"""Selected additive relationships with bounded LRU memoization.

A whole-graph topological rank guides the exact tabular recursion
``A_ij = (A_sj + A_dj)/2`` and ``A_ii = 1 + A_sd/2``; unknown parents
contribute zero. An explicit continuation stack avoids Python recursion and
keeps active values alive when the cache evicts entries. Cache size changes
work, never values, which agree bit-for-bit with the dense fill at realistic
pedigree depths.
"""
from collections import OrderedDict

import numpy as np


class _SelectedKinship:
    """Selected entries of A = twice kinship; ``graph`` must remain unchanged."""

    def __init__(self, graph, cache_max_entries=100_000):
        """Rank ``graph`` topologically (O(n + edges)); raise on any cycle."""
        if isinstance(cache_max_entries, (bool, np.bool_)) or not isinstance(
                cache_max_entries, (int, np.integer)):
            raise TypeError("kinship_cache_size must be an integer")
        if cache_max_entries < 0:
            raise ValueError("kinship_cache_size must be nonnegative")
        self.cache_max_entries = int(cache_max_entries)
        self._cache = OrderedDict()
        self._sire, self._dam = graph.sire, graph.dam

        # Validate all parent links once, including disconnected components.
        # Parents always have lower rank, so every pair dependency decreases.
        n = len(graph.ids)
        remaining = [int(s != -1) + int(d != -1)
                     for s, d in zip(graph.sire, graph.dam)]
        ready = [i for i, count in enumerate(remaining) if count == 0]
        self._rank = np.empty(n, dtype=np.intp)
        head = 0
        while head < len(ready):
            i = ready[head]
            self._rank[i] = head
            head += 1
            for child in graph.children[i]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        if head != n:
            raise ValueError("pedigree has a cycle (an individual is its own ancestor)")

    def _remember(self, key, value):
        """K6: insert as most recently used; evict the LRU entry beyond the cap."""
        if self.cache_max_entries:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def _pair(self, i, j):
        """``A_ij`` by steps K1-K6. Frame stages: 0 evaluate, 1 sire term done,
        2 combine ``(first + value) / 2``, 3 combine ``1 + value / 2``."""
        # A continuation retains its first child value while the second is
        # evaluated. Thus even a cache of size zero cannot evict a value that
        # the active computation still needs.
        stack = [(i, j, 0, 0.0)]
        value = 0.0
        while stack:
            i, j, stage, first = stack.pop()
            if stage == 0:
                if i == -1 or j == -1:
                    value = 0.0
                    continue
                if self._rank[i] < self._rank[j]:
                    i, j = j, i
                key = (i, j)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    value = cached
                    continue
                s, d = self._sire[i], self._dam[i]
                if i == j:
                    if s == -1 or d == -1:
                        value = 1.0
                        self._remember(key, value)
                    else:
                        stack.append((i, j, 3, 0.0))
                        stack.append((s, d, 0, 0.0))
                else:
                    stack.append((i, j, 1, 0.0))
                    stack.append((s, j, 0, 0.0))
            elif stage == 1:
                stack.append((i, j, 2, value))
                stack.append((self._dam[i], j, 0, 0.0))
            else:
                value = 1.0 + 0.5 * value if stage == 3 else 0.5 * (first + value)
                self._remember((i, j), value)
        return value

    def matrix(self, indices):
        """Return only the requested square relationship matrix, in row order.

        ``indices`` are parent-graph positions; the ``n (n + 1) / 2`` lower
        triangle entries are evaluated by `_pair` and mirrored."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("relationship indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= self._rank.size):
            raise ValueError("relationship index is outside the parent graph")
        n = len(indices)
        result = np.empty((n, n), dtype=np.float64)
        for a, i in enumerate(indices):
            for b in range(a + 1):
                result[a, b] = result[b, a] = self._pair(int(i), int(indices[b]))
        return result


def _covariance_reduction_is_safe(h2, n_members):
    """Certify when pedigree marginalisation needs no covariance repair.

    A conservative inverse-trace bound puts every principal covariance block's
    smallest eigenvalue above ``1e-6``. The bound uses pedigree diagonals in
    [1,2], genetic variance at least ``h2`` and residual variance at least
    ``(1-h2)/(1+h2)``. Near the heritability boundaries it fails closed, so
    callers retain the full matrix and legacy repair.
    """
    # The bound cannot exceed h2; reject tiny values before reciprocation.
    if not 1e-6 < h2 < 1.0:
        return False
    residual = (1.0 - h2) / (1.0 + h2)
    lower_bound = 1.0 / (1.0 / h2 + 3.0 * n_members / residual)
    return lower_bound > 1e-6
