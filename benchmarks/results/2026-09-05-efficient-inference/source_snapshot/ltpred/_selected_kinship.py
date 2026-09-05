"""Exact selected pedigree relationships with bounded memoization.

The tabular relationship recursion needs ancestors, not their entire dense
matrix. Evaluate requested pairs on the full parent graph, retaining at most
``cache_max_entries`` results. An explicit continuation stack handles arbitrarily
deep pedigrees without Python recursion; eviction changes work, never ancestry.
"""
from collections import OrderedDict

import numpy as np


class _SelectedKinship:
    """Selected entries of A = twice kinship; ``graph`` must remain unchanged."""

    def __init__(self, graph, cache_max_entries=100_000):
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
        if self.cache_max_entries:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def _pair(self, i, j):
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
        """Return only the requested square relationship matrix, in row order."""
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
    """Prove the full pedigree covariance needs no legacy PD repair.

    Write q = h2. Pedigree diagonals lie in [1, 2], so standardized target
    genetic variance a >= q and independent full-liability residual variance
    r >= (1-q)/(1+q). Regress the other genetic values on the target: each
    coefficient has square <= 2. The covariance therefore dominates the
    matrix for (g, beta*g + independent residuals). Its inverse trace is at
    most 1/q + 3*n_members/r, giving the reciprocal as an eigenvalue lower
    bound. Require 100 times the legacy 1e-8 repair threshold for roundoff.
    Near boundaries, use the unchanged full-pedigree correction instead.
    """
    # The bound cannot exceed h2; reject tiny values before reciprocation.
    if not 1e-6 < h2 < 1.0:
        return False
    residual = (1.0 - h2) / (1.0 + h2)
    lower_bound = 1.0 / (1.0 / h2 + 3.0 * n_members / residual)
    return lower_bound > 1e-6
