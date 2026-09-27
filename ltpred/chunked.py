"""Chunked and streaming drivers over the array liability kernels.

The ``*_chunked`` entry points take in-memory ``(F, k)`` bound arrays,
validate them once and run the array kernel on row slices, bounding the
kernel's working set. The ``*_batches`` entry points consume an iterator of
bound blocks, so the caller never holds every family's bounds at once; outputs
are concatenated in input order. All take one role set per call and keep the
``O(F)`` summary-memory contract of the array APIs (no draw arrays).

How results are preserved. PA is deterministic and per family, so chunking
changes its values by floating-point rounding at most. For Gibbs, each slice
or batch derives the seeds of its rows from their global row offset, the
same seeds `ltpred.estimate.estimate_liability_gibbs_arrays` gives those rows,
without building an ``O(F)`` seed array. A family's draws depend only on its
own bounds, seed block and the sampler settings (step G2 collapses each
family's own unbounded coordinates; methods report, Algorithm G), so the
estimates are bit-identical to the unchunked call for any chunk size.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._validation import validate_mixture_inputs
from .estimate import (_OUT_NAMES, _base_seeds, _check_unique_role_labels,
                       _gibbs_from_role_arrays, _pa_from_role_arrays,
                       _prepare_role_arrays, _single_out, _warn_unconverged)
from .gibbs import as_bounds

__all__ = ["estimate_liability_pa_chunked", "estimate_liability_gibbs_chunked",
           "estimate_liability_pa_batches", "estimate_liability_gibbs_batches"]


def _validate_chunk_size(chunk_size):
    """Return ``chunk_size`` as an integer >= 1; booleans are rejected."""
    if isinstance(chunk_size, (bool, np.bool_)):
        raise TypeError("chunk_size must be an integer >= 1, not bool")
    try:
        chunk_size = operator.index(chunk_size)
    except TypeError:
        raise TypeError("chunk_size must be an integer >= 1") from None
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return chunk_size


def _row_slices(n, chunk_size):
    """Yield row slices covering ``n`` families, including a ``(0, 0)`` empty."""
    if n == 0:
        yield slice(0, 0)
        return
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def _unpack_batch(item):
    """Parse one stream item as ``(lower, upper, K_i, K_pop)``.

    Accepts a mapping with keys ``lower``/``upper`` and optional ``K_i``/
    ``K_pop``, or a sequence of length 2 ``(lower, upper)`` or 4
    ``(lower, upper, K_i, K_pop)``.
    """
    if isinstance(item, Mapping):
        try:
            lower = item["lower"]
            upper = item["upper"]
        except KeyError as exc:
            raise ValueError(
                "a mapping batch must have 'lower' and 'upper' keys"
            ) from exc
        return lower, upper, item.get("K_i"), item.get("K_pop")
    if isinstance(item, np.ndarray):
        raise TypeError(
            "each batch must be a (lower, upper) tuple, "
            "(lower, upper, K_i, K_pop), or a mapping with those keys; "
            "got an ndarray")
    try:
        seq = tuple(item)
    except TypeError:
        raise TypeError(
            "each batch must be a (lower, upper) tuple, "
            "(lower, upper, K_i, K_pop), or a mapping with those keys"
        ) from None
    if len(seq) == 2:
        return seq[0], seq[1], None, None
    if len(seq) == 4:
        return seq[0], seq[1], seq[2], seq[3]
    raise ValueError(
        "each batch must be (lower, upper) or (lower, upper, K_i, K_pop) "
        f"or a mapping; got {len(seq)} items")


def _empty_bounds(roles):
    """``(0, len(roles))`` bounds, so an empty stream still goes through validation."""
    return np.empty((0, len(roles)))


def estimate_liability_pa_chunked(roles: Sequence[str], lower: ArrayLike,
                                  upper: ArrayLike, h2: float,
                                  out: str = "genetic",
                                  K_i: ArrayLike | None = None,
                                  K_pop: ArrayLike | None = None,
                                  use_mixture: bool = False,
                                  c2: float | None = None, m2: float | None = None,
                                  chunk_size: int = 65536
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """PA over row-chunks of one role-set. Returns ``(est, var)`` of length ``F``.

    Arguments and return values are those of
    `ltpred.estimate.estimate_liability_pa_arrays`; only the chunking differs.
    Validates the full bound arrays (and, with ``use_mixture``, the ``K_i``/
    ``K_pop`` pairs) once, then calls the array kernel on slices of
    ``chunk_size`` (default 65536). An in-memory array still holds every
    bound; stream with `estimate_liability_pa_batches` to avoid that.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    coord = _single_out(out)
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    if use_mixture:
        K_i, K_pop = validate_mixture_inputs(
            K_i, K_pop, expected_shape=lower.shape, require_pair=True,
            lower=lower, upper=upper,
            context="array estimator mixture inputs")
    est, var = np.empty(lower.shape[0]), np.empty(lower.shape[0])
    for sl in _row_slices(lower.shape[0], chunk_size):
        e, v = _pa_from_role_arrays(
            roles, lower[sl], upper[sl], h2, [coord],
            K_i=None if K_i is None else K_i[sl],
            K_pop=None if K_pop is None else K_pop[sl],
            use_mixture=use_mixture, c2=c2, m2=m2,
            mixture_require_pair=False)
        est[sl], var[sl] = e[coord], v[coord]
    return est, var


def estimate_liability_gibbs_chunked(roles: Sequence[str], lower: ArrayLike,
                                     upper: ArrayLike, h2: float,
                                     out: str = "genetic", tol: float = 0.01,
                                     n_sim: int = 100_000, burn_in: int = 1000,
                                     seed: int | None = None,
                                     max_rounds: int = 100,
                                     c2: float | None = None,
                                     m2: float | None = None,
                                     chunk_size: int = 4096,
                                     return_var: bool = False
                                     ) -> tuple[np.ndarray, ...]:
    """Gibbs over row-chunks of one role-set.

    Arguments and return values are those of
    `ltpred.estimate.estimate_liability_gibbs_arrays`; only the chunking
    differs. Each chunk's seeds come from its global row offset, so results
    are bit-identical to that function at the same ``seed`` for any
    ``chunk_size`` (default 4096; see the module docstring).
    """
    chunk_size = _validate_chunk_size(chunk_size)
    coord = _single_out(out)
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    F = lower.shape[0]
    est, se = np.empty(F), np.empty(F)
    var = np.empty(F) if return_var else None
    for sl in _row_slices(F, chunk_size):
        seeds = _base_seeds(seed, sl.stop - sl.start, max_rounds, start=sl.start)
        e, s, v = _gibbs_from_role_arrays(
            roles, lower[sl], upper[sl], h2, [coord], seeds,
            tol, n_sim, burn_in, max_rounds, c2=c2, m2=m2)
        est[sl], se[sl] = e[:, 0], s[:, 0]
        if return_var:
            var[sl] = v[:, 0]
    name = _OUT_NAMES[coord]
    _warn_unconverged({name: se}, [name], tol, max_rounds, F)
    if return_var:
        return est, se, var
    return est, se


def estimate_liability_pa_batches(roles: Sequence[str],
                                  batches: Iterable[Any],
                                  h2: float, out: str = "genetic",
                                  use_mixture: bool = False,
                                  c2: float | None = None, m2: float | None = None
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """PA from an iterator of bound batches, concatenated in order.

    Other arguments and return values are those of
    `ltpred.estimate.estimate_liability_pa_arrays`. Each item is
    ``(lower, upper)``, ``(lower, upper, K_i, K_pop)``, or a mapping with those
    keys (``K_i``/``K_pop`` optional). An empty iterator returns empty arrays.
    Mixture inputs are checked per batch, so with ``use_mixture=True`` a batch
    with no valid ``K`` pair raises; the chunked API gates that once on the
    whole cohort instead.
    """
    coord = _single_out(out)
    roles = list(roles)
    _check_unique_role_labels(roles)
    est_parts, var_parts = [], []
    for item in batches:
        lower, upper, K_i, K_pop = _unpack_batch(item)
        e, v = _pa_from_role_arrays(
            roles, lower, upper, h2, [coord],
            K_i=K_i, K_pop=K_pop, use_mixture=use_mixture,
            c2=c2, m2=m2)
        est_parts.append(e[coord])
        var_parts.append(v[coord])
    if not est_parts:
        empty = _empty_bounds(roles)
        e, v = _pa_from_role_arrays(
            roles, empty, empty, h2, [coord],
            use_mixture=use_mixture, c2=c2, m2=m2,
            mixture_require_pair=False)
        return e[coord], v[coord]
    return np.concatenate(est_parts), np.concatenate(var_parts)


def estimate_liability_gibbs_batches(roles: Sequence[str],
                                     batches: Iterable[Any],
                                     h2: float, out: str = "genetic",
                                     tol: float = 0.01, n_sim: int = 100_000,
                                     burn_in: int = 1000,
                                     seed: int | None = None,
                                     max_rounds: int = 100,
                                     c2: float | None = None,
                                     m2: float | None = None,
                                     return_var: bool = False
                                     ) -> tuple[np.ndarray, ...]:
    """Gibbs from consecutive bound batches of one virtual cohort.

    Other arguments and return values are those of
    `ltpred.estimate.estimate_liability_gibbs_arrays`. Each item is
    ``(lower, upper)`` or a mapping with those keys; a batch carrying
    ``K_i``/``K_pop`` raises (Gibbs has no censoring mixture). Rows are seeded
    by their position in the concatenated stream, so the result is
    bit-identical to one array call on the concatenated bounds at the same
    ``seed``, however the stream is split.
    """
    coord = _single_out(out)
    roles = list(roles)
    _check_unique_role_labels(roles)
    name = _OUT_NAMES[coord]
    est_parts, se_parts, var_parts = [], [], []
    start = 0
    for item in batches:
        lower, upper, K_i, K_pop = _unpack_batch(item)
        if K_i is not None or K_pop is not None:
            raise ValueError(
                "Gibbs batches have no censoring mixture; omit K_i and K_pop")
        lower = as_bounds(lower)
        n = 0 if lower.ndim != 2 else lower.shape[0]
        seeds = _base_seeds(seed, n, max_rounds, start=start)
        e, s, v = _gibbs_from_role_arrays(
            roles, lower, upper, h2, [coord], seeds,
            tol, n_sim, burn_in, max_rounds, c2=c2, m2=m2)
        est_parts.append(e[:, 0])
        se_parts.append(s[:, 0])
        var_parts.append(v[:, 0])
        start += n
    if not est_parts:
        empty = _empty_bounds(roles)
        seeds = _base_seeds(seed, 0, max_rounds)
        e, s, v = _gibbs_from_role_arrays(
            roles, empty, empty, h2, [coord], seeds,
            tol, n_sim, burn_in, max_rounds, c2=c2, m2=m2)
        est, se, var = e[:, 0], s[:, 0], v[:, 0]
    else:
        est = np.concatenate(est_parts)
        se = np.concatenate(se_parts)
        var = np.concatenate(var_parts)
    _warn_unconverged({name: se}, [name], tol, max_rounds, est.shape[0])
    if return_var:
        return est, se, var
    return est, se
