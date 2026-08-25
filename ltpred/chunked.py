"""Chunked and streaming drivers over the array liability kernels.

The in-memory ``*_chunked`` entry points still take ``(F, k)`` bound arrays;
they only bound the kernel's working set. The ``*_batches`` entry points
consume an iterator of bound blocks so the caller need never hold every
family's bounds at once. Both preserve deterministic grouping (one role-set
per call) and the ``O(F)`` summary-memory contract of the array APIs.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._validation import validate_bounds, validate_mixture_inputs
from .estimate import (_OUT_NAMES, _base_seeds, _check_unique_role_labels,
                       _gibbs_from_role_arrays, _pa_from_role_arrays,
                       _single_out, _warn_unconverged)
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


def _prepare_role_arrays(roles, lower, upper):
    """Validate roles and bound arrays once; return ``(roles, lower, upper)``."""
    roles = list(roles)
    _check_unique_role_labels(roles)
    lower = as_bounds(lower)
    upper = as_bounds(upper)
    if lower.ndim != 2 or upper.ndim != 2 or lower.shape[1] != len(roles):
        raise ValueError(
            f"lower and upper must be (n_families, {len(roles)}) -- one "
            f"column per role {roles}; got {lower.shape} and {upper.shape}")
    validate_bounds(lower, upper, context="array estimator bounds")
    return roles, lower, upper


def _row_slices(n, chunk_size):
    """Yield row slices covering ``n`` families, including a ``(0, 0)`` empty."""
    if n == 0:
        yield slice(0, 0)
        return
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def _concat(parts):
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


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
    return np.empty((0, len(roles)))


def estimate_liability_pa_chunked(roles: Sequence[str], lower: ArrayLike,
                                  upper: ArrayLike, h2: float = 0.5,
                                  out: str = "genetic",
                                  K_i: ArrayLike | None = None,
                                  K_pop: ArrayLike | None = None,
                                  use_mixture: bool = False,
                                  c2: float | None = None, m2: float | None = None,
                                  chunk_size: int = 65536
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Chunked Pearson-Aitken driver over the array kernel.

    Validates the full ``(F, k)`` bound (and mixture) arrays once, then streams
    homogeneous family rows through :func:`ltpred.estimate._pa_from_role_arrays`
    in blocks of ``chunk_size``. Default behaviour of
    :func:`~ltpred.estimate.estimate_liability_pa_arrays` is unchanged; this
    is an additional entry point.

    In-memory ``ndarray`` inputs still occupy ``F x k`` for the bounds; use a
    memmap for those arrays or :func:`estimate_liability_pa_batches` to actually
    avoid holding every family's bounds. Mixed pedigrees still need one call
    per role-set. Summary outputs stay ``O(F)``. Pearson-Aitken is
    deterministic, so there is no Monte-Carlo SE. Default ``chunk_size`` is
    65536: large enough that the hot kernel still saturates, small enough that
    float32 bounds for a nuclear family are a few MB. Rebuilding the tiny
    ``d x d`` covariance per chunk is intentional.

    Returns ``(est, var)`` of length ``F`` for the single ``out`` column.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    coord = _single_out(out)
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    if use_mixture:
        K_i, K_pop = validate_mixture_inputs(
            K_i, K_pop, expected_shape=lower.shape, require_pair=True,
            lower=lower, upper=upper,
            context="array estimator mixture inputs")
    est_parts, var_parts = [], []
    for sl in _row_slices(lower.shape[0], chunk_size):
        e, v = _pa_from_role_arrays(
            roles, lower[sl], upper[sl], h2, [coord],
            K_i=None if K_i is None else K_i[sl],
            K_pop=None if K_pop is None else K_pop[sl],
            use_mixture=use_mixture, c2=c2, m2=m2,
            mixture_require_pair=False)
        est_parts.append(e[coord])
        var_parts.append(v[coord])
    return _concat(est_parts), _concat(var_parts)


def estimate_liability_gibbs_chunked(roles: Sequence[str], lower: ArrayLike,
                                     upper: ArrayLike, h2: float = 0.5,
                                     out: str = "genetic", tol: float = 0.01,
                                     n_sim: int = 100_000, burn_in: int = 1000,
                                     seed: int | None = None,
                                     max_rounds: int = 100,
                                     c2: float | None = None,
                                     m2: float | None = None,
                                     chunk_size: int = 4096,
                                     return_var: bool = False
                                     ) -> tuple[np.ndarray, ...]:
    """Chunked Gibbs driver over the array kernel.

    Computes per-family seeds on the full cohort with
    :func:`ltpred.estimate._base_seeds` and passes ``seeds[sl]`` into each
    chunk, so the result agrees with
    :func:`~ltpred.estimate.estimate_liability_gibbs_arrays` at the same
    ``seed``. Restarting the sampler from ``seed=`` per chunk would break that
    agreement. Default ``chunk_size`` is 4096 because Gibbs accumulators are
    ``O(F_chunk x ncols)`` float64 and the sampler is heavier than PA.

    In-memory ``ndarray`` inputs still occupy ``F x k`` for the bounds; use a
    memmap or :func:`estimate_liability_gibbs_batches` to actually avoid holding
    every family's bounds. Mixed pedigrees still need one call per role-set.
    Summary outputs stay ``O(F)``. Unconverged families are warned once on the
    concatenated SE.

    Returns ``(est, se)`` of length ``F``, or ``(est, se, var)`` with
    ``return_var=True``.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    coord = _single_out(out)
    roles, lower, upper = _prepare_role_arrays(roles, lower, upper)
    F = lower.shape[0]
    seeds = _base_seeds(seed, F, max_rounds)
    est_parts, se_parts, var_parts = [], [], []
    for sl in _row_slices(F, chunk_size):
        e, s, v = _gibbs_from_role_arrays(
            roles, lower[sl], upper[sl], h2, [coord], seeds[sl],
            tol, n_sim, burn_in, max_rounds, c2=c2, m2=m2)
        est_parts.append(e[:, 0])
        se_parts.append(s[:, 0])
        var_parts.append(v[:, 0])
    est, se, var = _concat(est_parts), _concat(se_parts), _concat(var_parts)
    name = _OUT_NAMES[coord]
    _warn_unconverged({name: se}, [name], tol, max_rounds, F)
    if return_var:
        return est, se, var
    return est, se


def estimate_liability_pa_batches(roles: Sequence[str],
                                  batches: Iterable[Any],
                                  h2: float = 0.5, out: str = "genetic",
                                  use_mixture: bool = False,
                                  c2: float | None = None, m2: float | None = None
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Stream Pearson-Aitken estimates from an iterator of bound batches.

    ``batches`` yields either ``(lower, upper)``, ``(lower, upper, K_i, K_pop)``,
    or a mapping with keys ``lower``, ``upper`` and optional ``K_i``/``K_pop``.
    Results are concatenated in iterator order; an empty iterator returns empty
    arrays. The caller never has to hold every family's bounds at once — that
    is the memory contract. In-memory ndarrays still occupy ``F x k`` if you
    build them first; memmap or a true iterator is how you avoid that.

    Mixture inputs are validated **per batch** (the stream cannot see the whole
    cohort). A mixture batch with no valid ``K_i``/``K_pop`` pair raises, unlike
    the sliced ``*_chunked`` APIs which gate once globally so an all-cases
    chunk does not fail. Mixed pedigrees still need one call per role-set.
    Pearson-Aitken is deterministic (no SE). Summary outputs stay ``O(F)``.
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
                                     h2: float = 0.5, out: str = "genetic",
                                     tol: float = 0.01, n_sim: int = 100_000,
                                     burn_in: int = 1000,
                                     seed: int | None = None,
                                     max_rounds: int = 100,
                                     c2: float | None = None,
                                     m2: float | None = None,
                                     return_var: bool = False
                                     ) -> tuple[np.ndarray, ...]:
    """Stream Gibbs estimates from an iterator of bound batches.

    Batches are treated as consecutive blocks of a virtual concatenated
    cohort: a running family offset ``start`` assigns
    ``seeds = _base_seeds(seed, start + n, max_rounds)[start:]`` to a batch
    of size ``n``. Concatenating the same batches as one array therefore
    matches one-shot :func:`~ltpred.estimate.estimate_liability_gibbs_arrays`
    with the same ``seed``. Bound arrays are not retained across batches.

    In-memory ndarrays still occupy ``F x k`` if you build them first; memmap
    or a true iterator is how you avoid holding all bounds. Mixed pedigrees
    still need one call per role-set. Summary outputs stay ``O(F)``. Unconverged
    families are warned once on the concatenated SE.

    Returns ``(est, se)``, or ``(est, se, var)`` with ``return_var=True``.
    """
    coord = _single_out(out)
    roles = list(roles)
    _check_unique_role_labels(roles)
    name = _OUT_NAMES[coord]
    est_parts, se_parts, var_parts = [], [], []
    start = 0
    for item in batches:
        lower, upper, _K_i, _K_pop = _unpack_batch(item)
        lower = as_bounds(lower)
        n = 0 if lower.ndim != 2 else lower.shape[0]
        seeds = _base_seeds(seed, start + n, max_rounds)[start:]
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
