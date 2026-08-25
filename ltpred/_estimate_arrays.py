"""Array-level liability estimators with own_status support."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .gibbs import as_bounds
from ._own_status import _unbind_o_column
from ._estimate_core import _single_out
from ._estimate_group import _base_seeds
from ._estimate_role_arrays import _pa_from_role_arrays, _gibbs_from_role_arrays

def estimate_liability_pa_arrays(roles: Sequence[str], lower: ArrayLike,
                                 upper: ArrayLike, h2: float = 0.5,
                                 out: str = "genetic",
                                 K_i: ArrayLike | None = None,
                                 K_pop: ArrayLike | None = None,
                                 use_mixture: bool = False,
                                 c2: float | None = None, m2: float | None = None,
                                 own_status: Literal["in", "out"] = "in"
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Array-level Pearson-Aitken estimator — skips ``Family``/``Member`` objects.

    The production fast path for many same-structure probands: ``roles`` is the
    shared list of member roles (``o`` and relatives; ``g`` is added), and ``lower``
    / ``upper`` are ``(n_families, len(roles))`` bounds aligned to ``roles`` (build
        them straight from your columns, e.g. with a threshold helper). The covariance
    is built once (with the ``c2``/``m2`` sibship and couple shared-environment
    components, ``h2 + c2 + m2 <= 1``). ``out`` is ``"genetic"`` (target ``g``) or
    ``"full"`` (``E[l_o | own interval and relatives]``). ``use_mixture`` with
    ``K_i``/``K_pop`` (same shape) enables the
    censored-control mixture. ``own_status="out"`` unbinds the ``o`` column
    if present (use I) without dropping it; ``"in"`` (default) is use II.
    Only role ``o`` with ``own_status="out"`` raises. Returns PA
    sequential-moment approximations ``(est, var)`` of length ``n_families``."""
    coord = _single_out(out)
    lower, upper, K_i, K_pop = _unbind_o_column(
        roles, lower, upper, own_status, K_i=K_i, K_pop=K_pop)
    est, var = _pa_from_role_arrays(
        roles, lower, upper, h2, [coord], K_i=K_i, K_pop=K_pop,
        use_mixture=use_mixture, c2=c2, m2=m2)
    return est[coord], var[coord]


def estimate_liability_gibbs_arrays(roles: Sequence[str], lower: ArrayLike,
                                    upper: ArrayLike, h2: float = 0.5,
                                    out: str = "genetic", tol: float = 0.01,
                                    n_sim: int = 100_000, burn_in: int = 1000,
                                    seed: int | None = None,
                                    max_rounds: int = 100,
                                    c2: float | None = None,
                                    m2: float | None = None,
                                    return_var: bool = False,
                                    own_status: Literal["in", "out"] = "in"
                                    ) -> tuple[np.ndarray, ...]:
    """Array-level Gibbs inference — skips ``Family``/``Member`` objects.

        Same array inputs as :func:`estimate_liability_pa_arrays` (float32 ``lower``/
    ``upper`` halve their memory); the covariance takes the same ``c2``/``m2``
    shared-environment components. Returns ``(est, se)`` (posterior mean and
    batch-means Monte-Carlo SE) of length ``n_families`` for the single target
    selected by ``out``, or ``(est, se, var)`` with ``return_var=True``, where
    ``var`` is the Monte-Carlo estimate of the target's **posterior** variance —
    the comparable quantity to the ``var`` returned by
    :func:`estimate_liability_pa_arrays`, and a different thing from the sampler's
    own error ``se``. ``seed`` must be a non-boolean integer in
    ``[0, 2**32 - 1]`` or ``None``. ``own_status`` is as in
    :func:`estimate_liability_pa_arrays`."""
    coord = _single_out(out)
    lower, upper, _, _ = _unbind_o_column(roles, lower, upper, own_status)
    lower = as_bounds(lower)
    seeds = _base_seeds(seed, np.asarray(lower).shape[0], max_rounds)
    est, se, var = _gibbs_from_role_arrays(
        roles, lower, upper, h2, [coord], seeds, tol, n_sim, burn_in,
        max_rounds, c2=c2, m2=m2)
    if return_var:
        return est[:, 0], se[:, 0], var[:, 0]
    return est[:, 0], se[:, 0]
