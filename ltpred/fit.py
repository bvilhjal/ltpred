"""Fit liability-scale heritability from family data (data-augmentation Gibbs).

Everywhere else in ltpred the family covariance is *given* — you supply ``h2`` and
the estimator conditions on it. This module instead **fits** it: it estimates the
liability-scale heritability ``h2`` from the pattern of case/control (and
age-of-onset) statuses across relatives, by treating the latent liabilities as
missing data.

The sampler is inspired by bipred's joint effect/parameter Gibbs: each sweep it
(1) **augments** the latent liabilities — one persistent truncated-MVN draw per
family under the current covariance — and (2) **updates** the covariance parameter
from those draws with a damped moment step. The update is a Haseman–Elston-style
regression of the sampled liability cross-products on the additive relationship,

    h2_hat = sum_pairs A_ij * l_i l_j  /  sum_pairs A_ij^2 ,

pooled over all related pairs in all families, then damped
``h2 <- (1 - damp) h2 + damp h2_hat`` for cross-sweep stability. Because the
liabilities are drawn conditional on each family's observed intervals, pooling
their cross-products reconstructs the model moments, so the chain settles at the
``h2`` consistent with the observed familial resemblance — a threshold-model
variance-component estimate from pedigree affection data.

Scope: additive heritability (one variance component). The same machinery extends
to a shared-environment / maternal component by regressing on extra relationship
matrices, but those need contrasting relative types (e.g. MZ vs DZ, or
parent-offspring vs sib) to be identifiable, so only ``h2`` is fit here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._mathfun import norm_cdf, norm_ppf
from .covariance import get_relatedness, correct_positive_definite
from .gibbs import gibbs_params, gibbs_advance, _seed_rng
from .estimate import _group_by_structure, batch_means

__all__ = ["FitResult", "fit_heritability"]


@dataclass
class FitResult:
    """Result of :func:`fit_heritability`.

    ``h2`` is the posterior-mean liability-scale heritability (mean of the
    post-burn-in trace); ``h2_se`` its batch-means Monte-Carlo standard error.
    ``samples`` is the post-burn-in ``h2`` trace and ``trace`` the full one (for
    convergence diagnostics)."""
    h2: float
    h2_se: float
    samples: np.ndarray
    trace: np.ndarray
    n_iter: int
    burn_in: int


def _init_x(lowers, uppers):
    """Start each coordinate at the median of its marginal truncated normal
    (sd = 1); pinned coords collapse to their value, unbounded ones to 0."""
    p0 = (norm_cdf(lowers) + norm_cdf(uppers)) / 2.0
    x = norm_ppf(p0)
    return np.where(np.isfinite(x), x, 0.0)


def _prepare_group(families, idx):
    """Per-structure precompute: relationship matrix ``A``, related-pair list,
    per-family bounds/fixed mask, and the initial chain state ``x``."""
    roles = [m.role for m in families[idx[0]].members]
    k = len(roles)
    A = np.array([[get_relatedness(ri, rj, h2=1.0) for rj in roles] for ri in roles])
    A, _ = correct_positive_definite(A)               # ensure PSD (usually a no-op)
    pairs = [(i, j, A[i, j]) for i in range(k) for j in range(i + 1, k)
             if abs(A[i, j]) > 1e-12]
    F = len(idx)
    lowers = np.empty((F, k))
    uppers = np.empty((F, k))
    for slot, f in enumerate(idx):
        for c, m in enumerate(families[f].members):
            lowers[slot, c] = float(m.lower)
            uppers[slot, c] = float(m.upper)
    fixed = np.ascontiguousarray((uppers - lowers) < 1e-8)
    x = np.empty((F, k))
    for slot in range(F):
        x[slot] = _init_x(lowers[slot], uppers[slot])
    return dict(A=np.ascontiguousarray(A), pairs=pairs, k=k, F=F,
                lowers=np.ascontiguousarray(lowers),
                uppers=np.ascontiguousarray(uppers), fixed=fixed,
                x=np.ascontiguousarray(x))


def fit_heritability(families, *, h2_init=0.5, n_iter=1500, burn_in=500,
                     inner_sweeps=5, damp=0.2, seed=None, eps=1e-4):
    """Estimate liability-scale ``h2`` from family case/control (+age) statuses.

    ``families`` is a list of :class:`~ltpred.family.Family` whose members carry
    liability bounds (from a threshold builder). Runs a data-augmentation Gibbs
    sampler that fits ``h2`` from the familial resemblance among the latent
    liabilities (see the module docstring). ``inner_sweeps`` truncated-MVN sweeps
    are taken per outer iteration; ``damp`` controls the moment-update stability.
    Returns a :class:`FitResult`.

    Needs relatives (at least one related pair); a set of lone probands carries no
    information about ``h2`` and raises."""
    groups = [_prepare_group(families, idx) for _key, idx in _group_by_structure(families)]
    sxx = sum(sum(aij * aij for (_i, _j, aij) in g["pairs"]) * g["F"] for g in groups)
    if sxx <= 0:
        raise ValueError("no related pairs in the families — cannot fit h2 "
                         "(need relatives, not lone probands).")

    if seed is not None:
        _seed_rng(int(seed))

    h2 = float(h2_init)
    trace = np.empty(int(n_iter))
    for it in range(int(n_iter)):
        sxy = 0.0
        for g in groups:
            k = g["k"]
            sigma = (1.0 - h2) * np.eye(k) + h2 * g["A"]   # diag stays 1
            P, sd = gibbs_params(sigma)
            gibbs_advance(P, sd, g["lowers"], g["uppers"], g["fixed"], g["x"],
                          int(inner_sweeps))
            x = g["x"]
            for (i, j, aij) in g["pairs"]:
                sxy += aij * float(x[:, i] @ x[:, j])
        h2_hat = min(max(sxy / sxx, eps), 1.0 - eps)
        h2 = (1.0 - damp) * h2 + damp * h2_hat
        trace[it] = h2

    samples = trace[int(burn_in):].copy()
    est, se = batch_means(samples)
    return FitResult(h2=float(est[0]), h2_se=float(se[0]), samples=samples,
                     trace=trace, n_iter=int(n_iter), burn_in=int(burn_in))
