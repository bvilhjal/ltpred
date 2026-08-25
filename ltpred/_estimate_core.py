"""Liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

Internal helpers shared by the public estimators in :mod:`ltpred.estimate`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import validate_bounds

_PA_METHODS = {"pa", "pearson-aitken", "pearson_aitken", "aitken"}


def _resolve_method(method, default):
    """Normalise a method spelling to ``\"gibbs\"`` or ``\"pearson-aitken\"``.

    The single method-name -> engine gate shared by :func:`estimate_liability`
    and :func:`estimate_liability_from_kinship`: ``None`` maps to ``default``,
    the PA aliases (``\"pa\"``, ``\"pearson_aitken\"``, ``\"aitken\"``) to
    ``\"pearson-aitken\"``, and anything else raises."""
    if method is None:
        method = default
    name = str(method).lower()
    if name in _PA_METHODS:
        return "pearson-aitken"
    if name == "gibbs":
        return "gibbs"
    raise ValueError(f"unknown method {method!r}; use 'gibbs' or 'pearson-aitken'")


_OUT_COORDS = {"genetic": 0, "full": 1}
_OUT_NAMES = {0: "genetic", 1: "full"}


def _bounds_dtype(dtype):
    """Validate the per-family bounds dtype: float32 (half memory) or float64."""
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    return dt


def _out_entries(out):
    """Return a non-empty list of requested estimator output names."""
    if isinstance(out, str):
        entries = [out]
    else:
        try:
            entries = list(out)
        except TypeError:
            raise TypeError(
                "out must be 'genetic'/'full' or a non-empty sequence of them"
            ) from None
    if not entries:
        raise ValueError("out must contain at least one output name")
    return entries


def _resolve_out_entry(value):
    """Resolve one output name; only the exact strings ``\"genetic\"``/``\"full\"``."""
    if not isinstance(value, str):
        raise TypeError(f"out entry {value!r} must be one of genetic/full, "
                        f"not {type(value).__name__}")
    if value not in _OUT_COORDS:
        raise ValueError(f"out entry {value!r} must be one of genetic/full")
    return _OUT_COORDS[value]


@dataclass
class LiabilityResult:
    """Per-family liability estimates and their numerical uncertainty summaries.

    ``est``/``se``/``var`` map a column name to a per-family array (aligned with
    ``fam_ids``). Single-trait columns are ``\"genetic\"`` / ``\"full\"``; multi-trait
    columns are suffixed with the phenotype, e.g. ``\"genetic_height\"``.

    The two uncertainty fields answer different questions and neither substitutes
    for the other:

    * ``var`` is the target's **posterior** (conditional) variance — how uncertain
      this proband's liability is given their family. Both engines report it:
      Gibbs as the Monte-Carlo variance of its retained draws, Pearson-Aitken as
      its sequential-moment approximation. It does **not** shrink as you sample
      more.
    * ``se`` is the **estimator's own numerical error** in ``est``. Gibbs reports
      the batch-means Monte-Carlo error, which does shrink with more draws;
      Pearson-Aitken is deterministic and reports ``se = 0``. Zero PA SE means no
      Monte-Carlo error, not zero approximation error — PA's sequential fold stays
      approximate for more than one truncation, in both ``est`` and ``var``."""
    fam_ids: np.ndarray
    pids: object
    est: dict
    se: dict
    var: dict = None

    @property
    def genetic(self):
        """Shorthand for the single-trait genetic-liability estimate ``est['genetic']``
        (the usual output). Multi-trait results are keyed per trait, e.g.
        ``genetic_height`` — index ``.est`` directly for those."""
        if "genetic" not in self.est:
            raise AttributeError(
                "no 'genetic' column; multi-trait results use per-trait keys like "
                "'genetic_<trait>' — index .est directly")
        return self.est["genetic"]


def _normalise_out(out):
    coords = [_resolve_out_entry(value) for value in _out_entries(out)]
    return sorted(set(coords))


def _single_out(out):
    """Resolve ``out`` to one column index for the APIs that return a single array.

    Accepts ``\"genetic\"``/``\"full\"`` or a length-1 sequence, so every estimator
    takes the same spellings as :func:`estimate_liability`."""
    entries = _out_entries(out)
    if len(entries) != 1:
        raise ValueError("this API returns a single column; out must be one of "
                         "genetic/full (or a length-1 sequence)")
    return _resolve_out_entry(entries[0])


def _validate_mc_se_n_sim(n_sim):
    """Return an integer draw count large enough for the Gibbs MC-SE rule."""
    if isinstance(n_sim, (bool, np.bool_)):
        raise TypeError("n_sim must be an integer >= 4, not bool")
    try:
        n_sim = operator.index(n_sim)
    except TypeError:
        raise TypeError("n_sim must be an integer >= 4") from None
    if n_sim < 4:
        raise ValueError(
            "n_sim must be at least 4 to form two batches for the Monte-Carlo SE")
    return n_sim


def _validate_tol(tol):
    """Return the Gibbs convergence tolerance as a finite positive float.

    A NaN tolerance never satisfies ``se <= tol``, so the sampler runs to
    ``max_rounds`` yet the unconverged warning (keyed on the same comparison)
    stays silent; a non-positive one can never be reached."""
    if isinstance(tol, (bool, np.bool_)):
        raise TypeError("tol must be a positive real number, not bool")
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        raise TypeError("tol must be a positive real number") from None
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")
    return tol


def _validate_max_rounds(max_rounds):
    """Return ``max_rounds`` as a positive int: with none, the convergence loop
    never runs and every family keeps its all-zero initial estimates."""
    if isinstance(max_rounds, (bool, np.bool_)):
        raise TypeError("max_rounds must be a positive integer, not bool")
    try:
        max_rounds = operator.index(max_rounds)
    except TypeError:
        raise TypeError("max_rounds must be a positive integer") from None
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    return max_rounds


def batch_means(samples: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Batch-means estimate and Monte-Carlo SE of column means (Jones et al. 2006).

    Splits ``n`` samples into ``a = n // b`` consecutive batches of size
    ``b = floor(sqrt(n))``, then ``se = sqrt(b * var(batch_means) / (a*b))``
    — the denominator is the ``a*b`` draws that actually enter the batches,
    matching R ``batchmeans::bmmat``. Accepts a
    1-D or 2-D ``(n, ncols)`` array and returns ``(est, se)`` arrays over columns.
    Port of R ``batchmeans::bmmat`` -- the rule LTFHPlus uses to decide the Gibbs
    sampler has converged. (The estimator computes the same quantity online inside
    the kernel; this stays for direct use and tests.) Requires at least 4
    samples (two batches), the same minimum the estimator enforces on ``n_sim``."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 4:
        raise ValueError(
            "batch_means needs at least 4 samples to form two batches for the "
            "Monte-Carlo SE")
    b = int(np.floor(np.sqrt(n)))
    a = n // b
    used = x[:a * b].reshape(a, b, x.shape[1])
    batch_mean = used.mean(axis=1)             # (a, ncols)
    mu = batch_mean.mean(axis=0)               # (ncols,)
    sigma2 = b * np.sum((batch_mean - mu) ** 2, axis=0) / (a - 1)
    se = np.sqrt(sigma2 / (a * b))
    est = x.mean(axis=0)
    return est, se


def _ordered_thresholds(family, cov_roles):
    """Align a family's member bounds to the covariance's role ordering.

    Builds ``lower``/``upper`` (and pids) in ``cov_roles`` order, inserting the
    missing genetic row ``g`` -- and ``o`` if the proband gave no own status -- as
    the uninformative interval ``(-inf, inf)``. Mirrors LTFHPlus's
    ``add_missing_roles_for_proband``. Returns ``(lower, upper, pids)`` with
    per-phenotype columns when the inputs are vectors."""
    by_role = {m.role: m for m in family.members}
    n_pheno = 1
    for m in family.members:
        n_pheno = max(n_pheno, np.size(m.lower))

    def bounds(role):
        m = by_role.get(role)
        if m is None:  # g always missing; o missing when no proband status given
            return (np.full(n_pheno, -np.inf), np.full(n_pheno, np.inf), None)
        lo = np.broadcast_to(np.asarray(m.lower, dtype=float), (n_pheno,))
        hi = np.broadcast_to(np.asarray(m.upper, dtype=float), (n_pheno,))
        return lo, hi, m.pid

    lower, upper, pids = [], [], []
    for role in cov_roles:
        lo, hi, pid = bounds(role)
        lower.append(lo)
        upper.append(hi)
        pids.append(pid)
    lower = np.array(lower)
    upper = np.array(upper)
    validate_bounds(lower, upper, context=f"family {family.fam_id!r} bounds")
    return lower, upper, pids
