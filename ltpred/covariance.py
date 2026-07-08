"""Family covariance matrices on the liability scale.

Under the liability-threshold model a person's liability splits into a genetic
part ``l_g ~ N(0, h2)`` and an environmental part, summing to a full liability
``l_o ~ N(0, 1)``. Two relatives' genetic parts correlate by the fraction of DNA
they share, so every covariance entry is ``shared_DNA * h2`` (:func:`get_relatedness`).
:func:`construct_covmat_single` assembles the matrix for a proband's genetic
liability ``g``, full liability ``o`` and any relatives; :func:`construct_covmat_multi`
extends it to several genetically/environmentally correlated traits;
:func:`construct_covmat` dispatches on whether ``h2`` is a scalar or a vector.

These are ports of LTFHPlus's ``get_relatedness`` / ``construct_covmat*`` and the
matrix they build is exactly what :mod:`ltpred.gibbs` and :mod:`ltpred.estimate`
sample from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = ["Covmat", "get_relatedness", "construct_covmat_single",
           "construct_covmat_multi", "construct_covmat",
           "correct_positive_definite"]

# valid relative abbreviations (the LTFHPlus grammar), matched with re.fullmatch
# so trailing junk like "s1abc" or "c1x2" is rejected rather than silently treated
# as a sibling/child. Children must use the full ``c<group>.<idx>`` form (e.g.
# c1.1) so the partner-group logic in get_relatedness applies.
_VALID = re.compile(r"[gomf]|c\d+\.\d+|[mp]g[mf]|s\d*|[mp]hs\d*|[mp]au\d*")
# roles that may appear at most once (everything else can be numbered s1, s2, ...)
_SINGLE = re.compile(r"^[gomf]$|^[mp]g[mf]$")


@dataclass
class Covmat:
    """A liability-scale covariance matrix plus the role labelling of its rows.

    ``roles[i]`` is the family role of row ``i`` (``g``, ``o``, ``m``, ...). For
    the multi-trait case ``phen_names`` is set and rows are phenotype-major: row
    ``i`` belongs to phenotype ``phen_names[i // (d // n_pheno)]`` and the ``roles``
    list repeats the per-trait roles once per phenotype."""
    matrix: np.ndarray
    roles: list
    phen_names: list = None
    h2: object = None


def _match(pattern, s):
    return re.match(pattern, s) is not None


def _validate_relative(s):
    if not _VALID.fullmatch(s):
        raise ValueError(
            f"{s!r} is not a valid relative abbreviation. Use g, o, m, f, "
            "c<group>.<idx> (e.g. c1.1), mgm, mgf, pgm, pgf, s[0-9]*, mhs[0-9]*, "
            "phs[0-9]*, mau[0-9]*, or pau[0-9]*.")


def get_relatedness(s1, s2, h2=0.5):
    """Shared-DNA fraction between two family roles, times ``h2``.

    ``s1``/``s2`` are role strings (``g`` genetic liability, ``o`` full liability,
    ``m``/``f`` parents, ``s`` siblings, ``mgm``/``pgf`` grandparents, ``mhs``/``phs``
    half-sibs, ``mau``/``pau`` aunts/uncles, ``c`` children). Returns e.g.
    ``0.5*h2`` for parent/offspring or full sibs, ``0.25*h2`` for grandparents and
    half-sibs. Pass ``h2=1`` to get the bare shared-DNA fraction. A port of
    LTFHPlus::get_relatedness -- the relatedness table is reproduced branch for
    branch."""
    s1, s2 = s1.lower(), s2.lower()
    _validate_relative(s1)
    _validate_relative(s2)

    if _match(r"o$", s1):  # target individual's full liability
        if s1 == s2:
            return 1.0
        if _match(r"g$", s2):
            return 1.0 * h2
        if _match(r"m$", s2) or _match(r"f$", s2) or _match(r"c[0-9]*\.[0-9]*", s2) or _match(r"s[0-9]*", s2):
            return 0.5 * h2
        if _match(r"[mp]hs[0-9]*", s2) or _match(r"[mp]g[mf]", s2) or _match(r"[mp]au[0-9]*", s2):
            return 0.25 * h2

    elif _match(r"g$", s1):  # target individual's genetic liability
        if _match(r"[go]$", s2):
            return 1.0 * h2
        if _match(r"m$", s2) or _match(r"f$", s2) or _match(r"c[0-9]*\.[0-9]*", s2) or _match(r"s[0-9]*", s2):
            return 0.5 * h2
        if _match(r"[mp]hs[0-9]*", s2) or _match(r"[mp]g[mf]", s2) or _match(r"[mp]au[0-9]*", s2):
            return 0.25 * h2

    elif _match(r"m$", s1):  # mother
        if s1 == s2:
            return 1.0
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"mhs[0-9]*", s2) or _match(r"mg[mf]$", s2) or _match(r"mau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.25 * h2
        if _match(r"f$", s2) or _match(r"pg[mf]$", s2) or _match(r"phs[0-9]*", s2) or _match(r"pau[0-9]*", s2):
            return 0.0

    elif _match(r"f$", s1):  # father
        if s1 == s2:
            return 1.0
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"phs[0-9]*", s2) or _match(r"pg[mf]$", s2) or _match(r"pau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.25 * h2
        if _match(r"m$", s2) or _match(r"mg[mf]$", s2) or _match(r"mhs[0-9]*", s2) or _match(r"mau[0-9]*", s2):
            return 0.0

    elif _match(r"c[0-9]*\.[0-9]*", s1):  # children
        if s1 == s2:
            return 1.0
        if _match(r"[go]$", s2):
            return 0.5 * h2
        if _match(r"c[0-9]*\.[0-9]*", s2) and _child_group(s1) == _child_group(s2):
            return 0.5 * h2   # same partner group -> full siblings
        if _match(r"c[0-9]*\.[0-9]*", s2) and _child_group(s1) != _child_group(s2):
            return 0.25 * h2  # different partner group -> half siblings
        if _match(r"[mf]$", s2) or _match(r"s[0-9]*", s2):
            return 0.25 * h2
        if _match(r"[mp]g[mf]$", s2) or _match(r"[mp]au[0-9]*", s2) or _match(r"[mp]hs[0-9]*", s2):
            return 0.125 * h2

    elif _match(r"s[0-9]*", s1):  # full siblings
        if s1 == s2:
            return 1.0
        if _match(r"[go]$", s2) or _match(r"m$", s2) or _match(r"f$", s2) or _match(r"s[0-9]*", s2):
            return 0.5 * h2
        if _match(r"c[0-9]*\.[0-9]*", s2) or _match(r"[mp]hs[0-9]*", s2) or _match(r"[mp]g[mf]", s2) or _match(r"[mp]au[0-9]*", s2):
            return 0.25 * h2

    elif _match(r"mg[mf]$", s1):  # maternal grandparent
        if s1 == s2:
            return 1.0
        if _match(r"mg[mf]$", s2):
            return 0.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"mhs[0-9]*", s2):
            return 0.25 * h2
        if _match(r"m$", s2) or _match(r"mau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"f$", s2) or _match(r"pg[mf]$", s2) or _match(r"phs[0-9]*", s2) or _match(r"pau[0-9]*", s2):
            return 0.0

    elif _match(r"pg[mf]$", s1):  # paternal grandparent
        if s1 == s2:
            return 1.0
        if _match(r"pg[mf]$", s2):
            return 0.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"phs[0-9]*", s2):
            return 0.25 * h2
        if _match(r"f$", s2) or _match(r"pau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"m$", s2) or _match(r"mg[mf]$", s2) or _match(r"mhs[0-9]*", s2) or _match(r"mau[0-9]*", s2):
            return 0.0

    elif _match(r"mhs[0-9]*", s1):  # maternal half-siblings
        if s1 == s2:
            return 1.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"mg[mf]$", s2) or _match(r"mau[0-9]*", s2):
            return 0.25 * h2
        if _match(r"m$", s2) or _match(r"mhs[0-9]*", s2):
            return 0.5 * h2
        if _match(r"f$", s2) or _match(r"pg[mf]$", s2) or _match(r"phs[0-9]*", s2) or _match(r"pau[0-9]*", s2):
            return 0.0

    elif _match(r"phs[0-9]*", s1):  # paternal half-siblings
        if s1 == s2:
            return 1.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"pg[mf]$", s2) or _match(r"pau[0-9]*", s2):
            return 0.25 * h2
        if _match(r"f$", s2) or _match(r"phs[0-9]*", s2):
            return 0.5 * h2
        if _match(r"m$", s2) or _match(r"mg[mf]$", s2) or _match(r"mhs[0-9]*", s2) or _match(r"mau[0-9]*", s2):
            return 0.0

    elif _match(r"mau[0-9]*", s1):  # maternal aunts/uncles
        if s1 == s2:
            return 1.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"mhs[0-9]*", s2):
            return 0.25 * h2
        if _match(r"m$", s2) or _match(r"mg[mf]$", s2) or _match(r"mau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"f$", s2) or _match(r"pg[mf]$", s2) or _match(r"phs[0-9]*", s2) or _match(r"pau[0-9]*", s2):
            return 0.0

    elif _match(r"pau[0-9]*", s1):  # paternal aunts/uncles
        if s1 == s2:
            return 1.0
        if _match(r"c[0-9]*\.[0-9]*", s2):
            return 0.125 * h2
        if _match(r"[go]$", s2) or _match(r"s[0-9]*", s2) or _match(r"phs[0-9]*", s2):
            return 0.25 * h2
        if _match(r"f$", s2) or _match(r"pg[mf]$", s2) or _match(r"pau[0-9]*", s2):
            return 0.5 * h2
        if _match(r"m$", s2) or _match(r"mg[mf]$", s2) or _match(r"mhs[0-9]*", s2) or _match(r"mau[0-9]*", s2):
            return 0.0

    return np.nan


def _child_group(s):
    """Partner-group id of a child label ``c<group>.<idx>`` (``c1.2`` -> ``"1"``).

    Children sharing a partner group are full siblings (0.5 h2); different groups
    are half siblings (0.25 h2). This fixes an unescaped-dot bug in the R original
    (``gsub(".[0-9]*", "", s)``), which collapsed every child to the empty string
    and so wrongly treated cross-group children as full siblings."""
    m = re.match(r"c(\d+)", s)
    return m.group(1) if m else s


def _expand_family(fam_vec, n_fam, add_ind):
    """Normalise the family spec to an ordered list of role labels.

    Accepts either ``fam_vec`` (an explicit list like ``["m", "f", "s1"]``) or
    ``n_fam`` (a ``{role: count}`` mapping expanded to ``s1, s2, ...``). ``g`` and
    ``o`` are stripped from the input and, when ``add_ind``, re-inserted first so
    the proband's genetic and full liabilities lead the ordering."""
    if fam_vec is not None and n_fam is not None:
        raise ValueError("supply only one of fam_vec or n_fam")

    if fam_vec is None and n_fam is None:
        return ["g", "o"] if add_ind else []

    if n_fam is not None:
        for role, cnt in n_fam.items():
            _validate_relative(role)
            if cnt < 0:
                raise ValueError("n_fam counts must be non-negative")
        n_fam = {r: c for r, c in n_fam.items() if c > 0 and r not in ("g", "o")}
        roles = []
        for r, c in n_fam.items():
            if _SINGLE.match(r):
                roles.append(r)
            else:
                roles.extend(f"{r}{i}" for i in range(1, int(c) + 1))
        fam_vec = roles
    else:
        fam_vec = [r for r in fam_vec if r not in ("g", "o")]
        for r in fam_vec:
            _validate_relative(r)

    return (["g", "o"] + list(fam_vec)) if add_ind else list(fam_vec)


def construct_covmat_single(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                            n_fam=None, add_ind=True, h2=0.5):
    """Covariance matrix for one trait: proband ``g``/``o`` plus relatives.

    Entry ``(i, j)`` is ``get_relatedness(role_i, role_j, h2)``. With the defaults
    the matrix covers a proband and both parents, a sibling and all four
    grandparents. Returns a :class:`Covmat`; ``.roles`` gives the row ordering the
    Gibbs sampler expects (``g``, ``o`` first when ``add_ind``)."""
    if not (0.0 <= h2 <= 1.0):
        raise ValueError("h2 must be in [0, 1]")
    roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                           n_fam, add_ind)
    if not roles:
        return Covmat(np.empty((0, 0)), [], h2=h2)
    d = len(roles)
    cov = np.empty((d, d), dtype=np.float64)
    for i, ri in enumerate(roles):           # symmetric: fill upper, mirror to lower
        for j in range(i, d):
            val = get_relatedness(ri, roles[j], h2=h2)
            cov[i, j] = val
            cov[j, i] = val
    return Covmat(cov, roles, h2=h2)


def construct_covmat_multi(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                           n_fam=None, add_ind=True, *, genetic_corrmat,
                           full_corrmat, h2_vec, phen_names=None):
    """Covariance matrix for several correlated traits.

    Same-trait blocks use ``get_relatedness(., ., h2_p)``. Cross-trait blocks
    scale relatedness by the genetic covariance
    ``rho_g[p, q] * sqrt(h2_p h2_q)``; on the diagonal, the same individual's
    genetic liabilities across traits correlate by that genetic covariance while
    their full liabilities correlate by ``full_corrmat[p, q]``. Rows are
    phenotype-major (all roles of trait 1, then trait 2, ...). Port of
    LTFHPlus::construct_covmat_multi."""
    h2_vec = np.asarray(h2_vec, dtype=float)
    genetic_corrmat = np.asarray(genetic_corrmat, dtype=float)
    full_corrmat = np.asarray(full_corrmat, dtype=float)
    n_pheno = len(h2_vec)
    if np.any((h2_vec < 0) | (h2_vec > 1)):
        raise ValueError("all h2 must be in [0, 1]")
    for m in (genetic_corrmat, full_corrmat):
        if m.shape != (n_pheno, n_pheno):
            raise ValueError("correlation matrices must be n_pheno x n_pheno")
    if phen_names is None:
        phen_names = [f"phenotype{p + 1}" for p in range(n_pheno)]
    elif len(phen_names) != n_pheno:
        raise ValueError("phen_names length must match number of phenotypes")

    fam_roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                               n_fam, add_ind)
    if not fam_roles:  # bare g/o for every phenotype
        fam_roles = ["g", "o"]
    k = len(fam_roles)
    d = k * n_pheno
    cov = np.empty((d, d), dtype=np.float64)

    for p1 in range(n_pheno):
        for p2 in range(n_pheno):
            gcov = genetic_corrmat[p1, p2] * np.sqrt(h2_vec[p1] * h2_vec[p2])
            for a, ra in enumerate(fam_roles):
                for b, rb in enumerate(fam_roles):
                    if p1 == p2:
                        val = get_relatedness(ra, rb, h2=h2_vec[p1])
                    else:
                        val = get_relatedness(ra, rb, h2=gcov)
                    cov[p1 * k + a, p2 * k + b] = val
            if p1 != p2:
                # same individual across traits: genetic liab -> genetic cov,
                # everything else (full liab / relatives) -> full correlation
                for a in range(k):
                    cov[p1 * k + a, p2 * k + a] = gcov if a == 0 else full_corrmat[p1, p2]

    roles = fam_roles * n_pheno
    return Covmat(cov, roles, phen_names=list(phen_names), h2=h2_vec)


def construct_covmat(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                     n_fam=None, add_ind=True, h2=0.5, *, genetic_corrmat=None,
                     full_corrmat=None, phen_names=None):
    """Dispatch to the single- or multi-trait builder based on ``h2``.

    Scalar ``h2`` (and no correlation matrices) -> :func:`construct_covmat_single`;
    a vector ``h2`` with ``genetic_corrmat`` and ``full_corrmat`` ->
    :func:`construct_covmat_multi`."""
    is_multi = np.ndim(h2) > 0 or genetic_corrmat is not None or full_corrmat is not None
    if not is_multi:
        return construct_covmat_single(fam_vec=fam_vec, n_fam=n_fam,
                                       add_ind=add_ind, h2=h2)
    if genetic_corrmat is None or full_corrmat is None:
        raise ValueError("multi-trait covmat needs genetic_corrmat and full_corrmat")
    return construct_covmat_multi(fam_vec=fam_vec, n_fam=n_fam, add_ind=add_ind,
                                  genetic_corrmat=genetic_corrmat,
                                  full_corrmat=full_corrmat, h2_vec=h2,
                                  phen_names=phen_names)


def correct_positive_definite(covmat, correction_val=0.99, correction_limit=100,
                              eps=1e-8):
    """Nudge a not-quite-positive-definite covariance matrix back to PD.

    Relatedness rounding can leave the assembled matrix with a tiny (or negative)
    eigenvalue, which breaks the Gibbs conditional variances and the ``solve`` in
    :func:`ltpred.gibbs.gibbs_params`. Following LTFHPlus, this repeatedly shrinks
    the off-diagonal (multiply the whole matrix by ``correction_val``, then restore
    the diagonal) until the smallest eigenvalue exceeds ``eps`` (strictly PD, not
    merely PSD) or ``correction_limit`` is hit. Returns ``(corrected, n_iter)`` and
    leaves already-PD matrices untouched; raises if it cannot reach strict PD."""
    cov = np.array(covmat, dtype=np.float64, copy=True)
    if np.min(np.linalg.eigvalsh(cov)) > eps:
        return cov, 0
    diag = np.diag(cov).copy()
    n = 0
    while np.min(np.linalg.eigvalsh(cov)) <= eps and n <= correction_limit:
        cov *= correction_val
        np.fill_diagonal(cov, diag)
        n += 1
    if np.min(np.linalg.eigvalsh(cov)) <= eps:
        raise ValueError("unable to enforce a positive-definite covariance matrix")
    return cov, n
