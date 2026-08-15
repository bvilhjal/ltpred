"""Family covariance matrices on the liability scale.

Under the liability-threshold model a person's liability splits into a genetic
part ``l_g ~ N(0, h2)`` and an environmental part, summing to a full liability
``l_o ~ N(0, 1)``. Two relatives' genetic parts correlate by the fraction of DNA
they share, so every covariance entry is ``shared_DNA * h2`` (:func:`get_relatedness`).
:func:`construct_covmat_single` assembles the matrix for a proband's genetic
liability ``g``, full liability ``o`` and any relatives; :func:`construct_covmat_multi`
extends it to several genetically/environmentally correlated traits.

These are ports of LTFHPlus's ``get_relatedness`` / ``construct_covmat*`` and the
matrix they build is exactly what :mod:`ltpred.gibbs` and :mod:`ltpred.estimate`
sample from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Covmat", "get_relatedness", "construct_covmat_single",
           "construct_covmat_multi", "kinship_from_pedigree",
           "construct_covmat_from_kinship", "correct_positive_definite"]

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


def _validate_relative(s):
    if not _VALID.fullmatch(s):
        raise ValueError(
            f"{s!r} is not a valid relative abbreviation. Use g, o, m, f, "
            "c<group>.<idx> (e.g. c1.1), mgm, mgf, pgm, pgf, s[0-9]*, mhs[0-9]*, "
            "phs[0-9]*, mau[0-9]*, or pau[0-9]*.")


# Regex patterns for classifying numbered roles into relationship kinds.
_ROLE_SIB = re.compile(r"s\d*")
_ROLE_GP = re.compile(r"[mp]g[mf]")
_ROLE_HS = re.compile(r"[mp]hs\d*")
_ROLE_AVUNC = re.compile(r"[mp]au\d*")
_ROLE_CHILD = re.compile(r"c\d+\.\d+")


def _classify(role):
    """Return ``(kind, side)`` for a validated role string.

    Kinds: proband, genetic, parent, sib, gp, hs, avunc, child.
    Side: ``'m'`` (maternal), ``'p'`` (paternal), or ``''`` (neutral)."""
    if role == 'o':
        return 'proband', ''
    if role == 'g':
        return 'genetic', ''
    if role == 'm':
        return 'parent', 'm'
    if role == 'f':
        return 'parent', 'p'
    if _ROLE_SIB.fullmatch(role):
        return 'sib', ''
    if _ROLE_GP.fullmatch(role):
        return 'gp', 'm' if role.startswith('m') else 'p'
    if _ROLE_HS.fullmatch(role):
        return 'hs', 'm' if role.startswith('m') else 'p'
    if _ROLE_AVUNC.fullmatch(role):
        return 'avunc', 'm' if role.startswith('m') else 'p'
    if _ROLE_CHILD.fullmatch(role):
        return 'child', ''
    return 'unknown', ''


# Shared-DNA fraction between role kinds, for same-side or side-neutral pairs.
# Keys are alphabetically sorted kind pairs; values are the DNA fraction shared
# (multiplied by h² in get_relatedness for cross-role entries).
_SHARED_DNA = {
    ('avunc', 'avunc'): 0.5,
    ('avunc', 'child'): 0.125,
    ('avunc', 'genetic'): 0.25,
    ('avunc', 'gp'): 0.5,
    ('avunc', 'hs'): 0.25,
    ('avunc', 'parent'): 0.5,
    ('avunc', 'proband'): 0.25,
    ('avunc', 'sib'): 0.25,
    ('child', 'genetic'): 0.5,
    ('child', 'gp'): 0.125,
    ('child', 'hs'): 0.125,
    ('child', 'parent'): 0.25,
    ('child', 'proband'): 0.5,
    ('child', 'sib'): 0.25,
    ('genetic', 'genetic'): 1.0,
    ('genetic', 'gp'): 0.25,
    ('genetic', 'hs'): 0.25,
    ('genetic', 'parent'): 0.5,
    ('genetic', 'proband'): 1.0,
    ('genetic', 'sib'): 0.5,
    ('gp', 'gp'): 0.0,
    ('gp', 'hs'): 0.25,
    ('gp', 'parent'): 0.5,
    ('gp', 'proband'): 0.25,
    ('gp', 'sib'): 0.25,
    ('hs', 'hs'): 0.5,
    ('hs', 'parent'): 0.5,
    ('hs', 'proband'): 0.25,
    ('hs', 'sib'): 0.25,
    ('parent', 'parent'): 0.0,
    ('parent', 'proband'): 0.5,
    ('parent', 'sib'): 0.5,
    ('proband', 'proband'): 1.0,
    ('proband', 'sib'): 0.5,
    ('sib', 'sib'): 0.5,
}


def get_relatedness(s1: str, s2: str, h2: float = 0.5) -> float:
    """Shared-DNA fraction between two family roles, times ``h2``.

    ``s1``/``s2`` are role strings (``g`` genetic liability, ``o`` full liability,
    ``m``/``f`` parents, ``s`` siblings, ``mgm``/``pgf`` grandparents, ``mhs``/``phs``
    half-sibs, ``mau``/``pau`` aunts/uncles, ``c`` children). Returns e.g.
    ``0.5*h2`` for parent/offspring or full sibs, ``0.25*h2`` for grandparents and
    half-sibs. Pass ``h2=1`` to get the bare shared-DNA fraction.

    Two **same-side** half-sibs (``mhs1``/``mhs2`` or ``phs1``/``phs2``) are
    related ``0.5*h2`` *to each other* — the role grammar cannot name their second
    parents, so it implies a shared one (they are treated as full sibs of each
    other). A pedigree in which same-side half-sibs have distinct other parents is
    not expressible in this grammar; use :func:`construct_covmat_from_kinship`
    for those."""
    s1, s2 = s1.lower(), s2.lower()
    _validate_relative(s1)
    _validate_relative(s2)

    # Self-relatedness: the variance of the coordinate.
    # g (genetic liability) has variance h²; every other coordinate (full
    # liabilities) has variance 1.0.
    if s1 == s2:
        return float(h2) if s1 == 'g' else 1.0

    k1, side1 = _classify(s1)
    k2, side2 = _classify(s2)

    # Opposite-side relatives are unrelated through the proband.
    if side1 and side2 and side1 != side2:
        return 0.0

    # Child–child: depends on partner group.
    if k1 == 'child' and k2 == 'child':
        frac = 0.5 if _child_group(s1) == _child_group(s2) else 0.25
        return frac * h2

    key = (k1, k2) if k1 <= k2 else (k2, k1)
    frac = _SHARED_DNA.get(key, 0.0)
    return frac * h2


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
            if cnt > 1 and _SINGLE.match(role):
                raise ValueError(
                    f"n_fam count {cnt} for singleton role {role!r}: roles like "
                    f"{role!r} occur at most once per family — request multiples "
                    "with a numbered role instead (s1, s2, ...)")
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

    if len(set(fam_vec)) != len(fam_vec):
        # a repeated role names two individuals with one covariance coordinate,
        # silently building a perfectly-correlated (singular) pair of rows
        dup = sorted({r for r in fam_vec if fam_vec.count(r) > 1})
        raise ValueError(
            f"duplicate role(s) {dup} in the family specification; each role "
            "names one individual — number repeated relatives (s1, s2, ...).")
    return (["g", "o"] + list(fam_vec)) if add_ind else list(fam_vec)


_SIBSHIP = re.compile(r"o|s\d*")           # proband + full sibs (one sib-ship)
_CHILD = re.compile(r"c\d*\.\d*")          # the proband's children, by partner group
_PARENT = re.compile(r"[mf]")
_AVUNC = re.compile(r"[mp]au\d*")
_MAT_AVUNC = re.compile(r"mau\d*")         # mother's full sibs
_PAT_AVUNC = re.compile(r"pau\d*")         # father's full sibs


def _is_full_sib(a, b):
    """Whether roles ``a`` and ``b`` are **full siblings** — the pairs the common-
    environment component ``C`` loads on. Four cases: both in one sib-ship (proband
    ``o`` and its sibs ``s1``, ``s2``, …); a parent and their own sib (an
    aunt/uncle); two aunts/uncles on the **same** side (``mau1``/``mau2`` or
    ``pau1``/``pau2``), who are full sibs of that parent and of each other; or two
    of the proband's children in the **same partner group** (``c1.1``/``c1.2``),
    who are full sibs of each other. Including the avuncular case is what makes
    ``C`` form a complete sibship block ``{m, mau1, mau2, …}`` (a valid PSD
    component) rather than a non-PSD chain; the child case keeps ``C``'s structure
    the same whether a family is described from the parents' or the children's
    side. Cross-group children are only half sibs (``c1.1``/``c2.1``), so like
    ``mhs``/``phs`` they stay out of ``C``, and every block remains a disjoint
    equivalence class. The relatedness guard rejects unrelated look-alikes (e.g. a
    mother and a *paternal* aunt/uncle), which would otherwise match the
    parent/avuncular test."""
    if get_relatedness(a, b, 1.0) <= 0:
        return False

    def full(p, x):
        return p.fullmatch(x) is not None

    both_sibship = full(_SIBSHIP, a) and full(_SIBSHIP, b)
    parent_and_their_sib = ((full(_PARENT, a) and full(_AVUNC, b))
                            or (full(_PARENT, b) and full(_AVUNC, a)))
    same_side_avunc = ((full(_MAT_AVUNC, a) and full(_MAT_AVUNC, b))
                       or (full(_PAT_AVUNC, a) and full(_PAT_AVUNC, b)))
    same_group_children = (full(_CHILD, a) and full(_CHILD, b)
                           and _child_group(a) == _child_group(b))
    return (both_sibship or parent_and_their_sib or same_side_avunc
            or same_group_children)


# genetically-unrelated cohabiting couples in the role grammar; each is a mate
# pair that may share a couple/spousal environment (the ``M`` component).
_MATES = frozenset({frozenset({"m", "f"}), frozenset({"mgm", "mgf"}),
                    frozenset({"pgm", "pgf"})})


def _is_mates(a, b):
    """Whether roles ``a`` and ``b`` are a **mate pair** — the genetically-unrelated
    couples the couple-environment component ``M`` loads on: the proband's parents
    (``m``, ``f``) and the maternal/paternal grandparents (``mgm``/``mgf``,
    ``pgm``/``pgf``). Because mates share no DNA (``A_ab = 0``), their liability
    resemblance is not attributed to ``A`` — ``M`` captures it instead."""
    return frozenset({a, b}) in _MATES


def _apply_env_components(cov, fam_roles, c2, m2, h2):
    """Add sibship (``C``) and couple (``M``) shared-environment components to a
    liability covariance, in place.

    Only off-diagonal entries between non-``g`` rows change: a full-sib pair
    gets ``+ c2``, a mate pair ``+ m2``. The diagonal is untouched -- the
    residual environmental variance absorbs the components
    (``e2 = 1 - h2 - c2 - m2``), so every full liability keeps unit variance
    and the thresholds keep their prevalence meaning. The genetic target ``g``
    still couples to relatives only through ``h2 * A`` (it shares no
    environment), so its row is unchanged. ``h2 + c2 + m2 <= 1`` is required.
    """
    c2 = 0.0 if c2 is None else float(c2)
    m2 = 0.0 if m2 is None else float(m2)
    if c2 < 0 or m2 < 0:
        raise ValueError("c2 and m2 must be nonnegative")
    if h2 + c2 + m2 > 1.0:
        raise ValueError(
            "h2 + c2 + m2 must not exceed 1 (the residual environmental "
            f"variance would be negative: {h2} + {c2} + {m2})")
    d = len(fam_roles)
    for i in range(d):
        if fam_roles[i] == "g":
            continue
        for j in range(i + 1, d):
            if fam_roles[j] == "g":
                continue
            add = c2 * (_is_full_sib(fam_roles[i], fam_roles[j])) +                 m2 * (_is_mates(fam_roles[i], fam_roles[j]))
            if add:
                cov[i, j] += add
                cov[j, i] += add
    return cov


def construct_covmat_single(fam_vec: Sequence[str] | None = ("m", "f", "s1", "mgm",
                                                            "mgf", "pgm", "pgf"),
                            n_fam: Mapping[str, int] | None = None,
                            add_ind: bool = True, h2: float = 0.5,
                            c2: float | None = None,
                            m2: float | None = None) -> Covmat:
    """Covariance matrix for one trait: proband ``g``/``o`` plus relatives.

    Entry ``(i, j)`` is ``get_relatedness(role_i, role_j, h2)``. With the defaults
    the matrix covers a proband and both parents, a sibling and all four
    grandparents. Returns a :class:`Covmat`; ``.roles`` gives the row ordering the
    Gibbs sampler expects (``g``, ``o`` first when ``add_ind``). ``c2``/``m2`` add
    the sibship (``C``) and couple (``M``) shared-environment components of
    ``docs/algorithm.md`` to the relatives' covariance (off-diagonals only; the
    residual environmental variance absorbs them, so full liabilities keep unit
    variance and the genetic target stays coupled through ``h2 * A`` only).
    Requires ``h2 + c2 + m2 <= 1``."""
    if not (0.0 < h2 <= 1.0):
        raise ValueError(
            "h2 must be in (0, 1] -- a zero h2 makes the genetic liability "
            "identically zero, so its row of the covariance is degenerate "
            "and no positive-definite correction can recover it")
    roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                           n_fam, add_ind)
    if not roles:
        return Covmat(np.empty((0, 0)), [])
    d = len(roles)
    cov = np.empty((d, d), dtype=np.float64)
    for i, ri in enumerate(roles):           # symmetric: fill upper, mirror to lower
        for j in range(i, d):
            val = get_relatedness(ri, roles[j], h2=h2)
            cov[i, j] = val
            cov[j, i] = val
    if c2 is not None or m2 is not None:
        cov = _apply_env_components(cov, roles, c2, m2, h2)
    return Covmat(cov, roles)


def construct_covmat_multi(fam_vec: Sequence[str] | None = ("m", "f", "s1", "mgm",
                                                           "mgf", "pgm", "pgf"),
                           n_fam: Mapping[str, int] | None = None,
                           add_ind: bool = True, *,
                           genetic_corrmat: ArrayLike,
                           full_corrmat: ArrayLike, h2_vec: ArrayLike,
                           phen_names: Sequence[str] | None = None) -> Covmat:
    """Covariance matrix for several correlated traits.

    Same-trait blocks use ``get_relatedness(., ., h2_p)``. Cross-trait blocks
    scale relatedness by the genetic covariance
    ``rho_g[p, q] * sqrt(h2_p h2_q)``; on the diagonal, the same individual's
    genetic liabilities across traits correlate by that genetic covariance while
    their full liabilities correlate by ``full_corrmat[p, q]``. The three inputs
    must describe one coherent decomposition: with ``D = diag(sqrt(h2_vec))``,
    both ``G = D @ genetic_corrmat @ D`` and
    ``E = full_corrmat - G`` must be positive semi-definite. Rows are
    phenotype-major (all roles of trait 1, then trait 2, ...). Port of
    LTFHPlus::construct_covmat_multi."""
    h2_vec = np.asarray(h2_vec, dtype=float)
    genetic_corrmat = np.asarray(genetic_corrmat, dtype=float)
    full_corrmat = np.asarray(full_corrmat, dtype=float)
    if h2_vec.ndim != 1 or h2_vec.size == 0 or not np.all(np.isfinite(h2_vec)):
        raise ValueError("h2_vec must be a non-empty finite one-dimensional array")
    n_pheno = len(h2_vec)
    if np.any((h2_vec <= 0) | (h2_vec > 1)):
        raise ValueError(
            "all h2 must be in (0, 1] -- a zero h2 makes the genetic liability "
            "identically zero, so its row of the covariance is degenerate "
            "and no positive-definite correction can recover it")
    for name, m in (("genetic_corrmat", genetic_corrmat),
                    ("full_corrmat", full_corrmat)):
        if m.shape != (n_pheno, n_pheno):
            raise ValueError("correlation matrices must be n_pheno x n_pheno")
        if not np.all(np.isfinite(m)):
            raise ValueError(f"{name} must contain only finite values")
        if not np.allclose(m, m.T, atol=1e-8, rtol=0.0):
            raise ValueError(f"{name} must be symmetric")
        if not np.allclose(np.diag(m), 1.0, atol=1e-8, rtol=0.0):
            raise ValueError(f"{name} must have a unit diagonal")
        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (m + m.T))))
        if min_eig < -1e-8:
            raise ValueError(
                f"{name} must be positive semi-definite (minimum eigenvalue "
                f"{min_eig:.3g})")

    genetic_cov = genetic_corrmat * np.sqrt(np.outer(h2_vec, h2_vec))
    env_cov = full_corrmat - genetic_cov
    min_env_eig = float(np.min(np.linalg.eigvalsh(0.5 * (env_cov + env_cov.T))))
    if min_env_eig < -1e-8:
        raise ValueError(
            "genetic_corrmat, full_corrmat and h2_vec are incoherent: "
            "full_corrmat - D @ genetic_corrmat @ D must be positive "
            f"semi-definite (minimum eigenvalue {min_env_eig:.3g})")
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
            gcov = genetic_cov[p1, p2]
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
                # (keyed on the role label: with add_ind=False row 0 is a relative)
                for a in range(k):
                    cov[p1 * k + a, p2 * k + a] = (gcov if fam_roles[a] == "g"
                                                   else full_corrmat[p1, p2])

    roles = fam_roles * n_pheno
    return Covmat(cov, roles, phen_names=list(phen_names))


def kinship_from_pedigree(ids: Sequence, father: Sequence,
                          mother: Sequence) -> tuple[list, np.ndarray]:
    """Additive relationship matrix ``A`` (= 2×kinship) from a pedigree.

    Generalises the fixed role grammar (:func:`get_relatedness`) to **arbitrary
    pedigrees**: instead of naming relatives ``m``/``f``/``s1``/``mgm``… you give the
    parent of each individual and the relatedness is computed from the pedigree.

    ``ids`` is a sequence of unique individual ids; ``father`` and ``mother`` are the
    same-length sequences giving each individual's parents. A parent that is not
    itself one of ``ids`` (``None``, ``0``, ``""``, ``nan``, or any unlisted value)
    is treated as an unknown **founder**. Returns ``(ids, A)`` with ``A`` an
    ``(n, n)`` matrix in the given ``ids`` order: ``A[i,i] = 1 + F_i`` (``F_i`` the
    inbreeding coefficient) and ``A[i,j] = 2 × kinship(i, j)`` — e.g. 0.5 for
    parent–offspring and full sibs, 0.25 for grandparent/half-sib, 0.125 for first
    cousins. Computed by the recursive tabular method (Henderson 1976), which
    handles inbreeding and any pedigree depth.

    Feed ``A`` to :func:`construct_covmat_from_kinship` to build the liability
    covariance for these individuals."""
    ids = list(ids)
    father = list(father)
    mother = list(mother)
    n = len(ids)
    if not (len(father) == len(mother) == n):
        raise ValueError("ids, father and mother must share length")
    if len(set(ids)) != n:
        raise ValueError("ids must be unique")

    index = {pid: i for i, pid in enumerate(ids)}

    def _parent_idx(p):
        # unknown/founder markers, or any value not among the ids
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return -1
        return index.get(p, -1)

    sire = [_parent_idx(p) for p in father]
    dam = [_parent_idx(p) for p in mother]
    children = [[] for _ in range(n)]
    for i in range(n):
        if sire[i] == i or dam[i] == i:
            raise ValueError(f"individual {ids[i]!r} is its own parent")
        if sire[i] != -1:
            children[sire[i]].append(i)
        if dam[i] != -1:
            children[dam[i]].append(i)

    # Topological order: an individual comes after both its (known) parents.
    # Kahn's algorithm, O(n + edges). Any valid topological order yields the
    # same A, so the choice of order is free; the repeated-scan alternative
    # needs one pass per generation and degrades to O(n^2) when the records are
    # listed youngest-first.
    n_parents = [int(sire[i] != -1) + int(dam[i] != -1) for i in range(n)]
    ready = [i for i in range(n) if n_parents[i] == 0]
    order = []
    head = 0
    while head < len(ready):
        i = ready[head]
        head += 1
        order.append(i)
        for child in children[i]:
            n_parents[child] -= 1
            if n_parents[child] == 0:
                ready.append(child)
    if len(order) < n:
        raise ValueError("pedigree has a cycle (an individual is its own ancestor)")

    A = np.zeros((n, n), dtype=np.float64)
    for i in order:
        s, d = sire[i], dam[i]
        A[i, i] = 1.0 + (0.5 * A[s, d] if (s != -1 and d != -1) else 0.0)
        for j in order:
            if j == i:
                break                                    # only already-placed j
            aij = 0.5 * ((A[s, j] if s != -1 else 0.0) + (A[d, j] if d != -1 else 0.0))
            A[i, j] = A[j, i] = aij
    return ids, A


def construct_covmat_from_kinship(A: ArrayLike, h2: float = 0.5, target: int = 0,
                                  add_ind: bool = True) -> Covmat:
    """Liability-scale covariance from an additive relationship matrix ``A``.

    The kinship-based counterpart of :func:`construct_covmat_single`: given ``A``
    (``n×n``, e.g. from :func:`kinship_from_pedigree`) it first builds the raw
    covariance ``V = h2 * A + (1 - h2) * I`` and then divides row/column ``i`` by
    ``sqrt(V[i, i])``. This keeps every **full liability** ``o`` on the unit-variance
    threshold scale when inbreeding gives ``A[i, i] = 1 + F_i``; it is a no-op for
    non-inbred pedigrees. When ``add_ind``, the target's genetic liability divided
    by that **same full-liability SD** is prepended as ``g``, so its variance is
    ``h2 * A[t, t] / V[t, t]`` — equal to ``h2`` for a non-inbred target
    (``A[t, t] = 1``) but strictly below the role-based ``Var(g) = h2 * A[t, t]``
    under inbreeding. The kinship route's "genetic" estimates are therefore on the
    ``sd(o_target)`` scale when the target is inbred, not the role-based ``g``
    scale; rescale by ``sqrt(V[t, t])`` before comparing the two routes. Row order
    is ``[g, o_0, …, o_{n-1}]``; the ``target``'s own full-liability row is labelled
    ``o`` and the rest ``rel<i>``. Returns a :class:`Covmat`.

    This is exactly the matrix the Gibbs / PA samplers consume, so a kinship-derived
    covariance is a drop-in for the role-based one; for a standard pedigree the two
    agree entry for entry. One qualification: the role grammar's inherited
    convention takes two same-side half-sibs (``mhs1``, ``mhs2``) to share a second
    parent (:func:`get_relatedness`), so the entry-for-entry agreement holds only
    for pedigrees whose same-side half-sib sets satisfy that convention — a
    pedigree giving them distinct other parents legitimately disagrees with the
    role-based matrix there."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if not (0.0 < h2 <= 1.0):
        raise ValueError(
            "h2 must be in (0, 1] -- a zero h2 makes the genetic liability "
            "identically zero, so its row of the covariance is degenerate "
            "and no positive-definite correction can recover it")
    if not (0 <= target < n):
        raise ValueError(f"target {target} out of range for {n} individuals")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")
    if not np.allclose(A, A.T, atol=1e-8, rtol=0.0):
        raise ValueError("A must be symmetric")
    min_a_eig = float(np.min(np.linalg.eigvalsh(0.5 * (A + A.T))))
    if min_a_eig < -1e-8:
        raise ValueError(
            f"A must be positive semi-definite (minimum eigenvalue {min_a_eig:.3g})")

    # Inbreeding makes diag(h2*A + (1-h2)I) exceed one. Standardise each
    # member's full liability so the ordinary N(0, 1) prevalence thresholds remain
    # valid; this is exactly a no-op when every A[i, i] == 1.
    raw = h2 * A + (1.0 - h2) * np.eye(n)
    variances = np.diag(raw)
    if np.any(variances <= 0.0):
        raise ValueError("A and h2 must imply positive liability variances")
    scale = np.sqrt(variances)
    o_block = raw / np.outer(scale, scale)
    np.fill_diagonal(o_block, 1.0)
    o_roles = ["o" if i == target else f"rel{i}" for i in range(n)]
    if not add_ind:
        return Covmat(o_block, o_roles)
    d = n + 1
    cov = np.empty((d, d), dtype=np.float64)
    target_scale = scale[target]
    cov[0, 0] = h2 * A[target, target] / (target_scale * target_scale)
    cov[0, 1:] = cov[1:, 0] = h2 * A[target] / (target_scale * scale)
    cov[1:, 1:] = o_block
    return Covmat(cov, ["g"] + o_roles)


def correct_positive_definite(covmat: ArrayLike, correction_val: float = 0.99,
                              correction_limit: int = 100, eps: float = 1e-8
                              ) -> tuple[np.ndarray, int]:
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
