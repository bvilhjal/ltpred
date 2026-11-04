"""Liability-scale covariance from roles or arbitrary pedigrees.

Role constructors use the LTFHPlus relationship grammar and optional sibship
(C) and couple (M) kernels. The pedigree route computes ``A = 2 * kinship``
and constructs ``V = h2 A + c2 C + m2 M + e2 I``, standardising full
liabilities to unit variance before prepending the target genetic coordinate.
`correct_positive_definite` shrinks off-diagonals when numerical repair is
needed. See ``docs/algorithm.md`` for equations and model restrictions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    """Raise unless ``s`` fully matches the LTFHPlus role grammar ``_VALID``."""
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
    # Unreachable for validated roles; raise rather than silently return
    # relatedness 0 for a role the grammar accepted but this table forgot.
    raise ValueError(f"unclassifiable role {role!r}")


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


def get_relatedness(s1: str, s2: str, h2: float) -> float:
    """Return shared-DNA fraction times ``h2`` for two role labels.

    Roles cover the proband (``o``), genetic target (``g``), parents,
    siblings, grandparents, half-sibs, avuncular relatives and children
    (``c<group>.<idx>``). Use ``h2=1`` for the bare fraction. Opposite-side
    relatives and mates are unrelated; children in one partner group are full
    sibs. Same-side half-sib sets use the role grammar's shared-other-parent
    convention; more complex pedigrees need `construct_covmat_from_kinship`.
    """
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
    # Every kind pair except child/child (handled above) is in the table.
    return _SHARED_DNA[key] * h2


def _child_group(s):
    """Partner-group id of a child label ``c<group>.<idx>`` (``c1.2`` -> ``"1"``).

    Children sharing a partner group are full siblings (0.5 h2); different groups
    are half siblings (0.25 h2). This fixes an unescaped-dot bug in the R original
    (``gsub(".[0-9]*", "", s)``), which collapsed every child to the empty string
    and so wrongly treated cross-group children as full siblings."""
    m = re.match(r"c(\d+)", s)
    return m.group(1) if m else s


def _expand_family(fam_vec, add_ind):
    """Normalise the family spec to an ordered list of role labels.

    ``fam_vec`` is an explicit list like ``["m", "f", "s1"]`` (number repeated
    relatives ``s1, s2, ...``). ``g`` and ``o`` are stripped from the input
    and, when ``add_ind``, re-inserted first so the proband's genetic and
    full liabilities lead the ordering."""
    if fam_vec is None:
        return ["g", "o"] if add_ind else []

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
    """Return whether two roles share a full-sibship C block.

    Blocks cover the proband's sibship, each parent's avuncular sibship,
    children in one partner group, and same-side half-sib sets under the role
    grammar. The relatedness guard prevents unrelated look-alikes.
    """
    if get_relatedness(a, b, 1.0) <= 0:
        return False

    def full(p, x):
        """Whether pattern ``p`` matches all of role ``x``."""
        return p.fullmatch(x) is not None

    both_sibship = full(_SIBSHIP, a) and full(_SIBSHIP, b)
    parent_and_their_sib = ((full(_PARENT, a) and full(_AVUNC, b))
                            or (full(_PARENT, b) and full(_AVUNC, a)))
    same_side_avunc = ((full(_MAT_AVUNC, a) and full(_MAT_AVUNC, b))
                       or (full(_PAT_AVUNC, a) and full(_PAT_AVUNC, b)))
    same_group_children = (full(_CHILD, a) and full(_CHILD, b)
                           and _child_group(a) == _child_group(b))
    same_side_half_sibs = (full(_ROLE_HS, a) and full(_ROLE_HS, b)
                          and a[0] == b[0])
    return (both_sibship or parent_and_their_sib or same_side_avunc
            or same_group_children or same_side_half_sibs)


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


#: Heritabilities at or below this make the ``g``-``o`` block's smallest
#: eigenvalue (about ``h2``) fall under `correct_positive_definite`'s strict-PD
#: floor (``eps = 1e-8``), so no off-diagonal shrinking can repair the matrix.
#: Rejected here with a message naming ``h2`` rather than by that opaque
#: downstream failure.
_H2_FLOOR = 1e-8


def _validate_h2(h2):
    """Require heritability above the covariance repair's numerical floor."""
    if h2 is None or not (0.0 < h2 <= 1.0):
        raise ValueError(
            "h2 must be in (0, 1] -- a zero h2 makes the genetic liability "
            "identically zero, so its row of the covariance is degenerate "
            "and no positive-definite correction can recover it")
    if h2 <= _H2_FLOOR:
        raise ValueError(
            f"h2 must exceed {_H2_FLOOR:g}: the smallest eigenvalue of the "
            "g-o covariance is about h2, below the strict "
            "positive-definiteness floor the samplers require, so no "
            "correction can make the matrix usable -- supply a realistic "
            "liability-scale heritability")


def construct_covmat_single(fam_vec: Sequence[str] | None = ("m", "f", "s1", "mgm",
                                                            "mgf", "pgm", "pgf"),
                            add_ind: bool = True, *, h2: float,
                            c2: float | None = None,
                            m2: float | None = None) -> Covmat:
    """Construct a single-trait role covariance.

    Rows are ``g``, ``o`` and the supplied relative roles when ``add_ind``.
    ``h2`` must exceed ``1e-8`` and be at most one. Optional C/M components
    alter full-liability off-diagonals only and require
    ``h2 + c2 + m2 <= 1``. Returns a `Covmat` containing its row labels.
    """
    _validate_h2(h2)
    roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                           add_ind)
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
                           add_ind: bool = True, *,
                           genetic_corrmat: ArrayLike,
                           full_corrmat: ArrayLike, h2_vec: ArrayLike,
                           phen_names: Sequence[str] | None = None) -> Covmat:
    """Construct a phenotype-major multi-trait role covariance.

    Supply one heritability per trait and symmetric, unit-diagonal genetic and
    full-liability correlation matrices. With ``D = diag(sqrt(h2_vec))``, both
    ``G = D @ genetic_corrmat @ D`` and ``full_corrmat - G`` must be PSD.
    Same-person cross-trait entries use genetic covariance for ``g`` and full
    correlation otherwise.
    """
    h2_vec = np.asarray(h2_vec, dtype=float)
    genetic_corrmat = np.asarray(genetic_corrmat, dtype=float)
    full_corrmat = np.asarray(full_corrmat, dtype=float)
    if h2_vec.ndim != 1 or h2_vec.size == 0 or not np.all(np.isfinite(h2_vec)):
        raise ValueError("h2_vec must be a non-empty finite one-dimensional array")
    n_pheno = len(h2_vec)
    if np.any((h2_vec <= _H2_FLOOR) | (h2_vec > 1)):
        raise ValueError(
            f"all h2 must be in ({_H2_FLOOR:g}, 1] -- a zero h2 makes the "
            "genetic liability identically zero, and any h2 at or below the "
            "floor leaves the covariance under the strict "
            "positive-definiteness threshold, so no correction can recover it")
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
                               add_ind)
    if not fam_roles:  # bare g/o for every phenotype
        fam_roles = ["g", "o"]
    k = len(fam_roles)
    d = k * n_pheno
    cov = np.empty((d, d), dtype=np.float64)

    # One shared-DNA fraction table serves every phenotype pair: cross-role
    # relatedness is ``get_relatedness(ra, rb, h2) == frac(ra, rb) * h2`` with
    # ``frac * 1.0`` exact, so each block entry is the table scaled by that
    # pair's relatedness (``h2_vec[p1]`` within a trait, ``gcov`` across). The
    # same-role variances (1, or ``h2`` for ``g``) are the one exception and
    # keep going through ``get_relatedness`` directly.
    frac = np.ones((k, k))
    for a, ra in enumerate(fam_roles):
        for b, rb in enumerate(fam_roles):
            if a != b:
                frac[a, b] = get_relatedness(ra, rb, h2=1.0)

    for p1 in range(n_pheno):
        for p2 in range(n_pheno):
            gcov = genetic_cov[p1, p2]
            scale = h2_vec[p1] if p1 == p2 else gcov
            for a, ra in enumerate(fam_roles):
                # For p1 != p2 the same-role entries are overwritten just
                # below; don't pay a validated get_relatedness call for them.
                if p1 == p2:
                    cov[p1 * k + a, p2 * k + a] = get_relatedness(ra, ra, h2=scale)
                for b in range(a + 1, k):
                    val = frac[a, b] * scale
                    cov[p1 * k + a, p2 * k + b] = val
                    cov[p1 * k + b, p2 * k + a] = val
            if p1 != p2:
                # same individual across traits: genetic liab -> genetic cov,
                # everything else (full liab / relatives) -> full correlation
                # (keyed on the role label: with add_ind=False row 0 is a relative)
                for a in range(k):
                    cov[p1 * k + a, p2 * k + a] = (gcov if fam_roles[a] == "g"
                                                   else full_corrmat[p1, p2])

    roles = fam_roles * n_pheno
    return Covmat(cov, roles, phen_names=list(phen_names))


def _parent_links(ids, father, mother):
    """Index a pedigree: ``(ids, index, sire, dam, children, n_unresolved)``.

    Missing-id markers map to -1, as does any value not
    among ``ids``; the latter are counted in ``n_unresolved``. Raises on a
    length mismatch, a missing or duplicate id, and a person recorded as their
    own parent.
    """
    ids, father, mother = list(ids), list(father), list(mother)
    n = len(ids)
    if not (len(father) == len(mother) == n):
        raise ValueError("ids, father and mother must share length")
    from .family import _is_missing_id, _is_missing_parent
    if any(_is_missing_id(pid) for pid in ids):
        raise ValueError("ids must not contain missing values")
    if len(set(ids)) != n:
        raise ValueError("ids must be unique")
    index = {pid: i for i, pid in enumerate(ids)}
    unresolved = 0

    def _idx(p):
        """Index of parent ``p``, ``-1`` if missing or unlisted (counted)."""
        nonlocal unresolved
        if _is_missing_parent(p, index):
            return -1
        j = index.get(p, -1)
        unresolved += j == -1
        return j

    sire = [_idx(p) for p in father]
    dam = [_idx(p) for p in mother]
    children = [[] for _ in range(n)]
    for i in range(n):
        if sire[i] == i or dam[i] == i:
            raise ValueError(f"individual {ids[i]!r} is its own parent")
        if sire[i] != -1:
            children[sire[i]].append(i)
        if dam[i] != -1:
            children[dam[i]].append(i)
    return ids, index, sire, dam, children, unresolved


def _kinship_A(sire, dam, children=None):
    """Fill dense ``A = 2 * kinship`` from indexed parents.

    Kahn ordering puts parents first. Each row is the average of its parents'
    preceding rows, with ``A[k,k] = 1 + A[s,d]/2``. Unknown parents use a zero
    sentinel row. This indexed fast path shares phensim's arithmetic but keeps
    estimation independent of that optional dependency. O(n²) time and memory.
    """
    n = len(sire)
    if children is None:
        children = [[] for _ in range(n)]
        for i in range(n):
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

    # Fill in topological order, one vectorised row per individual. Row/column
    # ``n`` is an all-zero stand-in for an unknown parent, so the tabular
    # recursion needs no founder branches.
    pos = np.empty(n, dtype=np.intp)
    pos[order] = np.arange(n)
    s = [pos[sire[i]] if sire[i] != -1 else n for i in order]
    d = [pos[dam[i]] if dam[i] != -1 else n for i in order]
    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    for k in range(n):
        row = 0.5 * (A[s[k], :k] + A[d[k], :k])
        A[k, :k] = row
        A[:k, k] = row
        A[k, k] = 1.0 + 0.5 * A[s[k], d[k]]
    return A[np.ix_(pos, pos)]


def kinship_from_pedigree(ids: Sequence, father: Sequence,
                          mother: Sequence) -> tuple[list, np.ndarray]:
    """Return ``(ids, A)`` where ``A = 2 * kinship`` in input-id order.

    ``father`` and ``mother`` align to unique ``ids``. Missing or unlisted
    parents are founders; a zero id is a parent only when explicitly listed.
    The tabular recursion handles inbreeding and arbitrary depth, returning
    ``A[i,i] = 1 + F_i``. Missing/duplicate ids, self-parenting and cycles
    raise. Feed ``A`` to `construct_covmat_from_kinship`.
    """
    ids, _index, sire, dam, children, _unresolved = _parent_links(
        ids, father, mother)
    return ids, _kinship_A(sire, dam, children)


def _validate_kinship_component(value, kernel, name, n):
    """Validate one caller-supplied environmental variance component.

    ``A`` cannot identify environmental relationship classes: an additive
    relationship of 0.5 may be a full-sib or parent--offspring pair, and an
    additive relationship of 0 may be mates or strangers.  The kinship path
    therefore requires the caller to supply the component kernel rather than
    guessing it from ``A``.
    """
    if value is None:
        if kernel is not None:
            raise ValueError(f"{name}_kernel was supplied without {name}2")
        return 0.0, None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name}2 must be a finite nonnegative number") from None
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}2 must be a finite nonnegative number")
    if kernel is None:
        if value > 0.0:
            raise ValueError(
                f"{name}_kernel is required when {name}2 is nonzero; "
                "the environmental relationship cannot be inferred from A")
        return value, None

    K = np.ascontiguousarray(kernel, dtype=np.float64)
    label = f"{name}_kernel"
    if K.shape != (n, n):
        raise ValueError(f"{label} must have shape ({n}, {n})")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"{label} must contain only finite values")
    if not np.allclose(K, K.T, atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} must be symmetric")
    # The public tolerance accepts harmless floating-point asymmetry, while
    # downstream PA/Gibbs kernels require a more tightly symmetric covariance.
    # Canonicalise every accepted kernel rather than passing that mismatch on.
    K = 0.5 * (K + K.T)
    if not np.allclose(np.diag(K), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} must have a unit diagonal")
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (K + K.T))))
    if min_eig < -1e-8:
        raise ValueError(
            f"{label} must be positive semi-definite (minimum eigenvalue "
            f"{min_eig:.3g})")
    return value, K


# Internal certificates for pedigree-built matrices. _PSD_CERTIFIED skips
# redundant A validation; covariance repair still runs. _PD_CERTIFIED also
# certifies the assembled covariance, allowing the repair check to be skipped.
_PSD_CERTIFIED = object()
_PD_CERTIFIED = object()


def construct_covmat_from_kinship(A: ArrayLike, h2: float, target: int = 0,
                                  add_ind: bool = True, *,
                                  c2: float | None = None,
                                  c_kernel: ArrayLike | None = None,
                                  m2: float | None = None,
                                  m_kernel: ArrayLike | None = None,
                                  _certified_psd: object = None) -> Covmat:
    """Construct a standardised liability covariance from additive kinship ``A``.

    Forms ``V = h2 A + c2 C + m2 M + e2 I`` and scales full liabilities to
    unit variance. With ``add_ind``, prepends the target genetic coordinate on
    the same scale; rows are ``g, o_0, ..., o_{n-1}``. Environmental kernels
    must be supplied explicitly, symmetric PSD with unit diagonal, because
    they cannot be inferred from ``A``. Requires ``h2 + c2 + m2 <= 1``.

    ``A`` is validated as finite, symmetric and PSD unless an internal
    certificate is supplied. The role grammar's same-side half-sib convention
    is the only standard-pedigree case where role and pedigree covariances can
    legitimately differ.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    _validate_h2(h2)
    if not (0 <= target < n):
        raise ValueError(f"target {target} out of range for {n} individuals")
    if _certified_psd not in (_PSD_CERTIFIED, _PD_CERTIFIED):
        if not np.all(np.isfinite(A)):
            raise ValueError("A must contain only finite values")
        if not np.allclose(A, A.T, atol=1e-8, rtol=0.0):
            raise ValueError("A must be symmetric")
        # As for the optional component kernels, make the accepted tolerance
        # compatible with stricter downstream symmetry checks.
        A = 0.5 * (A + A.T)
        min_a_eig = float(np.min(np.linalg.eigvalsh(0.5 * (A + A.T))))
        if min_a_eig < -1e-8:
            raise ValueError(
                f"A must be positive semi-definite (minimum eigenvalue {min_a_eig:.3g})")

    c2, C = _validate_kinship_component(c2, c_kernel, "c", n)
    m2, M = _validate_kinship_component(m2, m_kernel, "m", n)
    if h2 + c2 + m2 > 1.0:
        raise ValueError(
            "h2 + c2 + m2 must not exceed 1 (the residual environmental "
            f"variance would be negative: {h2} + {c2} + {m2})")

    # Inbreeding makes diag(h2*A + c2*C + m2*M + e2*I) exceed one. Standardise each
    # member's full liability so the ordinary N(0, 1) prevalence thresholds remain
    # valid; this is a no-op (exact without C/M; within roundoff with them) when
    # every A[i, i] == 1 and all component kernels have their required unit
    # diagonal.
    residual = 1.0 - h2 - c2 - m2
    raw = h2 * A + residual * np.eye(n)
    if C is not None:
        raw += c2 * C
    if M is not None:
        raw += m2 * M
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
    """Shrink off-diagonals until the covariance is strictly positive definite.

    Each iteration multiplies off-diagonals by ``correction_val`` and preserves
    the diagonal. Returns ``(corrected_copy, n_iterations)``; already-PD input
    needs zero iterations. Raises after ``correction_limit`` unsuccessful
    steps. This changes the model and callers warn when it was necessary.
    """
    cov = np.array(covmat, dtype=np.float64, copy=True)
    # Cache the smallest eigenvalue: exactly one eigvalsh per shrink step
    # (the loop condition and the final check reuse the same value).
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig > eps:
        return cov, 0
    diag = np.diag(cov).copy()
    n = 0
    while min_eig <= eps and n < correction_limit:
        cov *= correction_val
        np.fill_diagonal(cov, diag)
        n += 1
        min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig <= eps:
        raise ValueError(
            "unable to enforce a positive-definite covariance matrix: after "
            f"{correction_limit} shrink steps the smallest eigenvalue is still "
            "<= eps (a tiny h2 at or below 1e-8 or strongly indefinite "
            "relatedness are the usual causes)")
    return cov, n
