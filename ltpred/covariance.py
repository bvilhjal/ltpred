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
           "correct_positive_definite", "kinship_from_pedigree",
           "construct_covmat_from_kinship"]

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


def construct_covmat_single(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                            n_fam=None, add_ind=True, h2=0.5, c2=None, m2=None):
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
        return Covmat(np.empty((0, 0)), [], h2=h2)
    d = len(roles)
    cov = np.empty((d, d), dtype=np.float64)
    for i, ri in enumerate(roles):           # symmetric: fill upper, mirror to lower
        for j in range(i, d):
            val = get_relatedness(ri, roles[j], h2=h2)
            cov[i, j] = val
            cov[j, i] = val
    if c2 is not None or m2 is not None:
        cov = _apply_env_components(cov, roles, c2, m2, h2)
    return Covmat(cov, roles, h2=h2)


# Roles whose sex the LTFHPlus grammar already fixes. Everything else -- the
# proband, siblings, children, half-sibs and aunts/uncles ("au" covers both) --
# is sex-ambiguous and must be supplied explicitly.
_ROLE_SEX = {"m": "F", "f": "M", "mgm": "F", "mgf": "M", "pgm": "F", "pgf": "M"}


def _resolve_sexes(roles, sex):
    """Map each role to ``"F"``/``"M"``, from the grammar plus explicit ``sex``.

    ``g`` follows ``o``: they are the same person's genetic and full liability.
    Raises if any role is left unresolved, naming the offenders -- a silent
    default would quietly impose one sex's heritability on the other.
    """
    given = {}
    for role, value in (sex or {}).items():
        key = str(role).lower()
        val = str(value).upper()
        if val not in ("F", "M"):
            raise ValueError(
                f"sex[{role!r}] must be 'F' or 'M', got {value!r}")
        given[key] = val
    if "g" not in given and "o" in given:
        given["g"] = given["o"]

    out, missing = [], []
    for role in roles:
        resolved = given.get(role, _ROLE_SEX.get(role))
        if resolved is None:
            missing.append(role)
        out.append(resolved)
    if missing:
        raise ValueError(
            "the sex of these roles is not fixed by the role grammar and must "
            f"be given in sex=: {sorted(set(missing))}. The grammar fixes only "
            f"{sorted(_ROLE_SEX)}; siblings, children, half-sibs and "
            "aunts/uncles are ambiguous.")
    return out


def construct_covmat_sex_limited(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                                 n_fam=None, add_ind=True, *, h2_female, h2_male,
                                 rg_cross=1.0, sex=None, c2=None, m2=None):
    """Covariance for one trait under a **sex-limitation** model.

    The ordinary single-trait covariance puts one scalar ``h2`` on every person,
    so sex can only enter through the thresholds, where it corrects calibration
    but cannot change the ranking. Here sex enters the covariance itself:

    .. code-block:: text

        Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) * rg_cross^[sex_i != sex_j]

    with ``h2_i`` the heritability of person ``i``'s sex. ``h2_female ==
    h2_male`` and ``rg_cross == 1`` reproduces :func:`construct_covmat_single`
    exactly. Unequal heritabilities give *scalar* (quantitative) sex limitation;
    ``rg_cross < 1`` adds *qualitative* sex limitation -- partly different
    genetic architectures between the sexes.

    Full liabilities keep unit variance, so thresholds retain their prevalence
    meaning; only the genetic scale differs by sex.

    ``sex`` maps the ambiguous roles to ``"F"``/``"M"`` (``g`` follows ``o``);
    parents and grandparents are taken from the role grammar. ``c2``/``m2``
    behave as in :func:`construct_covmat_single`, validated against the larger
    of the two heritabilities so the residual variance is nonnegative for both
    sexes.

    The result is positive semi-definite for ``|rg_cross| <= 1``: it is
    ``D^(1/2) (A o Rg) D^(1/2)`` with ``o`` the Hadamard product, and both ``A``
    and the cross-sex correlation matrix ``Rg`` are PSD, so their Schur product
    is too.
    """
    for name, value in (("h2_female", h2_female), ("h2_male", h2_male)):
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"{name} must be in (0, 1] -- a zero heritability makes that "
                "sex's genetic liability identically zero, leaving its row of "
                "the covariance degenerate")
    if not (-1.0 <= rg_cross <= 1.0):
        raise ValueError(
            f"rg_cross must be a correlation in [-1, 1], got {rg_cross}; "
            "outside that range the covariance is not positive semi-definite")

    roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                           n_fam, add_ind)
    if not roles:
        return Covmat(np.empty((0, 0)), [], h2=(h2_female, h2_male))

    sexes = _resolve_sexes(roles, sex)
    h2_of = {"F": float(h2_female), "M": float(h2_male)}
    d = len(roles)
    cov = np.empty((d, d), dtype=np.float64)
    for i, ri in enumerate(roles):           # symmetric: fill upper, mirror to lower
        for j in range(i, d):
            if i == j and ri != "g":
                val = 1.0                    # full liabilities stay standardised
            else:
                shared = get_relatedness(ri, roles[j], h2=1.0)   # bare 2*phi
                scale = (h2_of[sexes[i]] * h2_of[sexes[j]]) ** 0.5
                cross = rg_cross if sexes[i] != sexes[j] else 1.0
                val = shared * scale * cross
            cov[i, j] = val
            cov[j, i] = val
    if c2 is not None or m2 is not None:
        cov = _apply_env_components(cov, roles, c2, m2, max(h2_of.values()))
    return Covmat(cov, roles, h2=(float(h2_female), float(h2_male)))


def construct_covmat_multi(fam_vec=("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"),
                           n_fam=None, add_ind=True, *, genetic_corrmat,
                           full_corrmat, h2_vec, phen_names=None):
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


def kinship_from_pedigree(ids, father, mother):
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
    for i in range(n):
        if sire[i] == i or dam[i] == i:
            raise ValueError(f"individual {ids[i]!r} is its own parent")

    # topological order: an individual comes after both its (known) parents
    done = [False] * n
    order = []
    while len(order) < n:
        progressed = False
        for i in range(n):
            if done[i]:
                continue
            if (sire[i] == -1 or done[sire[i]]) and (dam[i] == -1 or done[dam[i]]):
                order.append(i)
                done[i] = True
                progressed = True
        if not progressed:
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


def construct_covmat_from_kinship(A, h2=0.5, target=0, add_ind=True):
    """Liability-scale covariance from an additive relationship matrix ``A``.

    The kinship-based counterpart of :func:`construct_covmat_single`: given ``A``
    (``n×n``, e.g. from :func:`kinship_from_pedigree`) it first builds the raw
    covariance ``V = h2 * A + (1 - h2) * I`` and then divides row/column ``i`` by
    ``sqrt(V[i, i])``. This keeps every **full liability** ``o`` on the unit-variance
    threshold scale when inbreeding gives ``A[i, i] = 1 + F_i``; it is a no-op for
    non-inbred pedigrees. When ``add_ind``, the similarly standardised **genetic
    liability contribution** ``g`` of the ``target`` is prepended. Row order is
    ``[g, o_0, …, o_{n-1}]``; the ``target``'s own full-liability row is labelled
    ``o`` and the rest ``rel<i>``. Returns a :class:`Covmat`.

    This is exactly the matrix the Gibbs / PA samplers consume, so a kinship-derived
    covariance is a drop-in for the role-based one; for a standard pedigree the two
    agree entry for entry."""
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
        return Covmat(o_block, o_roles, h2=h2)
    d = n + 1
    cov = np.empty((d, d), dtype=np.float64)
    target_scale = scale[target]
    cov[0, 0] = h2 * A[target, target] / (target_scale * target_scale)
    cov[0, 1:] = cov[1:, 0] = h2 * A[target] / (target_scale * scale)
    cov[1:, 1:] = o_block
    return Covmat(cov, ["g"] + o_roles, h2=h2)


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
