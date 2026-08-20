"""Covariance constructions not wired into any estimator.

**Unsupported research code.** These constructors were split out of
``ltpred.covariance``: nothing in the ``ltpred`` estimation path calls them,
they are not shipped in the wheel, and they may change or disappear without
notice. They are kept because ``benchmarks/bench_covariance_extensions.py``
exercises them; the private imports from
``ltpred.covariance`` are deliberate.

- :func:`construct_covmat_sex_limited` -- one trait under a **sex-limitation**
  model: sex-specific heritabilities and a cross-sex genetic correlation, so
  sex enters the covariance itself rather than only the thresholds.
- :func:`construct_covmat_nurture` -- separates a proband's **direct** genetic
  effect from parental **indirect** (genetic-nurture) effects in nuclear
  families, built from the path model rather than from kinship.

Shared machinery (the relatedness table, the family-spec expansion, the
shared-environment components, the PD correction) stays in
``ltpred.covariance`` and is imported from there.
"""

import re

import numpy as np

from ltpred.covariance import (Covmat, get_relatedness, _apply_env_components,
                               _expand_family)

__all__ = ["construct_covmat_sex_limited", "construct_covmat_nurture"]


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
    if "g" in given and "o" in given and given["g"] != given["o"]:
        # g is o's genetic component, not a second person: letting them differ
        # would silently apply one sex's heritability to the target row and the
        # other's to the same individual's full liability.
        raise ValueError(
            f"sex['g'] and sex['o'] must agree -- g is the genetic component "
            f"of o, the same person; got {given['g']!r} and {given['o']!r}. "
            f"Pass only sex['o'].")
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
    so sex can enter through personalised thresholds. Those thresholds affect
    calibration and can also change ordering through the truncated means, but
    they do not change the relative weights in the covariance. Here sex enters
    the covariance itself:

    .. code-block:: text

        Cov(g_i, g_j) = 2*phi_ij * sqrt(h2_i * h2_j) * rg_cross^[sex_i != sex_j]

    with ``h2_i`` the heritability of person ``i``'s sex. ``h2_female ==
    h2_male`` and ``rg_cross == 1`` reproduces
    :func:`ltpred.covariance.construct_covmat_single` exactly. Unequal
    heritabilities give *scalar* (quantitative) sex limitation;
    ``rg_cross < 1`` adds *qualitative* sex limitation -- partly different
    genetic architectures between the sexes.

    Full liabilities keep unit variance, so thresholds retain their prevalence
    meaning; only the genetic scale differs by sex.

    ``sex`` maps the ambiguous roles to ``"F"``/``"M"`` (``g`` follows ``o``);
    parents and grandparents are taken from the role grammar. ``c2``/``m2``
    behave as in :func:`ltpred.covariance.construct_covmat_single`, validated
    against the larger of the two heritabilities so the residual variance is
    nonnegative for both sexes.

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
        return Covmat(np.empty((0, 0)), [])

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
    return Covmat(cov, roles)


_NURTURE_ROLES = re.compile(r"^(g|o|m|f|s\d*)$")


def construct_covmat_nurture(fam_vec=("m", "f", "s1"), n_fam=None, add_ind=True,
                             *, h2, nurture):
    """Covariance separating a proband's **direct** genetic effect from parental
    **indirect** (genetic-nurture) effects.

    Everywhere else in ltpred an entry is ``2*phi * h2``: a function of
    relatedness alone. Genetic nurture is a *directional* path -- the parents'
    genotypes shape the offspring's environment, not the reverse -- so the
    covariance is built from the path model instead:

    .. code-block:: text

        A_o = (A_m + A_f)/2 + w_o          # transmission + Mendelian sampling
        l_o = A_o + nurture*(A_m + A_f) + e_o
        l_m = A_m + e_m                    # parents are founders here

    The target row ``g`` is the proband's **own** additive value ``A_o``, which
    is what a GWAS phenotype should predict; the indirect path contributes to
    the proband's *liability* without being part of their direct effect.

    The resulting entries are no longer a function of kinship. Writing ``h`` for
    ``h2`` and ``n`` for ``nurture``, parent-offspring covariance becomes
    ``h/2 + n*h`` while sib-sib becomes ``h/2 + 2*n*h + 2*n^2*h`` -- inflated by
    *different* amounts, which is what makes ``n`` identifiable from a nuclear
    family. The target row changes selectively: ``Cov(A_o, l_m)`` stays at
    ``h/2``, whereas ``Cov(A_o, l_o) = h*(1+n)`` and
    ``Cov(A_o, l_sibling) = h/2 + n*h`` because the nurture-driving parental
    genotypes correlate with ``A_o``. That pair-specific asymmetry between the
    ``g`` and ``o`` rows is precisely what a single symmetric kinship-scaled
    matrix cannot express.

    ``nurture = 0`` reproduces :func:`ltpred.covariance.construct_covmat_single`
    exactly.

    Only nuclear roles are accepted (``m``, ``f``, ``s...``). Grandparents and
    lateral relatives would require propagating the path model up the pedigree,
    which changes the parents from founders into offspring of their own parents;
    that recursion is not implemented, and silently treating them as founders
    would understate the covariance.

    Offspring liabilities are standardised to unit variance, so thresholds keep
    their prevalence meaning. This requires
    ``1 - h2 - 2*n^2*h2 - 2*n*h2 >= 0``; the shared nurture term is variance the
    residual has to give up.

    .. note::
       The sib-sib inflation ``2*n*h + 2*n^2*h`` is shared by all offspring of
       the couple, so on sibling covariance alone genetic nurture is
       indistinguishable from a sibship environment ``C``. Parent-offspring
       covariance is what separates them: ``C`` leaves it untouched, nurture
       raises it by ``n*h``. Fitting both from sibs only is not identified.
    """
    if not (0.0 < h2 <= 1.0):
        raise ValueError(
            "h2 must be in (0, 1] -- a zero direct heritability makes the "
            "genetic target identically zero, leaving its row degenerate")
    n = float(nurture)
    resid = 1.0 - h2 - 2.0 * n * n * h2 - 2.0 * n * h2
    if resid < -1e-12:
        raise ValueError(
            "the offspring residual variance would be negative "
            f"(1 - h2 - 2*nurture^2*h2 - 2*nurture*h2 = {resid:.4f}); the "
            "shared nurture term takes variance the residual must give up, so "
            "reduce h2 or |nurture|")

    roles = _expand_family(list(fam_vec) if fam_vec is not None else None,
                           n_fam, add_ind)
    bad = sorted({r for r in roles if not _NURTURE_ROLES.match(r)})
    if bad:
        raise ValueError(
            f"the genetic-nurture covariance is nuclear-only; got {bad}. "
            "Grandparents and lateral relatives need the path model propagated "
            "up the pedigree, which stops the parents being founders; that "
            "recursion is not implemented.")
    if not roles:
        return Covmat(np.empty((0, 0)), [])

    po = h2 / 2.0 + n * h2                       # parent - offspring liability
    ss = h2 / 2.0 + 2.0 * n * h2 + 2.0 * n * n * h2   # sib - sib liability

    def entry(a, b):
        pair = {a, b}
        if a == b:
            return h2 if a == "g" else 1.0       # g carries the direct variance
        if "g" in pair:
            other = b if a == "g" else a
            if other == "o":
                return h2 * (1.0 + n)            # own value plus the nurture it shares
            if other in ("m", "f"):
                return h2 / 2.0                  # transmission only: nurture-free
            return h2 / 2.0 + n * h2             # sib: shares the parental nurture
        if pair == {"m", "f"}:
            return 0.0                           # random mating
        if "o" in pair or all(r.startswith("s") for r in pair):
            return po if pair & {"m", "f"} else ss
        return po                                # parent with a sib

    d = len(roles)
    cov = np.empty((d, d), dtype=np.float64)
    for i, ri in enumerate(roles):               # symmetric: fill upper, mirror
        for j in range(i, d):
            val = entry(ri, roles[j])
            cov[i, j] = val
            cov[j, i] = val
    return Covmat(cov, roles)
