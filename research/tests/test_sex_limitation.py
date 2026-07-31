"""Sex-limitation covariance: scalar and qualitative sex effects in Sigma."""

import itertools

import numpy as np
import pytest

from research.covariance_extensions import construct_covmat_sex_limited
from ltpred.covariance import construct_covmat_single
from ltpred.pearson_aitken import pa_algorithm

FAM = ("m", "f", "s1", "mgm", "mgf", "pgm", "pgf")
SEX = {"o": "F", "s1": "M"}


def _index(covmat):
    return {role: i for i, role in enumerate(covmat.roles)}


@pytest.mark.parametrize("h2", [0.05, 0.4, 0.8, 1.0])
def test_reduces_to_scalar_model_exactly(h2):
    """Equal heritabilities and rg=1 is the ordinary single-trait covariance.

    Exact equality, not approximate: the sex-limited path must not perturb the
    established model when the sex effects are switched off.
    """
    got = construct_covmat_sex_limited(FAM, h2_female=h2, h2_male=h2,
                                       rg_cross=1.0, sex=SEX)
    want = construct_covmat_single(FAM, h2=h2)
    assert got.roles == want.roles
    assert np.array_equal(got.matrix, want.matrix)


@pytest.mark.parametrize("h2f,h2m,rg", list(itertools.product(
    [0.05, 0.5, 1.0], [0.05, 0.5, 1.0], [-1.0, -0.3, 0.0, 0.6, 1.0])))
def test_positive_semidefinite(h2f, h2m, rg):
    """PSD for every |rg| <= 1: D^(1/2) (A o Rg) D^(1/2) is a Schur product."""
    cov = construct_covmat_sex_limited(FAM, h2_female=h2f, h2_male=h2m,
                                       rg_cross=rg, sex=SEX).matrix
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_full_liabilities_stay_standardised():
    """Only the genetic scale is sex-specific; thresholds keep their meaning."""
    cov = construct_covmat_sex_limited(FAM, h2_female=0.6, h2_male=0.2,
                                       rg_cross=0.7, sex=SEX)
    idx = _index(cov)
    for role, i in idx.items():
        if role != "g":
            assert cov.matrix[i, i] == pytest.approx(1.0)
    # the genetic row carries the proband's own sex-specific heritability
    assert cov.matrix[idx["g"], idx["g"]] == pytest.approx(0.6)
    assert cov.matrix[idx["g"], idx["o"]] == pytest.approx(0.6)


def test_cross_sex_pairs_are_discounted_same_sex_pairs_are_not():
    h2f, h2m, rg = 0.6, 0.2, 0.7
    cov = construct_covmat_sex_limited(FAM, h2_female=h2f, h2_male=h2m,
                                       rg_cross=rg, sex=SEX)
    idx = _index(cov)
    # proband (F) and sibling (M): full sibs, 2*phi = 0.5, cross-sex
    assert cov.matrix[idx["o"], idx["s1"]] == pytest.approx(
        0.5 * (h2f * h2m) ** 0.5 * rg)
    # proband (F) and mother (F): full sibs' 2*phi = 0.5, same sex, no discount
    assert cov.matrix[idx["o"], idx["m"]] == pytest.approx(0.5 * h2f)
    # mother and father share no DNA: zero regardless of the sex model
    assert cov.matrix[idx["m"], idx["f"]] == 0.0


def test_rg_below_one_shrinks_cross_sex_covariance():
    """Lowering rg must strictly weaken cross-sex coupling and nothing else."""
    kw = dict(h2_female=0.5, h2_male=0.5, sex=SEX)
    hi = construct_covmat_sex_limited(FAM, rg_cross=1.0, **kw)
    lo = construct_covmat_sex_limited(FAM, rg_cross=0.4, **kw)
    idx = _index(hi)
    assert lo.matrix[idx["o"], idx["s1"]] < hi.matrix[idx["o"], idx["s1"]]
    # a same-sex pair is untouched
    assert lo.matrix[idx["o"], idx["m"]] == hi.matrix[idx["o"], idx["m"]]


def test_g_follows_o_without_being_named():
    cov = construct_covmat_sex_limited(("m", "f"), h2_female=0.3, h2_male=0.9,
                                       sex={"o": "M"})
    idx = _index(cov)
    assert cov.matrix[idx["g"], idx["g"]] == pytest.approx(0.9)


def test_ambiguous_roles_must_be_given_a_sex():
    """A silent default would impose one sex's heritability on the other."""
    with pytest.raises(ValueError, match="not fixed by the role grammar"):
        construct_covmat_sex_limited(FAM, h2_female=0.5, h2_male=0.5,
                                     sex={"o": "F"})          # s1 missing
    with pytest.raises(ValueError, match="not fixed by the role grammar"):
        construct_covmat_sex_limited(("m", "f"), h2_female=0.5, h2_male=0.5)


def test_grammar_fixes_parents_and_grandparents():
    """Only the ambiguous roles need declaring."""
    cov = construct_covmat_sex_limited(("m", "f", "mgm", "mgf", "pgm", "pgf"),
                                       h2_female=0.4, h2_male=0.8,
                                       sex={"o": "F"})
    idx = _index(cov)
    # mother is female, father male: their covariance with the proband differs
    assert cov.matrix[idx["o"], idx["m"]] == pytest.approx(0.5 * 0.4)
    assert cov.matrix[idx["o"], idx["f"]] == pytest.approx(
        0.5 * (0.4 * 0.8) ** 0.5)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_heritability(bad):
    with pytest.raises(ValueError, match="h2_female|h2_male"):
        construct_covmat_sex_limited(FAM, h2_female=bad, h2_male=0.5, sex=SEX)
    with pytest.raises(ValueError, match="h2_female|h2_male"):
        construct_covmat_sex_limited(FAM, h2_female=0.5, h2_male=bad, sex=SEX)


@pytest.mark.parametrize("bad", [-1.5, 1.01])
def test_rejects_out_of_range_rg(bad):
    with pytest.raises(ValueError, match="correlation in"):
        construct_covmat_sex_limited(FAM, h2_female=0.5, h2_male=0.5,
                                     rg_cross=bad, sex=SEX)


def test_rejects_unknown_sex_label():
    with pytest.raises(ValueError, match="must be 'F' or 'M'"):
        construct_covmat_sex_limited(FAM, h2_female=0.5, h2_male=0.5,
                                     sex={"o": "female", "s1": "M"})


def test_env_components_validated_against_the_larger_heritability():
    """Residual variance must stay nonnegative for *both* sexes."""
    with pytest.raises(ValueError, match="must not exceed 1"):
        construct_covmat_sex_limited(FAM, h2_female=0.3, h2_male=0.8,
                                     sex=SEX, c2=0.15, m2=0.1)   # 0.8+0.25 > 1
    ok = construct_covmat_sex_limited(FAM, h2_female=0.3, h2_male=0.5,
                                      sex=SEX, c2=0.15, m2=0.1)
    idx = _index(ok)
    assert ok.matrix[idx["m"], idx["f"]] == pytest.approx(0.1)   # mates get m2


def test_empty_family_returns_empty_matrix():
    cov = construct_covmat_sex_limited(None, add_ind=False, h2_female=0.5,
                                       h2_male=0.5)
    assert cov.matrix.shape == (0, 0)
    assert cov.roles == []


def test_feeds_the_covariance_level_pa_api():
    """End-to-end usability: the matrix drops into pa_algorithm unchanged."""
    cov = construct_covmat_sex_limited(("m", "f", "s1"), h2_female=0.6,
                                       h2_male=0.2, rg_cross=0.5, sex=SEX)
    d = len(cov.roles)
    lower = np.full(d, -np.inf)
    upper = np.full(d, np.inf)
    upper[cov.roles.index("m")] = -1.0        # mother a case-like tail bound
    lower[cov.roles.index("m")] = -np.inf
    est = pa_algorithm(cov.matrix, lower, upper, target=cov.roles.index("g"))
    assert np.isfinite(est[0])
    # conditioning on a low-liability mother must pull the proband's g down
    assert est[0] < 0.0
