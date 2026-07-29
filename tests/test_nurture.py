"""Direct vs indirect (genetic-nurture) covariance.

The reference matrix in :func:`test_matches_the_path_model` was verified against
a 4,000,000-family Monte-Carlo simulation of the path model itself (agreement to
8.5e-4, i.e. simulation noise), so these tests pin the closed form to the
generative model rather than to itself.
"""

import numpy as np
import pytest

from ltpred.covariance import (construct_covmat_nurture,
                               construct_covmat_single)

NUCLEAR = ("m", "f", "s1")


def _index(covmat):
    return {role: i for i, role in enumerate(covmat.roles)}


def test_matches_the_path_model():
    """h2 = 0.5, nurture = 0.35, checked against Monte Carlo of the path model."""
    got = construct_covmat_nurture(NUCLEAR, h2=0.5, nurture=0.35)
    want = np.array([
        [0.5000, 0.6750, 0.2500, 0.2500, 0.4250],
        [0.6750, 1.0000, 0.4250, 0.4250, 0.7225],
        [0.2500, 0.4250, 1.0000, 0.0000, 0.4250],
        [0.2500, 0.4250, 0.0000, 1.0000, 0.4250],
        [0.4250, 0.7225, 0.4250, 0.4250, 1.0000],
    ])
    assert got.roles == ["g", "o", "m", "f", "s1"]
    assert np.allclose(got.matrix, want, atol=1e-12)


@pytest.mark.parametrize("h2", [0.2, 0.5, 0.9])
def test_zero_nurture_reduces_to_the_additive_model_exactly(h2):
    """Switching the indirect path off must not perturb the established model."""
    got = construct_covmat_nurture(NUCLEAR, h2=h2, nurture=0.0)
    want = construct_covmat_single(NUCLEAR, h2=h2)
    assert got.roles == want.roles
    assert np.array_equal(got.matrix, want.matrix)


@pytest.mark.parametrize("h2", [0.05, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("n", [-0.3, -0.1, 0.0, 0.1, 0.3])
def test_positive_definite_across_valid_parameters(h2, n):
    if 1 - h2 - 2 * n * n * h2 - 2 * n * h2 < 0:
        pytest.skip("residual variance would be negative")
    cov = construct_covmat_nurture(("m", "f", "s1", "s2"), h2=h2, nurture=n).matrix
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 1e-10


def test_parent_offspring_and_sib_sib_inflate_by_different_amounts():
    """The unequal inflation is what identifies nurture in a nuclear family."""
    h2, n = 0.5, 0.3
    cov = construct_covmat_nurture(NUCLEAR, h2=h2, nurture=n)
    i = _index(cov)
    po = cov.matrix[i["o"], i["m"]]
    ss = cov.matrix[i["o"], i["s1"]]
    assert po == pytest.approx(h2 / 2 + n * h2)
    assert ss == pytest.approx(h2 / 2 + 2 * n * h2 + 2 * n * n * h2)
    # both exceed the kinship value, but not equally -- that is the signature
    assert po > h2 / 2 and ss > h2 / 2
    assert (ss - h2 / 2) > (po - h2 / 2)


def test_genetic_target_row_is_not_inflated_by_nurture():
    """Nurture changes how a parent's *liability* relates to the child, not how
    their *genotype* relates to the child's own genetic value."""
    h2, n = 0.5, 0.3
    cov = construct_covmat_nurture(NUCLEAR, h2=h2, nurture=n)
    i = _index(cov)
    assert cov.matrix[i["g"], i["m"]] == pytest.approx(h2 / 2)
    assert cov.matrix[i["g"], i["f"]] == pytest.approx(h2 / 2)
    # while the proband's own liability does absorb the shared nurture
    assert cov.matrix[i["g"], i["o"]] == pytest.approx(h2 * (1 + n))
    assert cov.matrix[i["g"], i["o"]] > cov.matrix[i["g"], i["g"]]


def test_this_is_not_expressible_as_kinship_times_a_scalar():
    """No single h2 makes the additive model reproduce the nurture covariance."""
    cov = construct_covmat_nurture(NUCLEAR, h2=0.5, nurture=0.3).matrix
    for h2_try in np.linspace(0.01, 1.0, 100):
        if np.allclose(construct_covmat_single(NUCLEAR, h2=h2_try).matrix, cov):
            raise AssertionError("unexpectedly matched an additive model")


def test_sibs_share_one_nurture_term():
    """All offspring of the couple get the same parental contribution."""
    cov = construct_covmat_nurture(("m", "f", "s1", "s2"), h2=0.5, nurture=0.3)
    i = _index(cov)
    ss = cov.matrix[i["s1"], i["s2"]]
    assert ss == pytest.approx(cov.matrix[i["o"], i["s1"]])
    assert cov.matrix[i["m"], i["s1"]] == pytest.approx(cov.matrix[i["m"], i["s2"]])


def test_parents_are_uncorrelated_random_mating():
    cov = construct_covmat_nurture(NUCLEAR, h2=0.5, nurture=0.3)
    i = _index(cov)
    assert cov.matrix[i["m"], i["f"]] == 0.0


def test_liabilities_stay_standardised():
    cov = construct_covmat_nurture(("m", "f", "s1", "s2"), h2=0.4, nurture=0.25)
    for role, i in _index(cov).items():
        if role != "g":
            assert cov.matrix[i, i] == pytest.approx(1.0)


def test_rejects_parameters_that_would_need_negative_residual_variance():
    with pytest.raises(ValueError, match="residual variance would be negative"):
        construct_covmat_nurture(NUCLEAR, h2=0.8, nurture=0.5)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_h2(bad):
    with pytest.raises(ValueError, match="h2 must be in"):
        construct_covmat_nurture(NUCLEAR, h2=bad, nurture=0.1)


@pytest.mark.parametrize("role", ["mgm", "pgf", "mhs1", "mau1"])
def test_rejects_non_nuclear_roles(role):
    """Silently treating a grandparent as a founder would understate the model."""
    with pytest.raises(ValueError, match="nuclear-only"):
        construct_covmat_nurture(("m", "f", role), h2=0.5, nurture=0.2)


def test_nurture_is_confounded_with_sibship_C_on_sibs_alone():
    """The documented identifiability trap, made executable.

    A sibship environment can be chosen to reproduce the nurture model's
    sib-sib covariance exactly. Only the parent-offspring covariance tells them
    apart -- `C` leaves it at the kinship value, nurture raises it.
    """
    h2, n = 0.5, 0.3
    nur = construct_covmat_nurture(NUCLEAR, h2=h2, nurture=n)
    i = _index(nur)
    sib_sib = nur.matrix[i["o"], i["s1"]]

    c2 = sib_sib - h2 / 2          # the C that mimics it
    ace = construct_covmat_single(NUCLEAR, h2=h2, c2=c2)
    j = _index(ace)

    # indistinguishable on sibs
    assert ace.matrix[j["o"], j["s1"]] == pytest.approx(sib_sib)
    # but not on parent-offspring
    assert ace.matrix[j["o"], j["m"]] == pytest.approx(h2 / 2)
    assert nur.matrix[i["o"], i["m"]] == pytest.approx(h2 / 2 + n * h2)
    assert nur.matrix[i["o"], i["m"]] > ace.matrix[j["o"], j["m"]]


def test_negative_nurture_is_allowed():
    """A contrast effect is a legitimate sign, and stays PSD."""
    cov = construct_covmat_nurture(NUCLEAR, h2=0.5, nurture=-0.2)
    i = _index(cov)
    assert cov.matrix[i["o"], i["m"]] < 0.5 / 2 + 1e-12
    assert np.linalg.eigvalsh(cov.matrix).min() > 1e-10
