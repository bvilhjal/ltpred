"""Analytic oracles for Gaussian observation reduction before interval PA."""
import numpy as np
import pytest
from scipy.stats import truncnorm

from ltpred.covariance import construct_covmat_single
from ltpred.pearson_aitken import pa_algorithm, pa_estimate_batched


@pytest.mark.parametrize('h2', [.2, .5, .8])
@pytest.mark.parametrize('pin', [1., 3., 4.])
def test_pin_plus_interval_matches_conditional_gaussian_oracle(h2, pin):
    cov = construct_covmat_single(['o', 'm'], h2=h2).matrix
    # g and mother's o conditional on the proband's full-liability pin.
    retained = [0, 2]
    m = cov[retained, 1] * pin
    v = cov[np.ix_(retained, retained)] - np.outer(cov[retained, 1], cov[1, retained])
    upper = 2.
    selected = truncnorm(-np.inf, (upper-m[1])/np.sqrt(v[1,1]), loc=m[1], scale=np.sqrt(v[1,1]))
    beta = v[0,1]/v[1,1]
    oracle = (m[0]+beta*(selected.mean()-m[1]),
              v[0,0]+beta**2*(selected.var()-v[1,1]))
    assert pa_algorithm(cov, [-np.inf,pin,-np.inf], [np.inf,pin,upper]) == pytest.approx(oracle, abs=2e-13)


def test_all_pins_match_joint_gaussian_conditioning_and_pin_order():
    rng = np.random.default_rng(48231)
    B = rng.normal(size=(6,6)); cov = B @ B.T + np.eye(6)
    values = rng.normal(size=(7,5))
    lo=np.c_[np.full(7,-np.inf),values];hi=np.c_[np.full(7,np.inf),values]
    weights=np.linalg.solve(cov[1:,1:],cov[1:,0])
    m=values@weights;v=cov[0,0]-cov[0,1:]@weights
    actual=pa_estimate_batched(cov,lo,hi)
    np.testing.assert_allclose(actual[0],m,rtol=0,atol=2e-14)
    np.testing.assert_allclose(actual[1],v,rtol=0,atol=2e-14)
    order=[0,3,5,2,1,4]
    perm=pa_estimate_batched(cov[np.ix_(order,order)],lo[:,order],hi[:,order])
    np.testing.assert_allclose(perm,actual,rtol=0,atol=2e-14)


def test_unobserved_coordinates_are_marginalized_without_changing_order():
    rng=np.random.default_rng(831)
    B=rng.normal(size=(10,10));cov=B@B.T+np.eye(10)
    keep=[0,2,4,8]
    lo=np.full(10,-np.inf);hi=np.full(10,np.inf)
    lo[[2,8]]=[1.,-.4];hi[4]=.2
    full=pa_algorithm(cov,lo,hi)
    reduced=pa_algorithm(cov[np.ix_(keep,keep)],lo[keep],hi[keep])
    np.testing.assert_array_equal(full,reduced)


def test_mixed_masks_and_nonzero_target_match_separate_calls():
    rng=np.random.default_rng(53)
    B=rng.normal(size=(4,4));cov=B@B.T+np.eye(4)
    lo=np.array([[-np.inf,1,-np.inf,2],[-np.inf,-np.inf,-1,-np.inf],
                 [-np.inf,2,3,0],[-np.inf,-np.inf,-np.inf,-np.inf]])
    hi=np.array([[np.inf,1,.5,2],[np.inf,.2,np.inf,np.inf],
                 [np.inf,2,3,0],[np.inf,np.inf,np.inf,np.inf]])
    for target in range(4):
        batch=pa_estimate_batched(cov,lo,hi,target=target)
        separate=np.array([pa_algorithm(cov,l,u,target=target) for l,u in zip(lo,hi)]).T
        np.testing.assert_allclose(batch,separate,rtol=0,atol=1e-14)


def test_singular_compatible_pins_have_exact_deterministic_target():
    cov=np.ones((3,3))
    assert pa_algorithm(cov,[-np.inf,2,2],[np.inf,2,2]) == pytest.approx((2.,0.),abs=1e-14)
    with pytest.raises(ValueError,match='incompatible'):
        pa_algorithm(cov,[-np.inf,2,3],[np.inf,2,3])
    with pytest.raises(ValueError,match='deterministic'):
        pa_algorithm(cov,[-np.inf,2,-np.inf],[np.inf,2,1])


def test_repeated_singular_pins_reject_impossible_remaining_observation():
    # Spectral subtraction previously left a variance of 2.22e-16 and treated
    # an impossible interval as merely extremely unlikely, returning ~1.
    with pytest.raises(ValueError, match="deterministic"):
        pa_algorithm(np.ones((4, 4)), [-np.inf, 2., 2., -np.inf],
                     [np.inf, 2., 2., 1.])


def test_singular_pin_support_tolerance_tracks_roundoff_at_large_values():
    with pytest.raises(ValueError, match="incompatible"):
        pa_algorithm(np.ones((3, 3)), [-np.inf, 1e10, 1e10 + .1],
                     [np.inf, 1e10, 1e10 + .1])


def test_scaled_singular_pins_preserve_support_and_boundary_observations():
    factor = np.array([1., .3, .7, 1.3])
    cov = np.outer(factor, factor)
    lower = [-np.inf, .6, 1.4, -np.inf]
    assert pa_algorithm(cov, lower, [np.inf, .6, 1.4, 2.6]) == pytest.approx((2., 0.), abs=1e-14)
    with pytest.raises(ValueError, match="deterministic"):
        pa_algorithm(cov, lower, [np.inf, .6, 1.4, 1.3])


def test_floating_point_rank_one_covariances_reject_impossible_evidence():
    rng = np.random.default_rng(236)
    for _ in range(100):
        factor = rng.uniform(.1, 2., 4)
        cov = np.outer(factor, factor)
        lower = [-np.inf, 2 * factor[1], 2 * factor[2], -np.inf]
        with pytest.raises(ValueError, match="deterministic"):
            pa_algorithm(cov, lower,
                         [np.inf, 2 * factor[1], 2 * factor[2], factor[3]])
        actual = pa_algorithm(cov, lower,
                              [np.inf, 2 * factor[1], 2 * factor[2], 3 * factor[3]])
        assert actual == pytest.approx((2 * factor[0], 0.), abs=1e-13)


@pytest.mark.parametrize("epsilon", [1e-10, 1e-14, 1e-15])
def test_small_positive_conditional_variance_is_not_rounded_to_determinism(epsilon):
    cov = np.array([[1. + epsilon, 1.], [1., 1.]])
    mean, variance = pa_algorithm(cov, [-np.inf, 2.], [np.inf, 2.])
    assert mean == pytest.approx(2.)
    assert variance > 0.
    # Near machine precision, conditioning loses relative digits. It must
    # nevertheless preserve the positive support, not return a point mass.
    assert variance == pytest.approx(cov[0, 0] - 1., rel=.25)


def test_singular_pin_block_does_not_discard_independent_small_variance():
    cov = np.array([[1e-20, 0., 0.], [0., 1., 1.], [0., 1., 1.]])
    assert pa_algorithm(cov, [-np.inf, 2., 2.], [np.inf, 2., 2.]) == pytest.approx(
        (0., 1e-20), rel=1e-14, abs=0.)


def test_small_variance_pins_are_not_lost_in_singular_block():
    # The independent low-variance pin matters for the target even though
    # the two ordinary-variance pins are redundant.
    cov = np.array([[1., 5e-11, 0., 0.], [5e-11, 1e-20, 0., 0.],
                    [0., 0., 1., 1.], [0., 0., 1., 1.]])
    assert pa_algorithm(cov, [-np.inf, 1e-10, 2., 2.],
                        [np.inf, 1e-10, 2., 2.]) == pytest.approx((.5, .75))


def test_target_pin_is_not_approximated_by_other_intervals():
    cov=np.array([[1.,.7],[.7,1.]])
    assert pa_algorithm(cov,[2.,-np.inf],[2.,.1]) == (2.,0.)


def test_empty_batch_and_all_unobserved():
    cov=np.array([[.5,.2],[.2,1.]])
    e,v=pa_estimate_batched(cov,np.empty((0,2)),np.empty((0,2)))
    assert e.size==v.size==0
    assert pa_algorithm(cov,[-np.inf,-np.inf],[np.inf,np.inf]) == pytest.approx((0.,.5))
