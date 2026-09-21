"""Regression checks for the register benchmark's calendar-time design."""

import numpy as np

from _helpers import load_script
from ltpred.covariance import kinship_from_pedigree


MODULE = load_script("benchmarks/bench_register_pipeline.py")


def test_simulated_birth_times_put_coparents_together_and_children_later():
    ids = ["f", "m", "c", "sp", "gc"]
    father = [None, None, "f", None, "c"]
    mother = [None, None, "m", None, "sp"]

    birth = MODULE.pedigree_birth_times(ids, father, mother)

    np.testing.assert_array_equal(birth, [1920.0, 1920.0, 1950.0,
                                          1950.0, 1980.0])


def test_prior_future_risk_is_cip_increment_conditional_on_survival():
    risk = MODULE.future_case_prob(
        est_g=np.array([0.0]), est_var=np.array([MODULE.H2]))
    cip_index = np.interp(MODULE.INDEX_AGE, MODULE.AGE_GRID, MODULE.TRUE_CIP)
    cip_eval = np.interp(MODULE.EVAL_AGE, MODULE.AGE_GRID, MODULE.TRUE_CIP)
    expected = (cip_eval - cip_index) / (1.0 - cip_index)

    np.testing.assert_allclose(risk, [expected], rtol=1e-14, atol=0.0)


def test_build_register_uses_the_public_inbreeding_scale():
    # x is the child of full siblings, so A_xx = 1.25. Reproduce the two RNG
    # draws and require both the simulated truth and observation process to use
    # the unit-full-liability scale used by construct_covmat_from_kinship.
    ids = ["gm", "gf", "sA", "sB", "x"]
    father = [None, None, "gf", "gf", "sA"]
    mother = [None, None, "gm", "gm", "sB"]
    seed = 913
    observed = MODULE.build_register(
        np.random.default_rng(seed), ids, father, mother)
    status, age, onset, genetic, _, residual_var = observed

    _, relationship = kinship_from_pedigree(ids, father, mother)
    replay = np.random.default_rng(seed)
    raw_genetic = replay.multivariate_normal(
        np.zeros(len(ids)), MODULE.H2 * relationship)
    residual = replay.standard_normal(len(ids)) * np.sqrt(1.0 - MODULE.H2)
    scale = np.sqrt(
        MODULE.H2 * np.diag(relationship) + (1.0 - MODULE.H2))
    liability = (raw_genetic + residual) / scale

    assert scale[-1] > 1.0
    np.testing.assert_allclose(genetic, raw_genetic / scale, rtol=0, atol=0)
    np.testing.assert_allclose(
        residual_var, (1.0 - MODULE.H2) / scale ** 2, rtol=0, atol=0)
    raw_full_cov = (MODULE.H2 * relationship
                    + (1.0 - MODULE.H2) * np.eye(len(ids)))
    standardized_cov = raw_full_cov / np.outer(scale, scale)
    np.testing.assert_allclose(
        np.diag(standardized_cov), np.ones(len(ids)), rtol=0, atol=1e-15)

    need = 1.0 - MODULE.norm.cdf(liability)
    expected_onset = np.full(len(ids), np.inf)
    event = need <= MODULE.TRUE_CIP[-1]
    expected_onset[event] = np.maximum(
        np.interp(need[event], MODULE.TRUE_CIP, MODULE.AGE_GRID), 1e-9)
    expected_status = expected_onset <= MODULE.EVAL_AGE
    expected_age = np.where(expected_status, expected_onset, MODULE.EVAL_AGE)
    np.testing.assert_array_equal(status, expected_status)
    np.testing.assert_allclose(age, expected_age, rtol=0, atol=0)
    np.testing.assert_allclose(onset, expected_onset, rtol=0, atol=0)


def test_inbred_prior_future_risk_uses_target_specific_residual_variance():
    self_relationship = 1.25
    scale2 = MODULE.H2 * self_relationship + (1.0 - MODULE.H2)
    genetic_var = MODULE.H2 * self_relationship / scale2
    residual_var = (1.0 - MODULE.H2) / scale2
    risk = MODULE.future_case_prob(
        est_g=np.array([0.0]), est_var=np.array([genetic_var]),
        residual_var=np.array([residual_var]))
    cip_index = np.interp(MODULE.INDEX_AGE, MODULE.AGE_GRID, MODULE.TRUE_CIP)
    cip_eval = np.interp(MODULE.EVAL_AGE, MODULE.AGE_GRID, MODULE.TRUE_CIP)
    expected = (cip_eval - cip_index) / (1.0 - cip_index)

    np.testing.assert_allclose(risk, [expected], rtol=1e-14, atol=0.0)


def test_benchmark_generators_are_the_public_ones():
    """The ledger and the tutorial must not be able to drift apart.

    ``build_register`` and ``pedigree_birth_times`` here are thin bindings of
    this benchmark's constants around :mod:`ltpred.simulate`. If either stops
    delegating, the promoted public generator and the published benchmark would
    be two different simulations producing incomparable numbers -- so pin the
    delegation bit for bit rather than trusting the import line.
    """
    from ltpred.simulate import pedigree_birth_times, simulate_register_liabilities

    ids, father, mother = MODULE.simulate_population(
        np.random.default_rng(17), n_founder_pairs=25, gens=2)

    np.testing.assert_array_equal(
        MODULE.pedigree_birth_times(ids, father, mother),
        pedigree_birth_times(ids, father, mother,
                             base_birth_year=MODULE.BASE_BIRTH_TIME,
                             generation_years=MODULE.GENERATION_YEARS))

    bound = simulate_register_liabilities(
        np.random.default_rng(23), ids, father, mother, h2=MODULE.H2,
        cip_ages=MODULE.AGE_GRID, cip_values=MODULE.TRUE_CIP,
        eval_age=MODULE.EVAL_AGE)
    expected = (bound.status, bound.age, bound.onset, bound.genetic,
                bound.birth_time, bound.residual_var)
    observed = MODULE.build_register(np.random.default_rng(23), ids, father,
                                     mother)
    assert len(observed) == len(expected)
    for got, want in zip(observed, expected):
        # atol=0/rtol=0: same generator, same seed, same constants must give the
        # same bits, not merely the same numbers to within a tolerance
        np.testing.assert_array_equal(got, want)
