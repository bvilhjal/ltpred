"""Supported trio-register pipeline: observation-set and calendar contracts."""

import numpy as np
import pytest

from ltpred import (build_parent_graph, estimate_liabilities,
                    estimate_liability_from_kinship, extract_pedigree,
                    kinship_from_pedigree, thresholds_from_cip)


IDS = ["o", "m", "f", "mgm", "mgf", "pgm", "pgf"]
FATHER = ["f", "mgf", "pgf", None, None, None, None]
MOTHER = ["m", "mgm", "pgm", None, None, None, None]
STATUS = np.array([1, 0, 0, 1, 0, 0, 0])
AGE = np.array([45.0, 70.0, 70.0, 65.0, 75.0, 80.0, 78.0])
BIRTH = np.array([1980.0, 1950.0, 1950.0, 1925.0, 1925.0, 1925.0, 1925.0])
CIP_AGES = np.arange(0.0, 121.0)
K_POP = 0.10
CIP_VALUES = K_POP / (1.0 + np.exp((60.0 - CIP_AGES) / 8.0))


def pipeline_kwargs(**updates):
    kwargs = dict(
        ids=IDS,
        father=FATHER,
        mother=MOTHER,
        probands=["o"],
        status=STATUS,
        age=AGE,
        use="gwas",
        cip_ages=CIP_AGES,
        cip_values=CIP_VALUES,
        k_pop=K_POP,
        max_degree=1,
    )
    kwargs.update(updates)
    return kwargs


def test_gwas_matches_manual_public_pieces_and_reports_observation_counts():
    out = estimate_liabilities(**pipeline_kwargs())

    graph = build_parent_graph(IDS, FATHER, MOTHER)
    ped = extract_pedigree(graph, "o", max_degree=1)
    member_index = np.array([graph.index[pid] for pid in ped.ids])
    lower, upper, _, _ = thresholds_from_cip(
        STATUS[member_index], AGE[member_index], CIP_AGES, CIP_VALUES,
        k_pop=K_POP, case_mode="pin")
    lower[ped.closure_only] = -np.inf
    upper[ped.closure_only] = np.inf
    _, relationship = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
    expected, _, expected_var = estimate_liability_from_kinship(
        relationship, lower[None, :], upper[None, :], h2=0.5,
        method="pearson-aitken")

    np.testing.assert_allclose(out.est, expected, rtol=0, atol=0)
    np.testing.assert_allclose(out.var, expected_var, rtol=0, atol=0)
    np.testing.assert_array_equal(out.n_relatives, [2])
    np.testing.assert_array_equal(out.n_conditioned, [3])
    np.testing.assert_array_equal(out.n_closure_only, [4])
    np.testing.assert_array_equal(out.degree_max, [1])


def test_closure_diagnoses_are_ignored_by_default_and_explicitly_opted_in():
    altered_status = STATUS.copy()
    altered_age = AGE.copy()
    altered_status[3:] = 1 - altered_status[3:]
    altered_age[3:] = [35.0, 40.0, 45.0, 50.0]

    base = estimate_liabilities(**pipeline_kwargs())
    altered = estimate_liabilities(**pipeline_kwargs(
        status=altered_status, age=altered_age))
    np.testing.assert_allclose(base.est, altered.est, rtol=0, atol=0)
    np.testing.assert_allclose(base.var, altered.var, rtol=0, atol=0)

    base_conditioned = estimate_liabilities(**pipeline_kwargs(
        condition_closure=True))
    altered_conditioned = estimate_liabilities(**pipeline_kwargs(
        status=altered_status, age=altered_age, condition_closure=True))
    assert not np.isclose(base_conditioned.est[0], altered_conditioned.est[0])
    np.testing.assert_array_equal(base_conditioned.n_conditioned, [7])


def test_calendar_prediction_uses_each_relatives_attained_age_at_index():
    # At the proband's 2020 landmark (age 40), the mother was already a case
    # (2015) while the father's 2025 diagnosis was still in the future. The
    # father is therefore a control at age 70, not at the proband's age 40.
    ids = ["o", "m", "f"]
    father = ["f", None, None]
    mother = ["m", None, None]
    status = np.array([1, 1, 1])
    age = np.array([45.0, 65.0, 75.0])
    birth = np.array([1980.0, 1950.0, 1950.0])
    out = estimate_liabilities(
        ids, father, mother, probands=["o"], status=status, age=age,
        use="prediction", birth_time=birth, index_time=np.array([2020.0]),
        cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP,
        max_degree=1)

    graph = build_parent_graph(ids, father, mother)
    ped = extract_pedigree(graph, "o", max_degree=1)
    assert ped.ids == ["o", "f", "m"]
    expected_status = np.array([0, 0, 1])
    expected_age = np.array([40.0, 70.0, 65.0])
    lower, upper, _, _ = thresholds_from_cip(
        expected_status, expected_age, CIP_AGES, CIP_VALUES,
        k_pop=K_POP, case_mode="pin")
    lower[0], upper[0] = -np.inf, np.inf
    _, relationship = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
    expected, _, expected_var = estimate_liability_from_kinship(
        relationship, lower[None, :], upper[None, :], h2=0.5,
        method="pearson-aitken")

    np.testing.assert_allclose(out.est, expected, rtol=0, atol=0)
    np.testing.assert_allclose(out.var, expected_var, rtol=0, atol=0)
    np.testing.assert_array_equal(out.n_conditioned, [2])


def test_prediction_is_invariant_to_post_index_outcomes_but_not_preindex_history():
    ids = ["o", "m", "f"]
    father = ["f", None, None]
    mother = ["m", None, None]
    common = dict(
        ids=ids, father=father, mother=mother, probands=["o"],
        use="prediction", birth_time=np.array([1980.0, 1950.0, 1950.0]),
        index_time=np.array([2020.0]), cip_ages=CIP_AGES,
        cip_values=CIP_VALUES, k_pop=K_POP, max_degree=1)

    # Proband and father differ only after 2020. Both versions reduce to the
    # same observation set at the landmark; the mother's 2015 case is shared.
    later_cases = estimate_liabilities(
        status=np.array([1, 1, 1]), age=np.array([45.0, 65.0, 75.0]), **common)
    later_controls = estimate_liabilities(
        status=np.array([0, 1, 0]), age=np.array([80.0, 65.0, 80.0]), **common)
    np.testing.assert_allclose(later_cases.est, later_controls.est, rtol=0, atol=0)
    np.testing.assert_allclose(later_cases.var, later_controls.var, rtol=0, atol=0)

    no_preindex_case = estimate_liabilities(
        status=np.array([0, 0, 0]), age=np.array([80.0, 65.0, 80.0]), **common)
    assert not np.isclose(later_cases.est[0], no_preindex_case.est[0])


@pytest.mark.parametrize("child_birth", [2020.0, 2025.0])
def test_prediction_leaves_newborn_and_not_yet_born_members_uninformative(child_birth):
    ids = ["o", "c", "sp"]
    father = [None, "o", None]
    mother = [None, "sp", None]
    out = estimate_liabilities(
        ids, father, mother, probands=["o"], status=np.array([1, 1, 1]),
        age=np.array([45.0, 5.0, 50.0]), use="prediction",
        birth_time=np.array([1980.0, child_birth, 1980.0]),
        index_time=np.array([2020.0]), cip_ages=CIP_AGES,
        cip_values=CIP_VALUES, k_pop=K_POP, max_degree=1)
    np.testing.assert_allclose(out.est, [0.0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(out.n_relatives, [1])
    np.testing.assert_array_equal(out.n_conditioned, [0])
    np.testing.assert_array_equal(out.n_closure_only, [1])
    np.testing.assert_array_equal(out.degree_max, [1])


def test_gwas_conditions_on_the_proband_status():
    common = dict(
        ids=["o"], father=[None], mother=[None], probands=["o"], age=[60.0],
        use="gwas", cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP,
        max_degree=1)
    case = estimate_liabilities(status=[1], **common)
    control = estimate_liabilities(status=[0], **common)
    assert case.est[0] > control.est[0]
    np.testing.assert_array_equal(case.n_conditioned, [1])


def test_population_record_permutation_preserves_stratified_prediction():
    strata = np.array(["A", "B", "A", "B", "A", "B", "A"])
    curve_a = (CIP_AGES, CIP_VALUES, K_POP)
    curve_b_values = 0.15 / (1.0 + np.exp((55.0 - CIP_AGES) / 9.0))
    curve_b = (CIP_AGES, curve_b_values, 0.15)
    curves = {"A": curve_a, "B": curve_b}
    base = estimate_liabilities(
        IDS, FATHER, MOTHER, probands=["o"], status=STATUS, age=AGE,
        use="prediction", birth_time=BIRTH, index_time=[2020.0],
        strata=strata, cip_by_stratum=curves, max_degree=1)

    order = np.arange(len(IDS))[::-1]
    permuted = estimate_liabilities(
        [IDS[i] for i in order], [FATHER[i] for i in order],
        [MOTHER[i] for i in order], probands=["o"], status=STATUS[order],
        age=AGE[order], use="prediction", birth_time=BIRTH[order],
        index_time=[2020.0], strata=strata[order],
        cip_by_stratum=curves, max_degree=1)
    np.testing.assert_allclose(base.est, permuted.est, rtol=0, atol=0)
    np.testing.assert_allclose(base.var, permuted.var, rtol=0, atol=0)


@pytest.mark.parametrize("bad_status", [
    [2, 0, 0, 1, 0, 0, 0],
    [np.nan, 0, 0, 1, 0, 0, 0],
])
def test_pipeline_rejects_nonbinary_status(bad_status):
    with pytest.raises(ValueError, match="boolean or 0/1"):
        estimate_liabilities(**pipeline_kwargs(status=bad_status))


def test_pipeline_rejects_invalid_use_calendar_and_closure_inputs():
    with pytest.raises(ValueError, match="use must"):
        estimate_liabilities(**pipeline_kwargs(use="risk"))
    with pytest.raises(ValueError, match="requires birth_time and index_time"):
        estimate_liabilities(**pipeline_kwargs(use="prediction"))
    with pytest.raises(ValueError, match="aligned to ids"):
        estimate_liabilities(**pipeline_kwargs(
            use="prediction", birth_time=BIRTH[:-1], index_time=[2020.0]))
    with pytest.raises(ValueError, match="aligned to probands"):
        estimate_liabilities(**pipeline_kwargs(
            use="prediction", birth_time=BIRTH, index_time=[2020.0, 2021.0]))
    with pytest.raises(ValueError, match="must not precede"):
        estimate_liabilities(**pipeline_kwargs(
            use="prediction", birth_time=BIRTH, index_time=[1979.0]))
    with pytest.raises(ValueError, match="must be omitted"):
        estimate_liabilities(**pipeline_kwargs(birth_time=BIRTH))
    with pytest.raises(TypeError, match="must be boolean"):
        estimate_liabilities(**pipeline_kwargs(condition_closure=1))
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_liabilities(**pipeline_kwargs(age=[-1.0] + AGE[1:].tolist()))


def test_pipeline_rejects_mixed_or_malformed_cip_routing():
    strata = np.array(["A"] * len(IDS))
    with pytest.raises(ValueError, match="mutually exclusive"):
        estimate_liabilities(**pipeline_kwargs(
            strata=strata,
            cip_by_stratum={"A": (CIP_AGES, CIP_VALUES, K_POP)}))
    base = pipeline_kwargs()
    base.pop("cip_ages")
    base.pop("cip_values")
    base.pop("k_pop")
    with pytest.raises(
            ValueError,
            match=r"must be \(cip_ages, cip_values, k_pop\)"):
        estimate_liabilities(**base, strata=strata,
                             cip_by_stratum={"A": (CIP_AGES, CIP_VALUES)})
