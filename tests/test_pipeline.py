"""Supported trio-register pipeline: observation-set and calendar contracts."""

import warnings

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
        h2=0.5,
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


@pytest.mark.parametrize("h2", [.5, 1., 1. - 1e-10, 1e-6])
def test_selected_pipeline_preserves_inbreeding_and_legacy_covariance_repair(h2):
    # The parents are full siblings; the grandparents are closure-only.
    ids = ["o", "a", "b", "gm", "gf"]
    father = ["a", "gf", "gf", None, None]
    mother = ["b", "gm", "gm", None, None]
    kwargs = pipeline_kwargs(ids=ids, father=father, mother=mother,
                             status=[1, 0, 1, 0, 0], age=[40., 60., 55., 85., 90.],
                             h2=h2)
    graph = build_parent_graph(ids, father, mother)
    ped = extract_pedigree(graph, "o", max_degree=1)
    rows = np.array([graph.index[i] for i in ped.ids])
    lower, upper, _, _ = thresholds_from_cip(
        np.asarray(kwargs["status"])[rows], np.asarray(kwargs["age"])[rows],
        CIP_AGES, CIP_VALUES, k_pop=K_POP)
    lower[ped.closure_only], upper[ped.closure_only] = -np.inf, np.inf
    _, full = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected, _, variance = estimate_liability_from_kinship(full, lower, upper, h2=h2)
        result = estimate_liabilities(**kwargs)
    np.testing.assert_allclose(result.est, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.var, variance, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(result.n_closure_only, [2])


def test_selected_pipeline_preserves_rejection_of_unrepairable_covariance():
    with pytest.raises(ValueError, match="unable to enforce"):
        estimate_liabilities(**pipeline_kwargs(h2=1e-10))


def test_pipeline_selected_relationships_are_bounded_and_cache_size_independent(monkeypatch):
    import ltpred.pipeline as pipeline_module

    reference = estimate_liabilities(**pipeline_kwargs(probands=["o", "m", "f"]))

    # Beyond the dense cap the safely-PD path must not build the closure A;
    # ancestors still determine the selected entries, whatever the cache.
    def forbidden(*args, **kwargs):
        raise AssertionError("large-pedigree inference constructed full closure kinship")

    monkeypatch.setattr(pipeline_module, "_DENSE_KINSHIP_MAX_MEMBERS", 0)
    monkeypatch.setattr(pipeline_module, "kinship_from_pedigree", forbidden)
    for size in [0, 1, 5]:
        result = estimate_liabilities(**pipeline_kwargs(
            probands=["o", "m", "f"], kinship_cache_size=size))
        np.testing.assert_array_equal(result.est, reference.est)
        np.testing.assert_array_equal(result.var, reference.var)


def test_pipeline_rejects_cycle_outside_selected_proband_component():
    with pytest.raises(ValueError, match="cycle"):
        estimate_liabilities(**pipeline_kwargs(
            ids=["o", "a", "b"], father=[None, "b", "a"],
            mother=[None, None, None], status=[0, 0, 0], age=[50., 50., 50.]))


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
        h2=0.5, max_degree=1)

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
        cip_values=CIP_VALUES, k_pop=K_POP, h2=0.5, max_degree=1)

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
        cip_values=CIP_VALUES, k_pop=K_POP, h2=0.5, max_degree=1)
    np.testing.assert_allclose(out.est, [0.0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(out.n_relatives, [1])
    np.testing.assert_array_equal(out.n_conditioned, [0])
    np.testing.assert_array_equal(out.n_closure_only, [1])
    np.testing.assert_array_equal(out.degree_max, [1])


def test_gwas_conditions_on_the_proband_status():
    common = dict(
        ids=["o"], father=[None], mother=[None], probands=["o"], age=[60.0],
        use="gwas", cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP,
        h2=0.5, max_degree=1)
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
        strata=strata, cip_by_stratum=curves, h2=0.5, max_degree=1)

    order = np.arange(len(IDS))[::-1]
    permuted = estimate_liabilities(
        [IDS[i] for i in order], [FATHER[i] for i in order],
        [MOTHER[i] for i in order], probands=["o"], status=STATUS[order],
        age=AGE[order], use="prediction", birth_time=BIRTH[order],
        index_time=[2020.0], strata=strata[order],
        cip_by_stratum=curves, h2=0.5, max_degree=1)
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


def test_unresolved_parents_are_surfaced_and_warned():
    # a and b carry unlisted non-null parents (2 of 3 records); c's resolve.
    ids = ["a", "b", "c"]
    father = ["x", "y", "a"]
    mother = [None, None, "b"]
    with pytest.warns(UserWarning, match="no id"):
        out = estimate_liabilities(
            ids, father, mother, probands=["c"],
            status=np.array([0, 0, 0]), age=np.array([65.0, 70.0, 45.0]),
            use="gwas", cip_ages=CIP_AGES, cip_values=CIP_VALUES,
            k_pop=K_POP, h2=0.5, max_degree=1)
    assert out.frac_records_with_unresolved_parents == pytest.approx(2.0 / 3.0)
    assert build_parent_graph(ids, father, mother).n_unresolved_parents == 2


def test_all_unresolved_parent_references_are_refused():
    # Integer ids with string parent references can never match; a register
    # boundary cannot explain zero resolved references, so this is refused.
    with pytest.raises(ValueError, match="0 matched"):
        estimate_liabilities(
            [1, 2, 3], ["3", None, None], ["2", None, None], probands=[1],
            status=np.array([0, 0, 0]), age=np.array([40.0, 65.0, 70.0]),
            use="gwas", cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP,
            h2=0.5)


def test_declared_unknown_founders_stay_silent():
    # The standard table's founders are all None: no diagnostic may fire.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = estimate_liabilities(**pipeline_kwargs())
        prediction = estimate_liabilities(**pipeline_kwargs(
            use="prediction", birth_time=BIRTH, index_time=[2020.0]))
    assert out.frac_records_with_unresolved_parents == 0.0
    assert prediction.frac_records_with_unresolved_parents == 0.0
    np.testing.assert_array_equal(prediction.proband_state,
                                  ["disease_free_and_followed"])


def test_proband_state_is_prediction_only():
    out = estimate_liabilities(**pipeline_kwargs())
    assert out.proband_state is None


def _trio_prediction(proband_status, proband_age, probands=None):
    ids = ["o", "m", "f"]
    common = dict(
        father=["f", None, None], mother=["m", None, None],
        probands=probands or ["o"], use="prediction",
        birth_time=np.array([1980.0, 1950.0, 1950.0]),
        index_time=np.array([2020.0] * len(probands or ["o"])),
        cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP, h2=0.5,
        max_degree=1)
    status = np.array([proband_status, 0, 0])
    age = np.array([proband_age, 65.0, 70.0])
    return estimate_liabilities(ids, status=status, age=age, **common)


def test_prediction_flags_prevalent_proband_without_changing_scores():
    # Proband diagnosed 2015, landmark 2020: not at risk, and now said so.
    with pytest.warns(UserWarning, match="prevalent_case"):
        prevalent = _trio_prediction(1, 35.0)
    np.testing.assert_array_equal(prevalent.proband_state, ["prevalent_case"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clean = _trio_prediction(1, 45.0)   # onset 2025, after the landmark
    np.testing.assert_array_equal(clean.proband_state,
                                  ["disease_free_and_followed"])
    # The diagnostic changes nothing the driver computes.
    np.testing.assert_allclose(prevalent.est, clean.est, rtol=0, atol=0)
    np.testing.assert_allclose(prevalent.var, clean.var, rtol=0, atol=0)


def test_prediction_flags_proband_exited_before_index():
    # Follow-up ended 2010, landmark 2020: named, but not a prevalent case.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _trio_prediction(0, 30.0)
    np.testing.assert_array_equal(out.proband_state, ["exited_before_index"])


@pytest.mark.parametrize("status,age,expected", [
    (1, 40.0, "prevalent_case"),            # onset exactly at the landmark
    (0, 40.0, "disease_free_and_followed"),  # follow-up ends exactly at it
])
def test_proband_state_landmark_boundary(status, age, expected):
    kwargs = dict(
        ids=["o"], father=[None], mother=[None], probands=["o"],
        status=np.array([status]), age=np.array([age]), use="prediction",
        birth_time=np.array([1980.0]), index_time=np.array([2020.0]),
        cip_ages=CIP_AGES, cip_values=CIP_VALUES, k_pop=K_POP, h2=0.5,
        max_degree=1)
    if expected == "prevalent_case":
        with pytest.warns(UserWarning, match="prevalent_case"):
            out = estimate_liabilities(**kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = estimate_liabilities(**kwargs)
    np.testing.assert_array_equal(out.proband_state, [expected])


def test_proband_state_aligns_to_probands():
    ids = ["o1", "o2"]
    with pytest.warns(UserWarning, match="1 of 2"):
        out = estimate_liabilities(
            ids, [None, None], [None, None], probands=["o1", "o2"],
            status=np.array([1, 0]), age=np.array([35.0, 60.0]),
            use="prediction", birth_time=np.array([1980.0, 1980.0]),
            index_time=np.array([2020.0, 2020.0]), cip_ages=CIP_AGES,
            cip_values=CIP_VALUES, k_pop=K_POP, h2=0.5, max_degree=1)
    np.testing.assert_array_equal(
        out.proband_state, ["prevalent_case", "disease_free_and_followed"])
