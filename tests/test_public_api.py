"""The top-level namespace stays small without breaking explicit imports."""


def test_primary_top_level_api_is_curated():
    import ltpred

    assert len(ltpred.__all__) <= 20
    assert "estimate_liability" in ltpred.__all__
    assert "thresholds_from_cip" in ltpred.__all__
    assert "rtmvnorm_gibbs" not in ltpred.__all__
    assert dir(ltpred) == sorted(ltpred.__all__)


def test_advanced_explicit_top_level_imports_remain_compatible():
    from ltpred import fit_variance_components, pa_algorithm, rtmvnorm_gibbs

    assert callable(fit_variance_components)
    assert callable(pa_algorithm)
    assert callable(rtmvnorm_gibbs)


def test_demoted_and_removed_names_are_gone_from_the_top_level():
    import ltpred
    import pytest

    for name in ("construct_covmat", "construct_covmat_sex_limited",
                 "construct_covmat_nurture", "extract_pedigrees",
                 "estimate_liabilities", "PopulationScores",
                 "estimate_liability_pa", "liability_sensitivity",
                 "SensitivityResult", "convert_observed_to_liability_scale",
                 "tnorm_moments", "tnorm_mixture_conditional",
                 "fit_genetic_correlation", "fit_genetic_factor",
                 "fit_nurture", "test_variance_component",
                 "observed_to_liability_gencov", "observed_to_liability_rg"):
        with pytest.raises(AttributeError):
            getattr(ltpred, name)
