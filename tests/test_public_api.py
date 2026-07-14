"""The top-level namespace stays small without breaking explicit imports."""


def test_primary_top_level_api_is_curated():
    import ltpred

    assert len(ltpred.__all__) <= 20
    assert "estimate_liability" in ltpred.__all__
    assert "thresholds_from_cip" in ltpred.__all__
    assert "rtmvnorm_gibbs" not in ltpred.__all__
    assert dir(ltpred) == sorted(ltpred.__all__)


def test_advanced_explicit_top_level_imports_remain_compatible():
    from ltpred import fit_genetic_factor, pa_algorithm, rtmvnorm_gibbs

    assert callable(fit_genetic_factor)
    assert callable(pa_algorithm)
    assert callable(rtmvnorm_gibbs)
