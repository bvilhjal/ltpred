"""Keep the numerical example's small public-register example executable."""

import numpy as np

from _helpers import load_script


EXAMPLE = load_script("examples/validation.py")


def test_register_example_preserves_calendar_and_closure_contracts():
    gwas, prediction, changed, risk, _ = EXAMPLE.register_example()

    assert prediction.probands == ["p"]
    np.testing.assert_array_equal(prediction.n_relatives, [3])
    np.testing.assert_array_equal(prediction.n_closure_only, [2])
    np.testing.assert_array_equal(prediction.n_conditioned, [3])
    np.testing.assert_array_equal(gwas.n_conditioned, [4])
    np.testing.assert_array_equal(prediction.degree_max, [1])
    np.testing.assert_allclose(prediction.est, changed.est, rtol=0, atol=0)
    np.testing.assert_allclose(prediction.var, changed.var, rtol=0, atol=0)
    assert gwas.est[0] > prediction.est[0]
    assert 0.0 <= risk[0] <= 1.0


def test_incident_risk_prior_matches_cip_increment_given_survival():
    risk = EXAMPLE._incident_risk(
        np.array([0.0]), np.array([EXAMPLE.H2]), cip_at_index=0.02,
        cip_at_horizon=0.05, h2=EXAMPLE.H2)

    np.testing.assert_allclose(risk, [(0.05 - 0.02) / (1.0 - 0.02)],
                               rtol=1e-14, atol=0)
