"""ltpred -- LT-FH++ in Python.

A faithful port of the R package **LTFHPlus**: the liability-threshold model
conditioned on family history, age of onset and sex (LT-FH++), and its
family-free age-dependent variant (ADuLT). Given each individual's case/control
status, age and relatives' statuses, it estimates the **posterior mean genetic
liability** -- a continuous phenotype that, used in a linear GWAS, recovers power
lost by a plain case/control label.

Typical use::

    from ltpred import simulate_under_LTM_single, estimate_liability
    sim = simulate_under_LTM_single(h2=0.5, pop_prev=0.05, n_sim=2000, seed=1)
    res = estimate_liability(sim.families, h2=0.5, out=("genetic",))
    res.est["genetic"]        # posterior mean genetic liability per proband

The pieces, if you want them directly:

* :func:`~ltpred.covariance.construct_covmat` -- family covariance from relatedness
* :func:`~ltpred.thresholds.age_thresholds` / ``prevalence_thresholds`` -- status+age -> bounds
* :func:`~ltpred.gibbs.rtmvnorm_gibbs` -- the truncated-MVN Gibbs sampler
* :func:`~ltpred.estimate.estimate_liability` -- the end-to-end estimator

numpy and scipy are required; an optional Numba JIT (the ``[fast]`` extra)
accelerates the Gibbs sweep. Names are imported lazily (PEP 562) so
``import ltpred`` stays cheap.
"""

import importlib

__version__ = "0.1.0.dev0"

# public name -> submodule it lives in
_EXPORTS = {
    "covariance": ["get_relatedness", "construct_covmat", "construct_covmat_single",
                   "construct_covmat_multi", "correct_positive_definite", "Covmat",
                   "kinship_from_pedigree", "construct_covmat_from_kinship"],
    "thresholds": ["convert_age_to_cir", "convert_age_to_thresh",
                   "convert_liability_to_aoo",
                   "convert_observed_to_liability_scale", "prevalence_thresholds",
                   "age_thresholds", "liability_threshold", "pa_thresholds",
                   "thresholds_from_cip"],
    "gibbs": ["rtmvnorm_gibbs", "gibbs_params"],
    "pearson_aitken": ["pa_algorithm", "pa_estimate_batched", "tnorm_moments",
                       "tnorm_mixture_conditional"],
    "family": ["Member", "Family", "families_from_columns"],
    "estimate": ["estimate_liability", "estimate_liability_pa",
                 "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
                 "estimate_liability_from_kinship", "liability_sensitivity",
                 "SensitivityResult", "batch_means", "LiabilityResult"],
    "simulate": ["simulate_under_LTM_single", "Simulation"],
    "fit": ["fit_heritability", "FitResult", "fit_variance_components",
            "VarCompResult", "fit_genetic_correlation", "GenCorrResult",
            "bootstrap_fit", "BootstrapResult", "test_variance_component",
            "test_genetic_correlation", "SignificanceTest"],
    "_numba": ["set_num_threads"],
}

_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = ["__version__", *_NAME_TO_MODULE]


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    mod = _NAME_TO_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = obj          # cache so subsequent access skips __getattr__
    return obj


def __dir__():
    return sorted(__all__)
