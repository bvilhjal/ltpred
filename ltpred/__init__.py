"""ltpred -- LT-FH++ in Python.

A faithful port of the R package **LTFHPlus**: the liability-threshold model
conditioned on family history and age-, birth-year- and sex-dependent prevalence
(LT-FH++), and the same personalised construction without family history
(ADuLT). Given each individual's case/control
status, age and relatives' statuses, it estimates the **posterior mean genetic
liability** -- a continuous phenotype that can improve power over a plain
case/control label when the liability model and supplied inputs are appropriate.

Typical use (single-trait inference defaults to Pearson-Aitken)::

    from ltpred import simulate_under_LTM_single, estimate_liability
    sim = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=0.5, pop_prev=0.05, n_sim=2000,
        use_age=False, seed=1)
    res = estimate_liability(sim.families, h2=0.5)
    res.est["genetic"]

numpy and scipy are required; an optional Numba JIT (the ``[fast]`` extra)
accelerates the Gibbs sweep. Names are imported lazily (PEP 562) so
``import ltpred`` stays cheap.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.7.1"

if TYPE_CHECKING:
    # Static re-exports. At runtime ``__getattr__`` below imports these lazily,
    # but a module-level ``__getattr__`` is a wildcard to a type checker: every
    # ``from ltpred import X`` would resolve to ``Any``, silently discarding the
    # annotations the ``py.typed`` marker promises. Naming them here restores
    # them without importing anything at run time. Kept in step with ``_EXPORTS``
    # by ``tests/test_public_api.py``.
    from .cip import CipCurve, aalen_johansen_cip, kaplan_meier_cip  # noqa: F401
    from .covariance import (Covmat, construct_covmat_from_kinship,  # noqa: F401
                             construct_covmat_multi, construct_covmat_single,
                             correct_positive_definite, get_relatedness,
                             kinship_from_pedigree)
    from .chunked import (estimate_liability_gibbs_batches,  # noqa: F401
                          estimate_liability_gibbs_chunked,
                          estimate_liability_pa_batches,
                          estimate_liability_pa_chunked)
    from .estimate import (LiabilityResult, batch_means,  # noqa: F401
                           estimate_liability,
                           estimate_liability_from_kinship,
                           estimate_liability_gibbs_arrays,
                           estimate_liability_pa_arrays)
    from .family import Family, Member, families_from_columns  # noqa: F401
    from .fit import (BootstrapResult, FitResult, VarCompResult,  # noqa: F401
                      bootstrap_fit, fit_heritability,
                      fit_variance_components)
    from .gibbs import gibbs_params, rtmvnorm_gibbs  # noqa: F401
    from .liability_scale import (liability_r2_from_z,  # noqa: F401
                                  liability_to_observed_h2,
                                  observed_to_liability_h2,
                                  probit_liability_r2)
    from ._numba import set_num_threads  # noqa: F401
    from .pearson_aitken import pa_algorithm, pa_estimate_batched  # noqa: F401
    from .pedigree import (Pedigree, ParentGraph, build_parent_graph,  # noqa: F401
                           extract_pedigree)
    from .pipeline import PopulationScores, estimate_liabilities  # noqa: F401
    from .quadrature import QuadratureResult, estimate_liability_quadrature_arrays  # noqa: F401
    from .pairwise import PairwiseFitResult, fit_pairwise  # noqa: F401
    from .pairwise_multi import MultiTraitPairwiseResult, fit_pairwise_multi  # noqa: F401
    from .simulate import (Simulation, simulate_under_LTM_single,  # noqa: F401
                           FollowupSimulation, MultiTraitSimulation,
                           RegisterSimulation, pedigree_birth_times,
                           simulate_followup_records, simulate_pedigree,
                           simulate_register_liabilities,
                           simulate_under_LTM_multi)
    from .tetrachoric import (TetrachoricResult, tetrachoric,  # noqa: F401
                              tetrachoric_matrix, tetrachoric_table)
    from .thresholds import (age_thresholds, convert_age_to_cir,  # noqa: F401
                             convert_age_to_thresh, convert_liability_to_aoo,
                             liability_threshold, pa_thresholds,
                             prevalence_thresholds, thresholds_from_cip)

# public name -> submodule it lives in
_EXPORTS = {
    "quadrature": ["QuadratureResult", "estimate_liability_quadrature_arrays"],
    "pairwise": ["PairwiseFitResult", "fit_pairwise"],
    "pairwise_multi": ["MultiTraitPairwiseResult", "fit_pairwise_multi"],
    "covariance": ["get_relatedness", "construct_covmat_single",
                   "construct_covmat_multi",
                   "correct_positive_definite", "Covmat",
                   "kinship_from_pedigree", "construct_covmat_from_kinship"],
    "cip": ["CipCurve", "kaplan_meier_cip", "aalen_johansen_cip"],
    "pedigree": ["ParentGraph", "Pedigree", "build_parent_graph",
                 "extract_pedigree"],
    "pipeline": ["PopulationScores", "estimate_liabilities"],
    "tetrachoric": ["TetrachoricResult", "tetrachoric", "tetrachoric_table",
                    "tetrachoric_matrix"],
    "liability_scale": ["observed_to_liability_h2", "liability_to_observed_h2",
                        "probit_liability_r2", "liability_r2_from_z"],
    "thresholds": ["convert_age_to_cir", "convert_age_to_thresh",
                   "convert_liability_to_aoo", "prevalence_thresholds",
                   "age_thresholds", "liability_threshold", "pa_thresholds",
                   "thresholds_from_cip"],
    "gibbs": ["rtmvnorm_gibbs", "gibbs_params"],
    "pearson_aitken": ["pa_algorithm", "pa_estimate_batched"],
    "family": ["Member", "Family", "families_from_columns"],
    "estimate": ["estimate_liability",
                 "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
                 "estimate_liability_from_kinship", "batch_means",
                 "LiabilityResult"],
    "chunked": ["estimate_liability_pa_chunked",
                "estimate_liability_gibbs_chunked",
                "estimate_liability_pa_batches",
                "estimate_liability_gibbs_batches"],
    "simulate": ["simulate_under_LTM_single", "Simulation",
                 "simulate_pedigree", "pedigree_birth_times",
                 "simulate_register_liabilities", "RegisterSimulation",
                 "simulate_followup_records", "FollowupSimulation",
                 "simulate_under_LTM_multi", "MultiTraitSimulation"],
    "fit": ["fit_heritability", "FitResult", "fit_variance_components",
            "VarCompResult", "bootstrap_fit", "BootstrapResult"],
    "_numba": ["set_num_threads"],
}

_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

# Keep wildcard imports and interactive completion focused on the ordinary
# single-trait workflow. Advanced names remain available as explicit top-level
# imports for compatibility, and from their owning modules documented in api.md.
__all__ = [
    "__version__",
    "Member", "Family", "families_from_columns",
    "prevalence_thresholds", "age_thresholds", "pa_thresholds",
    "thresholds_from_cip",
    "estimate_liability", "LiabilityResult",
    "kinship_from_pedigree", "estimate_liability_from_kinship",
    "simulate_under_LTM_single", "fit_heritability", "set_num_threads",
]


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
