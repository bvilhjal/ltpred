"""Liability estimates for LT-FH, LT-FH++, ADuLT and PA-FGRS.

For each proband this conditions a family covariance on every member's liability
interval and estimates the posterior mean of the proband's genetic liability ``g``
(and/or full liability ``o``). Gibbs estimates that target by Monte Carlo;
single-trait inference defaults to the deterministic Pearson-Aitken (PA)
sequential-moment approximation. Gibbs also handles multiple traits. The resulting
genetic-liability estimate is the continuous phenotype fed to a GWAS. Observation
bounds and relative rows distinguish LT-FH, LT-FH++, ADuLT; PA-FGRS additionally
requires its PA-specific ``K_i``/``K_pop`` censoring mixture. In particular, PA is
the single-trait default, but it does not turn classic LT-FH inputs into LT-FH++
or PA-FGRS automatically.

Families are independent, so the estimator groups those that share a family
structure (identical roles -> identical covariance) and samples the whole group
in one compiled, ``prange``-parallel kernel when Numba is available (otherwise a
serial Python fallback), accumulating the mean and the
batch-means Monte-Carlo SE online. The sampler is re-run, accumulating draws,
until every requested estimate's SE drops below ``tol`` (LTFHPlus's convergence
rule). ``_estimate_liability_single`` handles one trait,
``_estimate_liability_multi`` several correlated traits, and
:func:`estimate_liability` dispatches between them.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .covariance import (construct_covmat_single, construct_covmat_multi,
                         construct_covmat_from_kinship, correct_positive_definite)
from .gibbs import (gibbs_params, gibbs_estimate_batched, as_bounds, _MAX_SEED,
                    _validate_seed, _validate_burn_in)
from .pearson_aitken import pa_estimate_batched
from ._validation import (validate_bounds, validate_mixture_inputs,
                          validate_own_status)

__all__ = ["LiabilityResult", "batch_means", "estimate_liability",
           "estimate_liability_pa_arrays", "estimate_liability_gibbs_arrays",
           "estimate_liability_from_kinship"]
