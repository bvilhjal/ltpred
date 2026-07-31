"""Calibration of the inference machinery: p-values, intervals, and SEs.

The package ships three inference tools whose *calibration* had no benchmark
(external review, Tier 2): the parametric-bootstrap component test
(`test_variance_component`), the family-bootstrap interval (`bootstrap_fit`),
and the MCEM OPG/BHHH information SE (`fit_variance_components_mcem`, the former
`fit_variance_components(method="mcem")` route, now in `research.advanced_fitting`).
Point-estimate bias is benchmarked elsewhere (bench_fit_heritability,
bench_variance_components); this script asks the inferential question: are the
p-values uniform under the null, do the intervals cover at the nominal rate,
and do the reported SEs match the across-dataset sampling SD?

Design. R = 25 independent datasets (400 families, proband + parents + sib,
classic case/control bounds at prevalence 0.1) generated under an A-only model
(h2 = 0.5, C = 0). Fit settings are reduced (n_iter=600, burn_in=200,
n_boot=50) for runtime; each part notes the consequence.

  Part 1  Type-I error: `test_variance_component(fams, "C")` per dataset; the
          null is true, so p should be ~Uniform(0,1). Report the rejection
          rate at 0.05 with its binomial 95% interval, and the mean/median p.
  Part 2  Interval coverage: `bootstrap_fit` (family bootstrap, n_boot=50,
          95% percentile CI) of the fit_heritability point estimate per
          dataset; report the coverage of the true h2 = 0.5, and the mean
          bootstrap SE vs the across-dataset SD of the point estimates (the
          "bootstrap SE recovers the sampling SD" check; ROADMAP reports the
          internal h2_se understates that SD by >20x).
  Part 3  MCEM SEs: `fit_variance_components_mcem(("A","C"))` per
          dataset; compare the mean reported OPG SE with the across-dataset SD
          of the point estimates (ratio ~1 would mean the information SE is
          calibrated; the docs already call these approximate diagnostics).

Pre-registered reads: Part 1's rejection rate should be consistent with 0.05
(at R=25 the exact binomial interval is very wide, so this catches only gross
anti-conservatism); Part 2's coverage should be near 95% (same coarse
resolution); Part 3 reports the SE/SD ratio as-is (no gate -- the docs do not
claim calibration).

Run:  conda run -n ltpred python benchmarks/bench_inference_calibration.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.fit import bootstrap_fit, fit_heritability  # noqa: E402
from research.advanced_fitting import (  # noqa: E402
    fit_variance_components_mcem, test_variance_component)
from ltpred.simulate import simulate_under_LTM_single  # noqa: E402

SEED = 20260719
R = 25
N_FAM = 400
FAM_VEC = ["m", "f", "s1"]
H2_TRUE = 0.5
PREV = 0.1
N_BOOT = 50
FIT_KW = dict(n_iter=600, burn_in=200)


def make_dataset(rep):
    rng = np.random.default_rng(np.random.PCG64(SEED + 101 * rep))
    return simulate_under_LTM_single(fam_vec=FAM_VEC, h2=H2_TRUE,
                                     pop_prev=PREV, n_sim=N_FAM,
                                     seed=rng).families


def binom_ci(k, n, alpha=0.05):
    """Clopper–Pearson exact interval for a binomial proportion."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    p = k / n
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return p, lo, hi


def main():
    t0 = time.time()
    print("Inference-machinery calibration (Type-I / coverage / SE-vs-SD)")
    print(f"seed={SEED}  R={R}  n_fam={N_FAM}  true A-only model "
          f"(h2={H2_TRUE}, C=0)  n_boot={N_BOOT}  fit={FIT_KW}")

    # ---- Part 1: Type-I of the parametric-bootstrap component test ---------
    print("\nPart 1: test_variance_component('C') under the true null")
    pvals = []
    for rep in range(R):
        fams = make_dataset(rep)
        st = test_variance_component(fams, "C", n_boot=N_BOOT,
                                     seed=1000 + rep,
                                     sampling="population", **FIT_KW)
        pvals.append(st.p_value)
    pvals = np.array(pvals)
    p, lo, hi = binom_ci(int(np.sum(pvals < 0.05)), R)
    print(f"  rejection rate at 0.05: {p:.2f}  (95% binomial CI {lo:.2f}-{hi:.2f})")
    print(f"  p-values: mean {pvals.mean():.3f}  median {np.median(pvals):.3f}  "
          f"min {pvals.min():.3f}  (Uniform: 0.5 / 0.5 / ~1/(R+1))")

    # ---- Part 2: bootstrap_fit coverage + SE vs SD --------------------------
    print("\nPart 2: bootstrap_fit 95% CI coverage for fit_heritability")
    covers, ses, h2s = 0, [], []
    for rep in range(R):
        fams = make_dataset(rep)
        est = lambda f, seed=None: fit_heritability(
            f, seed=seed, sampling="population", **FIT_KW
        ).h2
        b = bootstrap_fit(fams, est, n_boot=N_BOOT, seed=2000 + rep)
        h2s.append(float(b.estimate))
        ses.append(float(b.se))
        covers += bool(b.ci_low <= H2_TRUE <= b.ci_high)
    p, lo, hi = binom_ci(covers, R)
    sd_across = float(np.std(h2s, ddof=1))
    print(f"  coverage of true h2: {covers}/{R} = {p:.2f}  "
          f"(95% binomial CI {lo:.2f}-{hi:.2f})")
    print(f"  mean bootstrap SE {np.mean(ses):.3f} vs across-dataset SD "
          f"{sd_across:.3f}  (ratio {np.mean(ses) / sd_across:.2f})")

    # ---- Part 3: MCEM OPG SE vs across-dataset SD ---------------------------
    print("\nPart 3: MCEM OPG SE vs across-dataset SD (A component)")
    se_mcem, a_hats = [], []
    for rep in range(R):
        fams = make_dataset(rep)
        r = fit_variance_components_mcem(fams, ("A", "C"),
                                         seed=3000 + rep, **FIT_KW)
        a_hats.append(float(r.components["A"]))
        se_mcem.append(float(r.se["A"]))
    sd_across = float(np.std(a_hats, ddof=1))
    print(f"  mean reported SE {np.mean(se_mcem):.3f} vs across-dataset SD "
          f"{sd_across:.3f}  (ratio {np.mean(se_mcem) / sd_across:.2f})")
    print(f"  point-estimate mean A {np.mean(a_hats):.3f} (truth {H2_TRUE})")

    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
