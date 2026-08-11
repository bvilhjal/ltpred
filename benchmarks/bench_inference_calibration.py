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
n_boot=50) for runtime; each part notes the consequence. Part 4 uses its own
two-trait null (below) at the same R, n_fam, and fit settings.

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
  Part 4  Type-I error: `test_genetic_correlation(fams)` per dataset under a
          true r_g = 0 generative model with a NONZERO phenotypic correlation
          (the null cell of bench_genetic_correlation: two traits,
          h2 = (0.5, 0.4), r_p = 0.2, prevalence 0.1 each, parents + 2 sibs;
          same R, n_fam, n_boot and fit settings as Part 1). The null is true,
          so p should be ~Uniform(0,1) and the test must not manufacture a
          genetic correlation from the phenotypic one.

Pre-registered reads: Part 1's and Part 4's rejection rates should be
consistent with 0.05 (at R=25 the exact binomial interval is very wide, so
this catches only gross anti-conservatism); Part 2's coverage should be near
95% (same coarse resolution); Part 3 reports the SE/SD ratio as-is (no gate --
the docs do not claim calibration).

Run:  conda run -n ltpred python benchmarks/bench_inference_calibration.py
      python benchmarks/bench_inference_calibration.py --parts 4   (one part)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _common import simulate_families_multi  # noqa: E402
from ltpred.fit import bootstrap_fit, fit_heritability  # noqa: E402
from research.advanced_fitting import (  # noqa: E402
    fit_variance_components_mcem, test_genetic_correlation,
    test_variance_component)
from ltpred.simulate import simulate_under_LTM_single  # noqa: E402

SEED = 20260719
R = 25
N_FAM = 400
FAM_VEC = ["m", "f", "s1"]
H2_TRUE = 0.5
PREV = 0.1
N_BOOT = 50
FIT_KW = dict(n_iter=600, burn_in=200)

# Part 4 null cell mirrors bench_genetic_correlation.py's r_g = 0 arm.
FAM_MULTI = ["m", "f", "s1", "s2"]
H2_MULTI = [0.5, 0.4]
RP_OFFDIAG = 0.2                     # phenotypic (full-liability) correlation


def make_dataset(rep, seed=SEED, n_fam=N_FAM):
    rng = np.random.default_rng(np.random.PCG64(seed + 101 * rep))
    return simulate_under_LTM_single(fam_vec=FAM_VEC, h2=H2_TRUE,
                                     pop_prev=PREV, n_sim=n_fam,
                                     seed=rng).families


def make_dataset_rg(rep, seed=SEED, n_fam=N_FAM):
    """Two-trait r_g = 0 null with nonzero phenotypic correlation."""
    return simulate_families_multi(FAM_MULTI, H2_MULTI, 0.0, RP_OFFDIAG,
                                   n_fam, (PREV, PREV), seed + 101 * rep)


def binom_ci(k, n, alpha=0.05):
    """Clopper–Pearson exact interval for a binomial proportion."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError("require 0 <= k <= n and n > 0")
    p = k / n
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return p, lo, hi


def part1(args):
    print("\nPart 1: test_variance_component('C') under the true null")
    pvals = []
    for rep in range(args.reps):
        fams = make_dataset(rep, args.seed, args.n_fam)
        st = test_variance_component(fams, "C", n_boot=args.n_boot,
                                     seed=1000 + rep,
                                     sampling="population", **args.fit_kw)
        pvals.append(st.p_value)
    report_pvals(pvals, args.reps)


def part2(args):
    print("\nPart 2: bootstrap_fit 95% CI coverage for fit_heritability")
    covers, ses, h2s = 0, [], []
    for rep in range(args.reps):
        fams = make_dataset(rep, args.seed, args.n_fam)
        est = lambda f, seed=None: fit_heritability(
            f, seed=seed, sampling="population", **args.fit_kw
        ).h2
        b = bootstrap_fit(fams, est, n_boot=args.n_boot, seed=2000 + rep)
        h2s.append(float(b.estimate))
        ses.append(float(b.se))
        covers += bool(b.ci_low <= H2_TRUE <= b.ci_high)
    p, lo, hi = binom_ci(covers, args.reps)
    sd_across = float(np.std(h2s, ddof=1))
    print(f"  coverage of true h2: {covers}/{args.reps} = {p:.2f}  "
          f"(95% binomial CI {lo:.2f}-{hi:.2f})")
    print(f"  mean bootstrap SE {np.mean(ses):.3f} vs across-dataset SD "
          f"{sd_across:.3f}  (ratio {np.mean(ses) / sd_across:.2f})")


def part3(args):
    print("\nPart 3: MCEM OPG SE vs across-dataset SD (A component)")
    se_mcem, a_hats = [], []
    for rep in range(args.reps):
        fams = make_dataset(rep, args.seed, args.n_fam)
        r = fit_variance_components_mcem(fams, ("A", "C"),
                                         seed=3000 + rep,
                                         sampling="population", **args.fit_kw)
        a_hats.append(float(r.components["A"]))
        se_mcem.append(float(r.se["A"]))
    sd_across = float(np.std(a_hats, ddof=1))
    print(f"  mean reported SE {np.mean(se_mcem):.3f} vs across-dataset SD "
          f"{sd_across:.3f}  (ratio {np.mean(se_mcem) / sd_across:.2f})")
    print(f"  point-estimate mean A {np.mean(a_hats):.3f} (truth {H2_TRUE})")


def part4(args):
    print("\nPart 4: test_genetic_correlation under the r_g=0 null "
          f"(r_p={RP_OFFDIAG}, h2={H2_MULTI})")
    pvals = []
    for rep in range(args.reps):
        fams = make_dataset_rg(rep, args.seed, args.n_fam)
        st = test_genetic_correlation(fams, n_boot=args.n_boot,
                                      seed=4000 + rep,
                                      sampling="population", **args.fit_kw)
        pvals.append(st.p_value)
    report_pvals(pvals, args.reps)


def report_pvals(pvals, reps):
    pvals = np.array(pvals)
    p, lo, hi = binom_ci(int(np.sum(pvals < 0.05)), reps)
    print(f"  rejection rate at 0.05: {p:.2f}  (95% binomial CI {lo:.2f}-{hi:.2f})")
    print(f"  p-values: mean {pvals.mean():.3f}  median {np.median(pvals):.3f}  "
          f"min {pvals.min():.3f}  (Uniform: 0.5 / 0.5 / ~1/(R+1))")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", default="1,2,3,4",
                    help="comma-separated parts to run (default: all)")
    ap.add_argument("--reps", type=int, default=R,
                    help="independent datasets per part")
    ap.add_argument("--n-fam", type=int, default=N_FAM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-iter", type=int, default=FIT_KW["n_iter"])
    ap.add_argument("--burn-in", type=int, default=FIT_KW["burn_in"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    args.parts = {int(p) for p in args.parts.split(",")}
    unknown = args.parts - {1, 2, 3, 4}
    if unknown:
        ap.error(f"unknown parts: {sorted(unknown)} (choose from 1,2,3,4)")
    args.fit_kw = dict(n_iter=args.n_iter, burn_in=args.burn_in)
    return args


def main():
    args = parse_args()
    t0 = time.time()
    print("Inference-machinery calibration (Type-I / coverage / SE-vs-SD)")
    print(f"seed={args.seed}  R={args.reps}  n_fam={args.n_fam}  "
          f"n_boot={args.n_boot}  fit={args.fit_kw}  "
          f"parts={sorted(args.parts)}")
    print(f"Part 1-3 model: A-only (h2={H2_TRUE}, C=0); "
          f"Part 4 model: two-trait r_g=0 null (h2={H2_MULTI}, "
          f"r_p={RP_OFFDIAG}, fam={FAM_MULTI})")

    for n, fn in ((1, part1), (2, part2), (3, part3), (4, part4)):
        if n in args.parts:
            fn(args)

    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
