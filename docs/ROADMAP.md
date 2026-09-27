# Roadmap

This page lists unfinished work, deliberate boundaries and the paper plan. The
[changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md) records
completed work, [benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
are the evidence ledger, the [review ledger](REVIEWS.md) tracks review findings,
and the [`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
inventories checkout-only prototypes. Numbers live in the evidence ledger, not
here.

## 1. Current boundary

The supported package provides LT-FH, LT-FH++, ADuLT and the single-trait
Pearson–Aitken PA-FGRS mixture; role-based and arbitrary-kinship inputs; a
population-register driver with explicit GWAS/prediction observation sets;
Gibbs, single-trait PA and additive nuclear-family quadrature; common-threshold
moment fitting; and opt-in pairwise likelihood fitting of single-trait A/C/M and
joint multi-trait heritabilities, correlations and shared-environment
covariances. Broad calibration across rare traits, family structures and
sampling designs, boundary inference and statistical-efficiency comparisons
remain unfinished. The [analysis choices](quickstart.md#choose-the-analysis),
[algorithm](algorithm.md) and [assumptions](assumptions.md) state the exact
contracts.

Experimental covariance models, selection-aware fitting, HE genetic-correlation
and onset-age-decay fitting, and factor models remain under `research/`
([research extensions](research.md)).

## 2. Priorities

**Table 1. Scientific and engineering priorities.**

| # | Item | What is needed |
|---|---|---|
| 1 | Selected family studies | An ascertainment likelihood for designs with unknown inclusion probabilities or zero-probability family-status strata. IPW is valid only when every relevant family pattern has a known, strictly positive inclusion probability; the marginal case-rate screen can falsify a design but not certify it. |
| 2 | Joint sampling diagnostics | Optional stratum labels or design probabilities; report joint-pattern support, extreme weights and effective sample size, without inferring positivity from marginals. |
| 3 | A single giant pedigree | Sparse relationship operators and solves for pedigrees with thousands of informative relatives; selected-pair recursion does not remove the dense matrix for one large family. |
| 4 | Relatives'-events contrast | RESULTS §§20–21 leave it unresolved at R = 5; rerun at larger R before quoting a payoff. |
| 5 | Real-LD workflow | Extend the opt-in HAPNEST path to a genotype → liability → LMM example with held-out calibration and null-marker diagnostics. |
| 6 | PyPI release | Register the trusted publisher, create a GitHub Release and verify the uploaded wheel and sdist ([RELEASING.md](RELEASING.md)). Tag `v0.7.4` first (review N3). |
| 7 | Multi-trait PA | Promote only if a benchmark shows adequate accuracy for rare traits, asymmetric truncation and larger pedigrees; otherwise Gibbs stays the multi-trait engine. |

**Table 2. Open review findings** (details and locations in the
[review ledger](REVIEWS.md#1-open-findings)).

| ID | Item | Next step |
|---|---|---|
| 09e T1-4 | PA object path still walks members three times | Fuse `_check_unique_roles` and `_group_by_structure` into the stacking pass; keep the bit-identity test. |
| 09e T2-10 | Covariance repair gate runs for every register proband | Prove the `_covariance_reduction_is_safe` bound for the g-prepended matrix, then skip the no-op repair. |
| 09e T2-3 | No time/memory capsule spans v0.7.1 → v0.7.4 | Run `bench_time_memory.py` through `run_benchmark.py` before the next release. |
| 09e T2-2 | `check_capsule_integrity` checks 3 of 6 capsules and passes a capsule with no thread record | Check each capsule schema; fail when thread provenance is missing. |
| 09e T2-4 | `bootstrap_fit` is serial | Parallelise replicates after confirming bit-identity on a quiet machine. |
| 09e T3-4 | Probe capsules lack machine-readable thread variables and load | Record `thread_variables` and `load_average` in future capsules. |
| N4 | Chunked Gibbs differs from the array API at Monte Carlo level for the same seed | Collapse per structure group, or keep the documented weaker contract. |
| N6 | Methods report imprecise on P1, G1, G4, G6 and quadrature details | Correct at the next PDF rebuild. |
| 09d §7 | Unverified boundary-test failure on NumPy 1.26 + Numba | Re-run on that stack; pin the warning path or add a CI leg. |

## 3. Deliberate boundaries

- **Validate new methods before changing defaults.** Broaden quadrature stress
  tests across rare and discordant large sibships, and assess pairwise interval
  coverage, boundary behaviour and efficiency against the moment fitter on
  equal cohorts. Direct incident-risk models and sparse pedigree message
  passing remain comparison studies.
- **Keep the PA-FGRS censoring mixture PA-only.** A Gibbs comparator needs
  mixture-aware full conditionals (or a latent lifetime-case indicator) plus
  exact checks on small pedigrees. Until then every PA–Gibbs agreement claim is
  restricted to no-mixture inputs.
- **Do not downgrade multi-trait shared-environment models.** `c2`/`m2`
  proportions do not define cross-trait `C`/`M` covariance, so the dispatcher
  raises; a supported extension must accept and validate those inputs.
- **Measured non-improvements stay out.** Review 09e measured several
  plausible optimisations that do not pay (PA bound-row deduplication is
  9.9–15.8% slower); see the [review ledger](REVIEWS.md#2-declined-by-design-or-deliberate-boundaries).

## 4. Graduation rule

A new statistical capability moves into `ltpred/` only after its estimand and
observation model are documented, invalid inputs fail clearly, known-truth
simulation covers calibration as well as ranking, runtime and memory are
measured at a representative scale, and the public path has tests and
committed benchmark evidence. A thin orchestration layer over supported pieces
may land on exact equivalence and invariance tests, but its payoff remains
unvalidated until representative evidence is regenerated. No graduation from
`research/` is scheduled.

## 5. Paper plan

The methods paper joins three uses of one latent-liability core: **I.** risk
prediction from family history (own status out; optional downstream PGS);
**II.** quantitative GWAS phenotypes (own status in); **III.** architecture,
relationships and aetiology from liability-scale h²/r_g and the CIP. PA and
Gibbs are inference engines, not disease models; LT-FH++, ADuLT and PA-FGRS
differ by their observation data. Onset-age decay, genetic nurture, sex
limitation and the factor model are follow-up work in `research/`.

**Table 3. Each paper claim has one canonical benchmark source.**

| Paper claim | Benchmark source | Remaining work |
|---|---|---|
| Engine agreement, speed, fold order | scaling, accuracy, PA robustness | Explain the tested exactness boundary |
| Independent-SNP NCP and calibration | GWAS power, LT-FH++ personalisation, confounding | Real-LD, relatedness-aware mixed-model confirmation (priority 5) |
| PGS plus family history | PGS comparison | Keep training/test separation explicit |
| Censoring and onset assumptions | PA-FGRS mixture, age of onset | Re-run after observation-model changes |
| Prospective register prediction | register pipeline, pedigree inference | Retain the familywise censoring contrast |
| Heritability and A/C/M | heritability, variance components, ascertainment | Separate population, IPW and unsupported designs |
| Inference calibration | inference calibration | State finite-replicate uncertainty |
| Misspecification | misspecification, CIP estimation | Keep assumptions next to conclusions |

Rules: never transcribe numbers by hand — `paper/tables/` are static snapshots
of committed CSVs that `scripts/check_evidence.py` recomputes cell by cell;
record commit, environment, seeds and command for every run; describe
`research/` results as exploratory. Writing order: freeze notation; draft
methods from the tested contracts; insert the tables; write results from the
ledger; add limitations (ascertainment, incidence misspecification, PA
approximation, pedigree overlap, transportability); rebuild the PDF and run
the checks. External dependencies: a HAPNEST/Linux environment for real LD;
R LTFHPlus 2.2.0 and LTFGRS 1.0.1 for `bench_ltfhplus_compare.py`; the owner's
PyPI setup; a Zenodo deposit after the final evidence freeze.
