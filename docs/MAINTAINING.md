# Maintaining ltpred

Release procedure, unfinished work, and the review ledger, in one file. The
[changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md) records
completed work; [benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
are the evidence ledger. This file is excluded from the published site.

## 1. Releasing

ltpred publishes to PyPI from a tagged GitHub Release via
[`.github/workflows/publish.yml`](https://github.com/bvilhjal/ltpred/blob/main/.github/workflows/publish.yml),
using PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). No API
token is stored in the repository.

### One-time setup (project owner)

Register the trusted publisher for the `ltpred` project at
<https://pypi.org/manage/account/publishing/> (or, before the first upload, as a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)):

| Field | Value |
|---|---|
| PyPI Project Name | `ltpred` |
| Owner | `bvilhjal` |
| Repository name | `ltpred` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

### Cutting a release

1. Bump `__version__` in `ltpred/__init__.py` (the sole source of truth for the
   distribution; `pyproject.toml` reads it dynamically) and move the
   `## Unreleased` section of `CHANGELOG.md` under the new `## X.Y.Z — <date>`
   heading. Four files carry the version independently of that attribute and
   drift when missed: `CITATION.cff` (`version` and `date-released`) and
   `report/ltpred_methods.tex` (the `v` string and `\date{}`), which
   `scripts/check_evidence.py` checks, and `report/README.md` and
   `docs/algorithm.md` (§ Methods report), which nothing checks.
2. Confirm CI is green on `main`: `test` (fast tier, 3.9–3.13 and 3.14t, plus macOS 3.12), `test-slow` (slow tier, 3.12),
   `research-tests`, `lint` (`ruff`), `test-no-numba`, `oldest-deps` (3.9 at the
   minimum NumPy/SciPy), `examples`, `build` (wheel and sdist) and `docs` (strict
   build plus `scripts/check_evidence.py`). The `docs` job also deploys the site
   on every push to `main`, so the published documentation tracks `main`, not
   the latest release.
3. Rebuild the tracked methods PDF and check its release-defining claims against
   the committed CSVs. If a benchmark artifact must change, first commit its
   source, then regenerate CSV outputs through `benchmarks/run_benchmark.py`
   with each retained output named by `--artifact`; commit the resulting JSONL
   provenance row with the artifact. The standalone
   [time/memory JSON comparison](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/README.md#time-and-memory-between-versions)
   instead retains its aggregate measurements, run log and before/after source
   provenance in a dated capsule. Keep the measured version explicit when a
   later documentation patch incorporates its results. Never backfill a source
   commit after a run.
   ```bash
   cd report && tectonic -X compile ltpred_methods.tex && cd ..
   python -m pip install "pypdf>=4"
   python scripts/check_evidence.py
   ```
4. Locally, sanity-check the package artifacts with modern tooling:
   ```bash
   python -m pip install --upgrade build "twine>=6.1" "packaging>=24.2"
   python -m build
   python -m twine check dist/*      # must PASS for the wheel and the sdist
   ```
   Older `packaging` (< 24.2) reports the PEP 639 `License-Expression` /
   `License-File` fields as malformed even though the metadata is valid; upgrade
   `packaging` rather than changing the license metadata.
5. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Create a GitHub Release for that tag. Publishing the release triggers
   `publish.yml`, which rebuilds, re-runs `twine check`, and uploads to PyPI
   through the trusted publisher.

`workflow_dispatch` runs the build-and-check jobs without publishing, so the
release path can be rehearsed at any time.

## 2. Roadmap

This page lists unfinished work, deliberate boundaries and the paper plan. The
[changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md) records
completed work, [benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
are the evidence ledger, §3 of this file tracks review findings,
and the [`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
inventories checkout-only prototypes. Numbers live in the evidence ledger, not
here.

### 2.1 Current boundary

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

Experimental covariance models, HE genetic-correlation and onset-age-decay
fitting, and factor models remain under `research/`
([research extensions](research.md)).

### 2.2 Priorities

**Table 1. Scientific and engineering priorities.**

| # | Item | What is needed |
|---|---|---|
| 1 | Selected family studies | An ascertainment likelihood for designs with unknown inclusion probabilities or zero-probability family-status strata. IPW is valid only when every relevant family pattern has a known, strictly positive inclusion probability; the marginal case-rate screen can falsify a design but not certify it. |
| 2 | Joint sampling diagnostics | Optional stratum labels or design probabilities; report joint-pattern support, extreme weights and effective sample size, without inferring positivity from marginals. |
| 3 | A single giant pedigree | Sparse relationship operators and solves for pedigrees with thousands of informative relatives; selected-pair recursion does not remove the dense matrix for one large family. |
| 4 | Relatives'-events contrast | RESULTS §§20–21 leave it unresolved at R = 5; rerun at larger R before quoting a payoff. |
| 5 | Real-LD workflow | Extend the opt-in HAPNEST path to a genotype → liability → LMM example with held-out calibration and null-marker diagnostics. |
| 6 | PyPI release | Register the trusted publisher, create a GitHub Release and verify the uploaded wheel and sdist ([§1 of this file](#1-releasing)). |
| 7 | Multi-trait PA | Promote only if a benchmark shows adequate accuracy for rare traits, asymmetric truncation and larger pedigrees; otherwise Gibbs stays the multi-trait engine. |

Open review findings are Table 2 in §3 below.

### 2.3 Deliberate boundaries

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
  plausible optimisations that do not pay, PA bound-row deduplication among
  them; see Table 3 in §3 below.

### 2.4 Graduation rule

A new statistical capability moves into `ltpred/` only after its estimand and
observation model are documented, invalid inputs fail clearly, known-truth
simulation covers calibration as well as ranking, runtime and memory are
measured at a representative scale, and the public path has tests and
committed benchmark evidence. A thin orchestration layer over supported pieces
may land on exact equivalence and invariance tests, but its payoff remains
unvalidated until representative evidence is regenerated. No graduation from
`research/` is scheduled.

### 2.5 Paper plan

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

## 3. Review findings

*Table 2. Open findings from reviews 08–09f, most consequential first. Checked against the tree on 2026-11-02 (line references date from then; symbols are the durable anchor). Finding F8 of 09f (tetrachoric via multivariate_normal.cdf) was fixed after that pass by the Plackett root solve and is dropped here. Resolved findings are in the CHANGELOG; full review texts are in git history (reviews 08–09e at `dc5336c`, 09f in history at the deletion commit).*

| ID | Review | Finding | Sev | Where | Next step |
|---|---|---|---|---|---|
| T1-4 (residual) | 09e | The PA object path still walks every member three times. `83dfd3a` fused only `_stack_object_members`. `_check_unique_roles` and `_group_by_structure` still run as separate passes. | T1 | `ltpred/estimate.py` `_check_unique_roles` (451; calls 635/692/748/796), `_group_by_structure` (558; calls 649/722/758/812), `_stack_object_members` (993; calls 655/727/760) | Fuse the other two walks. Let the mixture path fall through unchanged, and keep the bit-identity test. |
| T2-10 step 4 | 09e | The repair gate still runs for every proband. It could be skipped where `_covariance_reduction_is_safe` proves it a no-op. The bound and its interlacing step for the g-prepended matrix exist only as a docstring sketch (`ltpred/_selected_kinship.py:142-156`), not in the methods report. | T2 | `ltpred/_selected_kinship.py:139`, `ltpred/pipeline.py:486` | Write the proof in the methods report, then land the change with a bit-identity test. |
| T2-3 (residual) | 09e | No driver capsule (`bench_time_memory.py`) spans v0.7.1 → v0.7.4, so the v0.7.2–v0.7.4 performance changes have no end-to-end time/memory measurement. | T2 | `benchmarks/README.md` | Run one `bench_time_memory.py` capsule from v0.7.1 to the current release (the driver writes its own capsule; it does not use `run_benchmark.py`). |
| T2-2 (residual) | 09e | `check_capsule_integrity` checks only capsules that contain a `results.json`. That is 3 of the 6 under `benchmarks/results/`. The 2026-09-24 v0.7.3 capsule and both 2026-09-16 capsules are never checked. A capsule without `thread_variables` passes silently (`if threads:`), and lean-v072 is one such capsule. | T2 | `scripts/check_evidence.py:129-157` (`if threads:` at 148) | Check each capsule's own schema. Fail when thread provenance is missing. |
| T2-4 | 09e | `bootstrap_fit` runs strictly serially, although its replicates are independent. A probe measured 3.9× at 10 threads with max\|diff\| 0, under contention. | T2 | `ltpred/fit.py:946` | Parallelise, after re-establishing bit-identity on a quiet machine. |
| T3-4 | 09e | The v0.7.3 capsule JSONs record `threads: 1` but not the BLAS/OMP/MKL variables or a load average. Its README states them in prose only. | T3 | `benchmarks/results/2026-09-24-review/*.json` | Add machine-readable `thread_variables` and a `load_average` field to future probe capsules. |
| F1 (residual) | 09f | Pid normalization is cached on the member (0.7.5). `_stack_object_members` still rebuilds the row arrays on every call. | T2 | `ltpred/family.py` `_member_pid_key`; `ltpred/estimate.py` `_stack_object_members` | Vectorize the stacking, and keep the bit-identity test. |
| F2 (residual) | 09f | PA groups under 128 families use the serial kernel (0.7.5). Pin conditioning is still recomputed per chunk. | T2 | `ltpred/pearson_aitken.py` `_PARALLEL_MIN_FAMILIES` | Share pin reductions across chunks of one call. |
| F3 (residual) | 09f | The register loop reuses `Pedigree.member_index` (0.7.5) and still scores one proband per call. | T2 | `ltpred/pipeline.py` | Batch probands that share a structure and an observation mask. |
| F4 (residual) | 09f | Gibbs collapse invariants are computed once per group (0.7.5). `_validate_covmat` still factors with `eigvalsh` on every `gibbs_params` call. | T2 | `ltpred/gibbs.py` `_validate_covmat` | A Cholesky PD test, only if the error text stays the same. |
| F5 | 09f | Two `out` targets call `pa_estimate_batched` once each. | T2 | `ltpred/estimate.py` `_pa_from_role_arrays` | One fold that returns both targets, checked against both current orderings. |
| F7 | 09f | `_prepare_pairs` still scans each family's members in Python. | T2 | `ltpred/pairwise.py` `_prepare_pairs` | The same ingestion treatment as F1. |
| F9 | 09f | `_estimate_group` updates its accumulators in Python once per round. | T3 | `ltpred/estimate.py` `_estimate_group` | Leave it until a profile shows it. |
| F10 (residual) | 09f | Kinship subsetting still builds index arrays with `np.ix_`. The mating-loop half is N7, released in 0.7.5. | T3 | `ltpred/covariance.py` kinship extraction, and the other `np.ix_` sites | No change unless a profile shows it. |
| B1 (residual) | 09f | Personalised Gibbs and the censoring mixture are locked in RESULTS §34 (0.7.5, `d0e5102`). ADuLT and the Broad LT-FH scripts are not. | T2 | `benchmarks/RESULTS.md` §34 | Add those two arms. |
| B2 (residual) | 09f | §36 records register and array-PA throughput. It is not the workload matrix, and it does not time LTFHPlus at 350 000. | T2 | `benchmarks/RESULTS.md` §36 | Those cells, on a quiet host. |
| B3 | 09f | No comparison with PCGC, GREML, OpenMx or MCMCglmm. | T2 | `benchmarks/` | An external column on the existing fitter cohorts. |
| B4 | 09f | The suite does not report Z-based effective N, a locus count with a jackknife SE, Lee R² with a jackknife SE, or an S-LDSC attenuation ratio. | T2 | `benchmarks/RESULTS.md` | Add them beside the NCP ratio. The locus count needs B7. |
| B5 (residual) | 09f | Quadrature has RESULTS §35 (0.7.5). Multi-trait scoring still has no accuracy benchmark. | T2 | §2.2 priority 7 | That benchmark, before promoting multi-trait PA. |
| B6 | 09f | No FamEvent triangulation. | T3 | `benchmarks/` | An independent ascertained age-at-onset draw. |
| B7 | 09f | The HAPNEST real-LD arm has no artifact. | T2 | RESULTS limitations | One committed run. |
| B8 | 09f | The missing driver capsule is 09e T2-3. The case list still has no Gibbs, quadrature, fitter, or large-register case. | T2 | `benchmarks/bench_time_memory.py` | See T2-3, then extend the case list. |
| 09d §7 | 09d | **UNVERIFIED.** At `2722a59`, `test_pairwise_multi.py::test_boundary_withholds_uncertainty_and_zero_variance_correlations` failed under CPython 3.10 / NumPy 1.26 / Numba 0.66 ("Emitted warnings: []"). It passes under 3.14/NumPy 2.4 (09e) and under 3.11/NumPy 2.4 (2026-09-27). No CI job runs NumPy 1.26 with Numba. | T2 | `tests/test_pairwise_multi.py` | Re-run on that stack. If it still fails, pin the warning path or add a CI leg. |

*Table 3. Declined, by design, or deliberate boundaries — findings closed without the recommended change. Several are measured non-improvements; do not re-implement without new measurements.*

| ID | Review | Finding | Decision, and where it is recorded |
|---|---|---|---|
| F9 | 08 | Four CSVs had no manifest row, and schema-2 source states were unrecoverable. | A disclosed limit in the `benchmarks/RESULTS.md` preamble and its remaining limitations. The register and pedigree CSVs were rerun under the wrapper (`8219ce7`). `bench_pgs_comparison.csv` and `bench_ascertainment_h2null.csv` still have no manifest row. The h2null CSV was reproduced to 1.5e-14 (RESULTS §29). |
| T3-3 | 09 | Seeding `rtmvnorm_gibbs` reseeds the calling thread's Numba RNG (NumPy's global RNG without Numba). | Documented at `ltpred/gibbs.py:925-937`. The high-level estimators seed per family. |
| T3-4 | 09 | Per-family seed blocks wrap mod 2³², so streams can collide beyond about 43M families per call. | Documented at `ltpred/estimate.py:599-603`. |
| T3-7 | 09 | The collapsed-kernel Gibbs MC SE is about 10× below `batch_means` on raw draws. | By design: the kernel streams E[g\|y], which is Rao–Blackwellised. See methods report step G2 and `ltpred/gibbs.py:527-538`. |
| T3-2 (rest) | 09c | The docs should call quadrature a cross-check, not a cohort workhorse. | The measured part was fixed. The positioning was left to the maintainer. `docs/estimation.md:309` already says "numerical posterior cross-check". |
| T2-4 (rest) | 09d | De-duplicate the 28 "own status" restatements. | Declined by the review's own correction: only 6 are enumerations, and about 22 are distinct warnings. The fix was a separate tutorial page. |
| T3-6 | 09e | Reuse the invariant eigenbasis in `gibbs_params`. | Declined. The stage is 1.17% of an iteration, and the change is not bit-reproducible. |
| T3-7 | 09e | Skip the Hessian that the pairwise criterion discards. | Declined: it is 0.4% of a fit. |
| T3-8 | 09e | Check whether the workqueue launch lock serialises kernels. | Not a defect. It costs 160 ns per launch, and scaling is 3.65× at 8 threads. |
| T3-9 | 09e | Deduplicate PA bound rows, as was done for quadrature. | **Do not implement.** It is 9.9–15.8% slower, because `np.unique(axis=0)` costs more than the kernel it saves. |
| T3-10 | 09e | Vectorise the per-family Gibbs convergence loop. | Declined. The saving is noise at the defaults and pays off only beyond about 50 sweeps. |
| T3-11 | 09e | Allocation pressure. | Not a defect: 277 B/family for PA and 561 B/family for Gibbs. |
| T3-14 | 09e | Construct `_SelectedKinship` lazily. | Declined in `151898b`. The eager topological rank is the driver's whole-graph cycle check (`ltpred/pipeline.py:360-362`). |
| F11 | 09f | The Gibbs sweep and the PA fold are already the right shapes. | Do not micro-optimize them. Keep the pure-Python fallback, per-family reseeding, chunk bit-identity, and the time/memory output gate. |
| T3-15 | 09e | Compact the kinship cache into an array store. | Declined. Real occupancy is 0.7–4.5 MiB, and the default cap is within 1.16× of optimal. |
| — | 09e §6 | Exploit Kronecker structure in the multi-trait covariance. | Unavailable: no Kronecker matrix is ever built at scale. |
| — | 09e §6 | `simulate_register_liabilities(method="dense")` is quadratic-to-cubic in memory. | Covered qualitatively at `docs/estimation.md:564-569`, which recommends `mendelian` for large registers. The review measured 1.5 GB against 15 MB at n≈9,100. |
