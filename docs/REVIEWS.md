# Review ledger

Six independent, read-only reviews audited ltpred from v0.3.4 to v0.7.3
(Table 1). This ledger lists each numbered finding and what became of it. It
drops the reviews' narrative. The full texts are in git history:
`git show dc5336c:docs/reviews/REVIEW_2026-09e.md` (likewise `-08`, `-09`,
`-09b`, `-09c`, `-09d`). Statuses were checked on 2026-09-27 against the code,
docs, tests and `git show dc5336c:CHANGELOG.md`; "Unreleased" marks items
closed by the cleanup that created this ledger.

*Table 1. The six reviews.*

| Review | Date | Version reviewed (commit) | Focus | Findings | Added in |
|---|---|---|---|---|---|
| 08 | 2026-08-16 | v0.3.4 (`cf8cc99`) | Full audit: core, fitting, numerics, CI, evidence, docs, literature | 37 (F1–F37) | `6b9efda` |
| 09 | 2026-08-28 | v0.4.2 (`7806e55`) | Full audit against closed forms; second pass on its own fix branch | 7 (T3-1–T3-7) | `a6b18be` |
| 09b | 2026-09-02 | v0.5.1 (`a6b18be`) | 0.5.x delta: register driver `pipeline.py`, `pedigree.py` | 4 | `2006aef` |
| 09c | 2026-09-10 | v0.6.2 (`e9bfa66`) | 0.6.0 inference delta: quadrature, pairwise, selected kinship, PA pins | 4 | `64ed4c4` |
| 09d | 2026-09-21 | v0.7.0 (`2722a59`) | Documentation, onboarding, simulated data | 10 | `47cf4de` |
| 09e | 2026-09-25 | v0.7.3 (`6686541`) | Computational efficiency, benchmark provenance | 32 | `21b2361`–`0db95ea` |

Severity is each review's own tier. **T1** means correctness, or a false
published statement. **T2** means robustness, provenance or drift. **T3** means
polish. Review 08 numbers its findings F1–F37 on the same tiers: F1–F2 are T1,
F3–F16 are T2 and F17–F37 are T3.

*Table 2. Status counts (94 numbered findings).*

| Status | 08 | 09 | 09b | 09c | 09d | 09e | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIXED | 35 | 4 | 4 | 3 | 9 | 18 | 73 |
| PARTLY FIXED (open residual in §1) | — | — | — | — | — | 4 | 4 |
| OPEN | — | — | — | — | — | 2 | 2 |
| DECLINED / by design (§2) | 1 | 3 | — | — | — | 8 | 12 |
| SUPERSEDED / WITHDRAWN | 1 | — | — | 1 | 1 | — | 3 |

Checking the statuses found four further gaps (N1–N3, F2′); N1, N2 and F2′
were closed at once, and N3 and one UNVERIFIED item remain in §1. The
algorithm-documentation pass that followed (docstrings verified against the
code) added N4–N6.

## 1. Open findings

These were checked against the current tree and are still true. This section
feeds `docs/ROADMAP.md`.

*Table 3. Open findings, most consequential first.*

| ID | Review | Finding | Sev | Where | Next step |
|---|---|---|---|---|---|
| T1-4 (residual) | 09e | The PA object path still walks every member three times. `83dfd3a` fused only `_stack_object_members`. `_check_unique_roles` and `_group_by_structure` still run as separate passes. | T1 | `ltpred/estimate.py:561/575/581`, `618/648/653`, `674/684/686`, `722/738` | Fuse the other two walks. Let the mixture path fall through unchanged, and keep the bit-identity test. |
| T2-10 step 4 | 09e | The repair gate still runs for every proband. It could be skipped where `_covariance_reduction_is_safe` proves it a no-op. That bound is proved only for the full pedigree covariance, not for the g-prepended (n+1)×(n+1) matrix. The interlacing argument is not written down. | T2 | `ltpred/_selected_kinship.py:106`, `ltpred/pipeline.py:457` | Write the proof in the methods report, then land the change with a bit-identity test. |
| T2-3 (residual) | 09e | No driver capsule (`bench_time_memory.py`) spans v0.7.1 → v0.7.4, so the v0.7.2–v0.7.4 performance changes have no end-to-end time/memory measurement. | T2 | `benchmarks/README.md` | Run one capsule from v0.7.2 to v0.7.4 through `run_benchmark.py`. |
| T2-2 (residual) | 09e | `check_capsule_integrity` checks only capsules that contain a `results.json`. That is 3 of the 6 under `benchmarks/results/`. The 2026-09-24 v0.7.3 capsule and both 2026-09-16 capsules are never checked. A capsule without `thread_variables` passes silently (`if threads:`), and lean-v072 is one such capsule. | T2 | `scripts/check_evidence.py:131-156` (`:149`) | Check each capsule's own schema. Fail when thread provenance is missing. |
| T2-4 | 09e | `bootstrap_fit` runs strictly serially, although its replicates are independent. A probe measured 3.9× at 10 threads with max\|diff\| 0, under contention. | T2 | `ltpred/fit.py:993` | Parallelise, after re-establishing bit-identity on a quiet machine. |
| T3-4 | 09e | The v0.7.3 capsule JSONs record `threads: 1` but not the BLAS/OMP/MKL variables or a load average. Its README states them in prose only. | T3 | `benchmarks/results/2026-09-24-review/*.json` | Add machine-readable `thread_variables` and a `load_average` field to future probe capsules. |
| N3 | new | v0.7.4 is released (`__version__`, CHANGELOG 2026-09-25) but untagged: 09e T3-5 again. | T3 | git tags; `docs/RELEASING.md:64` | Tag `v0.7.4` at the release commit and push the tag. |
| N6 | new | The methods report is imprecise in places the docstrings now state exactly: P1 (only the object API sorts roles; array/kinship callers set fold order), G1 (P and sd are recomputed for the kept block each round), G4 (the far-tail draw is a leading-order approximation), G6 (fresh chains each round; stop at `se ≤ tol`), and Algorithm Q details (16-node start, last-two-changes acceptance, mode-search stop rules). | T3 | `report/ltpred_methods.tex`, `report/efficient_inference.tex` | Correct at the next PDF rebuild. |
| 09d §7 | 09d | **UNVERIFIED.** At `2722a59`, `test_pairwise_multi.py::test_boundary_withholds_uncertainty_and_zero_variance_correlations` failed under CPython 3.10 / NumPy 1.26 / Numba 0.66 ("Emitted warnings: []"). It passes under 3.14/NumPy 2.4 (09e) and under 3.11/NumPy 2.4 (2026-09-27). No CI job runs NumPy 1.26 with Numba. | T2 | `tests/test_pairwise_multi.py` | Re-run on that stack. If it still fails, pin the warning path or add a CI leg. |

## 2. Declined, by design, or deliberate boundaries

*Table 4. Findings that were closed without the recommended change.*

| ID | Review | Finding | Decision, and where it is recorded |
|---|---|---|---|
| F9 | 08 | Four CSVs had no manifest row, and schema-2 source states were unrecoverable. | A disclosed limit in the `benchmarks/RESULTS.md` preamble and its remaining limitations. The register and pedigree CSVs were rerun under the wrapper (`8219ce7`). `bench_pgs_comparison.csv` and `bench_ascertainment_h2null.csv` still have no manifest row. The h2null CSV was reproduced to 1.5e-14 (RESULTS §29). |
| T3-3 | 09 | Seeding `rtmvnorm_gibbs` mutates the global NumPy RNG. | Documented at `ltpred/gibbs.py:721`. The high-level estimators seed per family. |
| T3-4 | 09 | Per-family seed blocks wrap mod 2³², so streams can collide beyond about 43M families per call. | Documented at `ltpred/estimate.py:538-541`. |
| T3-7 | 09 | The collapsed-kernel Gibbs MC SE is about 10× below `batch_means` on raw draws. | By design: the kernel streams E[g\|y], which is Rao–Blackwellised. See methods report step G2 and `ltpred/gibbs.py:452-459`. |
| T3-2 (rest) | 09c | The docs should call quadrature a cross-check, not a cohort workhorse. | The measured part was fixed. The positioning was left to the maintainer. `docs/estimation.md:321` already says "numerical posterior cross-check". |
| T2-4 (rest) | 09d | De-duplicate the 28 "own status" restatements. | Declined by the review's own correction: only 6 are enumerations, and about 22 are distinct warnings. The fix was a separate tutorial page. |
| T3-6 | 09e | Reuse the invariant eigenbasis in `gibbs_params`. | Declined. The stage is 1.17% of an iteration, and the change is not bit-reproducible. |
| T3-7 | 09e | Skip the Hessian that the pairwise criterion discards. | Declined: it is 0.4% of a fit. |
| T3-8 | 09e | Check whether the workqueue launch lock serialises kernels. | Not a defect. It costs 160 ns per launch, and scaling is 3.65× at 8 threads. |
| T3-9 | 09e | Deduplicate PA bound rows, as was done for quadrature. | **Do not implement.** It is 9.9–15.8% slower, because `np.unique(axis=0)` costs more than the kernel it saves. |
| T3-10 | 09e | Vectorise the per-family Gibbs convergence loop. | Declined. The saving is noise at the defaults and pays off only beyond about 50 sweeps. |
| T3-11 | 09e | Allocation pressure. | Not a defect: 277 B/family for PA and 561 B/family for Gibbs. |
| T3-14 | 09e | Construct `_SelectedKinship` lazily. | Declined in `151898b`. The eager topological rank is the driver's whole-graph cycle check (`ltpred/pipeline.py:333-335`). |
| T3-15 | 09e | Compact the kinship cache into an array store. | Declined. Real occupancy is 0.7–4.5 MiB, and the default cap is within 1.16× of optimal. |
| — | 09e §6 | Exploit Kronecker structure in the multi-trait covariance. | Unavailable: no Kronecker matrix is ever built at scale. |
| — | 09e §6 | `simulate_register_liabilities(method="dense")` is quadratic-to-cubic in memory. | Covered qualitatively at `docs/estimation.md:577-582`, which recommends `mendelian` for large registers. The review measured 1.5 GB against 15 MB at n≈9,100. |

## 3. Resolved findings

*Table 5. Resolved findings by review. "Version" is the CHANGELOG release. Commits are given where the condensed CHANGELOG no longer names the finding.*

**08 (v0.3.4).** Nearly all were fixed in 0.4.0 by `6b9efda` and `ab690c9`.

| ID | Finding | Version |
|---|---|---|
| F1 | Stress-pedigree headline 0.9984 contradicted by its CSV (worst seed 0.99813) | 0.4.0 |
| F2 | Stratified-CIP research test could not fail | 0.4.0; the test later went with `research/pipeline.py` (carry-over is F2′) |
| F3 | NaN `fam_id` fragmented records into singleton families | 0.4.0 |
| F4 | Array APIs did not check column count against `roles` | 0.4.0 |
| F5 | Coincident infinite bounds gave NaN Gibbs draws | 0.4.0 |
| F6 | `estimate_liability_from_kinship` second element changed meaning with `method` | 0.4.0 (breaking change to `(est, se, var)`) |
| F7 | Sex-CIP gap 0.05091 / −0.03689, last digit wrong | 0.4.0 |
| F8 | Peak-RSS "respectively" clause misattributed | 0.4.0 |
| F10 | §2 load-average sentence described the superseded run | 0.4.0 |
| F11 | "≈0.97 cross-setting correlation" had no artifact | 0.4.0 (reworded to §12) |
| F12 | PGS×FH identity "four-setting verification" did not exist | 0.4.0 (cites §28 at its precision) |
| F13 | oldest-deps CI ran the full suite without Numba | 0.4.0 |
| F14 | Declared Python/OS support exceeded CI coverage | 0.4.0 (3.10, 3.11, 3.14t; Linux/macOS classifiers) |
| F15 | No sdist/wheel build verification | 0.4.0 (`build` job) |
| F16 | R lock existed only as an artifact, not a test | 0.4.0 (`tests/test_r_lock.py`) |
| F17 | README RMSE 0.0041 is Gibbs; PA is 0.0046 | 0.4.0; checked in 0.4.2 |
| F18 | Batch-means SE divided by all draws, not those batched | 0.4.0 |
| F19 | Gibbs-only controls silently ignored on the PA path | 0.7.3 (warns) |
| F20 | Length-1 `h2` routed to the multi-trait error | 0.4.0 |
| F21 | `case_encoding="pin"` accepted with non-threshold onset models | 0.4.0 (warns) |
| F22 | Tetrachoric SE is conditional/Wald, optimistic near \|ρ\|→1 | 0.4.0 (docstring) |
| F23 | `aalen_johansen_cip` rejected integral float `event_type` | 0.4.0 |
| F24 | `max_degree=2.9` truncated silently | 0.4.0 |
| F25 | `lower`/`upper` shape mismatch not checked | 0.4.0 |
| F26 | RELEASING listed stale CI Python versions | 0.4.0 |
| F27 | "recall g0 = w'μ" referred to nothing | 0.4.0 |
| F28 | "additive sharing only" true only by default | 0.4.0 |
| F29 | Detectable-enrichment formula omitted the 1.15× floor | 0.4.0 |
| F30 | §24 "M 0.17" should be 0.16 | 0.4.0 |
| F31 | §26/§27 normal CIs over 3–5 replicates | 0.4.0 (rerun with t half-widths) |
| F32 | CI comment "~285 tests" was stale | 0.4.0 |
| F33 | Quotation marks around a paraphrase of Dybdahl Krebs 2026 | 0.4.0 |
| F34 | "Public PA is LTFGRS" omitted `PAFGRS` | 0.4.0 (sentence since removed) |
| F35 | Unverifiable Hujoel 2022 "logistic combination" clause | 0.4.0 (dropped) |
| F36 | Four small untested branches | 0.4.0 |
| F37 | `check_results.py` guard-coverage gaps | SUPERSEDED: checker removed in 0.4.0 (`754e55f`), replaced by `check_evidence.py` |

**09 (v0.4.2).** Fixed in 0.5.0 (`1e36c5c`, `f69568f`).

| ID | Finding | Version |
|---|---|---|
| T3-1 | `tetrachoric_table` accepted fractional counts | 0.5.0 |
| T3-2 | `correct_positive_definite` did limit+1 corrections | 0.5.0 |
| T3-5 | Dead `_AGE_RANGES`/`_age_range` pinned by a test | 0.5.0 |
| T3-6 | Infinite count raised `OverflowError`, not `ValueError` | 0.5.0 |

**09b (v0.5.1).** Fixed in 0.5.2 (`b2bb8e3`, `bad41ad`, `8219ce7`).

| ID | Finding | Version |
|---|---|---|
| T2-1 | Unresolved parent references became founders silently | 0.5.2 (`frac_records_with_unresolved_parents`; raises at 0% resolved) |
| T2-2 | Prediction mode scored prevalent or exited probands silently | 0.5.2 (`proband_state`; warns) |
| T2-3 | Supported driver's payoff evidence labelled stale | 0.5.2 (RESULTS §§20–21 rerun under wrapper) |
| T3-1 | Prediction estimand not named | 0.5.2 (`docs/api.md:137`) |

**09c (v0.6.2).** Fixed in 0.7.0 (`64ed4c4`).

| ID | Finding | Version |
|---|---|---|
| T1-1 | Quadrature mode search stalled below float noise; 2–10% of inputs refused | 0.7.0 (objective-change stop rule) |
| T2-1 | Refinement refusal printed the passing change, not the failing one | 0.7.0 |
| T3-1 | `fit_pairwise` A bias −0.016 | WITHDRAWN: a 40-replicate artifact; 11,364 replicates give bias ≈ 0 (RESULTS §32) |
| T3-2 | Document the quadrature failure surface | 0.7.0 (measured part; rest in §2) |

**09d (v0.7.0).** Fixed in 0.7.1 (`47cf4de`, `31eb75d`), except T3-4.

| ID | Finding | Version |
|---|---|---|
| T1-1 | One exported simulator; register route could not be taught | 0.7.1 (`simulate_pedigree`, `simulate_register_liabilities`, …) |
| T1-2 | Vignette "run-book" unexecuted; register names unbound | 0.7.1 (`docs/tutorial.md` + `tests/test_tutorial.py`) |
| T2-1 | README quickstart used the weak `use_age=True` cohort | 0.7.1 |
| T2-2 | Multi-trait example imported from unshipped `examples/` | 0.7.1 (`simulate_under_LTM_multi`) |
| T2-3 | `registry_pipeline.py` never called the register driver | SUPERSEDED: renamed, then removed in 0.7.1 |
| T2-4 | Docs written as contract; tutorial too small | 0.7.1 (tutorial page; de-duplication declined, §2) |
| T3-1 | Examples read case status off bound finiteness | 0.7.1 |
| T3-2 | OMP deprecation notice on first run | 0.7.1 (`docs/quickstart.md:40`) |
| T3-3 | `age_thresholds` downgraded only on later pages | 0.7.1 (admonition, `quickstart.md:76`) |
| T3-4 | Nine exported names absent from docs | 0.7.3 (`docs/api.md`) |

**09e (v0.7.3).** Fixed in 0.7.4 (`151898b`, `83dfd3a`, `0050113`, `573d0cf`). All
outputs are bit-identical. Partial fixes have their residuals in §1.

| ID | Finding | Version |
|---|---|---|
| T1-1 | v0.4.0 scaling grid called "current"; four threads unstated | 0.7.4 (pinned by SHA-256, `48ed3ec`) |
| T1-2 | v0.7.2 wall-clock figures had no artifact | 0.7.4 (restated qualitatively) |
| T1-3 | Family-free register proband paid for matrix, eigen and PA | 0.7.4 (scalar ADuLT branch) |
| T1-4 | PA object path: three member walks, kernel 8–11% | 0.7.4, **partly** (§1) |
| T2-1 | v0.7.2 capsule gitignored and measured something else | 0.7.4 (`results/2026-09-23-lean-v072/`) |
| T2-2 | `check_evidence` pinned only the v0.6.1 capsule | 0.7.4, **partly** (§1) |
| T2-3 | v0.7.3 capsule unlisted; driver series broken | 0.7.4, **partly** (§1) |
| T2-5 | `fit_pairwise_multi` ran the pid pass twice | 0.7.4 |
| T2-6 | Three O(members×traits) scans of the same bounds | 0.7.4 |
| T2-7 | `_probability_limits` bisection repaid per bootstrap fit | 0.7.4 (memoised) |
| T2-8 | No cost signal between sampling and deterministic fitters | 0.7.4; lost in `dc5336c`, restored in `docs/inference.md` (Unreleased) |
| T2-9 | Single-family PA paid 28–67 µs of fixed dispatch | 0.7.4 |
| T2-10 | Three `eigvalsh` per register proband | 0.7.4, steps 1–3 (now one); step 4 in §1 |
| T2-11 | Dense/selected rule linear in m; crossover is quadratic | 0.7.4 (`0.03·m²`) |
| T2-12 | `_parent_links` round trip from index to id to index | 0.7.4 (`_kinship_A`) |
| T2-13 | `construct_covmat_multi` filled entry by entry | 0.7.4 |
| T3-1 | v0.7.1 headline multipliers (25×, 4×, 4×) had no artifact | Unreleased (restated qualitatively) |
| T3-2 | External 2026-09-05 efficiency review orphaned | 0.7.4 (`benchmarks/README.md:94-97`) |
| T3-3 | `bench_tetrachoric.py` printed "the same families" | 0.7.4 |
| T3-5 | v0.7.0 and v0.7.3 untagged | `573d0cf` (tags on remote; v0.7.4 is N3) |
| T3-12 | `kinship_cache_size=0` unguarded and 6,000× slower | 0.7.4 (warns) |
| T3-13 | `gibbs_chunked` built an O(F) seed array | 0.7.4 |

**Found while compiling this ledger.**

| ID | Finding | Version |
|---|---|---|
| N1 | 0.7.4's "1.22×" PA object-path figure had no artifact | Unreleased (restated qualitatively) |
| N2 | `docs/estimation.md` cited a `v0.4.0` tag that was never pushed | Unreleased (cites commit `1684fc5`; `check_evidence` pins the CSV by SHA-256) |
| F2′ | No test showed that `cip_by_stratum` changes a register score | Unreleased (`test_stratified_cips_route_each_record_to_its_own_curve`) |
| N4 | Chunked Gibbs was not seed-reproducible against the array API: the G2 collapse set was the coordinates unbounded in every family of a kernel call, so a family's draws depended on its batch | Unreleased (each family collapses its own unbounded set; chunked = array output bit for bit, `test_gibbs_family_result_does_not_depend_on_its_batch`) |
| N5 | The mixture path folded pins without P2's compatibility checks: a duplicated compatible pin crashed with an opaque `SystemError` (division by zero in the parallel kernel) and incompatible pins did not raise `ValueError` | Unreleased (`_check_pin_support` plus a collapsed-variance guard; ordinary mixture outputs bit-identical; `test_mixture_path_applies_the_pin_checks_of_the_standard_path`) |

## 4. Durable lessons

1. **A number in prose needs an artifact.** Unbacked or mis-transcribed numbers
   recur in every evidence review: 08 F1, F7, F8, F11, F12, F17 and F30; 09e
   T1-1, T1-2 and T3-1; and N1. Link to the ledger rather than repeat a number.
   Where prose must repeat one, pin it in `check_evidence.py`.
2. **Reproducing a number does not show the comparison was fair.** 08's fold
   time recomputed exactly from its CSV, yet it compared 4 ltpred threads with
   1 R worker. Record threads and load with every timing, and rerun under
   matched conditions (`bench_ltfhplus_compare_1thread.csv`).
3. **Consolidating docs can delete fixes.** 09e T2-8 was fixed in `573d0cf` and
   removed in `dc5336c` the same day. Before deleting or merging a page, grep
   this ledger for fixes that live on it.
4. **Inputs fail silently at the boundaries.** Examples are 08 F3–F6 and F20,
   09b T2-1/T2-2, and the 0.7.2/0.7.3 pid and `fam_id` fixes. Refuse input, or
   count and surface what was coerced. Never guess.
5. **A test that cannot fail is not a test.** Examples are 08 F2 (and F2′), the
   four weak oracles replaced in 0.4.0, and the negative control 09d used for
   the tutorial test. A new guard should be shown failing on the old tree.
6. **Reviews need benchmark-grade statistics.** 09c T3-1 was a 2.3-SE
   fluctuation over 40 replicates, with three components compared. 09d's "28
   restatements" was a phrase count. State the SE of a review's own estimates.
7. **Check "bit-identical" rather than assume it.** A probe called 09e T1-3
   bit-identical, but the variance was one ulp off. Compare exactly, and state
   ulp-level differences.
8. **Record measured negatives.** 09e T3-6–T3-11 and T3-15 look attractive but
   do not pay. PA row deduplication is the trap, because the quadrature win
   makes it look like a sure thing.
9. **A checker's scope drifts from its invariant.** `check_evidence` said
   "prose links, not repeats" while prose repeated numbers (09e T2-2). Its
   generalisation still skips 3 of 6 capsules. Discover inputs by glob or
   schema, not by name. A pass over nothing should be an error.
10. **Tag every release, and cite commits or hashes rather than tags.** Missing
    tags (09e T3-5, N3) and a v0.4.0 tag that was never pushed (`48ed3ec`, N2)
    leave historical numbers checkable only by git archaeology.

## 5. Independent verifications not recorded elsewhere

The reviews' headline-claim tables (08 Table 5) are superseded by later reruns;
`benchmarks/RESULTS.md` carries the current values. The oracle checks below
were written by the reviewers, and are recorded neither in RESULTS nor in
`docs/validation.md`.

*Table 6. Checks made independently of the package's tests. All passed.*

| Area | Check | Result | Review |
|---|---|---|---|
| CIP | KM and AJ estimates and Greenwood/Aalen SEs vs R `survival`/`cmprsk`, with ties, delayed entry and risk-set exhaustion | equal to all printed digits | 08 |
| Tetrachoric | MLE vs `polycor`; SE calibration over 300 tables | ~1e-5; SE 0.0586 vs SD 0.0607 | 08, 09 |
| Numerics | AS 241 vs `scipy.special.ndtri`, down to p = 1e-300 | 6e-16 relative | 08 |
| Numerics | Numba kernels vs pure-Python fallback | bit-identical | 08 |
| PA | single truncation vs closed form; pins plus ≤2 intervals vs dense oracle | 1e-10; ≤2.8e-17 | 09, 09c |
| Quadrature | exact reductions vs dense oracle; general case vs Monte Carlo (78 runs) | ≤2.3e-15; mean z +0.14, sd 0.87 | 09c |
| Pairwise | cell probabilities vs bivariate integration; analytic derivatives vs Richardson finite differences | 1.8e-16; 5e-9 / 7.6e-10 | 09c |
| Kinship | `_SelectedKinship` vs `kinship_from_pedigree`, caches 0–100,000, inbred and self-mated pedigrees | exact | 09c |
| Register | covariance-reduction guard vs legacy path (30 combinations) | bit-identical; bound margin ≥20.9× | 09c |
| Register | kinship-route sibship kernel vs role grammar; calendar censoring direction | 0.0; monotone | 09b |
| Multi-trait | Gibbs vs 2-D quadrature closed form | 0.3964 vs 0.3942 (within MC error) | 09 |
| Chunking | PA peak at F = 2M across chunk sizes; output hashes | 269.1 → 67.9 MiB; SHA-256 identical | 09e |
| Literature | every load-bearing citation vs publisher record, including Dybdahl Krebs 2026 (DOI 10.1016/j.ajhg.2025.11.016) | verified | 08 |

Design facts worth keeping. The kinship cache is a bounded LRU with an
explicit continuation stack: "eviction changes work, never ancestry" (09e §2).
From v0.5.2 to v0.7.3 the register cost profile kept its shape, at about 53%
kinship, 28% eigen/covariance and 7% PA, before 0.7.4 removed two of the
three `eigvalsh` calls (09e §6).
