# Review 2026-09f: computational efficiency and benchmark design

Date: 2026-09-28. Version reviewed: v0.7.4 (`5b33429`), working tree clean.
Focus: (1) where compute time goes and how to reduce it; (2) how to strengthen
the benchmark suite, including head-to-head comparisons with published
methods. This is a read-only review: nothing in the repository was changed
except this document.

## 0. Summary

Three conclusions carry the review.

1. **The inference kernels are in good shape; the orchestration around them is
   where the time goes.** At family sizes of roughly 5–20 coordinates the
   compiled PA and Gibbs kernels are a minority of end-to-end wall-time. The
   object API spends the large majority of a warm call in Python ingestion
   (per-member pid normalization and stacking), and the register pipeline
   spends essentially all of its time in a serial per-proband loop that pays
   full per-call setup for one family at a time. Batching and caching outside
   the kernels is worth more than any kernel micro-optimization.
2. **The flagship feature — personalised LT-FH++ — has no external lock.**
   `bench_ltfhplus_compare.py` locks classic LT-FH bounds only (single
   threshold, no personalisation, no mixture). The age/sex/cohort CIP path, the
   censoring mixture and ADuLT are only ever compared against internal variants
   and simulated truth. The R packages that implement the published methods
   (LTFHPlus, LTFGRS) cover exactly those features, so extending the lock is
   mostly harness work.
3. **The suite's metric vocabulary is stronger than the field's but not
   comparable to it.** Published papers quote mean χ² at causal SNPs, power at
   5×10⁻⁸, Z-based effective sample size, genome-wide-significant locus counts
   with block-jackknife s.e., and liability-scale R² with jackknife s.e. Two of
   these appear in no current benchmark, and the ones that do use differently
   defined proxies. A small metric-alignment addition would make every gain
   number directly quotable against Hujoel 2020, Pedersen 2022 and Dybdahl
   Krebs 2024.

Findings F1–F11 are the efficiency review (Table 2 ranks the resulting
recommendations R1–R8); §2 explains what was measured. Sections 3–5 give the
benchmark plan (B1–B8, Tables 3 and 4). Assumptions stated up front: family
sizes in the 2–20 coordinate range, role-grammar workloads dominant, thread
counts pinned as in `benchmarks/README.md`, PA as the default single-trait
engine. All measurements are relative shares on one machine; treat them as
ordering evidence, not portable constants.

## 1. What was measured

All timings below were taken on the reference macOS host in the `ltpred314`
environment (CPython 3.14.6 free-threaded, NumPy 2.4.6, SciPy 1.18.0, Numba
0.66.0) with `NUMBA_NUM_THREADS=4` and BLAS threads pinned to 1, after a JIT
warm-up call. Profiles are cProfile; counter figures come from instrumenting
`ltpred.gibbs` entry points. The workloads are synthetic but sized to sit in
the package's documented operating range.

**Methodological caveat found while acting on this review (2026-09-28).**
cProfile on the free-threaded build inflates wall-times of code paths that
enter Numba parallel kernels — the wrapper `launch` showed 2.3 ms where
profiler-free `timeit` measures a 30 µs dispatch, and whole workloads
inflated roughly 2–3× (workload F: 1.40 s profiled, 0.75 s timed). The
*relative* shares of pure-Python code are usable, and every number this
review acted on was re-measured with `timeit` before implementation; the
table below keeps the profiled shares as structure evidence with
profiler-free corrections where they were taken.

*Table 1. Measured wall-time shares per workload. "Share" is the fraction of
each call's wall-time spent in the named code, so the rows are directly
comparable despite different absolute sizes.*

| Workload | Wall-time | Where the time went | Reading |
|---|---|---|---|
| A. Object API `estimate_liability`, 4 000 five-member families (PA) | 52 ms profiled / 15.3 ms timed warm | Ingestion-dominated: `_check_unique_roles` 5.3 ms timed (20 000 `_pid_key` string normalizations), stacking a similar share; PA fold a small minority | F1 |
| B1. `estimate_liability_pa_chunked`, 65 536 rows, pin/absent-rich bounds, chunk 65 536 vs 256 | 31 ms vs 637 ms timed (≈20×) | ≈2.4 ms fixed setup per chunk: per-mask-group parallel kernel launches (≈30 µs each, dozens per chunk) and `_condition_pins` per distinct mask; the covariance build itself is 21 µs | F2 |
| B2. Gibbs chunked, 2 048 families × 25 000 sweeps | 2.57 s | 98.6% in the compiled sweep (`launch`); `gibbs_params`/`_blup_from_keep` recomputed 8× (once per convergence round); chunking 2% overhead | F4, F11 |
| C. `fit_heritability`, 120 iterations × 2 500 families | 0.79 s | 85% in the `gibbs_advance` kernel (0.60 s of that is first-call JIT); `_validate_covmat` (an `eigvalsh` each) called 192× | F4 |
| D. `fit_pairwise` A+C, 2 500 families | 85 ms | 61% in `_prepare_pairs` (per-family Python scan, 12 500 `_pid_key` calls); SLSQP + `quad` ≈ 2% at 2–3 design rows | F7 |
| E. `tetrachoric_matrix`, 6 traits / 8 000 pairs | 37 ms | 89% in `scipy.stats.multivariate_normal.cdf`; 299 CDF calls for 15 pairs (≈20 per pair incl. numeric Hessian), ≈52 µs per call | F8 |
| F. `estimate_liabilities` register run, 12 210 people, 2 500 probands, `max_degree=2` | 1.40 s profiled / 0.75 s timed (301 µs/proband) | 46% per-proband F=1 `estimate_liability_from_kinship` (incl. `validate_bounds` 80 ms, `construct_covmat_from_kinship` 80 ms, `_condition_pins` 96 ms), 23% dense `_kinship_A`, 13% `extract_pedigree`, 5% one-time `build_parent_graph` | F3 |

## 2. Efficiency findings

**F1 — The object API is ingestion-bound.** In workload A the PA fold is a
small minority of a warm call (timed: `_check_unique_roles` alone is 5.3 ms of
15.3 ms). The rest is per-member Python work: `_check_unique_roles`
(`ltpred/estimate.py:444`) normalizes every member's `pid` through
`_pid_key` (`ltpred/family.py:101`, a `.lower().strip()` per id) on every
call, `_stack_object_members` (`ltpred/estimate.py:958`) rebuilds row arrays,
and `_group_by_structure` re-derives role keys. At 4 000 families ingestion
costs several microseconds per family against roughly one microsecond of
kernel, so bulk users pay a multiple that grows with family count, not family
size. The same pattern reappears in the fitters' ingestion (F7). Suggested
fix: normalize pids once at construction (`families_from_columns` / `Member`)
or memoize `_pid_key` results per `Member`; pass precomputed role-index arrays
into stacking; document the columnar array API as the recommended bulk path.

**F2 — Per-call setup is inside the chunk/batch loops.** Every chunk of
`estimate_liability_{pa,gibbs}_chunked` (`ltpred/chunked.py:130-137`,
`:166-170`) re-enters `_pa_from_role_arrays` / `_gibbs_from_role_arrays`
(`ltpred/estimate.py:878`, `:926`). The originally suspected constant — the
covariance build plus its PD check — times at only 21 µs and is not the
problem. On pin- and censor-rich bounds (the realistic LT-FH++ case, where
cases are pinned, controls censored and unobserved relatives absent) the cost
is the per-chunk **mask machinery**: `_pa_reduced_nomix` groups families by
observation mask and launches one parallel kernel per group (~30 µs dispatch
each, dozens of groups per chunk) plus one `_condition_pins` reduction per
distinct mask, together ≈2.4 ms of fixed cost per 256-row chunk (≈20× wall
over one whole-array call). The defaults (65 536 / 4 096 rows) hide this, but
the streaming `batches` API invites small batches, and F3 pays the same setup
per proband. Suggested fixes: dispatch small groups to a serial kernel (the
crossover was measured near 128 families; values identical), and share
pin-conditioning reductions across chunks of one call. Chunk results are
seed-derived from global row offsets, so neither can affect values.

**F3 — The register pipeline is a serial per-proband loop with F=1 calls.**
`pipeline.py:445-520` handles one proband at a time, and each proband calls
`estimate_liability_from_kinship` with a single row (`pipeline.py:509-511`).
Consequences visible in workload F (560 µs/proband):

1. PA's mask-shared pin conditioning (`pearson_aitken.py:472-564`, grouped at
   `:600-613`) is recomputed per call instead of shared across probands with
   the same observation mask — the design's main amortization never fires
   outside `bench_scaling.py`.
2. Per-call validation and covariance construction (`validate_bounds`
   `ltpred/_validation.py:24`, `construct_covmat_from_kinship`
   `covariance.py:716`) are paid 2 500 times for one-row inputs.
3. Dense kinship is refilled per proband (`_kinship_A`, `covariance.py:563`;
   23% of the call) even though overlapping pedigrees share most ancestor
   pairs (only the selected-pair route has a cache, `_selected_kinship.py:39`).
4. The loop is single-threaded, and `pipeline.py:448` rebuilds `member_index`
   via per-id dict lookups although `Pedigree.member_index` already holds it
   (`pedigree.py:246`).

Suggested fix, in order of value: batch probands by (structure, observation
mask) and issue one `pa_estimate_batched` call per batch; parallelize the loop
across probands (PA is deterministic and seed-free, so this is safe); cache
kinship sub-blocks across overlapping extractions. Expected win 3–10× on the
register path — this is also the path the R-package fold-times are built on,
so any claim of the form "LT-FH++ at population scale" is bounded by it. This
finding is the workload-F counterpart of the ledger's open T1-4 ("PA object
path still walks members three times") and T2-10 ("covariance-repair gate runs
per register proband").

**F4 — Redundant factorizations and validations in inner loops.** Three
instances, all cheap at d ≤ 6 and all O(d³) as d grows (multi-trait d is
traits × members):

1. `_validate_covmat` runs `eigvalsh` on every `gibbs_params` call
   (`gibbs.py:221`), including internal ones (192 calls inside one
   `fit_heritability` run, workload C), immediately followed by `np.linalg.inv`.
   A Cholesky attempt would test positive-definiteness at O(d³/3), fail fast,
   and supply the factor the inverse needs anyway.
2. `_collapsed_round` recomputes `gibbs_params(cov[keep])` and
   `_blup_from_keep` on every convergence round (`gibbs.py:505`, `:520`) even
   though the covariance, bounds and unbounded patterns are constant across
   rounds (counters: 8 recomputes over 8 rounds in B2; 64 in C).
3. `correct_positive_definite` tests with `eigvalsh` per shrink iteration
   (`covariance.py:840-848`) and is re-entered per chunk/proband even for
   matrices the caller just built.

Suggested fix: memoize the kept-block parameters and BLUP map per
(group, unbounded-pattern); thread a "certified" flag (the mechanism already
exists as `_PSD_CERTIFIED`, `covariance.py:706-713`) through internal call
paths so validation runs once at public boundaries; use Cholesky for the PD
test. Keep the full check for user-supplied matrices — the error messages are
a documented safety contract.

**F5 — `out=("genetic", "full")` repeats the entire PA fold.**
`estimate.py:917-922` loops over requested coordinates and calls
`pa_estimate_batched` once per target, each call re-copying and re-folding Σ
(`pearson_aitken.py:776`). Benchmarks routinely ask for both outputs, so they
pay 2× on the fold. Fix: a multi-target fold that reports the target-0 moments
and the final updated first-row moments from one sweep. The two outputs
currently come from differently ordered folds, so agreement must be checked
against both existing results before switching.

**F6 — The mixture path folds uninformative rows.** `_pa_family`
(`pearson_aitken.py:372-376`) folds every row of Algorithm M, including absent
`(−∞, +∞)` rows, whose `_tnorm_mixture` returns the unchanged moments while
`_pa_update` still executes its O(i²) update with a zero factor. The
no-mixture kernel skips such rows (`pearson_aitken.py:394`). Algorithm M
deliberately has no P2 reduction, so deep pedigrees — where most rows are
closure-only or uninformative — pay full O(d³/6) per family for no information.
Fix: skip absent rows in `_pa_family` as `_pa_family_nomix` does. This is exact
(an absent row carries no bounds and no `K` contribution) and matters precisely
for the `max_degree=3` register case.

**F7 — The pairwise fitters are ingestion-bound, not quadrature-bound.**
`_prepare_pairs` (`pairwise.py:142-187`) is 61% of `fit_pairwise` in workload
D (per-family `sorted` of members, 12 500 `_pid_key` calls). The two adaptive
`quad` calls per design row per criterion evaluation
(`pairwise.py:129-130`) are real but minor at 2–3 design rows (~2% of the
call); the cost grows with the row count in `pairwise_multi`, whose objective
also runs PSD-eigenvalue constraints per SLSQP step
(`pairwise_multi.py:480-489`). Fix F1's ingestion pattern here first; then
consider fixed-order Plackett quadrature or a `(t, ρ)` memo along the lines of
`pairwise_multi._PROBABILITY_LIMITS_MEMO` (`pairwise_multi.py:289-325`, itself
unbounded — worth a size cap).

**F8 — `tetrachoric` uses heavyweight CDF machinery per likelihood
evaluation.** Each objective evaluation calls
`scipy.stats.multivariate_normal.cdf` with `maxpts=10⁶`, `abseps=releps=10⁻¹⁰`
(`tetrachoric.py:86-88`), ≈52 µs per call, about 20 calls per pair including
the numeric Hessian (`tetrachoric.py:160-166`). Measured determinism is good
(eight repeat calls bit-identical under SciPy 1.18), so the issue is cost and
duplication rather than reproducibility: the deterministic Plackett-identity
bivariate probability the fitter already uses (`pairwise.py:102-139`, unequal
thresholds would be a small generalization) computes the same quantity at a
fraction of the cost, and one implementation would serve both modules. At
m = 100 traits (4 950 pairs) the current cost is roughly 11 s per matrix
versus an estimated ~1 s after unification. A lock against R `polycor` (prior
art in `docs/REVIEWS.md:257`) should gate the change.

**F9 — Per-round Python bookkeeping in `_estimate_group` (minor).** The
accumulator updates at `estimate.py:374-417` are a Python loop over active
families per round. Invisible at the sizes measured (kernel-dominated), worth
vectorizing only if profiles at tiny d and large F show it.

**F10 — Indexing and allocation churn (minor).** `np.ix_` appears 15 415 times
in workload F (100 ms cumulative) around kinship subsetting; `_pa_batched`
copies the d×d working matrix per family (`pearson_aitken.py:414`, inherent to
the mutation semantics); `simulate.py:427-453` builds `Family`/`Member`
objects in nested loops, which slows benchmark data generation but not users.

**F11 — What is already near-optimal, and should not be touched lightly.**
The Gibbs sweep spends 98.6% of its call inside the compiled kernel with
streaming O(ncols) summaries; its cost is the algorithmic
F × (burn-in + n_sim) × d² term, and the levers there are statistical (the
batch-means convergence rounds, the G2 unbounded-set collapse, `tol`/`n_sim`
defaults), not micro-optimization. PA's active-block rank-1 fold, mask-shared
pin conditioning and structure grouping are the right shapes. Every change in
F1–F8 must preserve four contracts: the numerically identical pure-Python
fallback (`gibbs.py:303-310`), per-family reseeding so parallel results are
scheduler-independent (`gibbs.py:345-346`), chunk bit-identity from global row
offsets (`chunked.py:153-158`), and the `bench_time_memory` exact
output-agreement gate. That driver is the correctness net for this whole
section: each recommendation below should land with a before/after capsule
under its existing rules.

*Table 2. Recommendations ranked by expected value over effort. "Verify in"
names the evidence that must accompany the change under the repository's own
rule that a number in prose needs an artifact (`docs/REVIEWS.md`, lesson 1).*

| # | Recommendation | Expected win | Effort | Risk | Verify in |
|---|---|---|---|---|---|
| R1 | Batch and parallelize the register loop (F3) | 3–10× on `estimate_liabilities` | High | Medium (must stay bit-identical) | New `bench_time_memory` case at 200k probands; `bench_register_pipeline` part 4 |
| R2 | Serial dispatch for small PA batches/groups; share pin reductions across chunks (F2) | measured 1.5–1.6× chunked PA at chunk 64–1024 after the serial half alone | Low (done) / medium (pin sharing) | Low | `bench_time_memory` `pa_*` cases; `bench_scaling` |
| R3 | Cache pid/role normalization; vectorize object ingestion (F1, F7) | measured −17% on warm re-validation of 4 000 families; stacking still open | Low (pid half done) | Low (error messages pinned by tests) | `bench_scaling` object path; a new ingestion case |
| R4 | Cholesky PD test, certified-flag threading, memoize round invariants (F4) | small at d ≤ 6, material for multi-trait and large pedigrees | Low–medium (memo done) | Medium (validation contract) | `bench_time_memory` + the fitter benchmarks at multi-trait d |
| R5 | Skip absent rows in the mixture fold (F6) | proportional to uninformative-row share at depth 3 | Low (done) | Low | `bench_pafgrs_mixture` (paired before/after) |
| R6 | Multi-target PA fold (F5) | 2× whenever `out=("genetic","full")` | Medium | Medium (two orderings to reconcile) | `bench_accuracy` rows requesting both outputs |
| R7 | Unify bivariate CDF on the deterministic Plackett quadrature (F8) | 5–20× per tetrachoric pair | Low | Low | `bench_tetrachoric` + R `polycor` lock |
| R8 | Minor churn (F9, F10) | a few percent in orchestration | Low | Low | existing driver cases |

## 3. The benchmark suite today, and its gaps

The suite is unusually disciplined (provenance wrapper, evidence checker,
capsule archives). Its shape is also consistent: of 29 scripts, exactly two
touch external software — `bench_ltfhplus_compare.py` (LTFHPlus Gibbs, LTFGRS
PA, classic LT-FH bounds, 200 nuclear families) and the optional `ldpred3` arm
of `bench_pgs_comparison.py`. Everything else compares internal variants to
simulated truth. Three gaps follow directly, and they line up with gaps the
documents already admit:

1. **No external lock on the personalised path.** The LT-FH++/ADuLT
   personalisation (age/sex/cohort CIPs, censoring mixture, onset encodings)
   is the package's raison d'être and is benchmarked only against itself
   (`bench_ltfhpp_personalization`, `bench_pafgrs_mixture`, `bench_fh_prediction`).
2. **No benchmark at all for two advertised engines/features.** The
   quadrature engine (README's third engine) appears in no benchmark, and
   multi-trait *scoring* has none (`docs/ROADMAP.md:41` states the promotion
   gate is exactly such a benchmark). The time/memory driver covers neither
   Gibbs, quadrature nor the fitters in its seven cases.
3. **Gain metrics are not the field's metrics.** The strict causal-NCP ratio
   and squared-correlation eff-N proxy (defined in the `benchmarks/RESULTS.md`
   preamble, with the caveat that they are not interchangeable) are
   defensible, but papers in this literature quote a specific set of
   quantities (Table 4). Without them, "1.47× NCP" cannot be placed beside
   Hujoel 2020's "+22% mean χ² vs GWAX, +55% power" or Pedersen 2022's
   effective-sample-size increase.

Two further gaps worth naming because the roadmap already tracks them: the
pairwise/moment fitters have never been compared with published heritability
estimators on equal cohorts (`docs/ROADMAP.md:58-62`), and the HAPNEST real-LD
path "was not run for any artifact" (`benchmarks/RESULTS.md`), leaving every
association claim resting on independent SNPs.

## 4. Published methods to compare against

*Table 3. Comparator landscape. Citations verified against Crossref/Europe
PMC/CRAN in a 2026-09-28 literature scan; unverified items are flagged rather
than guessed. "Priority" is how much a lock against it adds beyond the current
suite.*

| Method / software | Citation | Implements | How to invoke | Compare on | Priority |
|---|---|---|---|---|---|
| LTFHPlus 2.2.0 (CRAN, GPL-3) | Reference implementation of Pedersen EM et al., *AJHG* 109:417–432 (2022), DOI 10.1016/j.ajhg.2022.01.009; CRAN DOI 10.32614/CRAN.package.LTFHPlus | LT-FH, LT-FH++ (age/sex/cohort thresholds), ADuLT, arbitrary pedigrees; Rcpp Gibbs; register/pedigree utilities since 2.2 | `estimate_liability(.tbl, family_graphs, h2, tol=0.01, ...)` via `Rscript` | **Personalised LT-FH++ and ADuLT score locks** (missing today), scores vs seeds/tolerance, per-family time, throughput at 350k scale | **1** |
| LTFGRS 1.0.1 (CRAN, GPL≥3) | Reference implementation of Dybdahl Krebs M et al., *AJHG* 111:2494–2509 (2024), DOI 10.1016/j.ajhg.2024.09.009; CRAN DOI 10.32614/CRAN.package.LTFGRS | PA engine, **censoring/onset mixture** (`useMixture=TRUE`), classic LT-FH, Kendler FGRS, censoring utilities | `estimate_liability(..., method="PA", useMixture=...)`, `kendler_simplified(...)` | Mixture lock (current lock sets `useMixture=FALSE`), PA-FGRS mixture panels, Kendler comparison | **1** |
| Original LT-FH scripts | Hujoel MLA et al., *Nat Genet* 52:541–547 (2020), DOI 10.1038/s41588-020-0613-6 | Monte-Carlo LT-FH over 377 configurations | Scripts at data.broadinstitute.org/alkesgroup/UKBB/LTFH/ (language unverified) | One-off compatibility lock of the classic-LT-FH mode; documents the MC-vs-analytic gap | 2 |
| GWAX | Liu, Erlich & Pickrell, *Nat Genet* (2017), DOI 10.1038/ng.3766 | Proxy-case binary phenotype | ~20 lines in `_common.py` | Baseline arm for every association benchmark (both key papers beat it); quote the proxy-GWAS bias caveats (Wu et al. 2024, DOI 10.1038/s41588-024-01963-9; rebuttal DOI 10.1038/s41588-024-02023-y) | 2 |
| PRS-FH | Hujoel MLA et al., *Cell Genomics* (2022), DOI 10.1016/j.xgen.2022.100152 | Combined PGS + family history with relative-class pseudo-heritabilities | Equations reimplementable (code statement unverified) | `bench_pgs_comparison` arm; liability-scale R² with jackknife | 3 |
| PCGC regression | Golan, Lander & Rosset, *PNAS* (2014), DOI 10.1073/pnas.1419064111 | Moment (pair) h² under ascertainment — the closest published analogue to `fit_pairwise` | Short reimplementation, no canonical package | Fitter comparisons: bias/SD/coverage/runtime on equal cohorts | 2 |
| GCTA-GREML (+ Lee 2011 correction, DOI 10.1016/j.ajhg.2011.02.002; Hayeck 2015, DOI 10.1016/j.ajhg.2015.03.004) | GCTA: Yang et al., *AJHG* (2011), DOI 10.1016/j.ajhg.2010.11.011 | SNP-based h² with ascertainment handling | CLI | Fitter comparison arm (genotype-based, so interpret differences in estimand) | 2 |
| OpenMx / MCMCglmm | Boker et al., *Psychometrika* (2011), DOI 10.1007/s11336-010-9200-6; Hadfield, *JSS* 33:2 (2010), DOI 10.18637/jss.v033.i02 | Pedigree/twin threshold FIML; Bayesian animal model | CRAN | `fit_variance_components` / tetrachoric locks (OpenMx is also an oracle for `polycor`-free tetrachorics) | 2 |
| FamEvent | Choi et al., *JSS* 97(7):1–30 (2021), DOI 10.18637/jss.v097.i07 | Ascertained age-at-onset family simulation + penetrance models | CRAN | **Independent simulator** for mixture/ADuLT arms (B6) | 2 |
| Kim–Kwak–Won LTM | *Genet Epidemiol*, DOI 10.1002/gepi.22244 (year discrepancy 2019/2022) | Heritability of dichotomous traits on ascertained pedigrees | Software unverified | Only if an implementation surfaces | 4 |
| Kendler FGRS | Kendler et al., *JAMA Psychiatry* (2021), DOI 10.1001/jamapsychiatry.2021.0336 | Kinship-weighted family scores (pragmatic incumbent) | `LTFGRS::kendler_simplified` | Practical baseline in the PA-FGRS panels (as its own paper does) | 3 |
| LT-FGRS preprint | Pedersen et al., medRxiv (2026), DOI 10.64898/2026.06.15.731517 (abstract verified; full text blocked) | Unifying R package; itself a benchmark-of-implementations paper | — | The closest existing template for this whole plan; read before finalizing B1/B2 designs | 1 (as a template) |

Not found / not usable: "famPRS", "LTML", "solium" (no such methods located;
search terms recorded in the scan), So et al. 2011 PA software, Fam-meta code,
FHAT code, SPACox citation. A "brute-force quadrature vs LT-FH++" runtime
comparison does not appear to exist in any of these papers — the published
speed contrast is Gibbs vs the original Monte Carlo (Pedersen 2022's
"32 cores, 350 000 individuals in <25 min" scaling claim being the headline to
reproduce-style against).

## 5. Proposed benchmark work

Each item states its question, its design and where it belongs (a new script or
an arm of an existing one). All of them inherit the provenance wrapper,
`check_evidence.py` pinning for any release-defining number, and the quoting
rules already learned: report thread counts with every fold, keep mixed
algorithm rows clearly labelled (the 8086× row of RESULTS §30 is
"LTFHPlus-Gibbs vs ltpred-PA", not "the same method, faster"), and separate
first-call JIT from warm timings.

**B1 — Turn the R lock into a published-methods suite.** Extend
`bench_ltfhplus_compare.py` (and `ltfhplus_compare.R` / `ltfgrs_compare.R`)
along four arms: (a) personalised LT-FH++ — age/sex/cohort thresholds and
interval/pinned bounds — against `LTFHPlus::estimate_liability`, which is the
missing flagship lock; (b) ADuLT (no proband phenotype) against the same;
(c) censoring mixture against `LTFGRS` `useMixture=TRUE`, on the exact
generative panels `bench_pafgrs_mixture.py` already simulates (threshold
crossing, stochastic onset, liability-dependent onset × MID/OLD censoring);
(d) classic LT-FH against the original Broad scripts as a one-off. Keep the
score-lock metrics (corr/RMSE/max-abs), fold-times with thread provenance and
isolated peak RSS, and extend the `tests/test_r_lock.py` committed-fixture
pattern to the new arms so port regressions fail CI without R. The current
lock's workload (200 equal nuclear families, LTFHPlus's own Gibbs settings)
should remain the constant cell so historical RESULTS §30 rows stay
comparable.

**B2 — A workload matrix and a population-scale throughput curve.** The lock
should cross: family size (2–11 members), depth (`max_degree` 1–3), trait
rarity (K ∈ {0.005, 0.05, 0.3}), personalisation on/off, mixture on/off, and
ascertainment (1× and the Pedersen 2022 design of downsampling controls to
50%; optionally the ADuLT 5× oversampling × two generative models). Report
ms/family and probands/s with explicit thread counts, plus peak RSS. The
headline cell is a 350 000-proband register-style run — directly comparable to
the published "350k in <25 min on 32 cores" claim — reported for ltpred PA,
ltpred Gibbs and LTFHPlus Gibbs on matched inputs. This is also the natural
home for R1's before/after evidence.

**B3 — Fitter comparisons on equal cohorts (the roadmap's own ask).** On the
same simulated cohorts, compare `fit_heritability` / `fit_variance_components`
/ `fit_pairwise` / `fit_pairwise_multi` against: PCGC regression
(reimplemented; the closest published analogue to the pairwise moment fit),
GCTA-GREML with the Lee/Hayeck ascertainment corrections where genotypes
exist, OpenMx threshold FIML (the gold standard for pedigree binary traits)
and MCMCglmm. Metrics: bias, across-replicate SD, reported-s.e. calibration,
95% coverage, boundary behaviour and wall-time — the columns
`bench_pairwise_recovery.py` and `bench_fit_heritability.py` already produce,
plus the external estimator columns. Include the ascertainment cells
(`bench_ascertainment.py`'s schemes) since ascertainment is exactly where the
published methods disagree (Golan 2014 shows REML underestimates). Add a
`polycor` lock for `tetrachoric` alongside (R7).

**B4 — Metric alignment to the field.** *Table 4* lists the mapping. The
additions are mechanical given the existing association helper: the Z-based
effective-sample-size comparison of Pedersen 2022 (mean χ²-based N_eff ratio
across matched analyses), the genome-wide-significant-locus count with a
200-block genomic jackknife s.e. (Hujoel 2020's real-data metric; needs the
HAPNEST LD arm of B7 to mean anything), liability-scale R² with jackknife s.e.
under the Lee conversion (PRS-FH's metric, for `bench_pgs_comparison`), and
the S-LDSC attenuation ratio as a confounding diagnostic. Keep the strict
causal-NCP ratio beside them — it is a cleaner estimand than the field's mean
χ² — but publish both so results are quotable in either vocabulary.

*Table 4. Metric alignment: what the comparators' papers quote, and what the
suite reports today.*

| Metric | Definition | Quoted by | In the suite today |
|---|---|---|---|
| Mean χ² at causal SNPs | NCP proxy | Hujoel 2020, ADuLT 2023 | Yes (`bench_gwas_power`) |
| Power at 5×10⁻⁸ | fraction of causal SNPs passing | Hujoel 2020, Pedersen 2022 | Yes |
| Z-based effective sample size | N_eff from matched Z-scores | Pedersen 2022, Hujoel 2020 (real data), Guan 2025 | No (squared-correlation eff-N proxy only) |
| GWS-locus count + 200-block jackknife s.e. | replication-scale discovery | Hujoel 2020, Pedersen 2022, ADuLT 2023 | No (needs real LD) |
| Liability-scale R² + jackknife s.e. | Lee conversion of prediction R² | Hujoel 2022 (PRS-FH) | Partial (test R² only) |
| λ_GC and S-LDSC attenuation ratio | null calibration / confounding | Hujoel 2020 | λ_GC yes; attenuation ratio no |
| AUC and joint-model incremental value | classification | Dybdahl Krebs 2024 | Yes (`bench_register_pipeline`, `bench_pgs_comparison`) |
| Calibration slope / decile curve | posterior-mean self-calibration | — (not used in these papers) | Yes — a differentiator; keep |
| Causal-SNP NCP ratio | strict NCP ratio | — (stricter than the field) | Yes — keep beside the field metrics |

**B5 — Evidence for the two unbenchmarked capabilities.** (a) The quadrature
engine: accuracy against PA and Gibbs on nuclear families (its documented
scope), a cost curve over family size and bound patterns, and the stress cells
the roadmap already lists (rare traits, discordant large sibships). (b)
Multi-trait scoring: the accuracy benchmark whose non-existence currently
blocks promoting multi-trait PA (`docs/ROADMAP.md:41`) — rare traits,
asymmetric truncation, larger pedigrees, with Gibbs as the reference arm. Both
also become `bench_time_memory` cases (B8), satisfying the graduation rule
that new capabilities ship with representative runtime evidence.

**B6 — Independent-simulator triangulation.** All statistical benchmarks draw
from ltpred's own liability-threshold simulator; a shared misunderstanding of
the observation model would be invisible. FamEvent (CRAN) generates ascertained
age-at-onset pedigrees with censoring and competing risks independently; feed
its output through the PA-FGRS mixture and ADuLT arms (B1c) and the
`bench_cip_estimation` cells. This also imports ADuLT 2023's evaluation grid
(two generative models × ascertainment severity), which is the most
transferable design in the literature.

**B7 — Run the real-LD arm once.** The HAPNEST opt-in of
`bench_gwas_power.py` / `bench_pgs_comparison.py` has never backed an
artifact. One committed run converts "results assume independent SNPs" from a
standing caveat into a bounded statement, and is a prerequisite for B4's
locus-count metric.

**B8 — Close the performance-regression gaps.** Three concrete items: (1) run
the `bench_time_memory` driver across v0.7.1 → v0.7.4 now, closing the
acknowledged capsule gap before the next release; (2) extend its seven-case
list with Gibbs, quadrature, a fitter and a large (≥200k proband) pipeline
case — the current cases cover none of the first three and only a 300-proband
pipeline; (3) adopt the driver's exact output-agreement gate as the
correctness net for R1–R8, each landing with a before/after capsule in the
`benchmarks/results/` format. For F3/R1 specifically, an allocation-profile
capsule (the `2026-09-23-lean-v072` pattern) is the right first evidence if
wall-clock on a shared host would be untrustworthy.

## 6. Scope and limitations

This review read the full core (`ltpred/*.py`, ~10k lines), all 29 benchmark
scripts' designs, the results/roadmap/review documents, and profiled six
workloads at small-to-moderate scale on one machine. It did not: rerun any
statistical benchmark; exercise the R comparators (whether LTFHPlus/LTFGRS
install and run on this host is untested here); measure multi-trait d beyond
noting its scaling shape; or verify the literature scan's flagged unverified
citations. Profile shares at d ≤ 6 will shift with scale — in particular F4
grows with d and F2 shrinks in relative terms as the kernel grows — so the
ranked table is a statement about the documented operating range, and R1–R8
each carry their own verification plan for exactly that reason.

## 7. Postscript: what was implemented on 2026-09-28

Acting on this review, five changes landed in the working tree (uncommitted;
1024-test fast suite, ruff and bit-identity goldens all green, two new
regression tests in `tests/test_pearson_aitken.py`):

1. **R5 (F6):** the PA-FGRS mixture fold skips absent rows exactly as the
   no-mixture kernel does. A family with extra absent relatives now scores
   bit-identically to the same family with those rows deleted; previously the
   posterior variance moved in the last ulp (golden check across PA, mixture,
   Gibbs, chunked, object, fitter and register paths: every output
   bit-identical except mixture variances, worst diff 4.4e-16).
2. **R2, serial half (F2):** batches and mask groups below 128 families run
   on serial kernel twins (`_PARALLEL_MIN_FAMILIES`); the crossover was
   measured, not assumed. Chunked PA on pin/absent-rich bounds, timed
   before → after: chunk 1024 154.9 → 102.7 ms, chunk 256 386.1 → 253.3 ms,
   chunk 64 900.0 → 557.4 ms (**1.5–1.6×**); whole-array calls unchanged.
3. **R3, pid half (F1):** `_member_pid_key` caches the normalization on the
   Member, invalidated on pid reassignment (a pattern the test suite itself
   uses). Warm re-validation of 4 000 families: 5.3 → 4.4 ms. Object-API
   first-pass wall-time is unchanged — the remaining ingestion cost is
   `_stack_object_members`, deliberately left alone here.
4. **R4, memo half (F4):** the Gibbs convergence loop computes each group's
   collapse invariants once across rounds (`_collapse_cache` through
   `gibbs_estimate_batched`); counters previously showed one recompute per
   round. Draws are bit-identical.
5. **R1, item 4 (F3):** the register driver reuses `Pedigree.member_index`
   instead of rebuilding it per proband. Pipeline timing moved 301 → 294
   µs/proband (≈2%); the structural levers of R1 (mask-batched probands,
   parallel loop, shared kinship blocks) remain open and are still the
   largest single win available.

Two corrections to this review's own first draft are recorded above: the
per-chunk tax is the mask machinery, not the covariance factorization (21 µs),
and cProfile on the free-threaded build inflates Numba-parallel paths 2–3× —
both found by re-measuring with `timeit` before implementing. The remaining
open items, in value order: R1's batching/parallelism, cross-chunk sharing of
`_condition_pins` reductions (the other ~half of the chunk tax; touches the
most numerically sensitive code in the PA path, so it wants its own careful
pass), R6, R7, and the Cholesky-based PD test of R4.
