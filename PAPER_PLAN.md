# Paper development plan — ltpred as a research paper

Status: living document, pinned 2026-08-10. Owner: project owner + assistant.
Scope: turning ltpred's three inference uses into a main methods paper plus a
JOSS companion. Not in scope (follow-up papers): the onset-age decay model,
nurture, sex-limitation, and the genetic factor model (all `research/`).

## The main paper

**Working title:** *Liability-threshold family-history inference at registry
scale: prediction, GWAS phenotypes, and variance-component estimation from
pedigrees.*

**Positioning.** Not "a faster LT-FH++" (PA-for-family-history is Krebs et al.
2024). The contribution is the conjunction:

1. The first fully personalised LT-FH++ implementation with two
   exactness-characterized inference engines (Gibbs; deterministic
   Pearson–Aitken with an explicit exactness boundary and measured fold-order
   spread).
2. Three inference tasks off one latent-liability core:
   - **Prediction** — posterior-mean liabilities as individual risk scores:
     personalised age/sex/cohort thresholds, the PA-FGRS censoring mixture,
     familywise censoring for honest prospective prediction, exact tabular
     kinship on arbitrary pedigrees (the LT-FGRS niche).
   - **GWAS phenotypes** — replicated 1.47 ± 0.04× causal-SNP NCP ratio;
     +0.145 ± 0.015 LT-FH++ increment over ADuLT (paired); cohort-blind
     thresholds inflate λ_GC up to 16× while cohort-aware stay ≈ 1
     (personalized thresholds as a confounding-control device).
   - **Variance components from pedigrees** — HE-on-augmented-liabilities
     fitters: `h²`, relationship-specific shared environments (C/M),
     cross-trait `r_g`; identifiability arithmetic; sampling contracts;
     bootstrap calibration. Twin-study quantities without twins.
3. An **evaluation framework** where every claim is calibrated and every
   quoted number re-derives from committed artifacts (this document's
   provenance rules are part of the contribution).

**Venue:** PLOS Genetics or AJHG (with the Tier-A evidence); Bioinformatics as
fallback. **Companion:** JOSS software paper once PyPI ships (one-time
trusted-publisher registration is the only remaining step; see
docs/RELEASING.md).

## Evidence map (paper artifact → benchmark → status)

| Paper element | Source benchmark | Status |
|---|---|---|
| Engines: speed, agreement, fold-order | bench_scaling, bench_accuracy, bench_pa_robustness | ✅ committed; pa_robustness → multi-seed (W2) |
| Power: 1.47× NCP, real-LD confirmation | bench_gwas_power (+HAPNEST) | ✅ indep.-SNP; ⛔ HAPNEST blocked (no singularity on this machine) |
| PGS baseline + PGS+FH joint | bench_pgs_comparison.py | ✅ §28 (joint R² 0.339 vs PGS 0.236, LT-FH 0.170; a·b·√p verified) |
| Personalization payoff (ADuLT vs LT-FH++) | bench_ltfhpp_personalization | ✅ gold standard (70 replicate rows) |
| Confounding λ_GC | bench_confounding (+HAPNEST later) | ✅ independent-SNP |
| Censoring mixture | bench_pafgrs_mixture | ✅ paired uncertainty (all Δcorr CIs < ±0.0002) |
| Prediction: prospective register, discrimination | bench_register_pipeline, bench_pedigree_inference | ✅ 5-replicate + AUC; leakage contrast resolved (ΔAUC +0.065 ± 0.014) |
| Variance components h²/C/M, r_g | bench_fit_heritability, bench_variance_components, bench_genetic_correlation | ✅ replicated (25 cohorts) |
| Inference calibration (Type-I, coverage) | bench_inference_calibration | ✅ + r_g-test null (W2) |
| R LTFHPlus numerical comparison | NEW (small) | ⛔ needs `remotes::install_github("EmilMiP/LTFHPlus")` (R exists at /usr/local/bin/R) |
| Competitor estimator for h² (LDSC/GREML-style) | via ldpred3 `ldsc_h2` arm in bench_pgs_comparison | 🚧 (W1, optional) |
| Misspecification / assumption stress | bench_misspecification + liability-dependent-onset arm | ✅ onset-timing arm in §16 (ρ=0.6 copula; pin slope 0.92) |

## Provenance and structure rules (durable fixes, W1)

- **Generated, not transcribed.** `scripts/make_results.py` renders paper
  tables from the committed CSVs; `scripts/check_results.py` re-derives the
  RESULTS.md headline values from the artifacts and fails on mismatch (CI-able
  guard against the 2026-08 doc/artifact skew class).
- **Provenance committed.** `run_manifest.jsonl` and `run_logs/` are tracked
  (since c1a6a23). Every new/changed benchmark run is recorded the same way
  (the `benchmarks/run_benchmark.py` driver does this).
- **Environments.** `ltpred314` = package verification (pytest, ruff, docs
  gates; dependency-minimal, CI-equivalent). `ldpred3` = benchmark analysis
  and paper artifacts (pandas, matplotlib, optional ldpred3 PGS backend).
  Documented in benchmarks/README.md.
- **paper/** (created when drafting starts): manuscript, one figure script per
  figure, figure→benchmark→CSV→seed manifest, pinned ltpred commit,
  reproducibility capsule; evidence bundle → Zenodo DOI.

## Work breakdown

**Wave 1 (this session, parallel):** ✅ done 2026-08-10
- W1-A `bench_pgs_comparison.py`: arms = case/control, LT-FH++, PGS-only,
  PGS+LT-FH++ joint. Design pinned: same generative framework as
  bench_gwas_power (independent SNPs); 50/50 train/test split (PGS trained on
  train-cohort CC GWAS, scored on test); default PGS = self-contained numpy
  (clumping-free independent SNPs: Z-scored marginal weights), optional
  `--pgs-backend ldpred3` for an LDpred-class baseline; reports NCP ratios,
  incremental R², PGS↔FH correlation vs the `a·b·√p` prediction, calibration.
  **Landed as RESULTS.md §28:** LT-FH NCP 1.446 ± 0.023×; test R²: PGS
  0.236, LT-FH 0.170, joint 0.339; PGS↔FH 0.2009 vs theory 0.2003.
- W1-B §20/§21 replication + discrimination: 5 paired replicates each, new
  long-format CSVs; prospective AUC added. **Resolved the leakage contrast:**
  relatives' post-index events genuinely help rank discrimination (ΔAUC
  +0.0648 ± 0.0139, CI [+0.026, +0.103]); the corr contrast stays unresolved
  (+0.0084 ± 0.0208) — score-scale effect (slope drops 0.26 → 0.16). §20
  payoff now 0.569 ± 0.013 vs 0.498 ± 0.016 (paired Δ +0.0714 ± 0.0064).
- W1-C `scripts/make_results.py` + `scripts/check_results.py` (30 guards,
  wired into tests via tests/test_check_results.py); benchmarks/README.md
  env-split note.

**Wave 2 (this session, after wave 1):** ✅ done 2026-08-10
- W2-A §16: bench_pafgrs_mixture retains per-replicate paired differences
  (new CSV). **Verdict (updated 2026-08-14):** all ten Δcorr CIs — including
  the liability-dependent ρ = 0.6 arm — tighter than ±0.0002; every Δslope
  CI excludes zero. Pinning is calibrated only under threshold crossing
  (slope 0.92 at ρ = 0.6, heavy censoring).
- W2-B: bench_inference_calibration Part 4 — `test_genetic_correlation`
  under the r_g = 0 null: **0/25 rejections at 0.05** (CP CI 0–0.14).
  bench_pa_robustness now 3-seed (§14 rewritten with across-seed mean ± SE;
  worst single-seed PA–Gibbs agreement 0.99813).

**Blocked / later (not this session):**
- HAPNEST real-LD run (needs singularity; Linux VM or remote host).
- R LTFHPlus numerical comparison (needs LTFHPlus installed in R).
- ~~Liability-dependent-onset-timing generative arm for the mixture~~
  done 2026-08-14 (`onset_model="liability_dependent"`, ρ=0.6; RESULTS §16).
- GREML-class competitor for the h² fitter (needs GCTA; LDSC arm via ldpred3
  covers part of the question in W1-A).
- Kendler-FGRS prediction baseline (implement from the 2021 paper; affected-
  relative-count baseline ships in W1-B instead).
- Chunked/streaming driver; paper/ directory + drafting; PyPI trusted
  publisher; Zenodo deposit.

**Definition of done for this session:** all W1/W2 items implemented, each
benchmark rerun with artifacts (CSV/log) committed-ready, RESULTS.md updated
from the new artifacts and `check_results.py` green, full pytest + ruff green.
No git commits without the owner's go-ahead.
