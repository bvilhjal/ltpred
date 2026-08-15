# Paper development plan

Status: living plan. Numerical evidence belongs in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) and its generated tables, not in
this file.

## Scope

The main methods paper joins three uses of one latent-liability core:

1. prospective prediction from personalised incidence and family history;
2. quantitative GWAS phenotypes; and
3. pedigree variance-component estimation under an explicit sampling contract.

Pearson--Aitken and Gibbs are inference engines, not separate disease models.
LT-FH++, ADuLT, and PA-FGRS are distinguished by their observation data and
bounds. Onset-age decay, genetic nurture, sex limitation, and the genetic-factor
model remain follow-up work in the unsupported `research/` package.

Table 1 is the live evidence map. Status and numerical conclusions must be read
from the linked benchmark report, which is checked against committed artifacts.

**Table 1. Each paper claim has one canonical benchmark source.**

| Paper claim | Canonical benchmark source | Remaining work |
|---|---|---|
| Engine agreement, speed, and fold order | scaling, accuracy, PA robustness | Explain the tested exactness boundary |
| GWAS power and calibration | GWAS power, LT-FH++ personalisation, confounding | Add real-LD confirmation when available |
| PGS plus family-history prediction | PGS comparison | Keep training/test separation explicit |
| Censoring and onset assumptions | PA-FGRS mixture, age/onset | Re-run after observation-model changes |
| Prospective register prediction | register pipeline, pedigree inference | Retain familywise censoring contrast |
| Heritability and A/C/M estimation | heritability, variance components, ascertainment | Separate population, IPW, and unsupported designs |
| Inference calibration | inference calibration | State finite-replicate uncertainty |
| Misspecification | misspecification, CIP estimation | Keep assumptions next to conclusions |

## Reproducibility rules

- Generate tables with `scripts/make_results.py`; do not transcribe numerical
  cells into the manuscript.
- Guard prose-level numerical claims with `scripts/check_results.py`.
- Run changed benchmarks through `benchmarks/run_benchmark.py`. A dirty-tree run
  must retain its tracked patch and untracked-source bundle; a clean committed
  run is preferred for final evidence.
- Record the manuscript commit, software environment, seeds, command, and
  artifact hashes in the evidence capsule.
- Describe unsupported `research/` results as exploratory.

## Writing order

1. Freeze the estimand and observation-model notation.
2. Draft methods from the tested implementation contracts.
3. Insert generated tables and figures from `paper/`.
4. Write results from the canonical benchmark report.
5. Add limitations: ascertainment, incidence misspecification, PA approximation,
   pedigree overlap, and transportability.
6. Rebuild the PDF and run tests, result guards, documentation checks, and the
   clean-source provenance audit.

## External dependencies

- A real-LD run requires an appropriate HAPNEST/Linux environment.
- Direct numerical comparison with LTFHPlus requires its R installation.
- PyPI publication requires the project owner's trusted-publisher setup and a
  tagged GitHub Release.
- Zenodo deposition follows the final evidence freeze.
