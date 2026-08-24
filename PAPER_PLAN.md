# Paper development plan

Status: living plan. Numerical evidence belongs in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) and its generated tables, not in
this file.

## Scope

The main methods paper joins three uses of one latent-liability core
(same numbering as the vignette and the methods note):

- **I.** risk prediction from family history (own status out; optional
  downstream PGS);
- **II.** quantitative GWAS phenotypes (own status in); and
- **III.** architecture, relationships and aetiology from liability-scale h²
  / r_g and/or the CIP, including optional pedigree
  variance-component estimation under an explicit sampling contract.

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

- Do not transcribe numerical cells into the manuscript; the tables in
  `paper/tables/` were generated from the committed benchmark CSVs (the
  generator script was removed in the 2026-08 lean-down; regenerate by hand
  from the CSVs if a table needs updating).
- Prose-level numerical claims are no longer machine-guarded; verify them
  against the committed CSVs when editing.
- Record the manuscript commit, software environment, seeds, and command for
  each benchmark run in `benchmarks/RESULTS.md`.
- Describe dormant `research/` results as exploratory.

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
- Direct numerical comparison with LTFHPlus and LTFGRS:
  `bench_ltfhplus_compare.py` (opt-in; requires the R packages). Locked
  against LTFHPlus 2.2.0 (Gibbs) and LTFGRS 1.0.1 (`method="PA"`) on
  2026-08-15, with isolated-process peak RSS and per-family times.
- PyPI publication requires the project owner's trusted-publisher setup and a
  tagged GitHub Release.
- Zenodo deposition follows the final evidence freeze.
