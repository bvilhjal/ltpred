# Technical report

`ltpred_methods.pdf` is a methods note for colleagues and PhD
students who already know GWAS, pedigree `h²`, and Falconer's
threshold model. It uses the same three uses as the user vignette:
(I) family-history risk prediction, (II) a quantitative GWAS
phenotype, (III) architecture / relationships / aetiology from
`h²`, `r_g` and/or the CIP. It fixes the estimand, the observation-model
menu, the PA exactness boundary, the standardized inbred target,
the public role/kinship/register interfaces and calendar-time observation set,
the distinction between cumulative and incident risk, and the implementation
(grouping, collapsed untruncated coordinates,
streaming batch-means, object vs array path, `var` vs `se`), and
the load-bearing simulation tables — agreement and speed, the
LTFHPlus / LTFGRS lock (total and per-family time, fold times versus
both R packages, isolated-process peak RSS), independent-SNP marginal-association
NCP (not real-LD mixed-model evidence), personalisation,
cohort confounding, calibration,
PGS complementarity, the censoring mixture (including
liability-dependent onset), pin versus interval encodings, and
ascertainment. Section "Reducing the inference problem" derives exact
marginalization and pin conditioning, scalar ADuLT moments, parental-factor
quadrature, bounded selected relationship recursion, and pairwise likelihood
with independent-family sandwich uncertainty. The earlier benchmark tables
remain evidence for their archived source snapshots, not a rerun of all
methods in v0.6.1. The separate
[time and memory review](../benchmarks/results/2026-09-09-time-memory/README.md)
records the v0.6.1 PA-batching and parent-graph changes against v0.6.0.

The historical register-depth and prospective benchmark numbers are withdrawn
from the note pending a clean-source rerun. The vignette's six-person register
example validates observation-set invariants, not prediction accuracy.

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd report
tectonic -X compile ltpred_methods.tex
cd ..
python scripts/check_evidence.py
```

Six load-bearing report tables are static LaTeX inputs derived from the
committed benchmark CSVs (the full generator was removed in the 2026-08
lean-down). The compact evidence check covers the release-defining rows and
also rejects a report source or included table newer than the tracked PDF.

The PDF is tracked as repository documentation. The lean source distribution
omits the report because it is not a runtime dependency of the installed
package.
