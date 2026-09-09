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
with independent-family sandwich uncertainty. Its subsection "Time and memory
between versions" reports the
[9 September 2026 rerun](../benchmarks/results/2026-09-09-time-memory-v061-rerun/README.md)
of v0.6.1 against v0.6.0: 2.25× faster mixed-mask PA and 1.73× faster
million-record graph construction, lower peak memory, and exact output
agreement across all seven workloads. The small register-scoring improvement
is only about 3%. The table separates warm runtime from whole-process RSS;
the run capsule also reports first-call time and separately traced allocations.
The report is versioned v0.6.2, a documentation patch with the same numerical
implementation as v0.6.1. Earlier statistical tables retain their historical
provenance; this efficiency rerun does not refresh their statistical evidence.

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
lean-down). The compact evidence check covers the release-defining rows, the
time/memory JSON capsule and its documentation tables, and
also rejects a report source or included table newer than the tracked PDF.

The PDF is tracked as repository documentation. The lean source distribution
omits the report because it is not a runtime dependency of the installed
package.
