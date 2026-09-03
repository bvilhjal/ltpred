# Roadmap

This page lists unfinished work only. The [changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md) records
completed implementation history, [benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
are the canonical evidence ledger, and the [`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
inventories checkout-only prototypes. Keeping those roles separate avoids the
usual archaeological sport of deciding which copied number is the current one.

## Current boundary

The supported package provides LT-FH, LT-FH++, ADuLT, and the single-trait
Pearson--Aitken PA-FGRS mixture; role-based and arbitrary-kinship inputs; a
narrow population-register driver with explicit GWAS/prediction observation
sets; Gibbs and single-trait Pearson--Aitken inference; and common-threshold
liability-scale moment fitting. See the [guide](guide.md),
[algorithm](algorithm.md), and [assumptions](assumptions.md) for the exact
contracts.

Experimental covariance models, selection-aware fitting, genetic-correlation
fitting, and factor models remain under `research/`, which is dormant and
unmaintained (not run in CI or the test suite). They are importable from a
checkout but are deliberately absent from wheels. Its old register file is a
superseded snapshot, not an alternative public pipeline.

## Priorities

1. **Model selected family studies.** Add and validate an ascertainment
   likelihood for designs with unknown inclusion probabilities or
   zero-probability family-status strata. The current IPW route is appropriate
   only when every relevant family pattern has a known, strictly positive
   inclusion probability; its marginal case-rate screen can falsify a bad
   design but cannot certify joint positivity or weight validity.

2. **Make joint sampling diagnostics explicit.** Accept optional sampling-stratum
   labels or design probabilities and report observed joint-pattern support,
   extreme weights, and effective sample size. Diagnostics must describe what
   the supplied design establishes, not infer positivity from marginals.

3. **Scale a single giant pedigree.** Replace dense covariance construction and
   factorisation with sparse relationship operators and sparse solves for
   pedigrees containing thousands of relatives.

4. **Regenerate the register evidence.** Done 2026-09-03: both repaired
   benchmarks rerun under the provenance wrapper (RESULTS §§20–21, committed
   CSVs, manifest records). The relatives'-events contrast is unresolved at
   R = 5 on both metrics -- a larger-R rerun is the remaining evidence gap
   before quoting a relatives'-events payoff.

5. **Add a chunked driver.** Stream homogeneous family batches through the array
   kernels so biobank analyses need not hold all families and bounds in memory.
   The design should preserve deterministic grouping and the existing `O(F)`
   summary-memory contract.

6. **Validate a real-LD workflow.** Extend the opt-in HAPNEST path into a complete
   genotype-to-liability-to-LMM example with held-out calibration and null-marker
   diagnostics. Independent-SNP evidence is not a substitute for this test.

7. **Publish the first package release.** Register the PyPI trusted publisher,
   create the version tag and GitHub Release, verify the workflow-built wheel and
   sdist, and deploy the documentation site. The mechanical checklist is in
   [RELEASING.md](RELEASING.md).

8. **Investigate multi-trait Pearson--Aitken inference.** Promote it only if a
   benchmark demonstrates adequate accuracy across rare traits, asymmetric
   truncation, and larger pedigrees; otherwise retain Gibbs as the honest
   multi-trait engine.

## Graduation rule

A new statistical capability moves into `ltpred/` only after its estimand and
observation model are documented, invalid inputs fail clearly, known-truth
simulation covers calibration as well as ranking, runtime and memory are
measured on a representative scale, and the public path has tests and committed
benchmark evidence. A thin orchestration layer over already supported pieces
may land on exact equivalence and invariance tests, but its scientific payoff
and performance remain unvalidated until representative benchmark evidence is
regenerated. Until then, experimental methods stay experimental — and
`research/` currently has no maintainer, so their graduation is hypothetical
rather than scheduled.
