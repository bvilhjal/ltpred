# Roadmap

This page lists unfinished work and deliberate boundaries. The [changelog](https://github.com/bvilhjal/ltpred/blob/main/CHANGELOG.md) records
completed implementation history, [benchmark results](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md)
are the canonical evidence ledger, and the [`research/` README](https://github.com/bvilhjal/ltpred/blob/main/research/README.md)
inventories checkout-only prototypes. Keeping those roles separate avoids the
usual archaeological sport of deciding which copied number is the current one.

## Current boundary

The supported package provides LT-FH, LT-FH++, ADuLT, and the single-trait
Pearson--Aitken PA-FGRS mixture; role-based and arbitrary-kinship inputs; a
narrow population-register driver with explicit GWAS/prediction observation
sets; Gibbs and single-trait Pearson--Aitken inference; optional additive
nuclear-family quadrature; and common-threshold liability-scale moment fitting.
Opt-in pairwise likelihood fitting covers single-trait A/C/M and joint
multi-trait heritability, genetic/residual correlations and shared-environment
covariances. Numerical, known-truth and replicated coverage checks are recorded
in the [evidence ledger](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/RESULTS.md).
Broad calibration across rare traits, family structures and sampling designs,
boundary inference and statistical-efficiency comparisons remain unfinished.
See the [guide](guide.md),
[algorithm](algorithm.md), and [assumptions](assumptions.md) for the exact
contracts.

Experimental covariance models, selection-aware fitting, HE genetic-correlation
and onset-age-decay fitting, and factor models remain under `research/`: unsupported, checkout-only
code that is deliberately absent from wheels. Its own suite runs in a CI job
and five benchmark scripts import it, so it is kept importable, but nothing in
it is part of the public API ([research extensions](research.md)).

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
   pedigrees containing thousands of informative relatives. Selected relationship
   recursion and bounded reuse now reduce the repeated local register work;
   they do not remove a dense matrix for a genuinely large informative family.

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

## Deliberate boundaries

- **Validate the new methods before changing defaults.** Broaden quadrature
  stress tests across rare and discordant large sibships, and assess pairwise
  interval coverage, boundary behavior, and efficiency against the existing
  fitter on equal cohorts. Direct incident-risk models and sparse pedigree
  message passing remain comparison studies; the new reductions do not validate
  either model as a replacement for LT-FH++.
- **Keep the PA-FGRS censoring mixture PA-only on the current roadmap.** Its
  observation is a two-component mixture, not the single hyperrectangular
  truncation distribution sampled by Algorithm G. A Gibbs comparator needs
  mixture-aware full conditionals (or a latent lifetime-case indicator) under
  the same `K_i`/`K_pop` onset-independence model, plus exact enumeration or
  quadrature checks on small pedigrees. It is not a release blocker; until such
  an implementation is added, every PA--Gibbs agreement claim remains
  restricted to no-mixture inputs.
- **Do not downgrade multi-trait shared-environment models.** `c2`/`m2`
  proportions alone do not define the cross-trait covariance of `C` or `M`.
  The dispatcher therefore raises rather than dropping them. A supported
  extension must accept and validate those cross-trait covariance inputs.

## Graduation rule

A new statistical capability moves into `ltpred/` only after its estimand and
observation model are documented, invalid inputs fail clearly, known-truth
simulation covers calibration as well as ranking, runtime and memory are
measured on a representative scale, and the public path has tests and committed
benchmark evidence. A thin orchestration layer over already supported pieces
may land on exact equivalence and invariance tests, but its scientific payoff
and performance remain unvalidated until representative benchmark evidence is
regenerated. Until then, experimental methods stay experimental; no graduation
from `research/` is currently scheduled.
