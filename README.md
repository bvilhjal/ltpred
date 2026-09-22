# ltpred

**Tutorial:** [five questions](docs/tutorial.md) on a simulated cohort whose
genetic liability is known. Steps 1–4 are one disease; step 5 is two traits.
Then [choose a method](docs/guide.md).
**Reference:** [vignette](https://bvilhjal.github.io/ltpred/vignette/)
· [markdown](docs/vignette.md)

**ltpred** is a Python implementation of **LT-FH++**, extending the original
**LT-FH** family-history phenotype
([Hujoel et al. 2020, *Nature Genetics*](https://doi.org/10.1038/s41588-020-0613-6))
with age of onset (sex and cohort effects enter through the sex/cohort-specific
thresholds or cumulative-incidence proportions (CIPs) you supply). It is a faithful
port of the R package [LTFHPlus](https://github.com/EmilMiP/LTFHPlus)
([Pedersen et al. 2022, AJHG](https://doi.org/10.1016/j.ajhg.2022.01.009); the
family-free age-dependent variant, ADuLT, in
[Pedersen et al. 2023, Nat Commun](https://www.nature.com/articles/s41467-023-41210-z)),
and it also ships a deterministic **Pearson–Aitken (PA)** inference engine plus the
PA-specific, optional PA-FGRS censoring model
([Dybdahl Krebs et al. 2024, AJHG](https://doi.org/10.1016/j.ajhg.2024.09.009)).
The published base PA-FGRS model uses lifetime-threshold case intervals and an
age-censored-control mixture. ltpred supports both LT-FH++ case encodings that
LTFHPlus emits — the onset pin (`use_fixed_case_thr = TRUE`, ltpred's default)
and the onset interval (`use_fixed_case_thr = FALSE`, the R default); pairing the
interval with the mixture is a PA-FGRS-style variant, not the paper's
PA-FGRS_ADT specification.

The score is the **posterior mean genetic liability**: conceptually a
liability-threshold, age-aware, family-history analogue of BLUP /
selection-index prediction, projecting binary and censored disease observations
onto the proband's additive genetic value the way a breeding value is predicted
from relatives (see
[algorithm.md](docs/algorithm.md#connection-to-selection-index-and-blup)) —
not a SNP polygenic score.

## Installation

PyPI publication is pending (see the [roadmap](docs/ROADMAP.md) and
[release checklist](docs/RELEASING.md)). Until then, installation from a source
checkout is the supported path:

```bash
git clone https://github.com/bvilhjal/ltpred.git
cd ltpred
pip install -e ".[fast,test]"
pytest -q          # optional: confirm the install
```

`numpy` and `scipy` are required. `[fast]` adds an optional **Numba** JIT for the
Gibbs sweep and is strongly recommended — the pure-Python fallback is numerically
identical, just slower. For biobank-scale runs see the guide's
[scaling section](docs/estimation.md#scaling-to-large-cohorts).

## Quickstart

```python
from ltpred import simulate_under_LTM_single, estimate_liability

sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=0.5, pop_prev=0.05, n_sim=2000,
    use_age=False, seed=1,
)
pa = estimate_liability(sim.families, h2=0.5)
pa.est["genetic"]
```

`use_age=False` is the classic LT-FH call. `h²` is required and
disease-specific. Own diagnosis in the input is a GWAS phenotype; leave it
out to predict from family history. The
[tutorial](docs/tutorial.md) walks the same calls on a simulated register:
its printouts are checked, and each step names the quantity it estimates.
Six hand-typed rows
are the [quickstart](docs/quickstart.md). The model is in
[algorithm.md](docs/algorithm.md).

## Benchmarks

[`benchmarks/`](benchmarks/) compares model encodings and inference engines on
simulated data, and [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) is the
evidence ledger: PA versus Gibbs agreement and speed, the locked comparison to
the public R packages LTFHPlus and LTFGRS, independent-SNP marginal-association
gains for classic LT-FH and personalised LT-FH++ (with a matched family-free
ADuLT arm), calibration, cohort confounding, the PA-FGRS censoring mixture,
fitter recovery and ascertainment, and matched time/memory comparisons between
package versions. Its provenance header says which numbers are historical
snapshots and how to rerun them, and `scripts/check_evidence.py` reconciles
the release-defining claims in that ledger and in the
[methods report](report/ltpred_methods.pdf) with the committed artifacts.
Real-LD runs use [HAPNEST](benchmarks/hapnest/README.md) genotypes (opt-in;
not run for the committed results).

## Scope

ltpred covers the core prediction and fitting APIs: role-based and arbitrary-
pedigree (`kinship_from_pedigree`) covariance construction, an installed
population-trio register driver (`estimate_liabilities`), threshold/age/CIP
conversions plus CIP estimation from follow-up records (Kaplan-Meier and
Aalen-Johansen, `ltpred.cip`), Gibbs, PA and nuclear-family quadrature, single-
and multi-trait `estimate_liability`, simulation, and **model fitting** —
heritability (`fit_heritability`), variance components A + C + M
(`fit_variance_components`), optional common-threshold pairwise likelihood
(`fit_pairwise`), joint multi-trait h²/rg/re and optional sibship/couple
covariance (`fit_pairwise_multi`), approximate iid-family cluster percentile intervals
(`bootstrap_fit`), liability-scale transformations (`ltpred.liability_scale`),
and tetrachoric-correlation diagnostics
(`ltpred.tetrachoric`) for liability correlations straight from 2x2
case/control tables.

The fitted sibship (`C`) and couple (`M`) components feed back into prediction
through `c2`/`m2` on the single-trait role/object and array estimators
(`estimate_liability` with scalar `h2` included); on an arbitrary pedigree,
pass the aligned `C` and `M` kernels too, since `A` alone cannot tell a
full-sib pair from parent--offspring, or mates from strangers. The
arbitrary-kinship estimator also accepts caller-supplied kernels via
`c2`/`c_kernel` and `m2`/`m_kernel`, and (PA only) `use_mixture=True` with
per-member `K_i`/`K_pop` for the PA-FGRS censored-control mixture; Gibbs still
has no mixture implementation. The fitters remain role-based and require
independent, non-overlapping families, and the multi-trait scoring route rejects
`c2`/`m2` rather than silently ignoring them.

Unsupported research code lives in the checkout-only **`research/` package**
(importable as `research.<module>` from a checkout; not installed with the
wheel, not part of the public API, interfaces may change): the
HE genetic-correlation, onset-age-decay, common-factor and genetic-nurture fits,
the MCEM variance-component fit and the parametric-bootstrap significance tests
in `research.advanced_fitting`, plus the sex-limited and genetic-nurture
covariance constructors in `research.covariance_extensions`. It has its own
test suite (`research/tests`, run by CI) and five benchmark scripts import it;
the models are documented in [docs/research.md](docs/research.md).

Not included: an igraph-style pedigree-object interface, plotting utilities, and
the xgboost heritability helpers from LTFHPlus; and ltpred does not build LD or run
the GWAS itself (it produces the family-history liability phenotype you feed to
one).

## Documentation

[Quickstart](docs/quickstart.md), [tutorial](docs/tutorial.md) (every block is
executed in `tests/test_tutorial.py`), and the
[vignette](https://bvilhjal.github.io/ltpred/vignette/). The generators used
there ship in the package. Model and evidence:
[algorithm.md](docs/algorithm.md),
[methods note](report/ltpred_methods.pdf),
[benchmarks](benchmarks/RESULTS.md).

## Development

The code and documentation were written together with AI. Developed with the
support of [SMARTbiomed](https://smartbiomed.dk/).

## License

[MIT](LICENSE). Please cite the method(s) used and this repository; see
[CITATION.cff](CITATION.cff) for curated method and related-work references.
