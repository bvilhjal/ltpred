# ltpred

**LTpred** estimates posterior mean genetic liability from disease records,
age of onset and family relationships. It implements LT-FH, LT-FH++, ADuLT and
PA-FGRS, with deterministic Pearson–Aitken (PA), Gibbs sampling and restricted
nuclear-family quadrature. It also fits heritability and genetic/environmental
covariances from independent family data.

The score is a family-history analogue of a predicted breeding value under the
liability-threshold model. It is not a SNP polygenic score or an absolute disease
risk. Include the proband's diagnosis for a GWAS phenotype; leave it uninformative
for family-history prediction.

**[Getting started](docs/quickstart.md)** · **[Worked tutorial](docs/tutorial.md)** ·
**[Documentation site](https://bvilhjal.github.io/ltpred/)**

## Install and run

Install from a source checkout. NumPy and SciPy are required; `[fast]` adds the
recommended Numba acceleration.

```bash
git clone https://github.com/bvilhjal/ltpred.git
cd ltpred
pip install -e ".[fast]"
```

```python
from ltpred import simulate_under_LTM_single, estimate_liability

sim = simulate_under_LTM_single(
    fam_vec=["m", "f", "s1"], h2=0.5, pop_prev=0.05, n_sim=2000,
    use_age=False, seed=1,
)
res = estimate_liability(sim.families, h2=0.5)
res.genetic  # posterior mean genetic liability, aligned to res.pids
```

This is classic LT-FH: `use_age=False` uses one prevalence threshold.
LT-FH++ uses personalised age/sex/cohort cumulative-incidence bounds with relatives;
ADuLT uses those bounds for the proband alone. Supply disease-specific
liability-scale `h2`. For prediction, retain role `o` and its personal `pid`
with `(-inf, inf)` bounds. The [tutorial](docs/tutorial.md) covers calendar-censored
prediction through the population-register driver.

## Documentation

Each page has one job:

- [Getting started](docs/quickstart.md): analysis choice, installation and input format.
- [Tutorial](docs/tutorial.md): an executable simulated-register workflow.
- [Data preparation](docs/data-preparation.md) and [CIP estimation](docs/cip-estimation.md):
  roles, pedigrees, follow-up records and thresholds.
- [Scoring](docs/estimation.md) and [Fitting](docs/inference.md): method choice,
  result interpretation, scaling and sampling contracts.
- [Model](docs/algorithm.md), [API](docs/api.md) and [assumptions](docs/assumptions.md):
  definitions and reference details.

The [methods PDF](report/ltpred_methods.pdf) supplies the detailed derivation.
[Numerical checks](docs/validation.md) explains reproducible simulation examples;
[benchmark results](benchmarks/RESULTS.md) owns dated accuracy, runtime and memory
measurements. Retained benchmark artifacts substantiate those measurements.
`python scripts/check_evidence.py` checks the release-defining claims against them.

## Scope and development

Scoring supports role families, arbitrary pedigree kinship and a population-trio
register driver. Multi-trait scoring uses Gibbs with A+E covariance; fitting C/M
components does not extend that scorer. Fitters require independent,
non-overlapping families with common case/control thresholds and an explicit
population or positive-IPW sampling design. LTpred constructs phenotypes; it does
not build LD or run the downstream GWAS.

Unsupported prototypes remain in the checkout-only [research package](research/README.md).
They are not installed with the wheel. See the [changelog](CHANGELOG.md),
[benchmark instructions](benchmarks/README.md), [roadmap](docs/ROADMAP.md) and
[release instructions](docs/RELEASING.md) for development.

The code and documentation were written together with AI, with support from
[SMARTbiomed](https://smartbiomed.dk/).

## License and citation

[MIT](LICENSE). Cite the methods used and this repository;
[CITATION.cff](CITATION.cff) collects the method references.
