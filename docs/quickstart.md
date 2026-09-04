# Quickstart

A runnable tour of the estimator — from a small status/age table to one
genetic-liability score per proband — on a single page. This is **use II**
of the [vignette](vignette.md) (a GWAS phenotype, own status in). The data
below are tutorial-only, not a production analysis. Skip steps the vignette
says your use does not need. Each step links to the deeper reference.

## Install

PyPI publication is pending. Until then, installation from a source checkout is
the supported path:

```bash
pip install -e ".[fast]"     # [fast] adds the Numba JIT — recommended for real runs
```

`numpy` and `scipy` are the only hard dependencies; `numba` (the `[fast]` extra) is
optional but strongly recommended at scale.

## 1. Describe your data

One row per **observed person**, grouped by the target proband's `fam_id`. Each
person has a role defined relative to that proband, a case/control `status`, and an
`age` (age of onset for cases, age at last follow-up for controls):

```python
import numpy as np

fam_id = ["F1", "F1", "F1", "F2", "F2", "F2"]
pid    = ["P1", "M1", "D1", "P2", "M2", "S2"]
role   = ["o",  "m",  "f",  "o",  "m",  "s1"]     # o = proband; see the role grammar
status = np.array([1,   0,    1,    0,    1,    0])   # 1 = case
age    = np.array([48,  71,   55,   36,   62,   40])  # onset (cases) / follow-up (controls)
```

See [data preparation](data-preparation.md) for the [role grammar](data-preparation.md#role-grammar)
and for [arbitrary pedigrees](data-preparation.md#beyond-the-role-grammar-arbitrary-pedigrees).

## 2. Turn status + age into liability bounds

Threshold builders set the observation encoding; family rows distinguish
LT-FH++ from ADuLT. This example includes relatives, so the onset-pinned bounds
form a family-history analysis. The logistic builder is tutorial-only:

```python
from ltpred import age_thresholds
lower, upper = age_thresholds(status, age, pop_prev=0.05)   # pop_prev = lifetime prevalence
```

For **real register data**, use [`thresholds_from_cip`](data-preparation.md#a-real-register-data-recipe)
with your population's cumulative-incidence curve instead of the built-in logistic one.

## 3. Build the families

```python
from ltpred import families_from_columns
families = families_from_columns(
    fam_id=fam_id, pid=pid, role=role, lower=lower, upper=upper
)
```

## 4. Estimate the genetic liability

```python
from ltpred import estimate_liability
res = estimate_liability(families, h2=0.5)        # PA is the single-trait default
score = res.genetic                                # == res.est["genetic"]
```

For full LT-FH++, replace the logistic helper with age-, birth-year- and
sex-stratified empirical CIPs. To run ADuLT instead, build the same personalised
bounds for role `o` only and omit every relative row.

`h2` is the **liability-scale** heritability. Prefer an external estimate, or
convert an observed-scale value — see
[getting `h²`](data-preparation.md#getting-heritability-on-the-liability-scale).
Do **not** fit `h²` from this two-family toy example. The built-in family-data
fitter assumes independent, non-overlapping families under a declared sampling
design — unascertained (`sampling="population"`), or selected on observed status
with known inclusion probabilities (`sampling="ipw"` with weights); see
[Inference](inference.md#ascertained-samples).

## 5. Inspect and align the score

`score` is aligned to `res.pids`, not to the input row order:

```python
list(zip(res.pids, score))
```

For a GWAS, explicitly join `res.pids` to the genotyped proband IDs, residualize
the phenotype for sex, cohort, ancestry PCs, batch and other design covariates,
and validate calibration under the actual sampling design; see
[Using the estimate in a GWAS](estimation.md#using-the-estimate-in-a-gwas).
The score is not an absolute disease-risk probability.

## Where to go next

| you want to… | see |
|---|---|
| how to run the pipeline (three uses, then `h²`, pedigree, CIP, family history) | [Vignette](vignette.md) |
| understand roles, pedigrees, CIPs, real-data prep | [Data preparation](data-preparation.md) |
| choose Gibbs vs PA, scale to biobank size, multi-trait, GWAS export | [Estimation](estimation.md) |
| fit `h²` or A/C/M from population-sampled or reweighted selected families | [Inference](inference.md) |
| know the modelling assumptions & the real-data checklist | [Assumptions & checklist](assumptions.md) |
| look up a function signature | [API reference](api.md) |
| the model & estimator maths | [algorithm.md](algorithm.md) |
