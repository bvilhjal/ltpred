# Quickstart

One complete analysis — from a status/age table to a genetic-liability score you
can feed a GWAS — on a single page. Each step links to the deeper reference.

## Install

From a checkout (not yet on PyPI):

```bash
pip install -e ".[fast]"     # [fast] adds the Numba JIT — recommended for real runs
```

`numpy` and `scipy` are the only hard dependencies; `numba` (the `[fast]` extra) is
optional but strongly recommended at scale.

## 1. Describe your data

One row per **(proband, relative)**, grouped into families by `fam_id`. Each person
has a role (relative to the proband), a case/control `status`, and an `age` (age of
onset for cases, age at last follow-up for controls):

```python
import numpy as np

fam_id = ["F1", "F1", "F1", "F2", "F2", "F2"]
role   = ["o",  "m",  "f",  "o",  "m",  "s1"]     # o = proband; see the role grammar
status = np.array([1,   0,    1,    0,    1,    0])   # 1 = case
age    = np.array([48,  71,   55,   36,   62,   40])  # onset (cases) / follow-up (controls)
```

See [data preparation](data-preparation.md) for the [role grammar](data-preparation.md#role-grammar)
and for [arbitrary pedigrees](data-preparation.md#beyond-the-role-grammar-arbitrary-pedigrees).

## 2. Turn status + age into liability bounds

Pick the threshold builder that matches your model (classic LT-FH, ADuLT/LT-FH++, or
PA-FGRS). Here, the age-of-onset (LT-FH++) encoding:

```python
from ltpred import age_thresholds
lower, upper = age_thresholds(status, age, pop_prev=0.05)   # pop_prev = lifetime prevalence
```

For **real register data**, use [`thresholds_from_cip`](data-preparation.md#a-real-register-data-recipe)
with your population's cumulative-incidence curve instead of the built-in logistic one.

## 3. Build the families

```python
from ltpred import families_from_columns
families = families_from_columns(fam_id=fam_id, role=role, lower=lower, upper=upper)
```

## 4. Estimate the genetic liability

```python
from ltpred import estimate_liability
res = estimate_liability(families, h2=0.5)        # PA-FGRS by default; fast + deterministic
score = res.genetic                                # == res.est["genetic"]
```

`h2` is the **liability-scale** heritability. Don't have one? Estimate it from the
families with [`fit_heritability`](inference.md#fitting-heritability-from-the-family-data), or
convert an observed-scale value — see
[getting `h²`](data-preparation.md#getting-heritability-on-the-liability-scale).

## 5. Use the score

`score` is a continuous per-proband phenotype on the liability scale — use it in a
linear-regression GWAS in place of the 0/1 label (that's where the power gain comes
from), or as a standalone family-based risk score:

```python
# after residualising for covariates (sex, cohort, PCs, batch) — see estimation.md
y = (score - score.mean()) / score.std()
```

## Where to go next

| you want to… | see |
|---|---|
| understand roles, pedigrees, CIPs, real-data prep | [Data preparation](data-preparation.md) |
| choose Gibbs vs PA, scale to biobank size, multi-trait, GWAS export | [Estimation](estimation.md) |
| fit `h²`, variance components (A/C/M), `r_g`, factor models, tests | [Inference](inference.md) |
| know the modelling assumptions & the real-data checklist | [Assumptions & checklist](assumptions.md) |
| look up a function signature | [API reference](api.md) |
| the model & estimator maths | [algorithm.md](algorithm.md) |
