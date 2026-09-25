# Getting started

LTpred estimates posterior mean genetic liability from disease records and family
relationships. It returns a score per proband, not a SNP polygenic score or an
absolute disease-risk probability. This page shows the input format and a complete
six-row example; the [tutorial](tutorial.md) runs a simulated population register.

## Choose the analysis

Table 1 fixes whether the proband's own diagnosis belongs in the score.

**Table 1. Analysis purpose determines the observation set.**

| Purpose | Proband observation | Next step |
|---|---|---|
| I. Family-history prediction | Retain role `o` and its personal `pid`, with `(-inf, inf)` bounds; censor relatives at the prediction landmark | [Prospective tutorial](tutorial.md#step-4-score-prospectively-without-the-probands-own-diagnosis) |
| II. GWAS phenotype | Include the proband's diagnosis | Example below, then [GWAS use](estimation.md#using-the-estimate-in-a-gwas) |
| III. Estimate model parameters | A score is optional; fitting has its own sampling and identification requirements | [Fit h²/covariances](inference.md) or [estimate CIP](cip-estimation.md) |

For either scoring use, supply disease-specific liability-scale `h2` and population
prevalence or cumulative incidence. With relatives, common lifetime bounds give
LT-FH and personalised CIP bounds give LT-FH++; personalised proband-only bounds
give ADuLT. [Scoring](estimation.md#choose-a-scoring-model) covers engines and PA-FGRS.

## Install

PyPI publication is pending. Until then, installation from a source checkout is
the supported path:

```bash
git clone https://github.com/bvilhjal/ltpred.git
cd ltpred
pip install -e ".[fast]"     # [fast] adds the Numba JIT — recommended for real runs
```

`numpy` and `scipy` are the only hard dependencies; `numba` (the `[fast]` extra) is
optional but strongly recommended at scale.

With Numba installed, the first run may print
`OMP: Info #276: omp_set_nested routine deprecated…` to stderr. That notice comes
from the OpenMP runtime Numba links against, not from ltpred, and is harmless.

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

Each person's status and age become an interval on the liability scale
(`lower`, `upper`): a case is pinned at the threshold for their onset age, and a
control lies below the threshold for their current age. Because relatives are
included, this is a family-history (LT-FH++-style) analysis. `age_thresholds`
uses one built-in logistic incidence curve (`mid_point=60`, `slope=1/8`) and is
for demonstration only:

```python
from ltpred import age_thresholds
lower, upper = age_thresholds(status, age, pop_prev=0.05)   # pop_prev = lifetime prevalence
```

!!! warning "This builder is a demonstration, not the production path"

    `age_thresholds` uses one built-in logistic incidence curve shared by every
    person, so it cannot represent the sex- and birth-cohort-specific risk that
    LT-FH++ is built around. It is here to make step 2 runnable on six typed
    rows. For **real register data**, use
    [`thresholds_from_cip`](data-preparation.md#a-real-register-data-recipe) with
    your population's estimated cumulative-incidence curve — see
    [CIP estimation](cip-estimation.md) for how to get that curve, and the
    [tutorial](tutorial.md) for CIP estimation and the register driver on
    simulated data (it feeds the generating curve back in; see its last
    section).

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
fitters assume independent, non-overlapping families under a declared sampling
design — unascertained (`sampling="population"`), or selected on observed status
with known inclusion probabilities (`sampling="ipw"` with weights); see
[Inference](inference.md#ascertained-samples).

## 5. Inspect and align the score

`score` has one value per family (proband), aligned to `res.pids`, not to the
input row order:

```python
for proband, value in zip(res.pids, score):
    print(proband, round(float(value), 3))
```

```text
P1 1.357
P2 0.462
```

For a GWAS, explicitly join `res.pids` to the genotyped proband IDs, residualize
the phenotype for sex, cohort, ancestry PCs, batch and other design covariates,
and validate calibration under the actual sampling design; see
[Using the estimate in a GWAS](estimation.md#using-the-estimate-in-a-gwas).
The score is not an absolute disease-risk probability.

## Population register input

For an `ids`/`father`/`mother` table, use the supported `estimate_liabilities`
driver instead of assigning role labels yourself. Supply aligned `status` and
`age` columns, `probands`, liability-scale `h2`, and either one empirical CIP
or `strata` with `cip_by_stratum={label: (ages, values, k_pop)}`.
Choose `use="gwas"` to include own diagnosis; prospective `use="prediction"`
also requires calendar `birth_time` and per-proband `index_time` so relatives'
records are censored at that landmark. The [register recipe](data-preparation.md#beyond-the-role-grammar-arbitrary-pedigrees)
shows a complete call. `result.to_dict()` exports aligned columns;
`result.to_frame()` additionally requires pandas.

## Continue with your data

[Data preparation](data-preparation.md) owns roles, pedigrees and threshold inputs;
[Scoring](estimation.md) owns engines, output columns and scaling;
[Fitting](inference.md) owns covariance estimation and sampling assumptions.
Read the [analysis checklist](assumptions.md) before applying the example to a cohort.
