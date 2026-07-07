# ltpred benchmark results

Summary of the ltpred benchmark suite, comparing the two fitting methods — the
**Gibbs sampler** (LT-FH++) and the deterministic **Pearson–Aitken** estimator
(PA-FGRS) — on simulated data where the true genetic liability is known.

- **Generated:** 2026-07-07, numpy 2.2.6 / scipy 1.15 / numba 0.66, 10 cores.
- **Reproduce:** `OMP_NUM_THREADS=10 python benchmarks/<script>.py` (see
  [`README.md`](README.md) for what each measures).
- **Caveat:** these are *stochastic* benchmarks — each number is one Monte-Carlo
  draw, so re-running shifts values by sampling noise. The conclusions are stable.

## Headline findings

- **PA-FGRS reproduces the Gibbs LT-FH++ posterior mean.** Across all 27
  accuracy cells the two estimates correlate **≥ 0.997** (usually ≥ 0.999), and
  in the genotype GWAS they give the **same** effective sample size (1.52×).
- **PA-FGRS is 100–360× faster** — a deterministic sweep with no MCMC —
  processing **~150 000 families/second** vs ~570/s for the Gibbs sampler, at
  identical accuracy.
- **Both recover 1.2–2.8× the effective sample size of a raw case/control
  label.** The gain grows with heritability, with *lower* prevalence, and with
  more informative relatives (siblings, extended pedigrees).
- **The power gain carries into a genotype GWAS with no inflation:** LT-FH++ and
  PA-FGRS both reach **1.52× effective N** over case/control at λ_GC ≈ 1.0; the
  oracle (true genetic liability) ceiling is 9.6×.
- **Age-of-onset (the liability→onset map) adds a further ~1–10%**, growing with
  prevalence; PA and Gibbs exploit it identically.

---

## 1. Accuracy — Gibbs vs PA-FGRS (`bench_accuracy.py`)

corr(estimate, true g) at **h²=0.5**, 1500 families, and the effective-N gain
over the raw case/control label. Gibbs and PA are indistinguishable; PA is
~250–350× faster per cell.

| Family | Prev | corr Gibbs | corr PA | eff-N gain | Gibbs≈PA |
|---|---:|---:|---:|---:|---:|
| parents | 0.05 | 0.394 | 0.394 | 1.30× | 0.9997 |
| parents+2 sibs | 0.05 | 0.454 | 0.454 | 1.88× | 0.9997 |
| extended (7 rel) | 0.05 | 0.374 | 0.374 | 2.16× | 0.9997 |
| parents+2 sibs | 0.01 | 0.296 | 0.297 | 2.83× | 0.9990 |
| parents+2 sibs | 0.20 | 0.601 | 0.601 | 1.54× | 0.9999 |

Accuracy rises with heritability and prevalence; the *relative* gain over
case/control is largest at low prevalence and with siblings in the pedigree
(rarer cases carry more family-history information). Minimum Gibbs–PA agreement
across the whole grid was 0.997.

## 2. Runtime scaling (`bench_scaling.py`)

Wall time, h²=0.5, K=0.05, Numba on 10 cores.

| #families (trios) | Gibbs | PA-FGRS | speed-up |
|---:|---:|---:|---:|
| 1 000 | 1.9 s | 0.007 s | 270× |
| 2 000 | 3.5 s | 0.012 s | 280× |
| 8 000 | 13.9 s | 0.051 s | 273× |

Both scale linearly in the number of families; PA sustains ~150 000 families/s
against ~570/s for Gibbs. Growing the family from 2 to 10 relatives raises the
Gibbs cost 2.7 s → 9.3 s (n=2000) while PA stays under 0.04 s — the speed-up
holds (220–360×) across family sizes.

## 3. Age-of-onset information (`bench_age_onset.py`)

Pinning cases at their onset threshold (ADuLT/LT-FH++) vs plain case/control,
both fit with PA (Gibbs shown to agree). corr(estimate, true g), 3000 families,
8-relative pedigree.

| h² | Prev | case/control | age-of-onset (PA) | Gibbs | gain |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 0.428 | 0.433 | 0.433 | 1.03× |
| 0.5 | 0.30 | 0.643 | 0.673 | 0.673 | 1.10× |
| 0.8 | 0.30 | 0.751 | 0.784 | 0.783 | 1.09× |

Onset information helps most when cases are common (more relatives contribute an
onset age); at low prevalence the gain is small because case relatives are rare.
PA and Gibbs use the onset map identically (curves overlap).

## 4. GWAS power (`bench_gwas_power.py`)

Linear-regression GWAS on 10 000 probands × 5 000 SNPs (30 causal), h²=0.5,
K=0.05, trios. Phenotypes: case/control, LT-FH++ (Gibbs), PA-FGRS, and the oracle
true genetic liability.

| Phenotype | mean χ² at causal | eff-N vs c/c | power (p<5e-8) | λ_GC |
|---|---:|---:|---:|---:|
| case/control | 35.4 | 1.00× | 36.7% | 0.99 |
| **LT-FH++ (Gibbs)** | 53.8 | **1.52×** | 43.3% | 1.01 |
| **PA-FGRS** | 53.8 | **1.52×** | 43.3% | 1.02 |
| oracle (true g) | 341.5 | 9.64× | 76.7% | 1.04 |

Both family-based estimators lift the mean association χ² at causal SNPs by ~52%
— a real effective-sample-size gain — and raise detection power, while staying
calibrated at null SNPs (λ_GC ≈ 1). LT-FH++ and PA-FGRS are interchangeable in
power; the oracle marks the ceiling if the genetic liability were known exactly.
For real-LD genotypes, rerun with `--plink` on a HAPNEST fileset
([`hapnest/README.md`](hapnest/README.md)).

## Bottom line

PA-FGRS is a drop-in, deterministic replacement for the LT-FH++ Gibbs sampler:
same accuracy and GWAS power to three decimals, two-to-three orders of magnitude
faster. Use `method="pearson-aitken"` for large biobank-scale runs and
`method="gibbs"` when you want posterior draws or a sampling-based check.
