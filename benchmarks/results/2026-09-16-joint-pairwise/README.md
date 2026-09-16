# Joint pairwise inference: development evidence, 16 September 2026

The opt-in `fit_pairwise_multi` implementation was checked in 1,800 independent-cohort fits: 200 replicates per scenario, each generated from 3,000 nuclear families (`o, m, f, s1, s2`). The same seed indexes pair scenarios for comparison; cohorts within a scenario are independent. Trait prevalences are 0.10 and 0.20. There were **zero failed fits**, with every attempt retained.

This is a development checkout based on `6a3c574`, with package version still 0.6.2; it is not a published release. The main and null campaigns used identical package source hashes. Each directory retains the source zip, individual source hashes, versions, command, configuration, all replicate estimates/SEs and summaries. The second campaign only extends the benchmark script with null scenarios. Both manifests confirm unchanged source during execution. The snapshots and current package hashes were checked after both runs.

The subsequent v0.7.0 commit changes only the version string in the archived
package sources; the measured numerical implementation is unchanged. The
manifests and source archives retain the original development version.

**Table 1. Bias and conditional 95% interval coverage for the two correlations.** Bias includes all finite point estimates. Parentheses give Monte Carlo SE of the bias. Coverage uses only fits with available interior SEs; the last column shows this denominator out of 200.

| Scenario | rg bias (MC SE) | re bias (MC SE) | rg coverage | re coverage | Interior fits |
|---|---:|---:|---:|---:|---:|
| positive | -0.0026 (0.0043) | -0.0006 (0.0030) | 0.935 | 0.945 | 200/200 |
| negative_g | -0.0009 (0.0047) | +0.0018 (0.0035) | 0.960 | 0.955 | 200/200 |
| negative_e | -0.0027 (0.0048) | -0.0002 (0.0035) | 0.940 | 0.925 | 200/200 |
| null_g | -0.0033 (0.0048) | +0.0019 (0.0034) | 0.935 | 0.920 | 200/200 |
| null_e | -0.0024 (0.0045) | +0.0000 (0.0032) | 0.935 | 0.940 | 200/200 |
| shared | -0.0035 (0.0059) | -0.0105 (0.0097) | 0.956 | 0.972 | 180/200 |
| missing | -0.0004 (0.0074) | -0.0169 (0.0123) | 0.968 | 0.981 | 156/200 |
| ipw | +0.0061 (0.0079) | -0.0026 (0.0055) | 0.950 | 0.945 | 200/200 |
| omit_shared | -0.0064 (0.0039) | +0.1831 (0.0039) | 0.945 | 0.080 | 200/200 |

Across correctly specified scenarios, the absolute heritability bias was at most 0.0032. Full per-trait values, empirical SDs, SE/SD ratios, bias/coverage Monte Carlo errors and all denominators are in the summary files.

The shared-environment model has genetic variances (0.35, 0.40), sibship variances (0.15, 0.15), couple variances (0.10, 0.10), genetic correlation 0.5, sibship correlation 0.5, couple correlation 0.2 and residual correlation -0.35. `missing` independently hides 20% of phenotype coordinates under that same model. `ipw` uses the positive A+E model, retaining all trait-1 proband cases and 20% of controls and supplying known reciprocal inclusion probabilities. It therefore fits fewer than 3,000 retained families per replicate.

`omit_shared` deliberately fits A+E to A+C+M+E data. Its heritability biases are +0.1079 and +0.1051. Its residual-correlation comparison targets the generating within-person E correlation, not total non-genetic covariance. The fitted mean residual correlation is -0.1669 against -0.35; the correctly specified A+C+M fit gives -0.3605. This diagnoses misspecification, not a failure of a correctly specified A+E fit.

Boundary fits occurred in 20/200 shared-environment replicates and 44/200 missing-data replicates. They retain point estimates but withhold normal SEs. Thus the coverage column does not establish boundary coverage. With 200 replicates, coverage itself has visible Monte Carlo uncertainty (typically 0.01–0.02); the observed 0.920–0.981 residual-correlation range is not a universal calibration guarantee. Rare traits, informative missingness, unknown ascertainment probabilities, overlapping pedigrees, personalised thresholds and other family structures remain outside this evidence.

The probability tests use independent conditional-normal integration and exact zero-threshold arcsine tables, including signed correlations and PSD boundaries. The statistical simulator constructs its nuclear-family relationship kernels independently of the fitter. Timings in replicate records are explicitly uncontrolled and must not be used for performance rankings.

Artifacts: [main summary](summary.json), [main replicates](replicates.json), [main manifest](manifest.json), [main source](source.zip); [null summary](../2026-09-16-joint-pairwise-nulls/summary.json), [null replicates](../2026-09-16-joint-pairwise-nulls/replicates.json), [null manifest](../2026-09-16-joint-pairwise-nulls/manifest.json).
