# Efficient inference pilot, 5 September 2026

This capsule records the v0.6.0 implementation on a modified checkout based on
`8856ba869ac7c94ad8dd1d3004e9fbbadb38f7ab`. It is a source-hashed development
pilot, not a clean-commit run of the historical benchmark campaign. The full
measured Python sources, launcher, input arrays, outputs and process logs are
included; generated Numba caches are omitted.

The composite source digest is
`67002dd285139547df9f0e3d0c292bc27760679a68b1fa847f21168f572f484d`.
`results.json` records every file hash and confirms source stability throughout
the run. Its absolute output paths name the original run directory; this
capsule is a byte-preserving copy of the retained files.

Run the current implementation from the repository root with:

```bash
.venv/bin/python benchmarks/bench_efficient_inference.py --output /tmp/ltpred-efficiency-rerun
```

To reproduce the measured source independently of later edits, use the same
interpreter with `source_snapshot/benchmarks/bench_efficient_inference.py`.
NumPy, SciPy and Numba are required; their measured versions are in the JSON.
The driver checks AC power and Low Power Mode on macOS. Each arm gets a fresh
process and empty Numba cache. First-call time includes compilation reached by
that call; warm time excludes imports/setup. Peak RSS includes the whole child
process and all its repetitions. Numba's single-thread setting is verified;
BLAS environment limits and build configuration are recorded, but runtime BLAS
thread counts were unavailable.

The selected driver is compared with the full-covariance construction using the
same current PA engine. Means and variances match exactly; peak RSS is slightly
higher with selected relationships. ADuLT is compared with the public 2x2 PA
covariance API on identical bounds. The scalar means match exactly and variances
agree within 1.12e-16.

Three identical fitting datasets compare pairwise likelihood with the moment
fitter's default 1500/500/5 controls. Their point estimates differ. These timings
do not establish convergence, equal accuracy, interval coverage or statistical
efficiency. The one-family integration check also does not establish uniform
quadrature error. See the methods report for the numerical results and scope.
