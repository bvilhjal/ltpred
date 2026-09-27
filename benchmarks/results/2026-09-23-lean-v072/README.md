# v0.7.2 lean-cleanup development capsule (2026-09-23)

Promoted from the gitignored `tmp/lean-review/` working directory, where it
was the only measurement taken while preparing v0.7.2; the build outputs
(`site/`, `dist/`, `installed/`) were left behind as reproducible artifacts.

**Scope: allocation, not runtime.** This capsule measures the chunked kernels'
working set on synthetic single-role probes — chunked PA at n = 1,000,000
(peak allocation 30.59 → 15.49 MiB, median 0.0487 → 0.0469 s) and chunked
Gibbs at n = 50,000 (2.85 → 1.66 MiB). Its own `results.json` states the
limit plainly: *"Similar runtime in these short local probes; no general
speedup claim."* It does **not** back the CIP-validation and seed-bookkeeping
wall-clock figures the v0.7.2 changelog originally quoted; those numbers had
no committed artifact and have been restated qualitatively there.

Contents: `measure.py` (the probe driver), `change.patch` (the candidate
delta, SHA-256 recorded in `results.json`; applied to `8926f25` it reproduces
the code of `d57e1f9`), and before/after time and memory JSONs. `baseline.npz`
holds the probe outputs; the candidate's were byte-identical (SHA-256
`6b5db017…`), so only one copy is kept. `results.json` records the test counts
(1,149 → 1,173 passing). Baseline commit `8926f25` (v0.7.1). One thread, AC
power, five warm repeats; memory measured in separate processes. The build,
install, docs and test logs and the report renders were dropped as
reproducible tool output.
