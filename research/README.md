# research/

Unsupported research code, split out of the lean `ltpred` core. Nothing here
ships in the wheel or belongs to the public API, and interfaces may change
without notice. Import it as `research.<module>` from a repository checkout
(repository root on `sys.path`, for example by running from the repo root).

It is not dead code: five benchmark scripts import it
(`bench_ascertainment.py`, `bench_aod_decay.py`,
`bench_covariance_extensions.py`, `bench_genetic_correlation.py`,
`bench_inference_calibration.py`, plus `benchmarks/_common.py`), and its own
suite in `tests/` runs in CI's `research-tests` job. Locally:

```bash
pytest -q research/tests
```

The core `pytest` run collects only `tests/` (see `testpaths` in
`pyproject.toml`). The models are documented on the documentation site's
[research extensions](../docs/research.md) page.

Layout:

- `advanced_fitting.py` — inferential fitting machinery moved out of
  `ltpred.fit`: cross-trait genetic-correlation fits, the onset-age decay EM,
  the common-factor model, the Monte-Carlo EM variance-component route,
  parametric-bootstrap significance tests, and the closed-form nurture fit.
- `covariance_extensions.py` — covariance constructions not wired into any
  estimator: sex-limited architecture and direct/indirect (genetic-nurture)
  effects.
- `tests/` — tests for the above.

Graduation rule: a capability moves into `ltpred` proper only when it is
wired into the core estimation path and benchmarked.
