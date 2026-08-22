# research/

Unsupported research code that was split out of the lean `ltpred` core
package. Nothing here is shipped in the wheel; import it as
`research.<module>` from a repository checkout (repo root on `sys.path`,
e.g. run from the repo root).

**Status: dormant and unmaintained.** This code is kept for reference but is
no longer exercised in CI or by the test suite (`testpaths` in
`pyproject.toml` covers only `tests/`). It may drift out of sync with the
core package at any time.

Layout:

- `advanced_fitting.py` — inferential fitting machinery moved out of
  `ltpred.fit`: cross-trait genetic-correlation fits, the onset-age decay EM,
  the common-factor model, the Monte-Carlo EM variance-component route,
  parametric-bootstrap significance tests, and the closed-form nurture fit.
- `covariance_extensions.py` — covariance constructions not wired into any
  estimator: sex-limited architecture and direct/indirect (genetic-nurture)
  effects.
- `pipeline.py` — the end-to-end register pipeline (trio records -> pedigree
  -> CIP thresholds -> kinship-estimated scores).
- `tests/` — tests for the above, moved from `tests/`. They are **not**
  collected: `testpaths` in `pyproject.toml` covers only `tests/`, and CI
  does not run this directory.

Graduation rule: a capability moves into `ltpred` proper only when it is
wired into the core estimation path and benchmarked.
