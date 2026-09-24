# Releasing ltpred

ltpred publishes to PyPI from a tagged GitHub Release via
[`.github/workflows/publish.yml`](https://github.com/bvilhjal/ltpred/blob/main/.github/workflows/publish.yml),
using PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). No API
token is stored in the repository.

## One-time setup (project owner)

Register the trusted publisher for the `ltpred` project at
<https://pypi.org/manage/account/publishing/> (or, before the first upload, as a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)):

| Field | Value |
|---|---|
| PyPI Project Name | `ltpred` |
| Owner | `bvilhjal` |
| Repository name | `ltpred` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Optionally add the same as a trusted publisher on
[TestPyPI](https://test.pypi.org/) first to rehearse.

## Cutting a release

1. Bump `__version__` in `ltpred/__init__.py` (the sole source of truth for the
   distribution; `pyproject.toml` reads it dynamically) and move the
   `## Unreleased` section of `CHANGELOG.md` under the new `## X.Y.Z — <date>`
   heading. Two files carry the version independently of that attribute and
   drift silently when missed: `CITATION.cff` (`version` and `date-released`)
   and `report/ltpred_methods.tex`.
2. Confirm CI is green on `main`: `test` (3.9–3.13 and 3.14t, plus macOS 3.12),
   `research-tests`, `lint` (`ruff`), `test-no-numba`, `oldest-deps` (3.9 at the
   minimum NumPy/SciPy), `examples`, `build` (wheel and sdist) and `docs` (strict
   build plus `scripts/check_evidence.py`). The `docs` job also deploys the site
   on every push to `main`, so the published documentation tracks `main`, not
   the latest release.
3. Rebuild the tracked methods PDF and check its release-defining claims against
   the committed CSVs. If a benchmark artifact must change, first commit its
   source, then regenerate CSV/PNG outputs through `benchmarks/run_benchmark.py`
   with each retained output named by `--artifact`; commit the resulting JSONL
   provenance row with the artifact. The standalone
   [time/memory JSON comparison](https://github.com/bvilhjal/ltpred/blob/main/benchmarks/README.md#time-and-memory-between-versions)
   instead retains its aggregate measurements, run log and before/after source
   provenance in a dated capsule. Keep the measured version explicit when a
   later documentation patch incorporates its results. Never backfill a source
   commit after a run.
   ```bash
   cd report && tectonic -X compile ltpred_methods.tex && cd ..
   python -m pip install "pypdf>=4"
   python scripts/check_evidence.py
   ```
4. Locally, sanity-check the package artifacts with modern tooling:
   ```bash
   python -m pip install --upgrade build "twine>=6.1" "packaging>=24.2"
   python -m build
   python -m twine check dist/*      # must PASS for the wheel and the sdist
   ```
   Older `packaging` (< 24.2) reports the PEP 639 `License-Expression` /
   `License-File` fields as malformed even though the metadata is valid; upgrade
   `packaging` rather than changing the license metadata.
5. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Create a GitHub Release for that tag. Publishing the release triggers
   `publish.yml`, which rebuilds, re-runs `twine check`, and uploads to PyPI
   through the trusted publisher.

`workflow_dispatch` runs the build-and-check jobs without publishing, so the
release path can be rehearsed at any time.
