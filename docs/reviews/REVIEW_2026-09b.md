# Independent review of ltpred v0.5.1 (2026-09-02)

A read-only audit of what changed since the v0.4.2 review, plus the two
installed modules that neither previous review examined. This review covers
the tree at commit `a6b18be` (v0.5.1). It follows, and is independent of, the
v0.3.4 audit in `REVIEW_2026-08.md` and the v0.4.2 audit in
`REVIEW_2026-09.md`. No package code was changed by the review; the only
edits are this document and its `mkdocs.yml` nav entry.

## 1. Scope and method

The statistical core (`covariance.py`, `thresholds.py`, `pearson_aitken.py`,
`gibbs.py`, `estimate.py`, `fit.py`, `cip.py`, `tetrachoric.py`,
`liability_scale.py`, `simulate.py`) was audited twice in the last month
against primary sources and closed forms, and its verdicts stand. This review
does not repeat that work. It concentrates on the package delta since commit
`7806e55` (nine files, +488/-57 lines) and on the two installed modules the
earlier reviews did not list: `pipeline.py`, the public register driver added
in 0.5.0, and `pedigree.py`, whose extraction gained ancestral-closure marking
and shortest-degree relaxation in the same release.

Three stages. First, the project's own gates were run (Table 1). Second, the
delta was read line by line against the 0.5.0 and 0.5.1 changelog entries and
the documentation. Third, every claim that matters was checked independently
by execution: alignment contracts, the new kernel path against the role
grammar, the direction of the driver's calendar censoring, and two suspected
silent-failure modes, each reproduced on a hand-built table (Table 2).
Severity tiers as before: **T1**, correctness or scientific validity;
**T2**, robustness, evidence integrity, or maintainability; **T3**, style,
clarity, or polish.

*Table 1. Baseline checks on the review machine (macOS, Apple silicon; repo
venv: CPython 3.10.20, NumPy 2.2.6, SciPy 1.15.3, Numba 0.67.0; BLAS pinned to
one thread as in CI).*

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q tests` | 618 passed, 61 s |
| Lint | `ruff check .` | clean |
| Evidence guard | `scripts/check_evidence.py` | internally consistent: scaling 392 to 518x; PA stress floor 0.9991; R locks 6.786 +/- 0.079x and 1418 +/- 30x; PGS 0.338 / 0.236 / 0.170 |
| Docs | `mkdocs build --strict` | builds; the red text is the Material for MkDocs 2.0 announcement, not a strict-mode warning |
| Example | `examples/vignette.py` | runs end to end; register example gives GWAS +1.017439, prediction -0.072034, post-index and closure invariance 0.0 |

## 2. Verdicts by area

**Register driver (`pipeline.py`).** Correct where it computes; the gaps are
at its boundaries. The prediction-mode censoring is right in every case
checked: a relative's record after the landmark becomes a control censored at
that relative's own attained age, a pre-landmark diagnosis stays a pinned
case, a pre-landmark exit stays a control at exit age, and a member born at or
after the landmark is uninformative. The proband's own row is made
uninformative under `use="prediction"` and conditioned under `use="gwas"`.
The alignment the driver relies on, that `kinship_from_pedigree` returns the
matrix in the order of the ids it was given so that `target=0` is the
proband, holds: the tabular method iterates in topological order but writes
`A[i, j]` at the original positions (`covariance.py:473-540`). What the driver
does not do is report on the two conditions that most often make a register
analysis wrong without making any single score wrong: parent references that
failed to resolve, and probands who are not at risk at the landmark
(findings T2-1 and T2-2).

**Pedigree extraction (`pedigree.py`).** Correct. Ancestral closure includes
every recorded ancestor of every extracted member, closure nodes included, so
the extracted kinship is the population kinship restricted to the set; the
docstring's claim holds and is under test. Shortest-degree relaxation during
closure can only lower a degree, and no closure node can end below
`max_degree` because the breadth-first pass already explores parent edges.
Sibling edges give half-sibs degree 2 through the shared parent and mates
degree 2 through a shared child, as documented.

**Kinship-route environmental kernels (`covariance.py`).** Correct. With a
sibship kernel over the proband and two full sibs, the kinship route
reproduces `construct_covmat_single(c2=...)` to 0.0 in every entry (Table 2).
The kernel does not reach the parents, so `Cov(o, m)` stays `h2/2`. Validation
is complete: symmetry to 1e-8 then canonicalised, unit diagonal, positive
semi-definite to -1e-8, `h2 + c2 + m2 <= 1`, and a nonzero proportion without
its kernel is refused rather than guessed from `A`.

**Threshold helpers and diagnostics.** The removal of the `pop_prev = 0.1`
default from the six public helpers is the right change: a package cannot
carry a disease-independent prevalence. `thresholds_from_cip` documents that
`k_pop` defaults to `max(cip_values)` and clips a young control's CIP at
`min_cip = 1e-5`, which is what keeps a disease-free two-year-old
uninformative without an infinite bound. The tetrachoric finiteness gate now
precedes the integer gate, closing T3-6 of the previous review.

**Documentation and changelog.** Accurate against the code for the delta.
Every 0.5.0 entry that names behaviour was found in the code, and the
vignette's register example reproduces the invariances it claims.

## 3. Independent verification

*Table 2. Checks re-derived for this review, independent of the package's
own tests.*

| Check | Result |
|---|---|
| `kinship_from_pedigree` row order | matrix returned in input order; the driver's `target=0` is the proband |
| Kinship-route sibship kernel vs role grammar (o, m, f, s1, s2; `h2 = 0.5`, `c2 = 0.15`) | max abs difference 0.0; `Cov(o, s1) = 0.40 = h2/2 + c2`; `Cov(o, m) = 0.25 = h2/2` |
| Direction of calendar censoring (father disease-free at 35, 45, 55, 65, 75 at the 2020 landmark) | estimate -0.0547, -0.0640, -0.0754, -0.0850, -0.0932: monotone, older disease-free relatives pull the proband down |
| Post-index and closure-only invariance | 0.0 in the test suite and in the vignette example |
| Closure relaxation and exact kinship | covered by `test_degree_is_the_shortest_route_not_the_first_found` and `test_extraction_preserves_full_pedigree_kinship` |
| Silent founder coercion (T2-1) | same trio table with parent ids in a different format: estimate +0.3246 with 2 relatives becomes -0.0220 with 0 relatives; no warning, no count |
| Missing landmark eligibility (T2-2) | proband diagnosed 2015, landmark 2020, `use="prediction"`: scored +0.3591, no warning, no field; proband exited follow-up 2010: same |

## 4. Findings

No T1 findings. Every score the package computes was correct for the inputs
it was given. The T2 findings are about what the register driver does not
tell its caller.

**T2-1. Unresolved parent references become founders silently.**
`pedigree.py:109-112` and `covariance.py:504-508` map any parent value that
is not among `ids` to the founder sentinel `-1`. That is documented in both
docstrings and tested (`test_founder_columns_and_ordering`), and at the
register boundary it is unavoidable: a parent born before the register
started is a founder. But the same rule absorbs an id-format mismatch, a
failed join, or a truncated column, and in the register driver that turns
every family-history score into an own-status-only score with nothing to show
for it except a low `n_relatives`. The reproduction in Table 2 is the whole
failure: 2 relatives to 0, no warning. Recommendation: count non-null parent
references that did not resolve, once per table in `build_parent_graph`
(`ParentGraph.n_unresolved_parents`), and surface the fraction of records
with at least one unresolved non-null parent on `PopulationScores`. Warn when
that fraction is implausible for a register boundary, and refuse when the
resolved fraction is zero with non-null references present, which cannot be a
boundary effect. A cheap companion check is dtype: all ids integers and all
parent references strings is a certain mismatch.

**T2-2. `use="prediction"` scores probands who are not at risk at the
landmark, without saying so.** The only proband-level check is that the
landmark does not precede birth (`pipeline.py:193-199`). A proband already
diagnosed before `index_time`, or already out of follow-up, is scored like any
other and `PopulationScores` carries no field that distinguishes them.
The vignette is honest that the driver "does not construct an eligible
incident-risk cohort" (line 757), so this is a documented boundary. It is
still the wrong place to leave it. The driver already holds
`record_time[proband]` and `index_time`, and including prevalent cases in a
prospective evaluation is exactly the cohort-level leakage the `use=` design
exists to prevent: their family histories are enriched, so discrimination is
inflated. Recommendation: return a per-proband state,
`disease_free_and_followed`, `prevalent_case`, or `exited_before_index`, and
warn when any proband is a prevalent case. Do not raise; a caller may want the
scores for a different purpose.

**T2-3. A supported driver whose payoff evidence is marked stale.**
`docs/api.md` calls `estimate_liabilities` a supported public API. The
evidence that would say what it buys, RESULTS.md section 20 (pedigree payoff)
and section 21 (end-to-end register pipeline), is labelled "Historical/stale
evidence (2026-08-30)" by the project itself, and ROADMAP item 4 schedules the
rerun. The graduation rule's clause for thin orchestration was applied, and
the exact-equivalence and invariance tests it asks for exist. But the module
docstring calls the glue "small but consequential", and calendar censoring and
observation-set choice are consequential. Recommendation: regenerate sections
20 and 21 under the provenance wrapper before the first release (ROADMAP item
7), or until then label the driver in `api.md` and the vignette as supported
with its payoff unquantified.

**T3-1. Name the prediction estimand.** Under `use="prediction"` the
proband's own row is made uninformative (`pipeline.py:264`), so the
estimand is `E[g | relatives observed at the landmark]`, not
`E[g | relatives, proband disease-free at the landmark]`. That matches the
LT-FH prediction and PA-FGRS convention of leaving own status out, and the
docstring says the row is left uninformative. It does not say that the second
estimand exists or how it differs: for a landmark cohort of mixed ages the two
agree on ranking within an age but not on level across ages, since survival
to an older age disease-free is evidence of lower liability. One sentence in
`api.md` and the vignette's landmark section would close it.

## 5. What this review did not do

It did not re-derive the statistical core; two reviews in the preceding month
did, and nothing in the 0.5.x delta touches the PA update, the Gibbs
conditionals, the CIP estimators, or the fitters. It did not re-run the R
lock comparisons, the HAPNEST path, or the benchmark suite; the evidence guard
was run instead, and the register-benchmark staleness in T2-3 is the
project's own label, not a recomputation. It did not read `research/`. The
numerical probes were run on one machine and one stack (Table 1).
