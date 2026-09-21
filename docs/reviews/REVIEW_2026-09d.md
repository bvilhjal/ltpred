---
title: Independent review (v0.7.0, documentation)
description: Documentation, tutorial and simulated-data audit of ltpred v0.7.0.
---

# Documentation and onboarding review of ltpred v0.7.0 (2026-09-21)

A read-only audit of the **user-facing surface**: the documentation set, the
runnable examples, and the simulated data a new user can obtain. This review
covers the tree at commit `2722a59` (v0.7.0). It follows, and is independent of,
the four preceding audits — v0.3.4 (`REVIEW_2026-08.md`), v0.4.2
(`REVIEW_2026-09.md`), v0.5.1 (`REVIEW_2026-09b.md`) and v0.6.2
(`REVIEW_2026-09c.md`, at `e9bfa66`), fifteen commits behind this one.

Those four audits re-derived the statistical core against independent oracles.
**This one does not, and deliberately so.** It examines a surface no previous
audit touched: whether a person who installs ltpred can actually learn to use
it. Every claim below is either a file/line citation or a measurement I ran;
nothing is re-derived from the model, and no numerical finding of the earlier
audits is reopened. The audit changed no package code.

Severity tiers follow the chain, re-read for this domain: **T1**, a user cannot
complete a documented task, or a document states something false; **T2**,
significant friction, a broken promise to installed users, or duplication that
will drift; **T3**, clarity and polish.

## 1. Scope and method

Two questions frame the audit. First: *can a reader execute what the
documentation shows?* Second: *can a user obtain simulated data for every
supported route?* The second question matters more than it looks, because
ltpred's whole value proposition is that a user supplies a cohort — and the
only way to learn a pipeline that consumes a cohort is to have one.

Method. I read the complete documentation set (twelve pages, 31,730 words) and
all four examples. I then **ran** every copy-paste path a new user is offered:
the README quickstart, `docs/quickstart.md` steps 1–5, the multi-trait recipe in
`docs/data-preparation.md`, and all four `examples/*.py`. I inventoried the
distribution artifacts to establish what an installed user actually receives,
and I counted the exported API surface against the documentation. Where a
suspicion proved wrong on execution, that is recorded too (§2, §3).

*Table 1. Baseline checks on the review machine (macOS arm64; CPython 3.10.20,
NumPy 1.26.4, SciPy 1.15.3, Numba 0.66.0; BLAS pinned to one thread as in CI).
The repository's own `.venv` could not be used: its SciPy fails to `dlopen`
(`_spropack...so`, "zero-fill section type, but offset field is not zero"), a
local packaging problem unrelated to ltpred.*

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | clean |
| All examples | `python examples/*.py` | 4/4 exit 0, 5.2–14.6 s each, empty stderr |
| README quickstart | verbatim into a scratch script | runs; `est`/`var` shape `(2000,)`; `se` 0; PA vs Gibbs on 200 families corr 0.99946, max abs diff **0.0104** — the claimed "~1e-2" is accurate |
| `docs/quickstart.md` §1–5 | verbatim into a scratch script | runs in 1.5 s; `[('P1', 1.357058752244415), ('P2', 0.4619554497275081)]` |
| `data-preparation.md` multi-trait recipe | verbatim, `(n_people, n_traits)` bounds | runs; `families_from_columns` accepts 2-D bounds and gives each `Member` per-trait arrays. I suspected this snippet was wrong; it is not |
| Wheel contents | `unzip -l dist/ltpred-0.4.0-py3-none-any.whl` | 22 files, **all** under `ltpred/` |
| Sdist contents | `tar tzf dist/ltpred-0.4.0.tar.gz` | `ltpred/`, README, CHANGELOG, CITATION, LICENSE, MANIFEST.in, PKG-INFO, pyproject.toml, setup.cfg. **No** `docs/`, `examples/`, `benchmarks/`, `tests/`, `research/` |
| Docs build | not re-run | CI builds `mkdocs build --strict` on every push |

Nothing here is broken in the sense of failing. Every documented command that
claims to run, runs. The findings are about what the documentation set does not
offer, and about a gap between the package a user installs and the package the
docs describe.

## 2. What holds up

The documentation is **accurate**. That is worth stating plainly, because it is
the hard part and it is done. I found no false statement. Every number I could
re-derive matched: the README's PA/Gibbs agreement claim reproduced at 0.0104
against a claimed "~1e-2"; the quickstart's scores came out of a verbatim
transcription; the multi-trait recipe I expected to be a documentation bug is
correct, because `families_from_columns` genuinely accepts per-trait bound
columns.

The **hedging discipline is a genuine strength** and should not be traded away.
The distinction between Monte-Carlo error and approximation error (`res.se` is
exactly 0 under PA, "which is not the same as zero approximation error"); the
warning that a family-history phenotype can *concentrate* a birth-cohort trend
rather than dilute it (λ = 10.275 with one lifetime K against 4.616 for the raw
label); the insistence that `use="prediction"` hides own diagnosis but does not
select a disease-free risk set; the note that the sib tetrachoric estimates
`h² + 2c²` and is not interchangeable with the parent–offspring one — these are
exactly the errors a competent analyst makes, and the docs prevent them. Most
packages would not think to mention them.

The **engineering hygiene is above the norm**. CI runs all four examples end to
end (`.github/workflows/ci.yml`, job `examples`), with a comment recording that
they "were previously run by neither the tests nor CI, so they could rot
silently". The docs job builds `--strict` and notes that strict mode still
leaves link resolution at INFO. `scripts/check_evidence.py` reconciles
release-defining numbers against committed artifacts. `tests/test_vignette.py`
pins the register example's calendar and closure contracts. The provenance
instinct is consistently good.

The **information architecture is sound at the page level**. `index → guide →
quickstart → vignette → workflow(5) → reference(4)` is a sensible shape, and
"choose a method" as a distinct page is the right call for a package where four
published method names describe observation models rather than algorithms.

So the problem is not accuracy or rigour. It is **genre**: the documentation set
is written as a contract, and a contract is what a user needs *second*.

## 3. Measurements

*Table 2. Volume, redundancy and executability of the user-facing documentation.*

| Page | Lines | Words | Python blocks | Blocks that define their own data |
|---|---:|---:|---:|---:|
| `README.md` | 255 | 1,626 | — | — |
| `docs/index.md` | 40 | 237 | 0 | — |
| `docs/guide.md` | 176 | 1,343 | 0 | — |
| `docs/quickstart.md` | 108 | 614 | 5 | 2 |
| `docs/vignette.md` | 948 | 6,481 | 14 | **2** |
| `docs/data-preparation.md` | 463 | 3,113 | 9 | 5 |
| `docs/cip-estimation.md` | 290 | 2,176 | — | — |
| `docs/estimation.md` | 486 | 3,597 | 10 | 2 |
| `docs/inference.md` | 420 | 2,876 | — | — |
| `docs/assumptions.md` | 192 | 1,639 | — | — |
| **subtotal before `api.md`** | | **23,702** | | |
| `docs/algorithm.md` | 1,056 | 8,028 | | |

*Table 3. Redundancy and register, measured on the documentation set.*

| Quantity | Value |
|---|---|
| Restatements of the "own status in / out" rule | **28** across 7 files (README 4, guide 3, quickstart 1, vignette **13**, data-preparation 1, estimation 2, assumptions 4) |
| Pages enumerating the three uses | 6 (README, index, guide, vignette, estimation, algorithm) |
| Vignette caveat vocabulary | "not" ×66, "do not" ×12, "does not" ×10, "must" ×4, "cannot" ×3, "rejected" ×3 |
| Vignette line-initial imperatives (Run/Call/Pass/Use/Supply/Build/Choose) | **9** |
| Exported names (`_EXPORTS`) | 60 |
| Exported names never mentioned in `docs/*.md` or README | **9**: `BootstrapResult`, `CipCurve`, `FitResult`, `MultiTraitPairwiseResult`, `TetrachoricResult`, `VarCompResult`, `liability_threshold`, `tetrachoric_matrix`, `tetrachoric_table` |
| Public simulation entry points | **1** (`simulate_under_LTM_single`) |
| Simulation helpers in `benchmarks/`, unreachable by an installed user | **6** (Table 4) |

*Table 4. Simulators that exist in the repository but ship in neither the wheel
nor the sdist (`MANIFEST.in` has `prune tests`, `prune benchmarks`,
`prune report`; `pyproject.toml` sets `packages = ["ltpred"]`).*

| Function | Location | What it produces | Tutorial use it would serve |
|---|---|---|---|
| `build_register` | `benchmarks/bench_register_pipeline.py:139` | `(status, age, onset, genetic, birth_time, residual_var)` over one consistent population liability field | **the register driver's input, with known truth** |
| `pedigree_birth_times` | `benchmarks/bench_register_pipeline.py:76` | generation-coherent calendar birth times (union-find over co-parents) | `birth_time` / `index_time` for use I |
| `simulate_registry` | `benchmarks/bench_cip_estimation.py:87` | per-person follow-up: entry age, exit age, event (0 censor / 1 diagnosis / 2 death), Gompertz mortality | step 2 — estimating a CIP, competing risks |
| `simulate_families_multi` | `benchmarks/_common.py:135` | multi-trait families from `construct_covmat_multi`, parameterised by `h2_vec`/`rg`/`rp`; optional onset-age decay branch | multi-trait scoring, `fit_pairwise_multi` |
| `simulate_families_components` | `benchmarks/_common.py:216` | families under `sum_c props[c] K_c + e2 I` for A/C/M proportions, optionally returning true `g` | `fit_variance_components`, accuracy checks |
| `simulate_genotype_families` | `benchmarks/_common.py:260` | standardised genotypes, causal index, true `g`, families | use II end-to-end, GWAS phenotype |

## 4. Findings

**T1-1. The package exports one simulator, and it cannot produce input for the
package's own register driver.**

`ltpred/__init__.py` sets `_EXPORTS["simulate"] = ["simulate_under_LTM_single",
"Simulation"]`. That single entry point builds **role-grammar** families
(`fam_vec=["m","f","s1",…]`) for **one trait** under **one logistic CIP**; its
own docstring concedes the limit — "This helper has one logistic CIP and does
not simulate the full sex/birth-cohort personalisation of LT-FH++".

The installed, supported, public route for real data is the population
trio-register driver `estimate_liabilities`, which needs `ids`, `father`,
`mother`, `status`, `age`, and for use I `birth_time` and `index_time`, plus
`strata` and `cip_by_stratum`. **No public function produces any of it.** There
is no `ltpred.datasets`, no bundled `.csv`/`.npz`, no loader, no fetcher; the
only data files in the repository are the R-lock fixtures under `tests/fixtures/`
(pruned from the sdist) and benchmark output CSVs (likewise pruned).

The consequence is concrete. The only worked register input anywhere in the
documentation is **six hand-typed people** in
`examples/vignette.py:register_example` — a leakage/invariance check with a toy
CIP, which the script itself correctly labels "not clinical calibration or new
performance evidence". A user who wants to learn the route the docs recommend
for real data has no simulated cohort to learn it on, and no way to reproduce
the RESULTS §§20–21 design on their own machinery.

This is not a missing-capability finding. **The capability exists and is
stranded.** Table 4 lists six simulators in `benchmarks/`, including one
(`build_register`) that returns exactly the register columns plus the true
genetic value, and one (`pedigree_birth_times`) that is a self-contained
union-find generation assigner with no statistical content at all. They are
unreachable because `MANIFEST.in` prunes `benchmarks` and `packages = ["ltpred"]`
— verified against the built wheel (22 files, all `ltpred/`) and the sdist.

`build_register` is parameterised by module globals (`H2`, `EVAL_AGE`,
`TRUE_CIP`, `AGE_GRID`, `BASE_BIRTH_TIME`, `GENERATION_YEARS`), so promotion
means turning six constants into arguments, not a rewrite. `pedigree_birth_times`
needs only two. Recommendation: promote them into `ltpred/simulate.py` (or a new
`ltpred/datasets.py`), and have `bench_register_pipeline.py` import the promoted
function so the evidence ledger and the tutorial **cannot drift apart** — the
same argument the repository already applies to its benchmark numbers via
`scripts/check_evidence.py`. `simulate_registry` should follow, since step 2
(CIP estimation) currently has no simulated follow-up data either, and competing
mortality is precisely the subtlety the CIP page warns about.

**T1-2. The vignette is described as the run-book, but nothing executes its
code: 12 of its 14 blocks are fragments, and the page binds only four names.**

`README.md` calls the vignette "the run-book"; `docs/index.md` and `guide.md`
both route a new user to it as the answer to "how to run ltpred". Measured
against that promise: 14 Python blocks, 118 lines of code, **2** of which define
their own data.

To be precise about what is and is not there, because the page deserves credit
for part of it: it *does* have a literal cohort preamble (§"How to read the code
on this page", lines 192–200) that binds `sim`, `h2`, `K` and `n_fam` from one
runnable `simulate_under_LTM_single` call, and it does disclose that "Most blocks
below are **fragments**: they show a call with your own columns". So this is a
design choice with a partial mitigation, not an oversight, and I do not report it
as a falsehood.

What remains is that the fragments reach far beyond what that preamble binds.
The names they use and no block on the page ever defines are `status`, `age`,
`cip_ages`, `cip_values`, `fam_id`, `role`, `pid`, `ids`, `father`, `mother`,
`birth_time`, `index_time`, `strata`, `cip_by_stratum`, `age_entry`, `age_exit`,
`event`, `status_o`, `status_m`, `c2`, `m2`, `C`, `M`, `gwas_ids`,
`prediction_ids` and `lifetime_K` — including the entire register-route input,
which is the route the page recommends for real data.

And nothing guards any of it. `tests/test_vignette.py` is 32 lines and exercises
only `register_example()` and `_incident_risk`. CI runs the examples, not the
prose. So a signature change in `thresholds_from_cip` or `estimate_liabilities`
would leave twelve unrunnable blocks in the page the README calls the run-book,
and every gate would stay green.

The consequence is that the documentation has **no executable middle ground**:
below the vignette is a 614-word quickstart on two families; above it is a
375-line example script that prints a wall of caveats. Neither is a dataset a
reader can modify and explore.

Recommendation: keep the vignette as the contract — it is good at that — and put
the runnable middle ground on a new page rather than retrofitting 948 lines.
Extend the existing preamble to bind the register columns too, and add a test
that executes the page's blocks. The repository already has the mechanism
(`tests/_helpers.load_script`) and the precedent in `test_vignette.py`; this is
an extension of an existing pattern, not new machinery.

**T2-1. The README's headline quickstart runs a materially weaker cohort than
the vignette's, and than every number quoted in the documentation.**

`README.md` opens with `simulate_under_LTM_single(..., use_age=True, seed=1)`;
the vignette's canonical cohort — the one behind Table 4, the law-of-total-variance
check, the tetrachoric 0.237, the fitted ĥ² = 0.363 — uses `use_age=False`.
Measured at `n_sim=2000`, seed 1, h² = 0.5, K = 0.05:

| Cohort | Proband cases | Score SD | corr(score, true *g*) | Mean posterior variance |
|---|---:|---:|---:|---:|
| `use_age=True` (README) | **11** / 2000 (0.55%) | 0.1846 | 0.2667 | 0.4682 |
| `use_age=False` (vignette) | 93 / 2000 (4.65%) | 0.2881 | 0.3967 | 0.4135 |

The first thing a new user runs therefore yields 11 informative cases instead of
93, a score with 36% less spread, and a correlation with the truth of 0.267
rather than 0.397. The README's PA/Gibbs agreement claim is *accurate* on that
cohort (I measured 0.0104), but it is a weak demonstration: with 98.5% of
probands carrying no case information, the two engines are largely agreeing
about the prior. A reader who then opens the vignette finds different numbers
from a different cohort with no statement of why.

Recommendation: make the README quickstart `use_age=False`, matching the
vignette and every quoted figure, and present the age-aware variant as a
labelled second block that says what changes. This is a one-token edit with a
disproportionate effect on first contact.

**T2-2. The only worked multi-trait example is imported from a directory that
ships in neither distribution — and duplicates a benchmark helper.**

`docs/vignette.md`, §0, instructs:

```python
from examples.joint_inference import example_families
```

`examples/` is not in `packages`, is not named in `MANIFEST.in`, and is absent
from both the built wheel and the sdist (Table 1). The import works from a
checkout only. The vignette does say "runs from a source checkout", so this is
disclosed rather than concealed — but ROADMAP priority 7 is *publish the first
package release*, and after that the vignette's only multi-trait example is dead
on arrival for every `pip install ltpred` user. The same applies to all four
`examples/*.py` links in `README.md` and `guide.md`, which resolve only on
GitHub.

Second, `example_families()` is not an example — it is a **simulator**: 40 lines
building a two-trait covariance as `np.kron(g, a) + np.kron(s, c) + np.kron(t, m)
+ np.kron(e, np.eye(5))`, Cholesky-factoring it, drawing 3,000 × 10 latents and
reshaping trait-major to person-major. It is built entirely from public API
(`kinship_from_pedigree` for the additive block, `prevalence_thresholds`,
`families_from_columns`) plus NumPy, so it needs no new dependency to promote —
it belongs beside `simulate_under_LTM_single`.

Third, it is the **third** private route to component-structured family
liabilities in this repository, and the three are not copies of one another,
which is worse: they are three different parameterisations of the same object,
so a reader cannot tell which is canonical.

| Generator | Traits | Parameterised by | Covariance route |
|---|---|---|---|
| `_common.py:135 simulate_families_multi` | multi | `h2_vec`, genetic corr `rg`, full-liability corr `rp` | `construct_covmat_multi`; optional onset-age decay branch importing `research.advanced_fitting._decay_cov` |
| `_common.py:216 simulate_families_components` | single | A/C/M variance **proportions** | `_component_matrix`, summed; can return true `g` |
| `examples/joint_inference.py:24 example_families` | two | h², `rg`, sibship-C, couple-M, residual `re` | explicit `np.kron` sum, A from `kinship_from_pedigree`, hand-built C/M indicators |

None is in the public API. Recommendation: promote one multi-trait simulator
into `ltpred` (the example's is the best candidate — public-API-only, no
`research/` dependency, and its truth is quoted in the vignette), have
`examples/joint_inference.py` call it, and document why the two benchmark
generators remain internal rather than leaving three unexplained variants.


**T2-3. `examples/registry_pipeline.py` is misnamed: it never calls the register
driver, and it is the register driver that real-data users need.**

The file's docstring promises "End-to-end template: a registry-style status/age
table -> a GWAS phenotype". Grepping it for `estimate_liabilities`,
`birth_time`, `index_time`, `father`, `mother`, `strata` returns one hit — the
word "mother" in a sentence about family structure. It imports
`age_thresholds`, `families_from_columns`, `estimate_liability`: the **role**
API. So the example named for the register route demonstrates the other route,
and the route it demonstrates is the one already covered by the quickstart.

The gap this leaves is the same one as T1-1 seen from the examples side: the
supported public driver for population data — the one with `use="gwas"` /
`use="prediction"`, calendar-time familywise censoring, `proband_state` and
`frac_records_with_unresolved_parents` diagnostics — has a single six-person
illustration and no realistic worked example anywhere.

Recommendation: either rename the file to reflect the role API
(`role_pipeline.py`), or — better, given T1-1 — retarget it at
`estimate_liabilities` over a promoted simulated register, which would make it
the missing end-to-end template for the route the docs actually recommend.

**T2-4. The documentation set is optimised as a contract, and the tutorial it
contains is 614 words on two families.**

23,702 words stand between a new user and the API reference; `algorithm.md` adds
8,028 more. The single most important rule in the package — whether the
proband's own diagnosis enters `D_F` — is stated **28 times** across seven
files, 13 of them in the vignette. The three uses are enumerated on six pages.
In the vignette, caveat vocabulary ("not" ×66, "do not" ×12, "does not" ×10,
"cannot" ×3, "rejected" ×3) outnumbers line-initial imperatives by roughly ten
to one.

None of those statements is wrong, and most are load-bearing; §2 argues they
should be kept. But repetition is a symptom of a document that cannot assume its
reader has internalised anything, because it is simultaneously the tutorial, the
reference, and the liability disclaimer. The two genres have opposite
requirements: a tutorial must let a reader *succeed quickly* and defer caveats;
a contract must state every caveat *at the point of risk*. One page cannot do
both, and the vignette currently tries, at 6,481 words.

The quickstart inherits the problem in miniature. At 614 words it is admirably
short, but its dataset is two families and six hand-typed rows — small enough
that the page must immediately warn "Do **not** fit `h²` from this two-family toy
example", and small enough that nothing about grouping, heterogeneity of family
structure, or score alignment is actually exercised. It is too small to learn
from and too caveated to be a fast start.

Recommendation: split the genres explicitly. Keep the vignette as the contract
and let it stay long. Add one **tutorial** page — a single simulated cohort
obtained from one call, five numbered steps, every block runnable, with all
caveats collected into one closing "what this tutorial skipped" admonition that
links into the contract rather than interrupting the flow. Then the 28
restatements can fall to a handful, because the rule will have one canonical
home and the tutorial will have already demonstrated it.

**T3-1. `make_toy_table` recovers case status from bound finiteness rather than
from the simulator's own `status` field.**

`examples/registry_pipeline.py:52` sets `is_case = np.isfinite(m.lower)`,
reading the answer off the interval encoding, when `sim.status[role]` is
authoritative and already returned by the simulator. It is correct for the
default `case_encoding="pin"` and would silently change meaning under
`"interval"` or an uninformative member. The same coupling appears in
`examples/vignette.py` (§2 and the final block, `int(np.isfinite(m.lower))`).
Cheap to fix and worth fixing, because these are the files a new user reads to
learn what the fields mean.

**T3-2. First-run stderr noise.** With Numba present, the README quickstart
prints `OMP: Info #276: omp_set_nested routine deprecated, please use
omp_set_max_active_levels instead.` It is informational and harmless, but it is
the first thing a new user sees immediately after their first successful run, it
is not from ltpred, and it reads like an error. Worth a note in the
installation section or a suppression at import.

**T3-3. The quickstart's headline threshold helper is downgraded on two
reference pages, without a warning at the point of use.** `docs/quickstart.md`
step 2 builds bounds with `age_thresholds(status, age, pop_prev=0.05)`.
`data-preparation.md`'s builder table labels that same function "personalised
demo", and `guide.md`'s route table labels the logistic path "age-only
demonstration, not full LT-FH++". Each is right; the quickstart is nonetheless
the page a beginner copies, and it does carry a one-line pointer to
`thresholds_from_cip`. An admonition at the point of use — rather than only
downstream — would prevent the copy-paste.

**T3-4. Nine exported names never appear in the documentation.** Mostly result
containers (`FitResult`, `VarCompResult`, `BootstrapResult`, `CipCurve`,
`TetrachoricResult`, `MultiTraitPairwiseResult`), plus `liability_threshold`,
`tetrachoric_matrix` and `tetrachoric_table`. `docs/api.md` is 175 lines /
1,097 words for a 60-name surface. Low impact — mkdocstrings renders the
docstrings — but a user searching the prose for "what does `fit_heritability`
return" will not find the type named.

## 5. Recommendations, in priority order

1. **Promote the stranded simulators (T1-1, T2-2, T2-3).** Add a register
   simulator, a multi-trait simulator, and a follow-up/CIP simulator to the
   public API by lifting `build_register`, `pedigree_birth_times`,
   `simulate_registry`, `simulate_families_multi` and
   `simulate_families_components` out of `benchmarks/`, converting their module
   globals into arguments. Have the benchmarks import the promoted functions so
   the ledger and the tutorial share one generator. This is the single change
   that unlocks the most: it makes the register route learnable, makes the
   multi-trait vignette example installable, removes a duplicate implementation,
   and gives the tutorial in (3) something to run on.
2. **Make the vignette executable (T1-2).** One literal cohort preamble; every
   block runnable; a test that concatenates and executes them, reusing
   `tests/_helpers.load_script`.
3. **Write the tutorial page (T2-4).** One cohort, five steps, no inline
   caveats, one closing "what this skipped" admonition. Link the contract, do
   not restate it. Then de-duplicate the 28 restatements down to the canonical
   home plus pointers.
4. **Fix the README cohort (T2-1).** `use_age=False`; move the age-aware variant
   to a labelled second block.
5. **Retarget or rename `examples/registry_pipeline.py` (T2-3)**, and fix the
   `isfinite(m.lower)` status recovery in both example scripts (T3-1).
6. **Polish:** the OMP notice (T3-2), a point-of-use admonition on
   `age_thresholds` (T3-3), and the nine undocumented export names (T3-4).

Items 1–4 are the ones that change what a new user can do. Items 5–6 are
hygiene. None of them requires touching the statistical core, and none of them
weakens a caveat: the recommendation throughout is to *relocate* rigour, not
reduce it.

## 6. What this review did not do

It did not re-derive the liability model, the covariance construction, the
Gibbs conditionals, the Pearson–Aitken recursion, the quadrature engine, the CIP
estimators, the fitters, or the tetrachoric and liability-scale helpers; four
audits in the preceding seven weeks did, and this one has nothing to add to
them. It did not re-run the benchmark suite, the R locks, the evidence guard, or
the HAPNEST path. It did not audit the `research/` tree beyond noting that its
documentation page (`docs/research.md`, 365 lines) is new since `e9bfa66` and
was read only for consistency with the package docs. It did not assess the
typeset technical report (`report/ltpred_methods.pdf`) or `CITATION.cff` beyond
confirming they ship. It did not build the documentation site, relying on the
CI `--strict` job. Its measurements were made on one machine and one stack
(Table 1), and the distribution inventory was taken from a **stale local
`dist/` at v0.4.0** rather than a fresh v0.7.0 build; the packaging
configuration that determines those contents (`MANIFEST.in`,
`[tool.setuptools] packages`) is unchanged since, but a rebuilt artifact should
be checked before release — which ROADMAP priority 7 and `docs/RELEASING.md`
already call for.

## 7. Dispositions (same day)

Most of §5 was implemented on 2026-09-21 in the same working tree, except where
the table below records otherwise; the detail is in the `[Unreleased]` section of
`CHANGELOG.md`. Per finding, with the evidence that each claim holds:

| Finding | Disposition | Evidence |
|---|---|---|
| T1-1 one simulator, register route unteachable | **Implemented.** `simulate_pedigree`, `pedigree_birth_times`, `simulate_register_liabilities`, `simulate_followup_records` promoted to `ltpred.simulate`; benchmarks delegate to them | bit-identical to the originals at `atol=0`/`rtol=0` in every arm, including all three `simulate_registry` arms; `test_benchmark_generators_are_the_public_ones` pins the delegation |
| T1-2 vignette not executable | **Implemented, narrowed.** Rather than retrofit 948 lines, a new runnable page carries the middle ground and the vignette keeps its contract role; its preamble now binds the register names too | `docs/tutorial.md`; `tests/test_tutorial.py` executes its blocks in order and diffs each against the quoted output. Teeth confirmed by negative control |
| T2-1 README cohort weaker than the vignette's | **Implemented.** README uses `use_age=False`; the age-aware variant is a labelled second block | re-ran verbatim: 4.65% proband cases, PA/Gibbs max abs diff 0.0132 |
| T2-2 multi-trait example unreachable when installed | **Implemented.** `simulate_under_LTM_multi` promoted; the vignette and the example both use it | `examples/joint_inference.py` output unchanged to the last printed digit; the new vignette block reproduces `rg = 0.5434515836721618` exactly |
| T2-3 `registry_pipeline.py` misnamed | **Not implemented — deferred.** The tutorial now covers the register route end to end, so retargeting the script would duplicate it; renaming is a separate call | recorded under "Not changed" in the CHANGELOG |
| T2-4 volume and genre | **Partly implemented.** The genres are now separated (tutorial vs contract) and the tutorial collects its caveats in one closing admonition. The 28 restatements of the own-status rule were **not** de-duplicated: doing so safely means editing the contract pages, which needs its own pass | `docs/tutorial.md` §"What this tutorial skips" |
| T3-1 status read off bound finiteness | **Implemented** in both example scripts | both examples' stdout byte-identical before and after |
| T3-2 OMP notice | **Implemented** — noted in the quickstart's install section | — |
| T3-3 `age_thresholds` downgraded only downstream | **Implemented** — a point-of-use admonition in quickstart step 2 | — |
| T3-4 nine undocumented exports | **Not implemented — deferred** | recorded under "Not changed" in the CHANGELOG |

Two corrections to this report, made during implementation when execution
disproved them:

- **T2-2 overstated the duplication.** `example_families` is not a copy of
  `_common.simulate_families_multi`. There are three private routes to
  component-structured family liabilities, differently parameterised (see the
  table in T2-2); the finding is stronger as written there, but the original
  "second private copy" phrasing was wrong.
- **T1-2 understated the vignette.** The page already had a literal cohort
  preamble binding `sim`/`h2`/`K`/`n_fam` and already disclosed that the rest are
  fragments. The finding stands only for the register-route names, which nothing
  on the page bound, and for the absent execution guard. T1-2 has been rewritten
  accordingly.

Also recorded: `tests/test_pairwise_multi.py::test_boundary_withholds_uncertainty_and_zero_variance_correlations`
**fails at clean `2722a59`** on this machine's stack (CPython 3.10.20, NumPy
1.26.4, SciPy 1.15.3, Numba 0.66.0) — a `pytest.warns` block reports "Emitted
warnings: []". Verified pre-existing by `git stash` and re-run; it is unrelated
to anything in this audit and was left alone. Everything else passes: 884 tests,
`ruff check .` clean, `mkdocs build --strict` clean, all four examples exit 0.
