---
title: Independent review (v0.7.3, computational efficiency)
description: Computational-efficiency and benchmark-provenance audit of ltpred v0.7.3.
---

# Computational-efficiency review of ltpred v0.7.3 (2026-09-25)

A read-only audit of **computational efficiency**: where the package spends time
and memory, and whether the speed and memory claims it publishes are supported
by artifacts a future maintainer can actually reach. This review covers the tree
at commit `6686541` (v0.7.3). It follows, and is independent of, the five
preceding audits — v0.3.4 (`REVIEW_2026-08.md`), v0.4.2 (`REVIEW_2026-09.md`),
v0.5.1 (`REVIEW_2026-09b.md`), v0.6.2 (`REVIEW_2026-09c.md`, at `e9bfa66`) and
v0.7.0 (`REVIEW_2026-09d.md`, at `2722a59`), fifteen commits behind this one.
Its scope delta is therefore v0.7.1 → v0.7.3, the three releases that carried
the bulk of the package's optimisation work.

It also reaches further back than the in-repo chain, because the efficiency
story does not start there. An external methods-and-efficiency review of v0.5.2
(commit `8856ba8`, 2026-09-05) recommended four algorithmic reductions; three
shipped in `52dec52` (v0.6.0) as `ltpred/_selected_kinship.py`,
`ltpred/pairwise.py` and `ltpred/quadrature.py`. Two later commits then removed
the benchmark script and report subsection that carried their measurements
(`785e433`). §2 records what that review asked for and what actually shipped;
§4 traces the consequence for provenance.

Those five audits re-derived the statistical core against independent oracles.
**This one does not, and deliberately so.** It re-derives no likelihood, no
covariance and no posterior. Every claim below is either a `file:line`
citation, a git fact, or a measurement I ran; the audit changed no package code
and left the working tree clean.

Severity tiers follow the chain, re-read for this domain: **T1**, a published
statement is false, or a release-defining efficiency claim has no reachable
artifact; **T2**, a material provenance or disclosure gap that will drift;
**T3**, polish, labelling and completeness.

## 1. Scope and method

Two questions frame the audit. First: *is the remaining cost where a user would
expect it, given three releases of optimisation?* Second — and it turned out to
be the more productive one: *can every speed and memory claim in this repository
be traced to an artifact that is in the repository?*

The second question matters more than it looks. This package has unusually
strong provenance instincts: a `run_manifest.jsonl` wrapper that records the
clean source commit, exact command, runtime stack, thread settings, machine
profile and artifact hashes; an evidence checker that reconciles prose against
committed CSVs; and a documented rule that "a speed-up is only interpretable
alongside the thread count it was measured at" (`benchmarks/README.md:14-18`).
The findings below are not about that discipline being absent. They are about
where it stops applying.

Method. I reconstructed the efficiency commit chain by git archaeology
(`52dec52` → `b516271` → `785e433` → `8926f25` → `d57e1f9` → `6686541`), read
the three retained evidence capsules, and then **independently re-derived every
numeric speed and memory claim** I could find in `docs/`, `CHANGELOG.md`,
`benchmarks/RESULTS.md` and `report/` against the artifact that supposedly backs
it. Where a claim had no artifact, that is recorded as the finding. I read the
selected-kinship cache and the covariance-reduction guard in full, since both
answer questions the 2026-09-05 review left explicitly open. I reproduced two
cases from the v0.7.3 capsule on the current tree. The implementation arm of §4
(T1-3 and T2-4 → T2-8) came from directed profiling probes confined to `/tmp`
and built on the public simulation API; §6 records how I verified their output
and the one correction that resulted.

*Table 1. Baseline checks on the review machine (macOS arm64, 10 cores;
`ltpred314` — CPython 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0; BLAS,
OpenMP and Numba pinned to one thread as in CI; AC power, Low Power Mode off).*

| Check | Command | Result |
|---|---|---|
| Core suite | `pytest -q` | **1008 passed, 2 skipped**, 0 failed, 72.0 s |
| Research suite | `pytest -q research/tests` | **193 passed, 1 skipped**, 183.6 s |
| Lint | `ruff check .` | clean |
| Strict docs | `.venv/bin/mkdocs build --strict -d /tmp/...` | exit 0, 2.93 s |
| Working tree | `git status --short` | empty before and after |

`pyproject.toml:55` sets `testpaths = ["tests"]`, so a bare `pytest -q` from the
repo root excludes `research/tests`. That is why the v0.7.3 capsule quotes
"1,202 passed, 1 skipped" for "package and research tests" while the core suite
alone reports 1,008 here; the two are consistent, not contradictory.

The core suite is **fully clean on this stack**. The
`test_pairwise_multi.py::test_boundary_withholds_uncertainty_and_zero_variance_correlations`
failure recorded at v0.7.0 (`REVIEW_2026-09d.md` §7) does not reproduce under
`ltpred314`; it was specific to the ambient CPython 3.10 / NumPy 1.26 stack.

One methodological limit governs everything in §3. Profiling ran concurrently
with this audit, so the 1-minute load average reached ~20. By this repository's
own standard — the v0.7.1 capsule discards a load-13–26 attempt rather than
quoting it — **no absolute wall-clock number collected during this review is
quoted as a clean measurement.** Where a timing appears below it is either a
reproduction of a committed capsule case, reported as *consistent* rather than
clean, or a git/CSV fact independent of machine load. Allocation counts are
load-independent and are quoted freely.

## 2. What holds up

**The 2026-09-05 review's central production gap was closed properly.** That
review measured selected-pair kinship recursion at 1.83×–3.40× on three synthetic
workloads with means and variances exactly equal to the original, but refused to
endorse it for production:

> "This prototype uses an unbounded Python cache: it is not a biobank-ready
> memory design. A production implementation needs compact storage and a bounded
> cache or component/chunk strategy; worst-case relationship requests can
> approach quadratic size."

The shipped `ltpred/_selected_kinship.py` answers this directly, and the answer
is good. The cache is **bounded** by default (`cache_max_entries=100_000`,
`_selected_kinship.py:17`) and implements true LRU eviction — `_remember`
(`:44-49`) calls `move_to_end` then `popitem(last=False)`. Deep pedigrees are
handled by an **explicit continuation stack** rather than Python recursion
(`:52-86`), so there is no recursion-depth cliff. And the subtle hazard that
bounded caching usually introduces — evicting a value the active computation
still needs — is designed out and documented at `:53-55`: a continuation retains
its first child value while the second is evaluated, so "even a cache of size
zero cannot evict a value that the active computation still needs". The module
docstring states the invariant precisely: "eviction changes work, never
ancestry." That is the right sentence to have written.

The one part of the recommendation only partly met is *compact* storage: entries
remain Python floats in an `OrderedDict` keyed by integer tuples, not an
array-backed store. At the 100,000-entry default that is bounded and acceptable;
it is simply not the most memory-dense design available. This is not a finding,
only a note on how much of the 2026-09-05 advice was taken.

**The other two 2026-09-05 recommendations also landed, and landed correctly.**
Recommendation 3 (condition pinned onsets exactly before approximating intervals)
was deliberately output-changing, and the change is the one the review asked for:
`pa_algorithm` now returns **1.475175201 / 0.245246263** on the review's
two-observation example, matching its pin-first target exactly rather than the
old 1.482323441. Recommendation 4 (parental-factor quadrature) reproduces the
review's Table 2 to nine decimal places — **1.239036113 / 0.213147156** at 64
nodes, refinement error 4.4e-15. Its scope limits are hard and worth knowing:
roles must match `o|m|f|s[1-9][0-9]*`, so grandparents, aunts and uncles raise
`ValueError`; and it requires scalar `0 ≤ h² < 1`, no `c2`/`m2`, no mixture, a
single trait and a nuclear family. **The extended-family register workload
therefore has no quadrature path at all** and stays on PA or Gibbs. That is a
scope fact rather than a defect, but it bounds how much of the register path the
v0.7.3 quadrature optimisations can ever help — which is consistent with T1-4
finding the PA object path, not the kernel, to be the bottleneck.

**`_covariance_reduction_is_safe` removes an eigendecomposition by proof, not by
omission.** The 2026-09-05 review counted three eigendecompositions per family
on the kinship scoring path: relationship validation, positive-definiteness
handling, and PA validation. `_selected_kinship.py:101-118` now establishes an
analytic eigenvalue lower bound — pedigree diagonals lie in [1, 2], so
standardised target genetic variance `a >= q` and residual variance
`r >= (1-q)/(1+q)`, bounding the inverse trace by `1/q + 3*n/r` — and requires
100× the legacy `1e-8` repair threshold before skipping the repair, falling back
to the unchanged full-pedigree correction near boundaries. This is the correct
way to delete a numerical safety check: replace it with a proof that the
condition it guarded cannot arise, keep a margin, and retain the fallback. The
v0.7.3 capsule's related note — that merely removing the register covariance
checks "is not justified by the review's timing percentage" — is the same
instinct, and both are right.

**Thread and load discipline is real, and I could not break it.** All 14
`run_manifest.jsonl` rows carry `environment.thread_settings` plus a resolved
`numba_threads`. Both time/memory driver capsules carry `thread_variables` and
`numba_threads: 1`. The load-average caveat is attached at *every* site that
quotes a contended number: `RESULTS.md:209-210` ("2.56 to 5.86"), `:941-943`
("load 2.1"), `:1688` ("2.46 to 3.29"), the v0.7.1 capsule ("2.8–4.1"), and
`report/efficient_inference.tex`. The v0.7.1 capsule additionally *discloses
that a load-13–26 attempt was discarded*, and the manifest retains that
discarded run as row 11 while row 12's hash matches the committed CSV
byte-for-byte. Retaining the run you threw away is stronger provenance than most
published packages offer. I found no contended number presented as clean — with
one exception, T1-1, and that one fails by omission rather than by fabrication.

**The committed capsule reproduces.** Two cases from
`benchmarks/results/2026-09-24-review/` were re-run on the current tree using
its own `probe.py` (Table 3). Both land within 0.6% of the recorded median, and
the traced peak allocation counts are **byte-identical**. Process RSS did not
reproduce (168.5 vs 143.0 MiB; 112.7 vs 119.8 MiB), which I attribute to
redirecting `NUMBA_CACHE_DIR` into `/tmp` rather than to a memory regression —
RSS includes interpreter, imports and JIT state and is the least transferable
number in the capsule.

**`check_evidence.py` is a genuine guard where it applies.** It pins the v0.6.1
capsule exhaustively — 28 workers, all `thread_variables` equal to `"1"`, the
power guard, source stability, baseline and candidate revisions, and a SHA-256
of `results.json` against `provenance.json` — plus the scaling object-path
speedups, the PA-robustness floor, IPW, the R-lock fold times *with thread
provenance asserted*, PGS, and three paper tables cell by cell. The finding in
T2-2 is that its scope stops at v0.6.1, not that it is weak.

## 3. Measurements

*Table 2. The PA-array scaling grid as committed, versus what `docs/estimation.md`
says about it. Source: `benchmarks/bench_scaling.csv`, 10 rows, axes `n_fam` and
`family_size`.*

| Quantity | Committed CSV | `docs/estimation.md:359-360` |
|---|---|---|
| `pa_array_speedup` range | 12.648 – 28.625 | "13–29×" — faithful rounding |
| `fam_per_s_pa_array` range | 2.024e6 – 6.290e6 | "2.02–6.29 million" — exact |
| `threads` | **4** on every row | not stated |
| CSV last regenerated | `1684fc5`, **tag v0.4.0**, 2026-08-20 | "the **current** warmed timing grid" |

The transcription is accurate to the digit. The currency claim is not; see T1-1.

*Table 3. Reproduction of two v0.7.3 capsule cases on the current tree. Capsule
figures carry a v0.7.2 working-tree stamp; rerun is v0.7.3 at `6686541`. Median
of three warm calls, one thread, `NUMBA_CACHE_DIR` redirected to `/tmp`. Load
average 20.00 → 20.16 across the run, so these are reported as **consistent**,
not as a clean rerun.*

| case | capsule (ms) | rerun (ms) | delta | traced peak allocation |
|---|---:|---:|---:|---|
| `moments` | 2.442 (2.402/2.442/2.474) | 2.457 (2.475/2.399/2.457) | **+0.6%** | 1,048,886 B — **identical** |
| `pid` | 10.070 (10.070/10.217/9.875) | 10.017 (9.875/10.017/10.274) | **−0.5%** | 1,512,848 B — **identical** |

No single numeric repeat band is documented. The nearest are the v0.7.1
capsule's "every warm median is within 2%" and `RESULTS.md:245-247` ("five
repeats quantify timing variation but do not abolish operating-system noise").
Both deltas sit inside 2% and inside the capsule's own 3-rep spread. The
byte-exact allocation peaks are the load-bearing part, and they are
load-independent.

*Table 4. The efficiency commit chain, and where each claim's evidence lives.*

| Release | Commit | Efficiency content | Artifact in repo? |
|---|---|---|---|
| v0.6.0 | `52dec52` | selected-pair kinship, nuclear-family quadrature, pairwise-probit fitting | **No** — script and capsule retired by `785e433`; external copy only |
| v0.6.1 | `b516271` | PA and pedigree time/memory | **Yes** — `results/2026-09-09-time-memory-v061-rerun/`, pinned by `check_evidence.py` |
| v0.7.1 | `8926f25`, `7f63253`, `5b42b13` | kinship 25×, register 4×, dense-vs-selected, simulation 4×, object-path PA 0.61→0.36 s | **Partial** — `results/2026-09-23-time-memory-v071/` covers graph, PA batches, register; not the 25×/4× claims |
| v0.7.2 | `d57e1f9` | CIP validation once per call, GWAS bounds once, batch seed bookkeeping 2.4 s→5 ms | **No** — `CHANGELOG.md:127-133` only; the capsule that might have carried them is gitignored |
| v0.7.3 | `6686541` | quadrature moments 39.99×, bound-row dedup 80.66×, pid 4.65×, Mendelian simulation | **Yes** — `results/2026-09-24-review/`, linked from `CHANGELOG.md:34` |

The pattern is the finding: provenance quality is **not monotone in release
date**. v0.6.1 and v0.7.3 are properly captioned; v0.6.0, v0.7.1's headline
multipliers and v0.7.2 are not.

## 4. Findings

The findings fall into three groups, and they have different characters.
**T1-1, T1-2 and T2-1 → T2-3 are provenance findings**: the code is fine and the
claims about it are not reachable. **T1-3, T1-4, T2-4 → T2-13 and T3-12 → T3-14
are implementation findings**: measured avoidable cost in shipping code. **T3-6 →
T3-11 and T3-15 are measured negatives** — changes that look attractive and are
not worth making, recorded so the next pass does not repeat the work. All three
groups are numbered in one sequence by severity, so the identifiers are not
grouped.

**T1-1. `docs/estimation.md` describes a v0.4.0 measurement as "current", omits
that it was taken at four threads, and quotes a ratio whose denominator has
since been made 1.69× faster.**

`docs/estimation.md:359-360` reads:

> This runs the covariance construction once and the parallel PA kernel directly
> — 13–29× faster than the object path in the **current** warmed timing grid, at
> 2.02–6.29 million already-aligned families/s

Three separate defects, each verified.

*Currency.* `git log -1 -- benchmarks/bench_scaling.csv` returns `1684fc5`,
**tag v0.4.0**, dated 2026-08-20. HEAD is v0.7.3. The grid is three minor
releases old and the word "current" is false. This is not a quibble about
vintage for its own sake: four of the intervening releases changed the code
being timed.

*Direction of the bias.* The ratio's denominator is the object path, and
`CHANGELOG.md:157-159` records v0.7.1 making exactly that path faster — "the
object-input PA path skips array coercion for scalar bounds (**0.61 s to 0.36 s
on 50k families**)". That is 1.69×. If the array path is unchanged, the honest
current ratio is roughly 7.5–17× rather than 13–29×. The stale number therefore
flatters the array API, which is the one the sentence is selling. I have not
re-measured the grid, so I state this as a bound implied by the changelog, not as
a measurement — but the direction is not in doubt.

*Thread count.* Every row of the CSV carries `threads=4`. The sentence names no
thread count, against the repository's own rule at `benchmarks/README.md:14-18`
and against `RESULTS.md:247-250`, which qualifies the same family of numbers
"*at four threads on this machine*". A reader cannot tell whether 6.29 million
families/s is a one-thread or a four-thread figure, and it is the latter.

The same two numbers are repeated into `report/ltpred_methods.tex:652-654` and
thence into the tracked `report/ltpred_methods.pdf`, so the staleness is
published, not merely internal.

`scripts/check_evidence.py` cannot catch this: it never reads `docs/`, and its
docstring asserts "Prose elsewhere links to the ledger instead of repeating its
numbers" — which this prose does not do. Neither `pa_array_speedup` nor
`fam_per_s_pa_array` is pinned anywhere, so the 13–29× and 2.02–6.29 M figures
are unreconciled in every location they appear.

Recommendation: regenerate the grid at v0.7.3 through the provenance wrapper so
it carries a manifest row, replace "current" with the measured version and date,
state the thread count in the sentence, and add `pa_array_speedup` to
`check_evidence.py`. Until the grid is regenerated, the honest fix is to delete
"current" and attribute the numbers to v0.4.0 at four threads.

**T1-2. v0.7.2's Performance figures exist only in the changelog.**

`CHANGELOG.md:127-133` makes two quantitative claims:

> With a 100,001-point curve, 1,000 probands took **1.32 s before and 0.90 s
> after**, the same as with a 121-point curve.
> Seed bookkeeping for 1,000 batches of 1,000 families fell from **2.4 s and
> 23 MiB to 5 ms**.

A repo-wide grep for `1.32 s`, `0.90 s`, `23 MiB` and `100,001-point` returns
only those changelog lines. The v0.7.2 section cites no capsule, no `RESULTS.md`
section and no CSV — in contrast to v0.7.3, which links
`benchmarks/results/2026-09-24-review/README.md` at `CHANGELOG.md:34`, and
v0.7.1, which links `REVIEW_2026-09d.md` at `:168`. These are release-defining
performance claims for a released version and they are unreproducible from the
repository.

The 478× implied by "2.4 s → 5 ms" is the larger claim and the easier to lose
credibility over, because nothing a reader can run reproduces it.

Recommendation: either attach a capsule, or restate both bullets qualitatively
("validates each CIP curve once per call instead of once per proband; scores
bit-identical") and drop the numbers. The qualitative form is already present
in both bullets and carries the engineering content; the numbers carry only the
risk.

**T1-3. A family-free proband in the register driver pays for a covariance
matrix, an eigendecomposition and a batched PA call to obtain what a scalar
formula gives to machine precision.**

Recommendation 6 of the 2026-09-05 review **did land**: `_adult_full_moments`
(`estimate.py:814`) computes family-free ADuLT moments directly, and
`_pa_from_role_arrays` dispatches to it at `estimate.py:855`. That dispatch is
real and correct.

It is also the *only* one. `estimate_liability_from_kinship` (`estimate.py:1002`)
has no equivalent branch — and that is the function `pipeline.py:445` calls,
**once per proband with a batch of one**, inside a Python loop that ends
`est[k] = estimate[0]`. So a register proband with no informative relatives
still builds `construct_covmat_from_kinship` over a 1×1 relationship, runs
`correct_positive_definite` (an eigendecomposition), and calls
`pa_estimate_batched`, when for a single non-inbred target the whole answer is
`E[g|D] = h2*m` and `Var(g|D) = h2(1-h2) + h2²v`.

I verified this myself over 500 random single-proband intervals at h² = 0.5, one
thread, comparing `estimate_liability_from_kinship` on a 1×1 `A` against the
scalar formula:

- the **estimate** agrees to `max|diff| = 0.0` — bit-identical;
- the **posterior variance** agrees to `max|diff| = 1.11e-16` — one ulp, and
  therefore **not** bit-identical.

That distinction is worth stating precisely, because this repository's
changelogs claim "bit-identical where checked" and its regression tests compare
arrays exactly. The reduction is numerics-preserving to machine precision but is
*not* bit-reproducible for `var`, so adopting it requires either an exactness
tolerance in the affected tests or an algebraic rearrangement that reproduces
the current rounding. Indicative cost, contended: 207 µs per call on the kinship
route against 1.6 µs on the scalar route, ~132×; a directed profiling probe
measured 138× independently. I quote the ratio and not the absolutes, for the
reason given in §1.

Recommendation: add the scalar branch to `estimate_liability_from_kinship` for
`n == 1`, no mixture, scalar `0 < h2 < 1`, mirroring the existing
`_pa_from_role_arrays` dispatch. This is the same shape of gap the v0.7.3 capsule
left open when it retained the register covariance checks — "a trusted internal
path may eventually avoid repeated work" — except that here the trusted path is
an algebraic identity for a 1×1 system, not a waiver of validation, so the
objection that justified retaining those checks does not apply.

**T1-4. The PA object path spends 89–93% of its time in Python marshalling, and
walks every member of every family three separate times to do it.**

The Gibbs engine is *kernel-bound* on documented workloads — at defaults,
F=2000, one thread, a 69.5 s run leaves orchestration at ≈0.1% and `rounds=1`.
The PA object path is the opposite, and there is no crossover to find: the Numba
kernel's share never exceeded **10.6%** at any cohort size tested.

| cohort | total | `_stack_object_members` | `_check_unique_roles` | `_group_by_structure` | Numba kernel |
|---|---:|---:|---:|---:|---:|
| small, n_sim=20000 | 111.1 ms | 47.7% | 26.8% | 11.2% | **8.1%** |
| large, n_sim=20000 | 180.1 ms | 48.2% | 27.4% | 8.6% | **10.6%** |

I verified the structural cause directly. Three functions each walk all members
of all families, and all three are called in sequence at every estimator entry
point (`estimate.py:559/573/579`, `616/646/651`, `672/682/684`, `720/736`):
`_check_unique_roles` (`:413`) builds a role list plus pid keys,
`_group_by_structure` (`:513`) builds a sorted role tuple per family, and
`_stack_object_members` (`:915`) builds a per-family dict plus 2·F·k NumPy scalar
setitems. `cProfile` corroborates: the top four `tottime` entries are
`_scalar_member_bounds` (80,000 calls), `_stack_object_members`,
`_check_unique_roles` and `_pid_key` (80,000 calls).

A directed probe fused the third walk only — iterating `fam.members` against a
role→column map, appending to flat Python lists, one `np.array(...).reshape`,
falling back to `_scalar_member_bounds` whenever a bound is not already a
`float` so the error contract is preserved — and measured **1.38× / 1.34×
end-to-end** (27.6% / 25.4% saved) with `est` and `var` **bit-identical**. Fusing
the remaining two walks into the same pass is the larger prize. Numerics-preserving.

One caveat the probe flagged and I pass on: its fused prototype returns `None`
for the mixture arrays, which matches the original only when
`use_mixture=False`. The mixture path is untested and must fall through to the
existing implementation.

**T2-1. The capsule that could have backed v0.7.2 is excluded by `.gitignore`,
and measures something else.**

`git check-ignore -v tmp/lean-review/results.json` returns
`.gitignore:27: tmp/`, and `git ls-files tmp` is empty. The v0.7.2
before/after capsule — baseline `8926f25`, candidate patch, baseline and
candidate time and memory JSONs, `measure.py`, `change.patch`, build and test
logs — is not in the repository at all.

It would not have supported T1-2's numbers in any case. What it actually
measures is chunked PA at n=1,000,000 (peak allocation 30.59 → 15.49 MiB, median
0.0487 → 0.0469 s) and chunked Gibbs at n=50,000 (2.85 → 1.66 MiB), and its own
`results.json` states: *"Similar runtime in these short local probes; no general
speedup claim."* That is a well-scoped, honest capsule about **allocation**, not
about the CIP-validation or seed-bookkeeping runtimes the changelog quotes. Its
scope note is more careful than the changelog entry it was presumably written
for.

So v0.7.2 has a real measurement that is untracked, and a published measurement
that is unsupported — the two do not overlap.

Recommendation: move the capsule to `benchmarks/results/2026-09-23-lean-v072/`
where the other capsules live and the `.gitignore` does not reach it, and link
it from `benchmarks/README.md`'s "Time and memory between versions" section. If
it is judged a development artifact not worth retaining, say so in the v0.7.2
changelog entry rather than leaving numbers that point nowhere.

**T2-2. `check_evidence.py` stops at v0.6.1, so three releases of efficiency
claims are unreconciled.**

`scripts/check_time_memory_rerun()` hardcodes a single capsule at
`scripts/check_evidence.py:363`:

```python
capsule = ROOT / "benchmarks/results/2026-09-09-time-memory-v061-rerun"
```

with `baseline_revision == "52dec529..."` and `candidate_revision == "b5162712..."`
asserted. Everything after v0.6.1 is outside its scope. Specifically not
reconciled: the v0.7.1 capsule and `RESULTS.md` §31a; the
`2026-09-24-review` capsule; `pa_array_speedup` and `fam_per_s_pa_array` (so
T1-1's numbers are unpinned everywhere); every CHANGELOG Performance bullet; and
the 987 probands/s in §21 and `docs/api.md:146`.

The checker's docstring claims "Prose elsewhere links to the ledger instead of
repeating its numbers". That invariant is what makes the checker's narrow scope
safe — and T1-1 and T1-2 are both counterexamples to it. The guard and the
invariant have come apart.

Recommendation: generalise `check_time_memory_rerun` to iterate over every
directory in `benchmarks/results/` that contains a `results.json`, asserting
each one's internal hash and thread provenance, rather than naming one. Then
either extend coverage to the doc prose that repeats ledger numbers, or remove
those repetitions so the docstring becomes true again.

**T2-3. v0.7.3 shipped five efficiency changes and has no driver capsule;
`benchmarks/README.md` never mentions the capsule it does have.**

The "Time and memory between versions" section of `benchmarks/README.md:88-91`
links exactly two comparisons: v0.7.1 ↔ v0.7.0 and v0.6.1 ↔ v0.6.0. It does
disclose that these are "historical source-bound measurements, not timings of
the current tree", so the staleness is honestly labelled and is not itself the
finding.

The finding is that `benchmarks/results/2026-09-24-review/` — the only v0.7.3
efficiency evidence, and a good capsule with hash-verified before/after sources
— is reachable **only** from `CHANGELOG.md:34`. The README that tells a
maintainer how to run and how to read a time/memory campaign does not mention
it. And no seven-case `bench_time_memory.py` capsule spans v0.7.2 → v0.7.3,
even though v0.7.3 changed quadrature moment compilation, bound-row
deduplication, pid normalisation and added a new simulation backend.

Note also that the 2026-09-24 capsule is a *microbenchmark* capsule (its own
README: "workload-specific results on a shared host, not host-isolated speed
rankings"), not a driver capsule. It is the right instrument for what it
measured and the wrong one for answering "what did v0.7.3 do to end-to-end
register scoring?"

Recommendation: add the 2026-09-24 capsule to `benchmarks/README.md`'s version
list with its scope stated, and run one `bench_time_memory.py` capsule across
v0.7.2 → v0.7.3 before the next release so the driver series stays unbroken.

**T2-4. `bootstrap_fit` is strictly serial, and its replicates are independent.**
`fit.py:982` loops `for b in range(n_boot)` with no pool; a grep for
`ProcessPool|ThreadPool|joblib|Pool(|prange|parallel` across `fit.py` returns
nothing, which I confirmed. Wall time is linear in `B` with no redundancy —
per-replicate cost is 0.86–0.89× a standalone fit at B = 8/16/32, below 1.0
because resampling with replacement merges duplicate families into one structure
group. A directed probe measured 3.9× at ten threads with `max|diff| = 0.0e+00`
across three reps. The sampler already looks thread-safe by construction:
`_advance_rng_state = threading.local()` (`gibbs.py:49`), `_seed_rng` reseeds per
fit, and `gibbs_advance` draws all uniforms in NumPy *before* its `prange`, so
the kernel itself is random-free. I verified the structural claim (no pool) but
did not re-run the parallel timing, so the 3.9× is probe-measured and contended;
the bit-identity claim is the part that would need confirming on a quiet machine
before it is relied on, since T1-3 shows how easily an exact-equality claim can
be one ulp off.

**T2-5. `fit_pairwise_multi` is the only fitter still running the within-family
pid pass twice.** v0.7.3 added `check_pids=False` to the fitters to stop this.
Verified by grep — the call sites are `fit.py:573`, `fit.py:876` and
`pairwise.py:248` with `check_pids=False`, and `pairwise_multi.py:344` with the
default `check_pids=True`. `_assert_nonoverlapping_pids` (`fit.py:345`) then
redoes the within-family uniqueness check with its own `_pid_key` map. The probe
measured 14.2 ms with the pid pass against 1.8 ms without, making the duplicate
43% of the two-call pid prelude. The fix is one keyword, exactly as applied to
the other three. Numerics-preserving for accepted input; for *rejected* input the
raised message becomes the overlap-checker's, which is what the other four
fitters already emit, so any test asserting the `_check_unique_roles` wording
needs updating.

**T2-6. Three separate O(members × traits) Python scans of the same bounds.**
`_member_bounds` (`fit.py:222`) walks every member and materialises `lo`/`hi`
through a list comprehension. `_assert_common_thresholds`
(`pairwise_multi.py:163`) and `_assert_population_case_rate` (`fit.py:222`'s
caller) each invoke it, and `_prepare_multi_pairs` (`pairwise_multi.py:344`
region) then re-derives the identical `np.isfinite(lower)`/`isfinite(upper)`
masks per member in its own loop. Probe measurements on a 3000-family, 5-member,
2-trait cohort: 12.24 + 14.11 + 48.55 ms = **45.7%** of a 163.7 ms
`fit_pairwise_multi`. Sharing one stacked `(n_members, n_pheno)` bound array
across all three consumers is numerics-preserving provided the call order, and
therefore the exception order, is unchanged. The `_prepare_multi_pairs` loop is
also vectorisable across families sharing a structure, since `cases`, `observed`
and `thresholds` are pure elementwise functions of the stacked bounds.

**T2-7. A safety bisection costs more than the optimisation it guards, and is
repaid per bootstrap replicate.** `_probability_limits` (`pairwise_multi.py:213`,
`:365`) runs a 32-step bisection per sign per distinct threshold pair, each step
a full `_bivariate_probabilities` (two adaptive `quad` calls). It is cached only
*within* one call, so `bootstrap_fit` repays it B+1 times: 48.13 ms, **22.3%** of
a 215.6 ms fit, against 4.2 ms for all twelve criterion evaluations — the
optimiser is ~2% of the fit. Memoising across fits on
`(float(t1), float(t2), eps)` is numerics-preserving, since the returned limits
are a deterministic function of those keys. To be clear about the framing: this
is not waste, it is a guard, and the guard should stay. The finding is only that
its lifetime is one call when its inputs outlive the call.

**T2-8. The documented default fitting route costs ~196× the deterministic one,
and the documentation gives no cost signal.** Recommendation 5 of the 2026-09-05
review landed as `fit_pairwise`/`fit_pairwise_multi`, but the sampling fitter
still defaults to `n_iter=1500, inner_sweeps=5` (`fit.py:486`) — 3× the
repository's own benchmark controls (`bench_fit_heritability.py` uses 500/150).
On one matched 3000-family cohort the probe measured: default 11.06 s, benchmark
controls 4.02 s, `fit_pairwise(components=("A",))` 0.056 s — **196× and 71×**.
Scaling in `n_iter` is clean and linear (7.36–7.81 ms per iteration from 125 to
1000), and h² has already settled by `n_iter=250` (0.5384 → 0.5378 at 1000), so
the extra 1250 default iterations buy Monte-Carlo SE rather than estimate.
`docs/guide.md:75` Table 2 selects fitters purely on statistical purpose and
carries no cost column; `docs/inference.md:134` says only that `fit_pairwise`
"samples no latent liabilities".

The cheap route is not a rough approximation, which is what makes this worth
stating: `REVIEW_2026-09c.md` Table 4 (11,364 replicates, 44M families) puts
`fit_pairwise`'s asymptotic bias for A at +0.00005 with SE/SD ≈ 1.00 and 95%
coverage 0.940–0.953, and it returns a genuine cluster-sandwich SE where
`FitResult.h2_se` is documented as a within-dataset MC diagnostic that "can
substantially understate" sampling variability. This is **not**
numerics-preserving — different estimator, h² 0.5306 against 0.5376 on a truth of
0.5 — so it is a default-and-documentation finding, not a code defect. I am not
proposing that `fit_pairwise` replace `fit_heritability`; only that the guide
state the two-orders-of-magnitude cost difference and that 1500 is 3× the value
the package's own benchmarks use.

**T2-9. A single-family PA call pays 28–67 µs of fixed NumPy dispatch for a
kernel that costs 0.8–1.4 µs.** This is the register path's specific problem, and
it compounds T1-3: `pipeline.py:404` calls PA with `F=1` per proband, so neither
batching nor deduplication is available — the probe confirmed **0 of 160 probands
share an extracted degree-3 pedigree**.

Medians of 300 calls, one thread: `_pa_reduced_nomix(F=1)` costs 28.3/28.3/28.1 µs
with no pin and 55.3/60.4/56.3 µs with one pin at d=10/20/40, against
`_pa_family_nomix` at 0.83/0.83/1.13 µs. The overhead is ~12–15 tiny
NumPy/LAPACK dispatches that are **constant in d** — one `np.ix_` is 2.1 µs and a
1×1 `cholesky` 2.7 µs, which together already exceed the entire kernel. A safe
`F==1` branch recovers **1.19–1.24×**, verified bit-identical on 160 random
cases spanning d=6–40 with pins and intervals. The remaining ~30 µs is
`_condition_pins`, whose support and compatibility checks the probe explicitly
does **not** propose removing — correctly, since those are the checks that make
the pinned conditioning safe. Numerics-preserving.

**T2-10. The register path still pays three eigendecompositions per proband —
the count the 2026-09-05 review recorded, unchanged — and 17.6% of degree-3 wall
time is removable bit-identically.** This is the measurement the v0.7.3 capsule
asked for when it retained the covariance checks and left the door open.

Counting `np.linalg` calls through the driver gives exactly **3.000 `eigvalsh`
per proband** (1200 calls over 400 probands). I verified the three sites: the
relationship-PSD check on `A` at `covariance.py:681` — which also re-symmetrises
an already-symmetrised matrix via `0.5 * (A + A.T)`, allocating a third n×n copy;
the positive-definiteness repair gate in `correct_positive_definite`
(`covariance.py:734`, with `:738` and `:742` on the repair path), invoked from
`estimate.py:1077`; and the PA input gate at `pearson_aitken.py:570`, **on the
same array the repair gate has just certified**.

Cumulative ablation, 11 interleaved reps with a duplicate control arm at 0.998×,
outputs hash-verified identical (`est` SHA-256 equal, `max|Δvar| = 0`): PA gate
3.0%, plus relationship gate 8.8%, plus repair gate **17.6%** — 739.8 → 608.2 ms
per 400 probands, 1849 → 1521 µs/proband, 1.216×. At degree 1 it is only ≈2%.

The capsule's objection was that "merely removing these checks is not justified by
the review's timing percentage", and that objection is fair — so this is not a
proposal to remove them. It is a proposal to *certify once and trust internally*,
with every public signature and default unchanged:

1. a private `covariance._kinship_A(sire, dam, order)` taking integer indices,
   which also removes T2-12;
2. `construct_covmat_from_kinship(..., _certified_psd=<module-private sentinel>)`
   skipping the O(n²) checks and `eigvalsh` when the caller produced `A` from a
   pedigree, where it is PSD by construction;
3. `pa_estimate_batched(..., _psd_certified=…)` — this one is safe
   unconditionally and needs no proof, because `correct_positive_definite`
   already returns min-eig > 1e-8, strictly stronger than the PA gate's
   ≥ −1e-10·scale;
4. skipping the repair gate only where the shipped `_covariance_reduction_is_safe(h2, m)`
   proof — already computed at `pipeline.py:421` — gives min-eig > 1e-6, i.e. 100×
   the 1e-8 repair threshold.

Both constraints the 2026-09-05 review attached to any reduction hold: pedigree
extraction/closure and the selected-submatrix reduction are untouched (only
*certification* changes), and no repair is altered — it is skipped exactly where
it provably returns its input unchanged, with the `h2`-boundary fallback still
repairing. **One gap must be closed before landing step 4**: that bound is proved
for the full pedigree covariance, so it has to be re-derived for the
`g`-prepended (n+1)×(n+1) matrix; the sub-block follows by Cauchy interlacing,
but that argument is not written down anywhere yet. Steps 1–3 need no new proof.
Numerics-preserving.

**T2-11. The dense-versus-selected kinship rule is calibrated in the wrong
variable.** `pipeline.py:432-433` chooses dense when the selected pairs exceed
`2m`, which is *linear* in pedigree size. The measured crossover is *quadratic*:
dense costs ≈ c₁·m + c₂·m² (dominated by `_parent_links` plus the O(m²) fill)
while warm-selected costs ≈0.6 µs per requested pair, so the true test is
`pairs ≳ 0.02·m²`, i.e. `n_sel ≳ 0.25·m`. The shipped rule implies
`n_sel ≈ 2√m`; the two agree only near m ≈ 156, so the driver switches to dense
about 3.5× too early for small pedigrees and too late for large ones.

On a 20-point (m, n_sel) grid with 5 warm reps and routes verified
`array_equal`, **7 of 20 cells are mis-selected**, the worst costing **4.07× per
proband** (m=765, n_sel=63: dense 10.63 ms against selected 2.61 ms). Replacing
the constant 2 with `pairs > 0.03·m²` fits 18/20 cells; the two misses are within
1.25× and cost under 1.5 ms.

Two honest limits on this finding. The rule is right where it matters most — on
the degree-3 register, forcing always-selected is 4.42× *slower*. And the
end-to-end cost on the standard register is small (degree 1: always-selected
323.9 ms against heuristic 331.7 ms, 1.024×; degrees 2 and 3 the heuristic is
correct). So this is a calibration fix for deep-closure pedigrees, not a headline
win. The 1500-member cap is **not** a defect: in the one cap-binding case built
(m=1805, n_sel=203) the refused dense route is 2.05× slower than the selected
route the cap forces. Numerics-preserving.

**T2-12. `_parent_links` re-derives an id→index map that the parent graph
already holds.** `covariance.py:534`, reached from `pipeline.py:434`, rebuilds
the mapping per proband, while `extract_pedigree` has just converted indices to
ids for it. Measured at **23.0% of `kinship_from_pedigree`** (120.2 of 522.0 µs
per proband at m=121.4, 20 reps) and **5.70% of degree-3 driver wall time**
(40.9 of 717 ms). The round trip — indices → ids → indices — is the whole cost,
and step 1 of the T2-10 design removes it. Numerics-preserving.

**T2-13. `construct_covmat_multi` fills the multi-trait covariance entry by
entry.** `covariance.py:451-470` makes `n_pheno²·k²` calls to `get_relatedness`,
each costing ~1.0–1.35 µs because it re-runs two `_VALID` fullmatches plus
`_classify` regexes, and the same role pair is recomputed for every phenotype
pair. One k×k relatedness table suffices: since
`get_relatedness(ra, rb, h2) == frac(ra, rb) * h2` with `frac * 1.0` exact,
building the fraction table once and scaling per phenotype pair reproduces the
matrix **element-for-element** — verified with `np.array_equal`, not `allclose`.
On the default family (k=9), 20 reps: 318.0/779.3/1671.3 µs become
41.6/55.4/90.2 µs, i.e. **7.6× / 14.1× / 18.5×** at n_traits 2/3/5, and 33.6× at
k=22 with n_traits=5.

Severity is conditional and I want to be precise about why: `_estimate_liability_multi`
builds this once per structure group (`estimate.py:738`), not once per family, so
the win only reaches a caller sitting in a resampling loop — a bootstrap over
multi-trait fits, say. Numerics-preserving.

**T3-1. v0.7.1's headline multipliers have no committed artifact.**
`CHANGELOG.md:148-159` claims `kinship_from_pedigree` is "25x faster on a
2,683-person register", the register driver gains "4x throughput on the
degree-3 register benchmark", and `simulate_under_LTM_single(use_age=True)` is
"4x faster". `bench_pedigree_inference.csv` has 22 metric rows and no
kinship-timing column; the 2,683-person population at `RESULTS.md:859` supports
*exactness* and *payoff* of the selected-pair route, not its speed. The v0.7.1
capsule covers parent-graph construction, PA batches and register scoring, which
is adjacent but is not these three claims. Lower severity than T1-2 only because
v0.7.1 does link a capsule for the release as a whole.

**T3-2. The external 2026-09-05 review is orphaned.**
`grep -rn "ltpred-efficiency-review"` over the repository returns **zero** hits.
The design rationale for `_selected_kinship.py`, `pairwise.py` and
`quadrature.py` — including the measured 1.83×–3.40× folds, the parent-factor
quadrature derivation, and the pairwise-probit feasibility probe — is recorded
only in an unshipped sibling directory on one machine, whose generating script
`785e433` deleted.

To be fair to that commit: it is deliberate, reasoned, and disclosed at
`CHANGELOG.md:446-453`, and "development pilot on a modified checkout" is
accurate — the capsule's own `git_status` records 17 modified and 9 untracked
files including `?? ltpred/_selected_kinship.py`. `RESULTS.md:1681-1682` also
states plainly that "No new statistical calibration, Gibbs, quadrature or
fitting comparison was performed in this rerun", which correctly limits what
§31 is evidence for. And I confirmed that **no speed or memory figure
attributable to those three modules survives anywhere in the repository** —
`report/efficient_inference.tex` now carries only analytic statements such as
`O(sQ^2)` at `:118`. So this is *orphaned provenance*, not an unsupported
claim: nothing in the repo asserts a number it cannot back.

That distinction is why this is T3 and not T1. The residual cost is
discoverability — a maintainer reading `_selected_kinship.py` cannot find out
why the design looks the way it does. Recommendation: a one-line pointer in
`benchmarks/README.md` or the module docstring naming the 2026-09-05 review and
noting that its capsule is retained outside the repository, or vendor the
`REVIEW.md` into `docs/reviews/` as a sixth chain entry.

**T3-3. `bench_tetrachoric.py`'s console output still says "the same families".**
`benchmarks/bench_tetrachoric.py:100` prints `fit_heritability on the same
families`, while `:80` passes `sim.families[:4000]` out of `N_FAM = 20_000`
(`:40`). The ledger was corrected — `RESULTS.md:976-978` now says "4,000-family
subsets … this SD comparison does not establish greater statistical efficiency
at equal sample size" — but the script's stdout label was not, so anyone running
it sees the unqualified claim. This is the surviving half of a limitation the
2026-09-05 review also recorded. One-word fix.

**T3-4. The v0.7.3 capsule records `threads: 1` but not the BLAS/OMP variables
or a load average.** All 12 JSONs in `benchmarks/results/2026-09-24-review/`
carry `threads: 1`; none records `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` or `NUMEXPR_NUM_THREADS`, and none
records the load average — both of which the driver capsules do capture, and
both of which `benchmarks/README.md` asks for. Its README does state "BLAS,
OpenMP and Numba were limited to one thread" in prose, so the information is
present but not machine-readable.

**T3-5. Two releases in this scope delta have no git tag.** `v0.7.1` and `v0.7.2`
are tagged; `v0.7.0` and `v0.7.3` are not. This has a concrete cost already:
`RESULTS.md:943` attributes "276 probands/s" to v0.7.0, which is substantively
correct — `git show ac78ba36:benchmarks/bench_register_pipeline.csv` gives
275.741 and that tree's `__version__` is `0.7.0` — but the number is recoverable
only by git archaeology, since the retained CSV now holds 987 (manifest row 12,
commit `5b42b139`). Tagging releases would let a cited historical number be
checked against a ref rather than a commit hash dug out of a manifest.

**T3-6. A negative result worth recording: the per-step covariance rebuild is
real but does not matter.** The central hypothesis going into this audit was
that a fitter optimising `sigma = residual*I + Σ h²_c K_c` must be refactoring an
invariant eigenbasis on every likelihood evaluation. The premise is correct —
`gibbs_params(sigma)` (`fit.py:778`) runs `_validate_covmat`, which does a full
`eigvalsh` purely as a positive-definiteness check, then `np.linalg.inv`, once
per outer iteration. And for the single-component case `A = QΛQ'` is fixed, so
reuse is available: the probe implemented it and measured **9.30× faster on that
stage** with `max|P_eig − P_direct| = 1.39e-16`.

The conclusion is nevertheless **do not make this change.** The stage is
0.0709 ms of a 6.06 ms iteration — **1.17%** — because `gibbs_advance` is 98.1%
of the iteration (5.94 ms; 22.5 million truncated draws at default controls). The
whole-fit prize is about 1%. The share grows only with the *number of distinct
family structures*, since per-group per-iteration cost is flat at ~21–48 µs:
0.7% at one group, 5.4% at twenty, 12.6% at fifty, 17.5% at 112 groups. One
hundred and twelve structures is beyond any documented cohort. It would also not
be bit-reproducible: a 1e-16 perturbation of `P`/`sd` reshuffles every downstream
truncated draw, so the seeded trace diverges — precisely the class of change the
v0.7.3 ledger rejected for shared Gibbs estimates. Two sub-notes, both below the
threshold for action: the `eigvalsh` check is provably redundant here
(`residual ≥ eps > 0` and each `K_c` is already PSD-verified in
`_component_matrix`) and skipping it *would* be bit-exact, but it is 4.9% of
1.17%, i.e. 0.06% of the fit.

**T3-7. The pairwise criterion builds a Hessian and per-cell scores that the
optimiser discards.** `_multi_criterion` (`pairwise_multi.py:268-269`) accumulates
`hessian += curvature * np.outer(row, row)` and appends `cell_scores` on every
call, and the caller takes only `(value, gradient, _, _)`; `pairwise.py:160-161`
has the same shape. Measured waste is 20.4% of one criterion call but **0.4% of
the fit** (0.78 ms of 215.6 ms over eleven discarded calls), because `quad` is
77% of a criterion call. Recorded for completeness; not worth restructuring.

**T3-8. Negative: the workqueue serialisation does *not* neuter the parallel
path.** Commit `afcb5ec` was a plausible suspect, so it was tested directly.
Nothing is serialised inside a kernel — the lock wraps each *launch*, is taken
only when `numba.threading_layer() == "workqueue"` (plus once before the layer is
fixed), and costs **160 ns per launch**, not per family (`_numba.py:35,46`). With
the threading layer forced to `workqueue` and the lock therefore active, scaling
survives: 1426.0 / 654.5 / 390.8 ms at 1/4/8 threads = **2.18× at four, 3.65× at
eight**. Under the `omp` layer default on this host the lock is not taken at all
(66 ns/launch) and scaling is 2.55×/4.63×. `benchmarks/README.md`'s warning that
"Gibbs is parallel while the PA object path is largely serial" remains accurate —
and T1-4 explains the second half of it.

**T3-9. Negative, and a change this review recommends against: PA bound-row
deduplication would make the path slower.** The v0.7.3 quadrature dedup was an
80.66× win, so the obvious question is whether the PA path has the same
opportunity. The duplication is there and it is extreme — 20,000 families reduce
to **16** distinct bound rows on the small cohort and **125** on the large one.
But a bit-identical dedup prototype measured **0.91× / 0.86×**, i.e. 9.9% and
15.8% *slower*, because `np.unique(..., axis=0)` costs more than the 9.0/19.0 ms
of kernel it removes. Quadrature pays because its `_family` evaluation is
expensive; a PA row costs ~0.45 µs. **Do not implement this.** It is recorded
because the analogy with the quadrature win is persuasive and the measurement
contradicts it.

**T3-10. Negative: the per-family Python convergence loop is fully amortised at
documented defaults.** A bit-identical vectorised drop-in saves 6.9% at a
550-sweep work unit (F=2000) but **−3%**, i.e. noise, at documented defaults,
where `rounds=1` and the loop is ~25 ms against 69.5 s. The crossover is around
50 sweeps per family. Not worth changing at default controls.

**T3-11. Negative: allocation pressure is not a defect.** tracemalloc at
F=20,000 gives a PA peak of 5.28 MiB (**277 B/family**) and Gibbs 10.70 MiB
(**561 B/family**). The kernel-side temporaries — seven per family in the
collapsed kernel, `cov.copy()` plus `mu` in PA — are per-`prange`-iteration and
cannot be hoisted without per-thread scratch. These are not comparable to the
lean-review capsule's PA figures, whose arm was the single-role
`_adult_full_moments` path rather than the general object path.

**T3-12. `kinship_cache_size=0` is a documented public option with no guard, and
it is 2,900–6,000× slower than either alternative.** `pipeline.py:158` exposes
`kinship_cache_size: int = 100_000` and `:194` documents it as bounding "the
number of ancestor-pair results reused". `_selected_kinship.py:19-21` rejects
non-integers and negatives, so **zero is accepted** — and with the cache off, the
continuation-stack recursion is O(depth) *per requested pair* with no reuse. One
203-member request on a pedigree of m=1805 at depth 1600 took **79,409 ms cold**,
against 13.2 ms warm and 27.0 ms for the dense route.

The bounded cache is a genuine strength (T2-10's design depends on it), which is
why the unguarded zero is worth a line: a user reading "set to 0 to save memory"
gets a 6,000× slowdown rather than an error or a warning. Recommendation: either
reject zero, or clamp it to a small positive floor with a warning, or document
that zero disables reuse entirely and is not a memory-saving setting.

**T3-13. `gibbs_chunked` materialises the full O(F) seed array, while its own
sibling does not.** `chunked.py:151` calls `_base_seeds(seed, F, max_rounds)` on
the whole cohort; `chunked.py:235`, in the `*_batches` sibling, calls
`_base_seeds(seed, n, max_rounds, start=start)` per batch. I verified both call
sites. The cost is 8.0 / 80.0 / 400.0 MB and 6.5 / 82.3 / 460.9 ms at
F = 1M / 10M / 50M. This is the one place where the module's bounded-memory
promise does not hold, and the fix is already written 84 lines below it.

**T3-14. `_SelectedKinship.__init__` is eager and O(population) even when no
proband uses it.** `pipeline.py:301` constructs it unconditionally; the
constructor runs a full topological rank over every person in the population
(`_selected_kinship.py:26-42`). Measured at 3.5 / 35.3 / 132.8 / 403.2 ms for
populations of 12k / 100k / 297k / 744k (5 reps, linear at 0.54 µs/person) — and
at degree 3 the measured cache occupancy is **0 entries**, because every proband
takes the dense route. So on the workload the register driver is optimised for,
this is pure setup cost. Lazy construction on first selected-route use would
remove it. Numerics-preserving.

**T3-15. The kinship cache is bounded and correctly sized, but not compact.**
Confirming and quantifying §2's note: entries cost **186.8 / 276.4 / 298.9 bytes**
at 10k / 100k / 1M entries, roughly 18× an array-backed store, and the cap holds
exactly (10,000 → 10,000 entries; 100,000 → 100,000). On real registers actual
occupancy is only 2,620–16,884 entries, i.e. **0.7–4.5 MiB**, so the 18× is
immaterial in practice. The default cap of 100,000 is also near-optimal: on a
saturating 300-generation, 3,000-proband workload, cap 10,000 is 1.16× slower and
cap 1,000,000 is 1.14× slower while holding 553,422 entries and +100 MiB. No
action recommended; recorded so the "compact storage" half of the 2026-09-05
advice is answered with a number rather than left open.

## 5. Recommendations, in priority order

1. **Fuse the three member walks in the PA object path (T1-4).** The largest
   aggregate win measured: 1.34–1.38× on the whole path, bit-identical, with the
   kernel at only 8–11% of runtime. Start with `_stack_object_members`
   (`estimate.py:915`), whose fusion is already prototyped and verified; keep the
   `_scalar_member_bounds` fallback for non-`float` bounds so the error contract
   holds, and let the mixture path fall through unchanged.
2. **Certify covariance once and trust it internally (T2-10).** 17.6% of degree-3
   register wall time, bit-identical, and it is the measurement the v0.7.3 capsule
   asked for. Land steps 1–3, which need no new mathematics — the PA gate is
   already strictly weaker than what `correct_positive_definite` guarantees. Hold
   step 4 until the `_covariance_reduction_is_safe` bound is re-derived for the
   `g`-prepended matrix. Step 1 also removes T2-12 for free.
3. **Dispatch the family-free scalar path in the register driver (T1-3).** Add
   the `n == 1`, no-mixture, scalar-`h2` branch to
   `estimate_liability_from_kinship`, mirroring `estimate.py:855`. The largest
   per-call ratio measured (~132× per family-free proband) and numerics-preserving
   — but confirm the one-ulp variance difference against the exact-equality tests
   before claiming bit-identity in a changelog.
4. **Fix `docs/estimation.md:359` (T1-1).** Immediately: delete "current",
   attribute the grid to v0.4.0, and state four threads. Properly: regenerate
   `bench_scaling.csv` at v0.7.3 through `run_benchmark.py` so it carries a
   manifest row, then requote. Correct `report/ltpred_methods.tex:652-654` and
   rebuild the PDF in the same pass.
5. **Resolve v0.7.2's orphaned numbers (T1-2, T2-1).** Either promote
   `tmp/lean-review/` into `benchmarks/results/` and link it, or strip the four
   numbers from `CHANGELOG.md:127-133` and keep the qualitative statements.
6. **Take the four one-line fixes (T2-5, T2-7, T2-9, T3-13).** `check_pids=False`
   at `pairwise_multi.py:344`, matching the other three fitters; a cross-call memo
   on `_probability_limits` keyed by `(float(t1), float(t2), eps)`; an `F==1` fast
   branch in the PA reduction; and `_base_seeds(..., start=start)` in
   `gibbs_chunked`, where the correct call is already written 84 lines below in
   its own sibling. Together these are 43% of the pid prelude, 22.3% of a
   multi-trait fit, 1.19–1.24× on register-style single-family calls and 400 MB at
   F=50M, and none changes a number.
7. **Recalibrate the dense/selected rule and drop the id round trip (T2-11,
   T2-12).** `pairs > 0.03·m²` fits 18 of 20 measured cells against the current
   13; `_parent_links` is 23.0% of `kinship_from_pedigree`. Both are calibration
   and plumbing rather than headline wins — T2-11 costs only 1.024× end-to-end on
   the standard register — so schedule them with T2-10 rather than ahead of it.
8. **Share one stacked bounds array across the three scans (T2-6).** 45.7% of a
   `fit_pairwise_multi` call. Numerics-preserving if call order, and therefore
   exception order, is unchanged.
9. **Build the multi-trait relatedness table once (T2-13).** 7.6–18.5× on
   `construct_covmat_multi`, element-for-element identical. Worth doing when a
   caller sits in a resampling loop; low priority otherwise, since the estimator
   builds it once per structure group.
10. **Give the fitting guide a cost signal (T2-8).** State in `docs/guide.md:75`
    that the sampling and deterministic routes differ by ~2 orders of magnitude on
    a matched cohort, and that `n_iter=1500` is 3× the value the package's own
    benchmarks use. Do not change the default estimator.
11. **Generalise `check_evidence.py` past v0.6.1 (T2-2).** Iterate the
    `benchmarks/results/` capsules instead of naming one; add
    `pa_array_speedup`; then either police the doc prose or stop repeating ledger
    numbers in it, so the docstring's invariant holds.
12. **Restore the driver-capsule series (T2-3).** One `bench_time_memory.py` run
    across v0.7.2 → v0.7.3, and a `benchmarks/README.md` entry for the 2026-09-24
    capsule with its microbenchmark scope stated.
13. **Parallelise `bootstrap_fit` (T2-4)** — but only after re-establishing
    bit-identity on a quiet machine. The 3.9× was measured under contention and
    T1-3 is a reminder that an exact-equality claim needs verifying, not
    assuming.
14. **Guards and polish:** reject or warn on `kinship_cache_size=0` (T3-12, a
    documented option that is 6,000× slower), construct `_SelectedKinship` lazily
    (T3-14, up to 403 ms of setup with measured zero occupancy), link the external
    review (T3-2), the `bench_tetrachoric.py:100` label (T3-3), machine-readable
    thread and load fields in the review-capsule JSONs (T3-4), and tags for v0.7.0
    and v0.7.3 (T3-5).

Items 1–3 and 6–9 make the package faster; 4–5 and 11–12 make its numbers
trustworthy; 10, 13 and 14 are disclosure, guards and hygiene. **Every code
change recommended here is numerics-preserving** — item 3 through an algebraic
identity for a 1×1 system, item 2 through certifying once rather than relaxing a
check, and the rest through removing recomputation of values that do not change.
Item 3 is the only one that is not bit-reproducible, and its one-ulp variance
difference is stated rather than buried.

Six changes this review explicitly recommends **against**, each measured and each
rejected on the evidence rather than on principle: reusing the invariant
eigenbasis inside `gibbs_params` (T3-6 — 9.30× on a stage that is 1.17% of the
iteration, and not bit-reproducible); restructuring the pairwise criterion to
skip the discarded Hessian (T3-7 — 0.4% of the fit); **deduplicating PA bound
rows** (T3-9 — bit-identical but 9.9–15.8% *slower*, despite 20,000 families
collapsing to 16 distinct rows, because `np.unique(axis=0)` costs more than the
kernel it removes); vectorising the per-family Gibbs convergence loop (T3-10 —
noise at documented defaults, paying off only past ~50 sweeps per family);
compacting the kinship cache into an array-backed store (T3-15 — real occupancy
is 0.7–4.5 MiB, and the current default cap is within 1.16× of optimal in both
directions); and exploiting Kronecker structure in the multi-trait covariance,
which cannot be done because **no Kronecker matrix is ever materialised at scale**
— the only `np.kron` in the package is `simulate.py:860-861`, a 10×10 sigma of
800 bytes costing 40.7 µs for all four terms. T3-9 is recorded most emphatically
because the analogy with the v0.7.3 quadrature dedup makes it look obvious.

The efficiency picture itself is good. Three releases of optimisation landed
with bit-identical or explicitly-disclosed numerical changes, the two structural
risks the 2026-09-05 review identified were closed with a bounded cache and an
analytic guard rather than with a bypass, and the measurement discipline in the
retained capsules is stronger than the prose that quotes them. The gap is
between the artifacts and the claims, and it is closable with documentation and
one checker change.

## 6. What this review did not do

It did not re-derive the liability model, the covariance construction, the
Gibbs conditionals, the Pearson–Aitken recursion, the quadrature engine, the CIP
estimators, the fitters, or the tetrachoric and liability-scale helpers; five
audits in the preceding eight weeks did.

It did not complete a full profiling pass, but the three arms it did complete
cover most of the package: the **fitting layer** (T1-3, T2-4 → T2-8, T3-6, T3-7),
the **inference engines** (T1-4, T2-9, T3-8 → T3-11) and the **covariance,
kinship, register and chunking layers** (T2-10 → T2-13, T3-12 → T3-15), including
likelihood-evaluation counts, the per-step covariance rebuild, sandwich and
bootstrap cost, the absence of finite differences, the PA object-path split,
single-family call overhead, thread scaling under the workqueue lock, allocation
per family, eigendecomposition counts through the register driver, the
dense/selected crossover, cache sizing and bounded-memory behaviour.

`ltpred/chunked.py`'s bounded-memory claim **was** verified and it holds, which
closes the gap an earlier draft of this report recorded as open. On PA at
F=2,000,000 with four roles, working-set peak over inputs falls 269.1 MiB
(`arrays`) → 78.3 (chunk 65,536) → 69.0 (4,096) → 67.9 (256); on Gibbs at
F=200,000, 104.6 → 35.7 MiB at the default 4,096, with `est`/`se` SHA-256
identical at every chunk size. The `*_batches` route with a lazy generator peaks
at 210 MiB against 626 MiB for `arrays`, a 3× reduction, bit-identical when
replaying the same data. The defaults are well chosen rather than merely safe: PA
chunk 65,536 was the *fastest* arm measured (1550.5 ms against 1885.4 for
`arrays`), and Gibbs at 4096 costs ≤5%. The chunk=256 cliff (5636 ms, 3.6×) is
97% kernel batch inefficiency — the repeated per-chunk covariance build is only
15.6 µs per chunk, ~3%. Two qualifications belong with that verdict: the one
genuine O(n) accumulation is `np.concatenate` (`chunked.py:200`, `:251-253`),
which transiently doubles outputs (a 108 MiB floor against 64 MiB of outputs) and
is inherent to streaming without a known total; and `gibbs_chunked`'s seed array
is O(F) outright (T3-13).

Also recorded as measured and *not* defects: no Kronecker matrix is materialised
at scale anywhere in the package; factorisation reuse in the array path is
already right (`_condition_pins` runs 15 times for F=100,000, independent of mask
count, and per-family PA cost amortises 84.8 µs at F=1 → 0.61 µs at F=100k, a
139× spread — so the register path's waste is repeated *validation*, not repeated
factorisation); `extract_pedigree` is linear in m with no cliff, and the `str()`
in its sort key (`pedigree.py:198`) is unmeasurable at 0.95–1.07× between string
and integer ids.

One efficiency fact worth surfacing for users rather than as a finding:
`simulate_register_liabilities(method="dense")` is quadratic-to-cubic — 0.41 s and
+287.6 MiB at n=2,983, 10.81 s and +1,533.4 MiB at n=9,100, and 7.1 GB for `A`
alone at n=29,808 — while `method="mendelian"` is linear at 0.10 s and +14.6 MiB
for n=9,100 and 0.41 s and +33.0 MiB for n=29,808. The v0.7.3 release notes
describe `dense` as the default retained for historical seeded draws, which is a
correct and sufficient reason; but the two-orders-of-magnitude memory difference
at n≈10,000 is the kind of thing a user simulating a large register would want
stated at the point of choice. The mendelian path's "bounded ancestor-pair cache"
is the same `_SelectedKinship` cache (`simulate.py:566-590`), default cap 100,000.

The register-scoring profile split *was* re-derived, and the result is worth
stating because it is counter-intuitive. At v0.5.2 the 2026-09-05 review measured
approximately 52% kinship construction, 30% eigenvalue calculations and 8% PA
kernel. At v0.7.3, on 160 probands with one thread and matched buckets, the
equivalent split is **53.0% kinship, 28.5% eigen/covariance preparation, 7.0%
PA-plus-pins** (pure Numba kernel 3.6%), with pipeline Python and CIP at 11.5%,
on a mean pedigree of 254.1; at mean pedigree 84.6 it is 41.0 / 34.3 / 8.2
(kernel 2.2) / 16.4%. **Three optimisation releases did not move the shape.**
Each release made the whole path faster — v0.7.1 alone claims 4× on the degree-3
register — but the distribution of cost across kinship, eigen and kernel is
essentially where it was, so the remaining prize is still structural rather than
kernel-level. The absolute seconds are not comparable to the 2026-09-05 figures
(the probe used its own synthetic CIP curve), only the shape is.

The implementation findings were produced by directed profiling probes writing
only to `/tmp`, using cohorts built from the public simulation API. I
re-verified the load-bearing results independently before filing them, and one
did not survive: the probe reported the family-free scalar reduction as
bit-identical for both estimate and variance, and my own 500-interval check found
the estimate bit-identical but the variance one ulp adrift (T1-3). Every
probe-reported percentage in §4 is a ratio taken within a single contended run,
which is more robust than an absolute, but none of the absolute millisecond
figures should be quoted as a clean measurement.

It executed **no** script under `benchmarks/` except the capsule's own
`probe.py`, and that only with `--output` directed outside the repository. This
was deliberate: `bench_scaling.py:131` and `bench_register_pipeline.py:73`
resolve `CSV_PATH` beside themselves via `os.path.dirname(os.path.abspath(__file__))`,
so running either in place would overwrite committed evidence that
`check_evidence.py`, `RESULTS.md`, `paper/tables/*.tex` and
`report/ltpred_methods.tex` are pinned against. The scaling re-measurement
recommended in §5 item 1 should be run through `run_benchmark.py` with the
source tree clean, not ad hoc.

It did not re-run the statistical benchmark campaigns, the R locks, or the
HAPNEST path. It did not audit the `research/` tree beyond running its test
suite. It did not assess `CITATION.cff`, and it assessed
`report/ltpred_methods.tex` only for the two numbers T1-1 traces into it. Its
measurements were made on one machine and one stack (Table 1), under the
contention limit disclosed in §1, and no absolute wall-clock figure collected
during the audit is quoted as clean.
