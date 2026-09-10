# Independent review of ltpred v0.6.2 (2026-09-10)

A read-only audit of the 0.6.0 inference delta. This review covers the tree at
commit `e9bfa66` (v0.6.2). It follows, and is independent of, the v0.3.4 audit
in `REVIEW_2026-08.md`, the v0.4.2 audit in `REVIEW_2026-09.md`, and the v0.5.1
audit in `REVIEW_2026-09b.md`. The audit itself changed no package code; the
T1-1 and T2-1 findings it reports were then fixed in the same change set that
adds this document, and T3-1 was **withdrawn** after the validation study it
asked for failed to reproduce it (section 4 records the dispositions, including
two corrections to the T1-1 measurements found while fixing it; see the
CHANGELOG for the release entry).

## 1. Scope and method

The previous three audits covered the statistical core and the register driver
through v0.5.1. Version 0.6.0 added an inference engine and a fitter, and
deliberately changed some existing scores:

- `quadrature.py`, a new opt-in engine that integrates at most two parental
  factors for additive nuclear families;
- `pairwise.py` (`fit_pairwise`), a new A/C/M composite-likelihood fitter;
- `_selected_kinship.py`, bounded memoized selected relationship entries, used
  by the register driver behind a new covariance-reduction guard;
- `pearson_aitken.py`, where no-mixture PA now conditions all pins jointly and
  marginalizes uninformative rows before folding intervals — a stated,
  deliberate change to some pinned-family scores.

This review concentrates on that delta. It does not re-derive the liability
model, the covariance construction, the Gibbs conditionals, or the existing
fitters; three audits in the preceding six weeks did, and nothing in the delta
alters them.

Two stages. First, the project's own gates were run (Table 1). Second, every
claim that matters was re-derived by execution against oracles written for this
review, not by the package: a dense linear-Gaussian conditioning plus 1-D
quadrature oracle for posterior moments (exact for at most two non-pin
intervals), a Monte-Carlo oracle drawn directly from the model definition
(`Sigma = h2 A + (1 - h2) I`) with exact pin conditioning, and direct numerical
integration of the bivariate normal for the pairwise cell probabilities. Severity
tiers as before: **T1**, correctness or scientific validity; **T2**, robustness,
evidence integrity, or maintainability; **T3**, style, clarity, or polish.

*Table 1. Baseline checks on the review machine (macOS, Apple silicon; repo venv:
CPython 3.10.20, NumPy 2.2.6, SciPy 1.15.3, Numba 0.67.0; BLAS pinned to one
thread as in CI).*

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest -q` | 800 passed, 60 s |
| Lint | `ruff check .` | clean |
| Evidence guard | `scripts/check_evidence.py` | internally consistent: scaling 392–518×; PA stress floor 0.9991; IPW 0.468/0.521; R locks 6.786 ± 0.079× and 1418 ± 30×; PGS 0.338 / 0.236 / 0.170; time/memory 7 matched cases; PDF v0.6.2 |
| Documented quadrature example | `docs/estimation.md` snippet, verbatim | runs; est 1.036112, var 0.229013, error 1.13e-14, 64 nodes |
| Example script | `examples/vignette.py` | runs end to end; GWAS +1.013217, prediction −0.072034, post-index and closure invariance 0.0 |

## 2. Verdicts by area

**Quadrature engine (`quadrature.py`).** The mathematics is correct and the
answers are accurate to floating-point; the mode-finding step has a hard
failure mode that makes the engine unusable on a few percent of ordinary
inputs. Every reduction was checked by hand and by execution. The analytic
branches are right: a proband pinned under `out="full"` returns the pin with
zero variance; `h2 = 0` returns the prior; the parents-only branch returns
`0.5 h2 E[l_m | interval]` for the genetic target and `h2 - 0.25 h2² (1 - v)`
for its variance, both of which follow from the covariance algebra
(`Var(g_o | l_m in I) = 0.25 Var(g_m | I) + 0.25 h2 + 0.5 h2`); and the
one-remaining-interval branch is the exact bivariate truncated-normal formula.
The Newton mode search is right too: at the returned iterate the gradient
vanishes to 1e-17 and the returned Hessian matches a finite-difference Hessian
of the same objective (Table 2), so my initial suspicion of a sign error in the
`(noise - v) / noise²` term was wrong — it is the correct
`(1 - dt/dm) / s²` with `dt/dm = v/s²`. The importance-weighted tensor grid is
a consistent self-normalized estimator with a valid proposal. End to end, the
estimator is exact to ~1e-15 against the dense oracle on every reduction it
claims to close, and consistent with independent Monte Carlo on the general
multi-interval case (mean z = +0.14, sd 0.87, no |z| > 3 in 78 runs). The defect
is not in the arithmetic but in the stopping rule (finding T1-1).

**PA pin conditioning (`pearson_aitken.py`).** Correct. Where the result is
claimed exact it is exact: all-pins, pin plus one interval, and pins plus two
intervals all reproduce the dense oracle to 0.0 or 1e-17 (Table 2). The
deliberate score change is real and is visible in the example script: the
vignette's GWAS score moved from +1.017439 (v0.5.1) to +1.013217, while the
prediction-mode score is unchanged at −0.072034, which is what one expects from
conditioning pins jointly before the interval fold.

**Pairwise fitter (`pairwise.py`).** Correct. The four cell probabilities match
independent bivariate integration to 1.8e-16 over `t` in [−0.7, 4] and
`rho` in [0.01, 0.99]; the first and second derivatives match
Richardson-extrapolated finite differences (5e-9 and 7.6e-10 relative), which
matters because the optimizer consumes them as analytic scores. The composite
likelihood, the cluster-robust sandwich, and the KKT re-check are standard and
are used consistently on a per-family scale. In a recovery experiment I ran
independently — 40 replicates of 3,000 simulated five-member families with
known `A = 0.30, C = 0.15, M = 0.10` — all three components were recovered
with well-scaled uncertainty (Table 3). This is new evidence for a fitter whose
changelog entry says efficiency and coverage "still require broader
validation"; one small caveat is recorded as T3-1.

**Selected kinship and the register driver.** Correct. `_SelectedKinship`
reproduces `kinship_from_pedigree` exactly for full pedigrees, random subsets,
and caches of 0 through 100,000 entries, including inbred pedigrees (the
full-sib-mating child gets the correct diagonal 1.25) and deliberate
self-mating. The bound in `_covariance_reduction_is_safe` is sound: over three
structures and ten values of `h2`, whenever it claimed safety the selected path
was bit-identical to the legacy full-matrix path, and its claimed eigenvalue
lower bound was never violated (worst sampled margin 20.9×); it correctly
defers at `h2 <= 1e-6` and at `h2 = 1`. The two boundary diagnostics added in
0.5.2 in response to the previous review behave as documented: an all-unresolved
parent column and an integer-id/string-parent mismatch both raise, a 20%
unresolved share scores silently, and `proband_state` classifies all three
proband states while leaving every score unchanged.

**Method dispatch and model boundaries.** Correct and never silent. A requested
`method="quadrature"` is refused rather than quietly downgraded for
multi-trait input, nonzero `c2`/`m2`, the mixture, `h2 = 1`, and arbitrary
kinship matrices.

## 3. Independent verification

*Table 2. Checks re-derived for this review, independent of the package's own
tests. "Oracle" is the dense Gaussian-conditioning reference written for this
review.*

| Check | Result |
|---|---|
| `_mode` gradient at the returned iterate | max abs 5.8e-9 (1-D), 4.3e-8 (2-D narrow); vanishes |
| `_mode` Hessian vs finite differences | max abs deviation 1.3e-5 on a scale of 2.0, at minimum eigenvalue 1.22–2.32 |
| `_grid(n, 1)` weights, `n = 16 … 512` | sum = √(2π) to 1e-14; E[x²] = 1, E[x⁴] = 3, E[x⁶] = 15 to machine precision; the `n = 512` underflow fallback is as accurate as SciPy's own weights |
| `_log_mass` vs direct integration | worst relative error 1.3e-14 over means ±9 and widths 1e-7 … 3 sd; deep tails finite (log mass −454.32 at [30, 31]) |
| `_moments` narrow-window branch | worst relative error 5.2e-12 |
| Quadrature vs oracle, exact reductions (pins, ≤ 2 intervals, both targets) | worst est 1.3e-15, worst var 2.3e-15 over 10 cases |
| Quadrature vs Monte Carlo, general structures | 78 runs, mean z +0.14, sd 0.87, max |z| 2.38, none beyond 3 |
| PA vs oracle, pins + ≤ 2 intervals | max difference 0.0 to 2.8e-17 |
| Pairwise cell probabilities vs bivariate integration | max error 1.8e-16; cells sum to 1 |
| Pairwise derivatives vs finite differences | 5e-9 (first), 7.6e-10 (second, Richardson) |
| `_SelectedKinship` vs `kinship_from_pedigree` | exact for full pedigrees, random subsets, caches 0–100,000, inbred and self-mated pedigrees |
| Register-driver guard vs legacy path | bit-identical where the guard claims safety (30 combinations); bound margin ≥ 20.9× |
| 0.5.2 boundary diagnostics | all-unresolved and dtype-mismatch raise; `proband_state` correct and score-neutral |

*Table 3. `fit_pairwise` recovery, 40 replicates of 3,000 five-member families
(`o, m, f, s1, s2`), common threshold at 10% prevalence, `sampling="population"`.
**Forty replicates give SE(mean) of 0.0068, 0.0054 and 0.0076 for A, C and M, so
the bias column below is not resolvable at the size of the deviations it shows.**
Table 4 repeats the design with 73 times more replicates; read that one.*

| Component | Truth | Mean estimate | Empirical SD | Mean reported SE | SE/SD | 95% coverage |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.300 | 0.2841 | 0.0429 | 0.0474 | 1.11 | 0.97 |
| C | 0.150 | 0.1634 | 0.0344 | 0.0353 | 1.03 | 0.95 |
| M | 0.100 | 0.1056 | 0.0480 | 0.0509 | 1.06 | 1.00 |

*Table 4. The same design and truths at five cohort sizes, 11,367 replicates
totalling 44,005,500 simulated families, root seed 20260910. Replicates are
allocated proportional to 1/N, so SE(bias) is constant across N by construction.
Coverage is of the nominal 95% normal interval; "M at 0" is the share of
replicates with the M component pinned at the non-negativity boundary.*

| N | replicates | bias A | bias C | bias M | SE(bias) A/C/M | SE/SD A | coverage A | M at 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,500 | 5,867 | −0.0006 | −0.0011 | −0.0016 | 0.0009 / 0.0006 / 0.0009 | 1.00 | 0.948 | 10.6% |
| 3,000 | 2,933 | −0.0011 | −0.0004 | −0.0022 | 0.0009 / 0.0006 / 0.0009 | 1.01 | 0.957 | 3.2% |
| 6,000 | 1,467 | −0.0018 | +0.0004 | −0.0016 | 0.0009 / 0.0006 / 0.0009 | 1.00 | 0.956 | 0.5% |
| 12,000 | 733 | −0.0011 | −0.0001 | +0.0001 | 0.0009 / 0.0006 / 0.0009 | 0.92 | 0.926 | 0% |
| 24,000 | 367 | −0.0008 | +0.0003 | −0.0012 | 0.0009 / 0.0007 / 0.0010 | 0.95 | 0.954 | 0% |

No fit failed and none was discarded, at any size. Weighted least squares on
`bias(N) = b0 + b1/N` gives an asymptotic bias `b0` of −0.0013 for A (95% CI
−0.0025 to −0.0001), +0.0004 for C (−0.0005 to +0.0012) and −0.0009 for M
(−0.0021 to +0.0004), with the per-N biases flat rather than decaying, so the
`1/N` term is not identified. Across replicates `r(Â, Ĉ)` is −0.54 to −0.63,
`r(Â, M̂)` is +0.10 to +0.17, and the bias of `Â+Ĉ` is −0.0005 to −0.0017.
The harness is `tmp/pairwise_scaling/` (untracked).

## 4. Findings

**T1-1. `method="quadrature"` refuses ordinary inputs because the mode-search
stopping rule sits below the floating-point noise floor.** This is the one
finding that changes what a user can do; everything else in the delta held up.

`_mode` (`quadrature.py:135`) stops when
`max|step| <= 1e-10 * (1 + max|x|)`. On affected inputs the Newton iterate
reaches the true optimum at the second iteration and then cannot shrink its
step below roughly `3e-10`, so the test can never pass, the loop runs out its
80 iterations, and the estimator raises. Instrumenting the real code path shows
the mechanism exactly: from iteration 2 to iteration 79 the objective is
bit-identical (0.459686633802) and `x` equals SciPy's minimizer to eight
decimals, while `|step|` stalls at 2.8e-10 against a tolerance of 1.13e-10; the
Armijo line search accepts only scales of 1e-8 to 1e-7 because the predicted
decrease `gradient @ step` is roundoff. The reproducibility is exact:

```python
from ltpred.quadrature import estimate_liability_quadrature_arrays as est
est(["o", "m", "f"], [[-1.324, -1.238, -1.508]], [[0.796, 1.200, 0.958]], h2=0.5)
# RuntimeError: family 0: quadrature posterior-mode iteration did not converge
```

Three properties make this T1 rather than a nuisance. First, the failure rate on
plausible input is not negligible: 48 of 720 randomized bound sets (6.7%),
broken down as 7.5% for two-sided intervals, 9.2% for one-sided population
bounds, and 3.3% for the classic LT-FH++ encoding with a pinned proband — the
regime the engine is advertised for. Second, **no public control rescues an
affected family**: `quadrature_atol` and `quadrature_max_nodes` do not reach
`_mode`, and every combination tried (atol from 1e-8 to 1e-2, max_nodes 128 and
512) still raised on all 16 two-sided refusals sampled. Third, because the
estimator raises on the first failing family, the method is unusable at cohort
scale: a batch of 50 realistic families raised at family 30, and a batch of
1,000 raised at family 0. The error text also misdescribes the cause — the mode
iteration did converge.

The code already contains the intended tolerance for this situation in the
line-search `else` branch, whose comment reads "At floating-point resolution the
objective may stop changing before the derivative. This point remains a valid
importance proposal; quadrature refinement still determines acceptance." That
branch is unreachable here, because the line search does find a (tiny) scale.
Recommendation: make the stopping rule noise-aware — accept when the objective
stops changing at representable resolution, or when the accepted line-search
scale falls below a threshold, or relax the tolerance to the gradient's
resolution (about `1e-8 * (1 + max|x|)`) — and treat the mode as a proposal
rather than a result, which is what the importance reweighting already does.
A randomized bound test alongside the existing hand-picked cases would have
caught this; `tests/test_quadrature.py` currently uses only round-number bounds.
The refinement diagnostic did not mask the problem, because the estimator
raises rather than returning a value.

*Two corrections to the measurements above, from re-running them while fixing
the defect.* The incidence is environment-dependent, because the noise floor of
the exact gradient follows SciPy's truncated-moment accuracy: on identical
inputs the two-sided/one-sided refusal rates are 6.2%/9.6% under NumPy 1.26
with SciPy 1.15 but 2.1%/4.2% under NumPy 2.4 with SciPy 1.18, and the minimal
reproduction above returns rather than raising on the latter. The defect is
present in both; only which families trip it changes. Second, the stall is
sharper than "the line search accepts tiny scales": the accepted step changes
the objective by *exactly zero*, because `scale * step` (about 1e-8 x 5e-9)
falls below an ulp of `x`, so `candidate` equals `x` bit-for-bit and the Armijo
threshold underflows to `value`. The iterate therefore cannot move at all,
rather than moving slowly.

**Disposition: fixed.** `_mode` now also returns when an accepted line-search
step fails to change the objective at representable resolution
(`value - candidate_value <= 8 * eps * (1 + |value|)`) — the tolerance the
`else` branch already documented but could not reach. The Newton decrement
`gradient @ step` was measured as the more principled alternative and rejected
on evidence: across 700 traces its stalled values (up to 2.9e-15) overlap the
smallest values seen on genuinely productive iterations (down to 6.4e-20), so
no threshold separates the two states.

Validation. Over 1,200 randomized families spanning four role sets, six bound
styles, six `h2` values and both `out` modes, mode refusals fall from 49 to 0
and no family that previously returned now raises: the 1,031 answered before
are a subset of the 1,076 answered after, 775 of them bit-identical and the
rest differing by at most 8.8e-14 in `est` and 3.0e-13 in `var`, against the
method's own 1e-8 criterion. Four rescued families are genuinely node-limited
and now raise the refinement message instead, which `max_nodes=128` resolves —
an actionable refusal in place of an unrescuable one. On nine previously
refused inputs both moments reproduce an independently written 800x800
Gauss-Legendre integration over the two parental breeding values to within
2.0e-15 (`est`) and 8.9e-16 (`var`), the oracle being self-consistent to 1e-16
under order doubling, and agree with 40M-draw rejection sampling (max |z| =
1.8). Two regression tests were added: one pins the four verified answers, the
other supplies the randomized bound coverage whose absence let this through.
Seven of the eight new test cases fail on the pre-fix tree, and the suite (808
tests) passes under both dependency sets above.

**T2-1. The node-refinement refusal reports a change that already satisfies the
criterion.** The convergence test requires two *consecutive* refinements within
`atol`, but the failure message prints only the last one:

```
quadrature did not converge by 128 nodes per dimension (last change 7.48e-14, atol 1e-08)
```

Here the last change is one hundred million times smaller than the tolerance, so
the message reads as a contradiction; the actual reason is that the previous
refinement exceeded `atol`. Separately, with the default `max_nodes=128` only
three refinement comparisons exist (16→32→64→128), so a family that stabilizes
at 256 nodes is refused; that case is rescued by `max_nodes=512` (error 7.5e-14
at 256 nodes), but the message does not mention the knob. Recommendation: report
`max(changes[-2:])` and name the refinement that failed, and mention
`quadrature_max_nodes` in the message.

**Disposition: fixed.** The refusal now reports `max(changes[-2:])`, the
quantity the criterion actually tests, worded as "largest of the last two
changes". Re-measuring while fixing it relocated the binding case: it is
`max_nodes=64`, the documented *minimum*, where the ladder 16→32→64 leaves only
two changes and so forces the coarse first refinement into the acceptance test.
A deep-tail family at `h2=0.95` refused with `changes = [7.76e-06, 9.58e-09]`
while reporting only 9.58e-09 — and that same family returns at
`max_nodes=128` carrying `error` equal to 9.58e-09, the very number the refusal
had quoted. The two-successive-refinements rule is documented behaviour and was
left alone; only the message changed, plus two sentences in
`docs/estimation.md` recording that both changes must pass and that raising
`quadrature_max_nodes` is the remedy.

**T3-1. WITHDRAWN. The reported negative bias in the `A` component of
`fit_pairwise` was an artifact of using 40 replicates.** The finding as first
written read: mean `A` of 0.284 against a truth of 0.300, about −0.37 empirical
SD with a nominal t near −2.4, offered as an observation for a validation study
to pursue. The validation study was run (Table 4) and does not reproduce it.

At the identical design and cohort size, 2,933 replicates give a bias in `A` of
−0.0011 ± 0.0009 — indistinguishable from zero and fourteen times smaller than
the −0.0159 first reported. The original design was replicated faithfully: the
sampling SDs of Table 3 agree with those of the large study to within 0.7
standard errors of an SD from 40 draws. What was not adequate was the number of
replicates. Forty draws put SE(mean) at 0.0068 for `A`, so the reported
deviation was a 2.3 SE fluctuation of the review's own mean; the `C` deviation
was +2.5 SE in the opposite direction. Two further facts complete the account.
Three components were examined, and the probability that at least one shows
|t| > 2.3 by chance is 0.06. And `Â` and `Ĉ` are negatively correlated across
replicates, at −0.54 to −0.63, so a single under-powered replicate set tends to
show exactly the observed signature of `A` low with `C` high.

The mechanism the finding proposed was also wrong on theory, independently of
the numbers. It suggested that ignoring within-family dependence beyond the
cluster sandwich would bias the additive component. It does not: a pairwise
composite likelihood with correctly specified bivariate margins is consistent
whatever the dependence among pairs, which costs efficiency and corrupts naive
variance estimates but not consistency. There is a real intrinsic bias in this
estimator, and it is negative, but it is `O(1/N)` and far too small to see here.
In the one-component, one-pair-per-family design at threshold zero the composite
MLE is exactly `2 sin(π(p̂ − ½))` in the concordant fraction, which is concave
for `A > 0` since `g''(p) = −π²A`; enumerating the binomial sampling
distribution gives `N × bias → −0.3667`, so about −0.0001 at N = 3,000.

**Disposition: withdrawn, and the study it asked for is now reported.** No
defect is demonstrated. The residual is at most a −0.0013 asymptotic bias in
`A`, under 0.5% of the parameter, marginally separated from zero on 44 million
simulated families and of no practical consequence. The positive result is the
calibration the changelog said was missing: SE/SD between 0.92 and 1.01,
coverage between 0.926 and 0.978, no failed or discarded fits in 11,367
replicates, and boundary pinning confined to `M` — the component nearest zero —
falling from 10.6% at N = 1,500 to nil by N = 12,000. Two cells deserve a
follow-up rather than a claim: at N = 12,000 the `A` interval covered 0.926 with
SE/SD 0.92, both about 3 SE anti-conservative, which is either the expected
outlier among fifteen cells or a genuine small-sample SE effect.

**T3-2. The quadrature documentation could name its practical failure
surface.** `docs/estimation.md` correctly states that failure to meet the
refinement criterion raises and that no unchecked value is returned. Once T1-1
is fixed it would still help to say that the engine is a cross-check on
nuclear-family structures rather than a cohort-scale workhorse, and that
refusals are input-dependent rather than rare.

**Disposition: partly addressed.** The concrete, measured part is now
documented (the two-change acceptance rule and the `max_nodes` remedy, under
T2-1). The broader positioning claim was left out: with mode refusals removed,
the remaining refusals are the documented refinement criterion doing its job,
and characterizing the engine's cohort-scale standing is a judgement for the
maintainer rather than a review finding.

## 5. What this review did not do

It did not re-derive the liability model, the covariance construction, the
Gibbs conditionals, the CIP estimators, the existing fitters, or the tetrachoric
and liability-scale helpers; three audits in the preceding six weeks did, and
the delta does not touch them. It did not re-run the benchmark suite, the R
lock comparisons, or the HAPNEST path; the evidence guard was run instead, and
the 0.6.1 time/memory claims were not recomputed. It did not audit the
`benchmarks/` or `research/` trees. The numerical probes were run on one machine
and one stack (Table 1); the quadrature Monte-Carlo comparisons used 2.5–4
million draws, so agreement is established at the 1e-4 level there and at
1e-15 only where a closed-form oracle applies.
