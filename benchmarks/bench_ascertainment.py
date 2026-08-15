"""What ascertainment does to `fit_heritability`, `fit_variance_components` and `fit_genetic_correlation`.

Every other fitter benchmark here draws **unascertained population families**.
This one measures what the fitters do when that contract is violated -- the
evidence behind two 0.3.1 features, so read the arms as the justification for
them rather than as a description of current behaviour:

* `sampling="population"` used to be an honour system (a string check, not a
  data check), so an ascertained cohort fitted straight through. Arms A-F
  measure what it returned. They therefore run under `unguarded()`, since the
  check they motivated would now stop them.
* `sampling="ipw"` with per-family weights corrects the subset of designs whose
  inclusion probability is known and positive everywhere. Arm G measures that,
  and runs through the real gate.

Design. Each replicate draws a **population** of families under a known
liability-threshold model, applies one ascertainment scheme to choose which
families are analysed, and fits the selected subset. Only the selection rule
changes between arms: the generative model, the analysed sample size, the
thresholds and the fitter settings are held fixed, so the contrast is
attributable to ascertainment alone.

Schemes (selection is on the proband `o` unless stated):

  population          every family, no selection        -- the supported contract
  proband_case        affected proband                  -- classic family study
  case_control        50/50 affected/unaffected probands -- standard GWAS cohort
  enriched_20         20% affected probands             -- registry/biobank enrichment
  family_history      >= 1 affected member ANYWHERE     -- note this includes the
                      proband, so it is a strict superset of proband_case
  fh_proband_control  proband unaffected AND >= 1 affected relative -- the
                      contrast that isolates family-history selection, and the
                      design ltpred's own prediction path targets

Six arms:

  A. h2 recovery under every scheme (`fit_heritability`);
  B. A+C recovery under every scheme (`fit_variance_components`);
  C. r_g recovery (`research.advanced_fitting.fit_genetic_correlation`),
     ascertained on trait 1, which also asks whether a *cross-trait* estimand
     inherits the same bias as the within-trait ones;
  D. bias vs N for population and proband_case. This is the arm that decides
     whether ascertainment is a precision problem or an accuracy problem: a
     sampling-noise problem shrinks as 1/sqrt(N), a bias does not.
  E. the same cohort re-fitted at increasing n_iter AND from several starting
     values, to establish that a displaced estimate is a fixed point rather
     than an unfinished run. n_iter=1500 is amply sufficient at every N
     tested; the multi-start is the decisive half, because an ascertained fit
     reaches the ceiling even when started at h2_init=0.05 -- it climbs there
     from below rather than failing to leave.
  F. the Haseman-Elston moment evaluated directly on the selected families'
     TRUE liabilities, centered and uncentered. Arms A-D can only report
     "pinned at the clamp", which is one bit; this arm gives the uncensored
     magnitude of the selected sample's own moment and splits off the
     mean-shift term. It is a decomposition, NOT the cause -- see `_he_moment`
     for the true-h2=0 cell that refutes the causal reading.

Prevalence defaults to 0.05 rather than the 0.10 the other fitter benchmarks
use, because selection intensity -- and therefore the distortion -- grows as the
disease gets rarer. `--prev` sets it (one value per run; re-run to compare).
Note `case_control`/`enriched_20` can only raise the case share, so a target
below the population prevalence is unreachable and the run says so.

Ground truth is the simulator's own components: h2 is a property of the
population the model defines, and ascertainment changes the sampling
distribution rather than that parameter, so the population value stays the
defensible target under every scheme. Uncertainty is the across-replicate SD
with its own chi-square interval. Two different saturation detectors are
reported because the two fitters pin differently: `boundary_frac` for the
elementwise [eps, 1-eps] clamp (`fit_heritability`), and an exhausted residual
for the simplex renormalisation (`fit_variance_components`) -- see `_saturated`.

The population arm is the control, not a claim of exactness: the fitter carries
its own small bias at these settings, which is why every conclusion below is a
contrast against that arm rather than against zero.

    python benchmarks/bench_ascertainment.py --quick
    python benchmarks/bench_ascertainment.py --reps 10 --n-fam 10000
Writes bench_ascertainment.csv (+ .png if matplotlib is present).
"""

import os
import csv
import sys
import time
import argparse

import contextlib

import numpy as np

from _common import get_plt, sd_ci
from ltpred.covariance import (construct_covmat_multi, correct_positive_definite,
                               get_relatedness)
from ltpred.family import Family, Member
import ltpred.fit as _fit_mod
from ltpred.fit import (_component_matrix, fit_heritability,
                        fit_variance_components)
from ltpred.thresholds import liability_threshold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.advanced_fitting as _af_mod  # noqa: E402
from research.advanced_fitting import fit_genetic_correlation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# `o` is included so there is a proband to ascertain on. Structures are
# sib-rich where C has to be identified (C loads on the full-sib excess).
STRUCTURES = {
    "nuclear": ["o", "m", "f", "s1"],
    "sibship": ["o", "m", "f", "s1", "s2", "s3"],
    "extended": ["o", "m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"],
}

SCHEMES = ("population", "random_50", "proband_case", "case_control",
           "enriched_20", "family_history", "fh_proband_control")

# The case share each enrichment scheme is named for. Only reachable when the
# population prevalence is below it (the schemes thin controls, never cases).
TARGET_CASE_FRAC = {"case_control": 0.5, "enriched_20": 0.2}

# fit_heritability/fit_variance_components clamp to [eps, 1-eps]; a replicate
# within this of either end is reported as pinned rather than estimated.
_EPS = 1e-4
_BOUNDARY_TOL = 1e-3


def _at_boundary(values):
    """Fraction of estimates pinned at the fitter's elementwise [eps, 1-eps] clamp.

    Correct for `fit_heritability`, whose single `h2_hat` is clipped elementwise
    (ltpred/fit.py). See :func:`_saturated` for why it is the WRONG detector for
    the multi-component fit.
    """
    v = np.asarray(values, dtype=float)
    return float(np.mean((v <= _EPS + _BOUNDARY_TOL) | (v >= 1.0 - _EPS - _BOUNDARY_TOL)))


def _saturated(residuals):
    """Fraction of multi-component fits that used up the whole liability variance.

    `fit_variance_components` does not stop at the elementwise clamp: after
    clipping it renormalises onto the simplex when the components would sum past
    `1 - eps`. A runaway fit therefore does **not** sit at `1 - eps` per
    component -- it lands wherever the raw Haseman-Elston solution's *ratio* puts
    it, with the residual squeezed to the floor. Measured under `proband_case`:
    A = 0.5000, C = 0.5000, residual = 0.000100, which `_at_boundary` scores as
    perfectly healthy. The residual is the quantity that actually pins, so it is
    what this checks.
    """
    r = np.asarray(residuals, dtype=float)
    return float(np.mean(r <= _EPS + _BOUNDARY_TOL))


# Data and fitter seeds must not collide: both route to np.random.default_rng, so
# a shared value seeds the augmentation sampler with the very stream that drew
# the liabilities. Arms get disjoint blocks so a cell that appears in two arms
# (N=10,000 is in both A and D) is a fresh replicate, not a byte-identical rerun.
_ARM_OFFSET = {"h2": 0, "ac": 1_000_000, "rg": 2_000_000, "scale": 3_000_000,
               "converge": 4_000_000, "mechanism": 0, "ipw": 5_000_000,
               "dose": 6_000_000, "lee": 7_000_000}


def _seeds(base_seed, arm, rep):
    """`(data_seed, fit_seed)` -- disjoint streams per arm and replicate."""
    base = base_seed + _ARM_OFFSET[arm]
    return base + 1000 * rep, base + 500_000 + rep


def _component_cov(roles, props):
    """Family liability covariance `sum_c props[c] K_c + e2 I` (unit diagonal)."""
    n = len(roles)
    total = sum(props.values())
    if total > 1.0:
        raise ValueError("component proportions exceed 1")
    Sig = (1.0 - total) * np.eye(n)
    for comp, v in props.items():
        if v > 0:
            Sig = Sig + v * _component_matrix(roles, comp)
    Sig, _ = correct_positive_definite(Sig)
    return Sig


def _select(status, roles, scheme, rng):
    """Boolean mask of families this scheme would analyse.

    `status` is `(n_draw, len(roles))` observed case/control. Selection is on the
    proband except for `family_history`. `case_control` and `enriched_20` return
    a mask whose case share is only approximately right per chunk; the caller
    accumulates to the requested count, which is what fixes the realised share.
    """
    o = roles.index("o")
    proband = status[:, o]
    if scheme == "population":
        return np.ones(status.shape[0], dtype=bool)
    if scheme == "random_50":
        # NEGATIVE CONTROL: selects half the population, but independently of
        # phenotype. It exercises the identical chunked accept/reject path,
        # acceptance-rate-driven redraw and sel[:needed] truncation as the
        # phenotype-selected schemes, differing only in what selection depends
        # on. If those mechanics were themselves introducing the artefact this
        # arm would show it too; it must recover h2.
        return rng.random(status.shape[0]) < 0.5
    if scheme == "proband_case":
        return proband
    if scheme == "family_history":
        # NB this includes the proband, so it is a strict SUPERSET of
        # proband_case and is dominated by it in small families. Kept because it
        # is what "ascertained on family history" usually means in practice.
        return status.any(axis=1)
    if scheme == "fh_proband_control":
        # The informative contrast: proband UNAFFECTED but at least one relative
        # affected. This isolates selection on family history from selection on
        # the proband, and is the design ltpred's own prediction path targets
        # (omit/unbind `o`, score from relatives).
        rel = np.delete(status, o, axis=1)
        return (~proband) & rel.any(axis=1)
    if scheme in ("case_control", "enriched_20"):
        target_case_frac = TARGET_CASE_FRAC[scheme]
        # Keep every case; keep controls with the probability that makes the
        # retained case share equal the target. Sampling controls (rather than
        # discarding cases) keeps the case count -- the scarce resource -- intact.
        # NOTE this can only *raise* the case share, so the target is
        # unreachable once population prevalence already exceeds it: at K=0.30,
        # enriched_20 realises 0.299, not 0.200. Harmless at the default K=0.05
        # (verified 0.200 through K=0.20), but silent, so `simulate_ascertained`
        # records the realised share and main() refuses to mislabel a run.
        n_case = int(proband.sum())
        if n_case == 0:
            return np.zeros(status.shape[0], dtype=bool)
        want_control = n_case * (1.0 - target_case_frac) / target_case_frac
        n_control = int((~proband).sum())
        if n_control == 0:
            return proband
        keep_p = min(1.0, want_control / n_control)
        mask = proband.copy()
        ctrl = np.flatnonzero(~proband)
        mask[ctrl[rng.random(ctrl.size) < keep_p]] = True
        return mask
    raise ValueError(f"unknown scheme {scheme!r}")


def _build_families(status, roles, threshold, offset=0):
    """Case/control interval families -- one common threshold, as the fitters require."""
    fams = []
    for i in range(status.shape[0]):
        members = [Member(role=r,
                          lower=(threshold if status[i, j] else -np.inf),
                          upper=(np.inf if status[i, j] else threshold))
                   for j, r in enumerate(roles)]
        fams.append(Family(fam_id=offset + i, members=members))
    return fams


def simulate_ascertained(roles, props, n_fam, prev, scheme, seed, chunk=200_000,
                         max_draws=40_000_000, return_liab=False):
    """`n_fam` families under `scheme`, drawn from a population of the same model.

    Liabilities are drawn and thresholded in bulk and only the *selected* rows
    become `Family` objects, so a scheme keeping 5% of the population costs 20x
    the draws but not 20x the Python objects.

    Returns `(families, stats)` with

      accept_rate  fraction of drawn population families the scheme analyses --
                   the selection intensity, 1.0 for `population`. Measured from
                   the masks, not from `n_fam / n_drawn`, which would just
                   report the chunk size.
      case_frac    realised affected-proband share of the analysed sample. This
                   is the quantity `case_control` / `enriched_20` are *named*
                   for, and the only way to see that a target was met: the
                   schemes can raise the case share but never lower it, so a
                   target below the population prevalence is silently
                   unreachable.
      target_met   None when the scheme has no target, else whether `case_frac`
                   landed within 0.02 of it.
    """
    rng = np.random.default_rng(seed)
    Sig = _component_cov(roles, props)
    t = float(liability_threshold(prev))
    mean = np.zeros(len(roles))
    o_col = roles.index("o")

    kept, kept_liab, n_kept, n_drawn, n_eligible = [], [], 0, 0, 0
    accept_est = 1.0
    while n_kept < n_fam:
        if n_drawn >= max_draws:
            raise RuntimeError(
                f"scheme {scheme!r} at prev={prev} kept only {n_kept}/{n_fam} "
                f"families in {n_drawn} draws")
        # draw roughly what is still needed, inflated by the observed acceptance
        # rate so a rare scheme does not crawl and `population` does not
        # over-draw by an order of magnitude
        needed = n_fam - n_kept
        take = int(min(chunk, max(1024, needed / max(accept_est, 1e-3) * 1.3)))
        liab = rng.multivariate_normal(mean, Sig, size=take)
        status = liab > t
        mask = _select(status, roles, scheme, rng)
        n_drawn += take
        n_eligible += int(mask.sum())
        accept_est = max(n_eligible / n_drawn, 1e-4)
        sel = status[mask][:needed]
        if sel.shape[0] == 0:
            continue
        kept.append(sel)
        if return_liab:
            kept_liab.append(liab[mask][:needed])
        n_kept += sel.shape[0]
    status = np.concatenate(kept, axis=0)
    liab = np.concatenate(kept_liab, axis=0) if return_liab else None
    case_frac = float(status[:, o_col].mean())
    target = TARGET_CASE_FRAC.get(scheme)
    stats = dict(accept_rate=n_eligible / n_drawn, case_frac=case_frac,
                 target_case_frac=target,
                 target_met=None if target is None else abs(case_frac - target) <= 0.02,
                 liab=liab)
    return _build_families(status, roles, t), stats


def simulate_ascertained_multi(roles, h2_vec, rg, rp, n_fam, prev, scheme, seed,
                               chunk=100_000, max_draws=40_000_000):
    """Two-trait counterpart; ascertainment is on **trait 1**'s proband status."""
    rng = np.random.default_rng(seed)
    P = len(h2_vec)
    rg_m = np.array([[1.0, rg], [rg, 1.0]])
    rp_m = np.array([[1.0, rp], [rp, 1.0]])
    cov = construct_covmat_multi(fam_vec=roles, add_ind=True, genetic_corrmat=rg_m,
                                 full_corrmat=rp_m, h2_vec=h2_vec)
    Sig, _ = correct_positive_definite(cov.matrix)
    all_roles = cov.roles
    k = len(all_roles) // P
    # Each per-trait block LEADS with the latent genetic liability `g`, whose
    # variance is h2, not 1. Thresholding that column at `t` would invent a
    # spurious "affected" indicator, which `family_history` (an any-member rule)
    # would then select on. Drop it before thresholding, so both the selection
    # and the emitted members see observed statuses only.
    fam_roles = all_roles[:k]
    obs = [r for r in fam_roles if r != "g"]
    obs_idx = np.array([fam_roles.index(r) for r in obs])
    t = float(liability_threshold(prev))
    mean = np.zeros(len(all_roles))
    o_col = obs.index("o")

    kept, n_kept, n_drawn, n_eligible = [], 0, 0, 0
    accept_est = 1.0
    while n_kept < n_fam:
        if n_drawn >= max_draws:
            raise RuntimeError(f"multi scheme {scheme!r} kept {n_kept}/{n_fam}")
        needed = n_fam - n_kept
        take = int(min(chunk, max(1024, needed / max(accept_est, 1e-3) * 1.3)))
        liab = rng.multivariate_normal(mean, Sig, size=take)
        # (n, P, len(obs)) observed status, latent g excluded; trait 1 selects
        status = np.stack([liab[:, p * k:(p + 1) * k][:, obs_idx] > t
                           for p in range(P)], axis=1)
        trait1 = status[:, 0, :]
        mask = _select(trait1, obs, scheme, rng)
        n_drawn += take
        n_eligible += int(mask.sum())
        accept_est = max(n_eligible / n_drawn, 1e-4)
        sel = status[mask][:needed]
        if sel.shape[0] == 0:
            continue
        kept.append(sel)
        n_kept += sel.shape[0]
    status = np.concatenate(kept, axis=0)

    fams = []
    for i in range(status.shape[0]):
        members = []
        for j, r in enumerate(obs):          # status columns are already obs-order
            lo = [t if status[i, p, j] else -np.inf for p in range(P)]
            hi = [np.inf if status[i, p, j] else t for p in range(P)]
            members.append(Member(role=r, lower=np.array(lo), upper=np.array(hi)))
        fams.append(Family(fam_id=i, members=members))
    case_frac = float(status[:, 0, o_col].mean())        # trait 1, proband
    target = TARGET_CASE_FRAC.get(scheme)
    stats = dict(accept_rate=n_eligible / n_drawn, case_frac=case_frac,
                 target_case_frac=target,
                 target_met=None if target is None else abs(case_frac - target) <= 0.02)
    return fams, stats


_TARGET_WARNED = set()


def _check_target(scheme, stats, args):
    """Refuse to mislabel a scheme that could not reach its named case share.

    `case_control` / `enriched_20` thin controls, so they can raise the case
    share but never lower it: a target below the population prevalence is
    silently unreachable (at K=0.30, `enriched_20` realises 0.299). Rather than
    write a row labelled `enriched_20` that is really `population`, say so once
    per scheme.
    """
    if stats.get("target_met") is False and scheme not in _TARGET_WARNED:
        _TARGET_WARNED.add(scheme)
        print(f"  !! {scheme}: target case share "
              f"{stats['target_case_frac']:.2f} unreachable at prev={args.prev} "
              f"-- realised {stats['case_frac']:.3f}. This row is NOT the "
              f"scheme it is named for; lower --prev below the target.")


@contextlib.contextmanager
def unguarded():
    """Disable the case-rate guard for the arms that exist to measure it.

    `sampling="population"` is now verified against the data
    (`ltpred.fit._assert_population_case_rate`), so the ascertained cells of
    arms A-D would raise -- which is the guard working. But the whole point of
    those arms is to record WHAT THE NUMBER WOULD HAVE BEEN without it, so they
    run with the check suppressed and say so. Arm G, which measures the IPW
    remedy, deliberately does NOT use this: it goes through the real gate.
    """
    # Patch EVERY module that holds a reference. research/advanced_fitting.py
    # does `from ltpred.fit import _assert_population_case_rate`, which binds
    # its own module-global; patching only ltpred.fit leaves that binding intact
    # and arm C still raises.
    targets = [_fit_mod, _af_mod]
    saved = [(m, m._assert_population_case_rate) for m in targets]
    for m in targets:
        m._assert_population_case_rate = lambda *a, **k: None
    try:
        yield
    finally:
        for m, fn in saved:
            m._assert_population_case_rate = fn


def _design_weights(scheme, status, o_col, prev):
    """1/inclusion-probability by design, or None where positivity fails.

    `case_control` / `enriched_20` keep every case and retain controls with
    probability `K(1-q)/(q(1-K))` for target case share `q`, so the weights are
    known exactly rather than estimated. Every other scheme drops an entire
    stratum (no unaffected proband, or no unaffected family), giving that
    stratum inclusion probability zero -- IPW is undefined, not merely noisy.
    """
    q = TARGET_CASE_FRAC.get(scheme)
    if scheme in ("population", "random_50"):
        return np.ones(status.shape[0])          # constant pi: weights cancel
    if q is None:
        return None
    keep_p = prev * (1.0 - q) / (q * (1.0 - prev))
    return np.where(status[:, o_col], 1.0, 1.0 / keep_p)


def _fit_kwargs(n_iter, burn_in):
    # eps is passed explicitly: _EPS/_at_boundary/_saturated encode the clamp
    # location, so relying on the fitter default would let a change there
    # silently invalidate every boundary column in the CSV.
    return dict(n_iter=n_iter, burn_in=burn_in, sampling="population", eps=_EPS)


def arm_h2(args, rows):
    """Arm A -- h2 under every scheme."""
    print("\n=== A. h2 recovery by ascertainment scheme ===")
    print(f"{'structure':>10} {'scheme':>15} {'fitted':>8} {'bias':>8} "
          f"{'SD':>7} {'bnd':>5} {'sel':>7} {'case%':>7}")
    for struct in args.structures:
        roles = STRUCTURES[struct]
        for scheme in args.schemes:
            fitted, intens, cfrac = [], [], []
            for r in range(args.reps):
                data_seed, fit_seed = _seeds(args.seed, "h2", r)
                fams, st = simulate_ascertained(
                    roles, {"A": args.h2}, args.n_fam, args.prev, scheme,
                    seed=data_seed)
                _check_target(scheme, st, args)
                with unguarded():
                    res = fit_heritability(fams, seed=fit_seed,
                                           **_fit_kwargs(args.n_iter, args.burn_in))
                fitted.append(res.h2)
                intens.append(st["accept_rate"])
                cfrac.append(st["case_frac"])
            fitted = np.asarray(fitted)
            sd = float(fitted.std(ddof=1)) if args.reps > 1 else float("nan")
            lo, hi = sd_ci(sd, args.reps) if args.reps > 1 else (np.nan, np.nan)
            bnd = _at_boundary(fitted)
            print(f"{struct:>10} {scheme:>15} {fitted.mean():8.3f} "
                  f"{fitted.mean() - args.h2:+8.3f} {sd:7.3f} {bnd:5.2f} "
                  f"{np.mean(intens):7.3f} {np.mean(cfrac):7.3f}")
            rows.append(dict(arm="h2", structure=struct, scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                             target="h2", truth=args.h2,
                             fitted_mean=fitted.mean(),
                             bias=fitted.mean() - args.h2, sd=sd,
                             sd_lo=lo, sd_hi=hi, boundary_frac=bnd,
                             selection_frac=float(np.mean(intens)),
                             case_frac=float(np.mean(cfrac))))


def arm_ac(args, rows):
    """Arm B -- A and C together under every scheme."""
    print("\n=== B. A+C recovery by ascertainment scheme ===")
    print(f"{'scheme':>15} {'A':>8} {'biasA':>8} {'sdA':>7} "
          f"{'C':>8} {'biasC':>8} {'sdC':>7} {'resid':>8} {'sat':>5}")
    roles = STRUCTURES["sibship"]
    props = {"A": args.a2, "C": args.c2}
    for scheme in args.schemes:
        fa, fc, resid = [], [], []
        for r in range(args.reps):
            data_seed, fit_seed = _seeds(args.seed, "ac", r)
            fams, st = simulate_ascertained(roles, props, args.n_fam, args.prev,
                                            scheme, seed=data_seed)
            _check_target(scheme, st, args)
            with unguarded():
                res = fit_variance_components(fams, ("A", "C"), seed=fit_seed,
                                              **_fit_kwargs(args.n_iter, args.burn_in))
            fa.append(res.components["A"])
            fc.append(res.components["C"])
            resid.append(res.residual)
        fa, fc, resid = np.asarray(fa), np.asarray(fc), np.asarray(resid)
        sda = float(fa.std(ddof=1)) if args.reps > 1 else float("nan")
        sdc = float(fc.std(ddof=1)) if args.reps > 1 else float("nan")
        # NOT _at_boundary: the multi-component update renormalises onto the
        # simplex, so a runaway fit shows up as an exhausted residual, not as a
        # component sitting at 1-eps (see _saturated).
        sat = _saturated(resid)
        print(f"{scheme:>15} {fa.mean():8.3f} {fa.mean() - args.a2:+8.3f} {sda:7.3f} "
              f"{fc.mean():8.3f} {fc.mean() - args.c2:+8.3f} {sdc:7.3f} "
              f"{resid.mean():8.4f} {sat:5.2f}")
        for name, vals, truth, sd in (("A", fa, args.a2, sda), ("C", fc, args.c2, sdc)):
            lo, hi = sd_ci(sd, args.reps) if args.reps > 1 else (np.nan, np.nan)
            rows.append(dict(arm="AC", structure="sibship", scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                             target=name, truth=truth, fitted_mean=vals.mean(),
                             bias=vals.mean() - truth, sd=sd, sd_lo=lo, sd_hi=hi,
                             boundary_frac=sat, selection_frac=np.nan,
                             case_frac=np.nan, residual=float(resid.mean())))


def arm_rg(args, rows):
    """Arm C -- genetic correlation, ascertained on trait 1."""
    print("\n=== C. r_g recovery (ascertained on trait 1) ===")
    print(f"{'scheme':>15} {'r_g':>8} {'bias':>8} {'SD':>7} "
          f"{'h2_1':>7} {'h2_2':>7} {'h2pin':>6}")
    roles = STRUCTURES["nuclear"]
    h2_vec = np.array([args.h2, args.h2])
    for scheme in args.rg_schemes:
        rg_hat, h1, h2_ = [], [], []
        for r in range(args.reps):
            data_seed, fit_seed = _seeds(args.seed, "rg", r)
            fams, st = simulate_ascertained_multi(
                roles, h2_vec, args.rg, args.rp, args.n_fam, args.prev, scheme,
                seed=data_seed)
            _check_target(scheme, st, args)
            with unguarded():
                res = fit_genetic_correlation(fams, seed=fit_seed,
                                              **_fit_kwargs(args.n_iter, args.burn_in))
            rg_hat.append(float(res.rg[0, 1]))
            h1.append(float(res.h2[0]))
            h2_.append(float(res.h2[1]))
        rg_hat = np.asarray(rg_hat)
        sd = float(rg_hat.std(ddof=1)) if args.reps > 1 else float("nan")
        lo, hi = sd_ci(sd, args.reps) if args.reps > 1 else (np.nan, np.nan)
        # rg = G / sqrt(h2_1 h2_2): if the per-trait h2 have run to the clamp
        # the ratio is mechanically squeezed toward a finite value, so a small
        # |bias| in rg means nothing without knowing whether its denominators
        # are pinned. Report that fraction next to it.
        pin = max(_at_boundary(h1), _at_boundary(h2_))
        print(f"{scheme:>15} {rg_hat.mean():8.3f} {rg_hat.mean() - args.rg:+8.3f} "
              f"{sd:7.3f} {np.mean(h1):7.3f} {np.mean(h2_):7.3f} {pin:6.2f}")
        rows.append(dict(arm="rg", structure="nuclear", scheme=scheme,
                         n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                         target="rg", truth=args.rg, fitted_mean=rg_hat.mean(),
                         bias=rg_hat.mean() - args.rg, sd=sd, sd_lo=lo, sd_hi=hi,
                         boundary_frac=pin, selection_frac=np.nan,
                         case_frac=float(st["case_frac"])))
        for name, vals in (("rg_h2_trait1", h1), ("rg_h2_trait2", h2_)):
            vals = np.asarray(vals)
            rows.append(dict(arm="rg", structure="nuclear", scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                             target=name, truth=args.h2, fitted_mean=vals.mean(),
                             bias=vals.mean() - args.h2,
                             sd=float(vals.std(ddof=1)) if args.reps > 1 else np.nan,
                             sd_lo=np.nan, sd_hi=np.nan,
                             boundary_frac=_at_boundary(vals),
                             selection_frac=np.nan))


def arm_scale(args, rows):
    """Arm D -- does the distortion shrink with N? (bias vs sampling noise)"""
    print("\n=== D. bias vs N: does more data fix it? ===")
    print(f"{'scheme':>15} {'N':>7} {'fitted':>8} {'bias':>8} {'SD':>7}")
    roles = STRUCTURES["nuclear"]
    for scheme in ("population", "proband_case"):
        for n_fam in args.scale_n:
            fitted = []
            for r in range(args.scale_reps):
                # offset by n_fam too: without it the N=10,000 cell would reuse
                # arm A's exact liabilities and fitter stream, and look like
                # independent corroboration of a number it merely copied
                data_seed, fit_seed = _seeds(args.seed + n_fam, "scale", r)
                fams, _st = simulate_ascertained(roles, {"A": args.h2}, n_fam,
                                                 args.prev, scheme,
                                                 seed=data_seed)
                with unguarded():
                    res = fit_heritability(fams, seed=fit_seed,
                                           **_fit_kwargs(args.n_iter, args.burn_in))
                fitted.append(res.h2)
            fitted = np.asarray(fitted)
            sd = float(fitted.std(ddof=1)) if args.scale_reps > 1 else float("nan")
            print(f"{scheme:>15} {n_fam:7d} {fitted.mean():8.3f} "
                  f"{fitted.mean() - args.h2:+8.3f} {sd:7.3f}")
            rows.append(dict(arm="scale", structure="nuclear", scheme=scheme,
                             n_fam=n_fam, prev=args.prev, reps=args.scale_reps,
                             target="h2", truth=args.h2, fitted_mean=fitted.mean(),
                             bias=fitted.mean() - args.h2, sd=sd,
                             sd_lo=np.nan, sd_hi=np.nan, boundary_frac=_at_boundary(fitted),
                             selection_frac=np.nan))


def arm_converge(args, rows):
    """Arm E -- is the distortion bias, or just an unconverged fixed point?

    The damped Haseman-Elston iteration is a stochastic fixed point, so an
    estimate far from the truth has two possible explanations: the fixed point
    is in the wrong place (bias), or the run stopped before reaching it
    (non-convergence). They are trivially confusable -- a pilot at n_iter=200
    gave even the *population* arm a -0.26 "bias" that was purely the latter.
    This arm separates them by re-fitting the same cohort at increasing n_iter
    and reporting the slope of the last 200 trace iterates: a converged fixed
    point has a flat tail wherever it sits.
    """
    print("\n=== E. bias or non-convergence? ===")
    print(f"{'scheme':>15} {'n_iter':>7} {'h2':>8} {'tail slope':>12}")
    roles = STRUCTURES["nuclear"]
    for scheme in ("population", "proband_case"):
        data_seed, fit_seed = _seeds(args.seed, "converge", 0)
        fams, _st = simulate_ascertained(roles, {"A": args.h2}, args.n_fam,
                                         args.prev, scheme, seed=data_seed)
        for n_iter in args.converge_iters:
            with unguarded():
                res = fit_heritability(fams, n_iter=n_iter, burn_in=n_iter // 3,
                                       seed=fit_seed,
                                   **{k: v for k, v in
                                      _fit_kwargs(n_iter, n_iter // 3).items()
                                      if k not in ("n_iter", "burn_in")})
            # last third of the trace, scaled by its own length: a fixed x200
            # over a short --quick run would extrapolate a slope across a window
            # that never existed (and would include burn-in)
            tail = res.trace[-max(20, res.trace.size // 3):]
            slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0] * tail.size)
            print(f"{scheme:>15} {n_iter:7d} {res.h2:8.4f} {slope:+12.2e}")
            rows.append(dict(arm="converge", structure="nuclear", scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=1,
                             target=f"h2@n_iter={n_iter}", truth=args.h2,
                             fitted_mean=res.h2, bias=res.h2 - args.h2,
                             sd=np.nan, sd_lo=np.nan, sd_hi=np.nan,
                             boundary_frac=_at_boundary([res.h2]),
                             selection_frac=np.nan, trace_tail_slope=slope))

        # A flat tail proves only that the iteration stopped moving, which a
        # wrong-but-stable value also does. Approaching the same estimate from
        # both above and below is the discriminating evidence.
        starts = []
        for h2_init in args.converge_starts:
            with unguarded():
                res = fit_heritability(fams, n_iter=args.n_iter,
                                       burn_in=args.burn_in, h2_init=h2_init,
                                       seed=fit_seed,
                                   **{k: v for k, v in
                                      _fit_kwargs(args.n_iter, args.burn_in).items()
                                      if k not in ("n_iter", "burn_in")})
            starts.append(res.h2)
            rows.append(dict(arm="converge", structure="nuclear", scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=1,
                             target=f"h2@init={h2_init}", truth=args.h2,
                             fitted_mean=res.h2, bias=res.h2 - args.h2,
                             sd=np.nan, sd_lo=np.nan, sd_hi=np.nan,
                             boundary_frac=_at_boundary([res.h2]),
                             selection_frac=np.nan, trace_tail_slope=np.nan))
        spread = float(np.max(starts) - np.min(starts))
        print(f"{scheme:>15} {'multi':>7} "
              + " ".join(f"{v:.4f}" for v in starts)
              + f"   (spread {spread:.1e} over h2_init "
              + "/".join(str(x) for x in args.converge_starts) + ")")


def _he_moment(liab, roles, centered):
    """The Haseman-Elston ratio the fitter targets, evaluated on true liabilities.

    `sum_pairs A_ij l_i l_j / sum_pairs A_ij^2` -- exactly the update in
    ltpred/fit.py, but computed on the *simulated* liabilities of the selected
    families instead of augmented draws. Two things this buys that the fitted
    number cannot:

    1. **A magnitude.** The fitter clamps to `1 - eps`, so every ascertained
       scheme reports 0.9999 and the CSV carries one bit ("pinned") rather than
       how far past the truth the selected sample's own moment sits. This
       statistic is uncensored.
    2. **A decomposition.** fit.py regresses *raw* products, so under selection,
       where E[l] != 0 within the selected families,
       `E[l_i l_j] = Cov_ij + mu_i mu_j`. Re-running with `centered=True`
       removes exactly the `mu_i mu_j` term, separating the mean shift from
       the rest.

    **This is NOT the cause of the runaway, and must not be reported as one.**
    The fitter never sees these liabilities; it sees truncated-MVN draws
    conditional on the selected status pattern at the current h2. The refuting
    cell is true h2 = 0: every ascertained scheme still fits to 0.9999 while
    this statistic sits at ~0 (proband_case +0.006) or negative
    (family_history -0.107, and -0.562 once centered, i.e. centering makes it
    worse). The runaway is augmentation feedback -- under proband_case every
    augmented proband is redrawn above threshold, relatives are pulled with it,
    and the cross-products stay positive whatever the truth -- which this
    complete-data statistic cannot see. Run `--h2 0` to reproduce.

    Only pairs with a nonzero relationship contribute -- the same filter fit.py
    applies (`abs(A_ij) > 1e-12`), so genetically unrelated mates never enter.
    """
    A = np.array([[get_relatedness(a, b, 1.0) for b in roles] for a in roles])
    pairs = [(i, j, A[i, j]) for i in range(len(roles))
             for j in range(i + 1, len(roles)) if abs(A[i, j]) > 1e-12]
    sxx = sum(a * a for _, _, a in pairs)
    x = liab - liab.mean(axis=0) if centered else liab
    sxy = sum(a * float(x[:, i] @ x[:, j]) for i, j, a in pairs)
    return sxy / (sxx * x.shape[0])


def arm_mechanism(args, rows):
    """Arm F -- how wrong, and why.

    The fitted arms can only say "pinned at the clamp". This one reports the
    uncensored target and splits it into a mean-shift component and the rest.
    No sampler runs here, so it is nearly free and is not confounded with
    augmentation or damping.
    """
    print("\n=== F. how wrong, and why (HE moment on true liabilities) ===")
    print(f"{'structure':>10} {'scheme':>19} {'uncentered':>11} {'centered':>9} "
          f"{'x truth':>8}")
    for struct in args.structures:
        roles = STRUCTURES[struct]
        for scheme in args.schemes:
            unc, cen = [], []
            for r in range(args.reps):
                data_seed, _ = _seeds(args.seed, "h2", r)
                _f, st = simulate_ascertained(
                    roles, {"A": args.h2}, args.n_fam, args.prev, scheme,
                    seed=data_seed, return_liab=True)
                unc.append(_he_moment(st["liab"], roles, centered=False))
                cen.append(_he_moment(st["liab"], roles, centered=True))
            unc, cen = np.asarray(unc), np.asarray(cen)
            # the ratio is meaningless at the h2=0 cell (and that cell is
            # exactly where this arm is most informative), so print the
            # absolute moment there instead of dividing by zero
            ratio = (f"{unc.mean() / args.h2:8.2f}" if args.h2 > 0
                     else f"{'--':>8}")
            print(f"{struct:>10} {scheme:>19} {unc.mean():11.3f} {cen.mean():9.3f} " + ratio)
            rows.append(dict(arm="mechanism", structure=struct, scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                             target="he_moment", truth=args.h2,
                             fitted_mean=unc.mean(),
                             bias=unc.mean() - args.h2,
                             sd=float(unc.std(ddof=1)) if args.reps > 1 else np.nan,
                             sd_lo=np.nan, sd_hi=np.nan, boundary_frac=np.nan,
                             selection_frac=np.nan,
                             he_uncentered=unc.mean(), he_centered=cen.mean()))


def arm_ipw(args, rows):
    """Arm G -- does inverse-probability weighting undo the damage?

    The augmentation for a GIVEN family with GIVEN statuses is already the right
    conditional distribution; what selection breaks is the *mix* of families.
    That is what IPW repairs, so it is a better-matched remedy than a scale
    transform. This arm runs the real `sampling="ipw"` path -- no `unguarded()`
    -- so it also exercises the weighted marginal-calibration screen.

    Reported per scheme: the unweighted fit (what arms A-D measure), the IPW
    fit, and both across-replicate SDs, because the standard objection to IPW is
    efficiency rather than bias and the weights here reach 19x.
    """
    print("\n=== G. does IPW fix it? ===")
    print(f"{'scheme':>19} {'maxw':>6} {'unweighted':>11} {'IPW':>8} {'bias':>8} "
          f"{'SD(unw)':>8} {'SD(ipw)':>8}")
    roles = STRUCTURES["nuclear"]
    o_col = roles.index("o")
    for scheme in args.schemes:
        unw, ipw, maxw, blocked = [], [], 0.0, False
        for r in range(args.reps):
            data_seed, fit_seed = _seeds(args.seed, "ipw", r)
            props = {"A": args.h2} if args.h2 > 0 else {}
            fams, _st = simulate_ascertained(roles, props, args.n_fam, args.prev,
                                             scheme, seed=data_seed)
            status = np.array([[np.isfinite(m.lower) for m in f.members]
                               for f in fams])
            w = _design_weights(scheme, status, o_col, args.prev)
            with unguarded():
                unw.append(fit_heritability(
                    fams, seed=fit_seed,
                    **_fit_kwargs(args.n_iter, args.burn_in)).h2)
            if w is None:
                blocked = True
                continue
            maxw = max(maxw, float(w.max()))
            # the REAL gate: sampling="ipw" re-runs the case-rate check on the
            # weighted counts, so a positivity failure still raises here
            try:
                ipw.append(fit_heritability(
                    fams, seed=fit_seed, weights=w,
                    **{k: v for k, v in
                       _fit_kwargs(args.n_iter, args.burn_in).items()
                       if k != "sampling"}, sampling="ipw").h2)
            except ValueError:
                blocked = True
                break
        unw = np.asarray(unw)
        sd_u = float(unw.std(ddof=1)) if len(unw) > 1 else float("nan")
        if blocked or not ipw:
            print(f"{scheme:>19} {'--':>6} {unw.mean():11.3f} {'--':>8} {'--':>8} "
                  f"{sd_u:8.3f} {'--':>8}   positivity fails: IPW undefined")
            rows.append(dict(arm="ipw", structure="nuclear", scheme=scheme,
                             n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                             target="ipw_undefined", truth=args.h2,
                             fitted_mean=unw.mean(), bias=unw.mean() - args.h2,
                             sd=sd_u, sd_lo=np.nan, sd_hi=np.nan,
                             boundary_frac=np.nan, selection_frac=np.nan))
            continue
        ipw = np.asarray(ipw)
        sd_i = float(ipw.std(ddof=1)) if len(ipw) > 1 else float("nan")
        print(f"{scheme:>19} {maxw:6.1f} {unw.mean():11.3f} {ipw.mean():8.3f} "
              f"{ipw.mean() - args.h2:+8.3f} {sd_u:8.3f} {sd_i:8.3f}")
        rows.append(dict(arm="ipw", structure="nuclear", scheme=scheme,
                         n_fam=args.n_fam, prev=args.prev, reps=args.reps,
                         target="h2_ipw", truth=args.h2, fitted_mean=ipw.mean(),
                         bias=ipw.mean() - args.h2, sd=sd_i,
                         sd_lo=np.nan, sd_hi=np.nan, boundary_frac=np.nan,
                         selection_frac=np.nan, max_weight=maxw,
                         unweighted_mean=unw.mean(), unweighted_sd=sd_u))


def _sample_at_case_share(roles, props, n_fam, prev, target, seed):
    """`n_fam` families whose proband case share is `target` (keep all cases,
    thin controls). Separate from `_select` because the dose-response needs the
    share as a swept parameter rather than one of the two named schemes."""
    rng = np.random.default_rng(seed)
    Sig = _component_cov(roles, props)
    t = float(liability_threshold(prev))
    o = roles.index("o")
    kept, n = [], 0
    while n < n_fam:
        st = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=200_000) > t
        pro = st[:, o]
        n_case, n_ctrl = int(pro.sum()), int((~pro).sum())
        if n_case == 0 or n_ctrl == 0:
            continue
        want = n_case * (1.0 - target) / target
        mask = pro.copy()
        ctrl = np.flatnonzero(~pro)
        mask[ctrl[rng.random(ctrl.size) < min(1.0, want / n_ctrl)]] = True
        kept.append(st[mask])
        n += int(mask.sum())
    st = np.concatenate(kept)[:n_fam]
    return _build_families(st, roles, t), float(st[:, o].mean())


def arm_dose(args, rows):
    """Arm H -- how much enrichment does it take?

    The schemes in arms A-G are fixed designs; this sweeps the realised case
    share continuously against the assumed prevalence, which is what calibrates
    the case-rate guard's tolerance in ltpred.fit. Previously computed ad hoc;
    it is an arm so the numbers have an artifact behind them.
    """
    print("\n=== H. enrichment dose-response ===")
    print(f"{'target':>7} {'realised':>9} {'enrich':>7} {'fitted h2':>10} {'bias':>8} {'SD':>7}")
    roles = STRUCTURES["nuclear"]
    for target in args.dose_targets:
        fits, shares = [], []
        for r in range(args.dose_reps):
            data_seed, fit_seed = _seeds(args.seed, "dose", r)
            fams, share = _sample_at_case_share(
                roles, {"A": args.h2}, args.dose_n, args.prev, target,
                seed=data_seed + int(target * 1e6))
            with unguarded():
                fits.append(fit_heritability(
                    fams, seed=fit_seed,
                    **_fit_kwargs(args.n_iter, args.burn_in)).h2)
            shares.append(share)
        fits = np.asarray(fits)
        sd = float(fits.std(ddof=1)) if args.dose_reps > 1 else float("nan")
        enrich = float(np.mean(shares)) / args.prev
        print(f"{target:7.3f} {np.mean(shares):9.3f} {enrich:7.2f}x "
              f"{fits.mean():10.3f} {fits.mean() - args.h2:+8.3f} {sd:7.3f}")
        rows.append(dict(arm="dose", structure="nuclear", scheme=f"target_{target:g}",
                         n_fam=args.dose_n, prev=args.prev, reps=args.dose_reps,
                         target="h2", truth=args.h2, fitted_mean=fits.mean(),
                         bias=fits.mean() - args.h2, sd=sd, sd_lo=np.nan,
                         sd_hi=np.nan, boundary_frac=_at_boundary(fits),
                         selection_frac=np.nan, case_frac=float(np.mean(shares)),
                         enrichment=enrich))


def arm_specificity(args, rows):
    """Arm I -- does the case-rate guard fire on legitimate population data?

    Sensitivity is arm A's job; this is the other half, and the one that decides
    whether the guard is safe to ship. Each cell is an unascertained cohort that
    MUST pass.
    """
    print("\n=== I. guard specificity on population cohorts ===")
    print(f"{'N':>7} {'K':>6} {'cohorts':>8} {'false positives':>16}")
    from ltpred.simulate import simulate_under_LTM_single
    for n_fam, prev in args.spec_grid:
        fp = 0
        for r in range(args.spec_reps):
            sim = simulate_under_LTM_single(
                fam_vec=["m", "f", "s1", "s2"], h2=args.h2, n_sim=n_fam,
                pop_prev=prev, seed=args.seed + 7919 * r + int(prev * 1000))
            try:
                fit_heritability(sim.families, n_iter=120, burn_in=40,
                                 seed=args.seed + r, sampling="population")
            except ValueError:
                fp += 1
        print(f"{n_fam:7d} {prev:6.2f} {args.spec_reps:8d} {fp:16d}")
        rows.append(dict(arm="specificity", structure="nuclear",
                         scheme=f"K{prev:g}", n_fam=n_fam, prev=prev,
                         reps=args.spec_reps, target="false_positives",
                         truth=0.0, fitted_mean=float(fp), bias=float(fp),
                         sd=np.nan, sd_lo=np.nan, sd_hi=np.nan,
                         boundary_frac=np.nan, selection_frac=np.nan))


def _he_observed(status, roles):
    """Observed-scale HE: standardised 0/1 cross-products regressed on A."""
    A = np.array([[get_relatedness(a, b, 1.0) for b in roles] for a in roles])
    pairs = [(i, j, A[i, j]) for i in range(len(roles))
             for j in range(i + 1, len(roles)) if abs(A[i, j]) > 1e-12]
    sxx = sum(a * a for _, _, a in pairs)
    y = status.astype(float)
    sd = y.std(axis=0, ddof=0)
    if np.any(sd == 0):
        return float("nan")
    z = (y - y.mean(axis=0)) / sd
    return sum(a * float(z[:, i] @ z[:, j]) for i, j, a in pairs) / (sxx * z.shape[0])


def arm_lee(args, rows):
    """Arm J -- can a Lee et al. observed->liability factor rescue this instead?

    The obvious alternative to reweighting, and the one a reader will ask about.
    It targets a different (observed-scale) estimand, so the fair test is the
    full pipeline: observed-scale HE on the raw statuses, then
    `observed_to_liability_h2(K, P)`.
    """
    print("\n=== J. observed-scale HE + Lee et al. correction ===")
    print(f"{'scheme':>15} {'P(case)':>8} {'h2_obs':>8} {'+Lee':>8} {'bias':>8}")
    from ltpred.liability_scale import observed_to_liability_h2
    roles = STRUCTURES["nuclear"]
    o = roles.index("o")
    for scheme in args.lee_schemes:
        obs, lee, ps = [], [], []
        for r in range(args.reps):
            data_seed, _ = _seeds(args.seed, "lee", r)
            props = {"A": args.h2} if args.h2 > 0 else {}
            fams, _st = simulate_ascertained(roles, props, args.n_fam,
                                             args.prev, scheme, seed=data_seed)
            status = np.array([[np.isfinite(m.lower) for m in f.members]
                               for f in fams])
            p_case = float(status[:, o].mean())
            if not 0.0 < p_case < 1.0:      # Lee's factor needs P(1-P) > 0
                continue
            h2o = _he_observed(status, roles)
            obs.append(h2o)
            lee.append(float(observed_to_liability_h2(h2o, args.prev,
                                                      prop_cases=p_case)))
            ps.append(p_case)
        if not lee:
            print(f"{scheme:>15} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'undefined: P(1-P) = 0':>8}")
            continue
        lee_a = np.asarray(lee)
        print(f"{scheme:>15} {np.mean(ps):8.3f} {np.mean(obs):8.3f} "
              f"{lee_a.mean():8.3f} {lee_a.mean() - args.h2:+8.3f}")
        rows.append(dict(arm="lee", structure="nuclear", scheme=scheme,
                         n_fam=args.n_fam, prev=args.prev, reps=len(lee),
                         target="h2_lee", truth=args.h2, fitted_mean=lee_a.mean(),
                         bias=lee_a.mean() - args.h2,
                         sd=float(lee_a.std(ddof=1)) if len(lee) > 1 else np.nan,
                         sd_lo=np.nan, sd_hi=np.nan, boundary_frac=np.nan,
                         selection_frac=np.nan, case_frac=float(np.mean(ps)),
                         he_uncentered=float(np.mean(obs))))


FIELDS = ["arm", "structure", "scheme", "n_fam", "prev", "reps", "target",
          "truth", "fitted_mean", "bias", "sd", "sd_lo", "sd_hi",
          "boundary_frac", "selection_frac", "case_frac", "residual",
          "he_uncentered", "he_centered", "trace_tail_slope",
          "max_weight", "unweighted_mean", "unweighted_sd", "enrichment"]


def write_csv(rows, tag=""):
    path = os.path.join(HERE, f"bench_ascertainment{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"\nwrote {path}")


def plot(rows, tag=""):
    plt = get_plt()
    if plt is None:
        return
    h2 = [r for r in rows if r["arm"] == "h2" and r["structure"] == "nuclear"]
    scale = [r for r in rows if r["arm"] == "scale"]
    if not h2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    names = [r["scheme"] for r in h2]
    bias = [r["bias"] for r in h2]
    err = [r["sd"] / max(np.sqrt(r["reps"]), 1) for r in h2]
    ax.barh(names, bias, xerr=err, color=["#4C72B0" if n == "population" else "#C44E52"
                                          for n in names])
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("bias in fitted h²")
    ax.set_title("A. h² bias by ascertainment scheme")
    ax = axes[1]
    scale_schemes = sorted({r["scheme"] for r in scale})
    for scheme in scale_schemes:
        pts = sorted((r["n_fam"], r["bias"], r["sd"]) for r in scale
                     if r["scheme"] == scheme)
        n = [p[0] for p in pts]
        b = [p[1] for p in pts]
        s = [p[2] for p in pts]
        ax.errorbar(n, b, yerr=s, marker="o", label=scheme, capsize=3)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("families analysed")
    ax.set_ylabel("bias in fitted h²")
    ax.set_title("D. does more data fix it?")
    if scale_schemes:            # --arms without "scale" leaves the panel empty
        ax.legend()
    fig.tight_layout()
    path = os.path.join(HERE, f"bench_ascertainment{tag}.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--n-fam", type=int, default=10_000)
    p.add_argument("--prev", type=float, default=0.05)
    p.add_argument("--h2", type=float, default=0.5)
    p.add_argument("--a2", type=float, default=0.4)
    p.add_argument("--c2", type=float, default=0.2)
    p.add_argument("--rg", type=float, default=0.5)
    p.add_argument("--rp", type=float, default=0.3)
    p.add_argument("--n-iter", type=int, default=1500)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--scale-n", type=int, nargs="+",
                   default=[2500, 10_000, 40_000])
    p.add_argument("--scale-reps", type=int, default=5)
    p.add_argument("--converge-iters", type=int, nargs="+",
                   default=[500, 1500, 4000])
    p.add_argument("--converge-starts", type=float, nargs="+",
                   default=[0.05, 0.5, 0.95])
    p.add_argument("--dose-targets", type=float, nargs="+",
                   default=[0.05, 0.06, 0.075, 0.10, 0.125, 0.15, 0.20])
    p.add_argument("--dose-n", type=int, default=4000)
    p.add_argument("--dose-reps", type=int, default=3)
    p.add_argument("--spec-reps", type=int, default=2)
    p.add_argument("--lee-schemes", nargs="+",
                   default=["population", "case_control", "enriched_20"],
                   choices=list(SCHEMES))
    p.add_argument("--structures", nargs="+", default=["nuclear", "sibship"],
                   choices=sorted(STRUCTURES))
    p.add_argument("--schemes", nargs="+", default=list(SCHEMES),
                   choices=list(SCHEMES))
    p.add_argument("--rg-schemes", nargs="+",
                   default=["population", "proband_case", "case_control"],
                   choices=list(SCHEMES))
    p.add_argument("--arms", nargs="+",
                   default=["h2", "ac", "rg", "scale", "converge", "mechanism",
                            "ipw", "dose", "specificity", "lee"],
                   choices=["h2", "ac", "rg", "scale", "converge", "mechanism",
                            "ipw", "dose", "specificity", "lee"])
    p.add_argument("--tag", default="",
                   help="suffix for the output filenames, so a supplementary "
                        "pass (e.g. --h2 0) does not overwrite the main grid")
    p.add_argument("--quick", action="store_true",
                   help="tiny grid for a smoke run")
    args = p.parse_args()
    if not hasattr(args, "spec_grid"):
        args.spec_grid = [(500, 0.05), (1500, 0.10), (4000, 0.05),
                          (4000, 0.20), (10_000, 0.02), (10_000, 0.10)]

    if args.quick:
        args.reps, args.n_fam, args.n_iter, args.burn_in = 2, 800, 200, 60
        args.structures = ["nuclear"]
        args.scale_n, args.scale_reps = [400, 1600], 2
        args.converge_iters = [100, 300]
        args.dose_targets, args.dose_n, args.dose_reps = [0.05, 0.15], 400, 2
        args.spec_grid, args.spec_reps = [(400, 0.05)], 1
        args.converge_starts = [0.05, 0.95]

    t0 = time.time()
    rows = []
    if "h2" in args.arms:
        arm_h2(args, rows)
    if "ac" in args.arms:
        arm_ac(args, rows)
    if "rg" in args.arms:
        arm_rg(args, rows)
    if "scale" in args.arms:
        arm_scale(args, rows)
    if "converge" in args.arms:
        arm_converge(args, rows)
    if "mechanism" in args.arms:
        arm_mechanism(args, rows)
    if "ipw" in args.arms:
        arm_ipw(args, rows)
    if "dose" in args.arms:
        arm_dose(args, rows)
    if "specificity" in args.arms:
        arm_specificity(args, rows)
    if "lee" in args.arms:
        arm_lee(args, rows)
    write_csv(rows, args.tag)
    plot(rows, args.tag)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
