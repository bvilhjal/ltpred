"""How much does modelling (and fitting) shared environment help prediction?

Family history clusters partly for **genetic** reasons (`A`) and partly for shared
**environment** (`C`, e.g. a rearing effect common to full sibs). This benchmark's
grouped Gibbs estimator approximates the posterior mean of the proband's *genetic*
value given the family's covariance and case/control bounds. If you model the family
as additive-only, the estimator wrongly credits the sib-shared environmental
resemblance to genetics, so the score is **contaminated** by the sibship's
environment. Modelling `C` lets the estimator attribute that resemblance to
environment instead, sharpening the genetic estimate.

This benchmark quantifies the gain. It simulates families under the true `A+C+E`
model (so the proband's *true genetic liability* `g` is known), then estimates the
genetic-liability score under several models and reports corr(estimate, true `g`):

  1. **ignore C, true h²** — additive-only covariance at the true h² (isolates the
     covariance misspecification);
  2. **ignore C, fitted h²** — the realistic ignore-C pipeline: `fit_heritability`
     (which *inflates* h² because C loads onto sib resemblance), then additive;
  3. **fit A+C** — the realistic model-C pipeline: `fit_variance_components(A, C)`,
     then the A+C covariance;
  4. **oracle A+C** — true (h², c²): the ceiling.

Swept over the true c². At c²=0 all four coincide (a sanity check).

Panel **(c)** runs the same idea through the **public** API (merged from the
former `bench_env_components.py`; its seeds and design are preserved): families
simulated with sibship (`C`) and couple (`M`) shared environment via
`construct_covmat_single(h2, c2, m2)`, then `estimate_liability(..., c2=, m2=)`
with the components omitted (additive-only, misspecified), wired at the truth
(oracle-wired), or wired from `fit_variance_components(("A", "C", "M"))`
(fitted-wired). It reports corr(g), the calibration slope of the genetic
estimate, and corr of the full-liability prediction E[l_o | family] with the
true full liability.

    python benchmarks/bench_shared_env.py
    python benchmarks/bench_shared_env.py --c2 0 0.1 0.2 0.3 --n-fam 3000
Writes bench_shared_env.csv (+ .png if matplotlib is present).
"""

import os
import argparse
import warnings

import numpy as np
from scipy.stats import t as student_t

from _common import get_plt, simulate_families_components, write_rows
from ltpred.covariance import (construct_covmat_single, get_relatedness,
                               correct_positive_definite)
from ltpred.fit import _component_matrix, fit_heritability, fit_variance_components
from ltpred.estimate import _estimate_group, estimate_liability
from ltpred.family import Family, Member
from ltpred.thresholds import liability_threshold

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET = 0                                   # estimate the proband's genetic liability
GIBBS_TOL = 0.02
GIBBS_MAX_ROUNDS = 20

# Panel (c): public-API C/M wiring arm (merged from bench_env_components.py).
# Family structure, seed schedule and fit settings are frozen so the numbers
# stay comparable to the historical bench_env_components results.
WIRE_FAM_VEC = ["m", "f", "s1", "s2"]
WIRE_SEED = 20260719
WIRE_FIT_KW = dict(n_iter=600, burn_in=200)


def roles_with_sibs(n_sib):
    """Proband + parents + ``n_sib`` full sibs; C is shared within {o, s1..s_n}."""
    return ["o", "m", "f"] + [f"s{i}" for i in range(1, n_sib + 1)]


ROLES = roles_with_sibs(3)


def _matrices(roles):
    A = np.array([[get_relatedness(a, b, 1.0) for b in roles] for a in roles])
    A = correct_positive_definite(A)[0]
    C = _component_matrix(roles, "C")
    return A, C


def estimate_g(fams, roles, h2, c2, *, n_sim, burn_in, seed):
    """Posterior mean and Gibbs MCSE for target genetic liability under (h2, c2)."""
    A, C = _matrices(roles)
    n = len(roles)
    e2 = max(1.0 - h2 - c2, 1e-4)
    d = n + 1
    cov = np.empty((d, d))                    # [g_target, o_0..o_{n-1}]
    cov[0, 0] = h2
    cov[0, 1:] = cov[1:, 0] = h2 * A[TARGET]
    cov[1:, 1:] = h2 * A + c2 * C + e2 * np.eye(n)     # diagonal = 1
    cov = correct_positive_definite(cov)[0]
    F = len(fams)
    lo = np.full((F, d), -np.inf)
    hi = np.full((F, d), np.inf)
    for i, fam in enumerate(fams):
        for j, m in enumerate(fam.members):
            lo[i, 1 + j], hi[i, 1 + j] = m.lower, m.upper
    seeds = seed + np.arange(F, dtype=np.int64) * 50
    est, se, _var = _estimate_group(
        cov, [0], lo, hi, seeds, GIBBS_TOL, int(n_sim), int(burn_in), GIBBS_MAX_ROUNDS
    )
    return est[:, 0], se[:, 0]


def _mean_uncertainty(values):
    """Return mean, across-replicate SD/SE, and a t-based 95% CI half-width."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, np.nan, np.nan, np.nan
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(values.size)
    ci95 = float(student_t.ppf(0.975, values.size - 1) * se)
    return mean, sd, float(se), ci95


def run_setting(roles, h2, c2, n_fam, prev, reps, n_sim, burn_in, seed0):
    """Mean corr(estimate, true g) for the four models at one (structure, c2)."""
    acc = {k: [] for k in ("add_true", "add_fit", "ace_fit", "ace_oracle")}
    mcse = {k: [] for k in acc}
    fitted = []
    for r in range(reps):
        seed = seed0 + 1000 * r
        fams, g = simulate_families_components(roles, {"A": h2, "C": c2}, n_fam,
                                               prev, seed, return_genetic=True)
        g_true = g[:, TARGET]
        h2_add = min(max(fit_heritability(
            fams, sampling="population", n_iter=500, burn_in=150, seed=seed + 101
        ).h2, 0.02), 0.95)
        vc = fit_variance_components(
            fams, ("A", "C"), sampling="population",
            n_iter=600, burn_in=200, seed=seed + 211
        )
        h2_ace, c2_ace = vc.components["A"], vc.components["C"]
        fitted.append((h2_add, h2_ace, c2_ace))
        for name, (hh, cc) in dict(add_true=(h2, 0.0), add_fit=(h2_add, 0.0),
                                   ace_fit=(h2_ace, c2_ace), ace_oracle=(h2, c2)).items():
            est, se = estimate_g(
                fams, roles, hh, cc, n_sim=n_sim, burn_in=burn_in, seed=seed + 7
            )
            acc[name].append(np.corrcoef(est, g_true)[0, 1])
            mcse[name].append(se)
    m = {"reps": reps}
    for name, values in acc.items():
        mean, sd, se, ci95 = _mean_uncertainty(values)
        m[name] = mean
        m[f"{name}_sd"] = sd
        m[f"{name}_se"] = se
        m[f"{name}_ci95"] = ci95
    paired_gain = np.asarray(acc["ace_fit"]) - np.asarray(acc["add_fit"])
    oracle_gain = np.asarray(acc["ace_oracle"]) - np.asarray(acc["add_fit"])
    for name, values in (("gain", paired_gain), ("oracle_gain", oracle_gain)):
        mean, sd, se, ci95 = _mean_uncertainty(values)
        m[name] = mean
        m[f"{name}_sd"] = sd
        m[f"{name}_se"] = se
        m[f"{name}_ci95"] = ci95
    unconverged = 0
    nonfinite = 0
    for name, chunks in mcse.items():
        values = np.concatenate(chunks)
        finite = values[np.isfinite(values)]
        bad = int(values.size - finite.size)
        count = int(np.count_nonzero(finite > GIBBS_TOL)) + bad
        m[f"{name}_mcse_max"] = float(finite.max()) if finite.size else np.nan
        m[f"{name}_nonfinite"] = bad
        m[f"{name}_unconverged"] = count
        nonfinite += bad
        unconverged += count
    finite_maxima = [m[f"{name}_mcse_max"] for name in mcse
                     if np.isfinite(m[f"{name}_mcse_max"])]
    m["mcse_max"] = max(finite_maxima) if finite_maxima else np.nan
    m["nonfinite"] = nonfinite
    m["unconverged"] = unconverged
    if unconverged:
        total = reps * n_fam * len(mcse)
        warnings.warn(
            f"{unconverged} of {total} model-family estimates did not reach "
            f"tol={GIBBS_TOL} within max_rounds={GIBBS_MAX_ROUNDS}; their reported "
            "MCSE exceeds tol — increase max_rounds or n_sim.",
            RuntimeWarning,
            stacklevel=2,
        )
    m["h2_add"] = float(np.mean([f[0] for f in fitted]))
    m["h2_ace"] = float(np.mean([f[1] for f in fitted]))
    m["c2_ace"] = float(np.mean([f[2] for f in fitted]))
    return m


def wire_metrics(fams, true_g, true_o, h2, **kw):
    """corr(g), calibration slope(g) and corr(o) from the public estimator."""
    res = estimate_liability(fams, h2=h2, out=("genetic", "full"), **kw)
    g = np.asarray(res.est["genetic"])
    o = np.asarray(res.est["full"])
    return (float(np.corrcoef(g, true_g)[0, 1]),
            float(np.polyfit(g, true_g, 1)[0]),
            float(np.corrcoef(o, true_o)[0, 1]))


def run_wiring(h2, c2, m2, prev, n_fam, reps):
    """Panel (c): the public-API wiring contrast, including the M component.

    Families (proband + parents + two sibs) are simulated with true additive
    genetics plus sibship (C) and couple (M) shared environment from
    ``construct_covmat_single(h2, c2, m2)``, then estimated through the public
    ``estimate_liability(..., c2=, m2=)`` API under three arms: additive-only
    (misspecified), oracle-wired (c2/m2 at the truth) and fitted-wired
    (components from ``fit_variance_components(("A", "C", "M"))``). Seed
    schedule and fit settings preserved from bench_env_components.py."""
    cov_obj = construct_covmat_single(fam_vec=WIRE_FAM_VEC, h2=h2, c2=c2, m2=m2)
    roles = cov_obj.roles                      # g, o, m, f, s1, s2
    cov = cov_obj.matrix
    thr = float(liability_threshold(prev))
    arms = ("additive-only", "oracle-wired", "fitted-wired")
    acc = {a: [] for a in arms}
    fitted = []
    for rep in range(reps):
        rng = np.random.default_rng(np.random.PCG64(WIRE_SEED + 41 * rep))
        liab = rng.multivariate_normal(np.zeros(len(roles)), cov, size=n_fam)
        status = {r: liab[:, roles.index(r)] > thr for r in roles if r != "g"}
        fams = []
        for i in range(n_fam):
            fams.append(Family(fam_id=i, members=[
                Member(role=r, lower=thr if status[r][i] else -np.inf,
                       upper=np.inf if status[r][i] else thr)
                for r in roles if r != "g"]))
        true_g = liab[:, roles.index("g")]
        true_o = liab[:, roles.index("o")]
        acc["additive-only"].append(wire_metrics(fams, true_g, true_o, h2))
        acc["oracle-wired"].append(wire_metrics(fams, true_g, true_o, h2,
                                                c2=c2, m2=m2))
        fit = fit_variance_components(
            fams, ("A", "C", "M"), sampling="population", seed=rep,
            **WIRE_FIT_KW)
        c2_hat = float(fit.components.get("C", 0.0))
        m2_hat = float(fit.components.get("M", 0.0))
        fitted.append((float(fit.components["A"]), c2_hat, m2_hat))
        acc["fitted-wired"].append(
            wire_metrics(fams, true_g, true_o, h2, c2=c2_hat, m2=m2_hat))
    m = {"reps": reps, "n_fam": n_fam, "m2": m2}
    keys = {"additive-only": "wire_add", "oracle-wired": "wire_oracle",
            "fitted-wired": "wire_fitted"}
    for arm, key in keys.items():
        for j, met in enumerate(("corrg", "slopeg", "corro")):
            mean, sd, se, _ci = _mean_uncertainty([v[j] for v in acc[arm]])
            m[f"{key}_{met}"] = mean
            m[f"{key}_{met}_sd"] = sd
            m[f"{key}_{met}_se"] = se
    for j, comp in enumerate("ACM"):
        mean, sd, _se, _ci = _mean_uncertainty([f[j] for f in fitted])
        m[f"wire_fit_{comp}"] = mean
        m[f"wire_fit_{comp}_sd"] = sd
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--sibs", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--c2-for-sibs", type=float, default=0.3)
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n-sim", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=600)
    ap.add_argument("--wire-h2", type=float, default=0.4,
                    help="panel (c): true h2 (bench_env_components design)")
    ap.add_argument("--wire-c2", type=float, default=0.15,
                    help="panel (c): true sibship environment c2")
    ap.add_argument("--wire-m2", type=float, default=0.15,
                    help="panel (c): true couple environment m2")
    ap.add_argument("--wire-fams", type=int, default=4000)
    ap.add_argument("--wire-reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    h2 = args.h2
    tested_c2 = np.asarray([*args.c2, args.c2_for_sibs], dtype=float)
    if not np.isfinite(h2) or h2 < 0:
        ap.error("--h2 must be finite and nonnegative")
    if np.any(~np.isfinite(tested_c2)) or np.any(tested_c2 < 0):
        ap.error("all --c2 and --c2-for-sibs values must be finite and nonnegative")
    if np.any(h2 + tested_c2 > 1):
        ap.error("--h2 + c2 must be <= 1 for every tested shared-environment setting")
    if not 0.0 < args.wire_h2 <= 1.0:
        ap.error("--wire-h2 must be in (0, 1]")
    for flag, value in (("--wire-c2", args.wire_c2), ("--wire-m2", args.wire_m2)):
        if not np.isfinite(value) or value < 0:
            ap.error(f"{flag} must be finite and nonnegative")
    if args.wire_h2 + args.wire_c2 + args.wire_m2 > 1:
        ap.error("--wire-h2 + --wire-c2 + --wire-m2 must be <= 1")
    kw = dict(n_fam=args.n_fam, prev=args.prev, reps=args.reps, n_sim=args.n_sim,
              burn_in=args.burn_in)

    warm_c2 = float(tested_c2[0])
    fw, _ = simulate_families_components(ROLES, {"A": h2, "C": warm_c2}, 60,
                                         args.prev, 0, return_genetic=True)  # warm JIT
    estimate_g(fw, ROLES, h2, warm_c2, n_sim=2000, burn_in=200, seed=0)
    fit_variance_components(
        fw, ("A", "C"), sampling="population", n_iter=20, burn_in=5
    )
    estimate_liability(fw, h2=h2, out=("genetic", "full"), c2=warm_c2)  # PA path
    fit_variance_components(
        fw, ("A", "C", "M"), sampling="population", n_iter=20, burn_in=5
    )

    print(f"h2={h2} n_fam={args.n_fam} reps={args.reps}  corr(estimate, true genetic liability)")

    # (a) sweep c2 at a fixed 3-sib structure -------------------------------------
    print(f"\n(a) vs c²  [roles = {'+'.join(roles_with_sibs(3))}]")
    print(f"{'c2':>5} | {'ignoreC(trueh2)':>15} {'ignoreC(fit)':>13} {'fit A+C':>9} "
          f"{'oracle A+C':>11} | gain(fit) ± 95% CI   fitted(h2add,h2ace,c2ace)")
    rows_c2 = []
    for c2 in args.c2:
        m = run_setting(roles_with_sibs(3), h2, c2, seed0=args.seed, **kw)
        rows_c2.append(dict(panel="c2", c2=c2, n_sib=3, n_fam=args.n_fam, **m))
        print(f"{c2:5.2f} | {m['add_true']:15.4f} {m['add_fit']:13.4f} {m['ace_fit']:9.4f} "
              f"{m['ace_oracle']:11.4f} | {m['gain']:+.4f} ± {m['gain_ci95']:.4f}   "
              f"({m['h2_add']:.2f},{m['h2_ace']:.2f},{m['c2_ace']:.2f})")

    # (b) sweep sib-ship size at a fixed c2 ---------------------------------------
    cf = args.c2_for_sibs
    print(f"\n(b) vs sib-ship size  [c² = {cf}]")
    print(f"{'nsib':>5} | {'ignoreC(fit)':>13} {'fit A+C':>9} {'oracle A+C':>11} | "
          "gain(fit) ± 95% CI")
    rows_sib = []
    for ns in args.sibs:
        m = run_setting(roles_with_sibs(ns), h2, cf, seed0=args.seed + 500, **kw)
        rows_sib.append(dict(panel="sibs", c2=cf, n_sib=ns, n_fam=args.n_fam, **m))
        print(f"{ns:5d} | {m['add_fit']:13.4f} {m['ace_fit']:9.4f} {m['ace_oracle']:11.4f} | "
              f"{m['gain']:+.4f} ± {m['gain_ci95']:.4f}")

    # (c) public-API wiring of C/M into estimation ------------------------------
    wh2, wc2, wm2 = args.wire_h2, args.wire_c2, args.wire_m2
    print(f"\n(c) public-API C/M wiring  [estimate_liability(c2=, m2=); "
          f"{args.wire_fams} fams, h2={wh2} c2={wc2} m2={wm2}]")
    mw = run_wiring(wh2, wc2, wm2, args.prev, args.wire_fams, args.wire_reps)
    rows_wire = [dict(panel="wiring", c2=wc2, n_sib=2, **mw)]
    print(f"  fitted A/C/M = {mw['wire_fit_A']:.3f}/{mw['wire_fit_C']:.3f}/"
          f"{mw['wire_fit_M']:.3f}  (truth {wh2}/{wc2}/{wm2})")
    print(f"  {'arm':16s} {'corr(g)':>8s} {'slope(g)':>9s} {'corr(o)':>8s}   (± SE)")
    for arm, key in (("additive-only", "wire_add"), ("oracle-wired", "wire_oracle"),
                     ("fitted-wired", "wire_fitted")):
        print(f"  {arm:16s} {mw[key + '_corrg']:8.4f} {mw[key + '_slopeg']:9.4f} "
              f"{mw[key + '_corro']:8.4f}   (± {mw[key + '_corrg_se']:.3f}/"
              f"{mw[key + '_slopeg_se']:.3f}/{mw[key + '_corro_se']:.3f})")

    write_csv(rows_c2 + rows_sib + rows_wire)
    plot(rows_c2, rows_sib, h2)
    print("\nwrote bench_shared_env.csv and bench_shared_env.png")


def write_csv(rows):
    corr = ("add_true", "add_fit", "ace_fit", "ace_oracle")
    fields = ["panel", "c2", "n_sib", "reps"]
    for name in corr:
        fields.extend([name, f"{name}_sd", f"{name}_se", f"{name}_ci95"])
    for name in ("gain", "oracle_gain"):
        fields.extend([name, f"{name}_sd", f"{name}_se", f"{name}_ci95"])
    for name in corr:
        fields.extend([f"{name}_mcse_max", f"{name}_nonfinite",
                       f"{name}_unconverged"])
    fields.extend(["mcse_max", "nonfinite", "unconverged"])
    fields.extend(["h2_add", "h2_ace", "c2_ace"])
    # panel (c) wiring columns (merged from bench_env_components.py)
    fields.extend(["n_fam", "m2"])
    for name in ("wire_add", "wire_oracle", "wire_fitted"):
        for met in ("corrg", "slopeg", "corro"):
            fields.extend([f"{name}_{met}", f"{name}_{met}_sd", f"{name}_{met}_se"])
    for comp in "ACM":
        fields.extend([f"wire_fit_{comp}", f"wire_fit_{comp}_sd"])
    return write_rows(os.path.join(HERE, "bench_shared_env.csv"), rows, fields)


def plot(rows_c2, rows_sib, h2):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    c2 = [r["c2"] for r in rows_c2]
    ax[0].errorbar(c2, [r["add_fit"] for r in rows_c2],
                   yerr=[r["add_fit_ci95"] for r in rows_c2], fmt="-o", capsize=3,
                   label="ignore C (fitted h²)")
    ax[0].errorbar(c2, [r["ace_fit"] for r in rows_c2],
                   yerr=[r["ace_fit_ci95"] for r in rows_c2], fmt="-o", capsize=3,
                   label="fit A+C")
    ax[0].errorbar(c2, [r["ace_oracle"] for r in rows_c2],
                   yerr=[r["ace_oracle_ci95"] for r in rows_c2], fmt="--", capsize=3,
                   color="gray", label="oracle A+C")
    ax[0].set_xlabel("true shared-environment c²")
    ax[0].set_ylabel("corr(estimate, true genetic liability)")
    ax[0].set_title(f"(a) accuracy vs c²  (h²={h2}, 3 sibs; 95% CI)")
    ax[0].legend(fontsize=8)
    ns = [r["n_sib"] for r in rows_sib]
    ax[1].errorbar(ns, [r["gain"] for r in rows_sib],
                   yerr=[r["gain_ci95"] for r in rows_sib], fmt="-o", capsize=3,
                   color="tab:green", label="fit A+C − ignore C")
    ax[1].errorbar(ns, [r["oracle_gain"] for r in rows_sib],
                   yerr=[r["oracle_gain_ci95"] for r in rows_sib], fmt="--", capsize=3,
                   color="gray", label="oracle − ignore C")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("number of full sibs")
    ax[1].set_ylabel("Δ corr accuracy gain")
    cf = rows_sib[0]["c2"] if rows_sib else ""
    ax[1].set_title(f"(b) gain from modelling C vs sib-ship size (c²={cf}; 95% CI)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_shared_env.png"), dpi=130)


if __name__ == "__main__":
    main()
