"""Calibration of the estimated genetic liability — not just its ranking.

Every other accuracy benchmark scores the estimate by ``corr(estimate, true g)``,
which measures **ranking** only. But the estimate is a posterior *mean*, so it also
has a **scale**: a well-specified posterior mean is *self-calibrating* — regressing
the true genetic liability on the estimate gives slope 1 and intercept 0 (because
``E[g | ĝ] = ĝ`` when ``ĝ = E[g | data]``). This benchmark checks that scale, which
is what matters if the score is used for **risk stratification** (who is in the top
decile of genetic liability?) rather than only as a GWAS phenotype (where it is
standardised anyway). Truth ``g`` is known from the simulation, so it reports:

  (a) **calibration when correctly specified** — slope, intercept and a decile
      calibration curve (mean true g vs mean estimate per decile) for Gibbs and PA;
      both should sit on the diagonal;
  (b) **calibration vs an assumed-h² that is wrong** — sweep the ``h²`` handed to the
      estimator away from the truth. The headline: ``corr`` barely moves (ranking is
      robust, as ``liability_sensitivity`` shows) while the **slope tilts** — assume
      too much h² and the estimate over-spreads (slope < 1), too little and it
      under-spreads (slope > 1). Ranking is robust; *scale is not*;
  (c) **decile curves under that misspecification** — the calibration curve tilting
      off the diagonal as the assumed h² moves.

    python benchmarks/bench_calibration.py
    python benchmarks/bench_calibration.py --n-fam 5000 --true-h2 0.5
Writes bench_calibration.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np

from _common import simulate_families, estimate, get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCTURES = {
    "parents+sibs": ["m", "f", "s1", "s2"],
    "extended": ["m", "f", "s1", "mgm", "mgf", "pgm", "pgf"],
}


def decile_curve(est, true_g, n_bins=10):
    """Per-quantile-bin (mean estimate, mean true g). On the diagonal = calibrated."""
    order = np.argsort(est)
    bins = np.array_split(order, n_bins)
    me = np.array([est[b].mean() for b in bins])
    mg = np.array([true_g[b].mean() for b in bins])
    return me, mg


def calib(est, true_g, n_bins=10):
    """Calibration metrics of an estimate against the known truth ``true_g``.

    ``slope``/``intercept`` regress ``true_g`` on ``est`` (slope 1, intercept 0 =
    calibrated). ``cal_rmse`` is the RMS gap between per-decile mean(true g) and
    mean(est). ``top_ratio`` is realised/predicted in the top decile — mean(true g)
    over mean(est) there, so < 1 means the top of the score is over-stated."""
    corr = float(np.corrcoef(est, true_g)[0, 1])
    slope, intercept = np.polyfit(est, true_g, 1)
    me, mg = decile_curve(est, true_g, n_bins)
    cal_rmse = float(np.sqrt(np.mean((mg - me) ** 2)))
    top_ratio = float(mg[-1] / me[-1]) if abs(me[-1]) > 1e-9 else float("nan")
    return dict(corr=corr, slope=float(slope), intercept=float(intercept),
                cal_rmse=cal_rmse, top_ratio=top_ratio)


def panel_correct(n_fam, h2, prevs, n_sim, seed):
    """(a) Calibration of the correctly-specified estimate (Gibbs and PA)."""
    print("== (a) calibration when correctly specified (h2=%.2f) ==" % h2)
    rows = []
    for sname, fam_vec in STRUCTURES.items():
        for prev in prevs:
            sim = simulate_families(fam_vec, h2, prev, n_fam, seed)
            g = sim.genetic
            gib, _ = estimate(sim.families, h2, "gibbs", n_sim=n_sim, seed=seed)
            pa, _ = estimate(sim.families, h2, "pearson-aitken")
            cg, cp = calib(gib, g), calib(pa, g)
            print("  %-13s K=%.2f | slope Gibbs=%.3f PA=%.3f | intercept=%+.3f | "
                  "corr=%.3f | top-decile realised/pred=%.3f"
                  % (sname, prev, cg["slope"], cp["slope"], cg["intercept"],
                     cg["corr"], cg["top_ratio"]))
            rows.append(dict(panel="correct", structure=sname, prevalence=prev,
                             assumed_h2=h2, true_h2=h2,
                             slope_gibbs=cg["slope"], slope_pa=cp["slope"],
                             intercept_gibbs=cg["intercept"],
                             intercept_pa=cp["intercept"],
                             corr_gibbs=cg["corr"], corr_pa=cp["corr"],
                             cal_rmse_gibbs=cg["cal_rmse"],
                             cal_rmse_pa=cp["cal_rmse"],
                             top_ratio_gibbs=cg["top_ratio"],
                             top_ratio_pa=cp["top_ratio"]))
    return rows


def panel_misspec(n_fam, true_h2, assumed_grid, prev, seed):
    """(b) Calibration vs a wrong assumed h2 (PA; Gibbs agrees). Ranking robust,
    scale not."""
    print("== (b) calibration vs assumed h2 (true h2=%.2f, K=%.2f, parents+2 sibs) =="
          % (true_h2, prev))
    fam_vec = STRUCTURES["parents+sibs"]
    sim = simulate_families(fam_vec, true_h2, prev, n_fam, seed)
    g = sim.genetic
    rows = []
    for a_h2 in assumed_grid:
        est, _ = estimate(sim.families, a_h2, "pearson-aitken")
        c = calib(est, g)
        flag = "  <- true" if abs(a_h2 - true_h2) < 1e-9 else ""
        print("  assumed h2=%.2f | slope=%.3f  corr=%.3f  top-decile r/p=%.3f%s"
              % (a_h2, c["slope"], c["corr"], c["top_ratio"], flag))
        rows.append(dict(panel="misspec", structure="parents+sibs", prevalence=prev,
                         assumed_h2=a_h2, true_h2=true_h2, slope_gibbs="",
                         slope_pa=c["slope"], intercept_gibbs="",
                         intercept_pa=c["intercept"], corr_gibbs="",
                         corr_pa=c["corr"], cal_rmse_gibbs="",
                         cal_rmse_pa=c["cal_rmse"], top_ratio_gibbs="",
                         top_ratio_pa=c["top_ratio"]))
    return rows, sim, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--true-h2", type=float, default=0.5)
    ap.add_argument("--prev", type=float, nargs="+", default=[0.01, 0.05, 0.20])
    ap.add_argument("--assumed", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--misspec-prev", type=float, default=0.05)
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    estimate(simulate_families(["m", "f", "s1"], 0.5, 0.1, 40, 0).families, 0.5,
             "gibbs", n_sim=500, seed=0)                 # warm JIT

    rows = panel_correct(args.n_fam, args.true_h2, args.prev, args.n_sim, args.seed)
    mrows, sim, g = panel_misspec(args.n_fam, args.true_h2, args.assumed,
                                  args.misspec_prev, args.seed)
    rows += mrows

    # (c) decile calibration curves under misspecification
    curves = {}
    for a_h2 in (min(args.assumed), args.true_h2, max(args.assumed)):
        est, _ = estimate(sim.families, a_h2, "pearson-aitken")
        curves[a_h2] = decile_curve(est, g)

    write_csv(rows)
    plot(rows, args.true_h2, curves)
    print("\nwrote bench_calibration.csv and bench_calibration.png")


def write_csv(rows):
    fields = ["panel", "structure", "prevalence", "assumed_h2", "true_h2",
              "slope_gibbs", "slope_pa", "intercept_gibbs", "intercept_pa",
              "corr_gibbs", "corr_pa", "cal_rmse_gibbs", "cal_rmse_pa",
              "top_ratio_gibbs", "top_ratio_pa"]
    path = os.path.join(HERE, "bench_calibration.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(rows, true_h2, curves):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) well-specified decile curve for the true-h2 misspec cell (Gibbs≈PA)
    me, mg = curves[true_h2]
    lim = [min(me.min(), mg.min()) - 0.03, max(me.max(), mg.max()) + 0.03]
    ax[0].plot(lim, lim, "k--", lw=1, label="calibrated (y=x)")
    ax[0].plot(me, mg, "-o", color="#2F7D4F", label="decile means")
    ax[0].set_xlabel("mean estimate (per decile)")
    ax[0].set_ylabel("mean true g (per decile)")
    ax[0].set_title("(a) correctly specified: on the diagonal")
    ax[0].legend(fontsize=8)

    # (b) slope & corr vs assumed h2
    ms = [r for r in rows if r["panel"] == "misspec"]
    ms.sort(key=lambda r: r["assumed_h2"])
    a = [r["assumed_h2"] for r in ms]
    slope = [r["slope_pa"] for r in ms]
    corr = [r["corr_pa"] for r in ms]
    ax[1].plot(a, slope, "-o", color="#B95C3C", label="calibration slope")
    ax[1].plot(a, corr, "-s", color="#3B4A9C", label="corr(est, true g)")
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].axvline(true_h2, color="gray", ls=":", lw=1)
    ax[1].set_xlabel("assumed h² handed to the estimator")
    ax[1].set_ylabel("slope / corr")
    ax[1].set_title("(b) ranking robust, scale is not")
    ax[1].legend(fontsize=8)

    # (c) decile curves under misspecification
    lo = min(curves); hi = max(curves)
    allv = np.concatenate([np.concatenate(curves[k]) for k in curves])
    lim = [allv.min() - 0.03, allv.max() + 0.03]
    ax[2].plot(lim, lim, "k--", lw=1)
    colors = {lo: "#3B4A9C", true_h2: "#2F7D4F", hi: "#B95C3C"}
    for k in (lo, true_h2, hi):
        me, mg = curves[k]
        tag = " (true)" if abs(k - true_h2) < 1e-9 else ""
        ax[2].plot(me, mg, "-o", color=colors[k], label=f"assumed h²={k:.1f}{tag}")
    ax[2].set_xlabel("mean estimate (per decile)")
    ax[2].set_ylabel("mean true g (per decile)")
    ax[2].set_title("(c) misspecified h² tilts the calibration")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_calibration.png"), dpi=130)


if __name__ == "__main__":
    main()
