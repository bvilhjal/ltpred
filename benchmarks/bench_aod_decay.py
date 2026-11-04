"""Benchmark onset-age-decay genetic-correlation fitting.

Independent two-trait cohorts measure recovery of rho_g and decay rate lambda,
data requirements, the lambda=0 and rg=0 nulls, and robustness to the wrong
decay kernel and unmodelled shared environment. The likelihood M-step is
compared with its known truth; onset ages are independent per member-trait,
which accentuates the amplitude-decay ridge. `--no-robustness` skips the
misspecification panel. Writes `bench_aod_decay.csv`.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: `research`

from _common import simulate_families_multi, write_rows
from research.advanced_fitting import fit_genetic_correlation_decay

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.5]
RP_OFFDIAG = 0.3                      # phenotypic (full-liability) correlation
AGE_LO, AGE_HI = 15.0, 65.0
KERNEL = "ou"


def fit_replicates(rg_val, lam, n_fam, prev, reps, seed0, n_em, n_draw, burn,
                   m_iter):
    rg_fit = np.empty(reps)
    lamx_fit = np.empty(reps)
    lamw_fit = np.empty((reps, 2))
    for r in range(reps):
        fams = simulate_families_multi(FAM, H2, rg_val, RP_OFFDIAG, n_fam,
                                       (prev, prev), seed0 + r,
                                       aod=(AGE_LO, AGE_HI), lam=lam,
                                       kernel=KERNEL)
        res = fit_genetic_correlation_decay(fams, kernel=KERNEL, n_em=n_em,
                                            n_draw=n_draw, burn=burn,
                                            m_iter=m_iter,
                                            seed=seed0 + 100_000 + r,
                                            sampling="population")
        rg_fit[r] = res.rg[0, 1]
        lamx_fit[r] = res.lambda_cross[0, 1]
        lamw_fit[r] = res.lambda_within
    return rg_fit, lamx_fit, lamw_fit


def fit_robustness_replicates(true_kernel, lam, c2, n_fam, prev, reps, seed0,
                              n_em, n_draw, burn, m_iter):
    """Panel (d): truth may be misspecified (non-OU kernel, shared env c2);
    the fit is always the OU kernel with no c2 term."""
    rg_fit = np.empty(reps)
    h2_fit = np.empty((reps, 2))
    lamx_fit = np.empty(reps)
    for r in range(reps):
        fams = simulate_families_multi(FAM, H2, 0.5, RP_OFFDIAG, n_fam,
                                       (prev, prev), seed0 + r,
                                       aod=(AGE_LO, AGE_HI), lam=lam,
                                       kernel=true_kernel, c2=c2)
        res = fit_genetic_correlation_decay(fams, kernel="ou", n_em=n_em,
                                            n_draw=n_draw, burn=burn,
                                            m_iter=m_iter,
                                            seed=seed0 + 100_000 + r,
                                            sampling="population")
        rg_fit[r] = res.rg[0, 1]
        h2_fit[r] = res.h2
        lamx_fit[r] = res.lambda_cross[0, 1]
    return rg_fit, h2_fit, lamx_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.04])
    ap.add_argument("--n-fam", type=int, default=2500)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500])
    ap.add_argument("--prev", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-em", type=int, default=45)
    ap.add_argument("--n-draw", type=int, default=100)
    ap.add_argument("--burn", type=int, default=40)
    ap.add_argument("--m-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=200)
    # panel (d) misspecification robustness -- separate sizing so its defaults
    # stay comparable to the historical standalone robustness run (RESULTS §25)
    ap.add_argument("--robustness", action=argparse.BooleanOptionalAction,
                    default=True, help="run misspecification panel (d)")
    ap.add_argument("--rob-n-fam", type=int, default=2000)
    ap.add_argument("--rob-reps", type=int, default=2)
    ap.add_argument("--rob-n-em", type=int, default=35)
    ap.add_argument("--rob-n-draw", type=int, default=80)
    ap.add_argument("--rob-burn", type=int, default=30)
    ap.add_argument("--rob-m-iter", type=int, default=80)
    ap.add_argument("--rob-seed", type=int, default=31)
    ap.add_argument("--c2", type=float, default=0.10,
                    help="shared family environment for robustness arm B")
    args = ap.parse_args()

    # warm JIT / smoke on a tiny fit
    fit_genetic_correlation_decay(
        simulate_families_multi(FAM, H2, 0.5, 0.3, 40, (0.2, 0.2), 0,
                                aod=(AGE_LO, AGE_HI), lam=0.04, kernel=KERNEL),
        kernel=KERNEL, n_em=8, n_draw=20, burn=10, m_iter=20,
        sampling="population")

    rows = []
    lam_max = 4.0 / (AGE_HI - AGE_LO)
    print(f"kernel={KERNEL}  h2={H2}  r_p={RP_OFFDIAG}  lam_max={lam_max:.3f}  "
          f"n_em={args.n_em}")

    # (a) recovery vs true lambda (incl scalar null lam=0) -------------------
    print(f"\n== (a) recovery vs true lambda (rg=0.5, n_fam={args.n_fam}) ==")
    for lam in args.lams:
        t0 = time.time()
        rg_f, lx_f, lw_f = fit_replicates(
            0.5, lam, args.n_fam, args.prev, args.reps, args.seed, args.n_em,
            args.n_draw, args.burn, args.m_iter)
        rec = dict(lam=lam, rg_mean=float(rg_f.mean()), rg_sd=float(rg_f.std(ddof=1)),
                   lx_mean=float(lx_f.mean()), lx_sd=float(lx_f.std(ddof=1)),
                   lw_mean=float(lw_f.mean()))
        rows.append(dict(panel="vs_lambda", n_fam=args.n_fam, **rec,
                         reps=args.reps))
        tag = " (scalar null)" if lam == 0.0 else ""
        print(f"  lam={lam:.3f}{tag} | rg={rec['rg_mean']:+.3f}±{rec['rg_sd']:.3f} "
              f"(true .5)  lam_x={rec['lx_mean']:.4f}±{rec['lx_sd']:.4f} "
              f"(true {lam})  [{time.time()-t0:.0f}s]")

    # (b) data requirement (rg=0.5, lam=0.04) --------------------------------
    print("\n== (b) data requirement (rg=0.5, lam=0.04) ==")
    for n in args.sizes:
        rg_f, lx_f, lw_f = fit_replicates(
            0.5, 0.04, n, args.prev, args.reps, args.seed + 1000, args.n_em,
            args.n_draw, args.burn, args.m_iter)
        rec = dict(n_fam=n, rg_mean=float(rg_f.mean()), rg_sd=float(rg_f.std(ddof=1)),
                   lx_mean=float(lx_f.mean()), lx_sd=float(lx_f.std(ddof=1)))
        rows.append(dict(panel="data_req", lam=0.04, **rec, reps=args.reps))
        print(f"  n_fam={n:5d} | rg={rec['rg_mean']:+.3f}±{rec['rg_sd']:.3f} "
              f"(true .5)  lam_x={rec['lx_mean']:.4f} (true .04)")

    # (c) null r_g = 0 --------------------------------------------------------
    print(f"\n== (c) null rg=0 (lam=0.04, n_fam={args.n_fam}) ==")
    rg_f, lx_f, lw_f = fit_replicates(
        0.0, 0.04, args.n_fam, args.prev, args.reps, args.seed + 2000, args.n_em,
        args.n_draw, args.burn, args.m_iter)
    rec = dict(lam=0.04, rg_mean=float(rg_f.mean()), rg_sd=float(rg_f.std(ddof=1)),
               lx_mean=float(lx_f.mean()), lx_sd=float(lx_f.std(ddof=1)))
    rows.append(dict(panel="null", n_fam=args.n_fam, **rec, reps=args.reps))
    print(f"  rg={rec['rg_mean']:+.3f}±{rec['rg_sd']:.3f} (true 0)  "
          f"lam_x={rec['lx_mean']:.4f} (true .04)")

    # (d) misspecification robustness -----------------------------------------
    if args.robustness:
        print(f"\n== (d) misspecification robustness (rg=0.5, "
              f"n_fam={args.rob_n_fam}) ==")
        arms = [
            ("A_kernel_misspec", dict(kernel="gauss", lam=0.05, c2=0.0),
             "truth gauss lam=0.05, fit ou"),
            ("B_shared_env", dict(kernel="ou", lam=0.04, c2=args.c2),
             f"truth ou lam=0.04 + shared env c2={args.c2}"),
            ("B0_reference", dict(kernel="ou", lam=0.04, c2=0.0),
             "truth ou lam=0.04, no shared env"),
        ]
        for name, kw, desc in arms:
            t0 = time.time()
            rg_f, h2_f, lx_f = fit_robustness_replicates(
                kw["kernel"], kw["lam"], kw["c2"], args.rob_n_fam, args.prev,
                args.rob_reps, args.rob_seed, args.rob_n_em, args.rob_n_draw,
                args.rob_burn, args.rob_m_iter)
            rec = dict(arm=name, desc=desc, lam=kw["lam"], c2=kw["c2"],
                       rg_mean=float(rg_f.mean()),
                       rg_sd=float(rg_f.std(ddof=1)) if args.rob_reps > 1
                       else float("nan"),
                       h2_0=float(h2_f[:, 0].mean()),
                       h2_1=float(h2_f[:, 1].mean()),
                       lx_mean=float(lx_f.mean()))
            rows.append(dict(panel="robustness", n_fam=args.rob_n_fam, **rec,
                             reps=args.rob_reps))
            print(f"  {name:18s} | rg={rec['rg_mean']:+.3f}  "
                  f"h2=[{rec['h2_0']:.3f},{rec['h2_1']:.3f}]  "
                  f"lam_x={rec['lx_mean']:.4f}   [{time.time()-t0:.0f}s]  ({desc})")

    write_csv(rows)
    print("\nwrote bench_aod_decay.csv")


def write_csv(rows):
    fields = ["panel", "lam", "n_fam", "rg_mean", "rg_sd", "lx_mean", "lx_sd",
              "lw_mean", "reps", "arm", "desc", "c2", "h2_0", "h2_1"]
    return write_rows(os.path.join(HERE, "bench_aod_decay.csv"), rows, fields)


if __name__ == "__main__":
    main()
