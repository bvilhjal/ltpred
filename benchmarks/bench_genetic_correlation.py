"""Genetic-correlation inference: does `fit_genetic_correlation` recover r_g?

`fit_genetic_correlation` estimates the genetic correlation between traits from
family case/control data by a cross-trait Haseman–Elston regression (the
multi-trait analogue of `fit_heritability`). As for the single-trait fits, the
properties that matter are **bias** and **across-dataset sampling variability**,
so this benchmark fits many independent simulated two-trait cohorts and reads the
spread of the fitted r_g as its sampling distribution. It reports:

  (a) **bias & precision vs true r_g** — including the **null (r_g = 0 with a
      non-zero phenotypic correlation)**, which must not manufacture a genetic
      correlation;
  (b) **precision vs #families** — the across-replicate SD of r_g as N grows.

Ground truth is the simulator's genetic-correlation matrix.

    python benchmarks/bench_genetic_correlation.py
    python benchmarks/bench_genetic_correlation.py --reps 40 --n-fam 4000
Writes bench_genetic_correlation.csv (+ .png if matplotlib is present).
"""

import os
import csv
import time
import argparse

import numpy as np

from _common import get_plt
from ltpred.covariance import construct_covmat_multi, correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from ltpred.fit import fit_genetic_correlation

HERE = os.path.dirname(os.path.abspath(__file__))

FAM = ["m", "f", "s1", "s2"]
H2 = [0.5, 0.4]
RP_OFFDIAG = 0.2                       # phenotypic (full-liability) correlation


def simulate_two_trait(fam_vec, h2_vec, rg_val, rp_val, n_fam, prev, seed):
    """Two-trait families under the LTM; each member gets one interval per trait."""
    P = len(h2_vec)
    rg = np.array([[1.0, rg_val], [rg_val, 1.0]])
    rp = np.array([[1.0, rp_val], [rp_val, 1.0]])
    cov = construct_covmat_multi(fam_vec=fam_vec, add_ind=True, genetic_corrmat=rg,
                                 full_corrmat=rp, h2_vec=np.asarray(h2_vec, float))
    roles = cov.roles
    k = len(roles) // P
    fam_roles = roles[:k]
    Sig, _ = correct_positive_definite(cov.matrix)
    rng = np.random.default_rng(seed)
    liab = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=n_fam)
    t = [float(liability_threshold(prev[p])) for p in range(P)]
    obs = [r for r in fam_roles if r != "g"]
    fams = []
    for i in range(n_fam):
        members = []
        for r in obs:
            lo, hi = [], []
            for p in range(P):
                case = liab[i, p * k + fam_roles.index(r)] > t[p]
                lo.append(t[p] if case else -np.inf)
                hi.append(np.inf if case else t[p])
            members.append(Member(role=r, lower=lo, upper=hi))
        fams.append(Family(fam_id=i, members=members))
    return fams


def fit_replicates(rg_val, n_fam, prev, reps, seed0, n_iter, burn_in):
    fitted = np.empty(reps)
    for r in range(reps):
        fams = simulate_two_trait(FAM, H2, rg_val, RP_OFFDIAG, n_fam, (prev, prev),
                                  seed0 + r)
        res = fit_genetic_correlation(fams, n_iter=n_iter, burn_in=burn_in, seed=1)
        fitted[r] = res.rg[0, 1]
    return fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--reps-scaling", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--burn-in", type=int, default=250)
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()

    fit_genetic_correlation(simulate_two_trait(FAM, H2, 0.5, 0.2, 60, (0.1, 0.1), 0),
                            n_iter=20, burn_in=5)   # warm JIT

    rows = []

    # (a) bias & precision vs true r_g (incl null) ------------------------------
    print("== (a) bias & precision vs true r_g (n_fam=%d, h2=%s, r_p=%.1f) =="
          % (args.n_fam, H2, RP_OFFDIAG))
    panel_a = []
    for rg in args.rg:
        t0 = time.time()
        fitted = fit_replicates(rg, args.n_fam, args.prev, args.reps, args.seed,
                                args.n_iter, args.burn_in)
        bias = float(fitted.mean() - rg)
        sd = float(fitted.std(ddof=1))
        panel_a.append(dict(rg=rg, bias=bias, sd=sd))
        rows.append(dict(panel="bias_precision", rg=rg, n_fam=args.n_fam,
                         bias=bias, sd=sd, reps=args.reps))
        tag = " (null)" if rg == 0.0 else ""
        print(f"  r_g={rg:.1f}{tag} | fitted={fitted.mean():+.3f}  bias={bias:+.3f}  "
              f"SD={sd:.3f}  [{time.time()-t0:.0f}s]")

    # (b) precision vs #families (r_g=0.5) --------------------------------------
    print("== (b) precision (SD of r_g) vs #families (r_g=0.5) ==")
    panel_b = []
    for n in args.sizes:
        fitted = fit_replicates(0.5, n, args.prev, args.reps_scaling, args.seed + 1000,
                                args.n_iter, args.burn_in)
        sd = float(fitted.std(ddof=1))
        panel_b.append(dict(n_fam=n, sd=sd))
        rows.append(dict(panel="scaling", rg=0.5, n_fam=n,
                         bias=float(fitted.mean() - 0.5), sd=sd, reps=args.reps_scaling))
        print(f"  n_fam={n:5d} | SD={sd:.3f}")

    write_csv(rows)
    plot(panel_a, panel_b)
    print("\nwrote bench_genetic_correlation.csv and bench_genetic_correlation.png")


def write_csv(rows):
    fields = ["panel", "rg", "n_fam", "bias", "sd", "reps"]
    path = os.path.join(HERE, "bench_genetic_correlation.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(panel_a, panel_b):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    rg = [r["rg"] for r in panel_a]
    fitted = [r["rg"] + r["bias"] for r in panel_a]
    sd = [r["sd"] for r in panel_a]
    ax[0].errorbar(rg, fitted, yerr=sd, fmt="o", capsize=4, label="fitted ± SD")
    lim = [min(rg) - 0.1, max(rg) + 0.1]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlabel("true r_g")
    ax[0].set_ylabel("fitted r_g")
    ax[0].set_title("(a) bias & precision (incl. null)")
    ax[0].legend(fontsize=8)
    n = np.array([r["n_fam"] for r in panel_b], float)
    sdb = np.array([r["sd"] for r in panel_b])
    ax[1].plot(n, sdb, "-o", label="across-replicate SD")
    ref = sdb[0] * np.sqrt(n[0] / n)
    ax[1].plot(n, ref, "k:", lw=1, label="∝ 1/√N")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("number of families")
    ax[1].set_ylabel("SD of fitted r_g")
    ax[1].set_title("(b) precision vs #families (r_g=0.5)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_genetic_correlation.png"), dpi=130)


if __name__ == "__main__":
    main()
