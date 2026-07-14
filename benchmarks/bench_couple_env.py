"""The couple / spousal environment component `M`: recovery, and why it behaves
differently from the sibship component `C`.

`fit_variance_components` fits a **bank** of relationship-specific shared-environment
components. Besides the sibship component `C` (full-sib rearing environment), it
ships `M`, a **couple / spousal** environment that loads on the genetically-unrelated
mate pairs — the proband's parents `(m, f)` and the grandparent couples. `M` captures
spousal resemblance from shared adult environment *or* assortative mating (parent
data alone cannot separate the two).

The two shared-environment components differ in a way that matters. Sibs share both
genes and `C` (`A = 0.5`), so **ignoring a real `C` inflates the additive estimate**
— sib resemblance is over-credited to genetics. Mates share `M` but no genes
(`A = 0`), so they carry ~zero weight in the additive regression: **ignoring a real
`M` biases the additive estimate much less than ignoring `C`** (a residual can remain
from imputing under the misspecified model, especially when `M` is large). `M` is
worth fitting for its own sake (quantifying / testing spousal resemblance), not only
to de-bias `h²`.

This benchmark makes both points empirically on simulated data (liabilities drawn
from `a2 A + s2 K + e2 I`, thresholded, so ground truth is known):

  (a) **A+M recovery** — sweep the true couple variance `m2`; fit `("A", "M")` over
      many independent cohorts and read mean(fitted) - true and the across-replicate
      SD. At `m2 = 0`, report the small positive boundary estimate expected from
      the nonnegative constraint; it is not a hypothesis-test rejection rate.
  (b) **bias from ignoring shared environment** — put the *same* shared-environment
      variance `s2` in once as `C` (sibship) and once as `M` (couple), then fit the
      **additive-only** model `("A",)`. The `C` version inflates `A`; the `M`
      version is much less biased. This is the identifiability contrast the theory
      predicts (algorithm.md, *Relationship-specific environments and identifiability*).

    python benchmarks/bench_couple_env.py
    python benchmarks/bench_couple_env.py --reps 40 --n-fam 4000
Writes bench_couple_env.csv (+ .png if matplotlib is present).
"""

import os
import csv
import time
import argparse

import numpy as np

from _common import get_plt
from ltpred.covariance import correct_positive_definite
from ltpred.thresholds import liability_threshold
from ltpred.family import Family, Member
from ltpred.fit import _component_matrix, fit_variance_components

HERE = os.path.dirname(os.path.abspath(__file__))

# grandparents give 3 mate pairs per family (m/f, mgm/mgf, pgm/pgf) so M is powered;
# the full sibs (o, s1, s2) power C for the (b) contrast.
STRUCT = ["o", "s1", "s2", "m", "f", "mgm", "mgf", "pgm", "pgf"]


def simulate_vc(roles, props, n_fam, prev, seed):
    """Families with liabilities ~ N(0, sum_c props[c] K_c + e2 I), thresholded.

    ``props`` maps a component ('A', 'C', 'M') to its variance proportion."""
    roles = list(roles)
    tot = sum(props.values())
    Sig = (1.0 - tot) * np.eye(len(roles))
    for c, v in props.items():
        if v > 0:
            Sig = Sig + v * _component_matrix(roles, c)
    Sig, _ = correct_positive_definite(Sig)
    rng = np.random.default_rng(seed)
    liab = rng.multivariate_normal(np.zeros(len(roles)), Sig, size=n_fam)
    t = float(liability_threshold(prev))
    fams = []
    for i in range(n_fam):
        members = [Member(role=r, lower=(t if liab[i, c] > t else -np.inf),
                          upper=(np.inf if liab[i, c] > t else t))
                   for c, r in enumerate(roles)]
        fams.append(Family(fam_id=i, members=members))
    return fams


def fit_replicates(roles, props, comps, n_fam, prev, reps, seed0, n_iter, burn_in):
    """Fit `reps` independent cohorts under `props`; return the fitted-component dict
    of arrays keyed by the components in `comps`."""
    out = {c: np.empty(reps) for c in comps}
    for r in range(reps):
        fams = simulate_vc(roles, props, n_fam, prev, seed0 + r)
        # Use a reproducible but distinct inference stream for every cohort. A fixed
        # fitter seed would suppress one source of across-replicate Monte-Carlo
        # variation and synchronize all chains unnecessarily.
        fit_seed = 1_000_000 + seed0 + r
        res = fit_variance_components(
            fams, comps, n_iter=n_iter, burn_in=burn_in, seed=fit_seed
        )
        for c in comps:
            out[c][r] = res.components[c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--m2", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--s2", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--a2", type=float, default=0.4)
    ap.add_argument("--prev", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--n-iter", type=int, default=800)
    ap.add_argument("--burn-in", type=int, default=250)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    if not np.isfinite(args.a2) or not 0.0 <= args.a2 <= 1.0:
        ap.error("--a2 must be finite and between 0 and 1")
    for flag, values in (("--m2", args.m2), ("--s2", args.s2)):
        for value in values:
            if not np.isfinite(value) or value < 0.0:
                ap.error(f"{flag} values must be finite and nonnegative")
            if args.a2 + value > 1.0:
                ap.error(f"--a2 + {flag} must be <= 1 (got {args.a2 + value:g})")

    a2 = args.a2
    kw = dict(n_fam=args.n_fam, prev=args.prev, reps=args.reps,
              n_iter=args.n_iter, burn_in=args.burn_in)

    # warm the JIT
    fit_variance_components(simulate_vc(STRUCT, {"A": 0.4, "M": 0.2}, 60, args.prev, 0),
                            ("A", "M"), n_iter=20, burn_in=5)

    rows = []

    # (a) A+M recovery ----------------------------------------------------------
    print("== (a) A+M recovery (a2=%.1f, n_fam=%d, %s) ==" % (a2, args.n_fam, "+".join(STRUCT)))
    panel_a = []
    for m2 in args.m2:
        t0 = time.time()
        f = fit_replicates(STRUCT, {"A": a2, "M": m2}, ("A", "M"), seed0=args.seed, **kw)
        fa, fm = f["A"], f["M"]
        rec = dict(m2=m2, A_mean=float(fa.mean()), A_bias=float(fa.mean() - a2),
                   A_sd=float(fa.std(ddof=1)), M_mean=float(fm.mean()),
                   M_bias=float(fm.mean() - m2), M_sd=float(fm.std(ddof=1)))
        panel_a.append(rec)
        rows.append(dict(panel="recovery", a2=a2, **rec, n_fam=args.n_fam, reps=args.reps))
        print(f"  m2={m2:.1f} | A={fa.mean():.3f}({rec['A_bias']:+.3f}) SD={rec['A_sd']:.3f}"
              f" | M={fm.mean():.3f}({rec['M_bias']:+.3f}) SD={rec['M_sd']:.3f}  [{time.time()-t0:.0f}s]")

    # (b) bias in additive-only fit from ignoring C vs M ------------------------
    print("== (b) additive-only A bias from ignoring shared environment ==")
    print("   (same s2 variance placed as C (sibship) vs M (couple); fit ('A',) only)")
    panel_b = []
    for s2 in args.s2:
        fc = fit_replicates(STRUCT, {"A": a2, "C": s2}, ("A",), seed0=args.seed + 300, **kw)["A"]
        fm = fit_replicates(STRUCT, {"A": a2, "M": s2}, ("A",), seed0=args.seed + 600, **kw)["A"]
        rec = dict(s2=s2, A_ignoreC=float(fc.mean()), A_ignoreC_bias=float(fc.mean() - a2),
                   A_ignoreC_sd=float(fc.std(ddof=1)), A_ignoreM=float(fm.mean()),
                   A_ignoreM_bias=float(fm.mean() - a2),
                   A_ignoreM_sd=float(fm.std(ddof=1)))
        panel_b.append(rec)
        rows.append(dict(panel="ignore_bias", a2=a2, **rec, n_fam=args.n_fam, reps=args.reps))
        print(f"  s2={s2:.1f} | ignore C: A={fc.mean():.3f} "
              f"({rec['A_ignoreC_bias']:+.3f}, SD={rec['A_ignoreC_sd']:.3f}; inflated)"
              f" | ignore M: A={fm.mean():.3f} "
              f"({rec['A_ignoreM_bias']:+.3f}, SD={rec['A_ignoreM_sd']:.3f}; much less biased)")

    write_csv(rows)
    plot(panel_a, panel_b, a2)
    print("\nwrote bench_couple_env.csv and bench_couple_env.png")


def write_csv(rows):
    fields = ["panel", "a2", "m2", "s2", "A_mean", "A_bias", "A_sd", "M_mean",
              "M_bias", "M_sd", "A_ignoreC", "A_ignoreC_bias", "A_ignoreC_sd",
              "A_ignoreM", "A_ignoreM_bias", "A_ignoreM_sd", "n_fam", "reps"]
    path = os.path.join(HERE, "bench_couple_env.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(panel_a, panel_b, a2):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) M recovery vs true m2
    m2 = [r["m2"] for r in panel_a]
    ax[0].errorbar(m2, [r["M_mean"] for r in panel_a], yerr=[r["M_sd"] for r in panel_a],
                   fmt="o", capsize=3, label="M fitted")
    ax[0].plot([min(m2), max(m2)], [min(m2), max(m2)], "k:", lw=1, label="M true (y=x)")
    ax[0].errorbar(m2, [r["A_mean"] for r in panel_a], yerr=[r["A_sd"] for r in panel_a],
                   fmt="s", capsize=3, color="tab:green", label="A fitted")
    ax[0].axhline(a2, color="tab:green", ls=":", lw=1, label=f"A true ({a2})")
    ax[0].set_xlabel("true couple variance m²")
    ax[0].set_ylabel("fitted proportion")
    ax[0].set_title("(a) A+M recovery")
    ax[0].legend(fontsize=8)
    # (b) additive-only A bias: ignore C vs ignore M
    s2 = [r["s2"] for r in panel_b]
    ax[1].errorbar(s2, [r["A_ignoreC_bias"] for r in panel_b],
                   yerr=[r["A_ignoreC_sd"] for r in panel_b], fmt="-o", capsize=3,
                   color="tab:red", label="ignore C (sibship) → A inflated (± SD)")
    ax[1].errorbar(s2, [r["A_ignoreM_bias"] for r in panel_b],
                   yerr=[r["A_ignoreM_sd"] for r in panel_b], fmt="-o", capsize=3,
                   color="tab:blue", label="ignore M (couple) → much less bias (± SD)")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("true shared-environment variance s²")
    ax[1].set_ylabel("bias in additive-only Â  (fitted − true)")
    ax[1].set_title("(b) why C and M differ: bias from omitting each")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_couple_env.png"), dpi=130)


if __name__ == "__main__":
    main()
