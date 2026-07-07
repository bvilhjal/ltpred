"""Runtime scaling: Gibbs (LT-FH++) vs Pearson-Aitken (PA-FGRS).

PA-FGRS's selling point over the Gibbs sampler is speed: it is a deterministic,
closed-form sweep with no Monte-Carlo draws. This measures wall-clock time for
both methods as a function of (a) the number of families and (b) the family size
(number of relatives conditioned on), on the same simulated data. Both back-ends
use the structure-grouped, Numba-parallel path, so this is an honest end-to-end
comparison. Reports families/second for each.

    python benchmarks/bench_scaling.py
    python benchmarks/bench_scaling.py --sizes 500 1000 2000 4000 8000
Writes bench_scaling.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np

from _common import simulate_families, estimate, get_plt

HERE = os.path.dirname(os.path.abspath(__file__))

# family structures of growing size, for the family-size sweep
SIZE_STRUCTURES = [
    ("parents", ["m", "f"]),
    ("+sib", ["m", "f", "s1"]),
    ("+2 grandp.", ["m", "f", "s1", "mgm", "mgf"]),
    ("extended", ["m", "f", "s1", "mgm", "mgf", "pgm", "pgf"]),
    ("extended+aunts", ["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf", "mau1", "pau1"]),
]


def scan_n_fam(sizes, h2, prev, n_sim, seed):
    rows = []
    for n in sizes:
        sim = simulate_families(["m", "f", "s1"], h2, prev, n, seed)
        _, t_g = estimate(sim.families, h2, "gibbs", n_sim=n_sim, seed=seed)
        _, t_p = estimate(sim.families, h2, "pearson-aitken")
        rows.append(dict(axis="n_fam", value=n, n_relatives=3,
                         t_gibbs=t_g, t_pa=t_p,
                         fam_per_s_gibbs=n / t_g, fam_per_s_pa=n / t_p,
                         speedup=t_g / t_p))
        print(f"n_fam={n:6d} | Gibbs {t_g:7.2f}s ({n/t_g:7.0f}/s) | "
              f"PA {t_p:6.3f}s ({n/t_p:9.0f}/s) | {t_g/t_p:5.0f}x")
    return rows


def scan_family_size(n_fam, h2, prev, n_sim, seed):
    rows = []
    for label, fam_vec in SIZE_STRUCTURES:
        sim = simulate_families(fam_vec, h2, prev, n_fam, seed)
        _, t_g = estimate(sim.families, h2, "gibbs", n_sim=n_sim, seed=seed)
        _, t_p = estimate(sim.families, h2, "pearson-aitken")
        rows.append(dict(axis="family_size", value=len(fam_vec), n_relatives=len(fam_vec),
                         label=label, t_gibbs=t_g, t_pa=t_p,
                         fam_per_s_gibbs=n_fam / t_g, fam_per_s_pa=n_fam / t_p,
                         speedup=t_g / t_p))
        print(f"{label:16s} ({len(fam_vec)} rel) | Gibbs {t_g:7.2f}s | "
              f"PA {t_p:6.3f}s | {t_g/t_p:5.0f}x")
    return rows


def write_csv(rows):
    fields = sorted({k for r in rows for k in r})
    path = os.path.join(HERE, "bench_scaling.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def plot(nfam_rows, size_rows):
    plt = get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    n = [r["value"] for r in nfam_rows]
    axes[0].plot(n, [r["t_gibbs"] for r in nfam_rows], "-o", label="Gibbs (LT-FH++)")
    axes[0].plot(n, [r["t_pa"] for r in nfam_rows], "-o", label="PA-FGRS")
    axes[0].set_xlabel("number of families")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) scaling with #families (trios)")
    axes[0].legend()
    sz = [r["value"] for r in size_rows]
    axes[1].plot(sz, [r["t_gibbs"] for r in size_rows], "-o", label="Gibbs (LT-FH++)")
    axes[1].plot(sz, [r["t_pa"] for r in size_rows], "-o", label="PA-FGRS")
    axes[1].set_xlabel("relatives per family")
    axes[1].set_ylabel("wall time (s)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) scaling with family size")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_scaling.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--size-n-fam", type=int, default=2000)
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--prev", type=float, default=0.05)
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("== scaling with number of families (trios) ==")
    nfam_rows = scan_n_fam(args.sizes, args.h2, args.prev, args.n_sim, args.seed)
    print("\n== scaling with family size (n_fam=%d) ==" % args.size_n_fam)
    size_rows = scan_family_size(args.size_n_fam, args.h2, args.prev, args.n_sim, args.seed)

    path = write_csv(nfam_rows + size_rows)
    plot(nfam_rows, size_rows)
    print(f"\nwrote {os.path.basename(path)} and bench_scaling.png")


if __name__ == "__main__":
    main()
