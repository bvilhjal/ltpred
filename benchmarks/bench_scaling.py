"""Runtime scaling: Gibbs vs Pearson-Aitken inference.

Pearson-Aitken's selling point over the Gibbs sampler is speed: it is a deterministic,
closed-form sweep with no Monte-Carlo draws. This measures wall-clock time for
both methods as a function of (a) the number of families and (b) the family size
(number of relatives conditioned on), on the same simulated data. Both back-ends
use the structure-grouped, Numba-parallel path, so this is an honest end-to-end
comparison. Both paths are warmed before measurement; each point is the median
of repeated timings. The array-level PA path is included separately so its
documented biobank-scale throughput is backed by this benchmark.

    python benchmarks/bench_scaling.py
    python benchmarks/bench_scaling.py --sizes 500 1000 2000 4000 8000
Writes bench_scaling.csv (+ .png if matplotlib is present).
"""

import os
import argparse
import time

import numpy as np

from _common import estimate, get_plt, simulate_families, write_rows
from ltpred import estimate_liability_pa_arrays

try:
    from numba import get_num_threads
except ImportError:  # pragma: no cover - pure-Python fallback environment
    def get_num_threads():
        return 1

HERE = os.path.dirname(os.path.abspath(__file__))

# family structures of growing size, for the family-size sweep
SIZE_STRUCTURES = [
    ("parents", ["m", "f"]),
    ("+sib", ["m", "f", "s1"]),
    ("+2 grandp.", ["m", "f", "s1", "mgm", "mgf"]),
    ("extended", ["m", "f", "s1", "mgm", "mgf", "pgm", "pgf"]),
    ("extended+aunts", ["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf", "mau1", "pau1"]),
]


def _bounds_arrays(families):
    roles = [m.role for m in families[0].members]
    lower = np.asarray([[m.lower for m in f.members] for f in families])
    upper = np.asarray([[m.upper for m in f.members] for f in families])
    return roles, lower, upper


def _summary(values):
    values = np.asarray(values, float)
    return (float(np.median(values)), float(np.quantile(values, 0.25)),
            float(np.quantile(values, 0.75)))


def _time_object(families, h2, method, n_sim, seed, timing_reps):
    times = []
    for r in range(timing_reps):
        kwargs = dict(n_sim=n_sim, seed=seed + r) if method == "gibbs" else {}
        start = time.perf_counter()
        estimate(families, h2, method, **kwargs)
        times.append(time.perf_counter() - start)
    return _summary(times)


def _time_array_pa(families, h2, timing_reps, min_window=0.05):
    roles, lower, upper = _bounds_arrays(families)
    samples = []
    for _ in range(timing_reps):
        calls = 0
        start = time.perf_counter()
        elapsed = 0.0
        while elapsed < min_window:
            estimate_liability_pa_arrays(roles, lower, upper, h2=h2)
            calls += 1
            elapsed = time.perf_counter() - start
        samples.append(elapsed / calls)
    return _summary(samples)


def _timing_row(families, h2, n_sim, seed, timing_reps):
    tg, tg25, tg75 = _time_object(families, h2, "gibbs", n_sim, seed,
                                  timing_reps)
    tp, tp25, tp75 = _time_object(families, h2, "pearson-aitken", n_sim,
                                  seed, timing_reps)
    ta, ta25, ta75 = _time_array_pa(families, h2, timing_reps)
    return dict(t_gibbs=tg, t_gibbs_q25=tg25, t_gibbs_q75=tg75,
                t_pa=tp, t_pa_q25=tp25, t_pa_q75=tp75,
                t_pa_array=ta, t_pa_array_q25=ta25, t_pa_array_q75=ta75,
                timing_reps=timing_reps, threads=int(get_num_threads()))


def scan_n_fam(sizes, h2, prev, n_sim, seed, timing_reps):
    rows = []
    for n in sizes:
        sim = simulate_families(["m", "f", "s1"], h2, prev, n, seed)
        timing = _timing_row(sim.families, h2, n_sim, seed, timing_reps)
        t_g, t_p, t_a = timing["t_gibbs"], timing["t_pa"], timing["t_pa_array"]
        rows.append(dict(axis="n_fam", value=n, n_relatives=3, h2=h2,
                         prevalence=prev, n_sim=n_sim, seed=seed,
                         **timing,
                         fam_per_s_gibbs=n / t_g, fam_per_s_pa=n / t_p,
                         fam_per_s_pa_array=n / t_a, speedup=t_g / t_p,
                         pa_array_speedup=t_p / t_a))
        print(f"n_fam={n:6d} | Gibbs {t_g:7.2f}s ({n/t_g:7.0f}/s) | "
              f"PA {t_p:6.3f}s ({n/t_p:9.0f}/s) | array {t_a:7.4f}s "
              f"({n/t_a:10.0f}/s) | {t_g/t_p:5.0f}x")
    return rows


def scan_family_size(n_fam, h2, prev, n_sim, seed, timing_reps):
    rows = []
    for label, fam_vec in SIZE_STRUCTURES:
        sim = simulate_families(fam_vec, h2, prev, n_fam, seed)
        timing = _timing_row(sim.families, h2, n_sim, seed, timing_reps)
        t_g, t_p, t_a = timing["t_gibbs"], timing["t_pa"], timing["t_pa_array"]
        rows.append(dict(axis="family_size", value=len(fam_vec),
                         n_relatives=len(fam_vec), label=label, h2=h2,
                         prevalence=prev, n_sim=n_sim, seed=seed, **timing,
                         fam_per_s_gibbs=n_fam / t_g, fam_per_s_pa=n_fam / t_p,
                         fam_per_s_pa_array=n_fam / t_a, speedup=t_g / t_p,
                         pa_array_speedup=t_p / t_a))
        print(f"{label:16s} ({len(fam_vec)} rel) | Gibbs {t_g:7.2f}s | "
              f"PA {t_p:6.3f}s | array {t_a:7.4f}s | {t_g/t_p:5.0f}x")
    return rows


def write_csv(rows):
    fields = sorted({k for r in rows for k in r})
    return write_rows(os.path.join(HERE, "bench_scaling.csv"), rows, fields)


def plot(nfam_rows, size_rows):
    plt = get_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    def timing_error(rows, key):
        values = np.asarray([r[key] for r in rows])
        q25 = np.asarray([r[f"{key}_q25"] for r in rows])
        q75 = np.asarray([r[f"{key}_q75"] for r in rows])
        return np.vstack((values - q25, q75 - values))

    n = [r["value"] for r in nfam_rows]
    for key, label, marker in (("t_gibbs", "Gibbs", "o"),
                               ("t_pa", "PA object", "s"),
                               ("t_pa_array", "PA array", "^")):
        axes[0].errorbar(n, [r[key] for r in nfam_rows],
                         yerr=timing_error(nfam_rows, key), fmt=f"-{marker}",
                         capsize=2, label=label)
    axes[0].set_xlabel("number of families")
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) scaling: parents + one sibling")
    axes[0].legend()
    sz = [r["value"] for r in size_rows]
    for key, label, marker in (("t_gibbs", "Gibbs", "o"),
                               ("t_pa", "PA object", "s"),
                               ("t_pa_array", "PA array", "^")):
        axes[1].errorbar(sz, [r[key] for r in size_rows],
                         yerr=timing_error(size_rows, key), fmt=f"-{marker}",
                         capsize=2, label=label)
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
    ap.add_argument("--timing-reps", type=int, default=5,
                    help="independent timings per point; CSV stores median and IQR")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.timing_reps < 1:
        ap.error("--timing-reps must be at least 1")

    # Compile every measured path before the first stopwatch starts.
    warm = simulate_families(["m", "f", "s1"], args.h2, args.prev, 64,
                             args.seed + 10_000)
    estimate(warm.families, args.h2, "gibbs", n_sim=1_000,
             seed=args.seed + 10_000)
    estimate(warm.families, args.h2, "pearson-aitken")
    wr, wl, wu = _bounds_arrays(warm.families)
    estimate_liability_pa_arrays(wr, wl, wu, h2=args.h2)

    print(f"threads={get_num_threads()} timing_reps={args.timing_reps}")
    print("== scaling with number of families (parents + one sibling; "
          "three relatives) ==")
    nfam_rows = scan_n_fam(args.sizes, args.h2, args.prev, args.n_sim, args.seed,
                           args.timing_reps)
    print("\n== scaling with family size (n_fam=%d) ==" % args.size_n_fam)
    size_rows = scan_family_size(args.size_n_fam, args.h2, args.prev, args.n_sim,
                                 args.seed, args.timing_reps)

    path = write_csv(nfam_rows + size_rows)
    plot(nfam_rows, size_rows)
    print(f"\nwrote {os.path.basename(path)} and bench_scaling.png")


if __name__ == "__main__":
    main()
