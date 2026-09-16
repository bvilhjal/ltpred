"""Locked comparison of ltpred against public R LTFHPlus and LTFGRS.

LTFHPlus 2.2.0 is Gibbs-only (no ``method`` argument). Public Pearson-Aitken
lives in LTFGRS 1.0.1 (``estimate_liability(..., method="PA")``). This script
feeds the *same* simulated families and bounds to four isolated estimator
processes:

  * LTFHPlus::estimate_liability (Rcpp Gibbs, n_sim=1e5, burn_in=1000)
  * LTFGRS::estimate_liability(method="PA", useMixture=FALSE)
  * ltpred Gibbs with the same tol / n_sim / burn_in
  * ltpred Pearson-Aitken (no mixture)

Score agreement (correlation, RMSE, max abs) is a software lock, not the
intra-package PA-vs-Gibbs grid in ``bench_scaling.py``.

Wall-clock is the estimator call after in-process warmup, reported as a
cohort total and as milliseconds per family (total / n_fam). Peak RSS uses
ldpred3's isolated-process pattern: a stdlib-only launcher
(``_peak_launcher.py``) forks the command so ``wait4``'s ``ru_maxrss`` is the
child's own high-water mark, not the fat driver's inherited floor.

Requires R with LTFHPlus. LTFGRS is required for the PA arm; if it is
missing the script still locks LTFHPlus Gibbs and writes empty LTFGRS
columns. Exits 2 if R or LTFHPlus is missing::

    install.packages(c("LTFHPlus", "LTFGRS"))
    python benchmarks/bench_ltfhplus_compare.py
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import sys
import tempfile

import numpy as np

from _common import estimate, simulate_families, write_rows
from _peak_launcher import run_peak

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R_LTFHPLUS = os.path.join(HERE, "ltfhplus_compare.R")
R_LTFGRS = os.path.join(HERE, "ltfgrs_compare.R")

try:
    import numba
    from numba import get_num_threads
    NUMBA_VERSION = numba.__version__
except ImportError:  # pragma: no cover
    NUMBA_VERSION = ""

    def get_num_threads():
        return 1

H2 = 0.5
PREV = 0.05
FAM_VEC = ["m", "f", "s1"]
N_FAM = 200
REPS = 3
TOL = 0.01
N_SIM = 100_000
BURN_IN = 1000
SEED = 20260815


def _check_r_pkg(name):
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None, "Rscript not on PATH"
    probe = (
        f"if (!requireNamespace('{name}', quietly=TRUE)) "
        f"quit(status=2); "
        f"cat(as.character(packageVersion('{name}')))"
    )
    proc = __import__("subprocess").run(
        [rscript, "-e", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or f"{name} not installed").strip()
        return None, err.splitlines()[-1] if err else f"{name} not installed"
    return proc.stdout.strip(), None


def families_to_tbl(families, path):
    """Write the shared .tbl (fam_ID, indiv_ID, role, lower, upper)."""
    rows = []
    for fam in families:
        fid = str(fam.fam_id)
        for member in fam.members:
            if member.role == "g":
                continue
            pid = (str(member.pid) if member.pid is not None
                   else f"{fid}_{member.role}")
            rows.append(dict(
                fam_ID=fid, indiv_ID=pid, role=member.role,
                lower=_fmt_bound(member.lower),
                upper=_fmt_bound(member.upper),
            ))
    write_rows(path, rows, fields=("fam_ID", "indiv_ID", "role", "lower",
                                   "upper"))


def _fmt_bound(value):
    value = float(value)
    if np.isneginf(value):
        return "-Inf"
    if np.isposinf(value):
        return "Inf"
    return repr(value)


def _parse_bound(text):
    text = str(text).strip()
    if text in ("-Inf", "-inf"):
        return -np.inf
    if text in ("Inf", "inf"):
        return np.inf
    return float(text)


def families_from_tbl(path):
    from ltpred.family import families_from_columns
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return families_from_columns(
        [r["fam_ID"] for r in rows],
        [r["role"] for r in rows],
        [_parse_bound(r["lower"]) for r in rows],
        [_parse_bound(r["upper"]) for r in rows],
        pid=[r["indiv_ID"] for r in rows],
    )


def read_scores(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"empty score file: {path}")
    est_col = "genetic_est" if "genetic_est" in rows[0] else "est"
    by_fam = {r["fam_ID"]: float(r[est_col]) for r in rows}
    seconds = float(rows[0]["seconds"])
    meta = dict(rows[0])
    return by_fam, seconds, meta


def align(families, by_fam):
    return np.array([by_fam[str(fam.fam_id)] for fam in families], dtype=float)


def metrics(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    max_abs = float(np.max(np.abs(a - b)))
    return corr, rmse, max_abs


def bytes_to_mib(n):
    if n is None or n == "":
        return float("nan")
    return float(n) / (1024.0 * 1024.0)


def run_r_helper(script, tbl_path, out_path, h2, tol, workers, extra=()):
    rscript = shutil.which("Rscript")
    cmd = [rscript, script, tbl_path, out_path,
           str(h2), str(tol), str(workers), *extra]
    proc, peak = run_peak(cmd, cwd=ROOT, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(script)} failed (rc={proc.returncode}):\n"
            f"{proc.stdout}")
    return proc.stdout, peak


def run_ltpred_worker(method, tbl_path, out_path, h2, tol, n_sim, burn_in, seed):
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", method,
        "--tbl", tbl_path,
        "--out", out_path,
        "--h2", str(h2),
        "--tol", str(tol),
        "--n-sim", str(n_sim),
        "--burn-in", str(burn_in),
        "--seed", str(seed),
    ]
    proc, peak = run_peak(cmd, cwd=HERE, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"ltpred {method} worker failed (rc={proc.returncode}):\n"
            f"{proc.stdout}")
    return proc.stdout, peak


def worker_main(args):
    """Isolated estimator: warmup, then time the estimate on --tbl."""
    families = families_from_tbl(args.tbl)
    warm = families[: min(8, len(families))]
    if args.worker == "pa":
        estimate(warm, args.h2, "pa", tol=args.tol)
        t0 = __import__("time").perf_counter()
        est, _ = estimate(families, args.h2, "pa", tol=args.tol)
        seconds = __import__("time").perf_counter() - t0
    else:
        estimate(warm, args.h2, "gibbs", n_sim=args.n_sim,
                 burn_in=args.burn_in, tol=args.tol, seed=args.seed)
        t0 = __import__("time").perf_counter()
        est, _ = estimate(
            families, args.h2, "gibbs", n_sim=args.n_sim,
            burn_in=args.burn_in, tol=args.tol, seed=args.seed + 1)
        seconds = __import__("time").perf_counter() - t0

    # fam_id order follows first appearance in the tbl, matching Family list.
    fam_ids = []
    seen = set()
    for fam in families:
        key = str(fam.fam_id)
        if key not in seen:
            seen.add(key)
            fam_ids.append(key)
    write_rows(args.out, [dict(
        fam_ID=fid, genetic_est=float(value),
        seconds=seconds, method=args.worker)
        for fid, value in zip(fam_ids, est)],
        fields=("fam_ID", "genetic_est", "seconds", "method"))
    print(f"ltpred {args.worker}  families={len(fam_ids)}  seconds={seconds:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fam", type=int, default=N_FAM)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--workers", type=int, default=1,
                        help="future workers for the R packages (1 = sequential)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-prefix",
                        default=os.path.join(HERE, "bench_ltfhplus_compare"))
    parser.add_argument("--worker", choices=("pa", "gibbs"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--tbl", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--h2", type=float, default=H2, help=argparse.SUPPRESS)
    parser.add_argument("--tol", type=float, default=TOL, help=argparse.SUPPRESS)
    parser.add_argument("--n-sim", type=int, default=N_SIM, help=argparse.SUPPRESS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.tbl or not args.out:
            print("--worker requires --tbl and --out", file=sys.stderr)
            return 2
        return worker_main(args)

    ltfh_ver, err = _check_r_pkg("LTFHPlus")
    if err:
        print(f"SKIP: LTFHPlus comparison unavailable ({err})")
        print("Install with: Rscript -e "
              "\"install.packages('LTFHPlus')\"")
        return 2
    ltfgrs_ver, ltfgrs_err = _check_r_pkg("LTFGRS")
    if ltfgrs_err:
        print(f"NOTE: LTFGRS PA arm skipped ({ltfgrs_err})")
        print("Public PA is LTFGRS, not LTFHPlus. Install with: Rscript -e "
              "\"install.packages('LTFGRS')\"")

    print("Locked LTFHPlus / LTFGRS comparison")
    print(f"LTFHPlus {ltfh_ver}  LTFGRS {ltfgrs_ver or 'missing'}  "
          f"Rscript={shutil.which('Rscript')}")
    print(f"n_fam={args.n_fam}  reps={args.reps}  fam={'+'.join(FAM_VEC)}  "
          f"h2={H2}  K={PREV}  tol={TOL}  n_sim={N_SIM}  burn_in={BURN_IN}")
    print(f"R future workers={args.workers}  "
          f"Numba threads={get_num_threads()}")
    print("Estimand: posterior-mean genetic liability on classic LT-FH bounds.")
    print("LTFHPlus is Gibbs-only. Public PA is LTFGRS method=PA.")
    print("Peak RSS: isolated process via _peak_launcher (wait4, not parent floor).\n")

    rows = []
    with tempfile.TemporaryDirectory(prefix="ltfhplus_") as tmp:
        for rep in range(args.reps):
            sim = simulate_families(
                FAM_VEC, H2, PREV, args.n_fam, seed=args.seed + 17 * rep)
            true_g = np.asarray(sim.genetic, dtype=float)
            families = sim.families
            tbl = os.path.join(tmp, f"tbl_{rep}.csv")
            families_to_tbl(families, tbl)

            r_out = os.path.join(tmp, f"r_ltfh_{rep}.csv")
            r_seed = args.seed + 3000 + rep
            helper_log, peak_ltfh = run_r_helper(
                R_LTFHPLUS, tbl, r_out, H2, TOL, args.workers,
                extra=(str(r_seed),))
            print(helper_log.rstrip())
            r_by_fam, r_s, r_meta = read_scores(r_out)
            r_est = align(families, r_by_fam)
            r_ver = r_meta.get("ltfhplus_version", ltfh_ver)
            r_workers = int(float(r_meta.get("workers", args.workers)))
            if "seed" not in r_meta:
                raise RuntimeError("LTFHPlus output did not record its RNG seed")
            recorded_r_seed = int(float(r_meta["seed"]))
            if recorded_r_seed != r_seed:
                raise RuntimeError(
                    f"LTFHPlus recorded seed {recorded_r_seed}, expected {r_seed}")

            peak_ltfgrs = ""
            ltfgrs_s = float("nan")
            ltfgrs_est = None
            ltfgrs_ver = ltfgrs_ver
            if ltfgrs_ver:
                g_out = os.path.join(tmp, f"r_ltfgrs_{rep}.csv")
                g_log, peak_ltfgrs = run_r_helper(
                    R_LTFGRS, tbl, g_out, H2, TOL, args.workers, extra=("PA",))
                print(g_log.rstrip())
                g_by_fam, ltfgrs_s, g_meta = read_scores(g_out)
                ltfgrs_est = align(families, g_by_fam)
                ltfgrs_ver = g_meta.get("ltfgrs_version", ltfgrs_ver)

            pa_out = os.path.join(tmp, f"py_pa_{rep}.csv")
            pa_log, peak_pa = run_ltpred_worker(
                "pa", tbl, pa_out, H2, TOL, N_SIM, BURN_IN,
                args.seed + 1000 + rep)
            print(pa_log.rstrip())
            pa_by_fam, pa_s, _ = read_scores(pa_out)
            pa = align(families, pa_by_fam)

            gibbs_out = os.path.join(tmp, f"py_gibbs_{rep}.csv")
            gb_log, peak_gibbs = run_ltpred_worker(
                "gibbs", tbl, gibbs_out, H2, TOL, N_SIM, BURN_IN,
                args.seed + 2000 + rep)
            print(gb_log.rstrip())
            gb_by_fam, gibbs_s, _ = read_scores(gibbs_out)
            gibbs = align(families, gb_by_fam)

            c_pg, rmse_pg, mx_pg = metrics(pa, gibbs)
            c_pr, rmse_pr, mx_pr = metrics(pa, r_est)
            c_gr, rmse_gr, mx_gr = metrics(gibbs, r_est)
            c_gt, _, _ = metrics(gibbs, true_g)
            c_rt, _, _ = metrics(r_est, true_g)
            c_pt, _, _ = metrics(pa, true_g)

            if ltfgrs_est is not None:
                c_pL, rmse_pL, mx_pL = metrics(pa, ltfgrs_est)
                c_gL, rmse_gL, mx_gL = metrics(gibbs, ltfgrs_est)
                c_RL, rmse_RL, mx_RL = metrics(r_est, ltfgrs_est)
                c_Lt, _, _ = metrics(ltfgrs_est, true_g)
            else:
                c_pL = rmse_pL = mx_pL = float("nan")
                c_gL = rmse_gL = mx_gL = float("nan")
                c_RL = rmse_RL = mx_RL = float("nan")
                c_Lt = float("nan")

            row = dict(
                rep=rep, n_fam=args.n_fam, h2=H2, prevalence=PREV,
                tol=TOL, n_sim=N_SIM, burn_in=BURN_IN,
                ltfhplus_version=r_ver, ltfhplus_workers=r_workers,
                ltfhplus_seed=recorded_r_seed,
                r_version=r_meta.get("r_version", ""),
                r_rng_kind=r_meta.get("rng_kind", ""),
                ltfgrs_version=ltfgrs_ver or "",
                python_version=platform.python_version(),
                numpy_version=np.__version__, numba_version=NUMBA_VERSION,
                numba_threads=get_num_threads(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", ""),
                openblas_num_threads=os.environ.get(
                    "OPENBLAS_NUM_THREADS", ""),
                seconds_ltfhplus=r_s, seconds_ltfgrs_pa=ltfgrs_s,
                seconds_ltpred_gibbs=gibbs_s, seconds_ltpred_pa=pa_s,
                ms_per_fam_ltfhplus=1000.0 * r_s / args.n_fam,
                ms_per_fam_ltfgrs_pa=1000.0 * ltfgrs_s / args.n_fam,
                ms_per_fam_ltpred_gibbs=1000.0 * gibbs_s / args.n_fam,
                ms_per_fam_ltpred_pa=1000.0 * pa_s / args.n_fam,
                fold_ltfhplus_over_gibbs=r_s / gibbs_s,
                fold_ltfhplus_over_pa=r_s / pa_s,
                fold_ltfgrs_over_pa=ltfgrs_s / pa_s,
                fold_ltfgrs_over_gibbs=ltfgrs_s / gibbs_s,
                peak_rss_bytes_ltfhplus=peak_ltfh,
                peak_rss_bytes_ltfgrs_pa=peak_ltfgrs,
                peak_rss_bytes_ltpred_gibbs=peak_gibbs,
                peak_rss_bytes_ltpred_pa=peak_pa,
                corr_pa_gibbs=c_pg, rmse_pa_gibbs=rmse_pg, maxabs_pa_gibbs=mx_pg,
                corr_pa_ltfhplus=c_pr, rmse_pa_ltfhplus=rmse_pr,
                maxabs_pa_ltfhplus=mx_pr,
                corr_gibbs_ltfhplus=c_gr, rmse_gibbs_ltfhplus=rmse_gr,
                maxabs_gibbs_ltfhplus=mx_gr,
                corr_pa_ltfgrs=c_pL, rmse_pa_ltfgrs=rmse_pL,
                maxabs_pa_ltfgrs=mx_pL,
                corr_gibbs_ltfgrs=c_gL, rmse_gibbs_ltfgrs=rmse_gL,
                maxabs_gibbs_ltfgrs=mx_gL,
                corr_ltfhplus_ltfgrs=c_RL, rmse_ltfhplus_ltfgrs=rmse_RL,
                maxabs_ltfhplus_ltfgrs=mx_RL,
                corr_gibbs_truth=c_gt, corr_ltfhplus_truth=c_rt,
                corr_pa_truth=c_pt, corr_ltfgrs_truth=c_Lt,
            )
            rows.append(row)
            ltfgrs_bit = (
                f"  corr(PA,LTFGRS)={c_pL:.5f}  t LTFGRS={ltfgrs_s:.3f}s"
                if ltfgrs_est is not None else "")
            print(
                f"rep {rep}: corr(Gibbs,LTFHPlus)={c_gr:.5f}  "
                f"corr(PA,LTFHPlus)={c_pr:.5f}{ltfgrs_bit}  "
                f"RMSE(Gibbs,R)={rmse_gr:.4f}  "
                f"t R={r_s:.2f}s  t Gibbs={gibbs_s:.2f}s  t PA={pa_s:.3f}s  "
                f"RSS MiB LTFHPlus={bytes_to_mib(peak_ltfh):.1f}  "
                f"LTFGRS={bytes_to_mib(peak_ltfgrs):.1f}  "
                f"Gibbs={bytes_to_mib(peak_gibbs):.1f}  "
                f"PA={bytes_to_mib(peak_pa):.1f}"
            )

    out_csv = f"{args.output_prefix}.csv"
    write_rows(out_csv, rows, fields=list(rows[0]))

    def mean_se(key):
        vals = np.array([r[key] for r in rows], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
        se = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
        return float(vals.mean()), float(se)

    print("\nAcross-replicate mean ± SE")
    for key, label in (
        ("corr_gibbs_ltfhplus", "corr(ltpred Gibbs, LTFHPlus)"),
        ("corr_pa_ltfhplus", "corr(ltpred PA, LTFHPlus)"),
        ("corr_pa_ltfgrs", "corr(ltpred PA, LTFGRS PA)"),
        ("rmse_gibbs_ltfhplus", "RMSE(ltpred Gibbs, LTFHPlus)"),
        ("rmse_pa_ltfgrs", "RMSE(ltpred PA, LTFGRS PA)"),
        ("seconds_ltfhplus", "LTFHPlus seconds (total)"),
        ("ms_per_fam_ltfhplus", "LTFHPlus ms / family"),
        ("seconds_ltfgrs_pa", "LTFGRS PA seconds (total)"),
        ("ms_per_fam_ltfgrs_pa", "LTFGRS PA ms / family"),
        ("seconds_ltpred_gibbs", "ltpred Gibbs seconds (total)"),
        ("ms_per_fam_ltpred_gibbs", "ltpred Gibbs ms / family"),
        ("seconds_ltpred_pa", "ltpred PA seconds (total)"),
        ("ms_per_fam_ltpred_pa", "ltpred PA ms / family"),
        ("peak_rss_bytes_ltfhplus", "LTFHPlus peak RSS bytes"),
        ("peak_rss_bytes_ltfgrs_pa", "LTFGRS PA peak RSS bytes"),
        ("peak_rss_bytes_ltpred_gibbs", "ltpred Gibbs peak RSS bytes"),
        ("peak_rss_bytes_ltpred_pa", "ltpred PA peak RSS bytes"),
        ("fold_ltfhplus_over_gibbs", "fold LTFHPlus / ltpred Gibbs"),
        ("fold_ltfgrs_over_pa", "fold LTFGRS PA / ltpred PA"),
        ("fold_ltfhplus_over_pa", "fold LTFHPlus / ltpred PA"),
        ("fold_ltfgrs_over_gibbs", "fold LTFGRS PA / ltpred Gibbs"),
    ):
        mean, se = mean_se(key)
        if "rss_bytes" in key:
            print(f"  {label:36s} {bytes_to_mib(mean):.2f} ± "
                  f"{bytes_to_mib(se):.2f} MiB")
        else:
            print(f"  {label:36s} {mean:.4f} ± {se:.4f}")

    r_t, _ = mean_se("seconds_ltfhplus")
    g_t, _ = mean_se("seconds_ltpred_gibbs")
    p_t, _ = mean_se("seconds_ltpred_pa")
    L_t, _ = mean_se("seconds_ltfgrs_pa")
    fg, fgs = mean_se("fold_ltfhplus_over_gibbs")
    fp, fps = mean_se("fold_ltfgrs_over_pa")
    fmix, fmixs = mean_se("fold_ltfhplus_over_pa")
    print(f"\nFold times on this machine "
          f"(mean of per-replicate ratios; "
          f"R workers={args.workers}, "
          f"Numba threads={get_num_threads()}):")
    print(f"  LTFHPlus / ltpred Gibbs = {fg:.2f} ± {fgs:.2f}  (same algorithm)")
    if np.isfinite(fp):
        print(f"  LTFGRS PA / ltpred PA   = {fp:.1f} ± {fps:.1f}  (same algorithm)")
    print(f"  LTFHPlus / ltpred PA    = {fmix:.0f} ± {fmixs:.0f}  (mixes algorithms)")
    print("These are not hardware-independent constants.")
    print("Peak RSS is the isolated child via wait4; it includes the")
    print("interpreter and packages, not just the estimator scratch.")
    print(f"\nwrote {os.path.basename(out_csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
