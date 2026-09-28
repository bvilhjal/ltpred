"""Locked comparison of ltpred's personalised LT-FH++ against LTFHPlus/LTFGRS.

The classic lock (``bench_ltfhplus_compare.py``, RESULTS section 30) covers
one-threshold LT-FH bounds. This script locks the two features that define
LT-FH++ beyond it, on families whose bounds are built from an age-dependent
cumulative-incidence curve (every member has their own age; cases are pinned
at their onset-age threshold, controls are censored with a finite
age-specific upper bound):

  * **personalised panel** -- ltpred Gibbs vs ``LTFHPlus::estimate_liability``
    (Rcpp Gibbs) on identical per-member interval/pin bounds, with ltpred PA
    as the deterministic reference. Locks the posterior under heterogeneous
    intervals, the input regime the classic lock never exercises.
  * **mixture panel** -- ltpred PA with the censored-control mixture
    (``use_mixture=True``, Algorithm M) vs ``LTFGRS::estimate_liability``
    (``method="PA", useMixture=TRUE``), with both packages' no-mixture PA as
    anchors.

Input-contract note (found building this lock): the two packages encode a
censored control differently. ltpred's mixture takes the age-specific bound
``Phi^-1(1 - CIP(age))`` as ``upper`` and ignores its exact value -- the
mixture splits at the *lifetime* threshold ``Phi^-1(1 - K_pop)`` (Dybdahl
Krebs et al. 2024, eqs. S3-S5), so an age-specific or a lifetime ``upper``
gives the same score. LTFGRS *consumes* the passed ``upper``; feeding it the
age-specific bound double-corrects the censoring (measured: corr 0.71,
RMSE 0.29). The mixture panel therefore writes each package its own table:
ltpred gets the age-specific intervals with NaN ``K`` off censored rows,
LTFGRS gets lifetime uppers there and ``K`` filled on every row (its
validator rejects NA). The no-mixture anchors share the age-specific table.

Wall-clock is the estimator call after in-process warmup (as in the classic
lock); peak RSS uses the same isolated-process ``_peak_launcher`` pattern.
Requires R with LTFHPlus and LTFGRS (exits 2 without them).::

    NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python benchmarks/run_benchmark.py \\
        --artifact bench_personalised_compare.csv \\
        bench_personalised_compare.py
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.stats import norm

from _common import estimate, simulate_families, write_rows
from bench_ltfhplus_compare import (R_LTFGRS, R_LTFHPLUS, NUMBA_VERSION,
                                    _check_r_pkg, align, bytes_to_mib,
                                    families_from_tbl, metrics, read_scores,
                                    run_peak, run_r_helper)
from ltpred.thresholds import thresholds_from_cip

try:
    from numba import get_num_threads
except ImportError:  # pragma: no cover
    def get_num_threads():
        return 1

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

H2 = 0.5
PREV = 0.10                     # richer case mix than the classic lock's 0.05
FAM_VEC = ["m", "f", "s1"]
N_FAM = 200
REPS = 3
TOL = 0.01
N_SIM = 100_000
BURN_IN = 1000
SEED = 20260928
CIP_AGES = np.arange(0.0, 111.0)
CIP_VALUES = PREV / (1.0 + np.exp((60.0 - CIP_AGES) / 8.0))
LIFETIME = float(norm.isf(PREV))    # Phi^-1(1 - K_pop): the mixture split
FIELDS = ("fam_ID", "indiv_ID", "role", "lower", "upper", "K_i", "K_pop")


def _fmt(value):
    value = float(value)
    if np.isneginf(value):
        return "-Inf"
    if np.isposinf(value):
        return "Inf"
    return repr(value)


def personalised_rows(sim):
    """One row per member with CIP-derived bounds and ltpred's K convention.

    A case is pinned at ``Phi^-1(1 - CIP(onset_age))``; a control keeps the
    interval ``(-inf, Phi^-1(1 - CIP(age)))`` and carries its ``K_i``/``K_pop``
    mixture pair. Rows follow family-first-appearance order, matching
    ``families_from_tbl``.
    """
    rows = []
    for i, fam in enumerate(sim.families):
        fid = str(fam.fam_id)
        for member in fam.members:
            role = member.role
            case = bool(sim.status[role][i])
            age = float(sim.onset[role][i] if case else sim.ages[role][i])
            lo, hi, ki, kp = thresholds_from_cip(
                np.array([case]), np.array([age]), CIP_AGES, CIP_VALUES,
                k_pop=PREV)
            rows.append(dict(fam_ID=fid, indiv_ID=f"{fid}_{role}", role=role,
                             lower=_fmt(lo[0]), upper=_fmt(hi[0]),
                             K_i=_fmt(ki[0]), K_pop=_fmt(kp[0])))
    return rows


def write_native_tbl(rows, path, with_k=True):
    """ltpred's encoding: age-specific uppers, NaN K off censored controls.

    ``with_k=False`` drops the K columns for the R packages' no-mixture arms:
    LTFGRS validates ``K_i``/``K_pop`` whenever the columns are present, even
    with ``useMixture=FALSE``, and its NA check rejects ltpred's NaN
    convention."""
    write_rows(path, rows, fields=FIELDS if with_k else FIELDS[:5])


def write_ltfgrs_tbl(rows, path):
    """LTFGRS's mixture encoding: lifetime uppers on censored controls, K
    filled on every row (its validator rejects NA in the ``K_i > K_pop``
    check).

    ``K_i = K_pop`` on non-mixture rows asserts "no future case remains",
    which is inert wherever the row is a pin or a one-sided case: only the
    censored controls engage the mixture in either package."""
    filled = []
    for r in rows:
        r = dict(r)
        if r["lower"] != r["upper"] and r["upper"] != "Inf":
            r["upper"] = repr(LIFETIME)
        if r["K_i"] == "nan":
            r["K_i"] = repr(PREV)
        if r["K_pop"] == "nan":
            r["K_pop"] = repr(PREV)
        filled.append(r)
    write_rows(path, filled, fields=FIELDS)


def run_ltpred_worker(tbl_path, out_path, h2, tol, n_sim, burn_in, seed,
                      method):
    mixture = method == "mixture"
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "pa" if mixture else method,
           "--tbl", tbl_path, "--out", out_path,
           "--h2", str(h2), "--tol", str(tol),
           "--n-sim", str(n_sim), "--burn-in", str(burn_in),
           "--seed", str(seed)]
    if mixture:
        cmd.append("--mixture")
    proc, peak = run_peak(cmd, cwd=HERE, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"ltpred worker failed (rc={proc.returncode}):\n"
                           f"{proc.stdout}")
    return proc.stdout, peak


def families_from_tbl_k(path):
    """``families_from_tbl`` keeping the ``K_i``/``K_pop`` mixture columns."""
    import csv
    from ltpred.family import families_from_columns
    from bench_ltfhplus_compare import _parse_bound
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return families_from_columns(
        [r["fam_ID"] for r in rows],
        [r["role"] for r in rows],
        [_parse_bound(r["lower"]) for r in rows],
        [_parse_bound(r["upper"]) for r in rows],
        pid=[r["indiv_ID"] for r in rows],
        K_i=[_parse_bound(r["K_i"]) for r in rows],
        K_pop=[_parse_bound(r["K_pop"]) for r in rows])


def worker_main(args):
    """Isolated estimator process; mirrors bench_ltfhplus_compare's worker."""
    families = families_from_tbl_k(args.tbl)
    warm = families[: min(8, len(families))]
    if args.mixture:
        estimate(warm, args.h2, "pa", tol=args.tol, use_mixture=True)
        t0 = time.perf_counter()
        est, _ = estimate(families, args.h2, "pa", tol=args.tol,
                          use_mixture=True)
        label = "pa_mixture"
    elif args.worker == "pa":        # pragma: no cover - anchor arm
        estimate(warm, args.h2, "pa", tol=args.tol)
        t0 = time.perf_counter()
        est, _ = estimate(families, args.h2, "pa", tol=args.tol)
        label = "pa"
    else:
        estimate(warm, args.h2, "gibbs", n_sim=args.n_sim,
                 burn_in=args.burn_in, tol=args.tol, seed=args.seed)
        t0 = time.perf_counter()
        est, _ = estimate(families, args.h2, "gibbs", n_sim=args.n_sim,
                          burn_in=args.burn_in, tol=args.tol,
                          seed=args.seed + 1)
        label = "gibbs"
    seconds = time.perf_counter() - t0
    fam_ids, seen = [], set()
    for fam in families:
        key = str(fam.fam_id)
        if key not in seen:
            seen.add(key)
            fam_ids.append(key)
    write_rows(args.out, [dict(fam_ID=fid, genetic_est=float(value),
                               seconds=seconds, method=label)
                          for fid, value in zip(fam_ids, est)],
               fields=("fam_ID", "genetic_est", "seconds", "method"))
    print(f"ltpred {label}  families={len(fam_ids)}  seconds={seconds:.4f}")
    return 0


def _timed_row(rep, panel, estimator, seconds, n_fam, est, truth, peak):
    return dict(rep=rep, panel=panel, estimator=estimator, seconds=seconds,
                ms_per_family=1000.0 * seconds / n_fam,
                corr_with_truth=float(np.corrcoef(est, truth)[0, 1]),
                peak_rss_bytes=peak)


def _lock_row(rep, panel, estimator, a, b):
    corr, rmse, max_abs = metrics(a, b)
    return dict(rep=rep, panel=panel, estimator=estimator,
                corr_with_truth=corr, rmse=rmse, max_abs=max_abs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fam", type=int, default=N_FAM)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-prefix",
                        default=os.path.join(HERE, "bench_personalised_compare"))
    parser.add_argument("--worker", choices=("pa", "gibbs"),
                        default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mixture", action="store_true", help=argparse.SUPPRESS)
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
        return 2
    ltfgrs_ver, ltfgrs_err = _check_r_pkg("LTFGRS")
    if ltfgrs_err:
        print(f"SKIP: the mixture panel needs LTFGRS ({ltfgrs_err})")
        return 2

    print("Locked personalised LT-FH++ / censoring-mixture comparison")
    print(f"LTFHPlus {ltfh_ver}  LTFGRS {ltfgrs_ver}  "
          f"Rscript={shutil.which('Rscript')}")
    print(f"n_fam={args.n_fam}  reps={args.reps}  fam={'+'.join(FAM_VEC)}  "
          f"h2={H2}  K={PREV}  tol={TOL}  n_sim={N_SIM}  burn_in={BURN_IN}")
    print(f"R future workers={args.workers}  Numba threads={get_num_threads()}")
    print("Estimand: posterior-mean genetic liability under age-CIP bounds")
    print("(cases pinned at onset threshold, controls censored with a finite")
    print("age-specific upper). ltpred mixture input: age-specific uppers, NaN")
    print("K off censored rows; LTFGRS mixture input: lifetime uppers, K")
    print("filled (it consumes the passed upper).")
    print("Peak RSS: isolated process via _peak_launcher.\n")

    rows_out = []
    with tempfile.TemporaryDirectory(prefix="personalised_") as tmp:
        for rep in range(args.reps):
            sim = simulate_families(FAM_VEC, H2, PREV, args.n_fam,
                                    seed=args.seed + 17 * rep, use_age=True)
            true_g = np.asarray(sim.genetic, dtype=float)
            families = sim.families
            rows = personalised_rows(sim)
            n_pin = sum(1 for r in rows if r["lower"] == r["upper"])
            n_cens = sum(1 for r in rows
                         if r["upper"] != "Inf" and r["lower"] != r["upper"])
            native = os.path.join(tmp, f"native_{rep}.csv")
            native_r = os.path.join(tmp, f"native_r_{rep}.csv")
            ltfgrs_tbl = os.path.join(tmp, f"ltfgrs_{rep}.csv")
            write_native_tbl(rows, native)
            write_native_tbl(rows, native_r, with_k=False)
            write_ltfgrs_tbl(rows, ltfgrs_tbl)

            out = {}
            r_seed = args.seed + 3000 + rep
            r_out = os.path.join(tmp, f"r_ltfh_{rep}.csv")
            log, peak = run_r_helper(R_LTFHPLUS, native_r, r_out, H2, TOL,
                                     args.workers, extra=(str(r_seed),))
            print(log.rstrip())
            by_fam, seconds, meta = read_scores(r_out)
            if int(float(meta["seed"])) != r_seed:
                raise RuntimeError("LTFHPlus output did not record its seed")
            out["ltfhplus_gibbs"] = align(families, by_fam)
            rows_out.append(_timed_row(rep, "personalised", "ltfhplus_gibbs",
                                       seconds, args.n_fam,
                                       out["ltfhplus_gibbs"], true_g, peak))

            for name, tbl, panel in (
                    ("ltfgrs_pa_mixture", ltfgrs_tbl, "mixture"),
                    ("ltfgrs_pa_nomixture", native_r, "mixture")):
                r_out = os.path.join(tmp, f"r_{name}_{rep}.csv")
                log, peak = run_r_helper(R_LTFGRS, tbl, r_out, H2, TOL,
                                         args.workers,
                                         extra=("PA", "FALSE" if "nomix" in name
                                                else "TRUE"))
                print(log.rstrip())
                by_fam, seconds, _ = read_scores(r_out)
                out[name] = align(families, by_fam)
                rows_out.append(_timed_row(rep, panel, name, seconds,
                                           args.n_fam, out[name], true_g, peak))

            for label, method in (("ltpred_gibbs", "gibbs"),
                                  ("ltpred_pa", "pa"),
                                  ("ltpred_pa_mixture", "mixture")):
                w_out = os.path.join(tmp, f"py_{label}_{rep}.csv")
                log, peak = run_ltpred_worker(
                    native, w_out, H2, TOL, N_SIM, BURN_IN,
                    args.seed + 1000 + 10 * rep + len(label), method)
                print(log.rstrip())
                by_fam, seconds, _ = read_scores(w_out)
                out[label] = align(families, by_fam)
                rows_out.append(_timed_row(
                    rep, "personalised" if label == "ltpred_gibbs"
                    else "mixture", label, seconds, args.n_fam,
                    out[label], true_g, peak))

            rows_out.append(_lock_row(rep, "personalised",
                                      "lock_gibbs_ltfhplus",
                                      out["ltpred_gibbs"], out["ltfhplus_gibbs"]))
            rows_out.append(_lock_row(rep, "mixture",
                                      "lock_pa_mixture_ltfgrs",
                                      out["ltpred_pa_mixture"],
                                      out["ltfgrs_pa_mixture"]))
            rows_out.append(_lock_row(rep, "mixture",
                                      "lock_pa_ltfgrs_nomixture",
                                      out["ltpred_pa"], out["ltfgrs_pa_nomix"] if
                                      "ltfgrs_pa_nomix" in out
                                      else out["ltfgrs_pa_nomixture"]))
            print(
                f"rep {rep}: {n_pin} pinned cases, {n_cens} censored controls; "
                f"corr(Gibbs,LTFHPlus)="
                f"{rows_out[-3]['corr_with_truth']:.5f} "
                f"RMSE={rows_out[-3]['rmse']:.4f}  "
                f"corr(PA-mix,LTFGRS-mix)="
                f"{rows_out[-2]['corr_with_truth']:.5f} "
                f"RMSE={rows_out[-2]['rmse']:.5f}  "
                f"corr(PA,LTFGRS-PA)="
                f"{rows_out[-1]['corr_with_truth']:.5f} "
                f"RMSE={rows_out[-1]['rmse']:.5f}"
            )

        rows_out.append(dict(
            rep=args.reps, panel="meta", estimator="environment",
            ltfhplus_version=ltfh_ver, ltfgrs_version=ltfgrs_ver,
            python_version=platform.python_version(),
            numpy_version=np.__version__, numba_version=NUMBA_VERSION,
            numba_threads=get_num_threads(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS", ""),
            openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS", ""),
            r_workers=args.workers, h2=H2, prevalence=PREV, tol=TOL,
            n_sim=N_SIM, burn_in=BURN_IN))

    fields = []
    for r in rows_out:
        for k in r:
            if k not in fields:
                fields.append(k)
    for r in rows_out:
        for k in fields:
            r.setdefault(k, "")
    out_csv = f"{args.output_prefix}.csv"
    write_rows(out_csv, rows_out, fields=fields)

    def stats(estimator, key):
        vals = np.array([r[key] for r in rows_out
                         if r["estimator"] == estimator
                         and isinstance(r[key], float)
                         and np.isfinite(r[key])], float)
        if not vals.size:
            return float("nan"), float("nan")
        se = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
        return float(vals.mean()), float(se)

    print("\nAcross-replicate mean ± SE")
    for estimator, label in (
            ("lock_gibbs_ltfhplus", "corr(ltpred Gibbs, LTFHPlus)"),
            ("lock_gibbs_ltfhplus", "RMSE(ltpred Gibbs, LTFHPlus)"),
            ("lock_pa_mixture_ltfgrs", "corr(ltpred PA-mix, LTFGRS-mix)"),
            ("lock_pa_mixture_ltfgrs", "RMSE(ltpred PA-mix, LTFGRS-mix)"),
            ("lock_pa_ltfgrs_nomixture", "corr(ltpred PA, LTFGRS PA)"),
            ("lock_pa_ltfgrs_nomixture", "RMSE(ltpred PA, LTFGRS PA)")):
        mean, se = stats(estimator,
                         "corr_with_truth" if label.startswith("corr")
                         else "rmse")
        print(f"  {label:36s} {mean:.5f} ± {se:.5f}")

    print("\nEstimator wall-time (ms/family) and isolated peak RSS (MiB):")
    for name in ("ltfhplus_gibbs", "ltpred_gibbs", "ltpred_pa",
                 "ltfgrs_pa_mixture", "ltpred_pa_mixture"):
        mean, _ = stats(name, "ms_per_family")
        rss = np.array([r["peak_rss_bytes"] for r in rows_out
                        if r["estimator"] == name
                        and isinstance(r["peak_rss_bytes"], (int, float))],
                       float)
        rss_m = float(rss.mean()) if rss.size else float("nan")
        print(f"  {name:20s} {mean:9.3f} ms/fam   RSS "
              f"{bytes_to_mib(rss_m):8.1f} MiB")
    print("\nThe mixture lock encodes each package's own input contract")
    print("(age-specific uppers for ltpred, lifetime uppers for LTFGRS);")
    print("these are not hardware-independent constants.")
    print(f"\nwrote {os.path.basename(out_csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
