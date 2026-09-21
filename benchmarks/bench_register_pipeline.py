"""End-to-end register-pipeline benchmark.

`ltpred.pipeline.estimate_liabilities` chains trio records -> pedigree
discovery -> CIP thresholds -> per-proband genetic-liability scores. This
benchmark builds synthetic registers (3-generation populations, remarriages;
a known logistic CIP; liabilities from the extracted pedigrees; onsets by
threshold crossing) and measures the pipeline end to end, over `--reps`
independent registers so the paired contrasts carry across-replicate SEs and
t-based 95% CIs (section 15 convention):

  Part 1 (accuracy):    corr(est, true g) and calibration slope at degree 3
         (first cousins included) vs degree 1 (first-degree only) -- the
         "which relatives you include matters" effect, through the driver.
         Replicated; the degree-3-minus-degree-1 contrast is paired.
  Part 2 (CIP from data): the same pipeline with the CIP ESTIMATED from the
         register itself (aalen_johansen_cip on the follow-up records) vs the
         oracle curve. Replicated; the estimated-minus-oracle contrast is
         paired.
  Part 3 (prospective): prospective prediction among probands unaffected at
         age 40, with every person assigned a birth time on one calendar
         scale -- (a) the correct familywise-censored score at the proband's
         own age-40 index time, (b) a leaky arm that conditions on relatives'
         full post-index records but still leaves the proband uninformative,
         and (c) a cumulative-leak arm using full family records including the
         proband's own outcome. Per arm and replicate: corr(score, future),
         the calibration slope of future on score, AUC (rank/Mann-Whitney,
         implemented directly -- no sklearn), and calibration-in-the-large
         (mean at-risk predicted future-case probability vs observed rate).
         The leakage contrasts (b)-(a) and (c)-(a) are paired.
  Part 4 (throughput):  probands/s for the full pipeline. Single run.

Writes `benchmarks/bench_register_pipeline.csv` in long format
(rep, metric, value; rep 0 marks the single-run part).

Run:  conda run -n ltpred python benchmarks/bench_register_pipeline.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import mean_se  # noqa: E402 — also puts the repo root on sys.path
from bench_pedigree_inference import paired_summary, simulate_population  # noqa: E402
from ltpred.cip import aalen_johansen_cip  # noqa: E402
from ltpred.pipeline import estimate_liabilities  # noqa: E402
from ltpred.simulate import (pedigree_birth_times as _pedigree_birth_times,  # noqa: E402
                             simulate_register_liabilities)
from scipy.stats import norm  # noqa: E402

SEED = 20260719
H2 = 0.5
K_POP = 0.10
MID = 60.0
SLOPE = 1.0 / 8.0
N_PROBANDS = 400
INDEX_AGE = 40.0
EVAL_AGE = 70.0
N_FOUNDER_PAIRS = 150
REPS = 5
BASE_BIRTH_TIME = 1920.0
GENERATION_YEARS = 30.0

AGE_GRID = np.arange(0, 121, 1.0)
TRUE_CIP = K_POP / (1.0 + np.exp((MID - AGE_GRID) * SLOPE))
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bench_register_pipeline.csv")


def pedigree_birth_times(ids, father, mother):
    """This benchmark's calendar scale, via the promoted public generator."""
    return _pedigree_birth_times(ids, father, mother,
                                 base_birth_year=BASE_BIRTH_TIME,
                                 generation_years=GENERATION_YEARS)


def build_register(rng, ids, father, mother):
    """One CONSISTENT liability field for the whole population, then records.

    Delegates to :func:`ltpred.simulate.simulate_register_liabilities`, binding
    this benchmark's ``H2``/``TRUE_CIP``/``AGE_GRID``/``EVAL_AGE``; see that
    function for why the population is drawn once rather than per proband, and
    how the inbred-person rescaling works. Returns the six columns aligned to
    ``ids``."""
    sim = simulate_register_liabilities(rng, ids, father, mother, h2=H2,
                                        cip_ages=AGE_GRID, cip_values=TRUE_CIP,
                                        eval_age=EVAL_AGE)
    return (sim.status, sim.age, sim.onset, sim.genetic, sim.birth_time,
            sim.residual_var)


def auc_rank(scores, labels):
    """Mann-Whitney AUC: P(score_case > score_control) + 0.5 * P(tie).

    Rank-based with midranks for ties; implemented directly (no sklearn).
    Returns nan when one class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def future_case_prob(est_g, est_var, h2=H2, residual_var=None,
                     index_age=INDEX_AGE, eval_age=EVAL_AGE):
    """Model-consistent P(onset by eval age | unaffected at index age).

    Integrates the proband's posterior g ~ N(est_g, est_var) and their residual
    E against the threshold-crossing onset model: onset in (a, b] iff
    CIP(a) < 1 - Phi(L) <= CIP(b) for full liability L = g + E, i.e. L in
    [Phi^-1(1 - CIP(b)), Phi^-1(1 - CIP(a))). ``residual_var`` defaults to
    ``1 - h2`` for an outbred target; the caller supplies the smaller aligned
    variance on the standardized scale for an inbred target. The benchmark
    cohort is known to be unaffected at age ``a``, so the interval probability
    is divided by P(L < Phi^-1(1 - CIP(a)))."""
    cip_index = float(np.interp(index_age, AGE_GRID, TRUE_CIP))
    cip_eval = float(np.interp(eval_age, AGE_GRID, TRUE_CIP))
    thr_index = norm.ppf(1.0 - cip_index)
    thr_eval = norm.ppf(1.0 - cip_eval)
    est_g = np.asarray(est_g, dtype=float)
    if residual_var is None:
        residual_var = 1.0 - h2
    residual_var = np.asarray(residual_var, dtype=float)
    sd = np.sqrt(residual_var
                 + np.maximum(np.asarray(est_var, dtype=float), 0.0))
    at_risk = norm.cdf((thr_index - est_g) / sd)
    incident = at_risk - norm.cdf((thr_eval - est_g) / sd)
    return np.clip(incident / at_risk, 0.0, 1.0)


def prospective_metrics(est, est_var, future, residual_var):
    """corr, calibration slope, AUC, calibration-in-the-large vs future."""
    y = future.astype(float)
    return {
        "corr": float(np.corrcoef(y, est)[0, 1]),
        "slope": float(np.polyfit(est, y, 1)[0]),
        "auc": auc_rank(est, future),
        "cil_pred": float(future_case_prob(
            est, est_var, residual_var=residual_var).mean()),
        "cil_obs": float(y.mean()),
    }


def corr_slope(g, est):
    return (float(np.corrcoef(g, est)[0, 1]),
            float(np.polyfit(est, g, 1)[0]))


def run_replicate(rng, n_founder_pairs, n_probands):
    """One independent register: Parts 1-3, all arms paired within it.

    Returns (metrics, register) where metrics is a flat CSV-ready dict and
    register the tuple needed by the single-run throughput part."""
    ids, father, mother = simulate_population(
        rng, n_founder_pairs=n_founder_pairs)
    status, age, onset, genetic, birth_time, residual_var = build_register(
        rng, ids, father, mother)

    # Parts 1-2 use an ordinary random GWAS sample from the completed register.
    selected = rng.choice(len(ids), size=min(n_probands, len(ids)),
                          replace=False)
    if selected.size < 2:
        raise ValueError("the accuracy arms require at least two probands")
    probands = [ids[i] for i in selected]
    g = genetic[selected]

    # Part 3 is a distinct prospective risk set: disease-free at age 40.
    eligible = np.flatnonzero(onset > INDEX_AGE)
    if eligible.size < 2:
        raise ValueError(
            "the prospective arm requires at least two age-40-unaffected probands")
    prediction_selected = rng.choice(
        eligible, size=min(n_probands, eligible.size), replace=False)
    prediction_future = onset[prediction_selected] <= EVAL_AGE
    if np.unique(prediction_future).size < 2:
        raise ValueError(
            "the prospective sample has only one outcome class; increase "
            "--n-probands or --n-founder-pairs")
    prediction_probands = [ids[i] for i in prediction_selected]
    prediction_residual_var = residual_var[prediction_selected]
    n_pro = len(prediction_probands)
    index_time = birth_time[prediction_selected] + INDEX_AGE
    metrics = {
        "n_persons": float(len(ids)),
        "n_prediction_probands": float(n_pro),
    }

    # ---- Part 1: oracle CIP, degree 3 vs degree 1 -----------------------------
    out3 = estimate_liabilities(ids, father, mother, probands=probands,
                                status=status, age=age, cip_ages=AGE_GRID,
                                cip_values=TRUE_CIP, k_pop=K_POP, h2=H2,
                                max_degree=3, use="gwas")
    out1 = estimate_liabilities(ids, father, mother, probands=probands,
                                status=status, age=age, cip_ages=AGE_GRID,
                                cip_values=TRUE_CIP, k_pop=K_POP, h2=H2,
                                max_degree=1, use="gwas")
    c3, s3 = corr_slope(g, out3.est)
    c1, s1 = corr_slope(g, out1.est)
    metrics["part1_corr_degree3"] = c3
    metrics["part1_slope_degree3"] = s3
    metrics["part1_corr_degree1"] = c1
    metrics["part1_slope_degree1"] = s1
    metrics["part1_delta_corr_degree3_minus_degree1"] = c3 - c1

    # ---- Part 2: estimated CIP --------------------------------------------------
    entry = np.zeros(len(ids))
    event = status.astype(int)
    curve = aalen_johansen_cip(entry, age, event)
    out_est = estimate_liabilities(ids, father, mother, probands=probands,
                                   status=status, age=age,
                                   cip_ages=curve.ages, cip_values=curve.values,
                                   k_pop=K_POP, h2=H2, max_degree=3,
                                   use="gwas")
    cest, _ = corr_slope(g, out_est.est)
    metrics["part2_corr_estimated_cip"] = cest
    metrics["part2_corr_oracle_cip"] = c3
    metrics["part2_delta_corr_estimated_minus_oracle"] = cest - c3

    # ---- Part 3: prospective prediction ---------------------------------------
    future = prediction_future
    out_pro = estimate_liabilities(
        ids, father, mother, probands=prediction_probands, status=status,
        age=age, cip_ages=AGE_GRID, cip_values=TRUE_CIP, k_pop=K_POP, h2=H2,
        use="prediction", birth_time=birth_time, index_time=index_time)
    # Leaky arm (b): use a calendar time after every full record, thereby
    # retaining relatives' post-index events while the public prediction mode
    # still leaves each proband uninformative.
    after_all_records = float(np.max(birth_time + age) + 1.0)
    out_leak_rel = estimate_liabilities(
        ids, father, mother, probands=prediction_probands, status=status,
        age=age, cip_ages=AGE_GRID, cip_values=TRUE_CIP, k_pop=K_POP, h2=H2,
        use="prediction", birth_time=birth_time,
        index_time=np.full(n_pro, after_all_records))
    # Cumulative-leak arm (c): full records for relatives and the proband.
    # This deliberately calls GWAS mode on the prospective risk set and is not
    # a valid prospective score.
    out_leak_full = estimate_liabilities(
        ids, father, mother, probands=prediction_probands, status=status,
        age=age, cip_ages=AGE_GRID, cip_values=TRUE_CIP, k_pop=K_POP,
        h2=H2, max_degree=3, use="gwas")
    for tag, out in (("honest", out_pro),
                     ("leak_rel", out_leak_rel),
                     ("leak_full", out_leak_full)):
        for key, value in prospective_metrics(
                out.est, out.var, future, prediction_residual_var).items():
            metrics[f"part3_{key}_{tag}"] = value
    metrics["part3_delta_corr_leak_rel_minus_honest"] = (
        metrics["part3_corr_leak_rel"] - metrics["part3_corr_honest"])
    metrics["part3_delta_corr_leak_full_minus_honest"] = (
        metrics["part3_corr_leak_full"] - metrics["part3_corr_honest"])
    metrics["part3_delta_auc_leak_rel_minus_honest"] = (
        metrics["part3_auc_leak_rel"] - metrics["part3_auc_honest"])
    metrics["part3_delta_auc_leak_full_minus_honest"] = (
        metrics["part3_auc_leak_full"] - metrics["part3_auc_honest"])
    register = (ids, father, mother, probands, status, age)
    return metrics, register


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="End-to-end register pipeline: accuracy, CIP-from-data, "
                    "prospective prediction with discrimination metrics, and "
                    "throughput.")
    parser.add_argument("--reps", type=int, default=REPS,
                        help="independent registers for Parts 1-3")
    parser.add_argument("--n-probands", type=int, default=N_PROBANDS)
    parser.add_argument("--n-founder-pairs", type=int, default=N_FOUNDER_PAIRS,
                        help="founder couples per simulated population")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=CSV_PATH,
                        help="CSV artifact path (long format: rep, metric, value)")
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f"Register-pipeline benchmark (trio records -> scores), "
          f"reps={args.reps}, h2={H2}")

    rep_rows, registers = [], []
    for rep in range(args.reps):
        rrng = np.random.default_rng(np.random.PCG64(args.seed + 1000 + rep))
        metrics, register = run_replicate(rrng, args.n_founder_pairs,
                                          args.n_probands)
        rep_rows.append(metrics)
        registers.append(register)
        print(f"  rep {rep + 1}: pop {int(metrics['n_persons'])}  "
              f"deg3 corr {metrics['part1_corr_degree3']:.4f}  "
              f"deg1 corr {metrics['part1_corr_degree1']:.4f}  "
              f"estCIP corr {metrics['part2_corr_estimated_cip']:.4f}  "
              f"pro a/b/c corr {metrics['part3_corr_honest']:.4f}/"
              f"{metrics['part3_corr_leak_rel']:.4f}/"
              f"{metrics['part3_corr_leak_full']:.4f}  "
              f"auc(a) {metrics['part3_auc_honest']:.4f}")

    def series(key):
        return np.array([row[key] for row in rep_rows])

    # ---- Part 1 summary -----------------------------------------------------
    print(f"\nPart 1 (R={args.reps}): accuracy, oracle CIP "
          f"(± across-replicate SE)")
    for tag, degree in (("degree 3 (cousins)", "degree3"),
                        ("degree 1 (first-degree only)", "degree1")):
        mc, sc = mean_se(series(f"part1_corr_{degree}"))
        ms, ss = mean_se(series(f"part1_slope_{degree}"))
        print(f"  {tag:34s} corr {mc:.4f} ± {sc:.4f}  "
              f"slope {ms:.4f} ± {ss:.4f}")
    d_mean, d_se, d_ci = paired_summary(
        series("part1_delta_corr_degree3_minus_degree1"))
    print(f"  paired degree3-minus-degree1 corr: {d_mean:+.4f} ± {d_se:.4f} SE; "
          f"t-based 95% CI [{d_mean - d_ci:+.4f}, {d_mean + d_ci:+.4f}]")

    # ---- Part 2 summary -------------------------------------------------------
    print(f"\nPart 2 (R={args.reps}): CIP estimated from the register vs oracle")
    me, se_ = mean_se(series("part2_corr_estimated_cip"))
    mo, so = mean_se(series("part2_corr_oracle_cip"))
    print(f"  estimated CIP corr {me:.4f} ± {se_:.4f}   "
          f"oracle corr {mo:.4f} ± {so:.4f}")
    d_mean, d_se, d_ci = paired_summary(
        series("part2_delta_corr_estimated_minus_oracle"))
    print(f"  paired estimated-minus-oracle corr: {d_mean:+.4f} ± {d_se:.4f} SE; "
          f"t-based 95% CI [{d_mean - d_ci:+.4f}, {d_mean + d_ci:+.4f}]")

    # ---- Part 3 summary -------------------------------------------------------
    print(f"\nPart 3 (R={args.reps}): prospective prediction of diagnosis in "
          f"({INDEX_AGE:.0f}, {EVAL_AGE:.0f}] (± across-replicate SE)")
    for tag in ("honest", "leak_rel", "leak_full"):
        label = {"honest": "(a) familywise-censored (honest)",
                 "leak_rel": "(b) + relatives' post-index events",
                 "leak_full": "(c) + full records incl. proband"}[tag]
        mc, sc = mean_se(series(f"part3_corr_{tag}"))
        ms, ss = mean_se(series(f"part3_slope_{tag}"))
        ma, sa = mean_se(series(f"part3_auc_{tag}"))
        mp, sp = mean_se(series(f"part3_cil_pred_{tag}"))
        print(f"  {label:34s} corr {mc:.4f} ± {sc:.4f}  "
              f"slope {ms:.4f} ± {ss:.4f}  auc {ma:.4f} ± {sa:.4f}  "
              f"CIL pred {mp:.4f} ± {sp:.4f}")
    mobs, sobs = mean_se(series("part3_cil_obs_honest"))
    print(f"  observed future-case rate {mobs:.4f} ± {sobs:.4f}")
    for tag, label in (("leak_rel", "(b)-(a) relatives' post-index events"),
                       ("leak_full", "(c)-(a) full records incl. proband")):
        dc_mean, dc_se, dc_ci = paired_summary(
            series(f"part3_delta_corr_{tag}_minus_honest"))
        da_mean, da_se, da_ci = paired_summary(
            series(f"part3_delta_auc_{tag}_minus_honest"))
        print(f"  paired {label}: Δcorr {dc_mean:+.4f} ± {dc_se:.4f} SE; "
              f"t 95% CI [{dc_mean - dc_ci:+.4f}, {dc_mean + dc_ci:+.4f}]   "
              f"ΔAUC {da_mean:+.4f} ± {da_se:.4f} SE; "
              f"t 95% CI [{da_mean - da_ci:+.4f}, {da_mean + da_ci:+.4f}]")

    # ---- Part 4: throughput (single run) --------------------------------------
    print("\nPart 4: throughput (single run)")
    ids, father, mother, probands, status, age = registers[0]
    t1 = time.time()
    estimate_liabilities(ids, father, mother, probands=probands,
                         status=status, age=age, cip_ages=AGE_GRID,
                         cip_values=TRUE_CIP, k_pop=K_POP, h2=H2,
                         max_degree=3, use="gwas")
    dt = time.time() - t1
    rate = len(probands) / dt
    print(f"  {len(probands)} probands in {dt:.2f}s ({rate:,.0f} probands/s)")

    # ---- CSV artifact ----------------------------------------------------------
    csv_rows = []
    for rep, metrics in enumerate(rep_rows, start=1):
        csv_rows.extend((rep, key, value) for key, value in metrics.items()
                        if key != "n_persons")
        csv_rows.append((rep, "population_size", metrics["n_persons"]))
    csv_rows.append((0, "part4_probands_per_second", rate))
    csv_rows.append((0, "part4_seconds", dt))
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rep", "metric", "value"])
        writer.writerows(csv_rows)
    print(f"\nwrote {args.output} ({len(csv_rows)} rows)")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
