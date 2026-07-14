"""GWAS power: case/control vs classic LT-FH inferred by Gibbs/PA vs an oracle.

The central classic-LT-FH claim is that regressing genotypes on a posterior
genetic-liability phenotype conditioned on family history recovers association
power that the proband's raw case/control label discards. This benchmark has no
age/sex/cohort personalisation and no ADuLT arm; it makes the LT-FH claim concrete
on simulated genotypes:

  1. simulate causal-SNP genotypes and build each proband's genetic liability from
     them (variance ``h2``);
  2. draw the relatives' liabilities conditional on that value and threshold them
     into a family history;
  3. estimate the proband's classic LT-FH liability with Gibbs and PA;
  4. run a linear-regression GWAS of the proband genotypes on four phenotypes --
     case/control, LT-FH (Gibbs), LT-FH (PA), and true genetic liability -- and
     score each.

Reported per phenotype: mean chi-square at causal SNPs (whose noncentral part,
mean chi-square minus 1, is proportional to effective sample size), power at
genome-wide (p<5e-8) and suggestive (p<1e-4) thresholds, and lambda_GC at null
SNPs (calibration -- should stay ~1). The two LT-FH inference
engines should beat case/control and match each other, below the oracle ceiling, with no
inflation at nulls.

Genotypes are independent SNPs by default (LD is not needed for the power
comparison, which turns on each phenotype's correlation to the true genetic
value). For real LD, pass HAPNEST-generated genotypes via ``--plink PREFIX``.
In that mode, variants above ``--ld-null-r2`` with any causal SNP are excluded
from the genomic-control/QQ set, because causal LD proxies are associated rather
than null (see ``hapnest/README.md``).

    python benchmarks/bench_gwas_power.py
    python benchmarks/bench_gwas_power.py --plink data/output/synthetic --n-causal 40
Writes bench_gwas_power.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np
from scipy import stats

from _common import (simulate_genotype_families, estimate, gwas_chisq,
                     lambda_gc, read_plink_bed, get_plt)

HERE = os.path.dirname(os.path.abspath(__file__))

GW = float(stats.chi2.isf(5e-8, 1))     # genome-wide chi2 threshold
SUG = float(stats.chi2.isf(1e-4, 1))    # suggestive chi2 threshold


def score(chisq, causal_mask, calibration_mask):
    causal = chisq[causal_mask]
    null = chisq[calibration_mask]
    return dict(mean_chi2_causal=float(causal.mean()),
                power_gw=float((causal > GW).mean()),
                power_sug=float((causal > SUG).mean()),
                lambda_gc=lambda_gc(null),
                n_calibration_snps=int(calibration_mask.sum()))


def run(args, genotypes, seed):
    data = simulate_genotype_families(
        fam_vec=args.fam, h2=args.h2, prevalence=args.prev, n_fam=args.n_fam,
        m_snps=args.m_snps, n_causal=args.n_causal, seed=seed,
        genotypes=genotypes)
    Xs, causal, true_g, status = data["Xs"], data["causal"], data["true_g"], data["status"]
    n_fam, m = Xs.shape
    causal_mask = np.zeros(m, dtype=bool)
    causal_mask[causal] = True
    calibration_mask = ~causal_mask
    if args.plink:
        # Standardised columns make dot(X_j, X_c)/N the sample correlation.
        # Exclude LD proxies from the set called null for lambda/QQ diagnostics.
        corr_to_causal = (Xs.T @ Xs[:, causal]) / n_fam
        ld_proxy = np.max(corr_to_causal * corr_to_causal, axis=1) >= args.ld_null_r2
        calibration_mask = ~(causal_mask | ld_proxy)
    print(f"n_fam={n_fam}  m_snps={m}  causal={len(causal)}  "
          f"calibration_snps={calibration_mask.sum()}  case rate={status.mean():.3f}")

    gibbs, _ = estimate(data["families"], args.h2, "gibbs", n_sim=args.n_sim,
                        seed=seed)
    pa, _ = estimate(data["families"], args.h2, "pearson-aitken")
    print("inference complete; use bench_scaling.py for controlled timings")

    phenos = {
        "case_control": status.astype(float),
        "LT-FH (Gibbs)": gibbs,
        "LT-FH (PA)": pa,
        "oracle (true g)": true_g,
    }
    rows = []
    all_chisq = {}
    base = None
    for name, y in phenos.items():
        chi2 = gwas_chisq(Xs, y)
        all_chisq[name] = chi2
        s = score(chi2, causal_mask, calibration_mask)
        s["phenotype"] = name
        if name == "case_control":
            base = s["mean_chi2_causal"]
        rows.append(s)
    for s in rows:
        # Under a 1-df alternative E[chi2] = 1 + noncentrality, and the
        # noncentrality—not the raw statistic—scales with effective sample size.
        denom = base - 1.0
        s["effN_vs_cc"] = ((s["mean_chi2_causal"] - 1.0) / denom
                           if denom > 0.0 else np.nan)

    return rows, all_chisq, calibration_mask


def aggregate(runs):
    out = []
    metrics = ["mean_chi2_causal", "effN_vs_cc", "power_gw", "power_sug",
               "lambda_gc"]
    for i, first in enumerate(runs[0]):
        row = dict(phenotype=first["phenotype"], reps=len(runs),
                   n_calibration_snps=first["n_calibration_snps"])
        for key in metrics:
            values = np.asarray([rep[i][key] for rep in runs], float)
            row[key] = float(values.mean())
            row[f"se_{key}"] = (float(values.std(ddof=1) / np.sqrt(values.size))
                                if values.size > 1 else np.nan)
        out.append(row)
    return out


def write_csv(rows):
    fields = ["phenotype", "reps", "mean_chi2_causal", "se_mean_chi2_causal",
              "effN_vs_cc", "se_effN_vs_cc", "power_gw", "se_power_gw",
              "power_sug", "se_power_sug", "lambda_gc", "se_lambda_gc",
              "n_calibration_snps"]
    path = os.path.join(HERE, "bench_gwas_power.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])
    return path


def plot(rows, chisq, calibration_mask):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    names = [r["phenotype"] for r in rows]
    ax[0].bar(range(len(names)), [r["mean_chi2_causal"] for r in rows],
              yerr=[r["se_mean_chi2_causal"] for r in rows], capsize=3,
              color=["#999", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("mean chi2 at causal SNPs")
    ax[0].set_title("(a) power (effective N) at causal SNPs")
    ax[1].bar(range(len(names)), [100 * r["power_sug"] for r in rows],
              yerr=[100 * r["se_power_sug"] for r in rows], capsize=3,
              color=["#999", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("% causal SNPs detected (p<1e-4)")
    ax[1].set_title("(b) detection rate")
    # (c) QQ of null chi2 -> calibration
    for name in names:
        null = np.sort(chisq[name][calibration_mask])
        expected = stats.chi2.ppf((np.arange(1, len(null) + 1) - 0.5) / len(null), 1)
        ax[2].plot(expected, null, lw=1, label=name)
    lim = [0, max(3, expected.max())]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set_xlabel("expected chi2 (null)")
    ax[2].set_ylabel("observed chi2 (null)")
    ax[2].set_title("(c) calibration at null SNPs (representative replicate)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_gwas_power.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=10_000)
    ap.add_argument("--m-snps", type=int, default=5000)
    ap.add_argument("--n-causal", type=int, default=30)
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--prev", type=float, default=0.05)
    ap.add_argument("--fam", nargs="+", default=["m", "f", "s1"])
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent genotype/effect/cohort replicates")
    ap.add_argument("--plink", default=None, help="PLINK prefix for real-LD genotypes")
    ap.add_argument("--ld-null-r2", type=float, default=0.1,
                    help="exclude real-LD causal proxies at or above this r² from lambda/QQ")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not 0.0 <= args.ld_null_r2 <= 1.0:
        ap.error("--ld-null-r2 must be in [0, 1]")
    if args.reps < 1:
        ap.error("--reps must be at least 1")
    genotypes = None
    if args.plink:
        X, _ = read_plink_bed(args.plink)
        genotypes = X[:args.n_fam] if args.n_fam < X.shape[0] else X
        print(f"loaded genotypes from {args.plink}: {genotypes.shape}")

    runs = []
    representative = None
    for rep in range(args.reps):
        print(f"\nreplicate {rep + 1}/{args.reps}")
        run_rows, chisq, calibration_mask = run(args, genotypes, args.seed + rep)
        runs.append(run_rows)
        if representative is None:
            representative = (chisq, calibration_mask)
    rows = aggregate(runs)
    print(f"\n{'phenotype':18s} {'meanChi2':>14s} {'effN':>12s} "
          f"{'pow5e-8':>14s} {'lambdaGC':>14s}")
    for row in rows:
        print(f"{row['phenotype']:18s} "
              f"{row['mean_chi2_causal']:7.2f}±{row['se_mean_chi2_causal']:.2f} "
              f"{row['effN_vs_cc']:5.2f}±{row['se_effN_vs_cc']:.2f}x "
              f"{row['power_gw']:6.2%}±{row['se_power_gw']:.2%} "
              f"{row['lambda_gc']:6.3f}±{row['se_lambda_gc']:.3f}")
    path = write_csv(rows)
    plot(rows, representative[0], representative[1])
    print(f"\nwrote {os.path.basename(path)} and bench_gwas_power.png")


if __name__ == "__main__":
    main()
