"""GWAS power: case/control vs LT-FH++ vs PA-FGRS vs an oracle (LT-FH++/ADuLT-style).

The central claim of LT-FH++ and ADuLT is that regressing genotypes on an
*estimated liability* -- which uses family history and age -- recovers association
power that a plain case/control label discards. This benchmark makes that
concrete on simulated genotypes:

  1. simulate causal-SNP genotypes and build each proband's genetic liability from
     them (variance ``h2``);
  2. draw the relatives' liabilities conditional on that value and threshold them
     into a family history;
  3. estimate the proband's genetic liability with the Gibbs (LT-FH++) and PA-FGRS
     back-ends;
  4. run a linear-regression GWAS of the proband genotypes on four phenotypes --
     case/control, LT-FH++, PA-FGRS and the true genetic liability (oracle) -- and
     score each.

Reported per phenotype: mean chi-square at causal SNPs (proportional to effective
sample size), power at genome-wide (p<5e-8) and suggestive (p<1e-4) thresholds,
and lambda_GC at null SNPs (calibration -- should stay ~1). LT-FH++ and PA-FGRS
should beat case/control and match each other, below the oracle ceiling, with no
inflation at nulls.

Genotypes are independent SNPs by default (LD is not needed for the power
comparison, which turns on each phenotype's correlation to the true genetic
value). For real LD, pass HAPNEST-generated genotypes via ``--plink PREFIX``
(see ``hapnest/README.md``).

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


def score(chisq, causal_mask):
    causal = chisq[causal_mask]
    null = chisq[~causal_mask]
    return dict(mean_chi2_causal=float(causal.mean()),
                power_gw=float((causal > GW).mean()),
                power_sug=float((causal > SUG).mean()),
                lambda_gc=lambda_gc(null))


def run(args):
    genotypes = None
    if args.plink:
        X, _ = read_plink_bed(args.plink)
        genotypes = X[:args.n_fam] if args.n_fam < X.shape[0] else X
        print(f"loaded genotypes from {args.plink}: {genotypes.shape}")

    data = simulate_genotype_families(
        fam_vec=args.fam, h2=args.h2, prevalence=args.prev, n_fam=args.n_fam,
        m_snps=args.m_snps, n_causal=args.n_causal, seed=args.seed,
        genotypes=genotypes)
    Xs, causal, true_g, status = data["Xs"], data["causal"], data["true_g"], data["status"]
    n_fam, m = Xs.shape
    causal_mask = np.zeros(m, dtype=bool)
    causal_mask[causal] = True
    print(f"n_fam={n_fam}  m_snps={m}  causal={len(causal)}  "
          f"case rate={status.mean():.3f}")

    gibbs, t_g = estimate(data["families"], args.h2, "gibbs", n_sim=args.n_sim, seed=args.seed)
    pa, t_p = estimate(data["families"], args.h2, "pearson-aitken")
    print(f"estimation: Gibbs {t_g:.2f}s  PA {t_p:.3f}s")

    phenos = {
        "case_control": status.astype(float),
        "LT-FH++ (Gibbs)": gibbs,
        "PA-FGRS": pa,
        "oracle (true g)": true_g,
    }
    rows = []
    base = None
    for name, y in phenos.items():
        s = score(gwas_chisq(Xs, y), causal_mask)
        s["phenotype"] = name
        if name == "case_control":
            base = s["mean_chi2_causal"]
        rows.append(s)
    for s in rows:
        s["effN_vs_cc"] = s["mean_chi2_causal"] / base if base else np.nan

    print(f"\n{'phenotype':18s} {'meanChi2':>9s} {'effN':>6s} "
          f"{'pow5e-8':>8s} {'pow1e-4':>8s} {'lambdaGC':>9s}")
    for s in rows:
        print(f"{s['phenotype']:18s} {s['mean_chi2_causal']:9.2f} "
              f"{s['effN_vs_cc']:5.2f}x {s['power_gw']:8.2%} "
              f"{s['power_sug']:8.2%} {s['lambda_gc']:9.3f}")
    return rows, {name: gwas_chisq(Xs, y) for name, y in phenos.items()}, causal_mask


def write_csv(rows):
    fields = ["phenotype", "mean_chi2_causal", "effN_vs_cc", "power_gw",
              "power_sug", "lambda_gc"]
    path = os.path.join(HERE, "bench_gwas_power.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])
    return path


def plot(rows, chisq, causal_mask):
    plt = get_plt()
    if plt is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    names = [r["phenotype"] for r in rows]
    ax[0].bar(range(len(names)), [r["mean_chi2_causal"] for r in rows],
              color=["#999", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("mean chi2 at causal SNPs")
    ax[0].set_title("(a) power (effective N) at causal SNPs")
    ax[1].bar(range(len(names)), [100 * r["power_sug"] for r in rows],
              color=["#999", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("% causal SNPs detected (p<1e-4)")
    ax[1].set_title("(b) detection rate")
    # (c) QQ of null chi2 -> calibration
    for name in names:
        null = np.sort(chisq[name][~causal_mask])
        expected = stats.chi2.ppf((np.arange(1, len(null) + 1) - 0.5) / len(null), 1)
        ax[2].plot(expected, null, lw=1, label=name)
    lim = [0, max(3, expected.max())]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set_xlabel("expected chi2 (null)")
    ax[2].set_ylabel("observed chi2 (null)")
    ax[2].set_title("(c) calibration at null SNPs")
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
    ap.add_argument("--plink", default=None, help="PLINK prefix for real-LD genotypes")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows, chisq, causal_mask = run(args)
    path = write_csv(rows)
    plot(rows, chisq, causal_mask)
    print(f"\nwrote {os.path.basename(path)} and bench_gwas_power.png")


if __name__ == "__main__":
    main()
