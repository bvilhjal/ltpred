"""PGS baseline versus family history, and the PGS + LT-FH joint model.

The paper needs an honest train/test answer to "what does family history add on
top of a polygenic score?"  Same generative framework as ``bench_gwas_power.py``
(independent SNPs, causal effects, proband genetic liability ``g = Xs @ beta``,
relatives drawn conditional on it), but the cohort is split 50/50:

  * TRAIN is the discovery cohort.  A linear-regression GWAS is run on the
    case/control label (and, for the GWAS-power arms, on the LT-FH estimate and
    the oracle ``g``); the case/control summary statistics define the PGS
    weights.
  * TEST is never touched by the GWAS or the PGS weight fitting.  The PGS is
    scored there, the family-history (LT-FH) estimate is computed there, and
    both are evaluated against the held-out true genetic liability.  The joint
    combiner is fit by deterministic cross-fitting within TEST, so each
    proband's joint prediction is fit without that proband's true liability.

Arms (all evaluated on TEST unless noted):

  1. case/control -- GWAS on the 0/1 label (train); the raw label is also the
     baseline *predictor* on test.
  2. LT-FH -- the package's Pearson--Aitken estimate of the proband's genetic
     liability from the family history.  The bounds are classic LT-FH (single
     prevalence threshold for every member): this generative framework has no
     age/sex/cohort structure, so personalised LT-FH++ thresholds would be
     identical for every member and add nothing; bench_ltfhpp_personalization
     covers the personalised case.  PA is the primary engine (it matches Gibbs
     to corr >= 0.997; RESULTS.md section 1).  The GWAS on this estimate gives
     the causal-SNP NCP ratio over case/control, exactly as in
     ``bench_gwas_power.py``.
  3. PGS-only -- marginal Z-scored weights ``w_j = sqrt(n_train) * corr(x_j,
     y)`` from the train case/control GWAS.  With independent SNPs there is no
     LD to shrink for, so this self-contained numpy score is the LDpred-inf
     limit; no external PGS package is required.  ``--pgs-backend ldpred3``
     instead fits LDpred3-auto on the train summary statistics (LD reference =
     train cohort) and scores the test cohort via
     ``ldpred3.run_ldpred3_prs`` + ``score_from_weights``; it needs the
     ``ldpred3`` package importable (the ldpred3 conda env, not ltpred314).
  4. PGS + LT-FH joint -- cross-fitted OLS of the held-out true ``g`` on both
     scores.  Each test fold is predicted by coefficients fit on the other
     test folds; reports squared-correlation R^2, the incremental R^2 of each
     score over the other, and corr(PGS, LT-FH).

Theory check.  docs/algorithm.md ("Expected correlation between a PGS and the
family-history score") gives, under a conditionally-independent measurement
model, ``Corr(PGS, FH) = a * b * sqrt(p)`` with ``a = Corr(PGS, s)`` (the PGS
accuracy against the SNP-captured part), ``b = Corr(FH, g)``, and
``p = h2_SNP / h2_total``.  In this design the true genetic value is built
entirely from the simulated SNPs, so ``h2_SNP = h2_total`` and ``p = 1``; the
prediction reported here is therefore ``a * b`` with ``a = Corr(PGS, g)`` and
``b = Corr(FH, g)`` measured on TEST (equivalently ``sqrt(R2_pgs * R2_fh *
p)`` at ``p = 1``).  Conditional independence holds by construction: the PGS
error is train-cohort sampling noise, the LT-FH error is posterior uncertainty
from the relatives, and the two cohorts do not overlap.

Two different gain metrics appear and are kept separate, as in the rest of the
suite: the GWAS arms report **causal-SNP NCP ratios** (``mean chi2 - 1``,
proportional to effective sample size), while the prediction arms report
**squared-correlation R^2** against held-out ``g``.  Their magnitudes are not
comparable to each other.

    python benchmarks/bench_pgs_comparison.py
    python benchmarks/bench_pgs_comparison.py --reps 10 --m-snps 5000
    python benchmarks/bench_pgs_comparison.py --pgs-backend ldpred3 --reps 1 --n-fam 2000

Writes bench_pgs_comparison.csv (+ .png if matplotlib is present).
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats

from _common import (simulate_genotype_families, gwas_chisq, lambda_gc,
                     get_plt)
from ltpred.estimate import estimate_liability_pa_arrays

HERE = os.path.dirname(os.path.abspath(__file__))

CC = "case/control"
LTFH = "LT-FH (PA)"
PGS = "PGS"
JOINT = "PGS + LT-FH joint"
ORACLE = "oracle (true g)"

MAF_LOW, MAF_HIGH = 0.05, 0.5   # same range simulate_genotype_families uses


# --------------------------------------------------------------------------- #
#  PGS backends                                                               #
# --------------------------------------------------------------------------- #
def numpy_pgs(Xs_train, y_train, Xs_test):
    """Independent-SNP score: marginal Z weights from the train GWAS.

    ``w_j = sqrt(n) * corr(x_j, y)`` is the signed Z statistic of SNP ``j``;
    with no LD this is the LDpred-infinitesimal weight up to shrinkage toward
    the heritability, and any overall rescaling is irrelevant for the
    correlation/R^2 metrics reported."""
    y = np.asarray(y_train, dtype=np.float64)
    y = (y - y.mean()) / y.std()
    w = (Xs_train.T @ y) / np.sqrt(len(y))
    return Xs_test @ w


def _write_plink(prefix, X):
    """Write a SNP-major PLINK fileset from an ``(n, m)`` 0/1/2 dosage matrix.

    Encoding inverts ``_common._BED_MAP``: dosage 2 -> code 00 (homozygous A1,
    the counted/effect allele), 1 -> 10, 0 -> 11.  All variants sit on
    chromosome 1 with A1=A / A2=G; only the ldpred3 backend consumes these."""
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    code = np.empty(X.shape, dtype=np.uint8)
    code[X == 2.0] = 0b00
    code[X == 1.0] = 0b10
    code[X == 0.0] = 0b11
    pad = (-n) % 4
    if pad:
        code = np.vstack([code, np.zeros((pad, m), dtype=np.uint8)])
    packed = np.zeros((m, (n + pad) // 4), dtype=np.uint8)
    for k in range(4):
        packed |= code[k::4].T << (2 * k)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(b"\x6c\x1b\x01")
        fh.write(packed.tobytes())
    with open(prefix + ".bim", "w") as fh:
        for j in range(m):
            fh.write(f"1\trs{j}\t0\t{j + 1}\tA\tG\n")
    with open(prefix + ".fam", "w") as fh:
        for i in range(n):
            fh.write(f"F{i}\tI{i}\t0\t0\t0\t-9\n")


def _write_sumstats(path, X_train, y_train):
    """Marginal linear-regression summary statistics on the train cohort."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    xc = X - X.mean(axis=0)
    yc = y - y.mean()
    var_x = (xc * xc).sum(axis=0)
    var_x[var_x == 0.0] = 1.0
    beta = (xc.T @ yc) / var_x
    yss = float(yc @ yc)
    rss = np.maximum(yss - beta * (xc.T @ yc), 1e-12)
    se = np.sqrt(rss / ((n - 2) * var_x))
    with open(path, "w") as fh:
        fh.write("ID\tCHR\tPOS\tA1\tA2\tBETA\tSE\tN\tEAF\n")
        for j in range(X.shape[1]):
            fh.write(f"rs{j}\t1\t{j + 1}\tA\tG\t{beta[j]:.6g}\t{se[j]:.6g}"
                     f"\t{n}\t{X[:, j].mean() / 2.0:.6f}\n")


def ldpred3_pgs(X_train, y_train, X_test):
    """LDpred3-auto baseline: fit on train summary stats, score the test cohort.

    Writes the two cohorts and the train summary statistics to a temporary
    PLINK fileset, fits with the train cohort as its own LD reference
    (immaterial here -- the SNPs are independent) and scores via saved weights
    so the test cohort is scored out-of-sample.  Requires the optional
    ``ldpred3`` package (the ldpred3 conda env; not a dependency of ltpred)."""
    try:
        import ldpred3
    except ImportError as exc:
        raise SystemExit(
            "--pgs-backend ldpred3 needs the optional ldpred3 package, which is "
            "not importable in this environment; rerun with the ldpred3 conda "
            "env (e.g. /Users/au507860/anaconda3/envs/ldpred3/bin/python) or "
            f"use the default numpy backend.  ({exc})")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        train_prefix = os.path.join(tmp, "train")
        test_prefix = os.path.join(tmp, "test")
        _write_plink(train_prefix, X_train)
        _write_plink(test_prefix, X_test)
        ss_path = os.path.join(tmp, "sumstats.tsv")
        _write_sumstats(ss_path, X_train, y_train)
        res = ldpred3.run_ldpred3_prs(ss_path, train_prefix, method="auto")
        weights_path = os.path.join(tmp, "weights.txt")
        res.write_weights(weights_path)
        scored = ldpred3.score_from_weights(weights_path, test_prefix)
        return np.asarray(scored.scores, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  One replicate                                                              #
# --------------------------------------------------------------------------- #
def _family_bounds(families):
    """``(roles, lower, upper)`` arrays from the simulated Family objects."""
    roles = [m.role for m in families[0].members]
    lower = np.array([[m.lower for m in fam.members] for fam in families], float)
    upper = np.array([[m.upper for m in fam.members] for fam in families], float)
    return roles, lower, upper


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _cross_fitted_joint_predictions(g, s1, s2, n_folds, seed):
    """Return deterministic out-of-fold predictions from the joint OLS.

    The discovery cohort has already been excluded.  This second layer of
    sample splitting keeps each test proband's true ``g`` out of the OLS fit
    used to predict that proband.
    """
    g = np.asarray(g, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    if g.ndim != 1 or s1.shape != g.shape or s2.shape != g.shape:
        raise ValueError("g, s1, and s2 must be one-dimensional and aligned")
    if not 2 <= n_folds <= len(g):
        raise ValueError("n_folds must be between 2 and the test sample size")

    order = np.random.default_rng(seed).permutation(len(g))
    fold_id = np.empty(len(g), dtype=np.int64)
    fold_id[order] = np.arange(len(g)) % n_folds
    prediction = np.empty(len(g), dtype=np.float64)
    Z = np.column_stack([np.ones(len(g)), s1, s2])
    for fold in range(n_folds):
        evaluate = fold_id == fold
        fit = ~evaluate
        beta, *_ = np.linalg.lstsq(Z[fit], g[fit], rcond=None)
        prediction[evaluate] = Z[evaluate] @ beta
    return prediction


def _cross_fitted_joint_r2(g, s1, s2, n_folds, seed):
    """Squared-correlation R^2 of the joint OLS out-of-fold predictions."""
    prediction = _cross_fitted_joint_predictions(
        g, s1, s2, n_folds=n_folds, seed=seed)
    corr = _corr(prediction, g)
    return corr * corr


def _joint_crossfit_seed(seed, rep):
    """Derive a replicate-specific stream independent of simulation streams."""
    state = np.random.SeedSequence(
        [seed, rep, 0x4A4F494E]).generate_state(1, dtype=np.uint32)
    return int(state[0])


def run_rep(args, rep):
    """One independent genotype/effect/cohort replicate; returns its rows."""
    seed = args.seed + rep
    rng = np.random.default_rng(args.seed + 10_000 * rep)
    maf = rng.uniform(MAF_LOW, MAF_HIGH, args.m_snps)
    X = rng.binomial(2, maf, size=(args.n_fam, args.m_snps)).astype(np.float64)
    data = simulate_genotype_families(
        fam_vec=args.fam, h2=args.h2, prevalence=args.prev, n_fam=args.n_fam,
        m_snps=args.m_snps, n_causal=args.n_causal, seed=seed, genotypes=X)
    Xs, g, status = data["Xs"], data["true_g"], data["status"]
    causal_mask = np.zeros(args.m_snps, dtype=bool)
    causal_mask[data["causal"]] = True
    null_mask = ~causal_mask

    roles, lower, upper = _family_bounds(data["families"])
    fh, _ = estimate_liability_pa_arrays(roles, lower, upper, h2=args.h2,
                                         out="genetic")

    n_train = int(round(args.train_frac * args.n_fam))
    perm = rng.permutation(args.n_fam)
    tr, te = perm[:n_train], perm[n_train:]
    print(f"  n_train={len(tr)}  n_test={len(te)}  "
          f"case rate={status.mean():.3f}")

    def row(arm):
        return dict(row_type="replicate", rep=rep, arm=arm, comparison="",
                    metric="", corr_g=np.nan, r2_g=np.nan,
                    mean_chi2_causal=np.nan, ncp_ratio_vs_cc=np.nan,
                    lambda_gc=np.nan, corr_pgs_fh=np.nan,
                    theory_corr_pgs_fh=np.nan, incr_r2_pgs_over_fh=np.nan,
                    incr_r2_fh_over_pgs=np.nan)

    # GWAS arms on the train cohort: causal-SNP NCP ratio and lambda_GC.
    train_phenos = {CC: status[tr].astype(float), LTFH: fh[tr], ORACLE: g[tr]}
    by_arm, base = {}, None
    for name, y in train_phenos.items():
        chi2 = gwas_chisq(Xs[tr], y)
        r = row(name)
        r["mean_chi2_causal"] = float(chi2[causal_mask].mean())
        r["lambda_gc"] = lambda_gc(chi2[null_mask])
        if name == CC:
            base = r["mean_chi2_causal"] - 1.0
        by_arm[name] = r
    for r in by_arm.values():
        denom = base
        r["ncp_ratio_vs_cc"] = ((r["mean_chi2_causal"] - 1.0) / denom
                                if denom and denom > 0.0 else np.nan)

    # Test-cohort prediction metrics against the held-out true g.
    g_te = g[te]
    if args.pgs_backend == "ldpred3":
        pgs = ldpred3_pgs(X[tr], status[tr].astype(float), X[te])
    else:
        pgs = numpy_pgs(Xs[tr], status[tr].astype(float), Xs[te])
    fh_te = fh[te]
    a = _corr(pgs, g_te)              # Corr(PGS, g); s = g here (p = 1)
    b = _corr(fh_te, g_te)            # Corr(FH, g)
    r2_pgs, r2_fh = a * a, b * b
    joint_seed = _joint_crossfit_seed(args.seed, rep)
    r2_joint = _cross_fitted_joint_r2(
        g_te, pgs, fh_te, args.joint_folds, joint_seed)

    by_arm[PGS] = row(PGS)
    by_arm[PGS]["corr_g"], by_arm[PGS]["r2_g"] = a, r2_pgs
    by_arm[LTFH]["corr_g"], by_arm[LTFH]["r2_g"] = b, r2_fh
    by_arm[CC]["corr_g"] = _corr(status[te].astype(float), g_te)
    by_arm[CC]["r2_g"] = by_arm[CC]["corr_g"] ** 2

    by_arm[JOINT] = row(JOINT)
    r = by_arm[JOINT]
    r["r2_g"] = r2_joint
    r["joint_seed"] = joint_seed
    r["incr_r2_pgs_over_fh"] = r2_joint - r2_fh
    r["incr_r2_fh_over_pgs"] = r2_joint - r2_pgs
    r["corr_pgs_fh"] = _corr(pgs, fh_te)
    # docs/algorithm.md: Corr(PGS, FH) = a * b * sqrt(p); p = h2_SNP/h2_total
    # is exactly 1 in this design (g is fully SNP-captured), so a * b.
    r["theory_corr_pgs_fh"] = a * b
    order = (CC, LTFH, PGS, JOINT, ORACLE)
    return [by_arm[name] for name in order]


# --------------------------------------------------------------------------- #
#  Paired contrasts and aggregation                                           #
# --------------------------------------------------------------------------- #
def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    if values.size == 1:
        return mean, np.nan, np.nan, np.nan, 1
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(values.size)
    ci95 = float(stats.t.ppf(0.975, values.size - 1) * se)
    return mean, sd, float(se), ci95, int(values.size)


def paired_contrasts(replicate_rows):
    """Paired per-replicate differences, reported mean/sd/se/t-CI (section 15 style)."""
    by_arm = {}
    for r in replicate_rows:
        by_arm.setdefault(r["arm"], {})[r["rep"]] = r
    specs = [
        (f"{LTFH} - {CC}", LTFH, CC, "ncp_ratio_vs_cc"),
        (f"{LTFH} - {CC}", LTFH, CC, "corr_g"),
        (f"{PGS} - {CC}", PGS, CC, "r2_g"),
        (f"{JOINT} - {PGS}", JOINT, PGS, "r2_g"),
        (f"{JOINT} - {LTFH}", JOINT, LTFH, "r2_g"),
    ]
    out = []
    for label, right, left, metric in specs:
        reps = sorted(set(by_arm.get(right, ())) & set(by_arm.get(left, ())))
        delta = [by_arm[right][rep][metric] - by_arm[left][rep][metric]
                 for rep in reps]
        mean, sd, se, ci95, n = _mean_ci(delta)
        out.append(dict(row_type="paired_contrast", rep="", arm="",
                        comparison=label, metric=metric, delta_mean=mean,
                        delta_sd=sd, delta_se=se, delta_ci95=ci95,
                        contrast_n=n))
    # theory agreement: predicted a*b*sqrt(p) minus observed corr(PGS, FH)
    joint = by_arm.get(JOINT, {})
    reps = sorted(joint)
    delta = [joint[rep]["theory_corr_pgs_fh"] - joint[rep]["corr_pgs_fh"]
             for rep in reps]
    mean, sd, se, ci95, n = _mean_ci(delta)
    out.append(dict(row_type="paired_contrast", rep="", arm="",
                    comparison="theory a*b*sqrt(p) - observed corr(PGS,FH)",
                    metric="corr_pgs_fh", delta_mean=mean, delta_sd=sd,
                    delta_se=se, delta_ci95=ci95, contrast_n=n))
    return out


def aggregate(replicate_rows):
    """Across-replicate mean +/- SE for every arm/metric."""
    metrics = ["corr_g", "r2_g", "mean_chi2_causal", "ncp_ratio_vs_cc",
               "lambda_gc", "corr_pgs_fh", "theory_corr_pgs_fh",
               "incr_r2_pgs_over_fh", "incr_r2_fh_over_pgs"]
    out = []
    for arm in (CC, LTFH, PGS, JOINT, ORACLE):
        sub = [r for r in replicate_rows if r["arm"] == arm]
        if not sub:
            continue
        row = dict(arm=arm, reps=len(sub))
        for key in metrics:
            mean, _, se, _, _ = _mean_ci([r[key] for r in sub])
            row[key], row[f"se_{key}"] = mean, se
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
#  Output                                                                     #
# --------------------------------------------------------------------------- #
def write_csv(rows, args):
    fields = ["row_type", "rep", "arm", "comparison", "metric",
              "n_fam", "m_snps", "n_causal", "h2", "prev", "fam",
              "train_frac", "pgs_backend", "seed", "joint_fit_design",
              "joint_folds", "joint_seed",
              "corr_g", "r2_g", "mean_chi2_causal", "ncp_ratio_vs_cc",
              "lambda_gc", "corr_pgs_fh", "theory_corr_pgs_fh",
              "incr_r2_pgs_over_fh", "incr_r2_fh_over_pgs",
              "delta_mean", "delta_sd", "delta_se", "delta_ci95", "contrast_n"]
    meta = dict(n_fam=args.n_fam, m_snps=args.m_snps, n_causal=args.n_causal,
                h2=args.h2, prev=args.prev, fam="+".join(args.fam),
                train_frac=args.train_frac, pgs_backend=args.pgs_backend,
                seed=args.seed,
                joint_fit_design="seeded_shuffled_test_kfold_ols",
                joint_folds=args.joint_folds)
    path = os.path.join(HERE, "bench_pgs_comparison.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**meta, **r})
    return path


def plot(agg, replicate_rows):
    plt = get_plt()
    if plt is None:
        return
    by_arm = {r["arm"]: r for r in agg}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    gwas_arms = [CC, LTFH, ORACLE]
    ax[0].bar(range(len(gwas_arms)),
              [by_arm[a]["ncp_ratio_vs_cc"] for a in gwas_arms],
              yerr=[by_arm[a]["se_ncp_ratio_vs_cc"] for a in gwas_arms],
              capsize=3, color=["#999", "#1f77b4", "#2ca02c"])
    ax[0].set_xticks(range(len(gwas_arms)))
    ax[0].set_xticklabels(gwas_arms, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("causal-SNP NCP ratio vs case/control")
    ax[0].set_title("(a) GWAS power on the train cohort")

    pred_arms = [CC, PGS, LTFH, JOINT]
    ax[1].bar(range(len(pred_arms)), [by_arm[a]["r2_g"] for a in pred_arms],
              yerr=[by_arm[a]["se_r2_g"] for a in pred_arms], capsize=3,
              color=["#999", "#ff7f0e", "#1f77b4", "#9467bd"])
    ax[1].set_xticks(range(len(pred_arms)))
    ax[1].set_xticklabels(pred_arms, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("R2 against held-out true g")
    ax[1].set_title("(b) prediction on the test cohort")

    joint = sorted((r for r in replicate_rows if r["arm"] == JOINT),
                   key=lambda r: r["rep"])
    theory = np.asarray([r["theory_corr_pgs_fh"] for r in joint])
    observed = np.asarray([r["corr_pgs_fh"] for r in joint])
    lim = [0.0, float(max(theory.max(initial=0.0),
                          observed.max(initial=0.0)) * 1.2 + 0.02)]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].plot(theory, observed, "o", color="#1f77b4")
    mean_t, _, se_t, _, _ = _mean_ci(theory)
    mean_o, _, se_o, _, _ = _mean_ci(observed)
    ax[2].errorbar([mean_t], [mean_o], xerr=[se_t], yerr=[se_o], fmt="s",
                   color="#d62728", capsize=4, label="mean ± SE")
    ax[2].set_xlabel("theory: a * b * sqrt(p)  (p = 1 here)")
    ax[2].set_ylabel("observed corr(PGS, LT-FH)")
    ax[2].set_title("(c) PGS vs family-history correlation")
    ax[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_pgs_comparison.png"), dpi=130)


def summarize(agg, contrasts):
    print(f"\n{'arm':22s} {'corr_g':>16s} {'R2_g':>16s} {'NCP/cc':>14s} "
          f"{'lambdaGC':>14s}")
    for r in agg:
        def cell(key, fmt):
            v, se = r[key], r[f"se_{key}"]
            return (fmt.format(v) + "±" + fmt.format(se)
                    if np.isfinite(v) else "—")
        print(f"{r['arm']:22s} {cell('corr_g', '{:.3f}'):>16s} "
              f"{cell('r2_g', '{:.3f}'):>16s} "
              f"{cell('ncp_ratio_vs_cc', '{:.2f}'):>14s} "
              f"{cell('lambda_gc', '{:.3f}'):>14s}")
    print("\npaired contrasts (right - left; t-based 95% CI)")
    for c in contrasts:
        print(f"  {c['comparison']} [{c['metric']}]: "
              f"Δ={c['delta_mean']:+.4f} ± {c['delta_ci95']:.4f} "
              f"(n={c['contrast_n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=10_000)
    ap.add_argument("--m-snps", type=int, default=2000)
    ap.add_argument("--n-causal", type=int, default=30)
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--prev", type=float, default=0.05)
    ap.add_argument("--fam", nargs="+", default=["m", "f", "s1"])
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="cohort fraction used for the discovery GWAS / PGS fit")
    ap.add_argument("--joint-folds", type=int, default=2,
                    help="cross-fitting folds for the joint OLS within test")
    ap.add_argument("--pgs-backend", choices=["numpy", "ldpred3"],
                    default="numpy",
                    help="numpy: self-contained Z-scored marginal weights; "
                         "ldpred3: optional LDpred3-auto fit (needs the "
                         "ldpred3 package importable)")
    ap.add_argument("--reps", type=int, default=5,
                    help="independent genotype/effect/cohort replicates")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.reps < 1:
        ap.error("--reps must be at least 1")
    if not 0.0 < args.train_frac < 1.0:
        ap.error("--train-frac must be strictly between 0 and 1")
    if not 0.0 < args.h2 < 1.0:
        ap.error("--h2 must be strictly between 0 and 1")
    if not 0.0 < args.prev < 0.5:
        ap.error("--prev must be strictly between 0 and 0.5")
    if args.n_causal < 1:
        ap.error("--n-causal must be at least 1")
    if args.m_snps < args.n_causal + 20:
        ap.error("--m-snps must leave at least 20 null SNPs for lambda_GC")
    n_train = int(round(args.train_frac * args.n_fam))
    n_test = args.n_fam - n_train
    if n_train < 10 or n_test < 10:
        ap.error("--n-fam/--train-frac must give at least 10 train and 10 test")
    if not 2 <= args.joint_folds <= n_test:
        ap.error("--joint-folds must be between 2 and the test sample size")

    replicate_rows = []
    for rep in range(args.reps):
        print(f"replicate {rep + 1}/{args.reps}")
        replicate_rows.extend(run_rep(args, rep))
    agg = aggregate(replicate_rows)
    contrasts = paired_contrasts(replicate_rows)
    summarize(agg, contrasts)
    path = write_csv(list(replicate_rows) + contrasts, args)
    plot(agg, replicate_rows)
    print(f"\nwrote {os.path.basename(path)} and bench_pgs_comparison.png")


if __name__ == "__main__":
    main()
