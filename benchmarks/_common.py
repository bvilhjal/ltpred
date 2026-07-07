"""Shared helpers for the ltpred benchmark suite.

Two simulation paths feed the benchmarks:

* :func:`simulate_families` -- draws whole families straight from the LT-FH++
  covariance (genetic ``g``, full ``o``, relatives jointly MVN) and thresholds
  them. The true genetic liability is known (``sim.genetic``), so accuracy,
  scaling and censoring benchmarks need nothing else.
* :func:`simulate_genotype_families` -- for the GWAS-power benchmark: simulates
  causal-SNP genotypes, builds each proband's genetic liability from them, then
  draws the relatives' liabilities *conditional on that value* from the same
  family covariance. This gives both a genotype matrix to associate against and a
  correctly correlated family history to estimate from.

Plus small utilities the scripts share: a vectorised linear-regression GWAS
(:func:`gwas_chisq`), method runners (:func:`estimate`), and Agg-safe plotting.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# import ltpred from the repo root without installing it
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.simulate import simulate_under_LTM_single          # noqa: E402
from ltpred.covariance import construct_covmat_single, correct_positive_definite  # noqa: E402
from ltpred.thresholds import liability_threshold, convert_age_to_thresh, convert_age_to_cir  # noqa: E402
from ltpred.family import Family, Member                       # noqa: E402
from ltpred.estimate import estimate_liability                 # noqa: E402


# --------------------------------------------------------------------------- #
#  Simulation                                                                 #
# --------------------------------------------------------------------------- #
def simulate_families(fam_vec, h2, prevalence, n_fam, seed, use_age=False,
                      mid_point=60.0, slope=1.0 / 8.0):
    """Families drawn from the LT-FH++ covariance; ``sim.genetic`` is the truth."""
    return simulate_under_LTM_single(fam_vec=fam_vec, h2=h2, n_sim=n_fam,
                                     pop_prev=prevalence, use_age=use_age,
                                     mid_point=mid_point, slope=slope, seed=seed)


def simulate_genotype_families(fam_vec, h2, prevalence, n_fam, m_snps,
                               n_causal, seed, maf_low=0.05, maf_high=0.5,
                               genotypes=None):
    """Genotype-anchored families for the GWAS benchmark.

    Simulates (or accepts) ``n_fam x m_snps`` genotypes, draws ``n_causal`` causal
    effects so the proband's genetic liability ``g = X_std @ beta`` has variance
    ``h2``, then samples each family's remaining liabilities (full ``o`` and the
    relatives) *conditional on* ``g`` from the LT-FH++ covariance and thresholds
    them at ``prevalence``. Returns a dict with the standardised genotypes, causal
    index, true ``g``, proband ``status`` and the list of :class:`Family` inputs."""
    rng = np.random.default_rng(seed)
    if genotypes is None:
        maf = rng.uniform(maf_low, maf_high, size=m_snps)
        X = rng.binomial(2, maf, size=(n_fam, m_snps)).astype(np.float64)
    else:
        X = np.asarray(genotypes, dtype=np.float64)
        n_fam, m_snps = X.shape
    # standardise genotypes (drop monomorphic columns to unit sd)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / sd

    causal = rng.choice(m_snps, size=n_causal, replace=False)
    beta = np.zeros(m_snps)
    beta[causal] = rng.normal(0.0, np.sqrt(h2 / n_causal), size=n_causal)
    g = Xs @ beta
    g *= np.sqrt(h2) / g.std()          # fix genetic-liability variance to h2

    # family covariance over [g, o, relatives]; sample the rest given g
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles                                  # g, o, <relatives>
    rest = list(range(1, len(roles)))
    cgg = cov[0, 0]
    cross = cov[np.ix_(rest, [0])][:, 0]                   # cov(rest, g)
    cond_mean = np.outer(g, cross / cgg)                   # (n_fam, d-1)
    cond_cov = cov[np.ix_(rest, rest)] - np.outer(cross, cross) / cgg
    cond_cov = correct_positive_definite(cond_cov)[0]
    noise = rng.multivariate_normal(np.zeros(len(rest)), cond_cov, size=n_fam)
    rest_liab = cond_mean + noise                          # columns follow roles[1:]

    t = float(liability_threshold(prevalence))
    status_rest = rest_liab > t                            # o + relatives
    non_g = roles[1:]
    families = []
    for i in range(n_fam):
        members = []
        for k, role in enumerate(non_g):
            case = bool(status_rest[i, k])
            lo, hi = (t, np.inf) if case else (-np.inf, t)
            members.append(Member(role=role, lower=lo, upper=hi))
        families.append(Family(fam_id=i, members=members))

    o_col = non_g.index("o")
    return dict(Xs=Xs, causal=causal, true_g=g, status=status_rest[:, o_col],
                families=families, threshold=t)


# --------------------------------------------------------------------------- #
#  Estimation                                                                 #
# --------------------------------------------------------------------------- #
def estimate(families, h2, method, *, n_sim=25_000, burn_in=800, tol=0.03,
             seed=0, use_mixture=False):
    """Run one estimator and time it; returns ``(genetic_estimate, seconds)``.

    ``method`` is ``"gibbs"`` (LT-FH++) or ``"pearson-aitken"`` / ``"pa"``
    (PA-FGRS). ``use_mixture`` turns on the PA censored-control correction."""
    t0 = time.time()
    res = estimate_liability(families, h2=h2, method=method, out=("genetic",),
                             use_mixture=use_mixture, tol=tol, n_sim=n_sim,
                             burn_in=burn_in, seed=seed)
    return res.est["genetic"], time.time() - t0


# --------------------------------------------------------------------------- #
#  GWAS                                                                        #
# --------------------------------------------------------------------------- #
def gwas_chisq(Xs, phenotype):
    """Per-SNP 1-df association chi-square for a linear-regression GWAS.

    ``Xs`` is column-standardised genotypes ``(n, m)`` and ``phenotype`` any
    ``(n,)`` outcome (case/control label or an estimated liability). Returns
    ``chi2_j = n * corr(x_j, y)^2`` for every SNP -- the score statistic whose
    non-centrality at a causal SNP scales with the phenotype's correlation to the
    true genetic value, i.e. with effective sample size."""
    y = np.asarray(phenotype, dtype=np.float64)
    y = y - y.mean()
    ysd = y.std()
    if ysd == 0:
        return np.zeros(Xs.shape[1])
    y = y / ysd
    n = len(y)
    corr = (Xs.T @ y) / n
    return n * corr * corr


def lambda_gc(chisq_null):
    """Genomic-control inflation ``lambda_GC`` = median(chi2) / 0.4549 (1 df)."""
    return float(np.median(chisq_null) / 0.454936)


# --------------------------------------------------------------------------- #
#  PLINK genotype reader (for the HAPNEST opt-in)                             #
# --------------------------------------------------------------------------- #
_BED_MAP = np.array([2.0, np.nan, 1.0, 0.0])   # 2-bit code -> a2 dosage; 01 = missing


def read_plink_bed(prefix):
    """Read a SNP-major PLINK ``.bed``/``.bim``/``.fam`` fileset to a dosage matrix.

    Returns ``(X, n_snp)`` where ``X`` is ``(n_indiv, n_snp)`` float dosages (0/1/2)
    with missing calls mean-imputed per SNP. Enough to feed HAPNEST-generated real
    -LD genotypes into :func:`simulate_genotype_families` (pass as ``genotypes=``).
    Standard 2-bit encoding, individual-major within each SNP byte block."""
    with open(prefix + ".fam") as fh:
        n = sum(1 for _ in fh)
    with open(prefix + ".bim") as fh:
        m = sum(1 for _ in fh)
    with open(prefix + ".bed", "rb") as fh:
        magic = fh.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError("not a PLINK .bed file")
        if magic[2] != 1:
            raise ValueError("only SNP-major .bed is supported")
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
    bytes_per_snp = (n + 3) // 4
    raw = raw.reshape(m, bytes_per_snp)
    # unpack 4 individuals per byte (2 bits each, low bits first)
    codes = np.empty((m, bytes_per_snp * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (raw >> (2 * k)) & 0b11
    codes = codes[:, :n]
    X = _BED_MAP[codes].T.copy()               # (n_indiv, n_snp)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, m


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #
def get_plt():
    """Return a headless (Agg) matplotlib pyplot, or ``None`` if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover
        return None
