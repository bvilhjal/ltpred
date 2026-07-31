"""Probit estimation of residual-scale genetic variance (and the Lee bridge).

The probit model IS the liability-threshold model (residual variance 1), so
per-SNP residual-scale genetic variance is q = 2 f (1-f) beta². It is not a
fraction of total latent variance; after aggregating independent or suitably
LD-pruned variants, that fraction is q / (1 + q). This benchmark
simulates a polygenic disease on that convention (liab = X beta + eps,
eps ~ N(0, 1); effects scaled to total Var(X beta) = 0.5; threshold at the
empirical quantile for exact prevalence K = 0.1) and checks the routes from
case-control data to residual-scale genetic variance and, separately, the
fraction of total liability variance:

  truth (probit scale): sum_j 2 f_j (1-f_j) beta_j² = 0.5; Lee fraction of
  total liability = 0.5 / 1.5 = 1/3.
  (a)  joint probit fit (all SNPs at once) -> sum_j 2 f_j (1-f_j) beta_j²:
       the exact route; should recover the truth.
  (a') marginal probit fits (one SNP at a time, the GWAS practice): same
       identity, but each marginal fit absorbs the other SNPs' effects into
       its residual, attenuating it by ~1/sqrt(1 + V_background) -- the
       honest caveat for marginal estimation at substantial h².
  (b)  probit z route: sum_j (z_j² - 1) / (N wbar), subtracting the null
       second moment, with the probit information
       weight wbar = phi(a)^2/(Phi(a)(1-Phi(a))) -- the summary-statistic
       version of (a').
  (c)  OLS + Lee 2011 bridge: the null-adjusted observed-scale total
       sum_j (z_OLS,j² - 1)/N,
       bridged with [K(1-K)/z_K^2][K(1-K)/(P(1-P))] -> the Lee fraction of
       total (1/3), NOT the probit residual-scale total (0.5) -- the
       convention difference made explicit.

Run:  conda run -n ltpred python benchmarks/bench_liability_scale.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.liability_scale import (liability_r2_from_z,  # noqa: E402
                                    probit_liability_r2)

SEED = 20260719
N = 20_000
M = 100
P_CAUSAL = 0.20
VXB = 0.5                    # Var(X beta) on the probit residual=1 scale
K_POP = 0.10
REPS = 3
LEARNING = dict(iters=30)


def probit_mle(x, y, iters=25):
    """Probit MLE for y ~ a + b x via Newton-Raphson; returns (b, se_b)."""
    a = float(ndtri(y.mean()))
    b = 0.0
    for _ in range(iters):
        eta = a + b * x
        p = np.clip(ndtr(eta), 1e-10, 1 - 1e-10)
        phi = np.exp(-0.5 * eta * eta) / np.sqrt(2 * np.pi)
        w = phi * phi / (p * (1 - p))
        r = (y - p) / (p * (1 - p)) * phi
        i_aa = w.sum()
        i_ab = (w * x).sum()
        i_bb = (w * x * x).sum()
        g_a = r.sum()
        g_b = (r * x).sum()
        det = i_aa * i_bb - i_ab * i_ab
        if abs(det) < 1e-12:
            break
        da = (i_bb * g_a - i_ab * g_b) / det
        db = (-i_ab * g_a + i_aa * g_b) / det
        a += da
        b += db
        if max(abs(da), abs(db)) < 1e-8:
            break
    eta = a + b * x
    p = np.clip(ndtr(eta), 1e-10, 1 - 1e-10)
    phi = np.exp(-0.5 * eta * eta) / np.sqrt(2 * np.pi)
    w = phi * phi / (p * (1 - p))
    i_bb = (w * x * x).sum() - (w * x).sum() ** 2 / w.sum()
    se_b = 1.0 / np.sqrt(max(i_bb, 1e-12))
    return b, se_b


def probit_mle_joint(X, y, iters=30):
    """Joint probit MLE for y ~ a + X beta via Newton-Raphson."""
    n, m = X.shape
    a = float(ndtri(y.mean()))
    b = np.zeros(m)
    for _ in range(iters):
        eta = a + X @ b
        p = np.clip(ndtr(eta), 1e-10, 1 - 1e-10)
        phi = np.exp(-0.5 * eta * eta) / np.sqrt(2 * np.pi)
        w = phi * phi / (p * (1 - p))
        r = (y - p) / (p * (1 - p)) * phi
        W = w[:, None]
        i_aa = w.sum()
        i_ab = (W * X).sum(0)
        i_bb = (X * W).T @ X
        g_a = r.sum()
        g_b = (r[:, None] * X).sum(0)
        A = np.block([[np.array([[i_aa]]), i_ab[None, :]],
                      [i_ab[:, None], i_bb]])
        g = np.concatenate(([g_a], g_b))
        try:
            step = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        a += step[0]
        b += step[1:]
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def main():
    t0 = time.time()
    print("Probit residual-scale genetic-variance benchmark")
    print(f"seed={SEED}  N={N}  M={M}  causal={P_CAUSAL}  Var(Xb)={VXB}"
          f"  prev={K_POP}  reps={REPS}")
    lee_truth = VXB / (1.0 + VXB)
    print(f"truth: probit residual-scale {VXB}; Lee fraction-of-total "
          f"{lee_truth:.4f}")

    rows = []
    for rep in range(REPS):
        rng = np.random.default_rng(np.random.PCG64(SEED + 23 * rep))
        maf = rng.uniform(0.05, 0.5, M)
        X = rng.binomial(2, maf, size=(N, M)).astype(float)
        beta = np.zeros(M)
        causal = rng.choice(M, int(P_CAUSAL * M), replace=False)
        beta[causal] = rng.standard_normal(causal.size)
        ve = (2 * maf * (1 - maf) * beta * beta).sum()
        beta *= np.sqrt(VXB / ve)
        liab = X @ beta + rng.standard_normal(N)
        t = float(np.quantile(liab, 1.0 - K_POP))
        y = liab > t
        P = y.mean()

        # (a) joint probit identity
        b_joint = probit_mle_joint(X, y)
        r2_joint = float(probit_liability_r2(b_joint, maf, fraction=False).sum())

        # (a') marginal probit identity
        b_marg = np.array([probit_mle(X[:, j], y)[0] for j in range(M)])
        r2_marg = float(probit_liability_r2(b_marg, maf, fraction=False).sum())

        # (b) marginal probit z with the Lee & Wray 2013 factor
        zs = np.array([probit_mle(X[:, j], y)[0]
                       / probit_mle(X[:, j], y)[1] for j in range(M)])
        r2_z = float(
            liability_r2_from_z(zs, N, K_POP, P, subtract_null=True).sum()
        )

        # (c) OLS + Lee 2011 bridge
        rs_ols = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(M)])
        r2_ols = float((rs_ols ** 2).sum())
        r2_lee = float(liability_r2_from_z(
            rs_ols * np.sqrt(N), N, K_POP, P, subtract_null=True
        ).sum())
        rows.append((r2_joint, r2_marg, r2_z, r2_ols, r2_lee))

    rows = np.array(rows)
    names = ["(a) joint probit identity", "(a') marginal probit identity",
             "(b) probit (z²-1)/(N wbar)", "(c) null-adjusted OLS",
             "(c) Lee-2011 bridged"]
    print(f"\n  {'route':28s} {'mean':>8s} {'sd/sqrt(R)':>10s}")
    print(f"  {'truth probit scale':28s} {VXB:8.4f}")
    print(f"  {'truth Lee fraction':28s} {lee_truth:8.4f}")
    for k, name in enumerate(names):
        m = rows[:, k].mean()
        s = rows[:, k].std(ddof=1) / np.sqrt(REPS)
        print(f"  {name:28s} {m:8.4f} ± {s:.4f}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
