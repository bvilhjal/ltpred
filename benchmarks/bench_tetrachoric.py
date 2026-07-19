"""Tetrachoric correlations on families: the Falconer relationships.

`ltpred.tetrachoric` estimates the latent liability correlation from 2x2
case/control tables. Under the liability-threshold model with heritability
h2, the expected tetrachoric correlation between relatives with additive
relationship A is h2 * A -- the classic Falconer route to heritability from
binary data (first-degree relatives give h2 / 2). This benchmark simulates
families under the model (h2 = 0.5, prevalence 0.1) and checks, over 5
replicates of 20,000 families:

  1. Pairwise tetrachoric correlations from STATUSES recover h2 * A for
     proband-parent, proband-sib, sib-sib, proband-grandmother,
     proband-uncle, and the mate pair (expected 0).
  2. They agree with the Pearson correlation computed on the LATENT
     liabilities themselves (the estimand the binary table approximates).
  3. The Falconer heritability estimate h2 ~ 2 * tetrachoric(first-degree)
     recovers the true h2, and agrees with the package's fit_heritability.

Run:  conda run -n ltpred python benchmarks/bench_tetrachoric.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.fit import fit_heritability  # noqa: E402
from ltpred.simulate import simulate_under_LTM_single  # noqa: E402
from ltpred.tetrachoric import tetrachoric, tetrachoric_matrix  # noqa: E402
from ltpred.thresholds import liability_threshold  # noqa: E402

SEED = 20260719
H2 = 0.5
PREV = 0.10
N_FAM = 20_000
REPS = 5
FAM_VEC = ["m", "f", "s1", "mgm", "mau1"]

PAIRS = [("o", "m", 0.25), ("o", "f", 0.25), ("o", "s1", 0.25),
         ("m", "s1", 0.25), ("o", "mgm", 0.125), ("o", "mau1", 0.125),
         ("m", "f", 0.0)]


def main():
    t0 = time.time()
    print("Tetrachoric benchmark (Falconer relationships, h2*A)")
    print(f"seed={SEED}  h2={H2}  prev={PREV}  fams={N_FAM}  reps={REPS}")
    thr = float(liability_threshold(PREV))

    est = {(a, b): [] for a, b, _ in PAIRS}
    latent_corr = {(a, b): [] for a, b, _ in PAIRS}
    h2_falconer, h2_fit = [], []
    for rep in range(REPS):
        rng = np.random.default_rng(np.random.PCG64(SEED + 13 * rep))
        sim = simulate_under_LTM_single(fam_vec=FAM_VEC, h2=H2, n_sim=N_FAM,
                                        pop_prev=PREV, seed=rng)
        roles = sim.roles
        L = sim.liabilities
        status = {r: L[:, roles.index(r)] > thr for r in roles if r != "g"}
        for a, b, _ in PAIRS:
            est[(a, b)].append(tetrachoric(status[a], status[b]).rho)
            la = L[:, roles.index(a)] if a != "o" else L[:, roles.index("o")]
            lb = L[:, roles.index(b)]
            latent_corr[(a, b)].append(np.corrcoef(la, lb)[0, 1])
        fd = [est[p][-1] for p in (("o", "m"), ("o", "f"), ("o", "s1"))]
        h2_falconer.append(2.0 * np.mean(fd))
        h2_fit.append(fit_heritability(sim.families[:4000], n_iter=600,
                                       burn_in=200, seed=rep).h2)

    print(f"\n  {'pair':10s} {'expected h2*A':>12s} {'tetrachoric':>12s} "
          f"{'latent corr':>12s}")
    for a, b, e in PAIRS:
        t = np.mean(est[(a, b)])
        tse = np.std(est[(a, b)]) / np.sqrt(REPS)
        l = np.mean(latent_corr[(a, b)])
        print(f"  {a + '-' + b:10s} {e:12.3f} {t:9.4f} ± {tse:.4f} {l:12.4f}")

    print(f"\n  Falconer h2 = 2 x tetrachoric(first-degree): "
          f"{np.mean(h2_falconer):.4f} ± {np.std(h2_falconer) / np.sqrt(REPS):.4f}"
          f"  (truth {H2})")
    print(f"  fit_heritability on the same families:      "
          f"{np.mean(h2_fit):.4f} ± {np.std(h2_fit) / np.sqrt(REPS):.4f}")

    # matrix form on the first-degree block
    rng = np.random.default_rng(np.random.PCG64(SEED + 99))
    sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=H2,
                                    n_sim=N_FAM, pop_prev=PREV, seed=rng)
    roles = sim.roles
    X = np.column_stack([sim.liabilities[:, roles.index(r)] > thr
                         for r in ("o", "m", "f", "s1")])
    R = tetrachoric_matrix(X)
    print("\n  tetrachoric_matrix on [o, m, f, s1] (expected h2*A block):")
    print("  " + np.array2string(R, precision=3, suppress_small=True))
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
