"""LT-FH++ vs. a raw case/control label, and Gibbs vs. Pearson-Aitken fitting.

GWAS association power scales with how well the analysed phenotype correlates
with an individual's *true* genetic liability. This demo simulates families under
the liability-threshold model, estimates each proband's posterior mean genetic
liability two ways -- the Gibbs sampler and the deterministic Pearson-Aitken
(PA-FGRS) estimator -- and compares each to the simulated truth and to the plain
0/1 case/control label. The squared ratio of correlations approximates the
effective-sample-size gain from using the estimated liability. It also reports
how closely the two fitting methods agree and their relative speed.

Run (from the repo root): python examples/ltfh_power_demo.py
(installs not required — this inserts the repo root on sys.path; or `pip install -e .`.)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ltpred import simulate_under_LTM_single, estimate_liability  # noqa: E402


def main():
    h2 = 0.5
    pop_prev = 0.05
    n_fam = 2000

    sim = simulate_under_LTM_single(
        fam_vec=["m", "f", "s1"], h2=h2, pop_prev=pop_prev, n_sim=n_fam, seed=1,
    )

    t0 = time.time()
    gibbs = estimate_liability(sim.families, h2=h2, method="gibbs", out=("genetic",),
                               tol=0.03, n_sim=25_000, burn_in=800, seed=0)
    t_gibbs = time.time() - t0

    t0 = time.time()
    pa = estimate_liability(sim.families, h2=h2, method="pearson-aitken",
                            out=("genetic",))
    t_pa = time.time() - t0

    true_g = sim.genetic
    status = sim.status["o"].astype(float)
    r_status = np.corrcoef(status, true_g)[0, 1]
    r_gibbs = np.corrcoef(gibbs.est["genetic"], true_g)[0, 1]
    r_pa = np.corrcoef(pa.est["genetic"], true_g)[0, 1]

    print(f"families                       : {n_fam}")
    print(f"heritability h2                : {h2}")
    print(f"prevalence                     : {pop_prev}")
    print(f"case rate (proband)            : {status.mean():.3f}")
    print()
    print(f"corr(case/control    , true)   : {r_status:.3f}")
    print(f"corr(Gibbs genetic   , true)   : {r_gibbs:.3f}   ({t_gibbs:.2f}s)")
    print(f"corr(PA-FGRS genetic , true)   : {r_pa:.3f}   ({t_pa:.3f}s)")
    print()
    print(f"approx. effective-N gain       : {(r_gibbs / r_status) ** 2:.2f}x")
    print(f"Gibbs vs PA agreement (corr)   : "
          f"{np.corrcoef(gibbs.est['genetic'], pa.est['genetic'])[0, 1]:.4f}")
    print(f"PA speed-up over Gibbs         : {t_gibbs / t_pa:.0f}x")


if __name__ == "__main__":
    main()
