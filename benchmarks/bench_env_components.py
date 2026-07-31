"""Do fitted environment components help estimation? (the C/M wiring benchmark)

`estimate_liability(..., c2=..., m2=...)` wires the sibship (C) and couple (M)
shared-environment components of `fit_variance_components` into the family
covariance (algorithm.md's "sharper genetic estimate" promise). This benchmark
simulates families (proband + parents + two sibs) with true additive genetics
(h2 = 0.4), a true sibship environment (c2 = 0.15) and a true couple
environment (m2 = 0.15), then compares, over 5 replicates of 4,000 families:

  additive-only  : estimate_liability(h2=0.4) -- the misspecified status quo
  oracle-wired   : estimate_liability(h2=0.4, c2=0.15, m2=0.15)
  fitted-wired   : components first fit by fit_variance_components(A, C, M),
                   then wired into estimate_liability

Metrics: corr and calibration slope of the genetic estimate against the true
genetic liability, and corr of the full-liability prediction E[l_o | family]
with the true full liability. The misspecified additive model over-credits
environmental clustering to genetics; wiring should recalibrate the genetic
estimate and sharpen the full-liability prediction.

Run:  conda run -n ltpred python benchmarks/bench_env_components.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.covariance import construct_covmat_single  # noqa: E402
from ltpred.estimate import estimate_liability  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.fit import fit_variance_components  # noqa: E402
from ltpred.thresholds import liability_threshold  # noqa: E402

SEED = 20260719
H2 = 0.4
C2 = 0.15
M2 = 0.15
PREV = 0.10
N_FAM = 4_000
REPS = 5
FAM_VEC = ["m", "f", "s1", "s2"]
FIT_KW = dict(n_iter=600, burn_in=200)


def est_metrics(fams, true_g, true_o, **kw):
    res = estimate_liability(fams, h2=H2, out=("genetic", "full"), **kw)
    g = np.asarray(res.est["genetic"] if hasattr(res, "est") else res.genetic)
    o = np.asarray(res.est["full"] if hasattr(res, "est") else res.full)
    return (float(np.corrcoef(g, true_g)[0, 1]),
            float(np.polyfit(g, true_g, 1)[0]),
            float(np.corrcoef(o, true_o)[0, 1]))


def main():
    t0 = time.time()
    print("Environment-component wiring benchmark (C/M in estimation)")
    print(f"seed={SEED}  h2={H2}  c2={C2}  m2={M2}  prev={PREV}  fams={N_FAM}"
          f"  reps={REPS}")
    cov_obj = construct_covmat_single(fam_vec=FAM_VEC, h2=H2, c2=C2, m2=M2)
    roles = cov_obj.roles                      # g, o, m, f, s1, s2
    cov = cov_obj.matrix
    thr = float(liability_threshold(PREV))

    arms = ("additive-only", "oracle-wired", "fitted-wired")
    out = {a: [] for a in arms}
    for rep in range(REPS):
        rng = np.random.default_rng(np.random.PCG64(SEED + 41 * rep))
        liab = rng.multivariate_normal(np.zeros(len(roles)), cov, size=N_FAM)
        status = {r: liab[:, roles.index(r)] > thr for r in roles if r != "g"}
        fams = []
        for i in range(N_FAM):
            fams.append(Family(fam_id=i, members=[
                Member(role=r, lower=thr if status[r][i] else -np.inf,
                       upper=np.inf if status[r][i] else thr)
                for r in roles if r != "g"]))
        true_g = liab[:, roles.index("g")]
        true_o = liab[:, roles.index("o")]

        out["additive-only"].append(est_metrics(fams, true_g, true_o))
        out["oracle-wired"].append(est_metrics(fams, true_g, true_o,
                                               c2=C2, m2=M2))
        fit = fit_variance_components(
            fams, ("A", "C", "M"), sampling="population", seed=rep, **FIT_KW
        )
        c2_hat = float(fit.components.get("C", 0.0))
        m2_hat = float(fit.components.get("M", 0.0))
        a_hat = float(fit.components["A"])
        out["fitted-wired"].append(
            est_metrics(fams, true_g, true_o, c2=c2_hat, m2=m2_hat))
        if rep == 0:
            print(f"  fitted components (rep 0): A={a_hat:.3f} C={c2_hat:.3f} "
                  f"M={m2_hat:.3f}  (truth {H2}/{C2}/{M2})")

    print(f"\n  {'arm':16s} {'corr(g)':>8s} {'slope(g)':>9s} {'corr(o)':>8s}")
    for a in arms:
        arr = np.array(out[a])
        m = arr.mean(0)
        s = arr.std(0) / np.sqrt(REPS)
        print(f"  {a:16s} {m[0]:8.4f} {m[1]:9.4f} {m[2]:8.4f}"
              f"   (± {s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f})")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
