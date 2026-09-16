"""Vignette step 0: joint h², genetic and residual environmental correlation.

Run ``python examples/joint_inference.py`` from the repository root. This
separate two-trait cohort leaves the single-trait vignette's numbers unchanged.
On real data, replace the simulation with independent family records prepared
as in docs/data-preparation.md. One simulation illustrates the API, not its
calibration; replicated evidence is in benchmarks/RESULTS.md, section 33.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ltpred import (  # noqa: E402
    families_from_columns, fit_pairwise_multi, kinship_from_pedigree,
    prevalence_thresholds,
)


def example_families(n_families=3000, seed=1):
    """Simulate independent nuclear families with two observed binary traits."""
    roles = ["o", "m", "f", "s1", "s2"]
    _, a = kinship_from_pedigree(
        roles, father=["f", None, None, "f", "f"],
        mother=["m", None, None, "m", "m"],
    )
    c, m = np.eye(5), np.eye(5)
    c[np.ix_([0, 3, 4], [0, 3, 4])] = 1.  # full sibship
    m[1, 2] = m[2, 1] = 1.               # couple

    # Trait covariances: h² = (0.35, 0.40), rg = 0.50, residual re = -0.35.
    g = np.diag([.35, .40])
    g[0, 1] = g[1, 0] = .50 * np.sqrt(.35 * .40)
    s = np.array([[.15, .075], [.075, .15]])
    t = np.array([[.10, .02], [.02, .10]])
    e = np.diag(1. - np.diag(g + s + t))
    e[0, 1] = e[1, 0] = -.35 * np.sqrt(e[0, 0] * e[1, 1])
    sigma = (np.kron(g, a) + np.kron(s, c) + np.kron(t, m)
             + np.kron(e, np.eye(5)))
    latent = np.random.default_rng(seed).standard_normal((n_families, 10))
    latent = latent @ np.linalg.cholesky(sigma).T
    # Covariance coordinates are trait-major; table rows are people.
    latent = latent.reshape(n_families, 2, 5).transpose(0, 2, 1).reshape(-1, 2)
    prevalence = [.10, .20]
    status = latent > -ndtri(prevalence)
    bounds = [prevalence_thresholds(status[:, p], pop_prev=k)
              for p, k in enumerate(prevalence)]
    lower = np.column_stack([lo for lo, hi in bounds])
    upper = np.column_stack([hi for lo, hi in bounds])
    return families_from_columns(
        fam_id=np.repeat(np.arange(n_families), 5),
        role=np.tile(roles, n_families), lower=lower, upper=upper,
        pid=np.arange(5 * n_families),
    )


def main():
    joint = fit_pairwise_multi(
        example_families(), components=("A", "C", "M"),
        sampling="population", phen_names=["trait_1", "trait_2"],
    )
    print("Truth: h2 = [0.35, 0.40], rg = 0.50, residual re = -0.35")
    print("h2:", joint.h2)
    print("rg:", joint.rg[0, 1], "residual re:", joint.re[0, 1])
    print("Shared sibship correlation:", joint.correlations["C"][0, 1])
    print("Shared couple correlation:", joint.correlations["M"][0, 1])
    print("Inference status:", joint.inference_status)
    print("Sampling SE(h2):", joint.se["h2"])
    print("Sampling SE(rg):", joint.se["rg"][0, 1])
    print("Sampling SE(re):", joint.se["re"][0, 1])
    print("SEs condition on prevalence and weights; boundary fits withhold them.")
    print("re describes residual E; shared C/M are separate components.")
    print("Multi-trait liability scoring currently supports A+E only.")


if __name__ == "__main__":
    main()
