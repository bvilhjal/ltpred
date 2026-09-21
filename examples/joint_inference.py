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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ltpred import fit_pairwise_multi, simulate_under_LTM_multi  # noqa: E402


def example_families(n_families=3000, seed=1):
    """Simulate independent nuclear families with two observed binary traits.

    Thin wrapper over :func:`ltpred.simulate_under_LTM_multi`, which carries the
    generating design (h² = (0.35, 0.40), rg = 0.50, residual re = -0.35, trait
    prevalences 0.10 and 0.20) and returns the simulated liabilities and truth
    alongside the families."""
    return simulate_under_LTM_multi(n_families=n_families, seed=seed).families


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
