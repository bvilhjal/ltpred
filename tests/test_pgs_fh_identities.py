"""Numerical verification of the PGS x family-history score identities.

Two scores that are conditionally independent, linear, noisy measurements of
the same standardised genetic liability ``g`` satisfy

    Corr(PGS, FH) = a*b*sqrt(p) = c_g * b
    R²_joint      = (c_g² + b² − 2 c_g b ρ) / (1 − ρ²),   ρ = c_g * b

with ``c_g = Corr(PGS, g) = sqrt(R²_pgs * h²_SNP / h²_total)`` and
``b = Corr(FH, g) = sqrt(R²_fh)`` (docs/algorithm.md, "Combining with a SNP
polygenic score"). The benchmark cross-check of the correlation identity
inside the full GWAS design is ``bench_pgs_comparison.py`` (RESULTS.md §28);
these tests pin the two identities themselves to Monte-Carlo precision.
"""
import numpy as np
import pytest


@pytest.mark.parametrize("h2_ratio, r2_pgs, r2_fh", [
    (0.7, 0.05, 0.15),
    (1.0, 0.10, 0.20),
    (0.5, 0.02, 0.30),
    (0.9, 0.15, 0.05),
])
def test_pgs_fh_identities(h2_ratio, r2_pgs, r2_fh):
    rng = np.random.default_rng(42)
    n = 500_000
    c_g = np.sqrt(r2_pgs * h2_ratio)     # Corr(PGS, g) = a*sqrt(p)
    b = np.sqrt(r2_fh)                   # Corr(FH, g)
    g = rng.normal(size=n)
    pgs = c_g * g + np.sqrt(1.0 - c_g ** 2) * rng.normal(size=n)
    fh = b * g + np.sqrt(1.0 - b ** 2) * rng.normal(size=n)

    rho_theory = c_g * b                 # = a*b*sqrt(p)
    rho_emp = np.corrcoef(pgs, fh)[0, 1]
    assert rho_emp == pytest.approx(rho_theory, abs=4e-3)

    beta = np.linalg.lstsq(np.column_stack([pgs, fh]), g, rcond=None)[0]
    r2_emp = np.corrcoef(np.column_stack([pgs, fh]) @ beta, g)[0, 1] ** 2
    r2_theory = ((c_g ** 2 + b ** 2 - 2 * c_g * b * rho_theory)
                 / (1.0 - rho_theory ** 2))
    assert r2_emp == pytest.approx(r2_theory, abs=4e-3)
