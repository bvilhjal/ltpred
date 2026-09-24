"""Every figure quoted in docs/vignette.md comes from examples/vignette.py.

The page states numbers -- correlations, variance decompositions, calibration
rates -- as if a reader could reproduce them by running the script. Nothing
used to check that, and every one of them drifted: they had been produced on a
machine whose LAPACK gave ``rng.multivariate_normal`` a different (equally
valid) sign convention, so the same seed simulated different families. That is
fixed in ``ltpred.simulate`` (see ``_stable_factor``); this module keeps the
page and the script from separating again for any other reason.

Each check names the page's own rounding, and passes when the page's literal is
what the live value rounds to at that precision -- so a real drift fails while a
last-digit difference in the page's formatting does not.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_VIGNETTE = _ROOT / "docs" / "vignette.md"
_SCRIPT = _ROOT / "examples" / "vignette.py"

# Cohorts the page names but the figures dict does not carry.
TETRACHORIC_COHORT = 25_000
N_RISK_REPLICATES = 10


@pytest.fixture(scope="module")
def figures():
    """Run examples/vignette.py once and return the figures it computed."""
    spec = importlib.util.spec_from_file_location("_vignette_example", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        return module.main()


@pytest.fixture(scope="module")
def page():
    return _VIGNETTE.read_text(encoding="utf-8")


# (pattern, [(key, decimals, scale), ...]) -- one capture group per key.
# `scale` divides the live value before comparison, for figures the page writes
# in other units (1.1e-3 is quoted as "1.1 \times 10^{-3}").
CHECKS = [
    # -- 0. heritability and covariance --
    (r"\\rho_\{\\mathrm\{tet\}\}=(-?[\d.]+) \\pm ([\d.]+)\$, so",
     [("tet_rho_small", 3, 1), ("tet_se_small", 3, 1)]),
    (r"\$2\\rho_\{\\mathrm\{tet\}\}=(-?[\d.]+) \\pm ([\d.]+)\$ against a truth",
     [("tet_rho_small", 3, 0.5), ("tet_se_small", 3, 0.5)]),
    (r"returns \$([\d.]+) \\pm ([\d.]+)\$, so \$2\\rho",
     [("tet_rho_big", 3, 1), ("tet_se_big", 3, 1)]),
    (r"so \$2\\rho_\{\\mathrm\{tet\}\}=([\d.]+) \\pm ([\d.]+)\$\.",
     [("tet_rho_big", 3, 0.5), ("tet_se_big", 3, 0.5)]),
    (r"sib tetrachoric on those 25,000 families is \$([\d.]+) \\pm ([\d.]+)\$",
     [("tet_sib_rho_big", 3, 1), ("tet_sib_se_big", 3, 1)]),
    (r"observed_to_liability_h2\(0\.20, pop_prev=0\.05\) # -> ([\d.]+)",
     [("lee", 3, 1)]),
    (r"returns\s+\$\\hat h\^2=([\d.]+)\$ against a truth", [("fit_h2", 3, 1)]),
    (r"Monte-Carlo standard error of \$([\d.]+)\$", [("fit_h2_se", 3, 1)]),
    # -- 4. estimate, and the "did it work?" checks --
    (r"\$\\mathrm\{Var\}\(\\hat\\mu\)=([\d.]+)\$ plus a mean posterior\n"
     r"variance of \$([\d.]+)\$ gives \$([\d.]+)\$",
     [("var_mu", 3, 1), ("mean_post_var", 3, 1), ("lotv_sum", 3, 1)]),
    (r"has mean\n\$(-?[\d.]+)\$ and standard deviation \$([\d.]+)\$; "
     r"cases average \$\+([\d.]+)\$ and\ncontrols \$(-[\d.]+)\$",
     [("mu_mean", 3, 1), ("mu_sd", 3, 1), ("cases_mean", 2, 1),
      ("ctrls_mean", 2, 1)]),
    # -- 5. risk calibration (10 replicates of 4,000) --
    (r"predicted rate is\n\$([\d.]+) \\pm ([\d.]+)\$ against an observed "
     r"\$([\d.]+) \\pm ([\d.]+)\$ \(a gap of\n([\d.]+) standard errors\)",
     [("risk_pred_all", 4, 1), ("risk_pred_all_se", 4, 1),
      ("risk_obs_all", 4, 1), ("risk_obs_all_se", 4, 1),
      ("risk_gap_all_se_units", 1, 1)]),
    (r"top decile is \$([\d.]+) \\pm ([\d.]+)\$ predicted\nagainst "
     r"\$([\d.]+) \\pm ([\d.]+)\$ observed \(([\d.]+) SE\)",
     [("risk_pred_top", 4, 1), ("risk_pred_top_se", 4, 1),
      ("risk_obs_top", 4, 1), ("risk_obs_top_se", 4, 1),
      ("risk_gap_top_se_units", 1, 1)]),
    (r"bottom decile is\n\$([\d.]+)\$ predicted against \$([\d.]+) \\pm ([\d.]+)\$ "
     r"observed \(([\d.]+) SE\)",
     [("risk_pred_bot", 4, 1), ("risk_obs_bot", 4, 1),
      ("risk_obs_bot_se", 4, 1), ("risk_gap_bot_se_units", 1, 1)]),
    # -- what the run produced --
    (r"np\.corrcoef\(status, true_g\)\[0, 1\] +# ([\d.]+)", [("corr_status", 3, 1)]),
    (r"np\.corrcoef\(pa\.genetic, true_g\)\[0, 1\] +# ([\d.]+)", [("corr_pa", 3, 1)]),
    (r"\| Proband 0/1 status \(baseline\) \| ([\d.]+) \|", [("corr_status", 3, 1)]),
    (r"\| Own status only \(no ages; ADuLT's degenerate case\) \| ([\d.]+) \|",
     [("corr_adult", 3, 1)]),
    (r"\| Relatives only, uninformative `o` \| ([\d.]+) \|", [("corr_rel", 3, 1)]),
    (r"\| Classic LT-FH via PA \(`o` plus `m`, `f`, `s1`\) \| ([\d.]+) \|",
     [("corr_pa", 3, 1)]),
    (r"squared-correlation gain of \$([\d.]+)\\times\$", [("eff_n", 2, 1)]),
    (r"\\hat\{\\mu\}_i=\+([\d.]+)\$ with posterior\nvariance \$([\d.]+)\$, "
     r"an unaffected one \$(-[\d.]+)\$ with variance \$([\d.]+)\$",
     [("case_mu", 3, 1), ("case_var", 3, 1), ("ctrl_mu", 3, 1),
      ("ctrl_var", 3, 1)]),
    (r"\$\\mathrm\{Corr\}=([\d.]+)\$, at a median Gibbs Monte-Carlo SE of "
     r"\$([\d.]+)\$", [("corr_pa_gibbs", 4, 1), ("gibbs_se_median", 4, 1)]),
    (r"agree to \$([\d.]+)\\times10\^\{-3\}\$", [("kinship_max_diff", 1, 1e-3)]),
    # -- the age-censored cohort --
    (r"from ([\d.]+) to ([\d.]+), so the block uses \*\*20,000\*\* families to "
     r"leave (\d+)\nobserved cases",
     [("case_rate", 3, 1), ("age_case_rate", 4, 1), ("age_cases", 0, 1)]),
    (r"\$\\mathrm\{Corr\}=([\d.]+)\$ there is \*\*not\*\* comparable with the "
     r"([\d.]+) above", [("age_corr_aware", 3, 1), ("corr_pa", 3, 1)]),
    (r"reaches ([\d.]+); classic one-\$K\$ LT-FH", [("age_corr_label", 3, 1)]),
    (r"already reaches ([\d.]+); the age-aware encoding, given\nthe simulator's "
     r"true incidence curve, then reaches ([\d.]+)\.",
     [("age_corr_classic", 3, 1), ("age_corr_aware", 3, 1)]),
    (r"\(\$([\d.]+)\\times\$ on the squared-correlation proxy\) and the age term "
     r"adds\n\$([\d.]+)\\times\$", [("age_fh_gain", 2, 1), ("age_gain", 2, 1)]),
]


def _quoted_matches(page, pattern, keys, figures):
    match = re.search(pattern, page)
    assert match is not None, f"docs/vignette.md no longer contains {pattern!r}"
    assert len(match.groups()) == len(keys), pattern
    for text, (key, decimals, scale) in zip(match.groups(), keys):
        live = figures[key] / scale
        quoted = float(text)
        # The page's literal must be what `live` rounds to at `decimals`.
        tolerance = 0.5 * 10.0 ** -decimals + 1e-9
        assert abs(live - quoted) <= tolerance, (
            f"docs/vignette.md quotes {key} as {text}; examples/vignette.py "
            f"now gives {live!r}. Re-run the script and update the page."
        )


@pytest.mark.parametrize("pattern,keys", CHECKS, ids=[c[0][:40] for c in CHECKS])
def test_quoted_figure_matches_the_script(page, figures, pattern, keys):
    _quoted_matches(page, pattern, keys, figures)


def test_cohort_sizes_on_the_page_match_the_script(page, figures):
    """The page names the cohort behind each block; the script defines them.

    A figure is only reproducible if the reader knows which cohort produced it,
    so the sizes are part of the claim, not decoration.
    """
    plain = page.replace("**", "")
    assert f"h2, K, n_fam = {figures['h2']}, {figures['K']}, {figures['n_fam']}" in plain
    assert f"$n={figures['n_fam']}$ nuclear" in plain
    assert f"{figures['age_n_fam']:,} families" in plain          # age-censored block
    assert f"{TETRACHORIC_COHORT:,} families" in plain            # convergence check
    assert f"{N_RISK_REPLICATES} replicates of 4,000" in plain    # use-I calibration



_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def test_register_route_counts_on_the_page_match_the_script(page, figures):
    """The six-person register example's counts, which the page spells out."""
    plain = " ".join(page.replace("**", "").split())
    assert (f"Prediction has {_WORDS[figures['reg_relatives']]} relatives, "
            f"{_WORDS[figures['reg_closure_only']]} closure-only ancestors and "
            f"{_WORDS[figures['reg_conditioned_prediction']]} conditioned records"
            ) in plain
    assert (f"GWAS has {_WORDS[figures['reg_conditioned_gwas']]} conditioned "
            "records") in plain


def test_law_of_total_variance_holds_in_the_script(figures):
    """The identity the page tells readers to assert."""
    assert figures["lotv_sum"] == pytest.approx(figures["h2"], abs=0.02)
    assert figures["max_post_var"] <= figures["h2"] + 1e-8
    assert abs(figures["mu_mean"]) < 0.05
    assert figures["cases_mean"] > figures["ctrls_mean"]
