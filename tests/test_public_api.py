"""The top-level namespace stays small without breaking explicit imports."""

import pytest


def test_primary_top_level_api_is_curated():
    import ltpred

    assert len(ltpred.__all__) <= 20
    assert "estimate_liability" in ltpred.__all__
    assert "thresholds_from_cip" in ltpred.__all__
    assert "rtmvnorm_gibbs" not in ltpred.__all__
    assert "estimate_liabilities" not in ltpred.__all__
    assert "PopulationScores" not in ltpred.__all__
    assert dir(ltpred) == sorted(ltpred.__all__)


def test_advanced_explicit_top_level_imports_remain_compatible():
    from ltpred import (PairwiseFitResult, PopulationScores, estimate_liabilities,
                        fit_pairwise, fit_variance_components, pa_algorithm,
                        rtmvnorm_gibbs)

    assert PopulationScores.__module__ == "ltpred.pipeline"
    assert fit_pairwise.__module__ == PairwiseFitResult.__module__ == "ltpred.pairwise"
    assert callable(estimate_liabilities)
    assert callable(fit_variance_components)
    assert callable(pa_algorithm)
    assert callable(rtmvnorm_gibbs)


def test_demoted_and_removed_names_are_gone_from_the_top_level():
    import ltpred
    import pytest

    for name in ("construct_covmat", "construct_covmat_sex_limited",
                 "construct_covmat_nurture", "extract_pedigrees",
                 "estimate_liability_pa", "liability_sensitivity",
                 "SensitivityResult", "convert_observed_to_liability_scale",
                 "tnorm_moments", "tnorm_mixture_conditional",
                 "fit_genetic_correlation", "fit_genetic_factor",
                 "fit_nurture", "test_variance_component",
                 "observed_to_liability_gencov", "observed_to_liability_rg"):
        with pytest.raises(AttributeError):
            getattr(ltpred, name)


def test_static_reexports_cover_every_lazily_exported_name():
    # A module-level __getattr__ is a wildcard to a type checker: without the
    # `if TYPE_CHECKING` re-export block in __init__.py every `from ltpred
    # import X` resolves to Any, silently voiding the py.typed promise. The two
    # lists are maintained by hand, so pin them together.
    import ast
    import pathlib

    import ltpred

    source = pathlib.Path(ltpred.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    reexported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom):
                    reexported.update(alias.name for alias in stmt.names)

    lazy = {name for names in ltpred._EXPORTS.values() for name in names}
    assert reexported, "the TYPE_CHECKING re-export block disappeared"
    assert lazy - reexported == set(), (
        "lazily exported but not statically re-exported (these resolve to Any "
        f"for type checkers): {sorted(lazy - reexported)}")
    assert reexported - lazy == set(), (
        f"re-exported but not in _EXPORTS: {sorted(reexported - lazy)}")


def test_py_typed_marker_is_present_and_backed_by_annotations():
    # The marker tells type checkers to trust inline annotations, so the public
    # signatures must actually carry them.
    import inspect
    import pathlib

    import ltpred
    from ltpred import (estimate_liabilities, estimate_liability,
                        fit_heritability, prevalence_thresholds)

    assert (pathlib.Path(ltpred.__file__).parent / "py.typed").is_file()
    for fn in (estimate_liability, estimate_liabilities, fit_heritability,
               prevalence_thresholds):
        # __annotations__ rather than inspect.get_annotations: the latter is
        # 3.10+, and every annotated module uses `from __future__ import
        # annotations`, so these are unevaluated strings on every version.
        hints = fn.__annotations__
        assert "return" in hints, f"{fn.__name__} has no return annotation"
        params = [p for p in inspect.signature(fn).parameters]
        missing = [p for p in params if p not in hints]
        assert not missing, f"{fn.__name__} parameters lack annotations: {missing}"


def test_h2_is_a_required_argument_on_the_score_surface():
    # h2 is as disease-specific as pop_prev, which already refuses a default;
    # a missing h2 must raise TypeError before any covariance work runs.
    import numpy as np
    import pytest

    from ltpred import estimate_liabilities, estimate_liability
    from ltpred.family import Family, Member

    fam = Family("f", [Member("o", -np.inf, 0.0)])
    with pytest.raises(TypeError):
        estimate_liability([fam])
    with pytest.raises(TypeError):
        estimate_liabilities(
            ["o"], [None], [None], probands=["o"], status=[1], age=[45.0],
            use="gwas", cip_ages=np.arange(121.0),
            cip_values=np.full(121, 0.1), k_pop=0.1)


@pytest.mark.parametrize("first", [
    "from ltpred.tetrachoric import tetrachoric_matrix",
    "from ltpred import tetrachoric_matrix",
    "import importlib; importlib.import_module('ltpred.tetrachoric')",
])
def test_package_attribute_tetrachoric_is_the_function_not_the_module(first):
    """`tetrachoric` names both a public function and its module.

    Loading the submodule makes the import system bind it as a package
    attribute, which used to shadow the lazily exported function: after
    `from ltpred.tetrachoric import tetrachoric_matrix`, the documented
    `from ltpred import tetrachoric` returned the module. The package now keeps
    the function bound, while the module stays importable by its dotted name.
    Run in a subprocess so the import order under test does not depend on what
    the rest of the suite happened to import first.
    """
    import subprocess
    import sys

    source = (first + "\n"
              "import sys\n"
              "from ltpred import tetrachoric\n"
              "from ltpred.tetrachoric import TetrachoricResult\n"
              "module = sys.modules['ltpred.tetrachoric']\n"
              "print(type(tetrachoric).__name__, type(module).__name__,\n"
              "      module.tetrachoric is tetrachoric,\n"
              "      TetrachoricResult.__name__)\n")
    out = subprocess.run([sys.executable, "-c", source], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "function module True TetrachoricResult", out
