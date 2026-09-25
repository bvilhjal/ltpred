"""Identity, uncertainty export and deterministic-optimization contracts."""
import warnings

import numpy as np
import pytest

from ltpred import (Family, Member, estimate_liability, estimate_liabilities,
                    families_from_columns, kinship_from_pedigree)
from ltpred.quadrature import _moments, _moments_array, estimate_liability_quadrature_arrays
from ltpred.simulate import _mendelian_draw, simulate_register_liabilities


@pytest.mark.parametrize('method', ['pa', 'gibbs', 'quadrature'])
def test_prediction_preserves_personal_join_key(method):
    relative = Member('m', 1., np.inf, pid='mother')
    with pytest.raises(ValueError, match='no proband pid'):
        estimate_liability([Family('family', [relative])], h2=.5, method=method)
    family = Family('family', [Member('o', -np.inf, np.inf, pid='target'), relative])
    kwargs = dict(seed=31, n_sim=100, tol=1.) if method == 'gibbs' else {}
    result = estimate_liability([family], h2=.5, method=method, **kwargs)
    assert result.pids.tolist() == ['target']
    assert result.fam_ids.tolist() == ['family']
    # Erasing the own-status interval preserves the relatives-only estimand.
    reference = estimate_liability([Family('family', [Member('m', 1., np.inf)])],
                                   h2=.5, method=method, **kwargs)
    np.testing.assert_array_equal(result.genetic, reference.genetic)
    assert reference.pids.tolist() == ['family']


def test_multitrait_rejects_missing_proband_pid():
    family = Family('family', [Member('m', [1., 1.], [np.inf, np.inf], pid='mother')])
    with pytest.raises(ValueError, match='no proband pid'):
        estimate_liability([family], h2=[.5, .5], genetic_corrmat=np.eye(2),
                           full_corrmat=np.eye(2))


@pytest.mark.parametrize('missing', [None, np.nan, '', 'NA', '<NA>', b'null'])
def test_missing_id_does_not_become_a_proband_key(missing):
    family = Family('family', [Member('o', 1., np.inf, pid=missing)])
    assert estimate_liability([family], h2=.5).pids.tolist() == ['family']


def _register(**updates):
    kwargs = dict(ids=['o', 'm'], father=['0', 'NA'], mother=['m', ''],
                  probands=['o'], status=[1, 0], age=[40., 70.], h2=.5, use='gwas',
                  cip_ages=[0., 100.], cip_values=[0., .1], k_pop=.1)
    kwargs.update(updates)
    return estimate_liabilities(**kwargs)


def test_parent_markers_and_actual_zero_ids():
    result = _register()
    assert result.frac_records_with_unresolved_parents == 0.
    for zero in [0, '0']:
        _, relationship = kinship_from_pedigree([zero, 'child'], [None, zero], [None, None])
        assert relationship[0, 1] == .5
    with pytest.raises(ValueError, match='probands must be unique'):
        _register(probands=['o', 'o'])


def test_pandas_missing_ids_across_entry_points():
    pd = pytest.importorskip('pandas')
    for missing in [pd.NA, pd.NaT]:
        with pytest.raises(ValueError, match='fam_id must not contain missing'):
            families_from_columns([missing], ['o'], [1.], [np.inf])
        with pytest.raises(ValueError, match='ids must not contain missing'):
            kinship_from_pedigree([missing], [None], [None])
        assert _register(father=[missing, missing]).frac_records_with_unresolved_parents == 0.
        result = estimate_liability([Family('family', [Member('o', 1., np.inf, pid=missing)])], h2=.5)
        assert result.pids.tolist() == ['family']


def test_clear_input_errors_and_sampler_warning():
    from ltpred import prevalence_thresholds
    family = Family('f', [Member('o', 1., np.inf)])
    with pytest.raises(ValueError, match='h2 must be numeric'):
        estimate_liability([family], h2=None)
    for method in ['pa', 'gibbs', 'quadrature']:
        with pytest.raises(ValueError, match='at least one family'):
            estimate_liability([], h2=.5, method=method)
    with pytest.raises(ValueError, match='fraction, not a percentage'):
        prevalence_thresholds([True], pop_prev=5)
    with pytest.warns(UserWarning, match='ignores Gibbs controls: n_sim, seed'):
        estimate_liability([family], h2=.5, n_sim=10, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        estimate_liability([family], h2=.5)
    assert not caught


@pytest.mark.parametrize('bounds', [(-np.inf, np.inf), (1.2, np.inf), (-np.inf, -8.),
                                     (9., 10.), (1., 1.), (8., 8.000001),
                                     (-8.000001, -8.), (0., 1e-12)])
def test_compiled_moments_match_scalar_oracle_including_tail_guards(bounds):
    means = np.array([-1e8, -20., -1., 0., 8., 20., 1e8])
    lower, upper = bounds
    expected = np.array([_moments(mean, .75, lower, upper) for mean in means])
    actual = _moments_array(means, .75, lower, upper)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    if hasattr(_moments_array, 'py_func'):
        np.testing.assert_allclose(_moments_array.py_func(means, .75, lower, upper),
                                   expected, rtol=2e-13, atol=2e-14)


def test_quadrature_duplicate_scatter_preserves_order_and_diagnostics(monkeypatch):
    import ltpred.quadrature as module
    lo = np.array([[1., -np.inf], [-np.inf, 1.], [1., -np.inf], [-np.inf, 1.]])
    hi = np.array([[np.inf, 1.], [1., np.inf], [np.inf, 1.], [1., np.inf]])
    expected = [estimate_liability_quadrature_arrays(['o', 's1'], a[None], b[None], .5)
                for a, b in zip(lo, hi)]
    calls = []
    original = module._family
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, '_family', counted)
    actual = estimate_liability_quadrature_arrays(['o', 's1'], lo, hi, .5)
    assert len(calls) == 2
    for name in ['est', 'var', 'error', 'n_nodes']:
        np.testing.assert_array_equal(getattr(actual, name),
                                      np.concatenate([getattr(x, name) for x in expected]))


def test_score_table_exports_copy_arrays_and_retain_repeated_pids():
    families = [Family(i, [Member('o', 1., np.inf, pid='shared')]) for i in [2, 1]]
    result = estimate_liability(families, h2=.5, method='quadrature')
    columns = result.to_dict()
    assert columns['pid'].tolist() == ['shared', 'shared']
    assert columns['fam_id'].tolist() == [2, 1]
    columns['genetic'][:] = -999
    assert np.all(result.genetic > 0)
    assert 'quadrature_error_genetic' in columns
    for value in [result, _register(), estimate_liability_quadrature_arrays(
            ['o'], np.array([[1.]]), np.array([[np.inf]]), .5)]:
        exported = value.to_dict()
        assert any(name == 'var' or name.startswith('var_') for name in exported)
        assert any(name == 'se' or name.startswith('se_') for name in exported)
        pd = pytest.importorskip('pandas')
        table = value.to_frame()
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == list(exported)
        for key, array in exported.items():
            np.testing.assert_array_equal(table[key], array)


def test_mendelian_covariance_is_exact_with_inbreeding_missing_parents_and_shuffled_rows():
    # Two siblings mate; their inbred child then has a child with one unknown parent.
    ids = ['x', 'a', 'grandchild', 'b', 'gm', 'gf']
    father = ['a', 'gf', 'x', 'gf', None, None]
    mother = ['b', 'gm', None, 'gm', None, None]
    _, relationship = kinship_from_pedigree(ids, father, mother)
    factor = np.column_stack([_mendelian_draw(ids, father, mother, e)[0] for e in np.eye(len(ids))])
    _, diagonal = _mendelian_draw(ids, father, mother, np.zeros(len(ids)))
    np.testing.assert_allclose(factor @ factor.T, relationship, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(diagonal, np.diag(relationship))
    assert diagonal[0] == 1.25
    kwargs = dict(ids=ids, father=father, mother=mother, h2=.5,
                  cip_ages=[0., 100.], cip_values=[.001, .1], eval_age=70.)
    result = simulate_register_liabilities(np.random.default_rng(12), method='mendelian', **kwargs)
    replay = simulate_register_liabilities(np.random.default_rng(12), method='mendelian', **kwargs)
    np.testing.assert_array_equal(result.genetic, replay.genetic)
    np.testing.assert_allclose(result.residual_var, .5 / (.5 * diagonal + .5))
    with pytest.raises(ValueError, match='method must be'):
        simulate_register_liabilities(np.random.default_rng(12), method='invalid', **kwargs)


def test_export_trait_names_do_not_collide_with_uncertainty_columns():
    from ltpred import LiabilityResult
    result = LiabilityResult(np.array(['f']), np.array(['p']),
        est={'genetic_x': np.array([1.]), 'genetic_x_se': np.array([2.])},
        se={'genetic_x': np.array([.1]), 'genetic_x_se': np.array([.2])},
        var={'genetic_x': np.array([.3]), 'genetic_x_se': np.array([.4])})
    columns = result.to_dict()
    assert len(columns) == 8
    assert columns['genetic_x_se'][0] == 2.
    assert columns['se_genetic_x'][0] == .1


# --- 2026-09e efficiency-review remediation: bit-identity of the shortcuts ---

def test_kinship_a_from_indices_matches_kinship_from_pedigree():
    from ltpred.covariance import _kinship_A
    from ltpred.pedigree import build_parent_graph, extract_pedigree
    rng = np.random.default_rng(4)
    ids = [f"p{i}" for i in range(90)]
    father = [None if i < 8 or rng.random() < .25 else ids[int(rng.integers(0, i))]
              for i in range(90)]
    mother = [None if i < 8 or rng.random() < .25 else ids[int(rng.integers(0, i))]
              for i in range(90)]
    graph = build_parent_graph(ids, father, mother)
    for proband in ("p50", "p89"):
        ped = extract_pedigree(graph, proband, max_degree=2)
        _, by_ids = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        by_indices = _kinship_A(ped.sire_index, ped.dam_index)
        np.testing.assert_array_equal(by_ids, by_indices)


def test_certified_psd_construct_covmat_is_bit_identical():
    from ltpred.covariance import (construct_covmat_from_kinship,
                                   kinship_from_pedigree, _PSD_CERTIFIED)
    _, A = kinship_from_pedigree(
        ["c", "f", "m", "mgm"], ["f", "mgm", None, None], ["m", None, "mgm", None])
    plain = construct_covmat_from_kinship(A, h2=.5, target=0)
    certified = construct_covmat_from_kinship(A, h2=.5, target=0,
                                              _certified_psd=_PSD_CERTIFIED)
    np.testing.assert_array_equal(plain.matrix, certified.matrix)
    assert plain.roles == certified.roles
    with pytest.raises(ValueError, match="positive semi-definite"):
        construct_covmat_from_kinship([[1., 2.], [2., 1.]], h2=.5)
    with pytest.raises(ValueError, match="positive semi-definite"):
        construct_covmat_from_kinship([[1., 2.], [2., 1.]], h2=.5,
                                      _certified_psd="not the sentinel")


def test_family_free_kinship_scalar_branch_matches_matrix_path():
    from ltpred import estimate_liability_from_kinship
    from ltpred.covariance import (construct_covmat_from_kinship,
                                   correct_positive_definite)
    from ltpred.pearson_aitken import pa_estimate_batched
    rng = np.random.default_rng(2026)
    kinds = (["pin", "control", "uninformative", "interval"] * 40)
    lower, upper = [], []
    for kind in kinds:
        if kind == "pin":
            t = rng.normal(1.8, .4)
            lower.append(t); upper.append(t)
        elif kind == "control":
            lower.append(-np.inf); upper.append(rng.normal(-.1, .3))
        elif kind == "uninformative":
            lower.append(-np.inf); upper.append(np.inf)
        else:
            a, b = np.sort(rng.normal(size=2) * .5)
            lower.append(a); upper.append(b)
    lower, upper = np.array(lower), np.array(upper)
    for h2 in (.5, .05, .99):
        for out in ("genetic", "full"):
            est, _se, var = estimate_liability_from_kinship(
                [[1.]], lower[:, None], upper[:, None], h2=h2, out=out)
            # the literal matrix route: build the 2x2 covariance, prepend g,
            # batch PA -- exactly what the branch must reproduce bit for bit
            cov_obj = construct_covmat_from_kinship([[1.]], h2=h2, target=0)
            cov, _ = correct_positive_definite(cov_obj.matrix)
            g_lo = np.full((len(lower), 1), -np.inf)
            g_hi = np.full((len(lower), 1), np.inf)
            lo = np.concatenate([g_lo, lower[:, None]], axis=1)
            hi = np.concatenate([g_hi, upper[:, None]], axis=1)
            target = 0 if out == "genetic" else 1
            ref_est, ref_var = pa_estimate_batched(cov, lo, hi, target=target)
            np.testing.assert_array_equal(est, ref_est)
            np.testing.assert_array_equal(var, ref_var)


def test_register_cache_size_zero_warns_but_scores_identically():
    kwargs = dict(
        ids=["o", "m", "f"], father=["f", None, None], mother=["m", None, None],
        probands=["o"], status=np.array([1, 0, 0]), age=np.array([45., 70., 68.]),
        use="gwas", cip_ages=np.arange(0., 121.),
        cip_values=.1 / (1 + np.exp((55 - np.arange(0., 121.)) / 8.)),
        k_pop=.1, h2=.5, max_degree=1)
    reference = estimate_liabilities(**kwargs)
    with pytest.warns(UserWarning, match="kinship_cache_size=0"):
        zero = estimate_liabilities(**dict(kwargs, kinship_cache_size=0))
    np.testing.assert_array_equal(zero.est, reference.est)
    np.testing.assert_array_equal(zero.var, reference.var)


def test_covmat_multi_fraction_table_matches_direct_relatedness():
    from ltpred.covariance import construct_covmat_multi, get_relatedness
    rng = np.random.default_rng(8)
    roles = ["m", "f", "s1", "mgm", "mgf", "mau1"]
    h2_vec = np.array([.4, .25])
    p = 2
    X = rng.normal(size=(p, p + 3)); s = np.sqrt(np.diag(X @ X.T))
    gc = (X @ X.T) / np.outer(s, s)
    Y = rng.normal(size=(p, p + 3)); s = np.sqrt(np.diag(Y @ Y.T))
    ec = (Y @ Y.T) / np.outer(s, s)
    D = np.sqrt(np.outer(h2_vec, h2_vec))
    full = gc * D + ec * np.sqrt(np.outer(1 - h2_vec, 1 - h2_vec))
    built = construct_covmat_multi(fam_vec=roles, add_ind=True,
                                   genetic_corrmat=gc, full_corrmat=full,
                                   h2_vec=h2_vec).matrix
    fam_roles = ["g", "o"] + roles
    k = len(fam_roles)
    genetic_cov = gc * D
    expected = np.empty((k * p, k * p))
    for p1 in range(p):
        for p2 in range(p):
            gcov = genetic_cov[p1, p2]
            for a, ra in enumerate(fam_roles):
                for b, rb in enumerate(fam_roles):
                    val = (get_relatedness(ra, rb, h2=h2_vec[p1]) if p1 == p2
                           else get_relatedness(ra, rb, h2=gcov))
                    expected[p1 * k + a, p2 * k + b] = val
            if p1 != p2:
                for a in range(k):
                    expected[p1 * k + a, p2 * k + a] = (gcov if fam_roles[a] == "g"
                                                         else full[p1, p2])
    np.testing.assert_array_equal(built, expected)
