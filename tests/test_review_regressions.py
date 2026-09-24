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
