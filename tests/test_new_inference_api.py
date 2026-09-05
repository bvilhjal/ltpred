"""Public adapters preserve scores, output alignment and numerical diagnostics."""
import numpy as np
import pytest
from scipy.stats import truncnorm
from ltpred import (Family, Member, estimate_liability,
                    estimate_liability_from_kinship,
                    estimate_liability_quadrature_arrays, fit_pairwise,
                    QuadratureResult, PairwiseFitResult)


def test_quadrature_object_adapter_preserves_alignment_and_diagnostics():
    families=[Family('a',[Member('o',2,2,pid='x'),Member('m',-np.inf,1.5)]),
              Family('b',[Member('o',-np.inf,1,pid='y')]),
              Family('c',[Member('m',-np.inf,1.5),Member('o',1,1,pid='z')])]
    result=estimate_liability(families,h2=.5,method='quadrature',out=['genetic','full'])
    assert result.fam_ids.tolist()==['a','b','c']
    assert result.pids.tolist()==['x','y','z']
    for name in ('genetic','full'):
        for i,family in enumerate(families):
            ref=estimate_liability_quadrature_arrays(
                [m.role for m in family.members],[[m.lower for m in family.members]],
                [[m.upper for m in family.members]],out=name)
            assert isinstance(ref,QuadratureResult)
            assert result.est[name][i]==pytest.approx(ref.est[0],abs=1e-14)
            assert result.var[name][i]==pytest.approx(ref.var[0],abs=1e-14)
            assert result.quadrature_error[name][i]==ref.error[0]
            assert result.quadrature_nodes[name][i]==ref.n_nodes[0]
        assert not result.se[name].any()


@pytest.mark.parametrize('kwargs,match',[
    ({'h2':[.5,.5]},'single-trait'), ({'use_mixture':True},'without'),
    ({'c2':.1},'without'), ({'m2':.1},'without'),
    ({'quadrature_max_nodes':32},'max_nodes'),
    ({'quadrature_atol':-1},'atol')])
def test_quadrature_dispatch_rejects_unsupported_requests(kwargs,match):
    with pytest.raises((ValueError,NotImplementedError),match=match):
        estimate_liability([Family('f',[Member('o',-np.inf,1)])],method='quadrature',**kwargs)


def test_quadrature_is_not_silently_used_as_gibbs_for_kinship():
    with pytest.raises(NotImplementedError,match='nuclear-family'):
        estimate_liability_from_kinship(np.eye(1),[[-np.inf]],[[1]],method='quadrature')


@pytest.mark.parametrize('h2',[.2,.5,.8])
def test_adult_public_scalar_dispatch_matches_analytic_posterior(h2):
    bounds=[(-np.inf,np.inf),(2.,2.),(-np.inf,1.),(1.,np.inf),(-.5,1.5)]
    families=[Family(str(i),[Member('o',lo,hi)]) for i,(lo,hi) in enumerate(bounds)]
    result=estimate_liability(families,h2=h2,out=['genetic','full'])
    assert result.quadrature_error is None
    for i,(lo,hi) in enumerate(bounds):
        if lo==hi:m,v=lo,0.
        elif lo==-np.inf and hi==np.inf:m,v=0.,1.
        else:m,v=truncnorm.stats(lo,hi,moments='mv')
        assert result.est['genetic'][i]==pytest.approx(h2*m,abs=1e-14)
        assert result.var['genetic'][i]==pytest.approx(h2*(1-h2)+h2*h2*v,abs=1e-14)
        assert result.est['full'][i]==pytest.approx(m,abs=1e-14)
        assert result.var['full'][i]==pytest.approx(v,abs=1e-14)


def test_new_fitter_is_explicitly_exported():
    assert fit_pairwise.__module__=='ltpred.pairwise'
    assert PairwiseFitResult.__module__=='ltpred.pairwise'
