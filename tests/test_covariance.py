"""Relatedness table and covariance-matrix construction."""

import numpy as np
import pytest

from ltpred.covariance import (get_relatedness, construct_covmat_single,
                               construct_covmat_multi, construct_covmat,
                               correct_positive_definite)


@pytest.mark.parametrize("s1,s2,expected", [
    ("g", "g", 1.0), ("g", "o", 1.0), ("o", "o", 1.0),
    ("o", "m", 0.5), ("o", "f", 0.5), ("o", "s1", 0.5),
    ("g", "s2", 0.5),
    ("o", "mgm", 0.25), ("o", "pgf", 0.25),
    ("o", "mhs1", 0.25), ("o", "mau1", 0.25),
    ("m", "f", 0.0), ("m", "pgm", 0.0), ("mgm", "pgf", 0.0),
    ("m", "s1", 0.5), ("m", "mgm", 0.5), ("m", "mau1", 0.5),
    ("s1", "s2", 0.5), ("mgm", "mhs1", 0.25),
])
def test_relatedness_h2_one(s1, s2, expected):
    # h2=1 gives the bare shared-DNA fraction
    assert get_relatedness(s1, s2, h2=1.0) == pytest.approx(expected)


def test_relatedness_scales_with_h2():
    assert get_relatedness("o", "m", h2=0.4) == pytest.approx(0.2)
    assert get_relatedness("g", "g", h2=0.3) == pytest.approx(0.3)


def test_relatedness_is_symmetric():
    roles = ["g", "o", "m", "f", "s1", "mgm", "mgf", "pgm", "pgf",
             "mhs1", "phs1", "mau1", "pau1"]
    for a in roles:
        for b in roles:
            assert get_relatedness(a, b, 0.5) == get_relatedness(b, a, 0.5)


def test_single_covmat_structure():
    cov = construct_covmat_single(fam_vec=["m", "f", "s1"], h2=0.5)
    assert cov.roles == ["g", "o", "m", "f", "s1"]
    m = cov.matrix
    assert np.allclose(m, m.T)
    assert m[0, 0] == pytest.approx(0.5)   # var(g) = h2
    assert m[1, 1] == pytest.approx(1.0)   # var(o) = 1
    assert m[0, 1] == pytest.approx(0.5)   # cov(g, o) = h2
    assert m[2, 3] == pytest.approx(0.0)   # parents unrelated
    assert np.min(np.linalg.eigvalsh(m)) > 0  # PD


def test_single_covmat_default_is_pd():
    cov = construct_covmat_single(h2=0.5)  # m, f, s1, 4 grandparents
    assert np.min(np.linalg.eigvalsh(cov.matrix)) > 0


def test_n_fam_expands_multiples():
    cov = construct_covmat_single(fam_vec=None,
                                  n_fam={"m": 1, "s": 2}, h2=0.5)
    assert cov.roles == ["g", "o", "m", "s1", "s2"]


def test_add_ind_false_omits_g_o():
    cov = construct_covmat_single(fam_vec=["m", "f"], add_ind=False, h2=0.5)
    assert cov.roles == ["m", "f"]


def test_multi_covmat_blocks():
    gcorr = np.array([[1.0, 0.4], [0.4, 1.0]])
    fcorr = np.array([[1.0, 0.5], [0.5, 1.0]])
    h2 = np.array([0.5, 0.3])
    cov = construct_covmat_multi(fam_vec=["m"], genetic_corrmat=gcorr,
                                 full_corrmat=fcorr, h2_vec=h2,
                                 phen_names=["A", "B"])
    # roles per trait: g, o, m -> k=3, phenotype-major
    assert cov.roles == ["g", "o", "m", "g", "o", "m"]
    m = cov.matrix
    assert m.shape == (6, 6)
    assert np.allclose(m, m.T)
    # same-trait diagonal blocks
    assert m[0, 0] == pytest.approx(0.5)   # var g, trait A
    assert m[3, 3] == pytest.approx(0.3)   # var g, trait B
    # cross-trait, same individual: g<->g = genetic covariance
    gcov = 0.4 * np.sqrt(0.5 * 0.3)
    assert m[0, 3] == pytest.approx(gcov)
    # cross-trait, same individual: o<->o = full correlation
    assert m[1, 4] == pytest.approx(0.5)


def test_construct_covmat_dispatch():
    single = construct_covmat(fam_vec=["m"], h2=0.5)
    assert single.phen_names is None
    multi = construct_covmat(fam_vec=["m"], h2=[0.5, 0.5],
                             genetic_corrmat=np.eye(2), full_corrmat=np.eye(2))
    assert multi.phen_names is not None


def test_correct_positive_definite_leaves_pd_alone():
    cov = construct_covmat_single(fam_vec=["m", "f"], h2=0.5).matrix
    fixed, n = correct_positive_definite(cov)
    assert n == 0
    assert np.array_equal(fixed, cov)


def test_correct_positive_definite_fixes_non_pd():
    bad = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]])
    assert np.min(np.linalg.eigvalsh(bad)) < 0
    fixed, n = correct_positive_definite(bad)
    assert n > 0
    assert np.min(np.linalg.eigvalsh(fixed)) >= 0
    assert np.allclose(np.diag(fixed), np.diag(bad))  # diagonal preserved


def test_invalid_role_raises():
    with pytest.raises(ValueError):
        get_relatedness("o", "zzz")


@pytest.mark.parametrize("bad", ["s1abc", "c1x2", "zzz", "go", "m1", "s1.2"])
def test_malformed_roles_rejected(bad):
    # fullmatch validation rejects trailing junk instead of silently accepting it
    with pytest.raises(ValueError):
        get_relatedness("o", bad)


@pytest.mark.parametrize("ok", ["g", "o", "m", "f", "s", "s1", "mgm", "c1.1", "mhs2", "pau3"])
def test_wellformed_roles_accepted(ok):
    assert not np.isnan(get_relatedness("g", ok, h2=1.0))


def test_children_same_vs_different_partner_group():
    # same partner group -> full sibs (0.5); different group -> half sibs (0.25)
    assert get_relatedness("c1.1", "c1.2", h2=1.0) == pytest.approx(0.5)
    assert get_relatedness("c1.1", "c2.1", h2=1.0) == pytest.approx(0.25)
    assert get_relatedness("c1.1", "c1.1", h2=1.0) == pytest.approx(1.0)  # self
    # a child is a half-relative (0.25*h2) of the proband's siblings
    assert get_relatedness("c1.1", "s1", h2=1.0) == pytest.approx(0.25)


def test_correct_positive_definite_is_strict():
    # a positive-*semi*definite matrix (min eig 0) is pushed strictly PD
    psd = np.array([[1.0, 1.0], [1.0, 1.0]])       # eigenvalues 2, 0
    fixed, n = correct_positive_definite(psd)
    assert n > 0
    assert np.min(np.linalg.eigvalsh(fixed)) > 0
