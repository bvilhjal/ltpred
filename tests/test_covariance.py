"""Relatedness table and covariance-matrix construction."""

import numpy as np
import pytest

from ltpred.covariance import (get_relatedness, construct_covmat_single,
                               construct_covmat_multi,
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


def test_multi_covmat_rejects_incoherent_genetic_environment_split():
    # R_g and R_p are each valid correlation matrices, but together with h2 they
    # imply E = R_p - D R_g D with a negative eigenvalue.
    rg = np.array([[1.0, 0.9], [0.9, 1.0]])
    rp = np.array([[1.0, 0.1], [0.1, 1.0]])
    with pytest.raises(ValueError, match="incoherent"):
        construct_covmat_multi(fam_vec=["m"], h2_vec=[0.8, 0.8],
                               genetic_corrmat=rg, full_corrmat=rp)


@pytest.mark.parametrize(
    "name,matrix,match",
    [
        ("genetic_corrmat", np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        ("full_corrmat", np.array([[0.9, 0.0], [0.0, 1.0]]), "unit diagonal"),
        ("genetic_corrmat", np.array([[1.0, 1.2], [1.2, 1.0]]),
         "positive semi-definite"),
    ],
)
def test_multi_covmat_validates_correlation_matrices(name, matrix, match):
    kwargs = dict(h2_vec=[0.2, 0.2], genetic_corrmat=np.eye(2),
                  full_corrmat=np.eye(2))
    kwargs[name] = matrix
    with pytest.raises(ValueError, match=match):
        construct_covmat_multi(fam_vec=["m"], **kwargs)


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


# --- pedigree / kinship input -------------------------------------------------
from ltpred.covariance import kinship_from_pedigree, construct_covmat_from_kinship


def test_kinship_standard_relationships():
    # proband o, parents m/f, sibs s1/s2, maternal grandparents (founders)
    ids    = ["o", "m", "f", "s1", "s2", "mgm", "mgf"]
    father = ["f", "mgf", None, "f", "f", None, None]
    mother = ["m", "mgm", None, "m", "m", None, None]
    _, A = kinship_from_pedigree(ids, father, mother)
    idx = {p: i for i, p in enumerate(ids)}
    assert A[idx["o"], idx["o"]] == pytest.approx(1.0)          # non-inbred self
    assert A[idx["o"], idx["m"]] == pytest.approx(0.5)          # parent-offspring
    assert A[idx["o"], idx["f"]] == pytest.approx(0.5)
    assert A[idx["s1"], idx["s2"]] == pytest.approx(0.5)        # full sibs
    assert A[idx["o"], idx["mgm"]] == pytest.approx(0.25)       # grandparent
    assert A[idx["m"], idx["f"]] == pytest.approx(0.0)          # unrelated spouses
    assert np.allclose(A, A.T)


def test_kinship_halfsibs_cousins_inbreeding():
    # paternal half-sibs share only the father
    _, A = kinship_from_pedigree(["fa", "m1", "m2", "c1", "c2"],
                                 [None, None, None, "fa", "fa"],
                                 [None, None, None, "m1", "m2"])
    assert A[3, 4] == pytest.approx(0.25)
    # first cousins: children of two full sibs
    ids = ["gm", "gf", "sA", "sB", "pA", "pB", "c1", "c2"]
    fa = [None, None, "gf", "gf", None, None, "pA", "pB"]
    mo = [None, None, "gm", "gm", None, None, "sA", "sB"]
    _, A = kinship_from_pedigree(ids, fa, mo)
    assert A[6, 7] == pytest.approx(0.125)                      # first cousins
    # inbreeding: child of two full sibs has A_ii = 1 + 0.5*0.5 = 1.25
    _, A = kinship_from_pedigree(["gm", "gf", "sA", "sB", "x"],
                                 [None, None, "gf", "gf", "sA"],
                                 [None, None, "gm", "gm", "sB"])
    assert A[4, 4] == pytest.approx(1.25)


def test_covmat_from_kinship_matches_role_grammar():
    h2 = 0.6
    ids    = ["o", "m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"]
    father = ["f", "mgf", "pgf", "f", "f", None, None, None, None]
    mother = ["m", "mgm", "pgm", "m", "m", None, None, None, None]
    _, A = kinship_from_pedigree(ids, father, mother)
    ckin = construct_covmat_from_kinship(A, h2=h2, target=0)
    crole = construct_covmat_single(fam_vec=["m", "f", "s1", "s2", "mgm", "mgf",
                                             "pgm", "pgf"], h2=h2)
    assert np.allclose(ckin.matrix, crole.matrix)              # exact reproduction
    assert ckin.roles[:2] == ["g", "o"]
    assert construct_covmat_from_kinship(A, h2=h2, add_ind=False).matrix.shape == (9, 9)


def test_covmat_from_kinship_scales_inbred_target_genetic_variance():
    ids = ["gm", "gf", "sA", "sB", "x"]
    father = [None, None, "gf", "gf", "sA"]
    mother = [None, None, "gm", "gm", "sB"]
    _, A = kinship_from_pedigree(ids, father, mother)
    target = ids.index("x")
    h2 = 0.8

    cov = construct_covmat_from_kinship(A, h2=h2, target=target).matrix

    raw_genetic_var = h2 * A[target, target]
    raw_liability_var = 1.0 + h2 * (A[target, target] - 1.0)
    expected = raw_genetic_var / raw_liability_var
    assert cov[0, 0] == pytest.approx(expected)
    assert cov[0, 1 + target] == pytest.approx(expected)
    assert np.allclose(np.diag(cov)[1:], 1.0)
    assert np.min(np.linalg.eigvalsh(cov)) > 0.0


def test_kinship_input_validation():
    with pytest.raises(ValueError, match="unique"):
        kinship_from_pedigree(["a", "a"], [None, None], [None, None])
    with pytest.raises(ValueError, match="share length"):
        kinship_from_pedigree(["a", "b"], [None], [None, None])
    with pytest.raises(ValueError, match="own parent"):
        kinship_from_pedigree(["a"], ["a"], [None])
    with pytest.raises(ValueError, match="cycle"):
        kinship_from_pedigree(["a", "b"], ["b", "a"], [None, None])
    _, A = kinship_from_pedigree(["a"], [None], [None])
    with pytest.raises(ValueError, match="h2"):
        construct_covmat_from_kinship(A, h2=1.5)
    with pytest.raises(ValueError, match="target"):
        construct_covmat_from_kinship(A, target=5)


def test_multi_covmat_add_ind_false_same_person_cross_trait_uses_full_corr():
    # add_ind=False strips g/o, so row 0 is a *relative*: its same-person
    # cross-trait covariance is the full correlation, not the genetic one
    # (regression: the code keyed on row index 0 instead of the "g" role label)
    gcorr = np.array([[1.0, 0.4], [0.4, 1.0]])
    fcorr = np.array([[1.0, 0.7], [0.7, 1.0]])
    h2 = np.array([0.5, 0.3])
    cov = construct_covmat_multi(fam_vec=["m", "f"], add_ind=False,
                                 genetic_corrmat=gcorr, full_corrmat=fcorr,
                                 h2_vec=h2)
    assert cov.roles == ["m", "f", "m", "f"]
    k = 2
    for a in range(k):
        assert cov.matrix[a, k + a] == pytest.approx(0.7)
        assert cov.matrix[k + a, a] == pytest.approx(0.7)


def test_multi_covmat_add_ind_true_same_person_cross_trait_by_role():
    # g gets the genetic covariance; o and every relative the full correlation
    gcorr = np.array([[1.0, 0.4], [0.4, 1.0]])
    fcorr = np.array([[1.0, 0.7], [0.7, 1.0]])
    h2 = np.array([0.5, 0.3])
    cov = construct_covmat_multi(fam_vec=["m"], add_ind=True,
                                 genetic_corrmat=gcorr, full_corrmat=fcorr,
                                 h2_vec=h2)
    assert cov.roles == ["g", "o", "m", "g", "o", "m"]
    gcov = 0.4 * np.sqrt(0.5 * 0.3)
    m = cov.matrix
    assert m[0, 3] == pytest.approx(gcov)   # g <-> g: genetic covariance
    assert m[1, 4] == pytest.approx(0.7)    # o <-> o: full correlation
    assert m[2, 5] == pytest.approx(0.7)    # m <-> m: full correlation


def test_zero_h2_is_rejected_with_an_explanation():
    # h2 = 0 makes the `g` row identically zero, so the covariance is singular
    # in a way correct_positive_definite cannot repair. It used to be accepted
    # here and then failed downstream with an opaque "unable to enforce a
    # positive-definite covariance matrix".
    from ltpred.covariance import construct_covmat_from_kinship
    with pytest.raises(ValueError, match=r"h2 must be in \(0, 1\]"):
        construct_covmat_single(fam_vec=["m", "f"], h2=0.0)
    with pytest.raises(ValueError, match=r"all h2 must be in \(0, 1\]"):
        construct_covmat_multi(fam_vec=["m"], h2_vec=[0.5, 0.0],
                               genetic_corrmat=np.eye(2),
                               full_corrmat=np.eye(2))
    with pytest.raises(ValueError, match=r"h2 must be in \(0, 1\]"):
        construct_covmat_from_kinship(np.eye(2), h2=0.0)
    # the open end of the interval is still fine
    construct_covmat_single(fam_vec=["m", "f"], h2=1.0)
    construct_covmat_single(fam_vec=["m", "f"], h2=1e-6)
