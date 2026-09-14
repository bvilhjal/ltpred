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
    # rows folded from the former kind-by-kind generator: every same-side or
    # side-neutral kind pair at one representative role each (symmetry is
    # test_relatedness_symmetric_exhaustive's job, so one orientation each)
    ("c1.1", "g", 0.5), ("c1.1", "m", 0.25), ("c1.1", "mau1", 0.125),
    ("c1.1", "mgm", 0.125), ("c1.1", "mhs1", 0.125), ("c1.1", "o", 0.5),
    ("c1.1", "s1", 0.25), ("g", "m", 0.5), ("g", "mau1", 0.25),
    ("g", "mgm", 0.25), ("g", "mhs1", 0.25), ("g", "s1", 0.5),
    ("m", "mhs1", 0.5), ("mau1", "mau2", 0.5), ("mau1", "mgm", 0.5),
    ("mau1", "mhs1", 0.25), ("mau1", "s1", 0.25), ("mgm", "mgf", 0.0),
    ("mgm", "s1", 0.25), ("mhs1", "mhs2", 0.5), ("mhs1", "s1", 0.25),
])
def test_relatedness_h2_one(s1, s2, expected):
    # h2=1 gives the bare shared-DNA fraction
    assert get_relatedness(s1, s2, h2=1.0) == pytest.approx(expected)


def test_relatedness_scales_with_h2():
    assert get_relatedness("o", "m", h2=0.4) == pytest.approx(0.2)
    assert get_relatedness("g", "g", h2=0.3) == pytest.approx(0.3)


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


def test_n_fam_preserves_explicit_numbered_roles():
    cov = construct_covmat_single(
        fam_vec=None, n_fam={"s1": 1, "mhs3": 1, "c2.4": 1}, h2=0.5,
    )
    assert cov.roles == ["g", "o", "s1", "mhs3", "c2.4"]
    with pytest.raises(ValueError, match="explicit role 's1'"):
        construct_covmat_single(fam_vec=None, n_fam={"s1": 2}, h2=0.5)


@pytest.mark.parametrize("role,count", [("s", 1.9), ("s1", 0.5),
                                         ("s", np.nan), ("s", np.inf),
                                         ("s", True), ("s", 10 ** 1000)])
def test_n_fam_rejects_non_integer_counts(role, count):
    with pytest.raises(ValueError, match="non-negative integers"):
        construct_covmat_single(fam_vec=None, n_fam={role: count}, h2=0.5)
    # Preserve the convenient, unambiguous case of an integral float.
    cov = construct_covmat_single(fam_vec=None, n_fam={"s": 2.0}, h2=0.5)
    assert cov.roles == ["g", "o", "s1", "s2"]


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


def test_correct_positive_definite_respects_the_limit():
    # correction_limit is the number of correction attempts allowed: a limit of
    # zero rejects an unrepaired matrix instead of correcting it once
    # (review 2026-09, F2)
    bad = np.array([[1.0, 1.0], [1.0, 1.0]])       # eigenvalues 2, 0
    with pytest.raises(ValueError, match="positive-definite"):
        correct_positive_definite(bad, correction_limit=0)
    fixed, n = correct_positive_definite(bad, correction_limit=2)
    assert 0 < n <= 2
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


def test_kinship_environment_kernels_match_role_grammar():
    from ltpred.fit import _component_matrix

    h2, c2, m2 = 0.4, 0.15, 0.10
    ids = ["o", "m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"]
    father = ["f", "mgf", "pgf", "f", "f", None, None, None, None]
    mother = ["m", "mgm", "pgm", "m", "m", None, None, None, None]
    _, A = kinship_from_pedigree(ids, father, mother)
    C = _component_matrix(ids, "C")
    M = _component_matrix(ids, "M")

    kin = construct_covmat_from_kinship(
        A, h2=h2, target=0, c2=c2, c_kernel=C, m2=m2, m_kernel=M)
    role = construct_covmat_single(
        fam_vec=ids[1:], h2=h2, c2=c2, m2=m2)
    np.testing.assert_allclose(kin.matrix, role.matrix, rtol=0, atol=1e-12)


def test_kinship_environment_kernel_validation():
    A = np.eye(2)
    I = np.eye(2)
    with pytest.raises(ValueError, match="required when c2 is nonzero"):
        construct_covmat_from_kinship(A, c2=0.1)
    with pytest.raises(ValueError, match="supplied without c2"):
        construct_covmat_from_kinship(A, c_kernel=I)
    with pytest.raises(ValueError, match="finite nonnegative"):
        construct_covmat_from_kinship(A, c2=np.nan)
    with pytest.raises(ValueError, match="finite nonnegative"):
        construct_covmat_from_kinship(A, c2=-0.1)
    with pytest.raises(ValueError, match="must not exceed 1"):
        construct_covmat_from_kinship(
            A, h2=0.8, c2=0.2, c_kernel=I, m2=0.1, m_kernel=I)
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        construct_covmat_from_kinship(A, c2=0.1, c_kernel=np.eye(3))
    with pytest.raises(ValueError, match="finite values"):
        construct_covmat_from_kinship(
            A, c2=0.1, c_kernel=np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        construct_covmat_from_kinship(
            A, c2=0.1, c_kernel=np.array([[1.0, 0.2], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="unit diagonal"):
        construct_covmat_from_kinship(A, c2=0.1, c_kernel=0.5 * I)
    with pytest.raises(ValueError, match="positive semi-definite"):
        construct_covmat_from_kinship(
            A, c2=0.1, c_kernel=np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_kinship_inputs_canonicalize_accepted_numeric_asymmetry():
    # The public 1e-8 validation tolerance is wider than the PA/Gibbs symmetry
    # tolerance. Accepted numerical noise must therefore be removed here.
    A = np.array([[1.0, 5e-9], [0.0, 1.0]])
    C = np.array([[1.0, 5e-9], [0.0, 1.0]])
    cov = construct_covmat_from_kinship(
        A, h2=0.4, c2=0.2, c_kernel=C).matrix
    np.testing.assert_array_equal(cov, cov.T)


def test_kinship_environment_kernels_preserve_inbred_threshold_scale():
    A = np.array([[1.25, 0.5], [0.5, 1.0]])
    C = np.ones((2, 2))
    h2, c2 = 0.5, 0.2
    cov = construct_covmat_from_kinship(
        A, h2=h2, target=0, c2=c2, c_kernel=C).matrix

    raw_target_var = 1.0 + h2 * (A[0, 0] - 1.0)
    assert np.array_equal(np.diag(cov)[1:], np.ones(2))
    assert cov[0, 0] == pytest.approx(h2 * A[0, 0] / raw_target_var)
    assert cov[0, 1] == pytest.approx(h2 * A[0, 0] / raw_target_var)


def test_kinship_is_independent_of_record_order():
    # The tabular method needs parents placed before their children; the
    # topological sort (Kahn's) supplies that regardless of how the records are
    # listed. Youngest-first listing is the ordering that costs the naive
    # repeated-scan sort one pass per generation, so pin it explicitly.
    ids    = ["o", "m", "f", "s1", "mgm", "mgf", "pgm", "pgf"]
    father = ["f", "mgf", "pgf", "f", None, None, None, None]
    mother = ["m", "mgm", "pgm", "m", None, None, None, None]
    _, A = kinship_from_pedigree(ids, father, mother)

    rng = np.random.default_rng(11)
    for _ in range(20):
        perm = rng.permutation(len(ids))
        p_ids = [ids[i] for i in perm]
        p_fa = [father[i] for i in perm]
        p_mo = [mother[i] for i in perm]
        _, A_p = kinship_from_pedigree(p_ids, p_fa, p_mo)
        # A_p is in permuted order; map it back and require an exact match
        back = np.empty_like(A_p)
        pos = {pid: k for k, pid in enumerate(p_ids)}
        for a, ia in enumerate(ids):
            for b, ib in enumerate(ids):
                back[a, b] = A_p[pos[ia], pos[ib]]
        assert np.array_equal(back, A)


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


def test_n_fam_rejects_singleton_counts_above_one():
    # a count > 1 for a singleton role used to be silently dropped; numbered
    # roles are the way to request multiples
    with pytest.raises(ValueError, match="singleton role"):
        construct_covmat_single(fam_vec=None, n_fam={"m": 2}, h2=0.5)
    with pytest.raises(ValueError, match="singleton role"):
        construct_covmat_single(fam_vec=None, n_fam={"mgm": 3}, h2=0.5)
    cov = construct_covmat_single(fam_vec=None, n_fam={"s": 2}, h2=0.5)
    assert cov.roles == ["g", "o", "s1", "s2"]


def test_same_side_half_sibs_share_a_second_parent_by_convention():
    # the inherited LTFHPlus convention (see the get_relatedness docstring): two
    # same-side half-sibs are related 0.5*h2 *to each other* — an implied shared
    # second parent — while cross-side half-sibs are unrelated
    assert get_relatedness("mhs1", "mhs2", h2=0.4) == pytest.approx(0.5 * 0.4)
    assert get_relatedness("phs1", "phs2", h2=0.4) == pytest.approx(0.5 * 0.4)
    assert get_relatedness("mhs1", "phs1", h2=0.4) == pytest.approx(0.0)


def test_relatedness_self():
    """Every role has self-relatedness 1.0; g has variance h²."""
    assert get_relatedness("g", "g", h2=0.5) == pytest.approx(0.5)
    for role in ["o", "s1", "s2", "c1.1", "c1.2", "c2.1", "m", "f", "mgm", "mgf",
                 "pgm", "pgf", "mhs1", "mhs2", "phs1", "phs2", "mau1", "mau2",
                 "pau1", "pau2"]:
        assert get_relatedness(role, role, h2=0.5) == pytest.approx(1.0)


def test_relatedness_opposite_sides_are_zero():
    """Two side-specific roles on opposite sides are unrelated."""
    maternal = ["m", "mgm", "mgf", "mhs1", "mau1"]
    paternal = ["f", "pgm", "pgf", "phs1", "pau1"]
    for ma in maternal:
        for pa in paternal:
            assert get_relatedness(ma, pa, h2=1.0) == pytest.approx(0.0), \
                f"{ma!r} ↔ {pa!r} should be 0.0"


def test_relatedness_child_groups():
    """Children in same partner group are full sibs; different groups are half sibs."""
    assert get_relatedness("c1.1", "c1.2", h2=1.0) == pytest.approx(0.5)
    assert get_relatedness("c1.1", "c2.1", h2=1.0) == pytest.approx(0.25)
    assert get_relatedness("c1.1", "c1.1", h2=1.0) == pytest.approx(1.0)


def test_relatedness_symmetric_exhaustive():
    """Every role pair is symmetric."""
    roles = [
        "g", "o", "m", "f", "s1", "s2",
        "mgm", "mgf", "pgm", "pgf",
        "mhs1", "phs1", "mau1", "pau1",
        "c1.1", "c1.2", "c2.1",
    ]
    for a in roles:
        for b in roles:
            assert get_relatedness(a, b, 0.4) == get_relatedness(b, a, 0.4), \
                f"{a!r} ↔ {b!r} not symmetric"


# --- shared-environment components on the role grammar (merged from
# --- tests/test_env_components.py)

def test_environment_components_add_to_the_right_pairs():
    m = construct_covmat_single(fam_vec=["m", "f", "s1"], h2=0.4, c2=0.15, m2=0.1)
    r = {role: i for i, role in enumerate(m.roles)}
    cov = m.matrix
    assert cov[r["o"], r["s1"]] == pytest.approx(0.2 + 0.15, abs=1e-12)   # full sibs: h2/2 + c2
    assert cov[r["m"], r["f"]] == pytest.approx(0.1, abs=1e-12)           # mates: m2
    assert cov[r["o"], r["m"]] == pytest.approx(0.2, abs=1e-12)           # parent-offspring unchanged
    non_g = [i for role, i in r.items() if role != "g"]
    np.testing.assert_allclose(np.diag(cov)[non_g], 1.0, rtol=0, atol=1e-12)
    assert cov[r["g"], r["g"]] == pytest.approx(0.4, abs=1e-12)
    # the genetic row is untouched (g shares no environment)
    np.testing.assert_allclose(cov[r["g"], :], [0.4, 0.4, 0.2, 0.2, 0.2], rtol=0, atol=1e-12)


@pytest.mark.parametrize("c2,m2", [(None, None), (0.0, None), (None, 0.0)])
def test_absent_or_zero_environment_components_reproduce_the_base_matrix(c2, m2):
    base = construct_covmat_single(fam_vec=["m", "f", "s1", "mgm"], h2=0.3)
    m = construct_covmat_single(fam_vec=["m", "f", "s1", "mgm"], h2=0.3, c2=c2, m2=m2)
    np.testing.assert_allclose(m.matrix, base.matrix, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kwargs,match", [
    (dict(h2=0.5, c2=-0.1), "nonnegative"),
    (dict(h2=0.5, m2=-0.1), "nonnegative"),
    (dict(h2=0.6, c2=0.3, m2=0.2), "must not exceed 1"),
])
def test_role_grammar_builder_validates_environment_components(kwargs, match):
    # the kinship builder has its own check (test_kinship_environment_kernel_validation)
    with pytest.raises(ValueError, match=match):
        construct_covmat_single(fam_vec=["m"], **kwargs)


def test_c_covers_the_probands_children_sibship():
    # c1.1/c1.2 are full sibs in one partner group; the sibship regex used
    # to match only o/s*, so C silently skipped them and a family described
    # from the children's side got a different C structure than from the
    # parents' side.
    m = construct_covmat_single(fam_vec=["c1.1", "c1.2", "c2.1", "s1"], h2=0.4, c2=0.2)
    r = {role: i for i, role in enumerate(m.roles)}
    cov = m.matrix
    assert cov[r["c1.1"], r["c1.2"]] == pytest.approx(0.2 + 0.2, abs=1e-12)
    assert cov[r["o"], r["s1"]] == pytest.approx(0.2 + 0.2, abs=1e-12)
    # cross-group children are only half sibs -> no C, like mhs/phs
    assert cov[r["c1.1"], r["c2.1"]] == pytest.approx(0.1, abs=1e-12)
    # the kernel stays a disjoint partition, so the matrix stays PD
    assert np.linalg.eigvalsh(cov).min() > 0.0
