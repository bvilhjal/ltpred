import numpy as np
import pytest

from ltpred.covariance import get_relatedness, kinship_from_pedigree
from ltpred.pedigree import build_parent_graph, extract_pedigree

# Two grandparent couples, their children m/f, a proband and full sibling,
# f's remarriage, m's brother and child, and the proband's own child and mate.
IDS = ["mgm", "mgf", "pgm", "pgf", "m", "f", "au", "f2", "au_sp", "sp",
       "o", "s1", "hs1", "c1", "c_o"]
FATHER = [None, None, None, None, "mgf", "pgf", "mgf", None, None, None,
          "f", "f", "f", "au", "o"]
MOTHER = [None, None, None, None, "mgm", "pgm", "mgm", None, None, None,
          "m", "m", "f2", "au_sp", "sp"]


def make_graph():
    return build_parent_graph(IDS, FATHER, MOTHER)


class TestBuildGraph:
    def test_sibling_adjacency_matches_shared_known_parents_in_row_order(self):
        rng = np.random.default_rng(89)
        ids = list(range(60))
        father = [None] * 10 + rng.integers(0, 5, size=50).tolist()
        mother = [None] * 10 + rng.integers(5, 10, size=50).tolist()
        mother[20] = mother[21] = None
        order = rng.permutation(len(ids))
        graph = build_parent_graph([ids[i] for i in order],
                                   [father[i] for i in order],
                                   [mother[i] for i in order])
        for i, siblings in enumerate(graph.sibs):
            expected = [
                j for j in range(len(ids))
                if j != i and graph.sire[i] != -1 and graph.dam[i] != -1
                and graph.sire[j] == graph.sire[i] and graph.dam[j] == graph.dam[i]
            ]
            assert siblings == expected
        assert len({id(siblings) for siblings in graph.sibs}) == len(ids)

    def test_rejects_duplicates_and_self_parenting(self):
        with pytest.raises(ValueError, match="unique"):
            build_parent_graph(["a", "a"], [None, None], [None, None])
        with pytest.raises(ValueError, match="own parent"):
            build_parent_graph(["a", "b"], ["a", None], [None, None])
        with pytest.raises(ValueError, match="share length"):
            build_parent_graph(["a"], [None, None], [None])

    def test_unresolved_nonnull_parents_are_counted(self):
        graph = build_parent_graph(["a", "b", "c"], ["c", "x", None],
                                   [None, float("nan"), None])
        assert graph.n_unresolved_parents == 1
        assert graph.sire == [2, -1, -1]
        assert make_graph().n_unresolved_parents == 0


class TestExtract:
    def test_degree_two_membership_and_degrees(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        assert ped.ids[0] == "o"
        assert set(ped.ids) == {
            "o", "m", "f", "s1", "hs1", "mgm", "mgf", "pgm", "pgf",
            "c_o", "sp", "au", "f2",
        }
        degree = dict(zip(ped.ids, ped.degree))
        assert degree["o"] == 0
        assert (degree["m"], degree["f"], degree["s1"], degree["c_o"]) == (1, 1, 1, 1)
        for role in ("hs1", "mgm", "mgf", "pgm", "pgf", "au", "sp"):
            assert degree[role] == 2, role
        assert degree["f2"] == 3
        closure = dict(zip(ped.ids, ped.closure_only.tolist()))
        assert closure["f2"]
        assert not any(closure[pid] for pid in ped.ids if pid != "f2")

    def test_deeper_degrees(self):
        ped2 = extract_pedigree(make_graph(), "o", max_degree=2)
        assert "c1" not in ped2.ids
        ped3 = extract_pedigree(make_graph(), "o", max_degree=3)
        assert "c1" in ped3.ids
        assert "f2" in ped3.ids
        assert "au_sp" in ped3.ids
        assert ped3.closure_only[ped3.ids.index("au_sp")]

    def test_founder_columns_and_ordering(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        index = {pid: i for i, pid in enumerate(ped.ids)}
        for grandparent in ("mgm", "mgf", "pgm", "pgf"):
            assert ped.father[index[grandparent]] is None
            assert ped.mother[index[grandparent]] is None
        assert ped.father[index["o"]] == "f"
        assert ped.mother[index["o"]] == "m"
        assert ped.mother[index["hs1"]] == "f2"

        order = np.arange(len(IDS))[::-1]
        shuffled = build_parent_graph([IDS[i] for i in order],
                                      [FATHER[i] for i in order],
                                      [MOTHER[i] for i in order])
        ped2 = extract_pedigree(shuffled, "o", max_degree=2)
        assert ped.ids == ped2.ids
        np.testing.assert_array_equal(ped.closure_only, ped2.closure_only)

    def test_validation(self):
        with pytest.raises(ValueError, match="not in the parent graph"):
            extract_pedigree(make_graph(), "nobody")
        with pytest.raises(ValueError, match="at least 1"):
            extract_pedigree(make_graph(), "o", max_degree=0)
        for bad in (2.9, True):
            with pytest.raises(TypeError, match="integer"):
                extract_pedigree(make_graph(), "o", max_degree=bad)
        assert "m" in extract_pedigree(make_graph(), "o", max_degree=np.int64(2)).ids

    def test_degree_is_shortest_route_not_first_found(self):
        graph = build_parent_graph(ids=["P", "F", "M", "B", "A"],
                                   father=["F", "A", "B", "A", None],
                                   mother=["M", None, None, None, None])
        ped = extract_pedigree(graph, "P", max_degree=1)
        degree = dict(zip(ped.ids, ped.degree.tolist()))
        assert degree["A"] == 2
        assert max(ped.degree) == 2
        closure = dict(zip(ped.ids, ped.closure_only.tolist()))
        assert closure["A"] and closure["B"]
        assert not any(closure[pid] for pid in ("P", "F", "M"))


class TestKinshipIntegration:
    def test_extracted_kinship_matches_role_grammar(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        _, relationship = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        index = {pid: i for i, pid in enumerate(ped.ids)}
        for role in ("m", "f", "s1", "mgm", "mgf", "pgm", "pgf"):
            assert relationship[index["o"], index[role]] == pytest.approx(
                get_relatedness("o", role, h2=1.0), abs=1e-12)
        assert relationship[index["o"], index["hs1"]] == pytest.approx(0.25, abs=1e-12)
        assert relationship[index["o"], index["sp"]] == 0.0

    def test_deep_kinship_values(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=4)
        _, relationship = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        index = {pid: i for i, pid in enumerate(ped.ids)}
        assert relationship[index["o"], index["au"]] == pytest.approx(0.25, abs=1e-12)
        assert relationship[index["o"], index["c1"]] == pytest.approx(0.125, abs=1e-12)

    def test_extraction_preserves_full_pedigree_kinship(self):
        _, full = kinship_from_pedigree(IDS, FATHER, MOTHER)
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        _, subset = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        full_index = {pid: i for i, pid in enumerate(IDS)}
        rows = [full_index[pid] for pid in ped.ids]
        np.testing.assert_allclose(subset, full[np.ix_(rows, rows)], rtol=0, atol=1e-12)
