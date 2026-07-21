import unittest

import numpy as np

from ltpred.covariance import get_relatedness, kinship_from_pedigree
from ltpred.pedigree import (build_parent_graph, extract_pedigree,
                             extract_pedigrees)

# Toy population: two grandparent couples, their children m and f, proband o
# and full sib s1, f's remarriage (half-sib hs1 with f2), m's brother au and
# his child c1, and o's own child c_o with mate sp.
IDS = ["mgm", "mgf", "pgm", "pgf", "m", "f", "au", "f2", "au_sp", "sp",
       "o", "s1", "hs1", "c1", "c_o"]
FATHER = [None, None, None, None, "mgf", "pgf", "mgf", None, None, None,
          "f", "f", "f", "au", "o"]
MOTHER = [None, None, None, None, "mgm", "pgm", "mgm", None, None, None,
          "m", "m", "f2", "au_sp", "sp"]


def make_graph():
    return build_parent_graph(IDS, FATHER, MOTHER)


class BuildGraphTests(unittest.TestCase):
    def test_rejects_duplicates_and_self_parenting(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_parent_graph(["a", "a"], [None, None], [None, None])
        with self.assertRaisesRegex(ValueError, "own parent"):
            build_parent_graph(["a", "b"], ["a", None], [None, None])
        with self.assertRaisesRegex(ValueError, "share length"):
            build_parent_graph(["a"], [None, None], [None])


class ExtractTests(unittest.TestCase):
    def test_degree_two_membership_and_degrees(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        self.assertEqual(ped.ids[0], "o")
        self.assertEqual(set(ped.ids),
                         {"o", "m", "f", "s1", "hs1", "mgm", "mgf", "pgm",
                          "pgf", "c_o", "sp", "au", "f2"})
        deg = dict(zip(ped.ids, ped.degree))
        self.assertEqual(deg["o"], 0)
        # first degree: parents, children, full siblings
        self.assertEqual((deg["m"], deg["f"], deg["s1"], deg["c_o"]),
                         (1, 1, 1, 1))
        # second degree: half-sibs, grandparents, aunts/uncles, mates via child
        for r in ("hs1", "mgm", "mgf", "pgm", "pgf", "au", "sp"):
            self.assertEqual(deg[r], 2, r)
        # parental closure: hs1's mother enters even just past the limit
        self.assertEqual(deg["f2"], 3)

    def test_deeper_degrees(self):
        ped2 = extract_pedigree(make_graph(), "o", max_degree=2)
        self.assertNotIn("c1", ped2.ids)       # cousin is third degree
        ped3 = extract_pedigree(make_graph(), "o", max_degree=3)
        self.assertIn("c1", ped3.ids)          # first cousin at degree 3
        self.assertIn("f2", ped3.ids)          # father's mate via hs1
        self.assertIn("au_sp", ped3.ids)       # closure parent (degree 4)

    def test_founder_columns_and_ordering(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        idx = {pid: i for i, pid in enumerate(ped.ids)}
        # grandparents' parents are beyond the limit -> founders
        for gp in ("mgm", "mgf", "pgm", "pgf"):
            self.assertIsNone(ped.father[idx[gp]])
            self.assertIsNone(ped.mother[idx[gp]])
        # extracted parents are expressed as ids
        self.assertEqual(ped.father[idx["o"]], "f")
        self.assertEqual(ped.mother[idx["o"]], "m")
        self.assertEqual(ped.mother[idx["hs1"]], "f2" if "f2" in ped.ids else None)
        # deterministic under record shuffling
        order = np.argsort(np.arange(len(IDS)))
        g2 = build_parent_graph([IDS[i] for i in order],
                                [FATHER[i] for i in order],
                                [MOTHER[i] for i in order])
        ped2 = extract_pedigree(g2, "o", max_degree=2)
        self.assertEqual(ped.ids, ped2.ids)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "not in the parent graph"):
            extract_pedigree(make_graph(), "nobody")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            extract_pedigree(make_graph(), "o", max_degree=0)


class KinshipIntegrationTests(unittest.TestCase):
    def test_extracted_kinship_matches_role_grammar(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        _, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        idx = {pid: i for i, pid in enumerate(ped.ids)}
        for role, pid in (("m", "m"), ("f", "f"), ("s1", "s1"),
                          ("mgm", "mgm"), ("mgf", "mgf"),
                          ("pgm", "pgm"), ("pgf", "pgf")):
            self.assertAlmostEqual(A[idx["o"], idx[pid]],
                                   get_relatedness("o", role, h2=1.0),
                                   places=12)
        # half-sib and mate beyond the grammar
        self.assertAlmostEqual(A[idx["o"], idx["hs1"]], 0.25, places=12)
        self.assertAlmostEqual(A[idx["o"], idx["sp"]], 0.0, places=12)

    def test_deep_kinship_values(self):
        ped = extract_pedigree(make_graph(), "o", max_degree=4)
        _, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        idx = {pid: i for i, pid in enumerate(ped.ids)}
        self.assertAlmostEqual(A[idx["o"], idx["au"]], 0.25, places=12)
        self.assertAlmostEqual(A[idx["o"], idx["c1"]], 0.125, places=12)

    def test_extraction_preserves_full_pedigree_kinship(self):
        # the sub-pedigree A equals the full-population A restricted to members
        _, A_full = kinship_from_pedigree(IDS, FATHER, MOTHER)
        ped = extract_pedigree(make_graph(), "o", max_degree=2)
        _, A_sub = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        fi = {pid: i for i, pid in enumerate(IDS)}
        sub_idx_full = [fi[p] for p in ped.ids]
        np.testing.assert_allclose(A_sub, A_full[np.ix_(sub_idx_full, sub_idx_full)],
                                   rtol=0, atol=1e-12)

    def test_degree_is_the_shortest_route_not_the_first_found(self):
        # A is P's paternal grandfather (degree 2) but is also reachable as
        # B's father via the maternal-uncle route; assigning on first visit
        # recorded whichever the traversal hit first, overstating the degree.
        g = build_parent_graph(ids=["P", "F", "M", "B", "A"],
                               father=["F", "A", "B", "A", None],
                               mother=["M", None, None, None, None])
        ped = extract_pedigree(g, "P", max_degree=1)
        degrees = dict(zip(ped.ids, ped.degree.tolist()))
        self.assertEqual(degrees["A"], 2)
        self.assertEqual(max(ped.degree), 2)


    def test_batch_extraction_matches_single(self):
        g = make_graph()
        singles = [extract_pedigree(g, p, max_degree=2) for p in ("o", "s1", "m")]
        batched = list(extract_pedigrees(g, ("o", "s1", "m"), max_degree=2))
        for a, b in zip(singles, batched):
            self.assertEqual(a.ids, b.ids)


if __name__ == "__main__":
    unittest.main()
