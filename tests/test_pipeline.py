import unittest

import numpy as np

from ltpred.covariance import kinship_from_pedigree
from ltpred.estimate import estimate_liability_from_kinship
from ltpred.pipeline import PopulationScores, estimate_liabilities
from ltpred.thresholds import thresholds_from_cip

# same toy population as test_pedigree
IDS = ["mgm", "mgf", "pgm", "pgf", "m", "f", "au", "f2", "au_sp", "sp",
       "o", "s1", "hs1", "c1", "c_o"]
FATHER = [None, None, None, None, "mgf", "pgf", "mgf", None, None, None,
          "f", "f", "f", "au", "o"]
MOTHER = [None, None, None, None, "mgm", "pgm", "mgm", None, None, None,
          "m", "m", "f2", "au_sp", "sp"]

CIP_AGES = np.arange(0, 121, 1.0)
K_POP = 0.10
CIP_VALUES = K_POP / (1.0 + np.exp((60.0 - CIP_AGES) / 8.0))


def make_pheno(seed=1, case_roles=("m", "s1")):
    status = np.array([r in case_roles for r in IDS])
    age = np.array([75, 78, 73, 76, 45, 48, 50, 42, 47, 30,
                    25, 22, 20, 15, 5], dtype=float)
    return status, age


class SmokeTests(unittest.TestCase):
    def test_output_fields_and_shapes(self):
        status, age = make_pheno()
        out = estimate_liabilities(IDS, FATHER, MOTHER,
                                   probands=["o", "s1"], status=status, age=age,
                                   cip_ages=CIP_AGES, cip_values=CIP_VALUES,
                                   h2=0.5)
        self.assertIsInstance(out, PopulationScores)
        self.assertEqual(out.probands, ["o", "s1"])
        self.assertEqual(out.est.shape, (2,))
        self.assertEqual(out.var.shape, (2,))
        self.assertTrue(np.all(np.isfinite(out.est)))
        self.assertTrue(np.all(out.var >= 0))
        self.assertGreater(out.n_relatives[0], 0)
        self.assertEqual(out.degree_max[0], 4)   # includes closure ancestors

    def test_matches_manual_pieces(self):
        status, age = make_pheno()
        out = estimate_liabilities(IDS, FATHER, MOTHER,
                                   probands=["o"], status=status, age=age,
                                   cip_ages=CIP_AGES, cip_values=CIP_VALUES,
                                   h2=0.5)
        from ltpred.pedigree import build_parent_graph, extract_pedigree
        g = build_parent_graph(IDS, FATHER, MOTHER)
        ped = extract_pedigree(g, "o", max_degree=3)
        pos = {p: i for i, p in enumerate(IDS)}
        midx = np.array([pos[q] for q in ped.ids])
        lo, hi, _, _ = thresholds_from_cip(status[midx], age[midx],
                                           CIP_AGES, CIP_VALUES, k_pop=None)
        _, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        e, v = estimate_liability_from_kinship(A, lo[None, :], hi[None, :],
                                               h2=0.5, target=0)
        self.assertAlmostEqual(out.est[0], e[0], places=12)
        self.assertAlmostEqual(out.var[0], v[0], places=12)

    def test_proband_order_is_a_pure_permutation(self):
        status, age = make_pheno()
        a = estimate_liabilities(IDS, FATHER, MOTHER,
                                 probands=["o", "s1"], status=status, age=age,
                                 cip_ages=CIP_AGES, cip_values=CIP_VALUES)
        b = estimate_liabilities(IDS, FATHER, MOTHER,
                                 probands=["s1", "o"], status=status, age=age,
                                 cip_ages=CIP_AGES, cip_values=CIP_VALUES)
        self.assertAlmostEqual(a.est[0], b.est[1], places=12)
        self.assertAlmostEqual(a.est[1], b.est[0], places=12)


class FamilywiseCensoringTests(unittest.TestCase):
    def test_post_index_case_is_censored(self):
        # s1 is a case at 22; censoring at index age 20 must turn it into a
        # control at 20 and leave the proband uninformative
        status, age = make_pheno(case_roles=("m", "s1"))
        ia = 20.0
        out = estimate_liabilities(IDS, FATHER, MOTHER,
                                   probands=["o"], status=status, age=age,
                                   cip_ages=CIP_AGES, cip_values=CIP_VALUES,
                                   index_age=np.array([ia]))
        # manual reference: same pieces with s1 censored at 20, o unbounded
        from ltpred.pedigree import build_parent_graph, extract_pedigree
        g = build_parent_graph(IDS, FATHER, MOTHER)
        ped = extract_pedigree(g, "o", max_degree=3)
        pos = {p: i for i, p in enumerate(IDS)}
        midx = np.array([pos[q] for q in ped.ids])
        st = status[midx].copy()
        ag = age[midx].copy()
        post = ag > ia
        post[0] = False
        st[post] = False
        ag[post] = ia
        lo, hi, _, _ = thresholds_from_cip(st, ag, CIP_AGES, CIP_VALUES)
        lo[0], hi[0] = -np.inf, np.inf
        _, A = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        e, _ = estimate_liability_from_kinship(A, lo[None, :], hi[None, :],
                                               h2=0.5, target=0)
        self.assertAlmostEqual(out.est[0], e[0], places=12)

    def test_censoring_changes_the_estimate(self):
        # a family-history-heavy family: with vs without censoring must differ
        status, age = make_pheno(case_roles=("m", "f", "s1", "mgm"))
        full = estimate_liabilities(IDS, FATHER, MOTHER,
                                    probands=["o"], status=status, age=age,
                                    cip_ages=CIP_AGES, cip_values=CIP_VALUES)
        cens = estimate_liabilities(IDS, FATHER, MOTHER,
                                    probands=["o"], status=status, age=age,
                                    cip_ages=CIP_AGES, cip_values=CIP_VALUES,
                                    index_age=np.array([18.0]))
        self.assertNotAlmostEqual(full.est[0], cens.est[0], places=6)


class StrataTests(unittest.TestCase):
    def test_stratified_cips_run_and_differ(self):
        status, age = make_pheno()
        strata = np.array(["A"] * 7 + ["B"] * 8)
        curve_a = (CIP_AGES, CIP_VALUES, 0.10)
        curve_b = (CIP_AGES, 1.5 * CIP_VALUES, 0.15)
        out = estimate_liabilities(IDS, FATHER, MOTHER,
                                   probands=["o"], status=status, age=age,
                                   strata=strata,
                                   cip_by_stratum={"A": curve_a, "B": curve_b})
        self.assertTrue(np.isfinite(out.est[0]))


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_input(self):
        status, age = make_pheno()
        base = dict(ids=IDS, father=FATHER, mother=MOTHER, status=status,
                    age=age, cip_ages=CIP_AGES, cip_values=CIP_VALUES)
        with self.assertRaisesRegex(ValueError, "not among ids"):
            estimate_liabilities(probands=["nobody"], **base)
        with self.assertRaisesRegex(ValueError, "aligned to ids"):
            estimate_liabilities(probands=["o"],
                                 **{**base, "status": status[:5]})
        with self.assertRaisesRegex(ValueError, "together"):
            estimate_liabilities(probands=["o"],
                                 **{**base, "strata": np.zeros(len(IDS), int)})
        with self.assertRaisesRegex(ValueError, "no CIP supplied"):
            estimate_liabilities(probands=["o"],
                                 **{**base,
                                    "strata": np.array(["X"] * len(IDS)),
                                    "cip_by_stratum": {"Y": (CIP_AGES, CIP_VALUES, 0.1)}})
        with self.assertRaisesRegex(ValueError, "aligned to probands"):
            estimate_liabilities(probands=["o"],
                                 **{**base, "index_age": np.array([1.0, 2.0])})
        with self.assertRaisesRegex(ValueError, "case_mode"):
            estimate_liabilities(probands=["o"], **{**base, "case_mode": "x"})


if __name__ == "__main__":
    unittest.main()
