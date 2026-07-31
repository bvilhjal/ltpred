"""Pedigree inference from trio records: extraction fidelity and payoff.

`ltpred.pedigree` discovers a proband's relatives from population
parent-offspring records (Pedersen et al. 2025's graph-based extraction).
This benchmark simulates a 3-generation population (founder couples,
children, remarriages, cousins), then checks:

  Part 1 (fidelity): for random probands, the kinship computed on the
         EXTRACTED sub-pedigree equals the full-population kinship
         restricted to the same members -- extraction does not perturb A.
  Part 2 (payoff):   for each proband, simulate liabilities on the extracted
         pedigree and estimate the genetic liability two ways -- the kinship
         path on ALL extracted relatives (up to third degree: cousins) vs the
         fixed named-role grammar subset it can encode (parents, full siblings,
         grandparents). The LT-FGRS (Pedersen et al. 2026) claim is
         that which relatives you include matters; this measures it.
  Part 3 (scale):    extraction timing for thousands of probands.

Run:  conda run -n ltpred python benchmarks/bench_pedigree_inference.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ltpred.covariance import (construct_covmat_from_kinship,  # noqa: E402
                               kinship_from_pedigree)
from ltpred.estimate import estimate_liability, estimate_liability_from_kinship  # noqa: E402
from ltpred.family import Family, Member  # noqa: E402
from ltpred.pedigree import build_parent_graph, extract_pedigree  # noqa: E402
from ltpred.thresholds import liability_threshold  # noqa: E402

SEED = 20260719
H2 = 0.5
PREV = 0.10
MAX_DEGREE = 3
N_CHECK = 300
N_EST = 300


def simulate_population(rng, n_founder_pairs=150, gens=3, remarry=0.10):
    """3-generation population with remarriages (half-sibs) and cousins."""
    ids, father, mother = [], [], []

    def add(f, m):
        pid = f"p{len(ids)}"
        ids.append(pid)
        father.append(f)
        mother.append(m)
        return pid

    couples = [(add(None, None), add(None, None)) for _ in range(n_founder_pairs)]
    prev_children = []
    for g in range(gens):
        if g > 0:
            pool = list(prev_children)
            rng.shuffle(pool)
            couples = []
            i = 0
            while i + 1 < len(pool):
                a, b = pool[i], pool[i + 1]
                i += 2
                # avoid mating recorded siblings (same recorded parent)
                fa, ma = father[ids.index(a)], mother[ids.index(a)]
                fb, mb = father[ids.index(b)], mother[ids.index(b)]
                if fa is not None and (fa in (fb, mb) or ma in (fb, mb)):
                    continue
                couples.append((a, b))
        next_children = []
        for fa, mo in couples:
            for _ in range(int(rng.integers(2, 5))):
                next_children.append(add(fa, mo))
            if rng.uniform() < remarry:           # second union -> half-sibs
                mate = add(None, None)
                next_children.append(add(fa, mate))
        prev_children = next_children
    return ids, father, mother


def main():
    t0 = time.time()
    print("Pedigree-inference benchmark (trio records -> relatives)")
    rng = np.random.default_rng(np.random.PCG64(SEED))
    ids, father, mother = simulate_population(rng)
    print(f"population: {len(ids)} persons")
    graph = build_parent_graph(ids, father, mother)

    # ---- Part 1: extraction fidelity -----------------------------------------
    _, A_full = kinship_from_pedigree(ids, father, mother)
    fi = {p: i for i, p in enumerate(ids)}
    check_ids = [ids[i] for i in rng.choice(len(ids), size=N_CHECK, replace=False)]
    max_abs = 0.0
    for p in check_ids:
        ped = extract_pedigree(graph, p, max_degree=MAX_DEGREE)
        _, A_sub = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        sub_full = [fi[q] for q in ped.ids]
        max_abs = max(max_abs,
                      float(np.max(np.abs(A_sub - A_full[np.ix_(sub_full, sub_full)]))))
    print(f"Part 1: extracted vs full-pedigree kinship, max abs diff over "
          f"{N_CHECK} probands: {max_abs:.2e}")

    # ---- Part 2: all relatives vs the fixed named-role grammar subset ----------
    thr = float(liability_threshold(PREV))
    est_ids = [ids[i] for i in rng.choice(len(ids), size=N_EST, replace=False)]
    truths, ests_all, ests_role = [], [], []
    for p in est_ids:
        ped = extract_pedigree(graph, p, max_degree=MAX_DEGREE)
        _, A_sub = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        cov = construct_covmat_from_kinship(A_sub, h2=H2).matrix
        liab = rng.multivariate_normal(np.zeros(cov.shape[0]), cov)
        # bounds: classic case/control on every member's full liability
        status = liab[1:] > thr
        lower = np.where(status, thr, -np.inf)
        upper = np.where(status, np.inf, thr)
        est_all, _ = estimate_liability_from_kinship(A_sub, lower[None, :],
                                                     upper[None, :], h2=H2)
        truths.append(liab[0])
        ests_all.append(est_all[0])

        # role arm: the fixed parent/sibling/grandparent names the grammar encodes
        p_idx = fi[p]
        pm, pf = mother[p_idx], father[p_idx]

        def parents_of(pid):
            j = fi[pid]
            return father[j], mother[j]

        role_of = {}
        if pm:
            role_of[pm] = "m"
        if pf:
            role_of[pf] = "f"
        nsib = 0
        for pid in ped.ids[1:]:
            j = fi[pid]
            if (father[j] is not None and father[j] == father[p_idx]
                    and mother[j] is not None and mother[j] == mother[p_idx]):
                nsib += 1
                role_of[pid] = f"s{nsib}"
        if pm:
            fa, mo = parents_of(pm)
            if mo:
                role_of.setdefault(mo, "mgm")
            if fa:
                role_of.setdefault(fa, "mgf")
        if pf:
            fa, mo = parents_of(pf)
            if mo:
                role_of.setdefault(mo, "pgm")
            if fa:
                role_of.setdefault(fa, "pgf")

        rel_bounds = dict(zip(ped.ids, zip(lower, upper)))
        members = [Member(role="o", lower=rel_bounds[p][0],
                          upper=rel_bounds[p][1])]
        for pid, role in role_of.items():
            if pid in rel_bounds:
                lo, hi = rel_bounds[pid]
                members.append(Member(role=role, lower=lo, upper=hi))
        est_role = estimate_liability([Family(fam_id=0, members=members)], h2=H2)
        est_role = np.asarray(est_role.est["genetic"]
                              if hasattr(est_role, "est") else est_role.genetic)
        ests_role.append(est_role[0])

    truths = np.array(truths)
    ests_all = np.array(ests_all)
    ests_role = np.array(ests_role)
    c_all = np.corrcoef(truths, ests_all)[0, 1]
    c_role = np.corrcoef(truths, ests_role)[0, 1]
    c_between = np.corrcoef(ests_all, ests_role)[0, 1]
    print(f"Part 2 (N={N_EST}, h2={H2}, prev={PREV}, degree<={MAX_DEGREE}):")
    print(f"  corr(est, true g)  all relatives: {c_all:.4f}   "
          f"named-role subset: {c_role:.4f}   ratio {c_all / c_role:.3f}")
    print(f"  corr(all-relatives est, named-role estimate): {c_between:.4f}")

    # ---- Part 3: extraction timing -------------------------------------------
    t1 = time.time()
    for p in ids[:3000]:
        extract_pedigree(graph, p, max_degree=MAX_DEGREE)
    print(f"Part 3: 3000 extractions at degree {MAX_DEGREE}: "
          f"{time.time() - t1:.2f}s")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
