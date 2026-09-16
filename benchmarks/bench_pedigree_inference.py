"""Pedigree inference from trio records: extraction fidelity and payoff.

`ltpred.pedigree` discovers a proband's relatives from population
parent-offspring records (Pedersen et al. 2025's graph-based extraction).
This benchmark simulates a 3-generation population (founder couples,
children, remarriages, cousins), then checks:

  Part 1 (fidelity): for random probands, the kinship computed on the
         EXTRACTED sub-pedigree equals the full-population kinship
         restricted to the same members -- extraction does not perturb A.
         Single run.
  Part 2 (payoff):   over `--reps` independent simulated populations, for
         each proband simulate liabilities on the exact extracted pedigree
         and estimate the genetic liability two ways -- the kinship path with
         observations from relatives reached within degree 3 (including
         cousins) vs the fixed named-role grammar subset it can encode
         (parents, full siblings, grandparents). Ancestors added solely for
         exact kinship remain in A but have uninformative bounds. The LT-FGRS
         (Pedersen et al., LTFGRS R package) claim that which relatives you
         include matters is assessed with per-replicate values,
         across-replicate SEs (sd/sqrt(R)), and a paired contrast with a
         t-based 95% CI (section 15 convention).
  Part 3 (scale):    extraction timing for thousands of probands. Single run.

Writes `benchmarks/bench_pedigree_inference.csv` in long format
(rep, metric, value; rep 0 marks the single-run parts).

Run:  conda run -n ltpred python benchmarks/bench_pedigree_inference.py
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
from scipy import stats

from _common import mean_se

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
N_FOUNDER_PAIRS = 150
REPS = 5
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "bench_pedigree_inference.csv")


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


def payoff_replicate(rng, n_est, n_founder_pairs):
    """One independent population: corr(est, true g) two ways, paired.

    Returns (corr_degree_bounded, corr_named_role, corr_between_arms)."""
    ids, father, mother = simulate_population(rng, n_founder_pairs=n_founder_pairs)
    graph = build_parent_graph(ids, father, mother)
    thr = float(liability_threshold(PREV))
    fi = {p: i for i, p in enumerate(ids)}
    est_ids = [ids[i] for i in rng.choice(len(ids), size=min(n_est, len(ids)),
                                          replace=False)]
    truths, ests_degree, ests_role = [], [], []
    for p in est_ids:
        ped = extract_pedigree(graph, p, max_degree=MAX_DEGREE)
        _, A_sub = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        cov = construct_covmat_from_kinship(A_sub, h2=H2).matrix
        liab = rng.multivariate_normal(np.zeros(cov.shape[0]), cov)
        # Case/control bounds enter only for members reached by the requested
        # traversal. Closure-only ancestors remain in A so kinship is exact,
        # but their diagnoses are outside this observation set.
        status = liab[1:] > thr
        lower = np.where(status, thr, -np.inf)
        upper = np.where(status, np.inf, thr)
        lower[ped.closure_only] = -np.inf
        upper[ped.closure_only] = np.inf
        est_degree, _, _ = estimate_liability_from_kinship(
            A_sub, lower[None, :], upper[None, :], h2=H2)
        truths.append(liab[0])
        ests_degree.append(est_degree[0])

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
        est_role = np.asarray(est_role.est["genetic"])
        ests_role.append(est_role[0])

    truths = np.array(truths)
    ests_degree = np.array(ests_degree)
    ests_role = np.array(ests_role)
    return (float(np.corrcoef(truths, ests_degree)[0, 1]),
            float(np.corrcoef(truths, ests_role)[0, 1]),
            float(np.corrcoef(ests_degree, ests_role)[0, 1]))


def paired_summary(deltas):
    """Paired contrast: mean, SE, and t-based 95% CI half-width (§15 style)."""
    d = np.asarray(deltas, dtype=float)
    mean, se = mean_se(d)
    ci = (float(stats.t.ppf(0.975, d.size - 1) * se) if d.size >= 2
          else float("nan"))
    return mean, se, ci


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pedigree extraction fidelity and the arbitrary-kinship "
                    "payoff over the fixed named-role grammar.")
    parser.add_argument("--reps", type=int, default=REPS,
                        help="independent populations for Part 2 (payoff)")
    parser.add_argument("--n-check", type=int, default=N_CHECK,
                        help="probands for the Part 1 kinship check")
    parser.add_argument("--n-est", type=int, default=N_EST,
                        help="probands per replicate in Part 2")
    parser.add_argument("--n-founder-pairs", type=int, default=N_FOUNDER_PAIRS,
                        help="founder couples per simulated population")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default=CSV_PATH,
                        help="CSV artifact path (long format: rep, metric, value)")
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f"Pedigree-inference benchmark (trio records -> relatives), "
          f"reps={args.reps}")
    rng = np.random.default_rng(np.random.PCG64(args.seed))
    ids, father, mother = simulate_population(
        rng, n_founder_pairs=args.n_founder_pairs)
    print(f"population: {len(ids)} persons")
    graph = build_parent_graph(ids, father, mother)
    csv_rows = []

    # ---- Part 1: extraction fidelity (single run) -------------------------
    _, A_full = kinship_from_pedigree(ids, father, mother)
    fi = {p: i for i, p in enumerate(ids)}
    check_ids = [ids[i] for i in rng.choice(len(ids), size=min(args.n_check, len(ids)),
                                            replace=False)]
    max_abs = 0.0
    for p in check_ids:
        ped = extract_pedigree(graph, p, max_degree=MAX_DEGREE)
        _, A_sub = kinship_from_pedigree(ped.ids, ped.father, ped.mother)
        sub_full = [fi[q] for q in ped.ids]
        max_abs = max(max_abs,
                      float(np.max(np.abs(A_sub - A_full[np.ix_(sub_full, sub_full)]))))
    print(f"Part 1: extracted vs full-pedigree kinship, max abs diff over "
          f"{len(check_ids)} probands: {max_abs:.2e}")
    csv_rows.append((0, "part1_max_abs_kinship_diff", max_abs))

    # ---- Part 2: degree-bounded observations vs named-role subset ---------
    print(f"Part 2: payoff over {args.reps} independent populations "
          f"(N={args.n_est} probands each, h2={H2}, prev={PREV}, "
          f"degree<={MAX_DEGREE})")
    c_degree_r, c_role_r, c_between_r = [], [], []
    for rep in range(args.reps):
        rrng = np.random.default_rng(np.random.PCG64(args.seed + 1000 + rep))
        c_degree, c_role, c_between = payoff_replicate(
            rrng, args.n_est, args.n_founder_pairs)
        c_degree_r.append(c_degree)
        c_role_r.append(c_role)
        c_between_r.append(c_between)
        csv_rows.extend([
            (rep + 1, "part2_corr_degree_bounded", c_degree),
            (rep + 1, "part2_corr_named_role", c_role),
            (rep + 1, "part2_corr_between_arms", c_between),
            (rep + 1, "part2_delta_corr_degree_minus_role", c_degree - c_role),
        ])
        print(f"  rep {rep + 1}: degree-bounded {c_degree:.4f}  "
              f"role {c_role:.4f}  delta {c_degree - c_role:+.4f}  "
              f"corr(arms) {c_between:.4f}")
    m_degree, s_degree = mean_se(c_degree_r)
    m_role, s_role = mean_se(c_role_r)
    m_bet, s_bet = mean_se(c_between_r)
    d_mean, d_se, d_ci = paired_summary(
        np.array(c_degree_r) - np.array(c_role_r))
    print("  means (± across-replicate SE):")
    print(f"  corr(est, true g)  degree-bounded: {m_degree:.4f} ± "
          f"{s_degree:.4f}   named-role subset: {m_role:.4f} ± "
          f"{s_role:.4f}   ratio {m_degree / m_role:.3f}")
    print(f"  paired degree-minus-role: {d_mean:+.4f} ± {d_se:.4f} SE; "
          f"t-based 95% CI [{d_mean - d_ci:+.4f}, {d_mean + d_ci:+.4f}]")
    print(f"  corr(degree-bounded est, named-role estimate): "
          f"{m_bet:.4f} ± {s_bet:.4f}")

    # ---- Part 3: extraction timing (single run) ---------------------------
    t1 = time.time()
    for p in ids[:3000]:
        extract_pedigree(graph, p, max_degree=MAX_DEGREE)
    dt3 = time.time() - t1
    print(f"Part 3 (single run): 3000 extractions at degree {MAX_DEGREE}: "
          f"{dt3:.2f}s")
    csv_rows.append((0, "part3_seconds_3000_extractions", dt3))

    # ---- CSV artifact ------------------------------------------------------
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rep", "metric", "value"])
        writer.writerows(csv_rows)
    print(f"wrote {args.output} ({len(csv_rows)} rows)")
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
