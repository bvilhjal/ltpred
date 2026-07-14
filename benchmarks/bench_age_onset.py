"""LT-FH++ age-of-onset component over classic LT-FH, inferred with PA.

The liability-threshold model maps a case's age of onset to a threshold: younger
onset implies a more extreme liability. Feeding that map into the estimator --
pinning a case at ``liability = thresh(age_of_onset)`` instead of the plain
``liability > thresh(prevalence)`` -- is the age-of-onset component of
LT-FH++. This benchmark retains the same relatives in both scores and quantifies
the component's gain in estimating the true
genetic liability, fit with the deterministic Pearson-Aitken algorithm (no
mixture).

For a grid of heritability x prevalence it simulates families under the LTM
(true ``g`` known), assigns each case an age of onset through the
liability->onset map, then scores two encodings of the same families:

  * classic LT-FH            -- family statuses at ``(thresh(K_pop), inf)``, age ignored
  * FH + age-of-onset CIP    -- the same family, cases pinned at ``thresh(onset)``

Both are fit with PA over independent replicates; the age-of-onset encoding is
also fit with Gibbs on the first replicate to confirm the two engines agree.
Metrics are corr(estimate, true g), its replicate SE, and the effective-N-style
gain of age-of-onset over classic LT-FH. This is not ADuLT because relatives are
included, and the comparator is not the raw proband case/control label. With
``--followup`` a censoring age can be
imposed (lifetime cases whose onset is later become censored controls).

    python benchmarks/bench_age_onset.py
    python benchmarks/bench_age_onset.py --followup 50
Writes bench_age_onset.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse

import numpy as np

from _common import estimate, get_plt
from ltpred.covariance import construct_covmat_single, correct_positive_definite
from ltpred.thresholds import (liability_threshold, convert_age_to_thresh,
                               convert_liability_to_aoo)
from ltpred.family import Family, Member

HERE = os.path.dirname(os.path.abspath(__file__))


def simulate_onset(fam_vec, h2, pop_prev, n_fam, seed, followup=None,
                   mid_point=60.0, slope=1.0 / 8.0):
    """Simulate families with case ages of onset (true ``g`` known).

    Liabilities come from the family liability covariance; a lifetime case has liability
    above the lifetime threshold, and its age of onset is the deterministic
    liability->onset map. With ``followup`` set, a lifetime case whose onset is
    later is a censored control (upper bound at ``thresh(followup)``); otherwise
    everyone is fully observed. Returns per-member records for building the two
    encodings, plus the true ``g``."""
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles
    non_g = roles[1:]
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal(np.zeros(len(roles)), cov, size=n_fam)

    t_pop = float(liability_threshold(pop_prev))
    thr_follow = t_pop if followup is None else float(
        convert_age_to_thresh(followup, pop_prev=pop_prev, mid_point=mid_point, slope=slope))

    recs = []
    for i in range(n_fam):
        row = []
        for role in non_g:
            liab = L[i, roles.index(role)]
            lifetime_case = liab > t_pop
            aoo = convert_liability_to_aoo(liab, pop_prev=pop_prev, mid_point=mid_point,
                                           slope=slope) if lifetime_case else np.inf
            aoo = float(aoo) if np.isfinite(aoo) else 0.0
            observed_case = lifetime_case and (followup is None or aoo <= followup)
            onset_thr = float(convert_age_to_thresh(round(aoo), pop_prev=pop_prev,
                                                    mid_point=mid_point, slope=slope)) \
                if observed_case else np.nan
            row.append((role, observed_case, onset_thr))
        recs.append(row)
    return dict(recs=recs, true_g=L[:, 0], t_pop=t_pop, thr_follow=thr_follow)


def build_families(raw, encoding):
    """Build families under classic ``ltfh`` or ``age_onset`` encoding.

    Classic LT-FH encodes a case as the one-sided ``(thresh(K_pop), inf)``;
    the age component *pins* the case at its
    onset threshold (``lower == upper == thresh(onset)``), which uses the onset to
    fix the liability rather than merely bound it. Controls are identical across
    encodings, so the two differ only in how cases carry information."""
    fams = []
    for i, row in enumerate(raw["recs"]):
        members = []
        for role, is_case, onset_thr in row:
            if is_case and encoding == "ltfh":
                members.append(Member(role, lower=raw["t_pop"], upper=np.inf))
            elif is_case:                      # age-of-onset: pin at the onset threshold
                members.append(Member(role, lower=onset_thr, upper=onset_thr))
            else:
                members.append(Member(role, lower=-np.inf, upper=raw["thr_follow"]))
        fams.append(Family(fam_id=i, members=members))
    return fams


def run(fam_vec, h2s, prevs, n_fam, followup, n_sim, seed, reps):
    rows = []
    for h2 in h2s:
        for prev in prevs:
            values = {key: [] for key in ("ltfh", "pa", "gain")}
            r_gibbs = agreement = np.nan
            for rep in range(reps):
                rep_seed = seed + rep
                raw = simulate_onset(fam_vec, h2, prev, n_fam, rep_seed,
                                     followup=followup)
                true_g = raw["true_g"]
                ltfh, _ = estimate(build_families(raw, "ltfh"), h2, "pa")
                onset_families = build_families(raw, "age_onset")
                aoo_pa, _ = estimate(onset_families, h2, "pa")
                r_ltfh = float(np.corrcoef(ltfh, true_g)[0, 1])
                r_aoo = float(np.corrcoef(aoo_pa, true_g)[0, 1])
                values["ltfh"].append(r_ltfh)
                values["pa"].append(r_aoo)
                values["gain"].append((r_aoo / r_ltfh) ** 2 if r_ltfh > 0 else np.nan)
                if rep == 0:
                    aoo_gibbs, _ = estimate(onset_families, h2, "gibbs",
                                            n_sim=n_sim, seed=seed)
                    r_gibbs = float(np.corrcoef(aoo_gibbs, true_g)[0, 1])
                    agreement = float(np.corrcoef(aoo_pa, aoo_gibbs)[0, 1])

            mean = {key: float(np.nanmean(value)) for key, value in values.items()}
            se = {}
            for key, value in values.items():
                finite = np.asarray(value, float)
                finite = finite[np.isfinite(finite)]
                se[key] = (float(finite.std(ddof=1) / np.sqrt(finite.size))
                           if finite.size > 1 else np.nan)
            rows.append(dict(h2=h2, prevalence=prev, followup=followup,
                             reps=reps, gibbs_reps=1,
                             corr_ltfh_pa=mean["ltfh"], se_ltfh_pa=se["ltfh"],
                             corr_age_onset_pa=mean["pa"], se_age_onset_pa=se["pa"],
                             corr_age_onset_gibbs_first_rep=r_gibbs,
                             gain=mean["gain"],
                             se_gain=se["gain"], pa_gibbs_agree=agreement))
            print(f"h2={h2:.1f} K={prev:.2f} | corr LT-FH={mean['ltfh']:.3f}"
                  f"±{se['ltfh']:.3f} FH+onset PA={mean['pa']:.3f}±{se['pa']:.3f} "
                  f"| gain={mean['gain']:.3f}±{se['gain']:.3f}x "
                  f"| first-rep Gibbs={r_gibbs:.3f}")
    return rows


def write_csv(rows):
    path = os.path.join(HERE, "bench_age_onset.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def plot(rows):
    plt = get_plt()
    if plt is None:
        return
    h2s = sorted({r["h2"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for h2 in h2s:
        sub = sorted([r for r in rows if r["h2"] == h2], key=lambda r: r["prevalence"])
        p = [r["prevalence"] for r in sub]
        ax[0].errorbar(p, [r["corr_ltfh_pa"] for r in sub],
                       yerr=[r["se_ltfh_pa"] for r in sub], fmt="--o",
                       capsize=3, color=f"C{h2s.index(h2)}", alpha=0.6,
                       label=f"h2={h2} classic LT-FH")
        ax[0].errorbar(p, [r["corr_age_onset_pa"] for r in sub],
                       yerr=[r["se_age_onset_pa"] for r in sub], fmt="-o",
                       capsize=3, color=f"C{h2s.index(h2)}", label=f"h2={h2} FH + onset")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("prevalence")
    ax[0].set_ylabel("corr(estimate, true g)")
    ax[0].set_title("(a) accuracy: classic LT-FH vs FH + onset")
    ax[0].legend(fontsize=7)
    for h2 in h2s:
        sub = sorted([r for r in rows if r["h2"] == h2], key=lambda r: r["prevalence"])
        ax[1].errorbar([r["prevalence"] for r in sub], [r["gain"] for r in sub],
                       yerr=[r["se_gain"] for r in sub], fmt="-o", capsize=3,
                       label=f"h2={h2}")
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("prevalence")
    ax[1].set_ylabel("eff-N proxy: FH + onset / classic LT-FH")
    ax[1].set_title("(b) incremental gain from onset info")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_age_onset.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", nargs="+",
                    default=["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"])
    ap.add_argument("--h2", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--prev", type=float, nargs="+", default=[0.02, 0.05, 0.15, 0.30])
    ap.add_argument("--n-fam", type=int, default=3000)
    ap.add_argument("--followup", type=int, default=None,
                    help="censor controls at this age (default: full follow-up)")
    ap.add_argument("--n-sim", type=int, default=25_000)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent PA cohorts per cell; Gibbs checks the first")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be at least 1")

    rows = run(args.fam, args.h2, args.prev, args.n_fam, args.followup,
               args.n_sim, args.seed, args.reps)
    path = write_csv(rows)
    plot(rows)
    print(f"\nwrote {os.path.basename(path)} and bench_age_onset.png")


if __name__ == "__main__":
    main()
