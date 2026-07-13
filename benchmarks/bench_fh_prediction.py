"""LT-FH / LT-FH++ genetic-liability phenotype vs case/control — heritability,
age-of-onset (CIP over age), and prevalence change over birth cohorts.

The GWAS *phenotype* is the posterior-mean genetic liability
`E[g | own status + family history (+ age of onset + birth cohort)]`, scored against
the plain **case/control** label, using **population** cumulative-incidence (CIP)
thresholds so it stays valid under case ascertainment (Pedersen 2022/2023).

Age- and cohort-consistent generative model. A liability `ℓ` is fixed; the threshold
`T(age; birth_year) = Φ⁻¹(1 − CIP(age; birth_year))` falls with age as risk accrues
and shifts with **birth cohort** (secular change in prevalence). So each person has
an onset age `a* = T⁻¹(ℓ)` under *their own cohort's* CIP. **Death is a competing
risk**: each relative has an age at death `D` (other-cause mortality), and is observed
only up to `c = min(D, age now)`; the proband is a living participant. **Observed
case** iff `a* ≤ c` (pinned at `T(a*) ≈ ℓ`), else **censored control** at `(−∞, T(c))`
— a relative who dies disease-free is a control censored at their death age, not a
phantom centenarian. `CIP(age; by) = K(by) / (1 + exp((mid − age)/slope))`, with the
cohort lifetime prevalence `K(by) = K · R^((by − 1965)/30)` (a secular ratio `R` per
30 years). Ages are generationally consistent (proband 40–70 born ≈1950–1980, parents
≈29–31 y older, grandparents ≈56–58 y); with mortality, grandparents are observed to
death (~80, ~98 % deceased) with their correct ≈1908 birth cohort.

Two threshold policies are compared throughout:
  * **cohort-aware** — the LT-FH++/ADuLT personalised threshold
    `Tᵢ = Φ⁻¹(1 − K(ageᵢ ; birth_yearᵢ))`, each person anchored to *their own* birth
    cohort's lifetime prevalence `K(by) = K·R^((by−1965)/30)`;
  * **single-K** — the classical-LTM / original-LT-FH baseline: **one** lifetime
    prevalence `K` (the 1965 reference) for *everyone*, i.e. the same CIP anchor
    regardless of birth cohort (cohort-blind; sex is not modelled either).
They coincide when there is no secular trend (`R=1`).

Predictors (all use the population CIP): `cc` (own case/control), `count`
(# observed-case relatives), `LT-FH` (binary, single `K`), `LT-FH++` (age of onset,
**cohort-aware** thresholds), and `LT-FH++(single-K)` (age of onset but single-K
thresholds — the cohort-blind mis-specification).

Sweeps: (a) **ascertainment** (family-history vs age-of-onset gains); (b)
**heritability** (accuracy scales with h²); (c) **secular prevalence trend R** —
cohort-aware vs single-K thresholds. The birth-cohort correction helps **both**
calibration and ranking: ignoring the trend biases the liability estimate (panel (c)
reports this bias, up to −0.07), *and* it loses ranking/power whenever cases span a
range of birth cohorts — two cases with the same age of onset but different cohorts
have different true liabilities (the earlier-born, lower-prevalence one is more
extreme), which a single-K analysis collapses. On this pedigree the ranking gain is
small (the high-weight proband is a living participant spanning ~1950–1980, and the
wide-cohort grandparents are low-relatedness controls), but for cases drawn across
1900–2000 at R=4 the own-onset estimate reaches corr 0.50 (cohort-aware) vs 0.38
(single-K), a ~1.7× effective-N gain — which is why LT-FH++ personalises thresholds
by birth year. Panel (d) isolates that ranking gain: cases across a widening birth-
cohort span, each estimated from its own age of onset, cohort-aware vs single-K. The
CIP shape (`--mid`, `--slope`) is configurable; GWAS λ_GC is not tested here.

    python benchmarks/bench_fh_prediction.py
    python benchmarks/bench_fh_prediction.py --trends 1 2 4 --h2s 0.3 0.6
Writes bench_fh_prediction.csv (+ .png if matplotlib is present).
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np

from _common import estimate, get_plt
from ltpred.covariance import construct_covmat_single, correct_positive_definite
from ltpred.thresholds import liability_threshold, convert_age_to_thresh, convert_liability_to_aoo
from ltpred.family import Family, Member

HERE = os.path.dirname(os.path.abspath(__file__))

STRUCT = ["m", "f", "s1", "s2", "mgm", "mgf", "pgm", "pgf"]     # 3-generation pedigree
CAL_NOW = 2020.0                                                # reference calendar year
BY_REF = 1965.0                                                 # reference birth cohort for K
_AGE_GAP = {"o": 0, "s": 0, "mhs": 0, "phs": 0, "m": 29, "f": 31, "mau": 29,
            "pau": 31, "mgm": 56, "mgf": 58, "pgm": 56, "pgf": 58, "c": -29}


def _draw_family(non_g, proband_age, n, rng, death_mean, death_sd):
    """Birth years and observation (censoring) ages, with **death as a competing risk**.

    The proband is a living participant (age ~ U(``proband_age``)); each relative gets
    a generationally consistent birth year and an independent age at death
    ``D ~ N(death_mean, death_sd)`` (other-cause mortality). A relative is observed up
    to ``min(D, age now)`` — the deceased up to their death age, the living up to the
    present — so a relative who dies disease-free is a control censored at death, not a
    phantom centenarian. Returns ``(birth_years, observation_ages)``, each ``(n, m)``."""
    pa = rng.integers(proband_age[0], proband_age[1] + 1, size=n).astype(float)
    proband_by = CAL_NOW - pa
    BY = np.empty((n, len(non_g)))
    OA = np.empty((n, len(non_g)))
    for j, r in enumerate(non_g):
        stem = r.rstrip("0123456789")
        gap = _AGE_GAP.get(stem, 0)
        sd = 0.0 if stem == "o" else (7.0 if gap == 0 else 5.0)
        by = proband_by - gap + rng.normal(0.0, sd, n)         # older relatives born earlier
        age_now = CAL_NOW - by                                 # biological age if alive
        if stem == "o":
            obs = age_now                                      # enrolled living participant
        else:
            death = np.clip(rng.normal(death_mean, death_sd, n), 1, 110)
            obs = np.minimum(death, age_now)                   # competing risk censors at death
        BY[:, j] = by
        OA[:, j] = np.clip(np.round(obs), 1, 110)
    return BY, OA


def _cohort_K(birth_year, K, trend_R):
    """Lifetime prevalence by birth cohort: ``K`` at 1965, scaled ``R`` per 30 years."""
    return np.clip(K * trend_R ** ((birth_year - BY_REF) / 30.0), 1e-4, 0.9)


def _thr(age, prev, mid, slope):
    return np.asarray(convert_age_to_thresh(age, dist="logistic", pop_prev=prev,
                                            mid_point=mid, slope=slope), dtype=float)


def simulate_cohort(fam_vec, h2, K, n_fam, seed, case_frac=None, proband_age=(40, 70),
                    trend_R=1.0, mid=60.0, slope=1.0 / 8.0, death_mean=80.0, death_sd=12.0):
    """Age-, cohort-, and mortality-consistent cohort; ``case_frac`` oversamples
    observed case probands (ascertainment). Death is a **competing risk**: a relative
    is observed only up to ``min(age at death, age now)``, so a disease-free death is a
    censored control at the death age. Returns true ``g``, proband observed status, the
    observed-case relative count, and three family encodings: binary (single ``K``),
    age-of-onset **cohort-aware** (per-person ``K_i``), and age-of-onset single-``K``."""
    cov_obj = construct_covmat_single(fam_vec=fam_vec, add_ind=True, h2=h2)
    cov, _ = correct_positive_definite(cov_obj.matrix)
    roles = cov_obj.roles
    non_g = roles[1:]
    o_pos = non_g.index("o")
    mem_cols = [roles.index(r) for r in non_g]
    T_K = float(liability_threshold(K))
    rng = np.random.default_rng(seed)

    def block(n):
        L = rng.multivariate_normal(np.zeros(len(roles)), cov, size=n)
        Lm = L[:, mem_cols]
        BY, OA = _draw_family(non_g, proband_age, n, rng, death_mean, death_sd)
        Ki = _cohort_K(BY, K, trend_R)                     # cohort prevalence by birth year
        aoo = convert_liability_to_aoo(Lm, pop_prev=Ki)    # onset under own cohort CIP
        aoo = np.where(np.isfinite(aoo), np.round(aoo), np.inf)
        obs = aoo <= OA                                    # observed case iff onset before censoring
        return L[:, 0], Lm, OA, Ki, aoo, obs

    if case_frac is None:
        g, Lm, OA, Ki, aoo, obs = block(n_fam)
    else:
        n_case = int(round(case_frac * n_fam)); n_ctrl = n_fam - n_case
        buf = {k: [] for k in "gLAKae"}; nc = nk = 0
        while nc < n_case or nk < n_ctrl:
            gb, Lmb, OAb, Kib, aoob, obsb = block(max(20000, 8 * n_fam))
            pc = obsb[:, o_pos]
            ci = np.where(pc)[0][:max(0, n_case - nc)]
            ki = np.where(~pc)[0][:max(0, n_ctrl - nk)]
            take = np.concatenate([ci, ki]); nc += len(ci); nk += len(ki)
            for key, arr in zip("gLAKae", (gb, Lmb, OAb, Kib, aoob, obsb)):
                buf[key].append(arr[take])
        g, Lm, OA, Ki, aoo, obs = (np.concatenate(buf[k]) for k in "gLAKae")
        perm = rng.permutation(len(g))
        g, Lm, OA, Ki, aoo, obs = (x[perm] for x in (g, Lm, OA, Ki, aoo, obs))

    aoo_c = np.where(obs, aoo, 60.0)                        # avoid inf in unused case slots
    th_coh_case = _thr(aoo_c, Ki, mid, slope)
    th_coh_ctrl = _thr(OA, Ki, mid, slope)
    th_one_case = _thr(aoo_c, K, mid, slope)
    th_one_ctrl = _thr(OA, K, mid, slope)

    fams_bin, fams_coh, fams_one = [], [], []
    for i in range(len(g)):
        mb, mc, mo = [], [], []
        for j, r in enumerate(non_g):
            if obs[i, j]:
                mb.append(Member(r, T_K, np.inf))
                mc.append(Member(r, th_coh_case[i, j], th_coh_case[i, j]))
                mo.append(Member(r, th_one_case[i, j], th_one_case[i, j]))
            else:
                mb.append(Member(r, -np.inf, T_K))
                mc.append(Member(r, -np.inf, th_coh_ctrl[i, j]))
                mo.append(Member(r, -np.inf, th_one_ctrl[i, j]))
        fams_bin.append(Family(i, mb)); fams_coh.append(Family(i, mc)); fams_one.append(Family(i, mo))

    rel_pos = [j for j, r in enumerate(non_g) if r != "o"]
    return dict(g=g, o_status=obs[:, o_pos].astype(float),
                count=obs[:, rel_pos].sum(axis=1).astype(float),
                fams_bin=fams_bin, fams_coh=fams_coh, fams_one=fams_one)


def _corr(pred, g):
    return 0.0 if np.std(pred) < 1e-12 else float(np.corrcoef(pred, g)[0, 1])


def run_point(fam_vec, h2, K, case_frac, n_fam, reps, method, seed0, proband_age, trend_R):
    acc = defaultdict(list)
    for r in range(reps):
        s = simulate_cohort(fam_vec, h2, K, n_fam, seed0 + r, case_frac, proband_age, trend_R)
        g = s["g"]
        preds = dict(cc=s["o_status"], count=s["count"],
                     ltfh=estimate(s["fams_bin"], h2, method, seed=seed0 + r)[0],
                     ltfhpp=estimate(s["fams_coh"], h2, method, seed=seed0 + r)[0],
                     ltfhpp_one=estimate(s["fams_one"], h2, method, seed=seed0 + r)[0])
        c = {k: _corr(v, g) for k, v in preds.items()}
        for k, v in c.items():
            acc[f"corr_{k}"].append(v)
        acc["effN_ltfh"].append((c["ltfh"] / c["cc"]) ** 2 if c["cc"] > 0 else np.nan)
        acc["effN_age"].append((c["ltfhpp"] / c["ltfh"]) ** 2 if c["ltfh"] > 0 else np.nan)
        # birth-cohort mis-specification is a bias, not an accuracy, effect:
        acc["rmse_ltfhpp"].append(float(np.sqrt(np.mean((preds["ltfhpp"] - g) ** 2))))
        acc["rmse_ltfhpp_one"].append(float(np.sqrt(np.mean((preds["ltfhpp_one"] - g) ** 2))))
        acc["cohort_bias"].append(float(preds["ltfhpp_one"].mean() - preds["ltfhpp"].mean()))
        acc["case_frac"].append(float(s["o_status"].mean()))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def run_cohort_span(h2, K, span, n, seed, mid, slope, trend_R):
    """Isolate the birth-cohort **ranking** gain that the pedigree in (a)–(c) dilutes.

    Cases spanning ±``span`` birth years around 1965, each estimated from its **own**
    age of onset (no family), cohort-aware vs single-``K``. Two cases with the same
    onset age but different cohorts have different true liabilities (the earlier-born,
    lower-prevalence one is more extreme); cohort-aware pins them correctly, single-``K``
    collapses them — so the accuracy gap grows with the cohort span."""
    rng = np.random.default_rng(seed)
    by = rng.uniform(BY_REF - span, BY_REF + span, n)
    Ki = _cohort_K(by, K, trend_R)
    g = rng.normal(0.0, np.sqrt(h2), n)
    L = g + rng.normal(0.0, np.sqrt(1.0 - h2), n)
    aoo = convert_liability_to_aoo(L, pop_prev=Ki)
    obs = rng.uniform(40, 90, n)
    m = np.isfinite(aoo) & (aoo <= obs)                 # observed cases
    a = np.round(aoo[m])
    coh = _thr(a, Ki[m], mid, slope)                    # cohort-aware pin ≈ true liability
    one = _thr(a, K, mid, slope)                        # single-K pin (ignores cohort)
    return dict(corr_cohort=_corr(h2 * coh, g[m]), corr_single=_corr(h2 * one, g[m]),
                n_cases=int(m.sum()))


def _fmt(m):
    return (f"cc={m['corr_cc']:.3f} LT-FH={m['corr_ltfh']:.3f} LT-FH++={m['corr_ltfhpp']:.3f} "
            f"| effN: fh={m['effN_ltfh']:.2f}x age={m['effN_age']:.3f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fam", type=int, default=4000)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--h2s", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--trends", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--spans", type=float, nargs="+", default=[10, 25, 40, 55],
                    help="birth-cohort half-spans (± years) for the ranking-gain panel")
    ap.add_argument("--span-trend", type=float, default=3.0, help="trend R for panel (d)")
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--K", type=float, default=0.05)
    ap.add_argument("--mid", type=float, default=60.0)
    ap.add_argument("--slope", type=float, default=1.0 / 8.0)
    ap.add_argument("--proband-age", type=int, nargs=2, default=[40, 70])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--method", default="pa")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    pa = tuple(args.proband_age)
    base = dict(fam_vec=STRUCT, n_fam=args.n_fam, reps=args.reps, method=args.method,
                proband_age=pa)

    estimate(simulate_cohort(STRUCT, args.h2, args.K, 40, 0, proband_age=pa)["fams_coh"],
             args.h2, args.method)

    print(f"struct={'+'.join(STRUCT)} n_fam={args.n_fam} reps={args.reps} K={args.K} "
          f"CIP(mid={args.mid}, slope={args.slope:.3f})")

    rows = []
    print(f"\n(a) vs ascertainment  [h2={args.h2}, K={args.K}, no cohort trend]")
    for cf in [None] + list(args.fractions):
        m = run_point(h2=args.h2, K=args.K, case_frac=cf, seed0=args.seed, trend_R=1.0, **base)
        rows.append(dict(panel="ascertain", h2=args.h2, K=args.K, trend_R=1.0,
                         target_frac=(cf or 0.0), **m))
        print(f"  P={m['case_frac']:.3f} | {_fmt(m)}")

    print(f"\n(b) vs heritability  [K={args.K}, 50% ascertained, no cohort trend]")
    for h2 in args.h2s:
        m = run_point(h2=h2, K=args.K, case_frac=0.5, seed0=args.seed + 100, trend_R=1.0, **base)
        rows.append(dict(panel="h2", h2=h2, K=args.K, trend_R=1.0, target_frac=0.5, **m))
        print(f"  h2={h2:.1f} | {_fmt(m)}")

    print(f"\n(c) vs secular prevalence trend R  [h2={args.h2}, K={args.K}, 50% ascertained]")
    for R in args.trends:
        m = run_point(h2=args.h2, K=args.K, case_frac=0.5, seed0=args.seed + 200, trend_R=R, **base)
        rows.append(dict(panel="trend", h2=args.h2, K=args.K, trend_R=R, target_frac=0.5, **m))
        print(f"  R={R:.1f}/30y | corr coh={m['corr_ltfhpp']:.3f} single-K={m['corr_ltfhpp_one']:.3f} "
              f"(~equal) | single-K bias={m['cohort_bias']:+.3f} rmse {m['rmse_ltfhpp']:.3f}/{m['rmse_ltfhpp_one']:.3f}")

    print(f"\n(d) cohort RANKING gain vs cohort span of cases  [own onset, R={args.span_trend}]")
    for span in args.spans:
        d = run_cohort_span(args.h2, args.K, span, 5 * args.n_fam, args.seed + 300,
                            args.mid, args.slope, args.span_trend)
        rows.append(dict(panel="cohortspan", span=span, trend_R=args.span_trend,
                         corr_ltfhpp=d["corr_cohort"], corr_ltfhpp_one=d["corr_single"]))
        print(f"  ±{span:.0f}y (born {int(BY_REF - span)}-{int(BY_REF + span)}): "
              f"cohort-aware={d['corr_cohort']:.3f} single-K={d['corr_single']:.3f} (n={d['n_cases']})")

    write_csv(rows)
    plot(rows, args)
    print("\nwrote bench_fh_prediction.csv and bench_fh_prediction.png")


def write_csv(rows):
    fields = ["panel", "h2", "K", "trend_R", "span", "target_frac", "case_frac",
              "corr_cc", "corr_count", "corr_ltfh", "corr_ltfhpp", "corr_ltfhpp_one",
              "effN_ltfh", "effN_age", "rmse_ltfhpp", "rmse_ltfhpp_one", "cohort_bias"]
    path = os.path.join(HERE, "bench_fh_prediction.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in rows])


def plot(rows, args):
    plt = get_plt()
    if plt is None:
        return
    asc = sorted([r for r in rows if r["panel"] == "ascertain"], key=lambda r: r["case_frac"])
    h2s = sorted([r for r in rows if r["panel"] == "h2"], key=lambda r: r["h2"])
    trd = sorted([r for r in rows if r["panel"] == "trend"], key=lambda r: r["trend_R"])
    csp = sorted([r for r in rows if r["panel"] == "cohortspan"], key=lambda r: r["span"])
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs.ravel()
    C = dict(ltfhpp="#2a78d6", ltfh="#1baf7a", count="#eda100", cc="#888780", one="#e34948")

    P = [r["case_frac"] for r in asc]
    ax[0].plot(P, [r["corr_ltfhpp"] for r in asc], "-o", color=C["ltfhpp"], label="LT-FH++")
    ax[0].plot(P, [r["corr_ltfh"] for r in asc], "-s", color=C["ltfh"], label="LT-FH")
    ax[0].plot(P, [r["corr_cc"] for r in asc], "--^", color=C["cc"], label="case/control")
    ax[0].plot(P, [r["corr_count"] for r in asc], ":d", color=C["count"], label="FH-count")
    ax[0].set_xlabel("proband case fraction (ascertainment)")
    ax[0].set_ylabel("corr(prediction, true g)")
    ax[0].set_title(f"(a) ascertainment (h²={args.h2}, K={args.K})")
    ax[0].legend(fontsize=8)

    H = [r["h2"] for r in h2s]
    ax[1].plot(H, [r["corr_ltfhpp"] for r in h2s], "-o", color=C["ltfhpp"], label="LT-FH++")
    ax[1].plot(H, [r["corr_ltfh"] for r in h2s], "-s", color=C["ltfh"], label="LT-FH")
    ax[1].plot(H, [r["corr_cc"] for r in h2s], "--^", color=C["cc"], label="case/control")
    ax[1].set_xlabel("heritability h²")
    ax[1].set_ylabel("corr(prediction, true g)")
    ax[1].set_title("(b) vs heritability (50% ascertained)")
    ax[1].legend(fontsize=8)

    R = [r["trend_R"] for r in trd]
    ax[2].axhline(0.0, color="k", ls=":", lw=1)
    ax[2].bar([str(r) for r in R], [r["cohort_bias"] for r in trd], color=C["one"], width=0.55)
    ax[2].set_xlabel("secular prevalence trend  R  (× per 30 y)")
    ax[2].set_ylabel("single-K liability bias")
    ax[2].set_title("(c) ignoring cohort biases the estimate (pedigree)")

    S = [r["span"] for r in csp]
    ax[3].plot(S, [r["corr_ltfhpp"] for r in csp], "-o", color=C["ltfhpp"], label="cohort-aware")
    ax[3].plot(S, [r["corr_ltfhpp_one"] for r in csp], "-x", color=C["one"], label="single-K")
    ax[3].set_xlabel("birth-cohort half-span of cases  (± years)")
    ax[3].set_ylabel("corr(own-onset estimate, true g)")
    ax[3].set_title(f"(d) cohort also gives a RANKING gain (R={args.span_trend})")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench_fh_prediction.png"), dpi=130)


if __name__ == "__main__":
    main()
