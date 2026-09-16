"""Replicated joint h2/rg/re recovery, uncertainty and misspecification checks.

Run from a checkout, for example:
  python benchmarks/bench_pairwise_multi.py --reps 100 --n-fam 3000 --jobs 4 \
    --output benchmarks/results/2026-09-16-joint-pairwise

All attempted fits are retained, including failures and boundaries. Coverage is
conditional on an available interior SE and is accompanied by its denominator.
This is a statistical campaign, not a controlled timing comparison. The output
contains every replicate, summaries, versions, source hashes and a source zip;
it can therefore describe an explicitly dirty development checkout faithfully.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
import zipfile

import numpy as np
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ltpred.family import Family, Member
from ltpred.pairwise_multi import fit_pairwise_multi

SCENARIOS = ("positive", "negative_g", "negative_e", "shared", "missing", "ipw", "omit_shared", "null_g", "null_e")
ROLES = ("o", "m", "f", "s1", "s2")


def truth(scenario):
    shared = scenario in ("shared", "missing", "omit_shared")
    h = np.array([.35, .4]) if shared else np.array([.45, .35])
    rg = 0. if scenario == "null_g" else (-.45 if scenario == "negative_g" else .5)
    re = 0. if scenario == "null_e" else (-.4 if scenario == "negative_e" else (-.35 if shared else .35))
    matrices = {"A": np.diag(h)}
    matrices["A"][0, 1] = matrices["A"][1, 0] = rg*np.sqrt(h.prod())
    if shared:
        matrices["C"] = np.array([[.15, .075], [.075, .15]])
        matrices["M"] = np.array([[.1, .02], [.02, .1]])
    e = 1-sum(np.diag(v) for v in matrices.values())
    matrices["E"] = np.diag(e)
    matrices["E"][0, 1] = matrices["E"][1, 0] = re*np.sqrt(e.prod())
    return matrices, dict(h2_0=h[0], h2_1=h[1], rg=rg, re=re)


def simulate(scenario, n, seed):
    """Independent nuclear-family kernel construction, then Gaussian sampling."""
    matrices, target = truth(scenario)
    k = len(ROLES)
    a = np.full((k, k), .5)
    np.fill_diagonal(a, 1.)
    a[1, 2] = a[2, 1] = 0.             # unrelated parents
    c, m = np.eye(k), np.eye(k)
    for i in (0, 3, 4):                # the three full siblings
        for j in (0, 3, 4):
            c[i, j] = 1.
    m[1, 2] = m[2, 1] = 1.
    kernels = dict(A=a, C=c, M=m, E=np.eye(k))
    sigma = sum(np.kron(matrix, kernels[name]) for name, matrix in matrices.items())
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 2*k)) @ np.linalg.cholesky(sigma).T
    threshold = -ndtri(np.array([.1, .2]))
    status = latent.reshape(n, 2, k).transpose(0, 2, 1) > threshold
    weights = None
    if scenario == "ipw":
        pi = np.where(status[:, 0, 0], 1., .2)
        keep = rng.random(n) < pi
        status, weights = status[keep], 1/pi[keep]
    observed = rng.random(status.shape) >= (.2 if scenario == "missing" else 0.)
    lo = np.where(status & observed, threshold, -np.inf)
    hi = np.where(~status & observed, threshold, np.inf)
    families = [Family(i, [Member(role, lo[i, j], hi[i, j]) for j, role in enumerate(ROLES)])
                for i in range(len(status))]
    return families, weights, target


def one(task):
    scenario, n, seed = task
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        families, weights, target = simulate(scenario, n, seed)
        comps = ("A", "C", "M") if scenario in ("shared", "missing") else ("A",)
        record = dict(scenario=scenario, seed=seed, n_population=n,
                      n_families=len(families), truth=target)
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = fit_pairwise_multi(families, components=comps,
                    sampling="population" if weights is None else "ipw", weights=weights)
            record.update(estimate=dict(h2_0=result.h2[0], h2_1=result.h2[1],
                                        rg=result.rg[0, 1], re=result.re[0, 1]),
                          se=dict(h2_0=result.se["h2"][0], h2_1=result.se["h2"][1],
                                  rg=result.se["rg"][0, 1], re=result.se["re"][0, 1]),
                          status=result.inference_status, boundary=result.at_boundary,
                          n_pairs=result.n_pairs, n_iter=result.n_iter,
                          warnings=[str(w.message) for w in caught])
        except Exception as error:
            record.update(status="failed", error=repr(error))
        record["fit_seconds_uncontrolled"] = time.perf_counter()-started
    return record


def summarise(records):
    out = []
    for scenario in dict.fromkeys(r["scenario"] for r in records):
        attempted = [r for r in records if r["scenario"] == scenario]
        fits = [r for r in attempted if r["status"] != "failed"]
        for parameter in ("h2_0", "h2_1", "rg", "re"):
            target = attempted[0]["truth"][parameter]
            x = np.array([r["estimate"][parameter] for r in fits], dtype=float)
            se = np.array([r["se"][parameter] for r in fits], dtype=float)
            finite = np.isfinite(x)
            available = finite & np.isfinite(se)
            sd = np.std(x[finite], ddof=1) if finite.sum() > 1 else np.nan
            covered = np.abs(x[available]-target) <= 1.96*se[available]
            coverage = covered.mean() if available.any() else np.nan
            # Empirical SD on the same interior subset used for SE calibration.
            interior_sd = np.std(x[available], ddof=1) if available.sum() > 1 else np.nan
            out.append(dict(scenario=scenario, parameter=parameter, truth=float(target),
                attempts=len(attempted), failures=len(attempted)-len(fits),
                boundary_fits=sum(r["boundary"] for r in fits), finite_estimates=int(finite.sum()),
                bias=float(x[finite].mean()-target) if finite.any() else np.nan,
                sd=float(sd), bias_mcse=float(sd/np.sqrt(finite.sum())) if finite.any() else np.nan,
                coverage_n=int(available.sum()), coverage=float(coverage),
                coverage_mcse=float(np.sqrt(coverage*(1-coverage)/available.sum())) if available.any() else np.nan,
                se_over_sd=float(se[available].mean()/interior_sd) if available.sum() > 1 else np.nan))
    return out


def source_files():
    return sorted([*ROOT.joinpath("ltpred").glob("*.py"), Path(__file__).resolve()])


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def clean_json(obj):
    if isinstance(obj, dict):
        return {key: clean_json(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean_json(value) for value in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-fam", type=int, default=3000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=91700)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 2 or args.n_fam < 2 or args.jobs < 1:
        parser.error("reps/n-fam must be at least 2 and jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    import scipy
    import ltpred
    metadata = dict(command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    ltpred=ltpred.__version__, platform=platform.platform(), source_hashes=before,
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    configuration={**vars(args), "output": str(args.output)},
                    timing_note="Uncontrolled statistical campaign; fit timings are not performance rankings.")
    with zipfile.ZipFile(args.output/"source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, str(path.relative_to(ROOT)))
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for scenario in args.scenarios:
            # Matched seed streams isolate omission/missingness; each replicate
            # remains independent. Process scheduling does not choose seeds.
            tasks = [(scenario, args.n_fam, args.seed+r) for r in range(args.reps)]
            records.extend(pool.map(one, tasks))
            (args.output/"replicates.json").write_text(json.dumps(clean_json(records), indent=2)+"\n")
            summary = summarise(records)
            print(scenario, [row for row in summary if row["scenario"] == scenario and row["parameter"] in ("rg", "re")], flush=True)
    metadata["source_unchanged"] = before == hashes()
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output/"summary.json").write_text(json.dumps(clean_json(summary), indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if not metadata["source_unchanged"]:
        raise RuntimeError("source changed during the campaign; results are not accepted")


if __name__ == "__main__":
    main()
