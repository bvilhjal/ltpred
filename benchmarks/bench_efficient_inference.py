"""Small, provenance-bound efficiency pilot; no calibration or scaling claims.

Run from the checkout with its supported interpreter::

    .venv/bin/python benchmarks/bench_efficient_inference.py --output /absolute/path

Each arm runs in a fresh process with its own empty Numba cache directory.
First-call timings include compilation reached by that call; warm timings reuse
the process and compiled kernels. Process wall time includes imports, setup and
all repetitions. Peak RSS is measured by the existing small-parent launcher.
The pipeline's full arm disables only selected-covariance construction, retaining
the current PA engine. The three fitting cohorts are illustrative, not a study
of bias, coverage, sampling-SE calibration or full moment-fitter convergence.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def _command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _power():
    if sys.platform != "darwin":
        return {"status": "not_applicable", "platform": sys.platform}
    battery, settings = _command("pmset", "-g", "batt"), _command("pmset", "-g", "custom")
    active = settings.split("AC Power:", 1)[-1].split("Battery Power:", 1)[0]
    if "AC Power" not in battery or "lowpowermode         0" not in active:
        raise RuntimeError("benchmark requires AC power with Low Power Mode off")
    return {"status": "passed", "battery": battery, "settings": settings}


def _sources():
    paths = sorted((ROOT / "ltpred").glob("*.py")) + [Path(__file__).resolve(), ROOT / "benchmarks" / "_peak_launcher.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes, digest


def _hash_arrays(roles, *arrays):
    import numpy as np
    digest = hashlib.sha256(json.dumps(roles, sort_keys=True).encode())
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _timed(call):
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def _worker(case, directory):
    import numpy as np
    import scipy
    import numba
    from scipy.stats import norm
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = "threadpoolctl unavailable; requested limits recorded, BLAS runtime thread count not independently queried"
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    sys.path.insert(0, str(ROOT))
    import ltpred
    from ltpred.estimate import estimate_liability_gibbs_arrays, estimate_liability_pa_arrays
    from ltpred.quadrature import estimate_liability_quadrature_arrays

    directory.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "environment": {"python": sys.version, "executable": sys.executable,
              "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__,
              "ltpred": ltpred.__version__, "numba_threads": numba.get_num_threads(),
              "threadpools": pools, "numpy_config": config.getvalue(), "thread_variables": {key: os.environ.get(key) for key in THREAD_VARS}},
              "source_digest": _sources()[1]}
    if case.startswith("inference_"):
        roles = ["o", "m", "f", "s1", "s2", "s3", "s4"]
        threshold = norm.isf(.05)
        lower = np.array([[2., -np.inf, threshold, threshold, -np.inf, -np.inf, -np.inf]])
        upper = np.array([[2., threshold, np.inf, np.inf, threshold, threshold, threshold]])
        np.savez(directory / "input.npz", roles=roles, lower=lower, upper=upper)
        result["input_sha256"] = _hash_arrays(roles, lower, upper)
        result["settings"] = {"h2": .5, "prevalence": .05, "families": 1,
                              "description": "own pinned2; mother control; father case; one case and three control siblings"}
        if case == "inference_pa":
            call = lambda: estimate_liability_pa_arrays(roles, lower, upper, h2=.5)
        elif case == "inference_quadrature":
            call = lambda: estimate_liability_quadrature_arrays(roles, lower, upper, h2=.5)
        else:
            result["settings"].update(n_sim=1_000_000, burn_in=2000, tol=.001, max_rounds=1, seed=187)
            call = lambda: estimate_liability_gibbs_arrays(roles, lower, upper, h2=.5,
                       n_sim=1_000_000, burn_in=2000, tol=.001, max_rounds=1, seed=187, return_var=True)
        value, result["first_call_seconds"] = _timed(call)
        result["warm_seconds"] = [_timed(call)[1] for _ in range(3)]
        if case == "inference_quadrature":
            result["moments"] = {"mean": float(value.est[0]), "variance": float(value.var[0]),
                                  "resolution_change": float(value.error[0]), "nodes_per_dimension": int(value.n_nodes[0])}
            tight = estimate_liability_quadrature_arrays(roles, lower, upper, h2=.5, atol=1e-12, max_nodes=256)
            result["tight_check"] = {"mean": float(tight.est[0]), "variance": float(tight.var[0]),
                                      "resolution_change": float(tight.error[0]), "nodes_per_dimension": int(tight.n_nodes[0])}
        elif case == "inference_pa":
            result["moments"] = {"mean": float(value[0][0]), "variance": float(value[1][0])}
            exact = estimate_liability_pa_arrays(["o", "m"], [[3., -np.inf]], [[3., threshold]], h2=.5)
            result["pin_first_oracle"] = {"mean": float(exact[0][0]), "variance": float(exact[1][0]),
                "expected_mean": 1.4591376235349836, "expected_variance": .24345482008488772}
            np.testing.assert_allclose([exact[0][0], exact[1][0]], [1.4591376235349836, .24345482008488772], rtol=0, atol=2e-12)
        else:
            result["moments"] = {"mean": float(value[0][0]), "mcse": float(value[1][0]), "variance": float(value[2][0])}
    elif case.startswith("adult_"):
        from ltpred.pearson_aitken import pa_estimate_batched
        n = 20_000
        lower, upper = np.full((n, 1), -np.inf), np.full((n, 1), np.inf)
        lower[::4, 0] = upper[::4, 0] = 2.0
        upper[1::4, 0] = norm.isf(.05)
        lower[2::4, 0] = norm.isf(.05)
        covariance = np.array([[.5, .5], [.5, 1.]])
        lo2, hi2 = np.column_stack((np.full(n, -np.inf), lower)), np.column_stack((np.full(n, np.inf), upper))
        scalar = lambda: estimate_liability_pa_arrays(["o"], lower, upper, h2=.5)
        full = lambda: pa_estimate_batched(covariance, lo2, hi2)
        call = scalar if case == "adult_scalar" else full
        value, result["first_call_seconds"] = _timed(call)
        result["warm_seconds"] = [_timed(call)[1] for _ in range(3)]
        other = full() if case == "adult_scalar" else scalar()
        result["equivalence"] = {"max_abs_mean_difference": float(np.max(np.abs(value[0]-other[0]))),
                                  "max_abs_variance_difference": float(np.max(np.abs(value[1]-other[1])))}
        np.testing.assert_allclose(value, other, rtol=0, atol=1e-12)
        result["input_sha256"] = _hash_arrays(["o"], lower, upper)
        result["settings"] = {"families": n, "h2": .5, "bounds": "equal groups: pin2, controlK=.05, caseK=.05, uninformative",
                              "comparator": "current public two-coordinate PA covariance API"}
        np.savez(directory / "input_and_scores.npz", lower=lower, upper=upper, est=value[0], var=value[1])
    elif case.startswith("pipeline_"):
        import ltpred.pipeline as pipeline
        n, depth = 200, 60
        ids = [f"o{i}" for i in range(n)] + [f"m{i}" for i in range(n)] + [f"a{i}" for i in range(depth + 1)]
        father = ["a0"] * n + [None] * n + [f"a{i+1}" for i in range(depth)] + [None]
        mother = [f"m{i}" for i in range(n)] + [None] * (n + depth + 1)
        status = np.zeros(len(ids), dtype=int)
        status[:n:10] = 1
        status[n:2*n:17] = 1
        ages = np.array([50.] * n + [75.] * n + [80.] * (depth + 1))
        cip_ages = np.arange(121.)
        cip = .1 / (1. + np.exp((60. - cip_ages) / 8.))
        inputs = dict(ids=ids, father=father, mother=mother, probands=ids[:n], status=status,
                      age=ages, use="gwas", cip_ages=cip_ages, cip_values=cip, k_pop=.1,
                      h2=.5, max_degree=1)
        np.savez(directory / "input.npz", ids=ids, father=np.array(["" if x is None else x for x in father]),
                 mother=np.array(["" if x is None else x for x in mother]), status=status,
                 age=ages, cip_ages=cip_ages, cip_values=cip)
        result["input_sha256"] = _hash_arrays([ids, father, mother], status, ages, cip_ages, cip)
        result["settings"] = {"probands": n, "population_size": len(ids), "shared_paternal_chain_depth": depth,
                              "h2": .5, "max_degree": 1, "full_arm_change": "_covariance_reduction_is_safe always false; current PA retained"}
        original = pipeline._covariance_reduction_is_safe
        try:
            if case == "pipeline_full":
                pipeline._covariance_reduction_is_safe = lambda h2, size: False
            call = lambda: pipeline.estimate_liabilities(**inputs)
            value, result["first_call_seconds"] = _timed(call)
            result["warm_seconds"] = [_timed(call)[1] for _ in range(3)]
        finally:
            pipeline._covariance_reduction_is_safe = original
        np.savez(directory / "scores.npz", est=value.est, var=value.var,
                 n_closure_only=value.n_closure_only, n_conditioned=value.n_conditioned)
        result["scores"] = {"est": value.est.tolist(), "var": value.var.tolist(),
                            "closure_only_range": [int(value.n_closure_only.min()), int(value.n_closure_only.max())]}
    else:
        from ltpred import simulate_under_LTM_single, fit_heritability
        from ltpred.pairwise import fit_pairwise
        result["settings"] = {"families": 1000, "roles": ["o", "m", "f", "s1"], "h2_true": .5,
                              "prevalence": .1, "data_seeds": [31, 32, 33], "sampling": "population",
                              "moment_fitter": {"n_iter": 1500, "burn_in": 500, "inner_sweeps": 5}}
        result["replicates"] = []
        for seed in [31, 32, 33]:
            sim = simulate_under_LTM_single(fam_vec=["m", "f", "s1"], h2=.5, n_sim=1000,
                                           pop_prev=.1, use_age=False, seed=seed)
            lower = np.array([[member.lower for member in family.members] for family in sim.families])
            upper = np.array([[member.upper for member in family.members] for family in sim.families])
            roles = [member.role for member in sim.families[0].members]
            np.savez(directory / f"input_seed{seed}.npz", roles=roles, lower=lower, upper=upper)
            if case == "fit_pairwise":
                call = lambda: fit_pairwise(sim.families, sampling="population")
            else:
                call = lambda: fit_heritability(sim.families, sampling="population", n_iter=1500,
                                               burn_in=500, inner_sweeps=5, seed=1000+seed)
            if seed == 31:
                _, result["first_call_seconds"] = _timed(call)
            value, elapsed = _timed(call)
            record = {"seed": seed, "seconds": elapsed, "input_sha256": _hash_arrays(roles, lower, upper)}
            if case == "fit_pairwise":
                record.update(h2=value.components["A"], family_sandwich_se=value.se["A"],
                              inference_status=value.inference_status, iterations=value.n_iter)
            else:
                record.update(h2=value.h2, mcse=value.h2_se)
            result["replicates"].append(record)
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", default=None)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker, args.output)
        return
    from _peak_launcher import run_peak
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    power = _power()
    hashes, digest = _sources()
    for relative in hashes:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
                "revision": _command("git", "rev-parse", "HEAD"), "git_status": _command("git", "status", "--short"),
                "source_hashes": hashes, "source_digest": digest, "platform": platform.platform(),
                "machine": platform.machine(), "logical_cpus": os.cpu_count(), "power_guard": power,
                "timing_contract": "fresh process and empty per-arm Numba cache; first_call vs warm; all imports/setup outside call timing; process timing includes all work",
                "limitations": ["synthetic small workloads", "one CPU thread", "three fitting seeds do not establish accuracy or coverage",
                                "moment-fitter default controls; convergence not established", "sandwich SE and fitter MCSE measure different uncertainty",
                                "pipeline full comparator retains current PA", "peak RSS includes process imports and all repetitions"]}
    if sys.platform == "darwin":
        # Preserve useful hardware fields, not machine/user identifiers.
        hardware = _command("system_profiler", "SPHardwareDataType")
        keep = ("Model Name:", "Model Identifier:", "Chip:", "Total Number of Cores:", "Memory:")
        metadata["hardware"] = [line.strip() for line in hardware.splitlines() if line.strip().startswith(keep)]
    cases = ["inference_pa", "inference_quadrature", "inference_gibbs", "adult_scalar", "adult_covariance", "pipeline_selected", "pipeline_full", "fit_pairwise", "fit_moment"]
    results = {}
    start = time.perf_counter()
    for case in cases:
        directory = output / case
        directory.mkdir(exist_ok=True)
        env = dict(os.environ, **{key: "1" for key in THREAD_VARS}, NUMBA_CACHE_DIR=str(directory / f"numba_cache_{time.time_ns()}"))
        proc_start = time.perf_counter()
        proc, peak = run_peak([sys.executable, str(Path(__file__).resolve()), "--worker", case, "--output", str(directory)], cwd=ROOT, env=env)
        elapsed = time.perf_counter() - proc_start
        (directory / "process.log").write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"{case} failed; see {directory / 'process.log'}")
        result = json.loads((directory / "result.json").read_text())
        result.update(process_seconds=elapsed, peak_rss_bytes=peak, launcher_floor_kb=proc.floor_kb)
        results[case] = result
        print(f"{case}: process {elapsed:.3f}s, peak RSS {peak / 1024**2:.1f}MiB", flush=True)
    metadata["elapsed_seconds"] = time.perf_counter() - start
    metadata["source_digest_after"] = _sources()[1]
    metadata["source_stable"] = metadata["source_digest_after"] == digest and all(item["source_digest"] == digest for item in results.values())
    metadata["power_guard_after"] = _power()
    selected, full = results["pipeline_selected"]["scores"], results["pipeline_full"]["scores"]
    checks = {"pipeline_max_abs_mean_difference": max(abs(a-b) for a,b in zip(selected["est"], full["est"])),
              "pipeline_max_abs_variance_difference": max(abs(a-b) for a,b in zip(selected["var"], full["var"])),
              "pipeline_input_equal": results["pipeline_selected"]["input_sha256"] == results["pipeline_full"]["input_sha256"],
              "fitting_inputs_equal": [r["input_sha256"] for r in results["fit_pairwise"]["replicates"]] == [r["input_sha256"] for r in results["fit_moment"]["replicates"]]}
    artifact = {"metadata": metadata, "checks": checks, "results": results}
    (output / "results.json").write_text(json.dumps(artifact, indent=2) + "\n")
    if not metadata["source_stable"] or not checks["pipeline_input_equal"] or not checks["fitting_inputs_equal"] or max(checks["pipeline_max_abs_mean_difference"], checks["pipeline_max_abs_variance_difference"]) > 1e-12:
        raise RuntimeError("source/input/equivalence check failed; see results.json")


if __name__ == "__main__":
    main()
