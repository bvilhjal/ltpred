"""Matched runtime/allocation pilot against a Git revision; synthetic inputs.

Run from the checkout: .venv/bin/python benchmarks/bench_time_memory.py
    --output /tmp/ltpred-time-memory --baseline-ref HEAD

Each source/case gets separate timing and allocation processes. Timing reports
first call (including reached JIT compilation) and warm calls; whole-process
peak RSS comes from the timing process only. Tracemalloc measures allocations
inside one call after warming kernels, separately from timings and RSS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ("graph_200k", "graph_1m", "pa_mixed", "pa_pin", "pa_intervals",
         "pa_mixture", "pipeline")


def _sources(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "ltpred").glob("*.py"))}


def _workload(case):
    import numpy as np
    from ltpred.pedigree import build_parent_graph
    from ltpred.pearson_aitken import pa_estimate_batched
    from ltpred.pipeline import estimate_liabilities

    if case.startswith("graph_"):
        n = 200_000 if case == "graph_200k" else 1_000_000
        ids = list(range(n))
        father = [None] * (n // 2) + list(range(n // 4)) * 2
        mother = [None] * (n // 2) + list(range(n // 4, n // 2)) * 2
        call = lambda: build_parent_graph(ids, father, mother)
        return call, lambda: None, {"records": n, "design": "half founders; two children per couple"}
    if case.startswith("pa_"):
        n, d = 200_000, 17
        rng = np.random.default_rng(77)
        cov = np.full((d, d), .2) + .8 * np.eye(d)
        lo, hi = np.full((n, d), -np.inf), np.full((n, d), 1.7)
        hi[:, 0] = np.inf
        kwargs = {}
        if case == "pa_mixed":
            for j in range(1, d):
                rows = rng.random(n) < .03
                lo[rows, j] = hi[rows, j] = 2. + rng.random(rows.sum())
        elif case == "pa_pin":
            lo[:, 1] = hi[:, 1] = rng.uniform(1.8, 3., n)
        elif case == "pa_mixture":
            ki, kp = np.full((n, d), .02), np.full((n, d), .1)
            ki[:, 0] = kp[:, 0] = np.nan
            kwargs = dict(K_is=ki, K_pops=kp)
        digest = hashlib.sha256()
        for array in (cov, lo, hi, *kwargs.values()):
            digest.update(memoryview(array).cast("B"))
        call = lambda: pa_estimate_batched(cov, lo, hi, **kwargs)
        warm = lambda: pa_estimate_batched(cov, lo[:64], hi[:64],
                                          **{k: v[:64] for k, v in kwargs.items()})
        return call, warm, {"families": n, "coordinates": d, "seed": 77,
                            "input_sha256": digest.hexdigest()}
    n, depth = 300, 60
    ids = [f"o{i}" for i in range(n)] + [f"m{i}" for i in range(n)] + [f"a{i}" for i in range(depth + 1)]
    father = ["a0"] * n + [None] * n + [f"a{i+1}" for i in range(depth)] + [None]
    mother = [f"m{i}" for i in range(n)] + [None] * (n + depth + 1)
    status = np.zeros(len(ids), dtype=int)
    status[:n:10], status[n:2*n:17] = 1, 1
    ages = np.array([50.] * n + [75.] * n + [80.] * (depth + 1))
    cip_ages = np.arange(121.)
    cip = .1 / (1. + np.exp((60. - cip_ages) / 8.))
    inputs = dict(ids=ids, father=father, mother=mother, probands=ids[:n],
                  status=status, age=ages, use="gwas", cip_ages=cip_ages,
                  cip_values=cip, k_pop=.1, h2=.5, max_degree=1)
    return (lambda: estimate_liabilities(**inputs),
            lambda: estimate_liabilities(**dict(inputs, probands=ids[:5])),
            {"probands": n, "records": len(ids), "shared_paternal_chain_depth": depth,
             "h2": .5, "max_degree": 1, "use": "gwas"})


def _worker(source, case, mode, output, reps):
    sys.path.insert(0, str(source))
    import numpy as np
    import scipy
    import numba
    import ltpred
    call, warm, settings = _workload(case)
    result = {"case": case, "mode": mode, "settings": settings,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "numba": numba.__version__, "ltpred": ltpred.__version__,
                              "numba_threads": numba.get_num_threads()},
              "load_before": os.getloadavg(), "source_hashes": _sources(source)}
    if mode == "allocation":
        import tracemalloc
        warm()
        gc.collect()
        tracemalloc.start()
        value = call()
        result["peak_allocated_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    else:
        elapsed = []
        for _ in range(reps + 1):
            gc.collect()
            start = time.perf_counter()
            value = call()
            elapsed.append(time.perf_counter() - start)
            del value
        result.update(first_call_seconds=elapsed[0], warm_seconds=elapsed[1:])
        # Export from the allocation process only: serialization cannot inflate
        # the timing process's RSS, and no previous output survives a call.
        value = None
    if mode == "allocation":
        if case.startswith("graph_"):
            digest = hashlib.sha256()
            for rows in (value.ids, value.sire, value.dam):
                digest.update(np.asarray(rows, dtype="<i8").tobytes())
            for adjacency in (value.children, value.sibs):
                for row in adjacency:
                    digest.update(np.asarray([len(row), *row], dtype="<i8").tobytes())
            result["graph_sha256"] = digest.hexdigest()
        else:
            arrays = dict(est=value[0], var=value[1]) if case.startswith("pa_") else {
                key: getattr(value, key) for key in ("est", "var", "n_relatives",
                    "n_conditioned", "n_closure_only", "degree_max")}
            np.savez(output.with_suffix(".npz"), **arrays)
    result["load_after"] = os.getloadavg()
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--worker", nargs=3, metavar=("SOURCE", "CASE", "MODE"))
    args = parser.parse_args()
    if args.worker:
        _worker(Path(args.worker[0]), *args.worker[1:], args.output, args.reps)
        return
    from bench_efficient_inference import _power, THREAD_VARS
    from _peak_launcher import run_peak
    import numpy as np
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    power = _power()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(["git", "archive", revision, "ltpred"], cwd=ROOT)
    baseline = output / "baseline"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                target = baseline / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    candidate = output / "candidate"
    shutil.copytree(ROOT / "ltpred", candidate / "ltpred", ignore=shutil.ignore_patterns("__pycache__"))
    sources = dict(baseline=baseline, candidate=candidate)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "baseline_revision": revision,
                "command": sys.argv, "power_guard": power, "results": [], "agreement": {},
                "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_hashes": {arm: _sources(path) for arm, path in sources.items()},
                "thread_variables": {key: "1" for key in THREAD_VARS},
                "timing_contract": "first call includes reached JIT compilation; five warm calls by default; GC outside timing; RSS from timing process only",
                "memory_contract": "tracemalloc call peak in separate warmed process; excludes inputs/imports; includes Python and NumPy allocations but not all native/JIT workspace"}
    for index, case in enumerate(args.case or CASES):
        arms = list(sources) if index % 2 == 0 else list(reversed(sources))
        for arm in arms:
            env = dict(os.environ, **metadata["thread_variables"])
            env["NUMBA_CACHE_DIR"] = str(output / f"cache-{arm}-{case}")
            for mode in ("time", "allocation"):
                path = output / f"{case}-{arm}-{mode}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--output", str(path),
                       "--reps", str(args.reps), "--worker", str(sources[arm]), case, mode]
                proc, peak = run_peak(cmd, cwd=ROOT, env=env)
                if proc.returncode:
                    raise RuntimeError(proc.stdout)
                row = json.loads(path.read_text())
                row["arm"] = arm
                if mode == "time":
                    row["peak_rss_bytes"] = peak
                metadata["results"].append(row)
        if case.startswith("graph_"):
            rows = [r for r in metadata["results"] if r["case"] == case and r["mode"] == "allocation"]
            same = rows[0]["graph_sha256"] == rows[1]["graph_sha256"]
            metadata["agreement"][case] = {"identical_graph": same}
            assert same
        else:
            with np.load(output / f"{case}-baseline-allocation.npz") as old, np.load(output / f"{case}-candidate-allocation.npz") as new:
                metadata["agreement"][case] = {key: float(np.max(np.abs(old[key] - new[key]))) for key in old.files}
                for key in old.files:
                    np.testing.assert_allclose(old[key], new[key], rtol=0, atol=1e-12)
        print(f"{case}: both sources measured; agreement passed", flush=True)
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert _sources(ROOT) == metadata["source_hashes"]["candidate"], "source changed during benchmark"


if __name__ == "__main__":
    main()
