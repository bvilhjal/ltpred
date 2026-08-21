#!/usr/bin/env python3
"""Run one benchmark and append a compact, machine-readable provenance row.

Example::

    python benchmarks/run_benchmark.py --artifact bench_accuracy.csv bench_accuracy.py -- --reps 5

The wrapper requires a clean *source* tree. Existing benchmark CSV,
PNG, and manifest changes are ignored by that gate so a full campaign can run
before its artifacts are committed together.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MANIFEST = HERE / "run_manifest.jsonl"
ARTIFACT_SUFFIXES = {".csv", ".png"}
THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _git(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout


def _source_state(ignored_paths=()):
    """Return the commit and source changes present before a run."""
    commit = _git("rev-parse", "HEAD").strip()
    exclusions = [
        ":(exclude)benchmarks/bench_*.csv",
        ":(exclude)benchmarks/bench_*.png",
        ":(exclude)benchmarks/run_manifest.jsonl",
    ]
    for path in ignored_paths:
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        # A tracked file remains part of the source tree even when the caller
        # also declares it as an input. Otherwise ``--input ltpred/foo.py``
        # could hide a local source edit while the manifest claimed a clean
        # commit. Untracked inputs are content-addressed in the record below.
        if _git("ls-files", "--cached", "--", relative.as_posix()).strip():
            continue
        exclusions.append(f":(exclude,literal){relative.as_posix()}")
    status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", "--", ".",
        *exclusions,
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_fingerprint(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _artifact_state():
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(HERE.glob("bench_*"))
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES
    }


def _changed_artifacts(before, after):
    return [
        {"path": name, **details}
        for name, details in after.items()
        if before.get(name) != details
    ]


def _resolve_artifact(name):
    path = (HERE / name).resolve()
    if path.parent != HERE or path.suffix not in ARTIFACT_SUFFIXES:
        raise ValueError("artifacts must be CSV/PNG files directly under benchmarks/")
    return path


def _resolve_input(name):
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"input is not a file: {name}")
    return path


def _display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _package_versions():
    versions = {}
    for name in ("numpy", "scipy", "numba", "matplotlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    source = (ROOT / "ltpred" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", source, re.M)
    versions["ltpred"] = match.group(1) if match else None
    return versions


def _cpu_model():
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], check=False,
                capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _environment():
    try:
        import numba
        numba_threads = numba.get_num_threads()
    except (ImportError, RuntimeError):
        numba_threads = None
    return {
        "python": sys.version,
        "free_threading": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "packages": _package_versions(),
        "numba_threads": numba_threads,
        "thread_settings": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _load_average():
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return None
    return {"1min": one, "5min": five, "15min": fifteen}


def _resolve_script(name):
    script = (HERE / name).resolve()
    if (script.parent != HERE or not script.name.startswith("bench_")
            or script.suffix != ".py" or not script.is_file()):
        raise ValueError("script must name an existing benchmarks/bench_*.py file")
    return script


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", action="append", required=True, metavar="FILE",
        help="CSV/PNG output to hash after the run; repeat for multiple outputs",
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="FILE",
        help="external input to hash before the run; repeat for multiple inputs",
    )
    parser.add_argument("script")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        script = _resolve_script(args.script)
        declared_artifacts = [_resolve_artifact(name) for name in args.artifact]
        declared_inputs = [_resolve_input(name) for name in args.input]
        source = _source_state(declared_inputs)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"run_benchmark: {exc}") from None
    if source["dirty"]:
        changed = "\n  ".join(source["status"])
        raise SystemExit(
            "run_benchmark: source tree is dirty; commit it first so the run "
            f"has an exact source commit:\n  {changed}")

    benchmark_args = list(args.benchmark_args)
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    command = [sys.executable, str(script), *benchmark_args]
    before = _artifact_state()
    artifact_fingerprints = {
        path: _stat_fingerprint(path) for path in declared_artifacts}
    input_fingerprints = {
        path: _stat_fingerprint(path) for path in declared_inputs}
    inputs = [
        {"path": _display_path(path), "bytes": path.stat().st_size,
         "sha256": _sha256(path)}
        for path in declared_inputs
    ]
    environment = _environment()
    load_before = _load_average()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - t0
    finished = datetime.now(timezone.utc)
    load_after = _load_average()
    after = _artifact_state()
    source_after = _source_state(declared_inputs)
    source["unchanged_after_run"] = (
        source_after["commit"] == source["commit"] and not source_after["dirty"])
    source["commit_after_run"] = source_after["commit"]
    source["status_after_run"] = source_after["status"]
    missing = [path.name for path in declared_artifacts if not path.is_file()]
    untouched = [
        path.name for path in declared_artifacts
        if path.is_file() and _stat_fingerprint(path) == artifact_fingerprints[path]
    ]
    changed_inputs = [
        _display_path(path) for path in declared_inputs
        if _stat_fingerprint(path) != input_fingerprints[path]
    ]
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size,
         "sha256": _sha256(path), "touched": path.name not in untouched}
        for path in declared_artifacts if path.is_file()
    ]
    changed = _changed_artifacts(before, after)
    problems = []
    if missing:
        problems.append(f"missing artifacts: {', '.join(missing)}")
    if untouched:
        problems.append(f"untouched artifacts: {', '.join(untouched)}")
    if changed_inputs:
        problems.append(f"inputs changed during run: {', '.join(changed_inputs)}")
    if not source["unchanged_after_run"]:
        problems.append("source changed during run")
    wrapper_exit_code = result.returncode or (2 if problems else 0)

    record = {
        "schema_version": 1,
        "run_id": (started.strftime("%Y%m%dT%H%M%S.%fZ")
                   + f"-{script.stem}-{os.getpid()}"),
        "script": str(script.relative_to(ROOT)),
        "command": command,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": elapsed,
        "exit_code": result.returncode,
        "wrapper_exit_code": wrapper_exit_code,
        "provenance_errors": problems,
        "source": source,
        "environment": environment,
        "load_average": {"before": load_before, "after": load_after},
        "inputs": inputs,
        "changed_inputs": changed_inputs,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "untouched_artifacts": untouched,
        "changed_artifacts": changed,
    }
    manifest = DEFAULT_MANIFEST.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    print(f"provenance: {manifest}")
    if problems:
        print("provenance incomplete; " + "; ".join(problems), file=sys.stderr)
    return wrapper_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
