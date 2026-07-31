"""Run one local benchmark and append a machine-readable provenance record.

Example:
    python benchmarks/run_benchmark.py bench_accuracy.py -- --reps 5

The wrapper preserves the benchmark's stdout/stderr in hashed log files and
records the invocation, source state, environment, exit status, and changed
output hashes.
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
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MANIFEST = HERE / "run_manifest.jsonl"
DEFAULT_LOG_DIR = HERE / "run_logs"
ARTIFACT_SUFFIXES = {".csv", ".png"}
SOURCE_SUFFIXES = {
    "", ".c", ".cpp", ".h", ".ini", ".json", ".md", ".py", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args, text=True):
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=text,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _git_state():
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    diff = _git("diff", "--binary", "HEAD", text=False)
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
    status_lines = [] if status is None else status.splitlines()
    untracked_source = {}
    for relative in [] if untracked is None else untracked.split("\0"):
        if not relative:
            continue
        path = ROOT / relative
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            untracked_source[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return {
        "commit": None if commit is None else commit.strip(),
        "dirty": bool(status_lines),
        "status_porcelain_v1": status_lines,
        "tracked_diff_sha256": None if diff is None else _sha256_bytes(diff),
        "untracked_source_files": untracked_source,
    }


def _package_versions():
    versions = {}
    for package in ("ltpred", "numpy", "scipy", "numba", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _artifact_state():
    state = {}
    for path in sorted(HERE.glob("bench_*")):
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
            state[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return state


def _changed_artifacts(before, after):
    return [
        {"path": name, **info}
        for name, info in after.items()
        if before.get(name) != info
    ]


def _file_record(path):
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSONL destination (default: benchmarks/run_manifest.jsonl)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="stdout/stderr directory (default: benchmarks/run_logs)",
    )
    parser.add_argument("script", help="benchmark filename, for example bench_accuracy.py")
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the benchmark; place them after --",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    script = (HERE / args.script).resolve()
    if script.parent != HERE or not script.name.startswith("bench_") or script.suffix != ".py":
        raise SystemExit("script must be a benchmarks/bench_*.py file")
    if not script.is_file():
        raise SystemExit(f"benchmark not found: {script.name}")

    benchmark_args = list(args.benchmark_args)
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    command = [sys.executable, str(script), *benchmark_args]

    started = datetime.now(timezone.utc)
    before = _artifact_state()
    git_before = _git_state()
    run_id = (
        started.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-{script.stem}-{os.getpid()}"
    )
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_id}.stdout.log"
    stderr_path = log_dir / f"{run_id}.stderr.log"

    t0 = time.perf_counter()
    with stdout_path.open("wb") as stdout_fh, stderr_path.open("wb") as stderr_fh:
        proc = subprocess.run(
            command, cwd=ROOT, check=False, stdout=stdout_fh, stderr=stderr_fh)
    elapsed = time.perf_counter() - t0
    finished = datetime.now(timezone.utc)
    after = _artifact_state()

    # Preserve the familiar console behavior after the child exits while keeping
    # exact byte-for-byte logs for print-only benchmarks.
    with stdout_path.open("rb") as source:
        shutil.copyfileobj(source, sys.stdout.buffer)
    with stderr_path.open("rb") as source:
        shutil.copyfileobj(source, sys.stderr.buffer)
    sys.stdout.flush()
    sys.stderr.flush()

    record = {
        "schema_version": 1,
        "script": str(script.relative_to(ROOT)),
        "command": command,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": elapsed,
        "exit_code": proc.returncode,
        "git_before": git_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "thread_settings": {
                name: os.environ.get(name)
                for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "logs": {
            "stdout": _file_record(stdout_path),
            "stderr": _file_record(stderr_path),
        },
        "changed_artifacts": _changed_artifacts(before, after),
    }

    manifest = args.manifest.expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        json.dump(record, fh, sort_keys=True)
        fh.write("\n")
    print(f"manifest: {manifest}")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
