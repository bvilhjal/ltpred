import json
import os

import pytest

from benchmarks import run_benchmark


def test_source_state_does_not_exclude_tracked_repo_inputs(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    input_path = root / "data" / "cohort.bed"
    calls = []

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[:3] == ("ls-files", "--cached", "--"):
            return "data/cohort.bed\n"
        calls.append(args)
        return " M data/cohort.bed\n"

    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "_git", fake_git)
    state = run_benchmark._source_state([input_path])

    assert state == {
        "commit": "a" * 40,
        "dirty": True,
        "status": [" M data/cohort.bed"],
    }
    assert ":(exclude,literal)data/cohort.bed" not in calls[0]


def test_source_state_excludes_declared_untracked_repo_inputs(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    input_path = root / "data" / "cohort.bed"
    calls = []

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args[:3] == ("ls-files", "--cached", "--"):
            return ""
        calls.append(args)
        return ""

    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "_git", fake_git)
    state = run_benchmark._source_state([input_path])

    assert state == {"commit": "a" * 40, "dirty": False, "status": []}
    assert ":(exclude,literal)data/cohort.bed" in calls[0]


def test_resolve_script_rejects_nonbenchmark_and_path_escape(tmp_path, monkeypatch):
    here = tmp_path / "benchmarks"
    here.mkdir()
    (here / "bench_ok.py").write_text("", encoding="utf-8")
    (here / "helper.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_benchmark, "HERE", here)

    assert run_benchmark._resolve_script("bench_ok.py") == here / "bench_ok.py"
    for invalid in ("helper.py", "../bench_ok.py", "missing.py"):
        with pytest.raises(ValueError, match=r"bench_\*\.py"):
            run_benchmark._resolve_script(invalid)


def test_runner_records_commit_environment_and_artifact_hash(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "benchmarks"
    here.mkdir(parents=True)
    (root / "ltpred").mkdir()
    (root / "ltpred" / "__init__.py").write_text(
        '__version__ = "9.8.7"\n', encoding="utf-8")
    script = here / "bench_smoke.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.csv').write_text('x\\n1\\n')\n",
        encoding="utf-8",
    )
    (here / "bench_smoke.csv").write_text("x\n1\n", encoding="utf-8")
    os.utime(here / "bench_smoke.csv", ns=(1_000_000_000, 1_000_000_000))
    input_path = root / "input.dat"
    input_path.write_bytes(b"input")
    manifest = here / "manifest.jsonl"
    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(run_benchmark, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": False, "status": []},
    )

    assert run_benchmark.main(
        ["--artifact", "bench_smoke.csv", "--input", str(input_path),
         "bench_smoke.py"]) == 0
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["source"]["commit"] == "a" * 40
    assert record["source"]["dirty"] is False
    assert record["source"]["status"] == []
    assert record["environment"]["packages"]["ltpred"] == "9.8.7"
    assert record["artifacts"][0]["path"] == "bench_smoke.csv"
    assert record["artifacts"][0]["touched"] is True
    assert len(record["artifacts"][0]["sha256"]) == 64
    assert record["inputs"][0]["sha256"] == (
        "c96c6d5be8d08a12e7b5cdc1b207fa6b2430974c86803d8891675e76fd992c20")
    assert record["source"]["unchanged_after_run"] is True
    assert record["changed_artifacts"] == []
    assert record["wrapper_exit_code"] == 0
    assert record["provenance_errors"] == []


def test_runner_refuses_dirty_source_by_default(tmp_path, monkeypatch):
    here = tmp_path / "benchmarks"
    here.mkdir()
    (here / "bench_smoke.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": True,
            "status": [" M ltpred/estimate.py"]},
    )

    with pytest.raises(SystemExit, match="source tree is dirty"):
        run_benchmark.main(["--artifact", "bench_smoke.csv", "bench_smoke.py"])


def test_runner_rejects_untouched_preexisting_artifact(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "benchmarks"
    here.mkdir(parents=True)
    (root / "ltpred").mkdir()
    (root / "ltpred" / "__init__.py").write_text(
        '__version__ = "9.8.7"\n', encoding="utf-8")
    (here / "bench_nop.py").write_text("", encoding="utf-8")
    (here / "bench_stale.csv").write_text("stale\n", encoding="utf-8")
    manifest = here / "manifest.jsonl"
    monkeypatch.setattr(run_benchmark, "ROOT", root)
    monkeypatch.setattr(run_benchmark, "HERE", here)
    monkeypatch.setattr(run_benchmark, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(
        run_benchmark, "_source_state",
        lambda ignored_paths=(): {
            "commit": "a" * 40, "dirty": False, "status": []},
    )

    assert run_benchmark.main(
        ["--artifact", "bench_stale.csv", "bench_nop.py"]) == 2
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["untouched_artifacts"] == ["bench_stale.csv"]
    assert record["artifacts"][0]["touched"] is False
    assert record["wrapper_exit_code"] == 2
