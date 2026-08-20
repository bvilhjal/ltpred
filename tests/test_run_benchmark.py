import zipfile

from benchmarks import run_benchmark


def test_dirty_source_snapshot_persists_patch_and_untracked_files(tmp_path, monkeypatch):
    untracked = tmp_path / "research" / "new_method.py"
    untracked.parent.mkdir()
    untracked.write_text("VALUE = 3\n", encoding="utf-8")
    patch = b"diff --git a/model.py b/model.py\n"

    monkeypatch.setattr(run_benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_benchmark, "_git",
        lambda *args, **kwargs: patch if args == ("diff", "--binary", "HEAD") else None,
    )
    state = {"untracked_source_files": {"research/new_method.py": {}}}

    record = run_benchmark._source_snapshot("run-1", state, tmp_path / "snapshots")

    patch_path = tmp_path / record["tracked_patch"]["path"]
    archive_path = tmp_path / record["untracked_sources"]["path"]
    assert patch_path.read_bytes() == patch
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["research/new_method.py"]
        assert archive.read("research/new_method.py") == b"VALUE = 3\n"


def test_clean_source_snapshot_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(run_benchmark, "_git", lambda *args, **kwargs: b"")

    record = run_benchmark._source_snapshot(
        "run-2", {"untracked_source_files": {}}, tmp_path / "snapshots")

    assert record == {}
    assert not (tmp_path / "snapshots").exists()


def test_env_mismatch_detects_a_different_interpreter_stack():
    # The manifest always *recorded* the interpreter, but recording is not
    # checking: a bare `python` resolves to whatever the PATH offers, and on a
    # typical dev box that is a different Python and NumPy from the one the
    # committed artifacts were produced under. Timings and thread-sensitive
    # results are then incomparable with nothing visibly wrong.
    reference = {
        "python": "3.14",
        "free_threading": True,
        "packages": {"numpy": "2.4", "scipy": "1.18", "numba": "0.66"},
    }
    assert run_benchmark._env_mismatches(reference=run_benchmark._env_fingerprint()) == []

    wrong = dict(reference, python="3.10", free_threading=False,
                 packages={**reference["packages"], "numpy": "1.26"})
    found = " ".join(run_benchmark._env_mismatches(reference=wrong))
    assert "python" in found and "free-threading" in found and "numpy" in found


def test_reference_env_file_matches_the_running_stack():
    # The committed declaration must describe the environment the project
    # actually benchmarks in, or the guard protects nothing.
    import json
    assert run_benchmark.REFERENCE_ENV.exists()
    ref = json.loads(run_benchmark.REFERENCE_ENV.read_text(encoding="utf-8"))
    for key in ("python", "free_threading", "packages"):
        assert key in ref
