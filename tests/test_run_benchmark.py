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
