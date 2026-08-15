import re
from datetime import datetime
from pathlib import Path

import ltpred


ROOT = Path(__file__).resolve().parents[1]


def _match(path, pattern):
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"version marker missing from {path}"
    return match.group(1)


def test_version_is_consistent_across_release_metadata():
    version = ltpred.__version__
    assert _match("CITATION.cff", r"^version:\s*[\"']?([^\"'\s]+)") == version
    assert _match("index.html", r'<span class="chip">v<b>([^<]+)</b>') == version
    assert _match("report/ltpred_methods.tex", r"\\quad v([^}]+)}}") == version
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(version)}(?:\s|$)", changelog, re.MULTILINE)


def test_release_date_is_consistent_across_release_metadata():
    version = re.escape(ltpred.__version__)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^## {version} — (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE)
    assert match is not None, "dated release heading missing from CHANGELOG.md"
    release_date = match.group(1)

    assert _match("CITATION.cff", r'^date-released:\s*["\']?([^"\'\s]+)') == release_date
    report_date = _match("report/ltpred_methods.tex", r"\\date\{([^}]+)\}")
    assert datetime.strptime(report_date, "%d %B %Y").date().isoformat() == release_date
