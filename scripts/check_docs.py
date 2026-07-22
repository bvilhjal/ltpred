#!/usr/bin/env python3
"""Documentation integrity checks that ``mkdocs build --strict`` does not make.

``--strict`` escalates *config* and *nav* problems to errors, but leaves link and
anchor resolution at INFO level -- so it exits 0 while the built site carries
dead cross-links. That gap is not hypothetical: a single stray closing fence in
``docs/algorithm.md`` once turned 60 lines of prose into a code block, deleting a
whole ``##`` heading and breaking four cross-references, and the CI docs job
stayed green for as long as the bug existed.

Two checks, both cheap:

1. **Fence matching** -- track each opening fence until a closing fence uses the
   same marker and at least the same run length. This catches unmatched fences
   without mistaking marker-like text inside a code block for a delimiter.
2. **Anchor resolution** -- run the strict build and fail on the ``anchor``
   diagnostics mkdocs only whispers. Defence in depth: it catches a heading that
   was renamed or removed without updating the links into it.

Run it from the repository root::

    python scripts/check_docs.py

Exits non-zero (and explains what to fix) when either check fails.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

FENCE = re.compile(r"^ {0,3}(?P<run>`{3,}|~{3,})(?P<rest>.*)$")
# mkdocs phrases these as "... contains a link '...', but there is no such
# anchor on this page" / "... the doc '...' does not contain an anchor '...'".
ANCHOR_HINT = re.compile(r"anchor", re.IGNORECASE)
SKIP_DIRS = {".git", "site", "node_modules", ".venv", "venv", "__pycache__"}


def markdown_files(root: pathlib.Path):
    for path in sorted(root.rglob("*.md")):
        if SKIP_DIRS.isdisjoint(part for part in path.relative_to(root).parts):
            yield path


def check_fences(root: pathlib.Path) -> list:
    """Report markdown files with an unmatched fenced code block."""
    problems = []
    for path in markdown_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        opener = None
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = FENCE.match(line)
            if match is None:
                continue

            run = match.group("run")
            marker = run[0]
            rest = match.group("rest")
            if opener is None:
                # CommonMark does not allow a backtick in a backtick fence's
                # info string; such a line is ordinary text, not an opener.
                if marker == "`" and "`" in rest:
                    continue
                opener = (marker, len(run), line_number)
                continue

            open_marker, open_length, _ = opener
            if (marker == open_marker and len(run) >= open_length
                    and not rest.strip()):
                opener = None

        if opener is not None:
            marker, length, line_number = opener
            rel = path.relative_to(root).as_posix()
            problems.append(
                f"{rel}: unmatched {marker * length} code fence opened at "
                f"line {line_number}; its closer must use at least {length} "
                f"{marker} characters")
    return problems


def check_anchors(root: pathlib.Path) -> list:
    """Build the docs strictly and surface the anchor diagnostics it hides."""
    if not (root / "mkdocs.yml").exists():
        return []
    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=root, capture_output=True, text=True, errors="replace")
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        return [f"mkdocs build --strict failed (exit {proc.returncode}):\n{output.strip()}"]
    return [line.strip() for line in output.splitlines() if ANCHOR_HINT.search(line)]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    failures = []

    fences = check_fences(root)
    print(f"fences       : {'FAIL' if fences else 'ok'}", flush=True)
    failures += fences

    anchors = check_anchors(root)
    print(f"anchors      : {'FAIL' if anchors else 'ok'}", flush=True)
    failures += anchors

    if failures:
        # one stream, so the CI log cannot interleave the summary and the detail
        print("\nDocumentation integrity checks failed:\n")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nDocumentation integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
