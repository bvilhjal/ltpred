"""The tutorial page must run, and must print what it says it prints.

``docs/tutorial.md`` is the one documentation page written as a sequence rather
than as a set of fragments, so it can be executed: this test extracts its
``python`` blocks in order, runs them in a single shared namespace, and compares
each block's stdout with the ``text`` block that follows it. That makes two
distinct failures visible -- code that no longer runs after a signature change,
and prose that quotes numbers the code no longer produces -- which is the drift
``docs/reviews/REVIEW_2026-09d.md`` (T1-2) found unguarded elsewhere in the docs.
"""

import contextlib
import io
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = REPO / "docs" / "tutorial.md"

_BLOCK = re.compile(r"```(python|text)\n(.*?)```", re.S)


def _blocks():
    """``(kind, body)`` for every fenced block, in document order."""
    return [(m.group(1), m.group(2)) for m in _BLOCK.finditer(
        TUTORIAL.read_text(encoding="utf-8"))]


def _normalise(text):
    return [line.rstrip() for line in text.strip().splitlines() if line.strip()]


def test_tutorial_exists_and_has_executable_steps():
    blocks = _blocks()
    code = [body for kind, body in blocks if kind == "python"]
    assert len(code) >= 5, "the tutorial lost its step blocks"
    assert any(body.strip().startswith("import numpy") for body in code), \
        "the first step should be self-contained"


def test_tutorial_blocks_run_in_order_and_print_the_documented_output():
    """Each python block runs against the namespace the earlier blocks built.

    A block followed by a ```text``` block must print exactly those lines; a
    block with no following text block is only required to execute. The shared
    namespace is the point -- it is what makes the page a sequence rather than a
    pile of fragments, so a block that silently started depending on an
    undefined name fails here instead of in a reader's REPL.
    """
    blocks = _blocks()
    namespace = {}
    checked = 0
    for index, (kind, body) in enumerate(blocks):
        if kind != "python":
            continue
        expected = None
        if index + 1 < len(blocks) and blocks[index + 1][0] == "text":
            expected = blocks[index + 1][1]
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(body, f"<tutorial block {index}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report any failure as a doc bug
            pytest.fail(f"docs/tutorial.md block {index} raised "
                        f"{type(exc).__name__}: {exc}\n{body}")
        if expected is None:
            continue
        got = _normalise(stream.getvalue())
        want = _normalise(expected)
        assert got == want, (
            f"docs/tutorial.md block {index} printed something other than the "
            f"output quoted below it.\n  expected: {want}\n  got:      {got}")
        checked += 1
    # steps 1-5 each quote output; if that ever drops, the page has lost its
    # self-checking property rather than merely changing
    assert checked >= 5, f"only {checked} blocks quote their output"


def test_tutorial_only_uses_the_public_api():
    """The page is what an installed user can run, so no private or
    checkout-only imports: ``examples/`` and ``benchmarks/`` ship in neither the
    wheel nor the sdist (REVIEW_2026-09d T2-2)."""
    imported = set()
    for kind, body in _blocks():
        if kind != "python":
            continue
        for match in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", body, re.M):
            imported.add(match.group(1).split(".")[0])
    assert imported <= {"numpy", "ltpred"}, \
        f"the tutorial imports something a pip install will not provide: {imported}"
    text = TUTORIAL.read_text(encoding="utf-8")
    for forbidden in ("from examples", "from benchmarks", "import examples",
                      "research."):
        assert forbidden not in text, f"the tutorial points at {forbidden!r}"
