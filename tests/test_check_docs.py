from scripts.check_docs import check_fences


def _check(tmp_path, markdown):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "example.md").write_text(markdown, encoding="utf-8")
    return check_fences(tmp_path)


def test_mixed_delimiter_fences_pair_independently(tmp_path):
    markdown = "```python\npass\n```\n\n~~~text\nvalue\n~~~\n"
    assert _check(tmp_path, markdown) == []


def test_different_marker_is_literal_inside_fence(tmp_path):
    markdown = "```text\n~~~\n```\n"
    assert _check(tmp_path, markdown) == []


def test_short_closer_does_not_close_fence(tmp_path):
    problems = _check(tmp_path, "````text\nvalue\n```\n")
    assert len(problems) == 1
    assert "opened at line 1" in problems[0]


def test_longer_closer_is_valid(tmp_path):
    assert _check(tmp_path, "```text\nvalue\n````\n") == []


def test_unmatched_fence_is_reported(tmp_path):
    problems = _check(tmp_path, "~~~text\nvalue\n")
    assert problems == [
        "docs/example.md: unmatched ~~~ code fence opened at line 1; "
        "its closer must use at least 3 ~ characters"
    ]
