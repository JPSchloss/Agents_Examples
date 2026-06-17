"""Unit tests for the path sandbox — the single most security-critical helper."""

from pathlib import Path

import pytest

from data_agent.tools._paths import PathNotAllowed, safe_resolve


def test_allows_simple_relative_path(tmp_path: Path):
    assert safe_resolve("clean.py", tmp_path) == (tmp_path / "clean.py").resolve()


def test_allows_nested_relative_path(tmp_path: Path):
    assert safe_resolve("sub/dir/file.csv", tmp_path) == (tmp_path / "sub/dir/file.csv").resolve()


def test_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(PathNotAllowed):
        safe_resolve("../escape.txt", tmp_path)


def test_rejects_deep_traversal(tmp_path: Path):
    with pytest.raises(PathNotAllowed):
        safe_resolve("a/b/../../../etc/passwd", tmp_path)


def test_rejects_absolute_path_outside_roots(tmp_path: Path):
    with pytest.raises(PathNotAllowed):
        safe_resolve("/etc/passwd", tmp_path)


def test_second_allowed_root_is_honored(tmp_path: Path):
    raw = tmp_path / "raw"
    work = tmp_path / "work"
    raw.mkdir()
    work.mkdir()
    # Resolving relative to `work` should be fine; a path landing in `raw` is allowed too.
    assert safe_resolve("x.csv", work, raw) == (work / "x.csv").resolve()
