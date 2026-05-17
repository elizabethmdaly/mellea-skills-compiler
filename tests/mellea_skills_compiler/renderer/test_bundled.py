"""Tests for the renderer's v0.3 ``bundled_resources[]`` handling.

Covers:
- copy-scheduling shape
- pyproject.toml package-data update
- binary-resource deferral with warning (plan §11.5)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.bundled import (
    BundledResourceCopy,
    is_binary_resource,
    render_bundled_resources,
    update_pyproject_package_data,
)


# --- Copy scheduling ------------------------------------------------------


def test_render_schedules_directory_copy(tmp_path: Path):
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "owasp.md").write_text("hello")
    desc = {
        "bundled_resources": [
            {"source_dir": "references", "package_dir": "references"}
        ]
    }
    art = render_bundled_resources(desc, skill_root=tmp_path)
    assert len(art.copies) == 1
    copy = art.copies[0]
    assert isinstance(copy, BundledResourceCopy)
    assert copy.source_dir == "references"
    assert copy.package_dir == "references"
    assert copy.is_package_data is True
    assert "references" in art.package_data_dirs


def test_render_respects_is_package_data_false(tmp_path: Path):
    (tmp_path / "scripts").mkdir()
    desc = {
        "bundled_resources": [
            {"source_dir": "scripts", "package_dir": "scripts",
             "is_package_data": False}
        ]
    }
    art = render_bundled_resources(desc, skill_root=tmp_path)
    assert art.copies[0].is_package_data is False
    assert "scripts" not in art.package_data_dirs


def test_render_handles_missing_skill_root_gracefully():
    """skill_root=None means we skip binary scanning but still schedule the
    copy entry."""
    desc = {
        "bundled_resources": [
            {"source_dir": "references", "package_dir": "references"}
        ]
    }
    art = render_bundled_resources(desc, skill_root=None)
    assert len(art.copies) == 1
    assert art.copies[0].deferred_binaries == []


def test_render_rejects_missing_source_dir():
    with pytest.raises(RendererError, match="missing source_dir"):
        render_bundled_resources({"bundled_resources": [{"package_dir": "x"}]})


def test_render_rejects_non_list_bundled():
    with pytest.raises(RendererError, match="must be a list"):
        render_bundled_resources({"bundled_resources": "not-a-list"})


# --- Binary deferral ------------------------------------------------------


def test_is_binary_detects_known_extensions(tmp_path: Path):
    assert is_binary_resource(tmp_path / "foo.png")
    assert is_binary_resource(tmp_path / "foo.pdf")
    assert is_binary_resource(tmp_path / "foo.zip")
    assert not is_binary_resource(tmp_path / "foo.md")
    assert not is_binary_resource(tmp_path / "foo.txt")


def test_is_binary_detects_large_files(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 100)
    assert not is_binary_resource(big, max_size=1000)
    assert is_binary_resource(big, max_size=50)


def test_binary_resource_emits_warning_and_records_todo(tmp_path: Path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "diagram.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "refs" / "ok.md").write_text("# ok")
    desc = {"bundled_resources": [{"source_dir": "refs", "package_dir": "refs"}]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        art = render_bundled_resources(desc, skill_root=tmp_path)
    msgs = [str(x.message) for x in w]
    assert any("diagram.png" in m and "deferred" in m for m in msgs)
    assert any("diagram.png" in t for t in art.pyproject_todo)
    # Binary appears in the per-copy deferred list.
    assert any("diagram.png" in d for d in art.copies[0].deferred_binaries)


# --- pyproject.toml mutation ----------------------------------------------


def test_update_pyproject_appends_block_when_absent():
    src = '[project]\nname = "foo"\n'
    out = update_pyproject_package_data(src, "foo", {"references"})
    assert "[tool.setuptools.package-data]" in out
    assert "foo = " in out
    assert "references/**/*" in out


def test_update_pyproject_inserts_into_existing_block():
    src = (
        '[project]\nname = "foo"\n\n'
        '[tool.setuptools.package-data]\n'
        'foo = ["existing/*"]\n'
        '[other.section]\n'
    )
    out = update_pyproject_package_data(src, "foo", {"references"})
    assert "[tool.setuptools.package-data]" in out
    assert "[other.section]" in out
    assert "references/**/*" in out


def test_update_pyproject_no_dirs_is_no_op():
    src = '[project]\nname = "foo"\n'
    out = update_pyproject_package_data(src, "foo", set())
    assert out == src
