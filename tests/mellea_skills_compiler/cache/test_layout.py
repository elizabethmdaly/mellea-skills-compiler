"""Unit tests for mellea_skills_compiler.cache.layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.cache.layout import (
    DEFAULT_CACHE_DIR,
    MELLEA_CACHE_DIR_ENV,
    CacheLayout,
    _api_relpath,
    resolve_cache_root,
)


class TestApiRelpath:
    def test_full_url(self):
        assert (
            _api_relpath("https://docs.mellea.ai/api/mellea.stdlib.session.md")
            == "mellea.stdlib.session.md"
        )

    def test_nested_full_url(self):
        assert (
            _api_relpath("https://docs.mellea.ai/api/requirements/md.md")
            == "requirements/md.md"
        )

    def test_relpath_with_api_prefix(self):
        assert _api_relpath("api/requirements/md.md") == "requirements/md.md"

    def test_relpath_without_api_prefix(self):
        assert _api_relpath("requirements/md.md") == "requirements/md.md"

    def test_leading_slash(self):
        assert _api_relpath("/api/foo.md") == "foo.md"


class TestResolveCacheRoot:
    def test_explicit_wins(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        env_dir = tmp_path / "envdir"
        out = resolve_cache_root(explicit, env={MELLEA_CACHE_DIR_ENV: str(env_dir)})
        assert out == explicit.resolve()

    def test_env_used_when_no_explicit(self, tmp_path: Path):
        env_dir = tmp_path / "envdir"
        out = resolve_cache_root(env={MELLEA_CACHE_DIR_ENV: str(env_dir)})
        assert out == env_dir.resolve()

    def test_default_uses_cwd(self, tmp_path: Path):
        out = resolve_cache_root(env={}, cwd=tmp_path)
        assert out == (tmp_path / DEFAULT_CACHE_DIR).resolve()

    def test_default_dir_name(self):
        assert DEFAULT_CACHE_DIR == ".mellea-cache"

    def test_expanduser(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        out = resolve_cache_root("~/mycache", env={})
        assert out == (tmp_path / "mycache").resolve()


class TestCacheLayout:
    def test_for_version_builds_paths(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        assert layout.cache_root == tmp_path.resolve()
        assert layout.mellea_version == "0.5.0"
        assert layout.version_dir == tmp_path.resolve() / "0.5.0"
        assert layout.manifest_path == tmp_path.resolve() / "0.5.0" / "manifest.json"
        assert layout.llms_txt_path == tmp_path.resolve() / "0.5.0" / "llms.txt"
        assert layout.api_dir == tmp_path.resolve() / "0.5.0" / "api"

    def test_api_doc_path_from_url(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        p = layout.api_doc_path(
            "https://docs.mellea.ai/api/mellea.stdlib.session.md"
        )
        assert p == tmp_path.resolve() / "0.5.0" / "api" / "mellea.stdlib.session.md"

    def test_api_doc_path_nested(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        p = layout.api_doc_path("https://docs.mellea.ai/api/requirements/md.md")
        assert p == tmp_path.resolve() / "0.5.0" / "api" / "requirements" / "md.md"

    def test_ensure_dirs_creates_api(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        assert not layout.api_dir.exists()
        layout.ensure_dirs()
        assert layout.api_dir.is_dir()
        # Idempotent.
        layout.ensure_dirs()
        assert layout.api_dir.is_dir()

    def test_frozen(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        with pytest.raises(Exception):
            layout.mellea_version = "0.6.0"  # type: ignore[misc]
