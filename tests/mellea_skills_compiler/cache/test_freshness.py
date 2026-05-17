"""Unit tests for mellea_skills_compiler.cache.freshness.

All tests synthesise cache state on disk and inject ``installed_version`` +
``now`` for hermetic, network-free runs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mellea_skills_compiler.cache.freshness import (
    DEFAULT_TTL_DAYS,
    MELLEA_CACHE_TTL_DAYS_ENV,
    is_fresh,
    resolve_ttl_days,
    sha256_of,
)
from mellea_skills_compiler.cache.layout import CacheLayout


# --- Test helpers -----------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_cache(
    tmp_path: Path,
    *,
    version: str = "0.5.0",
    mellea_version_in_manifest: str = "0.5.0",
    fetched_at: str | None = None,
    llms_txt_bytes: bytes = b"# llms.txt\n",
    extra_files: dict[str, bytes] | None = None,
    omit_manifest_files: bool = False,
    corrupt_sha_for: str | None = None,
) -> CacheLayout:
    layout = CacheLayout.for_version(tmp_path, version)
    layout.ensure_dirs()

    layout.llms_txt_path.write_bytes(llms_txt_bytes)
    files = {
        "llms.txt": {
            "sha256": _sha256_bytes(llms_txt_bytes),
            "size": len(llms_txt_bytes),
            "source_url": "https://docs.mellea.ai/llms.txt",
        }
    }
    for rel, body in (extra_files or {}).items():
        path = layout.version_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        files[rel] = {
            "sha256": _sha256_bytes(body),
            "size": len(body),
            "source_url": f"https://docs.mellea.ai/{rel}",
        }

    if corrupt_sha_for:
        files[corrupt_sha_for]["sha256"] = "0" * 64

    manifest: dict = {
        "schema_version": 1,
        "mellea_version": mellea_version_in_manifest,
        "fetched_at": fetched_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"llms_txt_url": "https://docs.mellea.ai/llms.txt"},
    }
    if not omit_manifest_files:
        manifest["files"] = files

    layout.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return layout


# --- TTL resolution ---------------------------------------------------------


class TestResolveTtlDays:
    def test_default(self):
        assert resolve_ttl_days(env={}) == float(DEFAULT_TTL_DAYS)

    def test_env_override(self):
        assert resolve_ttl_days(env={MELLEA_CACHE_TTL_DAYS_ENV: "7"}) == 7.0

    def test_invalid_env_falls_back_to_default(self):
        assert (
            resolve_ttl_days(env={MELLEA_CACHE_TTL_DAYS_ENV: "not-a-number"})
            == float(DEFAULT_TTL_DAYS)
        )

    def test_explicit_wins(self):
        assert (
            resolve_ttl_days(0.5, env={MELLEA_CACHE_TTL_DAYS_ENV: "99"}) == 0.5
        )


# --- sha256 helper ----------------------------------------------------------


class TestSha256Of:
    def test_known_bytes(self, tmp_path: Path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"hello world")
        assert sha256_of(path) == _sha256_bytes(b"hello world")


# --- is_fresh ---------------------------------------------------------------


class TestIsFreshHappyPath:
    def test_fresh_when_everything_aligns(self, tmp_path: Path):
        layout = _write_cache(
            tmp_path,
            extra_files={"api/mellea.stdlib.session.md": b"# session\n"},
        )
        report = is_fresh(layout, installed_version="0.5.0")
        assert report.fresh, report.reasons
        assert report.reasons == []
        assert report.installed_mellea_version == "0.5.0"
        assert report.cached_mellea_version == "0.5.0"
        assert report.age_days is not None and report.age_days >= 0
        assert report.stale is False


class TestIsFreshMissingFiles:
    def test_missing_manifest(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("manifest missing" in r for r in report.reasons)

    def test_manifest_unreadable_json(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        layout.manifest_path.write_text("{ not json", encoding="utf-8")
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("manifest unreadable" in r for r in report.reasons)

    def test_referenced_file_missing(self, tmp_path: Path):
        layout = _write_cache(
            tmp_path,
            extra_files={"api/mellea.stdlib.session.md": b"# session\n"},
        )
        # Remove the file but keep the manifest entry.
        (layout.api_dir / "mellea.stdlib.session.md").unlink()
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("missing file" in r for r in report.reasons)


class TestIsFreshVersionMismatch:
    def test_cached_version_differs(self, tmp_path: Path):
        layout = _write_cache(
            tmp_path, mellea_version_in_manifest="0.99.0-fake"
        )
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("mellea_version mismatch" in r for r in report.reasons)

    def test_mellea_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Force the freshness module's importlib lookup to return None,
        # simulating "Mellea not installed in this environment".
        monkeypatch.setattr(
            "mellea_skills_compiler.cache.freshness.installed_mellea_version",
            lambda: None,
        )
        layout = _write_cache(tmp_path)
        report = is_fresh(layout)
        assert not report.fresh
        assert any("installed Mellea version unknown" in r for r in report.reasons)


class TestIsFreshTtl:
    def test_stale_when_too_old(self, tmp_path: Path):
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(
            timespec="seconds"
        )
        layout = _write_cache(tmp_path, fetched_at=old)
        report = is_fresh(
            layout, installed_version="0.5.0", ttl_days=30
        )
        assert not report.fresh
        assert any("exceeds ttl" in r for r in report.reasons)
        assert report.age_days is not None and report.age_days > 30

    def test_fresh_within_ttl(self, tmp_path: Path):
        recent = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat(timespec="seconds")
        layout = _write_cache(tmp_path, fetched_at=recent)
        report = is_fresh(
            layout, installed_version="0.5.0", ttl_days=30
        )
        assert report.fresh, report.reasons

    def test_unparseable_fetched_at(self, tmp_path: Path):
        layout = _write_cache(tmp_path, fetched_at="not-a-timestamp")
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("fetched_at unparseable" in r for r in report.reasons)


class TestIsFreshSha256:
    def test_sha_mismatch_in_manifest(self, tmp_path: Path):
        layout = _write_cache(
            tmp_path,
            extra_files={"api/mellea.stdlib.session.md": b"# session\n"},
            corrupt_sha_for="api/mellea.stdlib.session.md",
        )
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("sha256 mismatch" in r for r in report.reasons)

    def test_tampered_file_detected(self, tmp_path: Path):
        layout = _write_cache(
            tmp_path,
            extra_files={"api/mellea.stdlib.session.md": b"# session\n"},
        )
        # Tamper with the cached file post-write.
        (layout.api_dir / "mellea.stdlib.session.md").write_bytes(
            b"GARBAGE"
        )
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("sha256 mismatch" in r for r in report.reasons)

    def test_manifest_files_not_object(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        layout.llms_txt_path.write_bytes(b"x")
        layout.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mellea_version": "0.5.0",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "files": ["not", "an", "object"],
                }
            ),
            encoding="utf-8",
        )
        report = is_fresh(layout, installed_version="0.5.0")
        assert not report.fresh
        assert any("manifest.files is not an object" in r for r in report.reasons)
