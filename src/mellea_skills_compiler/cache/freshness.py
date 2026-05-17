"""Freshness / staleness detection for the local docs cache.

A cache slice (``<cache_root>/<mellea_version>/``) is **fresh** iff:

1. Its ``manifest.json`` lists a ``mellea_version`` equal to the currently
   installed Mellea version (``importlib.metadata.version('mellea')``).
2. Every file recorded in the manifest exists and its sha256 still matches
   the manifest entry (detects partial-write corruption or hand-tampering).
3. The ``fetched_at`` timestamp is younger than ``staleness_ttl_days``
   (default 30 days; override via the ``MELLEA_CACHE_TTL_DAYS`` env var or
   the ``ttl_days`` argument).

If any check fails the cache layer reports ``stale`` and the caller chooses
whether to call :func:`mellea_skills_compiler.cache.refresh` or proceed
without a fresh cache. **The library never refetches automatically** — that
is a CLI / CI concern.

The :class:`FreshnessReport` returned by :func:`is_fresh` enumerates each
reason for staleness so the CLI can render a precise diagnostic.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mellea_skills_compiler.cache.layout import CacheLayout

#: Environment variable for overriding the staleness TTL.
MELLEA_CACHE_TTL_DAYS_ENV = "MELLEA_CACHE_TTL_DAYS"

#: Default staleness window. Mellea releases are infrequent enough that 30 days
#: is conservative; CI pipelines should bump this down to a few hours if they
#: want stricter guarantees.
DEFAULT_TTL_DAYS = 30


@dataclass
class FreshnessReport:
    """The outcome of a freshness check, with structured reasons.

    ``fresh`` is True iff ``reasons`` is empty. The CLI renders ``reasons``
    line-by-line so the user can see exactly why a cache slice is stale.
    """

    fresh: bool
    cache_layout: CacheLayout
    installed_mellea_version: Optional[str]
    cached_mellea_version: Optional[str] = None
    fetched_at: Optional[str] = None
    age_days: Optional[float] = None
    ttl_days: float = float(DEFAULT_TTL_DAYS)
    reasons: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return not self.fresh


def resolve_ttl_days(
    explicit: Optional[float] = None,
    *,
    env: Optional[dict[str, str]] = None,
) -> float:
    """Resolve the freshness TTL (days)."""
    if explicit is not None:
        return float(explicit)
    env_map = env if env is not None else os.environ
    value = env_map.get(MELLEA_CACHE_TTL_DAYS_ENV)
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return float(DEFAULT_TTL_DAYS)


def installed_mellea_version() -> Optional[str]:
    """Return the installed Mellea version, or None if Mellea is not present."""
    try:
        return importlib.metadata.version("mellea")
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_fresh(
    layout: CacheLayout,
    *,
    ttl_days: Optional[float] = None,
    now: Optional[datetime] = None,
    installed_version: Optional[str] = None,
) -> FreshnessReport:
    """Return a :class:`FreshnessReport` for the given cache slice.

    Parameters
    ----------
    layout
        Where to look on disk.
    ttl_days
        Override the default staleness window. Defaults to
        ``MELLEA_CACHE_TTL_DAYS`` env var or 30 days.
    now
        Inject a "now" for tests. Defaults to ``datetime.now(timezone.utc)``.
    installed_version
        Inject the installed Mellea version for tests. Defaults to the value
        from ``importlib.metadata.version('mellea')``.
    """
    resolved_ttl = resolve_ttl_days(ttl_days)
    resolved_now = now if now is not None else datetime.now(timezone.utc)
    resolved_installed = (
        installed_version
        if installed_version is not None
        else installed_mellea_version()
    )

    report = FreshnessReport(
        fresh=False,
        cache_layout=layout,
        installed_mellea_version=resolved_installed,
        ttl_days=resolved_ttl,
    )

    # 1. Manifest must exist.
    if not layout.manifest_path.is_file():
        report.reasons.append(f"manifest missing: {layout.manifest_path}")
        return report

    try:
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.reasons.append(f"manifest unreadable: {type(exc).__name__}: {exc}")
        return report

    cached_version = manifest.get("mellea_version")
    report.cached_mellea_version = cached_version
    report.fetched_at = manifest.get("fetched_at")

    # 2. Version match.
    if resolved_installed is None:
        report.reasons.append(
            "installed Mellea version unknown (package not importable)"
        )
    elif cached_version != resolved_installed:
        report.reasons.append(
            f"mellea_version mismatch: cached={cached_version!r}, "
            f"installed={resolved_installed!r}"
        )

    # 3. Timestamp.
    if report.fetched_at:
        try:
            fetched_dt = datetime.fromisoformat(report.fetched_at)
            if fetched_dt.tzinfo is None:
                fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
            age = resolved_now - fetched_dt
            age_days = age.total_seconds() / 86400.0
            report.age_days = age_days
            if age_days > resolved_ttl:
                report.reasons.append(
                    f"age {age_days:.1f}d exceeds ttl {resolved_ttl:.1f}d"
                )
        except (TypeError, ValueError):
            report.reasons.append(f"fetched_at unparseable: {report.fetched_at!r}")
    else:
        report.reasons.append("fetched_at missing from manifest")

    # 4. Per-file sha256 integrity.
    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        report.reasons.append("manifest.files is not an object")
    else:
        for relpath, entry in files.items():
            if not isinstance(entry, dict):
                report.reasons.append(f"manifest.files[{relpath!r}] is not an object")
                continue
            expected_sha = entry.get("sha256")
            file_path = layout.version_dir / relpath
            if not file_path.is_file():
                report.reasons.append(f"missing file: {relpath}")
                continue
            try:
                actual_sha = sha256_of(file_path)
            except OSError as exc:
                report.reasons.append(f"cannot read {relpath}: {exc}")
                continue
            if expected_sha != actual_sha:
                report.reasons.append(
                    f"sha256 mismatch: {relpath} "
                    f"(expected {expected_sha[:12] if expected_sha else '?'}…, "
                    f"got {actual_sha[:12]}…)"
                )

    report.fresh = not report.reasons
    return report
