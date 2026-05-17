"""On-disk layout for the local docs cache.

A single :class:`CacheLayout` instance encapsulates every path the cache
fetcher, freshness checker, and CLI need. The dataclass is intentionally pure
data: it does not touch the filesystem on its own — callers do the
``mkdir(parents=True, exist_ok=True)`` explicitly so test fixtures and CI
scripts can audit exactly when directories appear.

URL → file-path mapping
-----------------------

For an llms.txt-listed URL such as
``https://docs.mellea.ai/api/mellea.stdlib.session.md`` the cache stores the
content at::

    <cache_root>/<mellea_version>/api/mellea.stdlib.session.md

The URL path (everything after ``/api/``) is mirrored verbatim into the
``api/`` subdirectory, so URLs that contain extra path segments (e.g.
``api/requirements/md.md``) round-trip to ``api/requirements/md.md`` on disk.

Plan reference: section 6.1 of the introspection-driven descriptor compiler
plan. The plan suggests ``~/.mellea-skills-compiler/cache/``; this
implementation defaults to a repo-local ``.mellea-cache/`` so the cache is
inspectable next to the project that produced it and can be cleaned with a
single ``rm -rf``. Either location is supported via :func:`resolve_cache_root`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Directory name (relative to the repo root) used when no override is given.
DEFAULT_CACHE_DIR: str = ".mellea-cache"

#: Environment variable that, when set, overrides the default cache root.
MELLEA_CACHE_DIR_ENV: str = "MELLEA_CACHE_DIR"


def resolve_cache_root(
    explicit: Optional[Path | str] = None,
    *,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Path:
    """Resolve the cache root directory.

    Precedence (highest first):

    1. ``explicit`` argument (CLI ``--cache-dir`` flag).
    2. ``MELLEA_CACHE_DIR`` environment variable.
    3. ``<cwd>/.mellea-cache``.

    The path is returned as an absolute :class:`Path`. The directory is **not**
    created — that's the caller's responsibility.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env_map = env if env is not None else os.environ
    env_value = env_map.get(MELLEA_CACHE_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()

    base = cwd if cwd is not None else Path.cwd()
    return (base / DEFAULT_CACHE_DIR).resolve()


@dataclass(frozen=True)
class CacheLayout:
    """All paths for one Mellea-version's cache slice.

    Construction is cheap (pure path arithmetic). Use :meth:`for_version` to
    build an instance from a cache root + Mellea version string.
    """

    #: Root directory containing all version subdirectories.
    cache_root: Path
    #: Mellea version string (e.g. ``"0.5.0"``); becomes a subdirectory name.
    mellea_version: str

    # --- Derived paths ------------------------------------------------------

    @property
    def version_dir(self) -> Path:
        """``<cache_root>/<mellea_version>/`` — owns one version's artefacts."""
        return self.cache_root / self.mellea_version

    @property
    def manifest_path(self) -> Path:
        """The manifest.json describing what's cached for this version."""
        return self.version_dir / "manifest.json"

    @property
    def llms_txt_path(self) -> Path:
        """Cached snapshot of ``https://docs.mellea.ai/llms.txt``."""
        return self.version_dir / "llms.txt"

    @property
    def api_dir(self) -> Path:
        """Directory holding per-module ``.md`` pages."""
        return self.version_dir / "api"

    # --- Methods ------------------------------------------------------------

    @classmethod
    def for_version(
        cls,
        cache_root: Path | str,
        mellea_version: str,
    ) -> "CacheLayout":
        """Build a layout from a cache root + Mellea version string."""
        return cls(
            cache_root=Path(cache_root).expanduser().resolve(),
            mellea_version=mellea_version,
        )

    def api_doc_path(self, url_or_relpath: str) -> Path:
        """Resolve an llms.txt URL (or ``api/*`` relpath) to its on-disk path.

        Accepts either:

        - A full URL like ``https://docs.mellea.ai/api/foo/bar.md``, or
        - A relative path like ``api/foo/bar.md`` or ``foo/bar.md``.

        Returns the absolute path under :attr:`api_dir`. The path is **not**
        guaranteed to exist.
        """
        relpath = _api_relpath(url_or_relpath)
        return self.api_dir / relpath

    def ensure_dirs(self) -> None:
        """Create the version + api directories. Idempotent."""
        self.api_dir.mkdir(parents=True, exist_ok=True)


def _api_relpath(url_or_relpath: str) -> str:
    """Extract the ``api/`` sub-path from a URL or relative path.

    Examples
    --------

    >>> _api_relpath("https://docs.mellea.ai/api/mellea.stdlib.session.md")
    'mellea.stdlib.session.md'
    >>> _api_relpath("api/requirements/md.md")
    'requirements/md.md'
    >>> _api_relpath("requirements/md.md")
    'requirements/md.md'
    """
    s = url_or_relpath
    if "://" in s:
        # Strip scheme + host; keep the path component.
        s = s.split("://", 1)[1]
        s = s.split("/", 1)[1] if "/" in s else ""
    s = s.lstrip("/")
    if s.startswith("api/"):
        s = s[len("api/"):]
    return s
