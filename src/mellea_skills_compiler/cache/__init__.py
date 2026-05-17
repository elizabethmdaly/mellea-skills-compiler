"""Local docs cache — Phase 1.C of the introspection-driven descriptor compiler.

The cache amortises ``llms.txt`` and per-module markdown fetches across many
compile invocations. It is keyed by the installed Mellea version so a Mellea
bump automatically lands in a fresh subdirectory and the previous version's
artefacts remain inspectable side-by-side.

The on-disk layout (see :mod:`mellea_skills_compiler.cache.layout` for the
authoritative dataclass) is:

::

    .mellea-cache/
      <mellea_version>/
        manifest.json     # version, timestamps, sha256 of every file
        llms.txt          # raw bytes of https://docs.mellea.ai/llms.txt
        api/
          mellea.stdlib.session.md
          mellea.stdlib.requirements.md
          ...

Plan reference: ``melleafy-handoff/analyses/2026-05-16-introspection-driven-descriptor-compiler-plan.md``
section 6. The plan suggests ``~/.mellea-skills-compiler/cache/`` for the root;
this implementation prefers a repo-local ``.mellea-cache/`` so the cache is
visible to project tooling and the developer's editor without crossing the
``$HOME`` boundary. Override with ``MELLEA_CACHE_DIR`` env var or
``--cache-dir`` CLI flag.

Public API
----------

``fetch_index``      — fetch & write llms.txt; returns parsed API URLs.
``fetch_module_doc`` — fetch & write a single per-module markdown page.
``is_fresh``         — return a :class:`FreshnessReport` for the given dir.
``refresh``          — fetch llms.txt + all linked module docs in one go.
``CacheLayout``      — dataclass describing where everything lives on disk.
"""

from __future__ import annotations

from mellea_skills_compiler.cache.fetcher import (
    LLMS_TXT_URL,
    MELLEA_DOCS_BASE_URL,
    FetchOutcome,
    RefreshResult,
    fetch_index,
    fetch_module_doc,
    refresh,
)
from mellea_skills_compiler.cache.freshness import (
    FreshnessReport,
    is_fresh,
)
from mellea_skills_compiler.cache.layout import (
    DEFAULT_CACHE_DIR,
    MELLEA_CACHE_DIR_ENV,
    CacheLayout,
    resolve_cache_root,
)

__all__ = [
    "CacheLayout",
    "DEFAULT_CACHE_DIR",
    "FetchOutcome",
    "FreshnessReport",
    "LLMS_TXT_URL",
    "MELLEA_CACHE_DIR_ENV",
    "MELLEA_DOCS_BASE_URL",
    "RefreshResult",
    "fetch_index",
    "fetch_module_doc",
    "is_fresh",
    "refresh",
    "resolve_cache_root",
]
