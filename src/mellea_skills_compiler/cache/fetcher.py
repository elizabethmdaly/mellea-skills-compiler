"""Network fetchers for llms.txt and per-module markdown pages.

Stdlib-only HTTP via :mod:`urllib.request`. Hard-coded base URL
(``https://docs.mellea.ai``) — refusing to talk to other hosts is the
simplest defence against the cache layer being repointed at arbitrary
URLs by config.

Retry / timeout policy
----------------------

- 30s timeout per HTTP request.
- 2 attempts with 2s linear backoff between attempts.
- A 404 on a per-module page logs a warning to stderr and records the URL
  in the manifest's ``missing_urls`` list; the populate run continues so
  one removed page doesn't abort the whole refresh.

Manifest schema
---------------

Written to ``<version_dir>/manifest.json``::

    {
      "schema_version": 1,
      "mellea_version": "0.5.0",
      "fetched_at": "2026-05-16T12:34:56+00:00",
      "source": {
        "llms_txt_url": "https://docs.mellea.ai/llms.txt",
        "docs_base_url": "https://docs.mellea.ai"
      },
      "files": {
        "llms.txt": {"sha256": "...", "size": 4321, "source_url": "..."},
        "api/mellea.stdlib.session.md": {"sha256": "...", "size": 1234, "source_url": "..."},
        ...
      },
      "missing_urls": ["https://docs.mellea.ai/api/foo.md", ...],
      "module_urls": [
        ["mellea.stdlib.session", "https://docs.mellea.ai/api/mellea.stdlib.session.md"],
        ...
      ]
    }
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mellea_skills_compiler.cache.freshness import (
    installed_mellea_version,
    sha256_of,
)
from mellea_skills_compiler.cache.layout import CacheLayout

#: The only host the cache will fetch from.
MELLEA_DOCS_BASE_URL = "https://docs.mellea.ai"

#: The well-known llms.txt URL.
LLMS_TXT_URL = f"{MELLEA_DOCS_BASE_URL}/llms.txt"

#: Manifest schema version. Bump when the manifest shape changes
#: incompatibly so freshness checks can refuse mismatched older caches.
MANIFEST_SCHEMA_VERSION = 1

#: Per-request HTTP timeout in seconds.
HTTP_TIMEOUT_SECONDS = 30

#: How many times to attempt a fetch before giving up.
HTTP_MAX_ATTEMPTS = 2

#: Linear backoff between retries (seconds).
HTTP_RETRY_BACKOFF_SECONDS = 2.0

# Pattern matching the same `api/*.md` rows the introspection spike parses
# (see scripts/introspect_mellea_spike.py).
_API_LINK_RE = re.compile(
    r"^- \[(?P<text>[^\]]+)\]\((?P<url>https://docs\.mellea\.ai/api/[^)]+\.md)\)"
)

@dataclass
class FetchOutcome:
    """Result of a single URL fetch."""

    url: str
    ok: bool
    path: Optional[Path] = None
    status: Optional[int] = None
    bytes_written: int = 0
    error: Optional[str] = None


@dataclass
class RefreshResult:
    """Aggregate result of a :func:`refresh` call."""

    layout: CacheLayout
    llms_outcome: FetchOutcome
    module_outcomes: list[FetchOutcome] = field(default_factory=list)
    manifest_path: Optional[Path] = None

    @property
    def succeeded(self) -> int:
        """Count of successful per-module fetches (excludes llms.txt)."""
        return sum(1 for o in self.module_outcomes if o.ok)

    @property
    def failed(self) -> int:
        """Count of failed per-module fetches (excludes llms.txt)."""
        return sum(1 for o in self.module_outcomes if not o.ok)


# --- Internal HTTP helpers --------------------------------------------------


def _http_get(
    url: str,
    *,
    urlopen=urllib.request.urlopen,
    sleep=time.sleep,
) -> tuple[int, bytes]:
    """GET ``url`` with retries.

    Returns ``(status_code, body_bytes)``. Raises the last exception on
    exhaust. ``urlopen``/``sleep`` are injectable for tests.

    A non-2xx status is **not** retried — the only retryable conditions are
    network exceptions (URLError, socket timeout). 4xx/5xx are surfaced
    once so callers can distinguish "404, don't bother" from "we tried twice
    and the network was flaky".
    """
    if not url.startswith(MELLEA_DOCS_BASE_URL):
        raise ValueError(
            f"refusing to fetch outside {MELLEA_DOCS_BASE_URL}: {url!r}"
        )

    last_exc: Optional[BaseException] = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            with urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as resp:  # noqa: S310
                status = getattr(resp, "status", 200)
                body = resp.read()
                return status, body
        except urllib.error.HTTPError as exc:
            # HTTP-level failure (404, 500, ...). Don't retry — return.
            try:
                body = exc.read() or b""
            except Exception:  # pragma: no cover - defensive
                body = b""
            return exc.code, body
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < HTTP_MAX_ATTEMPTS:
                sleep(HTTP_RETRY_BACKOFF_SECONDS)
                continue
            raise

    # Unreachable — the loop either returns or raises.
    if last_exc is not None:  # pragma: no cover - defensive
        raise last_exc
    raise RuntimeError("unreachable")  # pragma: no cover - defensive


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a sibling temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Parsing ----------------------------------------------------------------


def parse_api_urls(llms_text: str) -> list[tuple[str, str]]:
    """Parse ``- [link_text](url)`` lines out of llms.txt.

    Returns a list of ``(link_text, url)`` pairs in declaration order.
    Matches the same regex the Phase 0 introspection spike uses so the cache
    populates exactly the URLs the coverage report measured.
    """
    out: list[tuple[str, str]] = []
    for line in llms_text.splitlines():
        m = _API_LINK_RE.match(line.strip())
        if not m:
            continue
        out.append((m["text"].strip(), m["url"].strip()))
    return out


# --- Public fetchers --------------------------------------------------------


def fetch_index(
    layout: CacheLayout,
    *,
    urlopen=urllib.request.urlopen,
    sleep=time.sleep,
) -> tuple[FetchOutcome, list[tuple[str, str]]]:
    """Fetch ``llms.txt`` and write it to ``layout.llms_txt_path``.

    Returns ``(outcome, api_url_pairs)`` where ``api_url_pairs`` is the
    parsed ``(link_text, url)`` list (empty if the fetch failed). The caller
    is responsible for creating the version directory (e.g. via
    :meth:`CacheLayout.ensure_dirs`).
    """
    layout.version_dir.mkdir(parents=True, exist_ok=True)
    try:
        status, body = _http_get(LLMS_TXT_URL, urlopen=urlopen, sleep=sleep)
    except Exception as exc:
        return (
            FetchOutcome(
                url=LLMS_TXT_URL,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
            [],
        )

    if status != 200:
        return (
            FetchOutcome(
                url=LLMS_TXT_URL,
                ok=False,
                status=status,
                error=f"HTTP {status}",
            ),
            [],
        )

    _atomic_write_bytes(layout.llms_txt_path, body)
    text = body.decode("utf-8", errors="replace")
    pairs = parse_api_urls(text)
    return (
        FetchOutcome(
            url=LLMS_TXT_URL,
            ok=True,
            path=layout.llms_txt_path,
            status=status,
            bytes_written=len(body),
        ),
        pairs,
    )


def fetch_module_doc(
    layout: CacheLayout,
    url: str,
    *,
    urlopen=urllib.request.urlopen,
    sleep=time.sleep,
) -> FetchOutcome:
    """Fetch a single ``api/*.md`` page and write it into the cache.

    A 404 is recorded as ``ok=False, status=404`` with a stderr warning so
    a single removed page doesn't abort a populate run.
    """
    dest = layout.api_doc_path(url)
    try:
        status, body = _http_get(url, urlopen=urlopen, sleep=sleep)
    except Exception as exc:
        return FetchOutcome(
            url=url,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if status == 404:
        print(
            f"warning: {url} returned 404; recording in manifest.missing_urls",
            file=sys.stderr,
        )
        return FetchOutcome(url=url, ok=False, status=status, error="HTTP 404")

    if status != 200:
        return FetchOutcome(
            url=url,
            ok=False,
            status=status,
            error=f"HTTP {status}",
        )

    _atomic_write_bytes(dest, body)
    return FetchOutcome(
        url=url,
        ok=True,
        path=dest,
        status=status,
        bytes_written=len(body),
    )


# --- High-level refresh -----------------------------------------------------


def refresh(
    layout: CacheLayout,
    *,
    urlopen=urllib.request.urlopen,
    sleep=time.sleep,
    progress=None,
) -> RefreshResult:
    """Fetch llms.txt + every linked ``api/*.md`` and write a manifest.

    Parameters
    ----------
    layout
        Where to write. The version directory is created if missing.
    urlopen, sleep
        Injectable urlopen / sleep — for tests.
    progress
        Optional callable ``progress(done, total, outcome)`` called after
        each per-module fetch. ``done`` is 1-indexed.
    """
    layout.ensure_dirs()
    llms_outcome, api_pairs = fetch_index(layout, urlopen=urlopen, sleep=sleep)
    result = RefreshResult(layout=layout, llms_outcome=llms_outcome)

    if not llms_outcome.ok:
        # llms.txt failed: write a minimal manifest so the freshness check
        # can report something coherent.
        _write_manifest(layout, result, module_urls=[])
        return result

    for i, (link_text, url) in enumerate(api_pairs, start=1):
        outcome = fetch_module_doc(layout, url, urlopen=urlopen, sleep=sleep)
        result.module_outcomes.append(outcome)
        if progress is not None:
            try:
                progress(i, len(api_pairs), outcome)
            except Exception:  # pragma: no cover - progress is best-effort
                pass

    _write_manifest(layout, result, module_urls=api_pairs)
    return result


def _write_manifest(
    layout: CacheLayout,
    result: RefreshResult,
    *,
    module_urls: list[tuple[str, str]],
) -> None:
    """Hash every cached file and write ``manifest.json``."""
    files: dict[str, dict] = {}

    # llms.txt
    if layout.llms_txt_path.is_file():
        files["llms.txt"] = {
            "sha256": sha256_of(layout.llms_txt_path),
            "size": layout.llms_txt_path.stat().st_size,
            "source_url": LLMS_TXT_URL,
        }

    # api/*.md
    missing_urls: list[str] = []
    for outcome in result.module_outcomes:
        if outcome.ok and outcome.path is not None and outcome.path.is_file():
            rel = outcome.path.relative_to(layout.version_dir).as_posix()
            files[rel] = {
                "sha256": sha256_of(outcome.path),
                "size": outcome.path.stat().st_size,
                "source_url": outcome.url,
            }
        else:
            missing_urls.append(outcome.url)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mellea_version": installed_mellea_version() or "unknown",
        "fetched_at": _now_iso(),
        "source": {
            "llms_txt_url": LLMS_TXT_URL,
            "docs_base_url": MELLEA_DOCS_BASE_URL,
        },
        "files": files,
        "missing_urls": missing_urls,
        "module_urls": [list(p) for p in module_urls],
    }
    _atomic_write_bytes(
        layout.manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    result.manifest_path = layout.manifest_path
