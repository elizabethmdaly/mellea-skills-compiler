"""Unit tests for mellea_skills_compiler.cache.fetcher.

All HTTP is mocked via an injectable ``urlopen`` callable; tests never touch
the network. One optional smoke test (`test_real_populate`) is marked
``@pytest.mark.network`` and skipped unless ``--run-network`` is passed.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Iterable, Optional

import pytest

from mellea_skills_compiler.cache.fetcher import (
    LLMS_TXT_URL,
    MELLEA_DOCS_BASE_URL,
    fetch_index,
    fetch_module_doc,
    parse_api_urls,
    refresh,
)
from mellea_skills_compiler.cache.freshness import is_fresh
from mellea_skills_compiler.cache.layout import CacheLayout


SAMPLE_LLMS_TXT = """\
# Mellea
- [Home](https://docs.mellea.ai/)
- [mellea.stdlib.session](https://docs.mellea.ai/api/mellea.stdlib.session.md)
- [mellea.stdlib.requirements](https://docs.mellea.ai/api/mellea.stdlib.requirements.md)
- [mellea.backends.ollama](https://docs.mellea.ai/api/mellea.backends.ollama.md)
- not a link line
"""


# --- Mock urlopen -----------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_urlopen(
    responses: dict[str, bytes],
    *,
    statuses: Optional[dict[str, int]] = None,
    flaky: Optional[Iterable[str]] = None,
):
    """Build a urlopen replacement.

    ``responses`` maps URL → body. ``statuses`` overrides HTTP status per URL
    (defaults to 200). ``flaky`` is a set of URLs whose first call raises a
    URLError; the second call succeeds (used to test retry).
    """
    statuses = statuses or {}
    flaky_set = set(flaky or [])
    call_counts: dict[str, int] = {}

    def _urlopen(url: str, timeout: int = 30):  # noqa: ARG001
        call_counts[url] = call_counts.get(url, 0) + 1
        if url in flaky_set and call_counts[url] == 1:
            raise urllib.error.URLError("simulated transient")
        status = statuses.get(url, 200)
        if status == 404:
            raise urllib.error.HTTPError(
                url, 404, "Not Found", {}, io.BytesIO(b"missing")
            )
        if status >= 400:
            raise urllib.error.HTTPError(
                url, status, "Err", {}, io.BytesIO(b"err")
            )
        body = responses.get(url, b"")
        return _FakeResponse(body, status=status)

    _urlopen.call_counts = call_counts  # type: ignore[attr-defined]
    return _urlopen


def _no_sleep(_seconds: float) -> None:
    return None


# --- parse_api_urls ---------------------------------------------------------


class TestParseApiUrls:
    def test_returns_only_api_md_rows(self):
        pairs = parse_api_urls(SAMPLE_LLMS_TXT)
        assert pairs == [
            (
                "mellea.stdlib.session",
                "https://docs.mellea.ai/api/mellea.stdlib.session.md",
            ),
            (
                "mellea.stdlib.requirements",
                "https://docs.mellea.ai/api/mellea.stdlib.requirements.md",
            ),
            (
                "mellea.backends.ollama",
                "https://docs.mellea.ai/api/mellea.backends.ollama.md",
            ),
        ]

    def test_empty_input(self):
        assert parse_api_urls("") == []

    def test_real_spike_format_compatibility(self):
        # Confirms our parser uses the same regex shape as the spike's, so the
        # 73-URL count carries across.
        line = "- [mellea.stdlib.requirements.md](https://docs.mellea.ai/api/mellea.stdlib.requirements/md.md)"
        pairs = parse_api_urls(line)
        assert len(pairs) == 1
        assert pairs[0][1].endswith("/md.md")


# --- fetch_index ------------------------------------------------------------


class TestFetchIndex:
    def test_happy_path_writes_llms_txt_and_parses(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        urlopen = make_urlopen({LLMS_TXT_URL: SAMPLE_LLMS_TXT.encode()})
        outcome, pairs = fetch_index(layout, urlopen=urlopen, sleep=_no_sleep)
        assert outcome.ok
        assert outcome.status == 200
        assert layout.llms_txt_path.read_bytes() == SAMPLE_LLMS_TXT.encode()
        assert len(pairs) == 3

    def test_http_error(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        urlopen = make_urlopen({}, statuses={LLMS_TXT_URL: 500})
        outcome, pairs = fetch_index(layout, urlopen=urlopen, sleep=_no_sleep)
        assert not outcome.ok
        assert outcome.status == 500
        assert pairs == []
        assert not layout.llms_txt_path.exists()

    def test_network_error(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")

        def _boom(url, timeout=30):  # noqa: ARG001
            raise urllib.error.URLError("dns")

        outcome, pairs = fetch_index(layout, urlopen=_boom, sleep=_no_sleep)
        assert not outcome.ok
        assert outcome.error is not None
        assert "URLError" in outcome.error
        assert pairs == []


# --- fetch_module_doc -------------------------------------------------------


class TestFetchModuleDoc:
    def test_writes_to_correct_path(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        url = "https://docs.mellea.ai/api/mellea.stdlib.session.md"
        urlopen = make_urlopen({url: b"# Session\n"})
        outcome = fetch_module_doc(layout, url, urlopen=urlopen, sleep=_no_sleep)
        assert outcome.ok
        assert outcome.path is not None
        assert outcome.path == layout.api_dir / "mellea.stdlib.session.md"
        assert outcome.path.read_bytes() == b"# Session\n"

    def test_404_returns_not_ok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        url = "https://docs.mellea.ai/api/missing.md"
        urlopen = make_urlopen({}, statuses={url: 404})
        outcome = fetch_module_doc(layout, url, urlopen=urlopen, sleep=_no_sleep)
        assert not outcome.ok
        assert outcome.status == 404
        captured = capsys.readouterr()
        assert "404" in captured.err

    def test_retry_on_transient_then_success(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        url = "https://docs.mellea.ai/api/mellea.stdlib.session.md"
        urlopen = make_urlopen({url: b"# Session\n"}, flaky=[url])
        slept: list[float] = []
        outcome = fetch_module_doc(
            layout, url, urlopen=urlopen, sleep=lambda s: slept.append(s)
        )
        assert outcome.ok
        assert urlopen.call_counts[url] == 2  # one retry
        assert slept == [2.0]  # one backoff

    def test_refuses_off_host(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        layout.ensure_dirs()
        outcome = fetch_module_doc(
            layout,
            "https://evil.example.com/api/x.md",
            urlopen=make_urlopen({}),
            sleep=_no_sleep,
        )
        assert not outcome.ok
        assert outcome.error is not None
        assert "refusing to fetch" in outcome.error


# --- refresh (end-to-end with mocks) ----------------------------------------


def _build_full_corpus() -> dict[str, bytes]:
    """Map every URL the sample llms.txt references to a synthetic body."""
    pairs = parse_api_urls(SAMPLE_LLMS_TXT)
    return {
        LLMS_TXT_URL: SAMPLE_LLMS_TXT.encode(),
        **{url: f"# {text}\n".encode() for text, url in pairs},
    }


class TestRefresh:
    def test_full_populate_writes_manifest_and_files(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        corpus = _build_full_corpus()
        urlopen = make_urlopen(corpus)
        result = refresh(layout, urlopen=urlopen, sleep=_no_sleep)
        assert result.llms_outcome.ok
        assert result.succeeded == 3
        assert result.failed == 0
        assert result.manifest_path == layout.manifest_path

        manifest = json.loads(layout.manifest_path.read_text())
        assert manifest["mellea_version"]  # not None
        assert "llms.txt" in manifest["files"]
        assert "api/mellea.stdlib.session.md" in manifest["files"]
        assert manifest["missing_urls"] == []
        assert manifest["source"]["docs_base_url"] == MELLEA_DOCS_BASE_URL

    def test_freshness_after_populate(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        corpus = _build_full_corpus()
        refresh(
            layout,
            urlopen=make_urlopen(corpus),
            sleep=_no_sleep,
        )
        # Use the manifest-declared version so this test is independent of
        # which Mellea is installed in the test env.
        manifest = json.loads(layout.manifest_path.read_text())
        report = is_fresh(layout, installed_version=manifest["mellea_version"])
        assert report.fresh, report.reasons

    def test_partial_failure_records_missing_url(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        corpus = _build_full_corpus()
        # Mark one URL as 404.
        missing_url = "https://docs.mellea.ai/api/mellea.backends.ollama.md"
        corpus.pop(missing_url, None)
        urlopen = make_urlopen(corpus, statuses={missing_url: 404})
        result = refresh(layout, urlopen=urlopen, sleep=_no_sleep)
        assert result.succeeded == 2
        assert result.failed == 1
        manifest = json.loads(layout.manifest_path.read_text())
        assert missing_url in manifest["missing_urls"]
        # The other two files are present.
        assert "api/mellea.stdlib.session.md" in manifest["files"]

    def test_progress_callback_invoked(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        corpus = _build_full_corpus()
        urlopen = make_urlopen(corpus)
        seen: list[tuple[int, int]] = []
        refresh(
            layout,
            urlopen=urlopen,
            sleep=_no_sleep,
            progress=lambda done, total, _o: seen.append((done, total)),
        )
        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_llms_txt_failure_short_circuits(self, tmp_path: Path):
        layout = CacheLayout.for_version(tmp_path, "0.5.0")
        urlopen = make_urlopen({}, statuses={LLMS_TXT_URL: 500})
        result = refresh(layout, urlopen=urlopen, sleep=_no_sleep)
        assert not result.llms_outcome.ok
        assert result.module_outcomes == []
        # Manifest still gets written (with no files) so status() has
        # something to report.
        assert layout.manifest_path.is_file()


# --- Optional online smoke test --------------------------------------------


@pytest.mark.network
def test_real_populate(tmp_path: Path):
    """Hit docs.mellea.ai for real. Skipped unless --run-network."""
    layout = CacheLayout.for_version(tmp_path, "0.5.0")
    result = refresh(layout)
    assert result.llms_outcome.ok
    # Expect ≥70 of 73 known module pages (per coverage report).
    assert result.succeeded >= 70, (
        f"expected ≥70 successful module fetches, got {result.succeeded} "
        f"({result.failed} failed)"
    )
