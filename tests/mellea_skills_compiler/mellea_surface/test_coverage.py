"""Unit tests for the llms.txt coverage cross-referencer."""

from __future__ import annotations

import pytest

from mellea_skills_compiler.mellea_surface.coverage import (
    assess_coverage,
    coverage_summary,
    link_text_to_module,
    parse_api_urls,
    render_coverage_report,
    url_to_module,
)
from mellea_skills_compiler.mellea_surface.types import Surface


# --- llms.txt parsing -------------------------------------------------------


_SAMPLE_LLMS_TXT = """\
# Mellea docs

- [Skip me](https://docs.mellea.ai/getting-started.md): not api/.
- [mellea.stdlib.session](https://docs.mellea.ai/api/stdlib/session.md): Session API.
- [mellea.core.backend](https://docs.mellea.ai/api/core/backend.md): Backend protocol.
- [mellea.stdlib.requirements.md](https://docs.mellea.ai/api/stdlib/requirements/md.md): Markdown requirement.
  - [mellea.helpers.cli](https://docs.mellea.ai/api/helpers/cli.md): CLI helpers.
"""


class TestParseApiUrls:
    def test_parses_api_rows_only(self):
        rows = parse_api_urls(_SAMPLE_LLMS_TXT)
        assert len(rows) == 4
        texts = [r[0] for r in rows]
        assert "mellea.stdlib.session" in texts
        assert "Skip me" not in texts

    def test_handles_indented_rows(self):
        # The last row in the sample is indented (sub-bullet).
        rows = parse_api_urls(_SAMPLE_LLMS_TXT)
        texts = [r[0] for r in rows]
        assert "mellea.helpers.cli" in texts

    def test_empty_input(self):
        assert parse_api_urls("") == []


class TestUrlToModule:
    def test_simple_path(self):
        url = "https://docs.mellea.ai/api/stdlib/session.md"
        assert url_to_module(url) == "stdlib.session"

    def test_collapses_duplicated_tail(self):
        # Docs site uses foo/foo.md for package-level pages.
        url = "https://docs.mellea.ai/api/core/core.md"
        assert url_to_module(url) == "core"


class TestLinkTextToModule:
    def test_link_text_wins_when_dotted(self):
        url = "https://docs.mellea.ai/api/anything/anywhere.md"
        assert (
            link_text_to_module("mellea.stdlib.session", url)
            == "mellea.stdlib.session"
        )

    def test_falls_back_to_url(self):
        url = "https://docs.mellea.ai/api/core/backend.md"
        # Non-dotted link text => fall back to URL conversion.
        assert link_text_to_module("Backend Protocol", url) == "core.backend"


# --- Coverage cross-reference ----------------------------------------------


def _surface(symbols_per_module: dict[str, int]) -> Surface:
    """Build a minimal fake surface with N public symbols per module."""
    modules: dict = {}
    for mod_name, n in symbols_per_module.items():
        if n < 0:
            modules[mod_name] = {"import_error": "Boom"}
            continue
        modules[mod_name] = {
            "doc": None,
            "symbols": {
                f"sym{i}": {"kind": "function", "defined_in": mod_name}
                for i in range(n)
            },
        }
    return {
        "mellea_version": "0.5.0",
        "python_version": "3.12.7",
        "modules": modules,
    }


class TestAssessCoverage:
    def test_found_status(self):
        surface = _surface({"mellea.stdlib.session": 3})
        rows = assess_coverage(
            surface,
            [("mellea.stdlib.session", "https://docs.mellea.ai/api/stdlib/session.md")],
        )
        assert len(rows) == 1
        assert rows[0].status == "found"
        assert "3 public symbols" in rows[0].note

    def test_missing_status(self):
        surface = _surface({"mellea.stdlib.session": 1})
        rows = assess_coverage(
            surface,
            [("mellea.core.gone", "https://docs.mellea.ai/api/core/gone.md")],
        )
        assert rows[0].status == "missing"

    def test_empty_status(self):
        surface = _surface({"mellea.stdlib.empty": 0})
        rows = assess_coverage(
            surface,
            [("mellea.stdlib.empty", "https://docs.mellea.ai/api/stdlib/empty.md")],
        )
        assert rows[0].status == "empty"

    def test_import_error_status(self):
        surface = _surface({"mellea.broken": -1})
        rows = assess_coverage(
            surface,
            [("mellea.broken", "https://docs.mellea.ai/api/broken.md")],
        )
        assert rows[0].status == "import_error"
        assert "Boom" in rows[0].note


class TestCoverageSummary:
    def test_percent_calc(self):
        surface = _surface(
            {
                "mellea.a": 1,
                "mellea.b": 1,
                "mellea.c": 0,  # empty
            }
        )
        rows = assess_coverage(
            surface,
            [
                ("mellea.a", "https://docs.mellea.ai/api/a.md"),
                ("mellea.b", "https://docs.mellea.ai/api/b.md"),
                ("mellea.c", "https://docs.mellea.ai/api/c.md"),
                ("mellea.d", "https://docs.mellea.ai/api/d.md"),
            ],
        )
        summary = coverage_summary(rows)
        assert summary["total"] == 4
        assert summary["found"] == 2
        assert summary["empty"] == 1
        assert summary["missing"] == 1
        assert summary["percent"] == pytest.approx(50.0)
        assert summary["gate_pass"] is False

    def test_per_package_grouping(self):
        surface = _surface(
            {"mellea.stdlib.a": 1, "mellea.stdlib.b": 1, "mellea.core.x": 1}
        )
        rows = assess_coverage(
            surface,
            [
                ("mellea.stdlib.a", "https://docs.mellea.ai/api/stdlib/a.md"),
                ("mellea.stdlib.b", "https://docs.mellea.ai/api/stdlib/b.md"),
                ("mellea.core.x", "https://docs.mellea.ai/api/core/x.md"),
            ],
        )
        summary = coverage_summary(rows)
        assert summary["per_package"]["mellea.stdlib"]["found"] == 2
        assert summary["per_package"]["mellea.core"]["found"] == 1

    def test_gate_passes_at_100(self):
        surface = _surface({"mellea.a": 1})
        rows = assess_coverage(
            surface, [("mellea.a", "https://docs.mellea.ai/api/a.md")]
        )
        assert coverage_summary(rows)["gate_pass"] is True


# --- Report rendering ------------------------------------------------------


class TestRenderCoverageReport:
    def test_includes_headline(self):
        surface = _surface({"mellea.a": 1})
        rows = assess_coverage(
            surface, [("mellea.a", "https://docs.mellea.ai/api/a.md")]
        )
        report = render_coverage_report(rows, surface)
        assert "# Mellea Introspection Coverage Report" in report
        assert "**Overall coverage: 100.0%**" in report
        assert "Gate: PASS" in report

    def test_emits_missing_section(self):
        surface = _surface({"mellea.a": 1})
        rows = assess_coverage(
            surface,
            [
                ("mellea.a", "https://docs.mellea.ai/api/a.md"),
                ("mellea.gone", "https://docs.mellea.ai/api/gone.md"),
            ],
        )
        report = render_coverage_report(rows, surface)
        assert "## Missing modules" in report
        assert "mellea.gone" in report

    def test_no_missing_section_when_clean(self):
        surface = _surface({"mellea.a": 1})
        rows = assess_coverage(
            surface, [("mellea.a", "https://docs.mellea.ai/api/a.md")]
        )
        report = render_coverage_report(rows, surface)
        assert "## Missing modules" not in report
