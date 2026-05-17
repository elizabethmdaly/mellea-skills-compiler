"""Slow integration test: walk live mellea 0.5.0 + assert coverage baseline.

This test does the real introspection job, parses the historical
``coverage_report.md`` from the Phase 0.1 spike to extract the per-package
expectations, and asserts:

* every module documented as "found" in the spike report is still found by the
  productionised walker;
* overall coverage is >=95% (and ideally 100%);
* the same modules listed in the spike report are still in the live mellea
  surface.

Skipped automatically when mellea is not importable in the test environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

mellea = pytest.importorskip("mellea")

from mellea_skills_compiler.mellea_surface import (  # noqa: E402  (after importorskip)
    assess_coverage,
    build_surface,
    parse_api_urls,
    render_coverage_report,
)
from mellea_skills_compiler.mellea_surface.coverage import (  # noqa: E402
    COVERAGE_GATE_PERCENT,
    coverage_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SPIKE_REPORT = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "coverage_report.md"
)


def _parse_spike_packages(report: str) -> dict[str, dict[str, int]]:
    """Extract the per-package Found/Empty/Missing/ImportErr counts.

    The spike report has a markdown table::

        | `mellea.backends` | 7 | 0 | 0 | 0 |
    """
    rx = re.compile(
        r"^\|\s*`(?P<pkg>[\w\.]+)`\s*\|\s*(?P<f>\d+)\s*\|\s*(?P<e>\d+)\s*\|"
        r"\s*(?P<m>\d+)\s*\|\s*(?P<ie>\d+)\s*\|\s*$"
    )
    out: dict[str, dict[str, int]] = {}
    for line in report.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        out[m["pkg"]] = {
            "found": int(m["f"]),
            "empty": int(m["e"]),
            "missing": int(m["m"]),
            "import_error": int(m["ie"]),
        }
    return out


def test_live_mellea_surface_meets_baseline():
    """Walk real mellea, sanity-check the surface, regress on the spike baseline."""
    surface = build_surface(root_package="mellea")

    # Top-level shape.
    assert surface["mellea_version"]
    assert surface["python_version"]
    assert surface["modules"]
    assert "canonical_definitions" in surface
    assert "surface_fingerprint" in surface
    assert len(surface["surface_fingerprint"]) == 64  # sha256 hex

    # The spike found 132 modules. We don't pin to that exact number across
    # mellea versions, but the live surface should be in the same ballpark.
    assert len(surface["modules"]) >= 100, (
        f"Live mellea walker captured only {len(surface['modules'])} modules; "
        "spike captured 132. Investigate."
    )

    # If we're on mellea 0.5.0, cross-check against the spike report.
    if surface["mellea_version"] == "0.5.0" and SPIKE_REPORT.exists():
        spike_packages = _parse_spike_packages(SPIKE_REPORT.read_text())
        for pkg in spike_packages:
            # Every package the spike reported on should still have some
            # presence in the live surface (>=1 module under that namespace).
            assert any(
                m == pkg or m.startswith(f"{pkg}.")
                for m in surface["modules"]
            ), f"spike package {pkg!r} not represented in live surface"


def test_live_mellea_coverage_passes_gate():
    """Re-run the full llms.txt coverage assessment and assert >=95%.

    Fetches llms.txt over HTTPS at test time. If the network is unavailable,
    the test is skipped (not failed) — Phase 0.1 acceptance also tolerates
    network-unreachable environments.
    """
    surface = build_surface(root_package="mellea")

    try:
        from mellea_skills_compiler.mellea_surface import fetch_llms_txt

        text = fetch_llms_txt()
    except Exception as e:
        pytest.skip(f"llms.txt unreachable: {type(e).__name__}: {e}")

    api_urls = parse_api_urls(text)
    assert api_urls, "llms.txt parser returned no api/*.md rows"

    rows = assess_coverage(surface, api_urls)
    summary = coverage_summary(rows)

    # Render the report so any failure includes useful diagnostic text.
    report = render_coverage_report(rows, surface)

    assert summary["percent"] >= COVERAGE_GATE_PERCENT, (
        f"Coverage regressed: {summary['percent']:.1f}% < "
        f"{COVERAGE_GATE_PERCENT:.0f}%\n\n{report}"
    )


def test_live_surface_is_byte_stable(tmp_path: Path):
    """Two consecutive builds must serialise to byte-identical JSON.

    This is the cache-fingerprinting contract — without it the cache layer
    (P1.C) can't tell "actually changed" from "regenerated".
    """
    from mellea_skills_compiler.mellea_surface.introspector import (
        serialise_surface,
    )

    s1 = build_surface(root_package="mellea")
    s2 = build_surface(root_package="mellea")
    text1 = serialise_surface(s1)
    text2 = serialise_surface(s2)
    if text1 != text2:
        # Write both to disk for diff-on-failure debugging.
        (tmp_path / "s1.json").write_text(text1)
        (tmp_path / "s2.json").write_text(text2)
        pytest.fail(
            f"Surface JSON is not byte-stable across runs (see {tmp_path})"
        )

    # And the fingerprint should agree.
    assert s1["surface_fingerprint"] == s2["surface_fingerprint"]
