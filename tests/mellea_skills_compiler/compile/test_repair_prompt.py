"""Tests for the F1 repair-prompt integration.

These tests exercise :func:`build_repair_prompt_lint_section` — the
orchestrator-side glue that reads a parsed ``step_7_report.json`` and
assembles the per-lint fix prescriptions section from
:mod:`compile.repair_templates`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.repair_prompt import (
    build_repair_prompt_lint_section,
)


# ─── Synthetic report builders ─────────────────────────────────────────


def _make_report(*lints: dict) -> dict:
    """Build a minimal step_7_report.json shape with the given lint entries."""
    return {
        "format_version": "1.1",
        "checked_at": "2026-05-17T12:00:00+00:00",
        "package_path": "/some/pkg",
        "overall_verdict": "fail",
        "blocking_failures": sum(
            1
            for lint in lints
            if lint.get("verdict") == "fail" and lint.get("severity") == "error"
        ),
        "warnings": 0,
        "info_failures": 0,
        "strict": False,
        "smoke_check_mode": "never",
        "warnings_escalated_by_smoke": [],
        "smoke_check": None,
        "lints": list(lints),
    }


def _make_lint(
    lint_id: str,
    *,
    verdict: str = "fail",
    severity: str = "error",
    effective_severity: str | None = None,
    failures: list[dict] | None = None,
) -> dict:
    """Build a single lint-entry dict matching format_version 1.1."""
    return {
        "lint_id": lint_id,
        "verdict": verdict,
        "severity": severity,
        "effective_severity": effective_severity
        if effective_severity is not None
        else severity,
        "files_checked": 1,
        "skipped_reason": None,
        "failures": failures or [],
    }


# ─── Concrete failure payloads pulled from real overnight failures ─────


RUNTIME_DEFAULTS_FAILURE_DICT = {
    "file": "config.py",
    "line": None,
    "column": None,
    "message": "config.py not found in compiled package — wrapper writer never ran.",
    "rule_ref": "C8 runtime defaults",
}


BUNDLED_ASSET_FAILURE_DICT = {
    "file": "loader.py",
    "line": 17,
    "column": 11,
    "message": (
        "Bundled asset path 'references' is resolved via '_PKG_DIR'. "
        "Must be `Path(__file__).parent / 'references'`."
    ),
    "rule_ref": "Rule OUT-6, Rule 2.5-2",
}


STDLIB_ARITY_FAILURE_DICT = {
    "file": "requirements.py",
    "line": 91,
    "column": 4,
    "message": (
        "`check(...)` called with 1 positional argument(s); signature "
        "requires between 2 and 2 positional arg(s)."
    ),
    "rule_ref": "Rule 4-4 (stdlib-arity)",
}


INSTRUCT_PARSE_FAILURE_DICT = {
    "file": "pipeline.py",
    "line": 251,
    "column": 18,
    "message": (
        "`thunk` is the result of `m.instruct(..., format=Model)` and is a "
        "`ComputedModelOutputThunk`. Use `model_validate_json(thunk.value)`."
    ),
    "rule_ref": "KB1 (instruct returns thunk)",
}


# A surface JSON shaped like the real intermediate/mellea_api_ref.json
SAMPLE_SURFACE_JSON = {
    "modules": {
        "mellea.stdlib.requirements": {
            "check": {
                "signature": (
                    "check(req: Requirement, ctx: Context) -> "
                    "mellea.core.requirement.Requirement"
                )
            }
        }
    }
}


# ─── Test cases ────────────────────────────────────────────────────────


def test_lint_section_assembles_prescriptions_from_real_report(tmp_path: Path):
    """A report with failures for all 4 wave-1 lints renders all 4 prescriptions."""
    report = _make_report(
        _make_lint(
            "runtime-defaults-bound", failures=[RUNTIME_DEFAULTS_FAILURE_DICT]
        ),
        _make_lint(
            "bundled-asset-path-resolution",
            failures=[BUNDLED_ASSET_FAILURE_DICT],
        ),
        _make_lint("stdlib-arity", failures=[STDLIB_ARITY_FAILURE_DICT]),
        _make_lint(
            "instruct-result-parse-before-access",
            failures=[INSTRUCT_PARSE_FAILURE_DICT],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    # All four template headers must appear, in registry/report order.
    assert "Fix prescription for `runtime-defaults-bound`" in out
    assert "Fix prescription for `bundled-asset-path-resolution`" in out
    assert "Fix prescription for `stdlib-arity`" in out
    assert "Fix prescription for `instruct-result-parse-before-access`" in out
    # The umbrella header is present.
    assert "## Per-lint fix prescriptions" in out
    # Sections are separated by a horizontal rule.
    assert "\n\n---\n\n" in out


def test_lint_section_filters_to_error_severity(tmp_path: Path):
    """Mix of error + warning + info → only error-severity prescriptions surface.

    The wave-1 registry only has templates for ERROR-severity lints, so we
    use one templated ERROR lint plus an unrelated WARNING and INFO entry
    to confirm the filter (not the registry) is what's excluding them.
    """
    # Construct a synthetic warning failure with a real, registered template id —
    # forcing severity=warning so the gate must exclude it.
    warn_failure_dict = dict(BUNDLED_ASSET_FAILURE_DICT)
    info_failure_dict = dict(STDLIB_ARITY_FAILURE_DICT)
    report = _make_report(
        _make_lint(
            "runtime-defaults-bound",
            severity="error",
            failures=[RUNTIME_DEFAULTS_FAILURE_DICT],
        ),
        _make_lint(
            "bundled-asset-path-resolution",
            severity="warning",
            failures=[warn_failure_dict],
        ),
        _make_lint(
            "stdlib-arity",
            severity="info",
            failures=[info_failure_dict],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    assert "Fix prescription for `runtime-defaults-bound`" in out
    # The warning-severity bundled-asset prescription must NOT appear.
    assert "Fix prescription for `bundled-asset-path-resolution`" not in out
    # Nor the info-severity stdlib-arity one.
    assert "Fix prescription for `stdlib-arity`" not in out


def test_lint_section_handles_effective_severity_escalation(tmp_path: Path):
    """A WARNING-declared lint with effective_severity=error (smoke escalation)
    IS included — the gate honours effective_severity."""
    report = _make_report(
        _make_lint(
            "bundled-asset-path-resolution",
            severity="warning",
            effective_severity="error",
            failures=[BUNDLED_ASSET_FAILURE_DICT],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    assert "Fix prescription for `bundled-asset-path-resolution`" in out


def test_lint_section_returns_none_when_no_error_failures(tmp_path: Path):
    """Report with only WARNING failures → None (caller falls back to generic)."""
    report = _make_report(
        _make_lint(
            "bundled-asset-path-resolution",
            severity="warning",
            failures=[BUNDLED_ASSET_FAILURE_DICT],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is None


def test_lint_section_returns_none_when_no_templates_match(tmp_path: Path):
    """ERROR-severity failures exist but no registered template → None."""
    report = _make_report(
        _make_lint(
            "session-boundary",  # ERROR severity but no wave-1 template
            severity="error",
            failures=[
                {
                    "file": "pipeline.py",
                    "line": 12,
                    "column": 0,
                    "message": "session boundary violated",
                    "rule_ref": "KB5",
                }
            ],
        ),
        _make_lint(
            "validator-soundness",
            severity="error",
            failures=[
                {
                    "file": "requirements.py",
                    "line": 50,
                    "column": 8,
                    "message": "validator returns wrong type",
                    "rule_ref": "Rule 4-2",
                }
            ],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is None


def test_lint_section_returns_none_when_only_passing_lints(tmp_path: Path):
    """Report with only verdict=pass entries → None."""
    report = _make_report(
        _make_lint(
            "runtime-defaults-bound", verdict="pass", failures=[]
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is None


def test_lint_section_uses_surface_json_when_available(tmp_path: Path):
    """A real surface JSON in the package's intermediate/ is picked up and
    its canonical signature appears in the stdlib-arity prescription."""
    pkg = tmp_path / "pkg"
    (pkg / "intermediate").mkdir(parents=True)
    (pkg / "intermediate" / "mellea_api_ref.json").write_text(
        json.dumps(SAMPLE_SURFACE_JSON)
    )
    report = _make_report(
        _make_lint("stdlib-arity", failures=[STDLIB_ARITY_FAILURE_DICT]),
    )
    out = build_repair_prompt_lint_section(report, pkg)
    assert out is not None
    # The canonical signature string from SAMPLE_SURFACE_JSON appears verbatim.
    assert "check(req: Requirement, ctx: Context)" in out


def test_lint_section_falls_back_gracefully_when_no_surface_json(tmp_path: Path):
    """No surface JSON → stdlib-arity template uses its fallback prose,
    no crash."""
    report = _make_report(
        _make_lint("stdlib-arity", failures=[STDLIB_ARITY_FAILURE_DICT]),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    assert "could not be resolved" in out


def test_lint_section_handles_grounding_unavailable_surface(tmp_path: Path):
    """Surface JSON marked grounding_unavailable=true → fallback prose."""
    pkg = tmp_path / "pkg"
    (pkg / "intermediate").mkdir(parents=True)
    (pkg / "intermediate" / "mellea_api_ref.json").write_text(
        json.dumps({"grounding_unavailable": True})
    )
    report = _make_report(
        _make_lint("stdlib-arity", failures=[STDLIB_ARITY_FAILURE_DICT]),
    )
    out = build_repair_prompt_lint_section(report, pkg)
    assert out is not None
    assert "could not be resolved" in out


def test_lint_section_handles_empty_report(tmp_path: Path):
    """Empty / minimal report → None."""
    assert build_repair_prompt_lint_section({}, tmp_path) is None
    assert build_repair_prompt_lint_section({"lints": []}, tmp_path) is None


def test_lint_section_handles_malformed_report_shape(tmp_path: Path):
    """Defensive: non-dict / non-list inputs → None, no crash."""
    assert build_repair_prompt_lint_section(None, tmp_path) is None  # type: ignore[arg-type]
    assert build_repair_prompt_lint_section({"lints": "not-a-list"}, tmp_path) is None


def test_lint_section_skips_failures_with_extra_fields(tmp_path: Path):
    """A failure dict with extra keys (e.g. `suggestion`) is still parseable.

    The wrapper-rendered lint failures in real reports sometimes carry
    additional fields beyond the LintFailure dataclass; the conversion
    should drop them and still render the prescription.
    """
    failure_with_extras = {
        **RUNTIME_DEFAULTS_FAILURE_DICT,
        "suggestion": "Re-run config_writer.py — out of scope for the LintFailure dataclass.",
        "future_field": "something else",
    }
    report = _make_report(
        _make_lint(
            "runtime-defaults-bound", failures=[failure_with_extras]
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    assert "Fix prescription for `runtime-defaults-bound`" in out


def test_lint_section_renders_multiple_failures_for_same_lint(tmp_path: Path):
    """A lint with N failures yields N prescription blocks for that lint."""
    failure_a = dict(BUNDLED_ASSET_FAILURE_DICT, line=17)
    failure_b = dict(BUNDLED_ASSET_FAILURE_DICT, line=42)
    report = _make_report(
        _make_lint(
            "bundled-asset-path-resolution",
            failures=[failure_a, failure_b],
        ),
    )
    out = build_repair_prompt_lint_section(report, tmp_path)
    assert out is not None
    assert out.count("Fix prescription for `bundled-asset-path-resolution`") == 2
    assert "Line: 17" in out
    assert "Line: 42" in out


def test_lint_section_against_real_failed_report(tmp_path: Path):
    """End-to-end: load the real failed sentry-find-bugs report → confirm
    the assembled section is non-empty and contains the expected lint id.

    Falls back to a synthetic equivalent if the real file is not accessible
    (the repo path can be different across worktrees).
    """
    real_report = Path(
        "/Users/elizabeth.daly/Documents/Dev/mellea-skills-compiler-release/"
        "ed-fork/mellea-skills-compiler/examples/sentry-find-bugs/"
        "sentry_find_bugs_mellea/intermediate/step_7_report.json"
    )
    if real_report.is_file():
        report = json.loads(real_report.read_text(encoding="utf-8"))
        pkg = real_report.parent.parent
    else:
        # Synthetic equivalent: a runtime-defaults-bound failure.
        report = _make_report(
            _make_lint(
                "runtime-defaults-bound",
                failures=[RUNTIME_DEFAULTS_FAILURE_DICT],
            ),
        )
        pkg = tmp_path
    out = build_repair_prompt_lint_section(report, pkg)
    assert out is not None, "expected at least one prescription"
    assert "## Per-lint fix prescriptions" in out
    assert "Fix prescription for `runtime-defaults-bound`" in out


def test_lint_section_passes_package_dir_to_get_repair_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The wrapper forwards package_dir + surface_json into the template
    dispatcher exactly as the template expects."""
    captured: list[dict] = []

    def fake_get_repair_template(
        lint_id, failure, *, package_dir=None, surface_json=None
    ):
        captured.append(
            {
                "lint_id": lint_id,
                "package_dir": package_dir,
                "surface_json": surface_json,
                "failure_file": failure.file,
            }
        )
        return f"## Fix prescription for `{lint_id}`\n\nrendered"

    monkeypatch.setattr(
        "mellea_skills_compiler.compile.repair_prompt.get_repair_template",
        fake_get_repair_template,
    )
    report = _make_report(
        _make_lint("stdlib-arity", failures=[STDLIB_ARITY_FAILURE_DICT]),
    )
    pkg = tmp_path / "pkg"
    (pkg / "intermediate").mkdir(parents=True)
    (pkg / "intermediate" / "mellea_api_ref.json").write_text(
        json.dumps(SAMPLE_SURFACE_JSON)
    )

    out = build_repair_prompt_lint_section(report, pkg)
    assert out is not None
    assert len(captured) == 1
    assert captured[0]["lint_id"] == "stdlib-arity"
    assert captured[0]["package_dir"] == pkg
    assert captured[0]["surface_json"] == SAMPLE_SURFACE_JSON
    assert captured[0]["failure_file"] == "requirements.py"
