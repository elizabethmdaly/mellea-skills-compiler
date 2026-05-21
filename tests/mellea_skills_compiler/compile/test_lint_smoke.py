"""Phase 2 tests: evidence-based downgrade via in-gate smoke check.

These tests never spawn a real ollama subprocess. Backend detection and
smoke execution are both mocked via ``monkeypatch.setattr`` on the
helpers exposed by ``mellea_skills_compiler.compile.lints``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile import lints as lints_mod
from mellea_skills_compiler.compile.lints import (
    LintResult,
    SmokeCheckOutcome,
    run_lints,
)


# ─── Fixtures: stub lints + smoke ───────────────────────────────────────


def _stub_lint_passing(_: Path) -> LintResult:
    """A lint that passes — never fails the gate."""
    return LintResult(lint_id="import-soundness", verdict="pass")


def _stub_lint_warning_fails(_: Path) -> LintResult:
    """A WARNING-severity lint that fails — does not block by itself."""
    return LintResult(lint_id="no-watsonx", verdict="fail")


def _stub_lint_error_fails(_: Path) -> LintResult:
    """An ERROR-severity lint that fails — always blocks."""
    return LintResult(lint_id="import-soundness", verdict="fail")


def _mock_backend_unavailable(*args, **kwargs):
    return False, "ollama binary not found on PATH"


def _mock_backend_available(*args, **kwargs):
    return True, None


# ─── Tests ──────────────────────────────────────────────────────────────


def test_smoke_check_skipped_when_no_backend(monkeypatch):
    """smoke_check='auto' + no backend → SKIPPED, gate verdict unchanged."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_warning_fails,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_unavailable)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="auto")

    assert result.smoke_check is not None
    assert result.smoke_check.verdict == "skipped"
    assert result.smoke_check.backend_available is False
    # warning-only failure + smoke=skipped → still pass (smoke had no
    # opportunity to escalate).
    assert result.overall_verdict == "pass"
    assert result.warnings == 1


def test_smoke_always_mode_fails_when_no_backend(monkeypatch):
    """smoke_check='always' + no backend → gate fails."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_passing,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_unavailable)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="always")

    assert result.smoke_check.verdict == "skipped"
    assert result.overall_verdict == "fail", (
        "smoke_check='always' must hard-fail when backend is unavailable"
    )


def test_smoke_check_passes_with_working_pipeline(monkeypatch):
    """Backend available + smoke passes → gate verdict preserved (pass)."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_warning_fails,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_available)

    # Substitute the inline runner so we never touch real smoke_check.py.
    def _fake_smoke(_: Path, **_kwargs) -> SmokeCheckOutcome:
        # Schema for step_7_report.smoke_check.verdict requires past-
        # tense values: passed/failed/skipped. Test fixtures must use
        # the same vocabulary so the writer's emission validates.
        return SmokeCheckOutcome(
            verdict="passed",
            fixture_used="fx-001",
            duration_seconds=1.2,
            backend_available=True,
        )

    monkeypatch.setattr(lints_mod, "_run_smoke_check_inline", _fake_smoke)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="auto")

    assert result.smoke_check.verdict == "passed"
    assert result.smoke_check.fixture_used == "fx-001"
    # WARNING failure + smoke confirmed working → overall stays pass
    assert result.overall_verdict == "pass"
    assert result.warnings == 1


def test_smoke_check_escalates_warnings_on_failure(monkeypatch):
    """Backend available + smoke fails + warning-only lint result →
    warnings are escalated to blocking for this run."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_warning_fails,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_available)

    def _fake_smoke_fail(_: Path, **_kwargs) -> SmokeCheckOutcome:
        return SmokeCheckOutcome(
            verdict="failed",  # schema-conformant past-tense
            fixture_used="fx-001",
            duration_seconds=0.7,
            backend_available=True,
            error="TypeError: missing argument",
        )

    monkeypatch.setattr(lints_mod, "_run_smoke_check_inline", _fake_smoke_fail)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="auto")
        report = json.loads((pkg / "intermediate" / "step_7_report.json").read_text())

    assert result.overall_verdict == "fail", (
        "warning-only lint + smoke fail must escalate to overall fail"
    )
    assert result.warnings == 1
    # ``effective_severity`` per-lint should mark the warning as error for this run.
    by_id = {entry["lint_id"]: entry for entry in report["lints"]}
    assert by_id["no-watsonx"]["severity"] == "warning"
    assert by_id["no-watsonx"]["effective_severity"] == "error"
    # Per the step_7_report schema, this field is the LIST of escalated
    # lint ids (not a flat bool). The single escalated warning should
    # appear by id.
    assert report["warnings_escalated_by_smoke"] == ["no-watsonx"]


def test_smoke_check_confirms_advisory_on_pass(monkeypatch):
    """Same as escalation test but smoke PASSES → warnings stay advisory."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_warning_fails,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_available)

    def _fake_smoke_pass(_: Path, **_kwargs) -> SmokeCheckOutcome:
        return SmokeCheckOutcome(
            verdict="passed",  # schema-conformant past-tense
            fixture_used="fx-001",
            duration_seconds=1.0,
            backend_available=True,
        )

    monkeypatch.setattr(lints_mod, "_run_smoke_check_inline", _fake_smoke_pass)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="auto")
        report = json.loads((pkg / "intermediate" / "step_7_report.json").read_text())

    assert result.overall_verdict == "pass"
    by_id = {entry["lint_id"]: entry for entry in report["lints"]}
    assert by_id["no-watsonx"]["effective_severity"] == "warning"
    # No warnings escalated → empty list (schema-typed as array).
    assert report["warnings_escalated_by_smoke"] == []


def test_smoke_never_mode(monkeypatch):
    """smoke_check='never' → smoke is skipped entirely; gate is severity-only."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_warning_fails,))
    # Sentinel: if the backend detector is called, blow up. With
    # smoke_check='never' it must not be.
    def _explode(*_a, **_kw):
        raise AssertionError("backend detector must not be called when smoke=never")

    monkeypatch.setattr(lints_mod, "_detect_backend_available", _explode)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="never")
        report = json.loads((pkg / "intermediate" / "step_7_report.json").read_text())

    assert result.smoke_check is None
    assert report["smoke_check"] is None
    assert result.overall_verdict == "pass"


def test_smoke_does_not_run_when_errors_already_block(monkeypatch):
    """ERROR-severity failures already block — smoke shouldn't run."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_error_fails,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_available)

    inline_calls = {"count": 0}

    def _fake_smoke(_: Path, **_kwargs) -> SmokeCheckOutcome:
        inline_calls["count"] += 1
        return SmokeCheckOutcome(verdict="passed", backend_available=True)

    monkeypatch.setattr(lints_mod, "_run_smoke_check_inline", _fake_smoke)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, smoke_check="auto")

    assert result.overall_verdict == "fail"
    assert result.smoke_check.verdict == "skipped"
    assert "error-severity" in (result.smoke_check.skipped_reason or "").lower()
    assert inline_calls["count"] == 0


def test_smoke_check_invalid_mode_raises(monkeypatch):
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_passing,))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            run_lints(Path(tmp), smoke_check="bogus")


def test_smoke_outcome_serialised_in_report(monkeypatch):
    """The report includes a top-level ``smoke_check`` block when smoke ran."""
    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_stub_lint_passing,))
    monkeypatch.setattr(lints_mod, "_detect_backend_available", _mock_backend_available)

    def _fake_smoke(_: Path, **_kwargs) -> SmokeCheckOutcome:
        return SmokeCheckOutcome(
            verdict="passed",  # schema-conformant past-tense
            fixture_used="hello-world",
            duration_seconds=0.42,
            backend_available=True,
        )

    monkeypatch.setattr(lints_mod, "_run_smoke_check_inline", _fake_smoke)

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        run_lints(pkg, smoke_check="auto")
        report = json.loads((pkg / "intermediate" / "step_7_report.json").read_text())

    smoke = report["smoke_check"]
    assert smoke["verdict"] == "passed"
    assert smoke["fixture_used"] == "hello-world"
    assert smoke["duration_seconds"] == pytest.approx(0.42)
    assert smoke["backend_available"] is True
    assert report["smoke_check_mode"] == "auto"
