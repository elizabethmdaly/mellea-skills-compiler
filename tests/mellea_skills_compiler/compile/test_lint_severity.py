"""Phase 1 tests: graduated lint severity + `--strict` gate.

Covers:
  - Every lint in ALL_LINTS has a severity entry in ``_LINT_SEVERITY``.
  - The 14+3 ERROR / 5 WARNING / 1 INFO classification matches the user's
    decision (parametrised — these are intentional product decisions, not
    incidental code).
  - ``run_lints`` gate:
      * WARNING-only failures → overall_verdict == "pass", warnings counted.
      * ERROR failure         → overall_verdict == "fail", blocking_failures counted.
      * strict=True           → WARNING failures are promoted to blocking.
  - step_7_report.json carries ``severity`` per lint and the new top-level
    counters (``blocking_failures``, ``warnings``, ``info_failures``).

These tests avoid touching the real lint detection logic — they synthesise
``LintResult`` objects and exercise the gate semantics directly where it
matters. The full end-to-end exercise (``test_run_lints_with_synthetic_pkg``)
goes through ``run_lints`` against a tempdir package so the report write
and report shape are covered too.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest

from mellea_skills_compiler.compile import lints as lints_mod
from mellea_skills_compiler.compile.lints import (
    ALL_LINTS,
    LintResult,
    LintSeverity,
    _LINT_SEVERITY,
    _apply_severity,
    run_lints,
)


# Expected severity classification (the user's product decision).
# This is the single source of truth tested below.
_EXPECTED_SEVERITY: Dict[str, LintSeverity] = {
    # ERROR: always-breaks-runtime (14)
    "parseable": LintSeverity.ERROR,
    "import-soundness": LintSeverity.ERROR,
    "stdlib-arity": LintSeverity.ERROR,
    "pipeline-entry-canonical": LintSeverity.ERROR,
    "fixture-signature-bound": LintSeverity.ERROR,
    "instruct-has-description": LintSeverity.ERROR,
    "instruct-result-parse-before-access": LintSeverity.ERROR,
    "generative-forbidden-params": LintSeverity.ERROR,
    "generative-call-passes-session": LintSeverity.ERROR,
    "validator-soundness": LintSeverity.ERROR,
    "grounding-context-types": LintSeverity.ERROR,
    "format-annotation": LintSeverity.ERROR,
    "variable-safety": LintSeverity.ERROR,
    "session-boundary": LintSeverity.ERROR,
    # ERROR: deployment-context-sensitive (3) + fixtures-loader-contract
    "bundled-asset-path-resolution": LintSeverity.ERROR,
    "runtime-defaults-bound": LintSeverity.ERROR,
    "import-side-effects": LintSeverity.ERROR,
    "fixtures-loader-contract": LintSeverity.ERROR,
    # WARNING (5)
    "complex-schema-needs-strategy-or-fallback": LintSeverity.WARNING,
    "optional-extraction-guidance": LintSeverity.WARNING,
    "prefix-persona": LintSeverity.WARNING,
    "no-watsonx": LintSeverity.WARNING,
    "run-pipeline-params-typed": LintSeverity.WARNING,
    # INFO (1)
    "melleafy-json-consistency": LintSeverity.INFO,
}


def _lint_ids_from_all_lints() -> set:
    """Discover every lint id ALL_LINTS produces by running each function on
    an empty tempdir. Each function returns a LintResult whose ``lint_id``
    is the stable identifier (the same one reported in step_7_report.json).
    """
    discovered = set()
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        for lint_fn in ALL_LINTS:
            result = lint_fn(pkg)
            discovered.add(result.lint_id)
    return discovered


def test_each_lint_has_declared_severity():
    """Every lint in ALL_LINTS must be classified in ``_LINT_SEVERITY``.

    Catches the "added a new lint, forgot to classify it" regression.
    Without an explicit entry the lint would default to ERROR via the
    fallback in ``_apply_severity`` — usually safe, but we want the
    classification to be a deliberate decision recorded in source.
    """
    discovered = _lint_ids_from_all_lints()
    missing = discovered - set(_LINT_SEVERITY)
    assert not missing, (
        f"Lints in ALL_LINTS missing severity classification: {missing}. "
        f"Add an entry to ``_LINT_SEVERITY`` in compile/lints.py."
    )


@pytest.mark.parametrize(
    "lint_id, expected_severity",
    sorted(_EXPECTED_SEVERITY.items()),
)
def test_severity_classification_matches_user_decision(
    lint_id: str, expected_severity: LintSeverity
):
    """The 14+3 ERROR / 5 WARNING / 1 INFO split is a product decision.

    If this test fails, do NOT just flip the expected value — re-read
    the original classification rationale in
    ``.claude/commands/mellea-fy-validate.md`` and confirm with the
    compiler owner before changing.
    """
    actual = _LINT_SEVERITY.get(lint_id)
    assert actual == expected_severity, (
        f"Severity mismatch for {lint_id!r}: expected {expected_severity}, "
        f"got {actual}. See .claude/commands/mellea-fy-validate.md for the "
        f"authoritative classification."
    )


def test_severity_counts_match_documented_classification():
    """The documented 14+3 / 5 / 1 split should be the actual content of
    ``_LINT_SEVERITY``. This catches drift between the docs and the table.
    """
    errors = [k for k, v in _LINT_SEVERITY.items() if v == LintSeverity.ERROR]
    warnings = [k for k, v in _LINT_SEVERITY.items() if v == LintSeverity.WARNING]
    infos = [k for k, v in _LINT_SEVERITY.items() if v == LintSeverity.INFO]

    # 14 always-breaks-runtime + 3 deployment-context-sensitive + 1
    # fixtures-loader-contract = 18 ERROR.
    assert len(errors) == 18, (
        f"Expected 18 ERROR lints (14 always-breaks + 3 deployment-context "
        f"+ fixtures-loader-contract); got {len(errors)}: {sorted(errors)}"
    )
    assert len(warnings) == 5, (
        f"Expected 5 WARNING lints; got {len(warnings)}: {sorted(warnings)}"
    )
    assert len(infos) == 1, (
        f"Expected 1 INFO lint; got {len(infos)}: {sorted(infos)}"
    )


# ─── Gate logic ───


def _stamp(lint_id: str, verdict: str) -> LintResult:
    """Construct a LintResult and stamp its severity via the gate's helper.

    This is what ``run_lints`` would do for a real lint invocation; we
    use it here so the gate sees the right severity without us having
    to manually thread it through every test.
    """
    return _apply_severity(LintResult(lint_id=lint_id, verdict=verdict))


def _write_minimal_pass_package(pkg: Path) -> None:
    """Make a package that passes the lint suite cleanly so we can mutate
    one file at a time to flip individual lints.
    """
    (pkg / "fixtures").mkdir()
    (pkg / "fixtures" / "__init__.py").write_text("ALL_FIXTURES = []\n")
    (pkg / "config.py").write_text(
        'BACKEND = "ollama"\nMODEL_ID = "granite4.1:8b"\n'
    )
    intermediate = pkg / "intermediate"
    intermediate.mkdir()
    (intermediate / "runtime_defaults_directive.json").write_text(
        json.dumps(
            {
                "backend": "ollama",
                "model_id": "granite4.1:8b",
                "source": "runtime_defaults.json",
            }
        )
    )


def test_run_lints_gate_passes_when_only_warnings_fail(monkeypatch):
    """Synthetic: a WARNING-severity lint failing → overall pass, warnings>0."""

    # Stub ALL_LINTS for this test so we can deterministically inject a
    # single WARNING-severity lint failure without depending on what real
    # lints do against an empty tempdir.
    def _warning_fail(_: Path) -> LintResult:
        return LintResult(
            lint_id="no-watsonx",  # known WARNING in _LINT_SEVERITY
            verdict="fail",
            failures=[],
        )

    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_warning_fail,))

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg)

    assert result.overall_verdict == "pass", (
        f"WARNING-only failure must not block; got {result.overall_verdict}"
    )
    assert result.warnings == 1
    assert result.blocking_failures == 0


def test_run_lints_gate_fails_when_error_fails(monkeypatch):
    """Synthetic: an ERROR-severity lint failing → overall fail, blocking>0."""

    def _error_fail(_: Path) -> LintResult:
        return LintResult(
            lint_id="import-soundness",  # known ERROR
            verdict="fail",
            failures=[],
        )

    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_error_fail,))

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg)

    assert result.overall_verdict == "fail"
    assert result.blocking_failures == 1
    assert result.warnings == 0


def test_strict_mode_promotes_warnings_to_blocking(monkeypatch):
    """Same warning-only input as the pass-case test, but ``strict=True``
    → the gate promotes it to a blocking failure."""

    def _warning_fail(_: Path) -> LintResult:
        return LintResult(
            lint_id="no-watsonx",
            verdict="fail",
            failures=[],
        )

    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_warning_fail,))

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, strict=True)

    assert result.overall_verdict == "fail", (
        "strict=True must promote WARNING-severity failures to blocking"
    )
    # The counters reflect the underlying classification — they don't lie
    # about severity. Only the gate semantic changes.
    assert result.warnings == 1
    assert result.blocking_failures == 0
    assert result.strict is True


def test_strict_mode_does_not_promote_info(monkeypatch):
    """INFO-severity failures stay non-blocking even under ``--strict``.

    INFO is telemetry/drift. The classification table puts only one
    lint there (``melleafy-json-consistency``) — promoting drift signals
    to blocking would be hostile in CI.
    """

    def _info_fail(_: Path) -> LintResult:
        return LintResult(
            lint_id="melleafy-json-consistency",
            verdict="fail",
            failures=[],
        )

    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_info_fail,))

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        result = run_lints(pkg, strict=True)

    assert result.overall_verdict == "pass"
    assert result.info_failures == 1


def test_report_includes_severity_per_lint(monkeypatch):
    """step_7_report.json must carry ``severity`` per lint plus the new
    top-level counters."""

    def _warning_fail(_: Path) -> LintResult:
        return LintResult(
            lint_id="no-watsonx",
            verdict="fail",
            failures=[],
        )

    def _error_pass(_: Path) -> LintResult:
        return LintResult(
            lint_id="import-soundness",
            verdict="pass",
            failures=[],
        )

    monkeypatch.setattr(lints_mod, "ALL_LINTS", (_warning_fail, _error_pass))

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp)
        run_lints(pkg)
        report_path = pkg / "intermediate" / "step_7_report.json"
        data = json.loads(report_path.read_text())

    assert data["format_version"] == "1.1"
    assert "blocking_failures" in data
    assert "warnings" in data
    assert "info_failures" in data
    assert "strict" in data
    assert data["warnings"] == 1
    assert data["blocking_failures"] == 0
    assert data["overall_verdict"] == "pass"

    by_id = {entry["lint_id"]: entry for entry in data["lints"]}
    assert by_id["no-watsonx"]["severity"] == "warning"
    assert by_id["import-soundness"]["severity"] == "error"


def test_lint_result_default_severity_is_error():
    """Construct a LintResult without specifying severity → default ERROR.

    Back-compat for any consumer that builds LintResult directly without
    going through the gate's ``_apply_severity`` plumbing.
    """
    r = LintResult(lint_id="anything", verdict="pass")
    assert r.severity == LintSeverity.ERROR
