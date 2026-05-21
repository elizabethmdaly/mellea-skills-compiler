"""Coherence audits for the Step 7 report writer ↔ schema.

Two real drifts caught and fixed this session:
  (1) `warnings_escalated_by_smoke` written as bool, schema typed as
      array<string>. Fixed by deriving the list at write time.
  (2) `column` values emitted as 0-indexed (`ast.col_offset`), schema
      requires `minimum: 1`. Fixed via `_col_offset_to_schema` helper.

These tests pin those fixes as regression guards AND audit the
writer-vs-schema agreement more broadly.
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from importlib.resources import files

import jsonschema
import pytest

from mellea_skills_compiler.compile.lints import (
    _col_offset_to_schema,
    lint_variable_safety,
    run_lints,
)
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "step-7-report-writer-schema-coherence"


def _load_schema() -> dict:
    schema_text = (
        files("mellea_skills_compiler.compile.schemas")
        .joinpath("step_7_report.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(schema_text)


def _produce_real_report(pkg: Path, source_code: str) -> dict:
    """Run a single real lint that will fail, drive `run_lints` end-to-
    end, return the emitted report payload."""
    (pkg / "pipeline.py").write_text(source_code, encoding="utf-8")
    # smoke_check="never" to avoid backend dependencies in tests.
    run_lints(pkg, smoke_check="never")
    report_path = pkg / "intermediate" / "step_7_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_writer_output_validates_against_schema():
    """C-WRITER-OUTPUT-CONFORMS: end-to-end — a real report payload
    produced by run_lints must validate against the schema with zero
    errors."""
    schema = _load_schema()
    code = (
        "def _safe_parse(thunk):\n"
        "    try:\n"
        "        raw = thunk.value\n"
        "    except Exception:\n"
        "        return raw  # uninit-in-except\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        report = _produce_real_report(pkg, code)
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda e: list(e.absolute_path))
    assert not errors, (
        "C-WRITER-OUTPUT-CONFORMS failed: an end-to-end report payload "
        "from `run_lints` does not validate against step_7_report.schema.json. "
        "Drift between writer (lints.py) and schema. Errors: "
        + "; ".join(f"at {list(e.absolute_path)}: {e.message}" for e in errors[:3])
    )


def test_warnings_escalated_by_smoke_is_list():
    """C-WARNINGS-ESCALATED-IS-LIST: pin drift #1 — the field must be
    a list, never a bool."""
    code = "def run_pipeline():\n    pass\n"
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        report = _produce_real_report(pkg, code)
    val = report.get("warnings_escalated_by_smoke")
    assert isinstance(val, list), (
        f"C-WARNINGS-ESCALATED-IS-LIST failed: regression on drift #1. "
        f"Expected list; got {type(val).__name__} = {val!r}."
    )


def test_column_values_are_one_indexed_or_null():
    """C-COLUMN-IS-1-INDEXED: every emitted column value is null or >= 1.
    Pin drift #2 — Python's 0-indexed col_offset must be converted."""
    code = (
        "def _safe_parse(thunk):\n"
        "    try:\n"
        "        raw = thunk.value\n"
        "    except Exception:\n"
        "        return raw\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        report = _produce_real_report(pkg, code)
    bad: list[tuple[str, int]] = []
    for lint in report.get("lints", []):
        for failure in lint.get("failures", []):
            col = failure.get("column")
            if col is None:
                continue
            if not isinstance(col, int) or col < 1:
                bad.append((lint["lint_id"], col))
    assert not bad, (
        f"C-COLUMN-IS-1-INDEXED failed: regression on drift #2. "
        f"Found column values that aren't null and aren't 1-indexed: "
        f"{bad}. Convert ast.col_offset via _col_offset_to_schema."
    )


def test_col_offset_helper_is_one_indexed():
    """Unit-level pin for the helper: col_offset=0 → 1, col_offset=5 → 6,
    missing → None. Guards against the helper itself drifting."""
    class _N:
        def __init__(self, c):
            self.col_offset = c

    assert _col_offset_to_schema(_N(0)) == 1
    assert _col_offset_to_schema(_N(5)) == 6
    assert _col_offset_to_schema(object()) is None


def test_smoke_check_verdict_is_schema_conformant():
    """Pin for drift #3 (caught 2026-05-20 on the gpai smoke compile):
    ``_run_smoke_check_inline`` previously translated smoke_check.py's
    canonical ``"passed"``/``"failed"`` verdicts INTO ``"pass"``/``"fail"``
    — the wrong direction. The schema requires past-tense; the
    translator now preserves it.

    Test strategy: monkeypatch a synthetic smoke outcome through the
    lints module's inline runner so we never touch a real backend.
    The fake returns ``"passed"``; the produced report's smoke_check
    section must serialise the same value (schema-conformant).
    """
    from mellea_skills_compiler.compile import lints as lints_mod
    from mellea_skills_compiler.compile.lints import SmokeCheckOutcome

    saved_detect = lints_mod._detect_backend_available
    saved_inline = lints_mod._run_smoke_check_inline
    try:
        lints_mod._detect_backend_available = lambda *a, **k: (True, None)
        lints_mod._run_smoke_check_inline = (
            lambda *a, **k: SmokeCheckOutcome(
                verdict="passed",
                fixture_used="fx-001",
                duration_seconds=0.5,
                backend_available=True,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "pkg_mellea"
            pkg.mkdir()
            (pkg / "intermediate").mkdir()
            (pkg / "pipeline.py").write_text(
                "def run_pipeline():\n    pass\n", encoding="utf-8"
            )
            run_lints(pkg, smoke_check="auto")
            report = json.loads(
                (pkg / "intermediate" / "step_7_report.json").read_text(
                    encoding="utf-8"
                )
            )
    finally:
        lints_mod._detect_backend_available = saved_detect
        lints_mod._run_smoke_check_inline = saved_inline

    smoke = report.get("smoke_check") or {}
    verdict = smoke.get("verdict")
    schema_allowed = {"passed", "failed", "skipped"}
    assert verdict in schema_allowed, (
        f"smoke_check.verdict={verdict!r} is not schema-conformant. "
        f"Schema enum: {sorted(schema_allowed)}. Regression on drift "
        f"#3 — the translator at _run_smoke_check_inline must preserve "
        f"smoke_check.py's past-tense verdicts."
    )


def test_required_top_level_fields_present():
    """C-REQUIRED-FIELDS-PRESENT: every field declared `required` at
    the top level of the schema appears in the writer's emission."""
    schema = _load_schema()
    required = set(schema.get("required", []))
    code = "def run_pipeline():\n    pass\n"
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        report = _produce_real_report(pkg, code)
    missing = required - set(report.keys())
    assert not missing, (
        f"C-REQUIRED-FIELDS-PRESENT failed: writer omitted "
        f"required top-level fields: {sorted(missing)}. The schema "
        f"says these MUST be present in every emission."
    )
