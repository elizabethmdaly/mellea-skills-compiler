"""Tests for the ``R-SEM-SIGNATURE-MATCH`` semantic rule (Phase 3.5.D).

The rule compares a descriptor's ``inputs`` / ``outputs`` / ``schemas``
against a canonical ``expected_signature.json`` artefact derived
deterministically from ``element_mapping.json`` + ``classification.json``
by Step 2 of the /mellea-fy workflow.

The rule is **non-firing when no expected_signature is supplied** so unit
tests on raw descriptors without intermediate artefacts (the dominant
case in the rest of this suite) continue to pass.
"""
from __future__ import annotations

from typing import Any

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.semantic_rules import (
    R_SIGNATURE_MATCH,
    _check_expected_signature,
)


def _base_descriptor() -> dict[str, Any]:
    """A minimal-but-valid v0.2 descriptor with one string input and one model output."""
    return {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [
            {"name": "diff", "schema": {"kind": "str"}},
        ],
        "outputs": [
            {
                "name": "findings_report",
                "schema": {"kind": "model_ref", "ref": "#/schemas/FindingsReport"},
            }
        ],
        "schemas": {
            "FindingsReport": {
                "kind": "model",
                "fields": {
                    "severity": {"type": "str"},
                    "issues": {"type": "list[str]"},
                },
            }
        },
        "state": [],
        "pipeline": [],
    }


def _base_expected_signature() -> dict[str, Any]:
    """A matching expected_signature for ``_base_descriptor``."""
    return {
        "format_version": "1.0",
        "function_name": "run_pipeline",
        "inputs": [
            {"name": "diff", "type": "str"},
        ],
        "outputs": [
            {
                "name": "findings_report",
                "type": "FindingsReport",
                "schema_ref": "#/schemas/FindingsReport",
            }
        ],
        "schemas": [
            {
                "name": "FindingsReport",
                "kind": "model",
                "fields": [
                    {"name": "severity", "type": "str"},
                    {"name": "issues", "type": "list[str]"},
                ],
            }
        ],
        "source_element_refs": ["elem_001"],
    }


# --- Happy path: matching signature → no errors ----------------------------- #


def test_matching_signature_produces_no_errors() -> None:
    """Identical descriptor + expected_signature → rule produces no errors."""
    errors = _check_expected_signature(
        descriptor=_base_descriptor(),
        expected_signature=_base_expected_signature(),
    )
    assert errors == []


def test_validate_with_matching_signature_returns_valid() -> None:
    """End-to-end through validate(): no R-SEM-SIGNATURE-MATCH errors fire."""
    report = validate(
        _base_descriptor(),
        schema_version="0.2",
        surface=None,
        expected_signature=_base_expected_signature(),
    )
    sig_errors = [e for e in report.errors if e.rule == R_SIGNATURE_MATCH]
    assert sig_errors == [], f"unexpected R-SEM-SIGNATURE-MATCH errors: {sig_errors}"


# --- Mismatched input name -------------------------------------------------- #


def test_input_name_mismatch_fails_with_specific_error() -> None:
    """Descriptor input renamed (``code`` instead of ``diff``) → rule fires
    with both a 'missing' and an 'extra' message naming the offending names.
    """
    desc = _base_descriptor()
    desc["inputs"][0]["name"] = "code"
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    rules = [e.rule for e in errors]
    assert all(r == R_SIGNATURE_MATCH for r in rules)
    messages = " ".join(e.message for e in errors)
    assert "missing required input 'diff'" in messages
    assert "extra input 'code'" in messages


def test_input_name_mismatch_via_validate_marks_invalid() -> None:
    """End-to-end: a mismatched input name produces an invalid ValidationReport."""
    desc = _base_descriptor()
    desc["inputs"][0]["name"] = "code"
    report = validate(
        desc,
        schema_version="0.2",
        surface=None,
        expected_signature=_base_expected_signature(),
    )
    assert not report.valid
    sig_errors = [e for e in report.errors if e.rule == R_SIGNATURE_MATCH]
    assert len(sig_errors) >= 2  # one missing + one extra


# --- Mismatched output type ------------------------------------------------- #


def test_output_type_mismatch_fails() -> None:
    """Descriptor output type str when expected_signature says it's a model
    → rule fires with a type-mismatch message naming both types.
    """
    desc = _base_descriptor()
    desc["outputs"][0]["schema"] = {"kind": "str"}
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    type_mismatch = [e for e in errors if "type mismatch" in e.message]
    assert type_mismatch, f"expected type-mismatch error, got: {[e.message for e in errors]}"
    msg = type_mismatch[0].message
    assert "'FindingsReport'" in msg
    assert "'str'" in msg
    assert type_mismatch[0].path == "/outputs/0/schema"


def test_input_type_mismatch_fails() -> None:
    """An input typed as ``list[str]`` when expected_signature says ``str``
    triggers a type-mismatch error pointing at /inputs/0/schema.
    """
    desc = _base_descriptor()
    desc["inputs"][0]["schema"] = {"kind": "list", "of": {"kind": "str"}}
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    type_errors = [
        e
        for e in errors
        if "type mismatch" in e.message and e.path == "/inputs/0/schema"
    ]
    assert type_errors
    assert "'list[str]'" in type_errors[0].message
    assert "'str'" in type_errors[0].message


# --- Extra input ------------------------------------------------------------ #


def test_extra_input_fails() -> None:
    """Descriptor declares an input not present in expected_signature → fail."""
    desc = _base_descriptor()
    desc["inputs"].append({"name": "verbose", "schema": {"kind": "bool"}})
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    extras = [e for e in errors if "extra input 'verbose'" in e.message]
    assert extras


# --- Missing required field ------------------------------------------------- #


def test_missing_required_schema_field_fails() -> None:
    """A required field declared in expected_signature.schemas but missing
    from the descriptor's schemas → rule fires as an error.
    """
    desc = _base_descriptor()
    # Drop the 'severity' field.
    del desc["schemas"]["FindingsReport"]["fields"]["severity"]
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    missing = [
        e
        for e in errors
        if "is missing field 'severity'" in e.message and e.severity == "error"
    ]
    assert missing, f"expected error for missing required field; got: {[e.message for e in errors]}"


def test_missing_optional_schema_field_is_warning() -> None:
    """Optional fields missing from the descriptor schema → warning, not error.

    A descriptor that legitimately drops an optional field should not block the
    repair loop hard — surface as a warning instead.
    """
    exp = _base_expected_signature()
    exp["schemas"][0]["fields"].append(
        {"name": "metadata", "type": "str", "optional": True}
    )
    desc = _base_descriptor()
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=exp,
    )
    metadata_errors = [e for e in errors if "'metadata'" in e.message]
    assert metadata_errors
    assert all(e.severity == "warning" for e in metadata_errors)


def test_extra_schema_field_fails() -> None:
    """A field declared on the descriptor schema but not in expected_signature
    → rule reports the extra field by name.
    """
    desc = _base_descriptor()
    desc["schemas"]["FindingsReport"]["fields"]["secret"] = {"type": "str"}
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    extras = [
        e
        for e in errors
        if "declares extra field 'secret'" in e.message
    ]
    assert extras
    assert extras[0].path == "/schemas/FindingsReport/fields/secret"


def test_schema_field_type_mismatch_fails() -> None:
    """When a field exists on both sides but the types differ, the rule reports
    a type-mismatch error citing both types and the field path.
    """
    desc = _base_descriptor()
    # change list[str] to str
    desc["schemas"]["FindingsReport"]["fields"]["issues"] = {"type": "str"}
    errors = _check_expected_signature(
        descriptor=desc,
        expected_signature=_base_expected_signature(),
    )
    type_errors = [
        e
        for e in errors
        if "type mismatch" in e.message
        and e.path == "/schemas/FindingsReport/fields/issues"
    ]
    assert type_errors
    msg = type_errors[0].message
    assert "'list[str]'" in msg
    assert "'str'" in msg


# --- Non-firing when expected_signature is None ----------------------------- #


def test_rule_non_firing_when_expected_signature_is_none() -> None:
    """When no expected_signature is passed, the rule must be silent — this
    is the contract that lets the rest of the descriptor test suite (which
    has no expected_signature) continue to pass.
    """
    report = validate(
        _base_descriptor(),
        schema_version="0.2",
        surface=None,
        expected_signature=None,
    )
    sig_errors = [e for e in report.errors if e.rule == R_SIGNATURE_MATCH]
    assert sig_errors == []


def test_rule_non_firing_when_expected_signature_is_missing_default() -> None:
    """Default-call (no expected_signature kwarg) is silent — the rule is opt-in."""
    report = validate(_base_descriptor(), schema_version="0.2")
    sig_errors = [e for e in report.errors if e.rule == R_SIGNATURE_MATCH]
    assert sig_errors == []


# --- Repair-loop demonstration --------------------------------------------- #


def test_repair_loop_trigger_message_is_specific_and_actionable() -> None:
    """The rule's error messages must contain enough specifics for the repair
    loop to point the LLM at the precise field to fix. This test demonstrates
    the trigger: descriptor emits ``code`` where expected_signature says
    ``diff`` → R-SEM-SIGNATURE-MATCH fires with both names visible.
    """
    desc = _base_descriptor()
    desc["inputs"][0]["name"] = "code"
    report = validate(
        desc,
        schema_version="0.2",
        surface=None,
        expected_signature=_base_expected_signature(),
    )
    assert not report.valid
    sig_errors = [e for e in report.errors if e.rule == R_SIGNATURE_MATCH]
    joined = " | ".join(e.message for e in sig_errors)
    # The repair prompt needs both the expected name and the offending one
    # so it can produce a targeted correction.
    assert "'diff'" in joined
    assert "'code'" in joined
