"""End-to-end :func:`validate` tests against the public :class:`ValidationReport` API."""
from __future__ import annotations

import pytest

from mellea_skills_compiler.descriptor import (
    SUPPORTED_SCHEMA_VERSIONS,
    ValidationError,
    ValidationReport,
    load_schema,
    validate,
)


def _minimal(version: str = "0.2") -> dict:
    return {
        "descriptor_version": version,
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }


def test_supported_schema_versions_exposed():
    assert "0.1" in SUPPORTED_SCHEMA_VERSIONS
    assert "0.2" in SUPPORTED_SCHEMA_VERSIONS


def test_load_schema_unknown_version_raises():
    with pytest.raises(ValueError):
        load_schema("9.9")


def test_validate_minimal_v02_descriptor():
    report = validate(_minimal("0.2"), schema_version="0.2")
    assert isinstance(report, ValidationReport)
    assert report.valid
    assert report.errors == []
    assert report.descriptor_version == "0.2"
    assert report.schema_version == "0.2"


def test_validate_minimal_v01_descriptor():
    report = validate(_minimal("0.1"), schema_version="0.1")
    assert report.valid
    assert report.schema_version == "0.1"


def test_validate_unsupported_schema_version_returns_report():
    report = validate(_minimal("0.2"), schema_version="9.9")
    assert not report.valid
    # Validator emits "schema_version 'X' not supported; expected one of ..."
    assert any(
        ("unsupported" in e.message.lower() or "not supported" in e.message.lower())
        for e in report.errors
    )


def test_validate_descriptor_missing_required_field():
    desc = _minimal()
    del desc["skill"]
    report = validate(desc, schema_version="0.2")
    assert not report.valid
    # JSON-Schema layer should report a `required` rule violation.
    assert any(e.rule.startswith("jsonschema:") for e in report.errors)


def test_validate_error_path_is_json_pointer():
    """JSON Pointer paths let consumers highlight the exact location."""
    desc = _minimal()
    # Inject a bad nested value: pipeline node missing required `args`.
    desc["pipeline"] = [{"id": "c", "kind": "call", "symbol": "x.y"}]
    report = validate(desc, schema_version="0.2")
    assert not report.valid
    # At least one error should have a JSON-pointer style path under /pipeline/0/
    pipeline_errors = [e for e in report.errors if e.path.startswith("/pipeline/0")]
    assert pipeline_errors


def test_validate_returns_structured_errors_not_strings():
    desc = _minimal()
    desc["descriptor_version"] = "0.2"
    desc["skill"]["classification"] = "NOT_A_VALID_CLASSIFICATION"
    report = validate(desc, schema_version="0.2")
    assert not report.valid
    for err in report.errors:
        assert isinstance(err, ValidationError)
        assert hasattr(err, "path")
        assert hasattr(err, "rule")
        assert hasattr(err, "message")
        assert err.severity in ("error", "warning")


def test_validate_with_warning_severity_keeps_valid_true():
    """A warning is not an error: validity is determined by error-severity entries."""
    surface = {
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {
                    "MelleaSession": {
                        "kind": "class",
                        "members": {"instruct": {"kind": "method"}},
                    }
                }
            }
        }
    }
    desc = _minimal("0.2")
    desc["state"] = [
        {
            "id": "session",
            "symbol": "mellea.stdlib.session.MelleaSession",
            "args": {},
        }
    ]
    desc["schemas"] = {
        "M": {
            "kind": "model",
            "fields": {
                "a": {"type": "str"},
                "b": {"type": "str"},
            },
        }
    }
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/M"}},
            "on_parse_failure": {
                "strategy": "default",
                "default": {"a": "x"},  # missing required field 'b' -> warning
            },
        }
    ]
    report = validate(desc, schema_version="0.2", surface=surface)
    # Should still be valid (warnings only), even though report.errors is non-empty.
    assert report.valid
    assert any(e.severity == "warning" for e in report.errors)


def test_descriptor_version_echoed_back():
    desc = _minimal("0.2")
    report = validate(desc, schema_version="0.2")
    assert report.descriptor_version == "0.2"


def test_descriptor_non_dict_input_is_handled_gracefully():
    """A non-dict input shouldn't crash the validator outright."""
    # Passing a list — JSON schema-level type check should reject it.
    report = validate([], schema_version="0.2")  # type: ignore[arg-type]
    assert not report.valid


def test_acceptance_criterion_1_known_good_descriptor(tmp_path):
    """Acceptance criterion 1: a known-good v0.1 descriptor passes."""
    import json
    from pathlib import Path

    descriptor_path = (
        Path(__file__).parents[3]
        / "melleafy-handoff"
        / "kickoff"
        / "spike-outputs"
        / "descriptors"
        / "sentry-find-bugs.descriptor.json"
    )
    surface_path = (
        Path(__file__).parents[3]
        / "melleafy-handoff"
        / "kickoff"
        / "spike-outputs"
        / "surface_0.5.0.json"
    )
    descriptor = json.loads(descriptor_path.read_text())
    surface = json.loads(surface_path.read_text())

    report = validate(descriptor, schema_version="0.1", surface=surface)
    assert report.valid, [e for e in report.errors if e.severity == "error"]


def test_acceptance_criterion_2_broken_descriptor():
    """Acceptance criterion 2: a deliberately-broken descriptor produces a
    structured error at the right JSON Pointer path."""
    desc = _minimal("0.1")
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "nonexistent.module.Symbol",
            "args": {"x": {"ref": "no_such_id"}},
        }
    ]
    surface = {
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {"MelleaSession": {"members": {"instruct": {}}}}
            }
        }
    }
    report = validate(desc, schema_version="0.1", surface=surface)
    assert not report.valid
    # Should include a ref-resolution error pointing at the bad ref site.
    ref_errors = [
        e
        for e in report.errors
        if "ref" in e.path and e.severity == "error"
    ]
    assert ref_errors
    assert any(e.path.startswith("/pipeline/0/args/x") for e in ref_errors)
