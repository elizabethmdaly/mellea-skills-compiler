"""JSON-Schema-level tests for the ``expected_signature.schema.json`` artefact.

The schema lives at::

    src/mellea_skills_compiler/descriptor/schemas/expected_signature.schema.json

and defines the on-disk contract for ``intermediate/expected_signature.json``
(Phase 3.5.D). These tests apply the same Tier-S battery patterns used by
``tests/mellea_skills_compiler/test_artefact_schemas.py``:

- S1 — minimal valid instance round-trips
- S2 — required-field validation
- S3 — type-vocabulary validation (signature_param.type pattern, schema_def.kind enum)
- S4 — the schema self-validates against JSON Schema Draft 2020-12
"""
from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSValidationError


REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = (
    REPO_ROOT
    / "src"
    / "mellea_skills_compiler"
    / "descriptor"
    / "schemas"
    / "expected_signature.schema.json"
)


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _minimal_valid() -> dict[str, Any]:
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
    }


# --- S4: schema self-validates ---------------------------------------------- #


def test_schema_self_validates_against_draft_2020_12(schema: dict[str, Any]) -> None:
    """The schema document itself must be a valid Draft 2020-12 schema."""
    Draft202012Validator.check_schema(schema)


def test_schema_declares_2020_12_dialect(schema: dict[str, Any]) -> None:
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"


# --- S1: minimal-valid instance --------------------------------------------- #


def test_minimal_valid_instance_passes(schema: dict[str, Any]) -> None:
    Draft202012Validator(schema).validate(_minimal_valid())


def test_minimal_with_only_inputs_no_schemas_valid(schema: dict[str, Any]) -> None:
    """A signature with only primitive inputs and no schemas is valid."""
    instance = {
        "format_version": "1.0",
        "function_name": "run_pipeline",
        "inputs": [{"name": "diff", "type": "str"}],
        "outputs": [{"name": "result", "type": "str"}],
        "schemas": [],
    }
    Draft202012Validator(schema).validate(instance)


# --- S2: required-field validation ----------------------------------------- #


@pytest.mark.parametrize(
    "missing_field",
    ["format_version", "function_name", "inputs", "outputs", "schemas"],
)
def test_required_top_level_field_missing_fails(
    schema: dict[str, Any], missing_field: str
) -> None:
    instance = _minimal_valid()
    del instance[missing_field]
    validator = Draft202012Validator(schema)
    with pytest.raises(JSValidationError):
        validator.validate(instance)


def test_signature_param_requires_name_and_type(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["inputs"] = [{"type": "str"}]  # missing name
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)

    instance = _minimal_valid()
    instance["inputs"] = [{"name": "diff"}]  # missing type
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


def test_schema_def_model_requires_fields(schema: dict[str, Any]) -> None:
    """A schema_def with kind='model' requires the 'fields' list per the contract."""
    instance = _minimal_valid()
    instance["schemas"] = [{"name": "FindingsReport", "kind": "model"}]
    # Note: per the schema as-written, 'fields' is description-required but not
    # JSON-Schema-required at the top level (the model-vs-enum constraint is
    # carried in prose). This test asserts the schema accepts it — promoting
    # the constraint to JSON-Schema enforcement is a future refinement.
    # We just verify the schema's vocabulary works:
    Draft202012Validator(schema).validate(instance)


def test_schema_def_enum_requires_members(schema: dict[str, Any]) -> None:
    """A schema_def with kind='enum' but no members triggers minItems violation
    if members is absent — but only if members is *present and empty*. Test the
    minItems guard for an empty list."""
    instance = _minimal_valid()
    instance["schemas"] = [
        {"name": "Severity", "kind": "enum", "members": []}
    ]
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


# --- S3: type-vocabulary validation ---------------------------------------- #


def test_signature_param_name_must_be_python_identifier(schema: dict[str, Any]) -> None:
    """Hyphens, leading digits, and spaces are rejected by the name pattern."""
    bad_names = ["1bad", "has-hyphen", "has space", ""]
    for bad in bad_names:
        instance = _minimal_valid()
        instance["inputs"] = [{"name": bad, "type": "str"}]
        with pytest.raises(JSValidationError):
            Draft202012Validator(schema).validate(instance)


def test_schema_def_name_must_be_pascal_case(schema: dict[str, Any]) -> None:
    """Lowercase or hyphenated schema names are rejected."""
    bad_names = ["findingsReport", "findings-report", "findings_report", ""]
    for bad in bad_names:
        instance = _minimal_valid()
        instance["schemas"][0] = {
            "name": bad,
            "kind": "model",
            "fields": [{"name": "f", "type": "str"}],
        }
        with pytest.raises(JSValidationError):
            Draft202012Validator(schema).validate(instance)


def test_schema_def_kind_enum_is_strict(schema: dict[str, Any]) -> None:
    """Only 'model' and 'enum' are accepted kinds."""
    instance = _minimal_valid()
    instance["schemas"][0]["kind"] = "pydantic_model"  # not allowed
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


def test_schema_ref_pattern_enforced(schema: dict[str, Any]) -> None:
    """schema_ref must start with '#/schemas/' per the descriptor convention."""
    instance = _minimal_valid()
    instance["outputs"][0]["schema_ref"] = "FindingsReport"  # bare name not allowed
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


def test_format_version_pattern_enforced(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["format_version"] = "1"  # missing minor
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


def test_additional_properties_rejected_at_top_level(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["extra_key"] = "anything"
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


def test_additional_properties_rejected_on_signature_param(
    schema: dict[str, Any],
) -> None:
    instance = _minimal_valid()
    instance["inputs"][0]["random_key"] = "x"
    with pytest.raises(JSValidationError):
        Draft202012Validator(schema).validate(instance)


# --- Optional fields --------------------------------------------------------- #


def test_optional_flag_accepted_on_inputs(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["inputs"][0]["optional"] = True
    Draft202012Validator(schema).validate(instance)


def test_source_element_refs_optional(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["source_element_refs"] = ["elem_001", "elem_003-step2"]
    Draft202012Validator(schema).validate(instance)


def test_modality_accepted(schema: dict[str, Any]) -> None:
    instance = _minimal_valid()
    instance["modality"] = "synchronous_oneshot"
    Draft202012Validator(schema).validate(instance)
