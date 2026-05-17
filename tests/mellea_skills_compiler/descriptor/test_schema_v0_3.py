"""Schema-level tests for descriptor schema v0.3.

These exercise the new shape on a per-feature basis: ``dependencies[]``,
``bundled_resources[]``, the expanded classification block, the
``agent_loop`` and ``human_approval`` operators, the input/output
permissiveness additions, and ``source_elements[]``. The semantic-rule
tests (R-SEM-*) live in their own files; here we pin only the JSON-Schema
acceptance/rejection envelope.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from mellea_skills_compiler.descriptor import load_schema, validate


REPO_ROOT = Path(__file__).resolve().parents[3]


def _minimal_v03() -> dict[str, Any]:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }


# ---------------------------------------------------------------------------
# Schema self-validation + minimal acceptance.
# ---------------------------------------------------------------------------


def test_v03_schema_loads_and_self_validates():
    schema = load_schema("0.3")
    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("descriptor.v0.3.json")


def test_v03_minimal_descriptor_is_valid():
    schema = load_schema("0.3")
    Draft202012Validator(schema).validate(_minimal_v03())


def test_v03_descriptor_version_must_be_03():
    desc = _minimal_v03()
    desc["descriptor_version"] = "0.2"
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "expected JSON-Schema-level rejection for wrong version const"


# ---------------------------------------------------------------------------
# §3.1 dependencies[]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "disposition,extra",
    [
        ("delegate_to_runtime", {"signature": "(query: str) -> list[str]"}),
        ("real_impl", {"signature": "(x: int) -> int"}),
        ("stub", {"signature": "(content: str) -> None"}),
        ("mock", {"signature": "() -> bool"}),
        ("external_input", {"signature": "() -> str"}),
        ("bundle", {"path": "references/owasp.md"}),
        ("load_from_disk", {"path": "references/owasp.md"}),
        ("remove", {}),
    ],
)
def test_dependencies_each_disposition_accepted(disposition, extra):
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "dep_one", "kind": "tool" if "signature" in extra else "file",
         "disposition": disposition, **extra}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors, errors


def test_dependencies_invalid_disposition_rejected():
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "x", "kind": "tool", "disposition": "wibble"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_dependencies_bundle_without_path_rejected():
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "x", "kind": "file", "disposition": "bundle"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "bundle disposition without path must be rejected"


def test_dependencies_id_pattern_enforced():
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "Bad-ID", "kind": "tool", "disposition": "remove"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_dependencies_unknown_kind_rejected():
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "x", "kind": "unicorn", "disposition": "remove"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_dependencies_additional_properties_rejected():
    desc = _minimal_v03()
    desc["dependencies"] = [
        {"id": "x", "kind": "tool", "disposition": "remove", "wibble": True}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---------------------------------------------------------------------------
# §3.2 bundled_resources[]
# ---------------------------------------------------------------------------


def test_bundled_resources_minimal_accepted():
    desc = _minimal_v03()
    desc["bundled_resources"] = [
        {"source_dir": "references", "package_dir": "references"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_bundled_resources_missing_required_rejected():
    desc = _minimal_v03()
    desc["bundled_resources"] = [{"source_dir": "references"}]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_bundled_resources_glob_filter_accepted():
    desc = _minimal_v03()
    desc["bundled_resources"] = [
        {"source_dir": "scripts", "package_dir": "scripts",
         "is_package_data": True, "include_glob": "*.sh"}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_bundled_resources_multiple_accepted():
    desc = _minimal_v03()
    desc["bundled_resources"] = [
        {"source_dir": "references", "package_dir": "references"},
        {"source_dir": "scripts", "package_dir": "scripts"},
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_bundled_resources_additional_properties_rejected():
    desc = _minimal_v03()
    desc["bundled_resources"] = [
        {"source_dir": "r", "package_dir": "r", "garbage": True}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---------------------------------------------------------------------------
# §3.3 agent_loop + human_approval operators
# ---------------------------------------------------------------------------


def _agent_loop_node() -> dict[str, Any]:
    return {
        "id": "loop_one",
        "kind": "composition",
        "operator": "agent_loop",
        "max_iterations": 5,
        "state_schema": {"ref": "#/schemas/LoopState"},
        "termination_condition": {"predicate": "field_truthy", "field": "done"},
        "body": [
            {
                "id": "body_one",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "args": {"description": {"value": "ping"}},
            }
        ],
    }


def test_agent_loop_minimal_accepted():
    desc = _minimal_v03()
    desc["schemas"]["LoopState"] = {
        "kind": "model",
        "fields": {"done": {"type": "bool"}},
    }
    desc["pipeline"] = [_agent_loop_node()]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors, errors


def test_agent_loop_termination_each_predicate():
    desc = _minimal_v03()
    desc["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "str"}}}
    for predicate in [
        {"predicate": "field_truthy", "field": "x"},
        {"predicate": "field_equals", "field": "x", "value": "y"},
        {"predicate": "field_in", "field": "x", "values": ["a", "b"]},
    ]:
        node = _agent_loop_node()
        node["state_schema"] = {"ref": "#/schemas/S"}
        node["termination_condition"] = predicate
        desc["pipeline"] = [node]
        schema = load_schema("0.3")
        errors = list(Draft202012Validator(schema).iter_errors(desc))
        assert not errors, (predicate, errors)


def test_agent_loop_max_iterations_bounds():
    schema = load_schema("0.3")
    for bad in (0, -1, 1001):
        node = _agent_loop_node()
        node["max_iterations"] = bad
        desc = _minimal_v03()
        desc["schemas"]["LoopState"] = {
            "kind": "model", "fields": {"done": {"type": "bool"}}
        }
        desc["pipeline"] = [node]
        errors = list(Draft202012Validator(schema).iter_errors(desc))
        assert errors, f"max_iterations={bad} should be rejected"


def test_human_approval_minimal_accepted():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "gate",
            "kind": "composition",
            "operator": "human_approval",
            "prompt": {"value": "approve?"},
            "branches": {"approved": [], "rejected": []},
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors, errors


def test_human_approval_missing_branches_rejected():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "gate",
            "kind": "composition",
            "operator": "human_approval",
            "prompt": {"value": "?"},
            "branches": {"approved": []},
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_human_approval_default_branch_optional():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "gate",
            "kind": "composition",
            "operator": "human_approval",
            "prompt": {"value": "?"},
            "branches": {"approved": [], "rejected": []},
            "default_branch": "approved",
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


# ---------------------------------------------------------------------------
# §3.4 classification + per-call streaming + image/document inputs + async
# ---------------------------------------------------------------------------


def test_classification_primary_axis_closed_vocab():
    schema = load_schema("0.3")
    for v in ("DSL", "COMP", "DOM", "AGENT", "PERSONA", "MIXED"):
        desc = _minimal_v03()
        desc["skill"]["classification"]["primary_axis"] = v
        errors = list(Draft202012Validator(schema).iter_errors(desc))
        assert not errors


def test_classification_primary_axis_invalid_rejected():
    desc = _minimal_v03()
    desc["skill"]["classification"]["primary_axis"] = "WHATEVER"
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_classification_state_persistence_cross_session_accepted():
    """v0.3: cross_session is declarative-only; schema accepts it."""
    desc = _minimal_v03()
    desc["skill"]["classification"]["state_persistence"] = "cross_session"
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_skill_async_flag_accepted():
    desc = _minimal_v03()
    desc["skill"]["async"] = True
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_call_streaming_flag_accepted():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "speak",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {"description": {"value": "say hi"}},
            "streaming": True,
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_input_kind_image_accepted():
    desc = _minimal_v03()
    desc["inputs"] = [{"name": "photo", "schema": {"kind": "image"}}]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_input_kind_document_accepted():
    desc = _minimal_v03()
    desc["inputs"] = [
        {"name": "policy", "schema": {"kind": "document", "format_hint": "pdf"}}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


# ---------------------------------------------------------------------------
# §3.5 input/output permissiveness — dict, literal_union, optional, description
# ---------------------------------------------------------------------------


def test_input_dict_str_to_str_accepted():
    desc = _minimal_v03()
    desc["inputs"] = [
        {"name": "options", "schema": {"kind": "dict", "key_type": "str",
                                        "value_type": "str"}}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_input_literal_union_accepted():
    desc = _minimal_v03()
    desc["inputs"] = [
        {"name": "color", "schema": {"kind": "literal_union",
                                      "members": ["red", "green", "blue"]}}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_slot_optional_accepted():
    desc = _minimal_v03()
    desc["inputs"] = [
        {"name": "context", "schema": {"kind": "str"}, "optional": True}
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_schema_field_description_accepted():
    desc = _minimal_v03()
    desc["schemas"]["R"] = {
        "kind": "model",
        "fields": {"v": {"type": "str", "description": "the verdict"}},
    }
    desc["outputs"] = [{"name": "result", "schema": {"kind": "model_ref",
                                                       "ref": "#/schemas/R"}}]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


# ---------------------------------------------------------------------------
# §3.6 source_elements on pipeline nodes
# ---------------------------------------------------------------------------


def test_source_elements_on_call_accepted():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "n1",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {"description": {"value": "x"}},
            "source_elements": ["E14", "E15"],
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert not errors


def test_source_elements_pattern_enforced():
    desc = _minimal_v03()
    desc["pipeline"] = [
        {
            "id": "n1",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {"description": {"value": "x"}},
            "source_elements": ["e14"],  # lowercase — rejected
        }
    ]
    schema = load_schema("0.3")
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---------------------------------------------------------------------------
# Backward-compat: bumping v0.2 fixtures' version to "0.3" still validates.
# ---------------------------------------------------------------------------


V02_FIXTURE_DIR = (
    REPO_ROOT
    / "tests"
    / "mellea_skills_compiler"
    / "descriptor"
    / "fixtures"
    / "v02"
)


@pytest.mark.parametrize(
    "fixture_path",
    sorted(V02_FIXTURE_DIR.glob("*.descriptor.json")),
    ids=lambda p: p.name,
)
def test_v02_fixtures_validate_under_v03_after_classification_uplift(fixture_path):
    """A v0.2 fixture, with descriptor_version bumped to 0.3 and the
    classification string uplifted to the structured block, must validate.

    This is the "backward compatibility constraint" from the RFC: any valid
    v0.1/v0.2 descriptor remains valid against v0.3 modulo the explicit shape
    changes (classification string → object).
    """
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    bumped = copy.deepcopy(raw)
    bumped["descriptor_version"] = "0.3"
    classification = bumped.get("skill", {}).get("classification")
    if isinstance(classification, str):
        bumped["skill"]["classification"] = {"primary_axis": classification}
    report = validate(bumped, schema_version="0.3")
    assert report.valid, [(e.path, e.message) for e in report.errors]


def test_v02_fixtures_still_validate_under_v02():
    """Verify v0.2 fixtures left at version 0.2 still validate via v0.2 route."""
    for fixture_path in sorted(V02_FIXTURE_DIR.glob("*.descriptor.json")):
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        report = validate(raw, schema_version="0.2")
        assert report.valid, (
            f"{fixture_path.name}: " +
            ", ".join(f"{e.path}: {e.message}" for e in report.errors)
        )
