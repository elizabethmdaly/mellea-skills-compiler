"""Tests for the v0.1 schema snapshot.

The v0.1 snapshot is a frozen JSON Schema that captures the Phase 0
strawman as the Phase 0.2 hand-authored descriptors used it. These tests
guard against accidental drift: any failure here likely means a Phase 0.2
descriptor would stop validating, which would be a backwards-compatibility
break.
"""
from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from mellea_skills_compiler.descriptor import load_schema


def _minimal_v01_descriptor() -> dict:
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }


def test_v01_schema_loads():
    schema = load_schema("0.1")
    assert schema["$id"].endswith("descriptor.v0.1.json")
    assert schema["properties"]["descriptor_version"]["const"] == "0.1"


def test_v01_schema_self_validates():
    """The schema itself must be a structurally valid Draft-2020-12 schema."""
    Draft202012Validator.check_schema(load_schema("0.1"))


def test_v01_minimal_descriptor_is_valid():
    schema = load_schema("0.1")
    Draft202012Validator(schema).validate(_minimal_v01_descriptor())


def test_v01_rejects_descriptor_version_0_2():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["descriptor_version"] = "0.2"
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(desc)


def test_v01_rejects_filter_operator():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["pipeline"] = [
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "doc"},
            "item_id": "x",
            "predicate": {"kind": "field_truthy", "select": "y"},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 must reject the 'filter' operator (v0.2-only)."


def test_v01_rejects_on_parse_failure():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
            "on_parse_failure": {"strategy": "raise"},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 must reject on_parse_failure (v0.2-only)."


def test_v01_rejects_capabilities():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["capabilities"] = {"tools": {"allowed": []}}
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 must reject the top-level capabilities block."


def test_v01_rejects_top_level_notes_without_underscore():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    # 'notes' (no underscore) is a v0.2 field.
    desc["notes"] = ["a note"]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 should not silently accept v0.2's 'notes' field."


def test_v01_accepts_underscore_notes():
    """v0.1 hand-authored descriptors carried `_notes`; the snapshot preserves that."""
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["_notes"] = ["author commentary"]
    Draft202012Validator(schema).validate(desc)


def test_v01_state_rejects_scope():
    """`scope` is a v0.2-only field on state entries."""
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["state"] = [
        {
            "id": "backend",
            "symbol": "mellea.backends.openai.OpenAIBackend",
            "scope": "function",
            "args": {"model_id": {"env": "MODEL_ID"}},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 must reject `scope` on state entries."


def test_v01_select_universal_on_ref():
    """Even though the v0.1 doc only declared `select` on branch.on, the
    Phase 0.2 descriptors used it on map.over too. The snapshot allows it
    everywhere so those descriptors validate."""
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["pipeline"] = [
        {
            "id": "first",
            "kind": "call",
            "symbol": "x.y",
            "args": {"a": {"ref": "doc", "select": "x"}},
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v01_pattern_for_id_rejects_uppercase():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["pipeline"] = [
        {"id": "BadId", "kind": "call", "symbol": "x.y", "args": {}}
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors, "v0.1 must reject ids with uppercase letters."


def test_v01_map_requires_item_id_and_collect():
    schema = load_schema("0.1")
    desc = _minimal_v01_descriptor()
    desc["pipeline"] = [
        {
            "id": "m",
            "kind": "composition",
            "operator": "map",
            "over": {"ref": "doc"},
            "body": [],
            # missing item_id and collect
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors
