"""Tests for the v0.2 schema (Phase 1.B RFC Appendix A).

These tests are the JSON-Schema-level acceptance check for every
**ADDRESS** item from the RFC summary:

* §3.2 ``select`` on every ``ref`` site.
* §3.3 ``state[].scope: "module" | "function"``.
* §3.4 ``CallNode.on_parse_failure: { strategy, default? }``.
* §3.5 ``filter`` composition operator + closed predicate set.
* §3.6 top-level ``capabilities.tools.allowed/denied``.
* §3.7 literal-value ``{ "value": ... }`` for state args (already in v0.1; v0.2
  confirms it is honoured everywhere).
* §3.11 top-level ``notes`` array.

The v0.2 schema must self-validate against Draft-2020-12 (acceptance
criterion 3).
"""
from __future__ import annotations

from jsonschema import Draft202012Validator

from mellea_skills_compiler.descriptor import load_schema


def _minimal_v02_descriptor() -> dict:
    return {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }


def test_v02_schema_loads():
    schema = load_schema("0.2")
    assert schema["$id"].endswith("descriptor.v0.2.json")
    assert schema["properties"]["descriptor_version"]["const"] == "0.2"


def test_v02_schema_self_validates():
    """Acceptance criterion 3: v0.2 schema passes Draft202012 self-check."""
    Draft202012Validator.check_schema(load_schema("0.2"))


def test_v02_minimal_descriptor_is_valid():
    schema = load_schema("0.2")
    Draft202012Validator(schema).validate(_minimal_v02_descriptor())


def test_v02_rejects_v01_version_string():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["descriptor_version"] = "0.1"
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---- §3.2 select on refs --------------------------------------------------- #


def test_v02_select_on_map_over():
    """§3.2: select is now formally permitted on map.over (not just branch.on)."""
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "extract",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {"format": {"schema_ref": "#/schemas/Foo"}},
        },
        {
            "id": "m",
            "kind": "composition",
            "operator": "map",
            "over": {"ref": "extract", "select": "files"},
            "item_id": "f",
            "collect": "list",
            "body": [],
        },
    ]
    desc["schemas"] = {
        "Foo": {"kind": "model", "fields": {"files": {"type": "list[str]"}}}
    }
    Draft202012Validator(schema).validate(desc)


def test_v02_select_on_branch_on():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "classify",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
        },
        {
            "id": "br",
            "kind": "composition",
            "operator": "branch",
            "on": {"ref": "classify", "select": "severity"},
            "cases": {"_default": []},
        },
    ]
    Draft202012Validator(schema).validate(desc)


# ---- §3.3 state scope ------------------------------------------------------ #


def test_v02_state_scope_module():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["state"] = [
        {
            "id": "backend",
            "symbol": "mellea.backends.openai.OpenAIBackend",
            "scope": "module",
            "args": {"model_id": {"env": "MODEL_ID"}},
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_state_scope_function():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["state"] = [
        {
            "id": "backend",
            "symbol": "mellea.backends.openai.OpenAIBackend",
            "scope": "function",
            "args": {},
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_state_scope_rejects_unknown():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["state"] = [
        {
            "id": "backend",
            "symbol": "x.y",
            "scope": "global",
            "args": {},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---- §3.4 on_parse_failure ------------------------------------------------- #


def test_v02_on_parse_failure_raise_strategy():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
            "on_parse_failure": {"strategy": "raise"},
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_on_parse_failure_default_strategy():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
            "on_parse_failure": {"strategy": "default", "default": {"files": []}},
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_on_parse_failure_rejects_unknown_strategy():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
            "on_parse_failure": {"strategy": "ignore"},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---- §3.5 filter operator -------------------------------------------------- #


def test_v02_filter_operator_field_equals():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {"id": "src", "kind": "call", "symbol": "x.y", "args": {}},
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "src"},
            "item_id": "item",
            "predicate": {
                "kind": "field_equals",
                "select": "verdict",
                "value": "issue",
            },
        },
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_filter_predicate_field_in():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {"id": "src", "kind": "call", "symbol": "x.y", "args": {}},
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "src"},
            "item_id": "i",
            "predicate": {
                "kind": "field_in",
                "select": "severity",
                "values": ["Critical", "High"],
            },
        },
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_filter_predicate_field_truthy_with_negate():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {"id": "src", "kind": "call", "symbol": "x.y", "args": {}},
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "src"},
            "item_id": "i",
            "predicate": {
                "kind": "field_truthy",
                "select": "issue",
                "negate": True,
            },
        },
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_filter_rejects_unknown_predicate_kind():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "src"},
            "item_id": "i",
            "predicate": {"kind": "field_greater_than", "select": "n", "value": 5},
        }
    ]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---- §3.6 capabilities ----------------------------------------------------- #


def test_v02_capabilities_block_accepted():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["capabilities"] = {
        "tools": {"allowed": ["Read", "Grep", "Glob", "Bash", "Task"], "denied": []}
    }
    Draft202012Validator(schema).validate(desc)


def test_v02_capabilities_rejects_extra_keys():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["capabilities"] = {"tools": {"allowed": []}, "something_else": True}
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


# ---- §3.7 model_id literal ------------------------------------------------- #


def test_v02_state_args_accept_literal_value():
    """§3.7: state args may now be a literal `{value: ...}`, not just env."""
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["state"] = [
        {
            "id": "backend",
            "symbol": "mellea.backends.litellm.LiteLLMBackend",
            "args": {"model_id": {"value": "anthropic/claude-opus-4-1"}},
        }
    ]
    Draft202012Validator(schema).validate(desc)


# ---- §3.11 notes ----------------------------------------------------------- #


def test_v02_top_level_notes():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["notes"] = ["author commentary 1", "commentary 2"]
    Draft202012Validator(schema).validate(desc)


def test_v02_node_level_notes_on_call():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "x.y",
            "args": {},
            "notes": ["this node is the entrypoint"],
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_node_level_notes_on_composition():
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["pipeline"] = [
        {
            "id": "br",
            "kind": "composition",
            "operator": "branch",
            "on": {"ref": "x"},
            "cases": {},
            "notes": ["one"],
        }
    ]
    Draft202012Validator(schema).validate(desc)


def test_v02_rejects_underscore_notes():
    """v0.2 expects `notes`, not `_notes`. The migration is mechanical."""
    schema = load_schema("0.2")
    desc = _minimal_v02_descriptor()
    desc["_notes"] = ["should fail"]
    errors = list(Draft202012Validator(schema).iter_errors(desc))
    assert errors


def test_v02_schema_uses_draft_2020_12_metaschema():
    schema = load_schema("0.2")
    assert schema["$schema"].endswith("/draft/2020-12/schema")
