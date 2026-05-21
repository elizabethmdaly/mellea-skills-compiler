"""Coherence audits for the ``r-sem-schema-ref`` semantic rule.

The rule enforces that JSON Pointer schema references (`#/schemas/<Name>`)
resolve to a declared key in the descriptor's top-level `schemas` dict.
Two-layer enforcement: the JSON Schema's `^#/schemas/` pattern catches
non-pointer shapes; this semantic rule catches well-formed pointers
that don't resolve.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from mellea_skills_compiler.descriptor.semantic_rules import R_SCHEMA_REF
from mellea_skills_compiler.descriptor.validator import validate
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "r-sem-schema-ref"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _descriptor_with_schemas(schemas_dict: dict, schema_ref_value: str) -> dict:
    """Build a v0.3 descriptor with declared `schemas` and a call node
    whose `args` carries a schema_ref ArgValue.
    """
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [],
        "outputs": [],
        "schemas": schemas_dict,
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": [{
            "kind": "call",
            "id": "c0",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {"format": {"schema_ref": schema_ref_value}},
        }],
    }


def _has_error(errors, rule_id: str) -> bool:
    return any(e.rule == rule_id for e in errors)


def test_declared_schema_resolves():
    """C-DECLARED-SCHEMA-RESOLVES."""
    descriptor = _descriptor_with_schemas(
        schemas_dict={
            "Foo": {
                "kind": "model",
                "name": "Foo",
                "fields": {"x": {"type": "str"}},
            }
        },
        schema_ref_value="#/schemas/Foo",
    )
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert not _has_error(rep.errors, R_SCHEMA_REF), (
        f"schema_ref to declared schema must not fire R-SEM-SCHEMA-REF; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


def test_undeclared_schema_fails():
    """C-UNDECLARED-SCHEMA-FAILS."""
    descriptor = _descriptor_with_schemas(
        schemas_dict={},
        schema_ref_value="#/schemas/Missing",
    )
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _has_error(rep.errors, R_SCHEMA_REF), (
        f"schema_ref to undeclared schema must fire R-SEM-SCHEMA-REF; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


def test_schema_pattern_rejects_non_schemas_pointers():
    """C-SCHEMA-PATTERN-ENFORCES-SYNTAX — structural layer rejects
    malformed pointers before the semantic rule sees them.
    """
    schema_path = (
        _REPO_ROOT
        / "src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    arg_value_with_defs = {
        "$ref": "#/$defs/ArgValue",
        "$defs": schema["$defs"],
    }
    validator = jsonschema.Draft202012Validator(arg_value_with_defs)
    for bad in [
        {"schema_ref": "#/state/X"},
        {"schema_ref": "Foo"},
        {"schema_ref": ""},
    ]:
        errors = list(validator.iter_errors(bad))
        assert errors, (
            f"schema must structurally reject schema_ref {bad!r} — "
            f"pattern is `^#/schemas/`"
        )


def test_nested_schema_ref_is_checked():
    """C-NESTED-SCHEMA-REF-CHECKED — schema_ref nested in args tree is
    still validated.
    """
    descriptor = _descriptor_with_schemas(
        schemas_dict={},  # no declared schemas
        schema_ref_value="#/schemas/Buried",
    )
    # Bury the schema_ref one level deeper via a symbol-arg-value chain.
    descriptor["pipeline"][0]["args"] = {
        "outer": {
            "symbol": "mellea.stdlib.session.MelleaSession.chat",
            "args": {"format": {"schema_ref": "#/schemas/Buried"}},
        }
    }
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _has_error(rep.errors, R_SCHEMA_REF), (
        f"nested schema_ref must still be checked; got: "
        f"{[(e.rule, e.path) for e in rep.errors]}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Directive doc not yet written for schema_ref shape (registry "
        "xfail_until: 2026-06-30). Likely subsumed by tomorrow's "
        "per-archetype canonical-descriptors work, which demonstrates "
        "schema_ref usage via worked examples."
    ),
)
def test_directive_doc_section_exists():
    """C-DIRECTIVE-DOC-EXISTS."""
    rule = get_rule(_RULE_ID)
    doc_pointer = rule["directive"]["doc"]
    section = rule["directive"]["section"]
    assert doc_pointer, "directive.doc must not be null"
    assert section, "directive.section must not be null"
