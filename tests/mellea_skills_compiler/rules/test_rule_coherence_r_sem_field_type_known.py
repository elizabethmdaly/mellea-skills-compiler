"""Coherence audits for the ``r-sem-field-type-known`` semantic rule.

The rule enforces a closed vocabulary for `SchemaField.type` strings.
JSON Schema's `SchemaField.type` is just `{type: string}` — any string
passes structural validation — so this semantic rule is the SOLE
enforcer of the closed vocabulary.

The vocabulary is descriptor-IR-scoped (not Mellea-API-scoped), so
version-coupling risk is low. The vocabulary list lives in the rule
body (`_KNOWN_FIELD_TYPE_TOKENS`), paired with a renderer case in
`renderer/nodes.py::lower_type` for every accepted token.
"""
from __future__ import annotations

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import (
    R_FIELD_TYPE_KNOWN,
    _KNOWN_FIELD_TYPE_TOKENS,
)
from mellea_skills_compiler.descriptor.validator import validate
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "r-sem-field-type-known"


def _descriptor_with_schema(schemas_dict: dict) -> dict:
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
            "args": {},
        }],
    }


def _make_schema(field_type: str) -> dict:
    return {
        "Foo": {
            "kind": "model",
            "name": "Foo",
            "fields": {"x": {"type": field_type}},
        }
    }


def _has_error(errors, rule_id: str) -> bool:
    return any(e.rule == rule_id for e in errors)


@pytest.mark.parametrize(
    "primitive",
    ["str", "int", "float", "bool", "bytes"],
)
def test_primitive_types_pass(primitive):
    """C-PRIMITIVE-TYPES-PASS — parametrised across the primitive set
    (including `bytes` added 2026-05-20)."""
    descriptor = _descriptor_with_schema(_make_schema(primitive))
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert not _has_error(rep.errors, R_FIELD_TYPE_KNOWN), (
        f"primitive {primitive!r} must not fire R-SEM-FIELD-TYPE-KNOWN; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


@pytest.mark.parametrize(
    "param_type",
    ["list[str]", "dict[str, int]", "Optional[float]", "Literal['a', 'b']"],
)
def test_parameterised_types_pass(param_type):
    """C-PARAMETERISED-TYPES-PASS."""
    descriptor = _descriptor_with_schema(_make_schema(param_type))
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert not _has_error(rep.errors, R_FIELD_TYPE_KNOWN), (
        f"parameterised {param_type!r} must not fire R-SEM-FIELD-TYPE-KNOWN; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


@pytest.mark.parametrize(
    "unknown_type",
    ["Decimal", "datetime", "complex"],
)
def test_unknown_type_fails(unknown_type):
    """C-UNKNOWN-TYPE-FAILS."""
    descriptor = _descriptor_with_schema(_make_schema(unknown_type))
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _has_error(rep.errors, R_FIELD_TYPE_KNOWN), (
        f"unknown type {unknown_type!r} must fire R-SEM-FIELD-TYPE-KNOWN; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


def test_declared_schema_name_passes():
    """C-DECLARED-SCHEMA-NAME-PASSES — a field type that names another
    schema declared in the same descriptor is accepted.
    """
    descriptor = _descriptor_with_schema({
        "Foo": {
            "kind": "model",
            "name": "Foo",
            "fields": {"nested": {"type": "Bar"}},
        },
        "Bar": {
            "kind": "model",
            "name": "Bar",
            "fields": {"x": {"type": "str"}},
        },
    })
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert not _has_error(rep.errors, R_FIELD_TYPE_KNOWN), (
        f"reference to declared schema 'Bar' must not fire R-SEM-FIELD-"
        f"TYPE-KNOWN; got: {[(e.rule, e.message) for e in rep.errors]}"
    )


def test_renderer_lower_type_agrees_with_vocabulary():
    """C-RENDERER-LOWER-TYPE-AGREES — every token in the vocabulary
    must lower cleanly via the renderer.
    """
    import ast
    from mellea_skills_compiler.renderer.nodes import lower_type

    # The vocabulary's parameterised forms need an inner; skip the
    # `any` / sentinel tokens for renderer-lowering since they're
    # semantic-only.
    primitive_tokens = {"str", "int", "float", "bool", "bytes"}
    for tok in primitive_tokens:
        assert tok in _KNOWN_FIELD_TYPE_TOKENS, (
            f"vocabulary token {tok!r} dropped from "
            f"_KNOWN_FIELD_TYPE_TOKENS — coherence test guard"
        )
        # Lower it and assert it produces a valid AST expr.
        expr = lower_type(tok)
        assert isinstance(expr, ast.expr), (
            f"renderer's lower_type({tok!r}) returned {expr!r}, not an "
            f"ast.expr — registry vocabulary and renderer have drifted"
        )
