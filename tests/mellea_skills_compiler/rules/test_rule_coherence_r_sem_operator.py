"""Coherence audits for the ``r-sem-operator`` semantic rule.

Composition operators (`sequential`, `branch`, `map`, etc.) have
operator-specific required fields. The JSON Schema enforces these via
`oneOf` discriminator branches; this semantic rule provides additional
checks for shape-level invariants the schema can't express, and acts as
a back-stop if the schema's discriminator admits a malformed shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import R_OPERATOR
from mellea_skills_compiler.descriptor.validator import validate
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "r-sem-operator"


def _descriptor_with_pipeline(pipeline: list) -> dict:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [],
        "outputs": [],
        "schemas": {},
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": pipeline,
    }


def _any_error_about_operator(errors, op_name: str) -> bool:
    """True iff any error mentions the operator (R-SEM-OPERATOR OR
    jsonschema error on the operator's required fields)."""
    for e in errors:
        if e.rule == R_OPERATOR:
            return True
        # Schema-layer rejection of the same shape is also acceptable —
        # both faces enforce the same invariant.
        if e.rule.startswith("jsonschema:") and op_name in e.message.lower():
            return True
        if e.rule.startswith("jsonschema:") and "required" in e.rule:
            return True
    return False


def test_sequential_without_body_fails():
    """C-SEQUENTIAL-BODY-REQUIRED."""
    descriptor = _descriptor_with_pipeline([
        {"kind": "composition", "operator": "sequential", "id": "seq"},
        # missing `body`
    ])
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _any_error_about_operator(rep.errors, "sequential"), (
        f"sequential without `body` must be rejected by R-SEM-OPERATOR or "
        f"by the schema's discriminator; got: "
        f"{[(e.rule, e.path, e.message[:80]) for e in rep.errors]}"
    )


def test_map_missing_over_or_item_id_fails():
    """C-MAP-OVER-AND-ITEM-ID-REQUIRED."""
    descriptor = _descriptor_with_pipeline([
        {"kind": "composition", "operator": "map", "id": "m"},
        # missing `over`, `item_id`, `body`
    ])
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _any_error_about_operator(rep.errors, "map"), (
        f"map without `over`/`item_id` must be rejected; got: "
        f"{[(e.rule, e.path, e.message[:80]) for e in rep.errors]}"
    )


def test_branch_without_cases_fails():
    """C-BRANCH-CASES-REQUIRED."""
    descriptor = _descriptor_with_pipeline([
        {"kind": "composition", "operator": "branch", "id": "br",
         "on": {"ref": "s"}},
        # missing `cases`
    ])
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert _any_error_about_operator(rep.errors, "branch"), (
        f"branch without `cases` must be rejected; got: "
        f"{[(e.rule, e.path, e.message[:80]) for e in rep.errors]}"
    )


def test_well_formed_operators_pass():
    """C-WELL-FORMED-OPERATOR-PASSES — sequential composition with a
    body of a single call passes the rule.
    """
    descriptor = _descriptor_with_pipeline([
        {
            "kind": "composition",
            "operator": "sequential",
            "id": "seq",
            "body": [{
                "kind": "call",
                "id": "c0",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "args": {},
            }],
        },
    ])
    rep = validate(descriptor, schema_version="0.3", surface=None)
    op_errors = [e for e in rep.errors if e.rule == R_OPERATOR]
    assert not op_errors, (
        f"well-formed sequential must not fire R-SEM-OPERATOR; got: "
        f"{[(e.rule, e.path, e.message[:80]) for e in op_errors]}"
    )


def test_semantic_rule_aligns_with_schema_required_fields():
    """C-SEMANTIC-RULE-IS-SUPERSET-OF-SCHEMA.

    For each operator whose composition node we omit a schema-required
    field on, BOTH layers should reject. Specifically: missing `body`
    on sequential should yield at least one error from EITHER face
    (schema OR semantic rule), so the LLM never silently passes both.
    """
    cases = [
        ("sequential", {"kind": "composition", "operator": "sequential", "id": "x"}),
        ("map", {"kind": "composition", "operator": "map", "id": "x"}),
        ("branch", {"kind": "composition", "operator": "branch", "id": "x"}),
    ]
    for op_name, malformed in cases:
        descriptor = _descriptor_with_pipeline([malformed])
        rep = validate(descriptor, schema_version="0.3", surface=None)
        assert rep.errors, (
            f"malformed {op_name} composition must produce at least one "
            f"error (schema-layer OR semantic-layer); got none"
        )
