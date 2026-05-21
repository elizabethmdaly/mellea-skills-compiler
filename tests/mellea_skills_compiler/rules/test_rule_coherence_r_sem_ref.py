"""Coherence audits for the ``r-sem-ref`` semantic rule.

The rule enforces that every value-binding ``ref`` in the descriptor IR
(`bound_to`, `over`, `on`, ArgValue's `{ref: ...}` variant) resolves to
a member of ``visible_ids`` — the union of ``state[].id``,
``inputs[].name``, and prior pipeline-node ``id``s.

Two faces enforce the contract at two layers:

* **Schema** (``descriptor.schema.v0.3.json:$defs.Ref.ref``) carries
  ``pattern: ^[a-z_][a-z0-9_]*$``. JSON Pointer strings (`#/state/...`)
  are rejected here with a clean structural message.
* **Semantic rule** (``semantic_rules.py``) verifies the identifier is
  actually in scope. This is the back-stop for typos / out-of-scope
  refs that the pattern can't catch.

The distinct ``SchemaRef`` shape (``{ref: \"#/schemas/Foo\"}``) is
intentionally NOT constrained by the same pattern — it carries its own
``^#/schemas/`` pattern. The two shapes are orthogonal.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from mellea_skills_compiler.descriptor.semantic_rules import (
    R_REF,
    _validate_call_node,
)
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "r-sem-ref"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_descriptor_schema() -> dict:
    schema_path = (
        _REPO_ROOT
        / "src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _has_error(errors, rule_id: str) -> bool:
    return any(err.rule == rule_id for err in errors)


# ─── Coherence checks ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref_target, visible_ids",
    [
        ("session", {"session"}),
        ("organization_description", {"organization_description"}),
        ("extract", {"extract"}),
    ],
    ids=["state-id", "input-name", "prior-node-id"],
)
def test_plain_identifier_in_visible_ids_resolves(ref_target, visible_ids):
    """C-PLAIN-ID-IN-SCOPE-RESOLVES — parametrised across the three
    sources of visible_ids."""
    node = {
        "kind": "call",
        "id": "test_call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": ref_target},
    }
    errors = _validate_call_node(
        node=node,
        node_path="/pipeline/0",
        visible_ids=visible_ids,
        schemas={},
        surface=None,
        schema_version="0.3",
    )
    assert not _has_error(errors, R_REF), (
        f"ref={ref_target!r} against visible_ids={visible_ids!r} should "
        f"resolve; got {errors!r}"
    )


def test_unknown_identifier_fails():
    """C-UNKNOWN-ID-FAILS."""
    node = {
        "kind": "call",
        "id": "test_call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "no_such_id"},
    }
    errors = _validate_call_node(
        node=node,
        node_path="/pipeline/0",
        visible_ids={"session", "input_x"},
        schemas={},
        surface=None,
        schema_version="0.3",
    )
    assert _has_error(errors, R_REF), (
        f"ref to identifier not in visible_ids must fire R-SEM-REF; "
        f"got {errors!r}"
    )


def test_json_pointer_ref_fails_at_schema():
    """C-JSON-POINTER-FAILS-AT-SCHEMA — the structural fix."""
    schema = _load_descriptor_schema()
    ref_schema = schema["$defs"]["Ref"]
    validator = jsonschema.Draft202012Validator(ref_schema)
    bad_ref = {"ref": "#/state/session"}
    errors = list(validator.iter_errors(bad_ref))
    assert errors, (
        "schema must structurally reject JSON Pointer syntax in Ref.ref "
        "via the `^[a-z_][a-z0-9_]*$` pattern"
    )


def test_json_pointer_ref_fails_at_semantic():
    """C-JSON-POINTER-FAILS-AT-SEMANTIC — back-stop layer."""
    node = {
        "kind": "call",
        "id": "test_call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "#/state/session"},
    }
    errors = _validate_call_node(
        node=node,
        node_path="/pipeline/0",
        visible_ids={"session"},
        schemas={},
        surface=None,
        schema_version="0.3",
    )
    assert _has_error(errors, R_REF), (
        f"JSON Pointer ref string must also fail the semantic rule "
        f"(visible_ids lookup); got {errors!r}"
    )


def test_schema_ref_pointer_still_accepted():
    """C-SCHEMA-REF-UNAFFECTED."""
    schema = _load_descriptor_schema()
    # ArgValue's `schema_ref` variant carries pattern `^#/schemas/`.
    # Wrap with the full $defs so internal $refs (e.g. ArgValue ->
    # Ref) can resolve during sub-schema validation.
    arg_value_with_defs = {
        "$ref": "#/$defs/ArgValue",
        "$defs": schema["$defs"],
    }
    validator = jsonschema.Draft202012Validator(arg_value_with_defs)
    good_schema_ref = {"schema_ref": "#/schemas/Answer"}
    errors = list(validator.iter_errors(good_schema_ref))
    assert not errors, (
        f"SchemaRef shape with JSON Pointer must still validate — the "
        f"Ref.ref pattern tightening must not break SchemaRef; "
        f"got errors: {errors!r}"
    )


def test_directive_doc_section_exists():
    """C-DIRECTIVE-DOC-EXISTS."""
    rule = get_rule(_RULE_ID)
    doc_pointer = rule["directive"]["doc"]
    section = rule["directive"]["section"]
    assert doc_pointer, "directive.doc must not be null"
    assert section, "directive.section must not be null"
    doc_path = _REPO_ROOT / doc_pointer
    assert doc_path.is_file(), f"directive doc {doc_pointer!r} does not exist"
    doc_text = doc_path.read_text(encoding="utf-8")
    assert section in doc_text, (
        f"directive section {section!r} not found in {doc_pointer!r}"
    )


# ─── Examples-as-fourth-face cornerstones (pilot 2026-05-20) ─────────


def _run_all_validators(descriptor: dict) -> list:
    """Run the full two-layer validator pipeline against ``descriptor``.

    Returns the aggregated list of ``ValidationError`` records (empty
    list iff every layer passed). Mirrors what the wrapper does at
    descriptor-validate time, minus the optional surface/expected-sig/
    skill-root/inventory inputs that don't matter for R-SEM-REF.
    """
    from mellea_skills_compiler.descriptor.validator import validate

    report = validate(
        descriptor,
        schema_version="0.3",
        surface=None,
    )
    return list(report.errors)


def test_positive_examples_pass_all_validators():
    """C-POSITIVE-EXAMPLES-PASS-ALL-VALIDATORS.

    The first cornerstone of the example face: every registered positive
    example MUST pass every validator layer. A failure here means either
    the example is stale OR the validator regressed; either way the
    maintainer must reconcile.
    """
    rule = get_rule(_RULE_ID)
    positives = rule.get("examples", {}).get("positive", [])
    assert positives, (
        f"{_RULE_ID} declares example-face coherence checks but has no "
        f"positive examples"
    )
    for example in positives:
        errors = _run_all_validators(example["descriptor"])
        assert not errors, (
            f"positive example {example['id']!r} expected to pass all "
            f"validators, but got errors: "
            f"{[(e.rule, e.path, e.message) for e in errors]}"
        )


def test_negative_examples_fail_validators():
    """C-NEGATIVE-EXAMPLES-FAIL-VALIDATORS.

    The second cornerstone: every registered negative example MUST fail
    at least one validator, AND the failure must be attributable to the
    layer the example's ``fails_at`` field claims (schema vs semantic-
    rule). Couples examples and validators by construction.
    """
    rule = get_rule(_RULE_ID)
    negatives = rule.get("examples", {}).get("negative", [])
    assert negatives, (
        f"{_RULE_ID} declares example-face coherence checks but has no "
        f"negative examples"
    )
    for example in negatives:
        errors = _run_all_validators(example["descriptor"])
        assert errors, (
            f"negative example {example['id']!r} expected to fail at "
            f"least one validator but passed all"
        )
        fails_at = example.get("fails_at")
        if fails_at == "schema":
            # jsonschema errors carry rule prefix "jsonschema:..."
            schema_errors = [e for e in errors if e.rule.startswith("jsonschema:")]
            assert schema_errors, (
                f"negative example {example['id']!r} claims fails_at="
                f"'schema' but no jsonschema-layer errors fired; got: "
                f"{[(e.rule, e.message) for e in errors]}"
            )
        elif fails_at == "semantic-rule":
            sem_errors = [e for e in errors if e.rule.startswith("R-SEM-")]
            assert sem_errors, (
                f"negative example {example['id']!r} claims fails_at="
                f"'semantic-rule' but no R-SEM-* errors fired; got: "
                f"{[(e.rule, e.message) for e in errors]}"
            )


def test_near_miss_examples_fail_with_r_sem_ref_attribution():
    """C-NEAR-MISS-FAILS-WITH-ATTRIBUTION.

    Near-miss examples are plausible-mistake shapes the LLM is likely to
    emit (typos, common hallucinations). They must fail with at least
    one error attributed to R-SEM-REF specifically — not random noise
    from unrelated rules.
    """
    rule = get_rule(_RULE_ID)
    near_misses = rule.get("examples", {}).get("near_miss", [])
    for example in near_misses:
        errors = _run_all_validators(example["descriptor"])
        assert errors, (
            f"near-miss example {example['id']!r} expected to fail but "
            f"passed all validators"
        )
        ref_errors = [e for e in errors if e.rule == R_REF]
        assert ref_errors, (
            f"near-miss example {example['id']!r} expected to fail with "
            f"R-SEM-REF attribution; got: "
            f"{[(e.rule, e.message) for e in errors]}"
        )


def test_directive_embeds_registry_example_snippets():
    """C-DIRECTIVE-EMBEDS-REGISTRY-SNIPPETS.

    The directive doc's ref-shape bullet must contain the literal
    ``bound_to`` snippet from each positive and negative example. Drift
    detector: if a maintainer updates a snippet in either place without
    the other, this test catches the asymmetry.
    """
    rule = get_rule(_RULE_ID)
    doc_path = _REPO_ROOT / rule["directive"]["doc"]
    doc_text = doc_path.read_text(encoding="utf-8")
    for example in rule.get("examples", {}).get("positive", []) + rule.get(
        "examples", {}
    ).get("negative", []):
        # The doc embeds the `bound_to` ref fragment — small enough to
        # paste verbatim, distinctive enough to find unambiguously.
        bound_to = example["descriptor"]["pipeline"][0]["bound_to"]
        ref_value = bound_to["ref"]
        # Match the JSON-style snippet `"ref": "<value>"` (the doc uses
        # JSON syntax inline). Whitespace tolerant.
        snippet = f'"ref": "{ref_value}"'
        assert snippet in doc_text, (
            f"directive doc {rule['directive']['doc']!r} does not "
            f"contain example {example['id']!r}'s snippet {snippet!r}. "
            f"Either update the doc to embed the registry's example "
            f"verbatim, or update the registry example to match the doc."
        )
