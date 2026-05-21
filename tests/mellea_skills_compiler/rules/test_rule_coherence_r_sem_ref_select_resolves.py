"""Coherence audits for the ``r-sem-ref-select-resolves`` semantic rule.

Warning-severity rule. Best-effort traversal of `{ref, select}` dotted
paths against the ref's resolved target schema. Permissive about cases
it can't reason about (false-positive-averse).
"""
from __future__ import annotations

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import R_REF_SELECT_RESOLVES
from mellea_skills_compiler.descriptor.validator import validate


def _descriptor_with_select(args: dict, schemas: dict | None = None) -> dict:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [],
        "outputs": [],
        "schemas": schemas or {},
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": [
            {
                "kind": "call",
                "id": "extract",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "args": {"format": {"schema_ref": "#/schemas/Profile"}},
            },
            {
                "kind": "call",
                "id": "use",
                "symbol": "mellea.stdlib.session.MelleaSession.chat",
                "args": args,
            },
        ],
    }


def test_valid_select_passes():
    """C-VALID-SELECT-PASSES — select path matching a declared field
    on the target schema passes without warning.
    """
    schemas = {
        "Profile": {
            "kind": "model",
            "name": "Profile",
            "fields": {"name": {"type": "str"}, "email": {"type": "str"}},
        }
    }
    descriptor = _descriptor_with_select(
        args={"context": {"ref": "extract", "select": "name"}},
        schemas=schemas,
    )
    rep = validate(descriptor, schema_version="0.3", surface=None)
    ref_select_errors = [e for e in rep.errors if e.rule == R_REF_SELECT_RESOLVES]
    assert not ref_select_errors, (
        f"valid select path must not fire R-SEM-REF-SELECT-RESOLVES; "
        f"got: {[(e.rule, e.message) for e in ref_select_errors]}"
    )


def test_rule_severity_is_warning():
    """C-RULE-EMITS-WARNING-SEVERITY.

    Inspect the rule's emission semantics. We verify the rule's
    declared severity matches the registry's `validation.severity`
    field. Confirms the rule has not been accidentally promoted to
    error severity.
    """
    from mellea_skills_compiler.rules import get_rule

    rule = get_rule("r-sem-ref-select-resolves")
    assert rule["validation"]["severity"] == "warning", (
        f"r-sem-ref-select-resolves severity must be 'warning' "
        f"(best-effort traversal — error severity would create false "
        f"positives the design avoids); got "
        f"{rule['validation']['severity']!r}"
    )


def test_unresolvable_target_skips():
    """C-UNRESOLVABLE-TARGET-NO-OPS — when the ref's target type
    cannot be inferred, the rule does not fire spuriously.
    """
    # ref to an input that has no schema -> rule should not fire even
    # though `select` doesn't resolve to anything inspectable.
    descriptor = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [{"name": "opaque_input"}],
        "outputs": [],
        "schemas": {},
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": [{
            "kind": "call",
            "id": "c0",
            "symbol": "mellea.stdlib.session.MelleaSession.chat",
            "args": {
                "context": {
                    "ref": "opaque_input",
                    "select": "field_we_cant_check",
                }
            },
        }],
    }
    rep = validate(descriptor, schema_version="0.3", surface=None)
    ref_select_errors = [e for e in rep.errors if e.rule == R_REF_SELECT_RESOLVES]
    # The contract: when the target can't be resolved, the rule should
    # stay silent (best-effort discipline). It's acceptable for the
    # rule to fire IF and only if it can resolve the target.
    # This test pins the silent-on-unresolvable behavior.
    assert not ref_select_errors, (
        f"unresolvable target must not fire spurious warnings; got: "
        f"{[(e.rule, e.message) for e in ref_select_errors]}"
    )
