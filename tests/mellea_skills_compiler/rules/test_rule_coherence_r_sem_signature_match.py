"""Coherence audits for the ``r-sem-signature-match`` semantic rule.

Compares descriptor `inputs`/`outputs` against the locked
`expected_signature.json` artifact (P3.5.D). Cross-artefact rule that
fires only when `expected_signature` is supplied to `validate()`.
"""
from __future__ import annotations

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import R_SIGNATURE_MATCH
from mellea_skills_compiler.descriptor.validator import validate


def _descriptor_with_inputs(inputs: list) -> dict:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": inputs,
        "outputs": [],
        "schemas": {},
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": [{
            "kind": "call",
            "id": "c0",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {},
        }],
    }


def _has_error(errors, rule_id: str) -> bool:
    return any(e.rule == rule_id for e in errors)


def test_matching_inputs_pass():
    """C-LOCKED-INPUTS-DESCRIBED."""
    descriptor = _descriptor_with_inputs([
        {"name": "session_id", "schema": {"kind": "str"}},
    ])
    expected = {
        "inputs": [{"name": "session_id", "type": "str"}],
        "outputs": [],
        "schemas": [],
    }
    rep = validate(
        descriptor, schema_version="0.3", surface=None, expected_signature=expected
    )
    assert not _has_error(rep.errors, R_SIGNATURE_MATCH), (
        f"matching inputs must not fire R-SEM-SIGNATURE-MATCH; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_missing_required_input_fails():
    """C-MISSING-REQUIRED-INPUT-FAILS."""
    descriptor = _descriptor_with_inputs([])
    expected = {
        "inputs": [{"name": "session_id", "type": "str"}],
        "outputs": [],
        "schemas": [],
    }
    rep = validate(
        descriptor, schema_version="0.3", surface=None, expected_signature=expected
    )
    assert _has_error(rep.errors, R_SIGNATURE_MATCH), (
        f"missing required input must fire R-SEM-SIGNATURE-MATCH; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_optional_input_may_be_omitted():
    """C-OPTIONAL-INPUT-MAY-BE-OMITTED.

    Optional inputs missing from the descriptor degrade to WARNING
    severity (telemetry) rather than ERROR. The rule may still emit a
    warning entry — the contract is that no ERROR-severity entry fires
    for optional-marked missing inputs.
    """
    descriptor = _descriptor_with_inputs([])
    expected = {
        "inputs": [{"name": "user_template", "type": "str", "optional": True}],
        "outputs": [],
        "schemas": [],
    }
    rep = validate(
        descriptor, schema_version="0.3", surface=None, expected_signature=expected
    )
    error_severity = [
        e for e in rep.errors
        if e.rule == R_SIGNATURE_MATCH and e.severity == "error"
    ]
    assert not error_severity, (
        f"optional input may be omitted without firing an ERROR-severity "
        f"R-SEM-SIGNATURE-MATCH; got: "
        f"{[(e.rule, e.severity, e.message) for e in error_severity]}"
    )


def test_no_expected_signature_skips_rule():
    """C-NO-EXPECTED-SIG-NOOPS."""
    descriptor = _descriptor_with_inputs([])
    rep = validate(
        descriptor, schema_version="0.3", surface=None, expected_signature=None
    )
    assert not _has_error(rep.errors, R_SIGNATURE_MATCH), (
        f"R-SEM-SIGNATURE-MATCH must no-op when expected_signature=None"
    )
