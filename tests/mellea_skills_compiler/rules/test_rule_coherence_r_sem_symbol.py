"""Coherence audits for the ``r-sem-symbol`` semantic rule.

The rule enforces that every `symbol` field in `state[]` and pipeline
`call` nodes resolves to a known entry in the introspected Mellea
surface. Surface-driven by construction: when Mellea updates, the
surface refreshes and the rule's verdicts follow automatically.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import (
    R_SYMBOL,
    _symbol_in_surface,
)
from mellea_skills_compiler.descriptor.validator import validate
from mellea_skills_compiler.rules import get_rule


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _minimal_descriptor(*, state_symbol: str, call_symbol: str) -> dict:
    """Build a minimal-valid v0.3 descriptor with the given symbols."""
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [],
        "outputs": [],
        "schemas": {},
        "state": [{"id": "s", "symbol": state_symbol}],
        "pipeline": [{
            "kind": "call",
            "id": "c0",
            "symbol": call_symbol,
            "args": {},
        }],
    }


def _synthetic_surface() -> dict:
    """Small synthetic surface — declares the symbols used in tests.

    Surface shape per `_symbol_in_surface` docstring: top-level symbols
    are keys under `modules.<path>.symbols`; class members live under
    `<ClassSymbol>.members.<member_name>`.
    """
    return {
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {
                    "start_session": {"kind": "function"},
                    "MelleaSession": {
                        "kind": "class",
                        "members": {
                            "instruct": {"kind": "method"},
                            "chat": {"kind": "method"},
                        },
                    },
                }
            }
        }
    }


def _has_error(errors, rule_id: str) -> bool:
    return any(e.rule == rule_id for e in errors)


def test_valid_surface_symbol_passes():
    """C-VALID-SURFACE-SYMBOL-PASSES."""
    descriptor = _minimal_descriptor(
        state_symbol="mellea.stdlib.session.start_session",
        call_symbol="mellea.stdlib.session.MelleaSession.instruct",
    )
    rep = validate(descriptor, schema_version="0.3", surface=_synthetic_surface())
    assert not _has_error(rep.errors, R_SYMBOL), (
        f"valid surface symbols must not fire R-SEM-SYMBOL; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_unknown_symbol_fails():
    """C-UNKNOWN-SYMBOL-FAILS."""
    descriptor = _minimal_descriptor(
        state_symbol="mellea.stdlib.session.start_session",
        call_symbol="mellea.stdlib.session.MelleaSession.nonexistent_method",
    )
    rep = validate(descriptor, schema_version="0.3", surface=_synthetic_surface())
    assert _has_error(rep.errors, R_SYMBOL), (
        f"call symbol not in surface must fire R-SEM-SYMBOL; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_surface_none_skips_rule():
    """C-SURFACE-NONE-NO-OPS — when surface=None, the rule no-ops."""
    descriptor = _minimal_descriptor(
        state_symbol="any.bogus.path",
        call_symbol="another.bogus.thing",
    )
    rep = validate(descriptor, schema_version="0.3", surface=None)
    assert not _has_error(rep.errors, R_SYMBOL), (
        f"R-SEM-SYMBOL must no-op when surface=None; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_state_symbol_also_validated():
    """C-STATE-SYMBOL-CHECKED — rule fires on state[].symbol too."""
    descriptor = _minimal_descriptor(
        state_symbol="not.in.surface",
        call_symbol="mellea.stdlib.session.MelleaSession.instruct",
    )
    rep = validate(descriptor, schema_version="0.3", surface=_synthetic_surface())
    state_symbol_errs = [
        e for e in rep.errors
        if e.rule == R_SYMBOL and e.path.startswith("/state/")
    ]
    assert state_symbol_errs, (
        f"state[].symbol that doesn't resolve must fire R-SEM-SYMBOL on a "
        f"/state/... path; got: {[(e.rule, e.path) for e in rep.errors]}"
    )


def test_rule_body_has_no_hardcoded_mellea_names():
    """C-NO-MELLEA-NAME-IN-RULE-BODY.

    The version-coupling smell test: inspect the source of
    `_symbol_in_surface` and the call-validation site, and assert they
    don't contain hardcoded Mellea API names (the modality-rule failure
    pattern the audit-coverage plan deliberately avoids).
    """
    import inspect
    from mellea_skills_compiler.descriptor import semantic_rules

    body = inspect.getsource(semantic_rules._symbol_in_surface)
    # Tokens that would indicate hardcoding (specific Mellea class /
    # method names appearing as string literals in the function body).
    forbidden = [
        '"MelleaSession"',
        "'MelleaSession'",
        '"start_session"',
        "'start_session'",
        '"instruct"',
        "'instruct'",
    ]
    for tok in forbidden:
        assert tok not in body, (
            f"_symbol_in_surface contains hardcoded Mellea name {tok!r} — "
            f"the rule should consume the introspected surface as data, "
            f"not bake in specific symbol names"
        )
