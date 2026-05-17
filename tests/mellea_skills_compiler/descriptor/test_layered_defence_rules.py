"""Tests for the two v0.3 layered-defence rules (RFC §1.7 + §4 + Plan §11.6).

``R-SEM-CALL-ARGS-MATCH-SIG`` has been promoted from WARNING to ERROR
(2026-05-17, post-gdpr-privacy-notice evidence) so the descriptor-side
rule catches arity mismatches the post-render ``stdlib-arity`` lint
catches — saving a 50+ minute render round-trip per skill.

``R-SEM-REF-SELECT-RESOLVES`` remains at WARNING severity per the RFC §4
telemetry window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.semantic_rules import (
    R_CALL_ARGS_MATCH_SIG,
    R_REF_SELECT_RESOLVES,
)


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _v03() -> dict[str, Any]:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [],
    }


def _findings(report, rule: str):
    return [e for e in report.errors if e.rule == rule]


class TestCallArgsMatchSig:

    def test_pre_not_met_when_no_surface(self):
        """No surface → rule cannot fire."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {"hallucinated_arg": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=None)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_exact_match_args_pass(self, surface):
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_hallucinated_arg_warning(self, surface):
        """Hallucinated kwarg now fires at ERROR severity (post-2026-05-17 promotion)."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"},
                      "wibble_factor": {"value": 42}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings
        assert all(f.severity == "error" for f in findings)
        assert any("wibble_factor" in f.message for f in findings)

    def test_self_stripped_for_bound_methods(self, surface):
        """Descriptor call's args do not need a 'self' key."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        # No 'self' warning even though signature contains it.
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_unknown_symbol_does_not_fire(self, surface):
        """If the symbol isn't in surface, R-SEM-SYMBOL handles it — not this rule."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "non.existent.Whatever.method",
             "args": {"some_arg": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_kwargs_signature_passes_through(self, surface):
        """A symbol whose signature has **kwargs accepts any arg."""
        # Construct a fake-surface-like override.
        fake_surface = {
            "modules": {
                "myhelper": {
                    "symbols": {
                        "go": {
                            "kind": "function",
                            "signature": "(target: str, **kwargs) -> None",
                        }
                    }
                }
            }
        }
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "myhelper.go",
             "args": {"target": {"value": "x"},
                      "any_other_arg": {"value": 1}}}
        ]
        report = validate(d, schema_version="0.3", surface=fake_surface)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    # --- Tightened arity coverage (2026-05-17) -----------------------------
    # These tests close the gap between R-SEM-CALL-ARGS-MATCH-SIG and the
    # post-render ``stdlib-arity`` lint that caught gdpr-privacy-notice's
    # ``check("..."))`` (1 positional, signature requires 2).

    def test_rule_catches_extra_positional(self, surface):
        """A call with more positional-style args than the static signature accepts fires ERROR."""
        d = _v03()
        # ``check`` has static signature min_pos=2, max_pos=2.
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.requirements.check",
             "args": {"requirement": {"value": "x"},
                      "output": {"value": "y"},
                      "extra": {"value": "z"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings, "expected an arity error for too-many args on check()"
        assert any(f.severity == "error" for f in findings)
        assert any("at most 2" in f.message or "3 arg" in f.message for f in findings)

    def test_rule_catches_missing_required(self, surface):
        """A call missing a required parameter fires ERROR with a repair-able message."""
        d = _v03()
        # ``check`` requires both ``requirement`` and ``output``.
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.requirements.check",
             "args": {"requirement": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings, "expected a missing-required error"
        assert all(f.severity == "error" for f in findings)
        assert any("output" in f.message and "missing" in f.message.lower()
                   for f in findings), (
            "expected a message naming the missing 'output' parameter"
        )

    def test_rule_catches_unknown_keyword(self, surface):
        """Unknown keyword fires ERROR (severity-promoted from WARNING)."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"},
                      "definitely_not_a_real_kwarg": {"value": 1}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings
        assert all(f.severity == "error" for f in findings)
        assert any("definitely_not_a_real_kwarg" in f.message for f in findings)

    def test_rule_passes_when_args_match_exactly(self, surface):
        """Valid call → no findings of this rule."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_rule_passes_with_kwargs_signature(self):
        """When the signature includes ``**kwargs`` we don't flag unknown keys.

        But we still flag missing required (positional-or-keyword without
        default).
        """
        fake_surface = {
            "modules": {
                "myhelper": {
                    "symbols": {
                        "go": {
                            "kind": "function",
                            "signature": "(target: str, **kwargs) -> None",
                        }
                    }
                }
            }
        }
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "myhelper.go",
             "args": {"target": {"value": "x"},
                      "anything_at_all": {"value": 1},
                      "you_like": {"value": 2}}}
        ]
        report = validate(d, schema_version="0.3", surface=fake_surface)
        assert not _findings(report, R_CALL_ARGS_MATCH_SIG)

    def test_rule_against_failing_skill_shape(self, surface):
        """Reproduces gdpr-privacy-notice's failure shape: ``check("...")`` with 1 arg.

        Pre-tightening this passed validation and only failed at the
        post-render ``stdlib-arity`` lint after a full Mellea compile.
        """
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.requirements.check",
             "args": {"requirement": {
                 "value": "Notice must not reference this skill, "
                          "internal tool names, or source reference files"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings, (
            "tightened R-SEM-CALL-ARGS-MATCH-SIG must catch the gdpr-privacy "
            "shape (check with 1 arg) — same shape the post-render "
            "stdlib-arity lint catches"
        )
        assert all(f.severity == "error" for f in findings)
        # Repair-able: must name the symbol AND either the count or the
        # missing parameter name.
        assert any("check" in f.message for f in findings)
        assert any(("output" in f.message) or ("2" in f.message)
                   for f in findings)

    def test_rule_promotes_to_error_severity(self, surface):
        """Severity is ERROR, not WARNING (post-2026-05-17 promotion)."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"},
                      "bogus_kwarg": {"value": 1}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings
        # ERROR severity blocks the compile — must not be downgraded.
        assert all(f.severity == "error" for f in findings)
        # And it actually invalidates the report (because severity is error).
        assert not report.valid

    def test_rule_validates_against_real_surface_json(self, surface):
        """Load the real surface JSON, build a descriptor that violates, confirm fire."""
        # Use a real symbol with a known required positional-or-keyword.
        # MelleaSession.instruct has ``description`` as the only required
        # non-self positional. A call with no description must fire.
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             # No 'description' — required.
             "args": {"model_options": {"value": {}}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings
        assert all(f.severity == "error" for f in findings)
        assert any("description" in f.message for f in findings)

    def test_kwargs_signature_does_not_mask_missing_required(self):
        """Signature with **kwargs still requires the positional-or-keyword target.

        Edge case: a ``go(target: str, **kwargs)`` signature accepts ANY
        keyword, but ``target`` is still required.
        """
        fake_surface = {
            "modules": {
                "myhelper": {
                    "symbols": {
                        "go": {
                            "kind": "function",
                            "signature": "(target: str, **kwargs) -> None",
                        }
                    }
                }
            }
        }
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "myhelper.go",
             # No ``target`` — required.
             "args": {"some_other": {"value": 1}}}
        ]
        report = validate(d, schema_version="0.3", surface=fake_surface)
        findings = _findings(report, R_CALL_ARGS_MATCH_SIG)
        assert findings
        assert any("target" in f.message and "missing" in f.message.lower()
                   for f in findings)


class TestRefSelectResolves:

    def test_pre_not_met_no_select_no_fire(self, surface):
        d = _v03()
        d["pipeline"] = [
            {"id": "c", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {"description": {"value": "x"}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_REF_SELECT_RESOLVES)

    def test_valid_dot_path_passes(self, surface):
        d = _v03()
        d["schemas"]["Pred"] = {
            "kind": "model",
            "fields": {"verdict": {"type": "str"}},
        }
        d["pipeline"] = [
            {"id": "first",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"value": "x"},
                 "format": {"schema_ref": "#/schemas/Pred"}}},
            {"id": "second",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"template": "use {x}",
                                  "args": {"x": {"ref": "first", "select": "verdict"}}}}},
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_REF_SELECT_RESOLVES)

    def test_unknown_field_warns(self, surface):
        d = _v03()
        d["schemas"]["Pred"] = {
            "kind": "model",
            "fields": {"verdict": {"type": "str"}},
        }
        d["pipeline"] = [
            {"id": "first",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"value": "x"},
                 "format": {"schema_ref": "#/schemas/Pred"}}},
            {"id": "second",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"template": "use {x}",
                                  "args": {"x": {"ref": "first", "select": "nonexistent_field"}}}}},
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_REF_SELECT_RESOLVES)
        assert findings
        assert all(f.severity == "warning" for f in findings)

    def test_indexing_syntax_warns(self, surface):
        d = _v03()
        d["schemas"]["Pred"] = {
            "kind": "model",
            "fields": {"verdict": {"type": "str"}},
        }
        d["pipeline"] = [
            {"id": "first",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"value": "x"},
                 "format": {"schema_ref": "#/schemas/Pred"}}},
            {"id": "second",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {
                 "description": {"template": "use {x}",
                                  "args": {"x": {"ref": "first", "select": "items.0.foo"}}}}},
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        findings = _findings(report, R_REF_SELECT_RESOLVES)
        assert findings
        assert any("indexing" in f.message for f in findings)

    def test_state_id_ref_does_not_fire(self, surface):
        """A ref to a state-id (not a pipeline node with format) is not this rule's concern."""
        d = _v03()
        d["pipeline"] = [
            {"id": "c",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {
                 "description": {"template": "see {x}",
                                  "args": {"x": {"ref": "session", "select": "backend"}}}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_REF_SELECT_RESOLVES)

    def test_no_schemas_no_fire(self, surface):
        d = _v03()
        d["pipeline"] = [
            {"id": "c",
             "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {
                 "description": {"template": "x",
                                  "args": {"y": {"ref": "doc", "select": "first"}}}}}
        ]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_REF_SELECT_RESOLVES)
