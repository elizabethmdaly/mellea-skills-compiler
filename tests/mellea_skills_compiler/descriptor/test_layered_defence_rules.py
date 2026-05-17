"""Tests for the two v0.3 layered-defence rules (RFC §1.7 + §4 + Plan §11.6).

Both rules ship at **warning** severity in v0.3 per the RFC's promotion-gate
discipline; tests assert the warning fires (or does not) without expecting
overall ``valid == False``.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


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
        assert all(f.severity == "warning" for f in findings)
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
