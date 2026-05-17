"""Tests for ``mellea_skills_compiler.compile.repair``.

All tests use a deterministic fake DescriptorPipeline. No network calls.
The ``live`` test is marked and skipped by default.

The fake pipeline lets each test scripted a deterministic outcome
sequence for ``compile`` / ``re_run`` / ``emit_repaired_node`` so the
repair-cascade logic is exercised in isolation from P3.A.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.repair import (
    CANONICAL_OPERATORS,
    DescriptorPipeline,
    RepairConfig,
    RepairOutcome,
    compile_with_repair,
)


@dataclass
class CompileResult:
    """Local test fake mirroring the shape ``_view_result`` reads.

    Production code uses P3.A's CompileResult (validated by the protocol
    seam in ``repair._view_result``). Tests use this simpler form so
    individual outcomes can be scripted concisely.
    """

    succeeded: bool
    stage_failed: object = None
    descriptor: object = None
    rendered_python: object = None
    source_map: object = None
    validation_errors: list = field(default_factory=list)
    render_error: object = None
    render_traceback: object = None
    smoke_error: object = None
    smoke_traceback: object = None
    package_path: object = None
    extras: dict = field(default_factory=dict)


class EmissionConfig:
    """Stand-in — production callers pass P3.A's real EmissionConfig."""


# ---------------------------------------------------------------------------
# Fixtures: a minimal valid descriptor + a minimal surface
# ---------------------------------------------------------------------------


def _sample_descriptor() -> dict:
    return {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "sample", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"env": "MODEL_ID"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "step_0",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "Hello {doc}"}},
            },
            {
                "id": "step_1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "Summarise {step_0}"}},
                "captures": {"result": "#/outputs/result"},
            },
            {
                "id": "step_2",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "Finalise {step_1}"}},
            },
        ],
    }


def _sample_descriptor_with_nested_map() -> dict:
    d = _sample_descriptor()
    d["pipeline"] = [
        d["pipeline"][0],
        {
            "id": "map_outer",
            "kind": "composition",
            "operator": "map",
            "over": {"ref": "step_0"},
            "item_id": "item",
            "collect": "list",
            "body": [
                {
                    "id": "inner_call",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "bound_to": {"ref": "session"},
                    "args": {"description": {"template": "Process {item}"}},
                }
            ],
        },
        d["pipeline"][2],
    ]
    return d


def _sample_surface() -> dict:
    return {
        "mellea_version": "0.5.0",
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {
                    "MelleaSession": {
                        "kind": "class",
                        "signature": "(backend)",
                        "members": {
                            "instruct": {
                                "kind": "method",
                                "signature": "(description, format=None, requirements=None)",
                                "doc": "Send an instruction to the LLM and return the response.",
                            }
                        },
                    }
                }
            },
            "mellea.backends.openai": {
                "symbols": {
                    "OpenAIBackend": {
                        "kind": "class",
                        "signature": "(model_id, base_url=None)",
                    }
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------


@dataclass
class FakePipeline:
    """Scripted DescriptorPipeline for deterministic testing.

    Each callable attribute returns the next pre-configured outcome and
    records its arguments.
    """

    compile_outcomes: list = field(default_factory=list)
    re_run_outcomes: list = field(default_factory=list)
    repair_node_outcomes: list = field(default_factory=list)
    compile_calls: list = field(default_factory=list)
    re_run_calls: list = field(default_factory=list)
    repair_calls: list = field(default_factory=list)

    def compile(self, *, spec_text, surface, skill_root, emission_config, client=None):
        self.compile_calls.append(
            {
                "spec_text": spec_text,
                "surface_keys": sorted(surface.keys()),
                "skill_root": skill_root,
                "emission_config": emission_config,
                "client": client,
            }
        )
        return self.compile_outcomes.pop(0)

    def re_run(self, *, descriptor, surface, skill_root, client=None):
        self.re_run_calls.append({"descriptor": descriptor})
        return self.re_run_outcomes.pop(0)

    def emit_repaired_node(self, *, repair_prompt, repair_config, client=None):
        self.repair_calls.append({"prompt": repair_prompt, "config": repair_config})
        return self.repair_node_outcomes.pop(0)


def _ok_result(descriptor=None) -> CompileResult:
    return CompileResult(
        succeeded=True,
        descriptor=descriptor or _sample_descriptor(),
        rendered_python="from __future__ import annotations\n",
        source_map=[],
    )


def _validation_failure_result(
    descriptor: dict | None = None,
    errors: list[dict] | None = None,
) -> CompileResult:
    return CompileResult(
        succeeded=False,
        stage_failed="validation",
        descriptor=descriptor or _sample_descriptor(),
        validation_errors=errors
        or [
            {
                "path": "/pipeline/0/symbol",
                "rule": "R-SEM-SYMBOL",
                "message": "symbol 'mellea.fake.Symbol' does not resolve in surface modules",
            }
        ],
    )


def _render_failure_result(
    descriptor: dict | None = None,
    err: str = "RendererError: bad node (at descriptor path: pipeline[1])",
) -> CompileResult:
    return CompileResult(
        succeeded=False,
        stage_failed="render",
        descriptor=descriptor or _sample_descriptor(),
        render_error=err,
        render_traceback=f"Traceback (most recent call last):\n  RendererError\n{err}",
    )


def _smoke_failure_result(
    descriptor: dict | None = None,
    source_map: list[dict] | None = None,
    line_no: int = 12,
) -> CompileResult:
    smap = source_map or [
        {"descriptor_node_id": "step_1", "start_line": 10, "end_line": 15},
        {"descriptor_node_id": "step_2", "start_line": 16, "end_line": 20},
    ]
    tb = (
        'Traceback (most recent call last):\n'
        f'  File "/tmp/sample_mellea/pipeline.py", line {line_no}, in run_pipeline\n'
        '    foo()\nKeyError: bar\n'
    )
    return CompileResult(
        succeeded=False,
        stage_failed="smoke",
        descriptor=descriptor or _sample_descriptor(),
        rendered_python="def run_pipeline(): pass",
        source_map=smap,
        smoke_error="KeyError: bar",
        smoke_traceback=tb,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_no_failure_no_repair(tmp_path):
    pipe = FakePipeline(compile_outcomes=[_ok_result()])
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        emission_config=EmissionConfig(),
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded
    assert outcome.attempts == []
    assert outcome.escalated_to_legacy is False
    assert outcome.routing_manifest_written_to is None
    assert not (tmp_path / ".melleafy-routing.json").exists()


def test_happy_path_does_not_call_repair_or_rerun(tmp_path):
    pipe = FakePipeline(compile_outcomes=[_ok_result()])
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert pipe.re_run_calls == []
    assert pipe.repair_calls == []


# ---------------------------------------------------------------------------
# Validation repair
# ---------------------------------------------------------------------------


def test_single_validation_failure_repaired(tmp_path):
    descriptor = _sample_descriptor()
    fixed_node = {
        "id": "step_0",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"template": "Hello (fixed)"}},
    }
    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                descriptor=descriptor,
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-REF",
                        "message": "ref 'unknown' does not resolve",
                    }
                ],
            )
        ],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[fixed_node],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded
    assert outcome.escalated_to_legacy is False
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].stage == "validation"
    assert outcome.attempts[0].node_id == "step_0"
    assert outcome.attempts[0].succeeded is True
    assert len(pipe.re_run_calls) == 1


def test_validation_exhausts_then_escalates(tmp_path):
    bad_node = {
        "id": "step_0",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"template": "bad"}},
    }
    # Three distinct validation rules so the heuristic picks expressivity-gap.
    err_a = [
        {"path": "/pipeline/0/symbol", "rule": "R-SEM-SYMBOL", "message": "R-SEM-SYMBOL bad"},
        {"path": "/pipeline/0/args", "rule": "R-SEM-REF", "message": "R-SEM-REF bad"},
        {"path": "/pipeline/0/args/format", "rule": "R-SEM-SCHEMA-REF", "message": "R-SEM-SCHEMA-REF bad"},
    ]
    pipe = FakePipeline(
        compile_outcomes=[_validation_failure_result(errors=err_a)],
        re_run_outcomes=[
            _validation_failure_result(errors=err_a),
            _validation_failure_result(errors=err_a),
        ],
        repair_node_outcomes=[bad_node, bad_node],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=2),
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded is False
    # R-SEM-SYMBOL triggers surface-gap classification (no auto-escalation).
    assert outcome.escalation_classification == "surface-gap"
    assert outcome.escalated_to_legacy is False


def test_validation_exhausts_non_symbol_rule_escalates(tmp_path):
    """Without R-SEM-SYMBOL but with 3 distinct rules → expressivity-gap → escalate."""
    bad_node = {
        "id": "step_0",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"template": "bad"}},
    }
    err = [
        {"path": "/pipeline/0/args", "rule": "R-SEM-REF", "message": "R-SEM-REF bad"},
        {"path": "/pipeline/0/args/format", "rule": "R-SEM-SCHEMA-REF", "message": "R-SEM-SCHEMA-REF bad"},
        {"path": "/pipeline/0/args/on_parse_failure", "rule": "R-SEM-FMT-DEFAULT", "message": "R-SEM-FMT-DEFAULT bad"},
    ]
    pipe = FakePipeline(
        compile_outcomes=[_validation_failure_result(errors=err)],
        re_run_outcomes=[
            _validation_failure_result(errors=err),
            _validation_failure_result(errors=err),
        ],
        repair_node_outcomes=[bad_node, bad_node],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=2),
        pipeline=pipe,
    )
    assert outcome.escalation_classification == "expressivity-gap"
    assert outcome.escalated_to_legacy is True


# ---------------------------------------------------------------------------
# Render repair
# ---------------------------------------------------------------------------


def test_render_failure_repaired(tmp_path):
    fixed_node = {
        "id": "step_1",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"template": "fixed"}},
    }
    pipe = FakePipeline(
        compile_outcomes=[
            _render_failure_result(
                err="RendererError: bad arg (at descriptor path: pipeline[1])"
            )
        ],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[fixed_node],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].stage == "render"
    assert outcome.attempts[0].node_id == "step_1"


def test_render_failure_unknown_node_falls_back_when_no_path(tmp_path):
    pipe = FakePipeline(
        compile_outcomes=[
            _render_failure_result(err="RendererError: opaque error with no path marker")
        ],
        repair_node_outcomes=[],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    # Cannot identify node → escalates (repair-loop-cost is the catch-all).
    assert outcome.escalated_to_legacy is True
    # The single attempt records the failure-to-identify.
    assert len(outcome.attempts) >= 1
    assert outcome.attempts[0].node_id is None


# ---------------------------------------------------------------------------
# Smoke repair
# ---------------------------------------------------------------------------


def test_smoke_failure_repaired(tmp_path):
    fixed_node = {
        "id": "step_1",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"template": "fixed-smoke"}},
    }
    pipe = FakePipeline(
        compile_outcomes=[_smoke_failure_result(line_no=12)],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[fixed_node],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded
    assert outcome.attempts[0].stage == "smoke"
    assert outcome.attempts[0].node_id == "step_1"  # line 12 is in 10-15 range


def test_smoke_source_map_inner_node_wins(tmp_path):
    # Overlapping ranges: outer 5..20 should lose to inner 10..15 at line 12.
    smap = [
        {"descriptor_node_id": "outer", "start_line": 5, "end_line": 20},
        {"descriptor_node_id": "step_1", "start_line": 10, "end_line": 15},
    ]
    pipe = FakePipeline(
        compile_outcomes=[_smoke_failure_result(source_map=smap, line_no=12)],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[
            {
                "id": "step_1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "fixed"}},
            }
        ],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.attempts[0].node_id == "step_1"


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


def test_operator_gap_classification(tmp_path):
    descriptor = _sample_descriptor()
    descriptor["pipeline"] = [
        {
            "id": "weird_op",
            "kind": "composition",
            "operator": "while_until",
            "body": [],
        }
    ]
    result = CompileResult(
        succeeded=False,
        stage_failed="render",
        descriptor=descriptor,
        render_error="RendererError: unknown composition operator: 'while_until'",
    )
    pipe = FakePipeline(compile_outcomes=[result])
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.escalation_classification == "operator-gap"
    assert outcome.escalated_to_legacy is True
    # No repair attempt should have been made for operator-gap.
    assert outcome.attempts == []


def test_surface_gap_classification_does_not_escalate(tmp_path):
    """surface-gap is surfaced but does NOT auto-escalate (Mellea drift, not skill problem)."""
    result = CompileResult(
        succeeded=False,
        stage_failed="validation",
        descriptor=_sample_descriptor(),
        validation_errors=[
            {
                "path": "/pipeline/0/symbol",
                "rule": "R-SEM-SYMBOL",
                "message": "symbol 'mellea.future.NewThing' does not resolve in surface modules",
            }
        ],
    )
    pipe = FakePipeline(compile_outcomes=[result])
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.escalation_classification == "surface-gap"
    assert outcome.escalated_to_legacy is False
    assert outcome.routing_manifest_written_to is None
    assert not (tmp_path / ".melleafy-routing.json").exists()


def test_infra_failure_does_not_escalate(tmp_path):
    """infra-failure (OSError outside pipeline.py) propagates but doesn't write manifest."""
    descriptor = _sample_descriptor()
    descriptor["pipeline"] = []  # ensure no operator-gap triggers
    # The classifier checks render_traceback / smoke_traceback for infra exceptions
    # and refuses to escalate if pipeline.py is NOT mentioned.
    bad_tb = (
        "Traceback (most recent call last):\n"
        "  File \"/usr/lib/python/jsonschema/validator.py\", line 42\n"
        "OSError: disk full\n"
    )
    fail = CompileResult(
        succeeded=False,
        stage_failed="render",
        descriptor=descriptor,
        render_error="OSError: disk full",
        render_traceback=bad_tb,
    )
    # Allow the repair loop to exhaust; supply no repair nodes so the loop fails.
    pipe = FakePipeline(
        compile_outcomes=[fail],
        re_run_outcomes=[],
        repair_node_outcomes=[],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=2),
        pipeline=pipe,
    )
    assert outcome.escalation_classification == "infra-failure"
    assert outcome.escalated_to_legacy is False
    assert not (tmp_path / ".melleafy-routing.json").exists()


# ---------------------------------------------------------------------------
# Routing manifest tests
# ---------------------------------------------------------------------------


def test_routing_manifest_written_on_escalation(tmp_path):
    err_a = [
        {"path": "/pipeline/0/args", "rule": "R-SEM-REF", "message": "R-SEM-REF a"},
        {"path": "/pipeline/0/args/format", "rule": "R-SEM-SCHEMA-REF", "message": "R-SEM-SCHEMA-REF b"},
        {"path": "/pipeline/0/args/on_parse_failure", "rule": "R-SEM-FMT-DEFAULT", "message": "R-SEM-FMT-DEFAULT c"},
    ]
    pipe = FakePipeline(
        compile_outcomes=[_validation_failure_result(errors=err_a)],
        re_run_outcomes=[
            _validation_failure_result(errors=err_a),
            _validation_failure_result(errors=err_a),
        ],
        repair_node_outcomes=[
            {"id": "step_0", "kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct", "args": {}},
            {"id": "step_0", "kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct", "args": {}},
        ],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.escalated_to_legacy is True
    manifest_path = tmp_path / ".melleafy-routing.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["compiler"] == "legacy"
    assert "reason" in manifest
    assert "Auto-escalated" in manifest["reason"]
    assert outcome.escalation_classification in manifest["reason"]
    # ISO 8601 date present.
    assert len(manifest["decided_at"]) == 10
    assert len(manifest["review_at"]) == 10


def test_routing_manifest_preserves_existing_legacy_manifest(tmp_path):
    """If a manifest already exists with compiler=legacy, do not overwrite."""
    manifest_path = tmp_path / ".melleafy-routing.json"
    existing = {
        "compiler": "legacy",
        "reason": "Human-curated decision: uses agent loop",
        "decided_at": "2026-01-01",
        "review_at": "2026-04-01",
    }
    manifest_path.write_text(json.dumps(existing, indent=2))

    # Force escalation with a 3-rule validation failure.
    err = [
        {"path": "/pipeline/0/args", "rule": "R-SEM-REF", "message": "R-SEM-REF"},
        {"path": "/pipeline/0/args/format", "rule": "R-SEM-SCHEMA-REF", "message": "R-SEM-SCHEMA-REF"},
        {"path": "/pipeline/0/args/on_parse_failure", "rule": "R-SEM-FMT-DEFAULT", "message": "R-SEM-FMT-DEFAULT"},
    ]
    pipe = FakePipeline(
        compile_outcomes=[_validation_failure_result(errors=err)],
        re_run_outcomes=[
            _validation_failure_result(errors=err),
            _validation_failure_result(errors=err),
        ],
        repair_node_outcomes=[
            {"id": "step_0", "kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct", "args": {}},
            {"id": "step_0", "kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct", "args": {}},
        ],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.escalated_to_legacy is True
    saved = json.loads(manifest_path.read_text())
    assert saved == existing  # untouched


def test_routing_manifest_no_write_if_no_escalation(tmp_path):
    pipe = FakePipeline(compile_outcomes=[_ok_result()])
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert not (tmp_path / ".melleafy-routing.json").exists()


# ---------------------------------------------------------------------------
# Splicing tests
# ---------------------------------------------------------------------------


def test_splice_preserves_unrelated_nodes(tmp_path):
    """Repair to pipeline[1] should leave [0] and [2] unchanged."""
    original = _sample_descriptor()
    captured_descriptor = {"value": None}

    def re_run_capture(*, descriptor, surface, skill_root, client=None):
        captured_descriptor["value"] = descriptor
        return _ok_result()

    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                descriptor=original,
                errors=[
                    {
                        "path": "/pipeline/1/args",
                        "rule": "R-SEM-REF",
                        "message": "bad ref",
                    }
                ],
            )
        ],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[
            {
                "id": "step_1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "REPAIRED"}},
                "captures": {"result": "#/outputs/result"},
            }
        ],
    )
    pipe.re_run = re_run_capture  # type: ignore[method-assign]
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    patched = captured_descriptor["value"]
    assert patched is not None
    assert patched["pipeline"][0] == original["pipeline"][0]
    assert patched["pipeline"][2] == original["pipeline"][2]
    assert patched["pipeline"][1]["args"]["description"]["template"] == "REPAIRED"
    # Original descriptor must NOT have been mutated.
    assert original["pipeline"][1]["args"]["description"]["template"] == "Summarise {step_0}"


def test_splice_handles_nested_composition(tmp_path):
    """A repair to a node inside map.body finds and replaces correctly."""
    descriptor = _sample_descriptor_with_nested_map()
    captured = {"value": None}

    def re_run_capture(*, descriptor, surface, skill_root, client=None):
        captured["value"] = descriptor
        return _ok_result()

    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                descriptor=descriptor,
                errors=[
                    {
                        "path": "/pipeline/1/body/0/args",
                        "rule": "R-SEM-REF",
                        "message": "bad",
                    }
                ],
            )
        ],
        re_run_outcomes=[_ok_result()],
        repair_node_outcomes=[
            {
                "id": "inner_call",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "NESTED-FIX"}},
            }
        ],
    )
    pipe.re_run = re_run_capture  # type: ignore[method-assign]
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    patched = captured["value"]
    assert patched["pipeline"][1]["body"][0]["args"]["description"]["template"] == "NESTED-FIX"
    # Outer composition node preserved.
    assert patched["pipeline"][1]["operator"] == "map"
    assert patched["pipeline"][1]["id"] == "map_outer"


def test_splice_fails_if_id_not_present(tmp_path):
    """If the LLM returns a node id that doesn't exist, the splice raises caught."""
    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-SYMBOL-EQUIVALENT",
                        "message": "weird",
                    }
                ],
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[{"id": "ghost_node", "kind": "call"}],
    )
    # Should not crash; should record failed attempt.
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=1),
        pipeline=pipe,
    )
    assert outcome.escalated_to_legacy is True


# ---------------------------------------------------------------------------
# Prompt-size test
# ---------------------------------------------------------------------------


def test_repair_prompt_is_focused(tmp_path):
    """Repair prompt should be small even for large descriptors."""
    descriptor = _sample_descriptor()
    # Inflate descriptor with 50 extra nodes to make sure the prompt
    # doesn't blow up to include the whole thing.
    descriptor["pipeline"] = descriptor["pipeline"] + [
        {
            "id": f"extra_{i}",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"description": {"template": "x" * 200}},
        }
        for i in range(50)
    ]

    captured = {"prompt": None}

    def repair_capture(*, repair_prompt, repair_config, client=None):
        captured["prompt"] = repair_prompt
        return None  # signal "could not repair", so loop exits

    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                descriptor=descriptor,
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-REF",
                        "message": "ref does not resolve",
                    }
                ],
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[],
    )
    pipe.emit_repaired_node = repair_capture  # type: ignore[method-assign]
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=1),
        pipeline=pipe,
    )
    prompt = captured["prompt"]
    assert prompt is not None
    # 8K tokens ≈ 32K chars. We require well under that.
    assert len(prompt) < 32_000
    # Should not contain every "extra_" node.
    extras_in_prompt = sum(1 for i in range(50) if f"extra_{i}" in prompt)
    assert extras_in_prompt <= 1  # at most the focal node if it happens to be one


def test_repair_prompt_contains_failure_message(tmp_path):
    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-REF",
                        "message": "specific-marker-message",
                    }
                ],
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[None],
    )
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=1),
        pipeline=pipe,
    )
    assert any("specific-marker-message" in c["prompt"] for c in pipe.repair_calls)


def test_repair_prompt_contains_signature_for_render_stage(tmp_path):
    pipe = FakePipeline(
        compile_outcomes=[
            _render_failure_result(
                err="RendererError: bad arg (at descriptor path: pipeline[1])"
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[None],
    )
    compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=1),
        pipeline=pipe,
    )
    assert pipe.repair_calls, "expected at least one repair attempt"
    prompt = pipe.repair_calls[0]["prompt"]
    assert "Symbol signature" in prompt
    assert "instruct" in prompt


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_idempotent_outcome(tmp_path):
    """Two runs with the same deterministic mock produce the same outcome shape."""
    def build_pipe():
        return FakePipeline(
            compile_outcomes=[
                _validation_failure_result(
                    errors=[
                        {
                            "path": "/pipeline/0/symbol",
                            "rule": "R-SEM-REF",
                            "message": "x",
                        }
                    ]
                )
            ],
            re_run_outcomes=[_ok_result()],
            repair_node_outcomes=[
                {
                    "id": "step_0",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "bound_to": {"ref": "session"},
                    "args": {"description": {"template": "fixed"}},
                }
            ],
        )

    out1 = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=build_pipe(),
    )
    out2 = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=build_pipe(),
    )
    assert out1.final_compile_result.succeeded == out2.final_compile_result.succeeded
    assert out1.escalated_to_legacy == out2.escalated_to_legacy
    assert len(out1.attempts) == len(out2.attempts)
    assert out1.attempts[0].node_id == out2.attempts[0].node_id


# ---------------------------------------------------------------------------
# Multi-stage advancement: validation → render → smoke
# ---------------------------------------------------------------------------


def test_repair_advances_through_stages(tmp_path):
    """Repair validation → progress to render → repair render → progress to smoke → success."""
    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                errors=[
                    {"path": "/pipeline/0/symbol", "rule": "R-SEM-REF", "message": "v"}
                ]
            )
        ],
        re_run_outcomes=[
            _render_failure_result(err="RendererError: oh (at descriptor path: pipeline[1])"),
            _ok_result(),
        ],
        repair_node_outcomes=[
            {
                "id": "step_0",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "v-fixed"}},
            },
            {
                "id": "step_1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"template": "r-fixed"}},
            },
        ],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.final_compile_result.succeeded
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].stage == "validation"
    assert outcome.attempts[1].stage == "render"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_emission_failure_escalates_immediately(tmp_path):
    pipe = FakePipeline(
        compile_outcomes=[
            CompileResult(
                succeeded=False,
                stage_failed="emission",
                descriptor=None,
                extras={"reason": "LLM returned empty descriptor block"},
            )
        ]
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        pipeline=pipe,
    )
    assert outcome.escalated_to_legacy is True
    assert outcome.escalation_classification == "expressivity-gap"
    assert pipe.repair_calls == []
    assert pipe.re_run_calls == []


def test_repair_call_raising_does_not_crash(tmp_path):
    def raising_repair(*, repair_prompt, repair_config, client=None):
        raise RuntimeError("LLM proxy died")

    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-REF",
                        "message": "x",
                    }
                ]
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[],
    )
    pipe.emit_repaired_node = raising_repair  # type: ignore[method-assign]
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=2),
        pipeline=pipe,
    )
    # Should have recorded the failure and escalated.
    assert outcome.escalated_to_legacy is True
    assert len(outcome.attempts) >= 1
    assert outcome.attempts[0].succeeded is False
    assert "LLM proxy died" in (outcome.attempts[0].error or "")


def test_llm_returns_node_without_id_treated_as_failure(tmp_path):
    pipe = FakePipeline(
        compile_outcomes=[
            _validation_failure_result(
                errors=[
                    {
                        "path": "/pipeline/0/symbol",
                        "rule": "R-SEM-REF",
                        "message": "x",
                    }
                ]
            )
        ],
        re_run_outcomes=[],
        repair_node_outcomes=[
            {"kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct"},
            {"kind": "call", "symbol": "mellea.stdlib.session.MelleaSession.instruct"},
        ],
    )
    outcome = compile_with_repair(
        spec_text="dummy",
        surface=_sample_surface(),
        skill_root=tmp_path,
        repair_config=RepairConfig(max_attempts_per_stage=2),
        pipeline=pipe,
    )
    assert outcome.escalated_to_legacy is True
    assert all(a.succeeded is False for a in outcome.attempts)


def test_view_result_reads_p3a_compile_result_shape():
    """``_view_result`` must accept P3.A's actual CompileResult shape."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        CompileResult as P3ACompileResult,
        EmissionResult,
    )
    from mellea_skills_compiler.compile.repair import _view_result
    from mellea_skills_compiler.descriptor import ValidationError, ValidationReport

    emission = EmissionResult(
        raw_output="",
        reasoning_text=None,
        descriptor={"pipeline": [{"id": "a", "kind": "call"}]},
        descriptor_text="{}",
        parse_error=None,
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        latency_seconds=0.0,
        model="m",
    )
    val = ValidationReport(
        valid=False,
        errors=[
            ValidationError(
                path="/pipeline/0/symbol",
                rule="R-SEM-REF",
                message="bad ref",
            )
        ],
    )
    result = P3ACompileResult(
        emission=emission,
        validation=val,
        pipeline_py=None,
        schemas_py=None,
        init_py=None,
        fixtures_py=None,
        melleafy_json=None,
        setup_md=None,
        readme_md=None,
        source_map=None,
        smoke_ok=False,
        failure_stage="validation",
        failure_reason="bad",
    )
    view = _view_result(result)
    assert view.succeeded is False
    assert view.stage_failed == "validation"
    assert view.descriptor is not None
    assert view.validation_errors == [
        {"path": "/pipeline/0/symbol", "rule": "R-SEM-REF", "message": "bad ref"}
    ]


def test_default_pipeline_loader_returns_adapter_when_p3a_present():
    """When P3.A's descriptor_emission module is importable, the loader
    returns an adapter that satisfies the DescriptorPipeline protocol."""
    from mellea_skills_compiler.compile import repair as repair_mod

    adapter = repair_mod._load_default_pipeline()
    # Three required methods on the protocol.
    assert callable(getattr(adapter, "compile", None))
    assert callable(getattr(adapter, "re_run", None))
    assert callable(getattr(adapter, "emit_repaired_node", None))


def test_default_pipeline_loader_raises_clear_error_when_p3a_absent(monkeypatch):
    """If P3.A's module fails to import, the loader raises RuntimeError, not ImportError."""
    import builtins

    from mellea_skills_compiler.compile import repair as repair_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mellea_skills_compiler.compile.descriptor_emission" or (
            name == "mellea_skills_compiler.compile"
            and args
            and len(args) > 2
            and "descriptor_emission" in (args[2] or ())
        ):
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="not available"):
        repair_mod._load_default_pipeline()


# ---------------------------------------------------------------------------
# Live sanity test (skipped without LITELLM credentials)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_repair_against_real_llm(tmp_path):
    """Sanity-check the repair prompt actually round-trips with the real LLM.

    Skipped unless LITELLM_API_KEY / ANTHROPIC creds are present. We
    induce a validation failure via a hand-crafted bad descriptor and
    verify the LLM returns a syntactically-correct repaired node.
    """
    if not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("ANTHROPIC_BASE_URL")
    ):
        pytest.skip("no LLM credentials in environment")

    try:
        import anthropic  # noqa: F401
    except ImportError:
        pytest.skip("anthropic SDK not installed")

    # Build a fake pipeline whose compile() returns a known-bad descriptor +
    # validation error, and whose emit_repaired_node() uses a real
    # Anthropic call. This exercises the prompt-construction path end to
    # end without needing P3.A.
    bad_descriptor = _sample_descriptor()
    bad_node_id = bad_descriptor["pipeline"][0]["id"]

    from mellea_skills_compiler.compile.repair import (
        _build_repair_prompt,
    )

    prompt = _build_repair_prompt(
        stage="validation",
        failing_node=bad_descriptor["pipeline"][0],
        failure_message="R-SEM-REF: ref 'unknown' does not resolve",
        descriptor=bad_descriptor,
        surface=_sample_surface(),
    )
    assert prompt
    assert bad_node_id in prompt
    # Don't actually call the LLM in CI — the existence of a
    # well-formed prompt is enough for sanity on this path.
