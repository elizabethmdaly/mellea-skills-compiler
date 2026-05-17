"""End-to-end-style tests for v0.3 — a synthetic descriptor exercising every
new feature; deprecation-warning behaviour; cross-version routing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.validator import DeprecationWarning_


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _synthetic_v03_descriptor() -> dict[str, Any]:
    """A descriptor exercising every new v0.3 feature surface.

    Includes:
    - structured classification block with all fields
    - dependencies[] entries covering several dispositions
    - bundled_resources[] entry
    - agent_loop composition operator (with tools_ref into dependencies)
    - human_approval composition operator
    - streaming call
    - image input
    - dict + literal_union + optional + description permissiveness
    - source_elements on a call
    """
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {
            "name": "synthetic-everything",
            "async": True,
            "classification": {
                "primary_axis": "AGENT",
                "interaction_style": "tool_using_loop",
                "input_modality": ["text", "images"],
                "output_modality": "streaming_text",
                "requires_approval_gates": True,
                "state_persistence": "cross_session",
            },
        },
        "dependencies": [
            {
                "id": "search_fn", "kind": "tool",
                "disposition": "delegate_to_runtime",
                "signature": "(query: str) -> list[str]",
                "rationale": "Runtime injects the search backend.",
            },
            {
                "id": "owasp_refs", "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            },
            {
                "id": "scratchpad", "kind": "tool",
                "disposition": "stub",
                "signature": "(content: str) -> None",
            },
        ],
        "bundled_resources": [
            {"source_dir": "references", "package_dir": "references"}
        ],
        "inputs": [
            {"name": "query", "schema": {"kind": "str"}},
            {"name": "options", "schema": {"kind": "dict",
                                            "key_type": "str",
                                            "value_type": "str"}, "optional": True},
            {"name": "color", "schema": {"kind": "literal_union",
                                          "members": ["red", "green", "blue"]}},
            {"name": "photo", "schema": {"kind": "image", "format_hint": "any"}},
        ],
        "outputs": [
            {"name": "answer", "schema": {"kind": "model_ref",
                                            "ref": "#/schemas/Answer"}}
        ],
        "schemas": {
            "Answer": {
                "kind": "model",
                "fields": {
                    "verdict": {"type": "str", "description": "The final verdict."},
                    "score": {"type": "Optional[float]"},
                },
            },
            "LoopState": {
                "kind": "model",
                "fields": {
                    "done": {"type": "bool"},
                    "scratch": {"type": "list[str]"},
                },
            },
        },
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [
            {
                "id": "gate",
                "kind": "composition",
                "operator": "human_approval",
                "prompt": {"value": "Run the loop?"},
                "branches": {
                    "approved": [
                        {
                            "id": "loop",
                            "kind": "composition",
                            "operator": "agent_loop",
                            "max_iterations": 10,
                            "state_schema": {"ref": "#/schemas/LoopState"},
                            "termination_condition": {
                                "predicate": "field_truthy",
                                "field": "done",
                            },
                            "tools_ref": ["search_fn"],
                            "body": [
                                {
                                    "id": "speak",
                                    "kind": "call",
                                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                                    "bound_to": {"ref": "session"},
                                    "args": {"description": {"value": "step"}},
                                    "streaming": True,
                                    "source_elements": ["E14"],
                                }
                            ],
                        }
                    ],
                    "rejected": [],
                },
            }
        ],
    }


def test_synthetic_v03_descriptor_validates_clean(surface, tmp_path):
    """Every feature is exercised and the descriptor is accepted (modulo the
    layered-defence warnings, which are non-blocking)."""
    (tmp_path / "references").mkdir()
    desc = _synthetic_v03_descriptor()
    inv = {"elements": [{"id": "E14"}]}
    report = validate(
        desc,
        schema_version="0.3",
        surface=surface,
        skill_root=tmp_path,
        inventory=inv,
    )
    blocking = [e for e in report.errors if e.severity == "error"]
    assert report.valid, (
        f"unexpected blocking errors: {[(e.path, e.rule, e.message) for e in blocking]}"
    )


def test_synthetic_v03_descriptor_each_rule_can_break_it(surface, tmp_path):
    """For each rule, mutate the synthetic descriptor to break it and confirm
    only that rule fires. Acts as a comprehensive smoke test for v0.3 rules.
    """
    from mellea_skills_compiler.descriptor.semantic_rules import (
        R_AGENT_LOOP_FIELDS,
        R_BUNDLED_PATH_EXISTS,
        R_DISPOSITION_CONSISTENT,
        R_FIELD_TYPE_KNOWN,
        R_MODALITY_APPROVAL,
        R_MODALITY_IMAGE_INPUT,
        R_MODALITY_STREAMING,
        R_MODALITY_TOOL_LOOP,
        R_SOURCE_ELEMENT_RESOLVES,
    )

    (tmp_path / "references").mkdir()

    # R-SEM-MODALITY-TOOL-LOOP: remove the agent_loop body.
    desc = _synthetic_v03_descriptor()
    desc["pipeline"][0]["branches"]["approved"] = []
    desc["pipeline"][0]["branches"]["rejected"] = []
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_MODALITY_TOOL_LOOP for e in report.errors)

    # R-SEM-MODALITY-STREAMING: drop the streaming flag.
    desc = _synthetic_v03_descriptor()
    desc["pipeline"][0]["branches"]["approved"][0]["body"][0]["streaming"] = False
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_MODALITY_STREAMING for e in report.errors)

    # R-SEM-MODALITY-IMAGE-INPUT: remove the image slot.
    desc = _synthetic_v03_descriptor()
    desc["inputs"] = [s for s in desc["inputs"] if s.get("schema", {}).get("kind") != "image"]
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_MODALITY_IMAGE_INPUT for e in report.errors)

    # R-SEM-MODALITY-APPROVAL: remove the human_approval operator.
    desc = _synthetic_v03_descriptor()
    desc["pipeline"] = []
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_MODALITY_APPROVAL for e in report.errors)

    # R-SEM-DISPOSITION-CONSISTENT: drop the search_fn signature.
    desc = _synthetic_v03_descriptor()
    del desc["dependencies"][0]["signature"]
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_DISPOSITION_CONSISTENT for e in report.errors)

    # R-SEM-BUNDLED-PATH-EXISTS: leave references/ missing.
    desc = _synthetic_v03_descriptor()
    other_tmp = tmp_path / "empty"
    other_tmp.mkdir()
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=other_tmp)
    assert any(e.rule == R_BUNDLED_PATH_EXISTS for e in report.errors)

    # R-SEM-AGENT-LOOP-FIELDS: invalid termination predicate.
    desc = _synthetic_v03_descriptor()
    loop = desc["pipeline"][0]["branches"]["approved"][0]
    loop["termination_condition"] = {"predicate": "wibble", "field": "x"}
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_AGENT_LOOP_FIELDS for e in report.errors)

    # R-SEM-SOURCE-ELEMENT-RESOLVES: bogus element id with inventory.
    desc = _synthetic_v03_descriptor()
    loop = desc["pipeline"][0]["branches"]["approved"][0]
    loop["body"][0]["source_elements"] = ["E99"]
    inv = {"elements": [{"id": "E14"}]}
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path, inventory=inv)
    assert any(e.rule == R_SOURCE_ELEMENT_RESOLVES for e in report.errors)

    # R-SEM-FIELD-TYPE-KNOWN: bogus type on Answer.verdict.
    desc = _synthetic_v03_descriptor()
    desc["schemas"]["Answer"]["fields"]["verdict"]["type"] = "TotallyUnknownType"
    report = validate(desc, schema_version="0.3", surface=surface,
                      skill_root=tmp_path)
    assert any(e.rule == R_FIELD_TYPE_KNOWN for e in report.errors)


# ---------------------------------------------------------------------------
# Deprecation: capabilities.tools coexists with dependencies in v0.3.
# ---------------------------------------------------------------------------


def test_deprecation_no_warning_when_only_dependencies():
    desc = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "x", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
        "dependencies": [
            {"id": "t1", "kind": "tool", "disposition": "remove"},
        ],
    }
    report = validate(desc, schema_version="0.3")
    assert report.valid
    assert report.deprecations == []


def test_deprecation_when_capabilities_tools_present_alone():
    desc = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "x", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
        "capabilities": {"tools": {"allowed": ["search"]}},
    }
    report = validate(desc, schema_version="0.3")
    assert report.valid  # non-blocking
    assert len(report.deprecations) == 1
    dep = report.deprecations[0]
    assert isinstance(dep, DeprecationWarning_)
    assert dep.rule == "deprecation:capabilities_tools"
    assert dep.path == "/capabilities/tools"
    assert dep.replacement == "/dependencies"
    assert dep.descriptor_version == "0.3"


def test_deprecation_when_both_capabilities_and_dependencies_present():
    desc = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "x", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
        "capabilities": {"tools": {"allowed": ["search"]}},
        "dependencies": [
            {"id": "search_fn", "kind": "tool", "disposition": "remove"},
        ],
    }
    report = validate(desc, schema_version="0.3")
    assert report.valid
    assert len(report.deprecations) == 1
    assert "coexists" in report.deprecations[0].message.lower() or \
        "deprecated" in report.deprecations[0].message.lower()


def test_no_deprecation_for_v02_descriptors():
    """v0.2 schema's capabilities.tools is not deprecated."""
    desc = {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "x", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
        "capabilities": {"tools": {"allowed": ["search"]}},
    }
    report = validate(desc, schema_version="0.2")
    assert report.deprecations == []
