"""Tests for the four v0.3 R-SEM-MODALITY-* rules (RFC §4 + Plan §11.3)."""
from __future__ import annotations

from typing import Any

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.semantic_rules import (
    R_MODALITY_APPROVAL,
    R_MODALITY_IMAGE_INPUT,
    R_MODALITY_STREAMING,
    R_MODALITY_TOOL_LOOP,
)


def _v03(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }
    base.update(overrides)
    return base


def _rule_fired(report, rule: str) -> bool:
    return any(e.rule == rule for e in report.errors)


# ---------------------------------------------------------------------------
# R-SEM-MODALITY-TOOL-LOOP
# ---------------------------------------------------------------------------


class TestModalityToolLoop:
    def _desc(self, *, pipeline, interaction_style=None):
        d = _v03()
        d["pipeline"] = pipeline
        if interaction_style is not None:
            d["skill"]["classification"]["interaction_style"] = interaction_style
        return d

    def test_pre_not_met_does_not_fire(self):
        """No tool_using_loop classification → rule does not fire."""
        d = self._desc(pipeline=[])
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_TOOL_LOOP)

    def test_tool_using_loop_with_aact_passes(self):
        d = self._desc(
            interaction_style="tool_using_loop",
            pipeline=[
                {
                    "id": "loop_call",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.aact",
                    "args": {},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_TOOL_LOOP)

    def test_tool_using_loop_with_agent_loop_passes(self):
        d = self._desc(
            interaction_style="tool_using_loop",
            pipeline=[
                {
                    "id": "loop",
                    "kind": "composition",
                    "operator": "agent_loop",
                    "max_iterations": 5,
                    "state_schema": {"ref": "#/schemas/S"},
                    "termination_condition": {"predicate": "field_truthy", "field": "x"},
                    "body": [
                        {
                            "id": "b1",
                            "kind": "call",
                            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                            "args": {},
                        }
                    ],
                }
            ],
        )
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_TOOL_LOOP)

    def test_tool_using_loop_with_neither_fires(self):
        d = self._desc(
            interaction_style="tool_using_loop",
            pipeline=[
                {
                    "id": "instruct_call",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {"description": {"value": "x"}},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_TOOL_LOOP)

    def test_non_tool_using_loop_no_aact_does_not_fire(self):
        """interaction_style other than tool_using_loop never fires this rule."""
        d = self._desc(
            interaction_style="single_turn",
            pipeline=[
                {
                    "id": "c",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_TOOL_LOOP)

    def test_aact_inside_branch_case_passes(self):
        d = self._desc(
            interaction_style="tool_using_loop",
            pipeline=[
                {
                    "id": "br",
                    "kind": "composition",
                    "operator": "branch",
                    "on": {"ref": "doc"},
                    "cases": {
                        "yes": [
                            {
                                "id": "aact_call",
                                "kind": "call",
                                "symbol": "mellea.stdlib.session.MelleaSession.aact",
                                "args": {},
                            }
                        ],
                        "no": [],
                    },
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_TOOL_LOOP)


# ---------------------------------------------------------------------------
# R-SEM-MODALITY-STREAMING
# ---------------------------------------------------------------------------


class TestModalityStreaming:
    def _desc(self, *, pipeline, output_modality=None):
        d = _v03()
        d["pipeline"] = pipeline
        if output_modality is not None:
            d["skill"]["classification"]["output_modality"] = output_modality
        return d

    def test_pre_not_met_does_not_fire(self):
        d = self._desc(pipeline=[])
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_STREAMING)

    def test_streaming_text_with_terminal_streaming_passes(self):
        d = self._desc(
            output_modality="streaming_text",
            pipeline=[
                {
                    "id": "c",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {},
                    "streaming": True,
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_STREAMING)

    def test_streaming_text_without_streaming_fires(self):
        d = self._desc(
            output_modality="streaming_text",
            pipeline=[
                {
                    "id": "c",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_STREAMING)

    def test_streaming_text_empty_pipeline_fires(self):
        d = self._desc(output_modality="streaming_text", pipeline=[])
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_STREAMING)

    def test_non_streaming_text_does_not_fire(self):
        d = self._desc(
            output_modality="structured",
            pipeline=[
                {
                    "id": "c",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_STREAMING)

    def test_streaming_text_parallel_all_terminals_pass(self):
        d = self._desc(
            output_modality="streaming_text",
            pipeline=[
                {
                    "id": "par",
                    "kind": "composition",
                    "operator": "parallel",
                    "branches": [
                        [
                            {
                                "id": "a",
                                "kind": "call",
                                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                                "args": {},
                                "streaming": True,
                            }
                        ],
                        [
                            {
                                "id": "b",
                                "kind": "call",
                                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                                "args": {},
                                "streaming": True,
                            }
                        ],
                    ],
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_STREAMING)


# ---------------------------------------------------------------------------
# R-SEM-MODALITY-IMAGE-INPUT
# ---------------------------------------------------------------------------


class TestModalityImageInput:
    def _desc(self, *, inputs, input_modality=None):
        d = _v03()
        d["inputs"] = inputs
        if input_modality is not None:
            d["skill"]["classification"]["input_modality"] = input_modality
        return d

    def test_pre_not_met_does_not_fire(self):
        d = self._desc(inputs=[{"name": "doc", "schema": {"kind": "str"}}])
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_IMAGE_INPUT)

    def test_images_with_image_slot_passes(self):
        d = self._desc(
            input_modality=["images"],
            inputs=[{"name": "photo", "schema": {"kind": "image"}}],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_IMAGE_INPUT)

    def test_images_without_image_slot_fires(self):
        d = self._desc(
            input_modality=["images"],
            inputs=[{"name": "doc", "schema": {"kind": "str"}}],
        )
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_IMAGE_INPUT)

    def test_documents_with_document_slot_passes(self):
        d = self._desc(
            input_modality=["documents"],
            inputs=[{"name": "policy", "schema": {"kind": "document"}}],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_IMAGE_INPUT)

    def test_documents_without_document_slot_fires(self):
        d = self._desc(
            input_modality=["documents"],
            inputs=[{"name": "doc", "schema": {"kind": "str"}}],
        )
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_IMAGE_INPUT)

    def test_list_of_images_satisfies(self):
        """list[image] also satisfies the rule."""
        d = self._desc(
            input_modality=["images"],
            inputs=[
                {"name": "photos", "schema": {"kind": "list",
                                                "of": {"kind": "image"}}}
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_IMAGE_INPUT)


# ---------------------------------------------------------------------------
# R-SEM-MODALITY-APPROVAL
# ---------------------------------------------------------------------------


class TestModalityApproval:
    def _desc(self, *, pipeline, requires_approval_gates=None):
        d = _v03()
        d["pipeline"] = pipeline
        if requires_approval_gates is not None:
            d["skill"]["classification"]["requires_approval_gates"] = requires_approval_gates
        return d

    def test_pre_not_met_does_not_fire(self):
        d = self._desc(pipeline=[])
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_APPROVAL)

    def test_approval_required_with_operator_passes(self):
        d = self._desc(
            requires_approval_gates=True,
            pipeline=[
                {
                    "id": "gate",
                    "kind": "composition",
                    "operator": "human_approval",
                    "prompt": {"value": "?"},
                    "branches": {"approved": [], "rejected": []},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_APPROVAL)

    def test_approval_required_without_operator_fires(self):
        d = self._desc(
            requires_approval_gates=True,
            pipeline=[
                {
                    "id": "c",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "args": {},
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert _rule_fired(report, R_MODALITY_APPROVAL)

    def test_default_false_no_operator_passes(self):
        d = self._desc(pipeline=[])
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_APPROVAL)

    def test_approval_inside_branch_case_passes(self):
        d = self._desc(
            requires_approval_gates=True,
            pipeline=[
                {
                    "id": "br",
                    "kind": "composition",
                    "operator": "branch",
                    "on": {"ref": "doc"},
                    "cases": {
                        "yes": [
                            {
                                "id": "gate",
                                "kind": "composition",
                                "operator": "human_approval",
                                "prompt": {"value": "?"},
                                "branches": {"approved": [], "rejected": []},
                            }
                        ],
                        "no": [],
                    },
                }
            ],
        )
        report = validate(d, schema_version="0.3")
        assert not _rule_fired(report, R_MODALITY_APPROVAL)
