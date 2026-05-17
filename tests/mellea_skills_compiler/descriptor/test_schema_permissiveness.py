"""Schema-permissiveness battery — written from the descriptor schema CONTRACT,
not the validator's implementation.

Strategy applied (per `melleafy-handoff/process/testing-strategy.md` §Strategies):
- §S2 — Category enumeration first. The failure-mode categories below are
  derived from "what could a descriptor author legitimately want to express"
  rather than from "what does the current schema reject."
- §S3 — Adversarial role-play: each test imagines a descriptor author who
  is faithfully transcribing a spec's intent, then asks whether the schema
  accepts that intent.
- §S7 — Domain failure-mode categories explicit in the test class names.

Each test class corresponds to one schema-permissiveness category. Each test
inside a class is one concrete instance of the category. Today most tests
xfail (the schema rejects); after Phase 3.5.B's v0.3 schema expansion they
should pass.

What this battery is NOT testing (acknowledged gaps, candidates for future
expansion):
- Schema STRICTNESS edge cases (descriptors that SHOULD be rejected but
  aren't). Covered by a separate negative-test battery TBD.
- Cross-version compatibility (v0.1 descriptors validating against v0.2
  schema). Covered by P2.D's existing test suite.
- Performance / resource exhaustion (very deeply nested schemas, very large
  field counts). Out of scope for permissiveness battery; performance
  battery TBD.
- Semantic mismatches between declared schema kind and actual field
  contents. That's a separate type-checking concern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.descriptor import validate


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _minimal_descriptor(
    *,
    inputs: list,
    outputs: list,
    schemas: dict | None = None,
    descriptor_version: str = "0.1",
) -> dict[str, Any]:
    """Return the smallest valid descriptor where we vary inputs/outputs/schemas only."""
    if descriptor_version == "0.3":
        classification: Any = {"primary_axis": "DSL"}
    else:
        classification = "DSL"
    return {
        "descriptor_version": descriptor_version,
        "mellea_version": "0.5.0",
        "skill": {"name": "schema-test", "classification": classification},
        "inputs": inputs,
        "outputs": outputs,
        "schemas": schemas or {},
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [],
    }


# =============================================================================
# Category 1: Compound input types — list, dict
# =============================================================================
# Adversarial framing: "I'm transcribing a skill whose input is a *list of
# files* or a *dict of options*. The schema should accept that shape directly."


class TestCompoundInputs:

    def test_input_kind_list_of_model_ref(self, surface):
        """A spec's input is a list of structured items. Author transcribes as
        kind: 'list' with of: pointing at a declared schema.

        v0.3: the schema must accept ``kind: 'list'`` of a ``{ref}``-shaped
        SchemaRef. (Was xfailed prior to Phase 3.5.B — now passes against
        v0.3.)
        """
        desc = _minimal_descriptor(
            inputs=[
                {"name": "files", "schema": {"kind": "list", "of": {"kind": "model_ref", "ref": "#/schemas/FileItem"}}}
            ],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            schemas={"FileItem": {"kind": "model", "fields": {"path": {"type": "str"}}}},
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"

    def test_input_kind_list_of_str(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "lines", "schema": {"kind": "list", "of": {"kind": "str"}}}],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"

    def test_input_kind_dict_str_to_str(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "options", "schema": {"kind": "dict", "key_type": "str", "value_type": "str"}}],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"


# =============================================================================
# Category 2: Optional flag at input level
# =============================================================================
# Adversarial framing: "The spec says 'optionally provide a config'. The schema
# allows 'optional' on internal fields but not on input slots."


class TestOptionalInputs:

    def test_input_str_with_optional_true(self, surface):
        """Common pattern: a string input that's optional. Spec says 'optionally
        provide a context'.

        v0.3: Slot now accepts ``optional: true``.
        """
        desc = _minimal_descriptor(
            inputs=[
                {"name": "context", "schema": {"kind": "str"}, "optional": True},
                {"name": "required_input", "schema": {"kind": "str"}},
            ],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"

    def test_input_model_ref_with_optional_true(self, surface):
        desc = _minimal_descriptor(
            inputs=[
                {"name": "config", "schema": {"kind": "model_ref", "ref": "#/schemas/Config"}, "optional": True}
            ],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            schemas={"Config": {"kind": "model", "fields": {"verbose": {"type": "bool"}}}},
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"


# =============================================================================
# Category 3: Literal-union input types
# =============================================================================
# Adversarial framing: "The spec says input is 'either red, blue, or green'.
# Pydantic models this as Literal['red','blue','green']. Can the schema
# express this at input level without forcing a separate enum class?"


class TestLiteralUnionInputs:

    def test_input_literal_union_of_strings(self, surface):
        """v0.3 adds ``kind: 'literal_union'`` at the input/output boundary."""
        desc = _minimal_descriptor(
            inputs=[{"name": "color", "schema": {"kind": "literal_union", "members": ["red", "blue", "green"]}}],
            outputs=[{"name": "result", "schema": {"kind": "str"}}],
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"


# =============================================================================
# Category 4: Symmetric expressiveness — outputs vs inputs
# =============================================================================
# Adversarial framing: "Outputs should accept the same kind range as inputs.
# If kind: 'list' works at input, it must work at output. Asymmetric
# permissiveness is a footgun."


class TestSymmetricOutputs:

    def test_output_kind_list_of_model_ref(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "query", "schema": {"kind": "str"}}],
            outputs=[
                {"name": "findings", "schema": {"kind": "list", "of": {"kind": "model_ref", "ref": "#/schemas/Finding"}}}
            ],
            schemas={"Finding": {"kind": "model", "fields": {"file": {"type": "str"}}}},
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"

    def test_output_kind_dict_str_to_model_ref(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "query", "schema": {"kind": "str"}}],
            outputs=[
                {"name": "results", "schema": {"kind": "dict", "key_type": "str",
                                                "value_type": {"ref": "#/schemas/Item"}}}
            ],
            schemas={"Item": {"kind": "model", "fields": {"value": {"type": "int"}}}},
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"


# =============================================================================
# Category 5: Schema documentation / description fields
# =============================================================================
# Adversarial framing: "Pydantic models accept a 'description' on every field
# (used for `Field(description=...)`). Reference compilation `schemas.py`
# uses these. Can the descriptor express them?"


class TestFieldDescriptions:

    def test_schema_field_with_description(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "query", "schema": {"kind": "str"}}],
            outputs=[{"name": "result", "schema": {"kind": "model_ref", "ref": "#/schemas/Result"}}],
            schemas={
                "Result": {
                    "kind": "model",
                    "fields": {
                        "verdict": {
                            "type": "str",
                            "description": "The final verdict — one of 'pass', 'fail', 'inconclusive'."
                        }
                    },
                }
            },
            descriptor_version="0.3",
        )
        report = validate(desc, schema_version="0.3", surface=surface)
        assert report.valid, f"validator rejected: {[(e.path, e.message) for e in report.errors]}"


# =============================================================================
# Category 6: Edge-of-vocabulary kinds
# =============================================================================
# Adversarial framing: "What kinds did v0.1's strawman support that the
# validator might have implicitly dropped? Test every documented kind."


class TestDocumentedKindsBaseline:
    """Baseline: every kind in the schema's documented vocabulary MUST be
    accepted. These should ALL pass today — if they don't, the validator has
    a regression against its own documentation."""

    def test_input_str(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "x", "schema": {"kind": "str"}}],
            outputs=[{"name": "y", "schema": {"kind": "str"}}],
        )
        assert validate(desc, schema_version="0.1", surface=surface).valid

    def test_input_pydantic_model(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "doc", "schema": {"kind": "pydantic_model", "ref": "#/schemas/Doc"}}],
            outputs=[{"name": "y", "schema": {"kind": "str"}}],
            schemas={"Doc": {"kind": "model", "fields": {"text": {"type": "str"}}}},
        )
        assert validate(desc, schema_version="0.1", surface=surface).valid

    def test_input_model_ref(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "doc", "schema": {"kind": "model_ref", "ref": "#/schemas/Doc"}}],
            outputs=[{"name": "y", "schema": {"kind": "str"}}],
            schemas={"Doc": {"kind": "model", "fields": {"text": {"type": "str"}}}},
        )
        assert validate(desc, schema_version="0.1", surface=surface).valid

    def test_input_enum_ref(self, surface):
        desc = _minimal_descriptor(
            inputs=[{"name": "color", "schema": {"kind": "enum_ref", "ref": "#/schemas/Color"}}],
            outputs=[{"name": "y", "schema": {"kind": "str"}}],
            schemas={"Color": {"kind": "enum", "members": ["red", "blue", "green"]}},
        )
        assert validate(desc, schema_version="0.1", surface=surface).valid


# =============================================================================
# Category 7: Property — symmetric input/output kind vocabulary
# =============================================================================
# Strategy §S4 (property-based): for every accepted INPUT kind, the same kind
# must be accepted as an OUTPUT. Asymmetric expressiveness is a footgun and
# should be caught as a property violation.


@pytest.mark.parametrize(
    "kind_spec",
    [
        {"kind": "str"},
        {"kind": "pydantic_model", "ref": "#/schemas/X"},
        {"kind": "model_ref", "ref": "#/schemas/X"},
        {"kind": "enum_ref", "ref": "#/schemas/E"},
    ],
)
def test_property_input_output_symmetric_for_documented_kinds(surface, kind_spec):
    """Property: any kind valid at inputs is valid at outputs (and vice versa)."""
    schemas = {
        "X": {"kind": "model", "fields": {"v": {"type": "str"}}},
        "E": {"kind": "enum", "members": ["a", "b"]},
    }
    as_input = _minimal_descriptor(
        inputs=[{"name": "x", "schema": kind_spec}],
        outputs=[{"name": "y", "schema": {"kind": "str"}}],
        schemas=schemas,
    )
    as_output = _minimal_descriptor(
        inputs=[{"name": "x", "schema": {"kind": "str"}}],
        outputs=[{"name": "y", "schema": kind_spec}],
        schemas=schemas,
    )
    in_report = validate(as_input, schema_version="0.1", surface=surface)
    out_report = validate(as_output, schema_version="0.1", surface=surface)
    assert in_report.valid == out_report.valid, (
        f"kind {kind_spec!r} asymmetric: input.valid={in_report.valid} "
        f"vs output.valid={out_report.valid}. "
        f"input errors: {[e.message for e in in_report.errors]} "
        f"output errors: {[e.message for e in out_report.errors]}"
    )
