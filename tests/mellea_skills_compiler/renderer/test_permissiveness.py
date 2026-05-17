"""Tests for v0.3 input/output type permissiveness (RFC §3.5).

Covers:
- ``list[T]``
- ``dict[K, V]``
- ``literal_union``
- ``optional: true`` on Slot
- field ``description`` propagation to ``Field(description=...)``
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import render_descriptor
from mellea_skills_compiler.renderer.schemas import render_schemas


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _wrap(inputs: list[dict], outputs: list[dict] | None = None, schemas: dict | None = None) -> dict:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "perm-tests",
                  "classification": {"primary_axis": "DSL"}},
        "inputs": inputs,
        "outputs": outputs or [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": schemas or {},
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [
            {"id": "ans", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"value": "x"}}}
        ],
    }


# --- list -----------------------------------------------------------------


def test_list_of_ref_emits_list_with_class(surface):
    desc = _wrap(
        inputs=[{"name": "items", "schema": {"kind": "list",
                                              "of": {"ref": "#/schemas/Item"}}}],
        schemas={"Item": {"kind": "model", "fields": {"name": {"type": "str"}}}},
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "items: list[Item]" in src


def test_list_of_str(surface):
    desc = _wrap(
        inputs=[{"name": "tags", "schema": {"kind": "list",
                                             "of": {"kind": "str"}}}],
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "tags: list[str]" in src


# --- dict -----------------------------------------------------------------


def test_dict_str_to_str(surface):
    desc = _wrap(
        inputs=[{"name": "opts", "schema": {"kind": "dict",
                                              "key_type": "str",
                                              "value_type": "str"}}],
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "opts: dict[str, str]" in src


def test_dict_str_to_ref(surface):
    desc = _wrap(
        inputs=[{"name": "results",
                  "schema": {"kind": "dict", "key_type": "str",
                              "value_type": {"ref": "#/schemas/Result"}}}],
        schemas={"Result": {"kind": "model", "fields": {"ok": {"type": "bool"}}}},
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "results: dict[str, Result]" in src


# --- literal_union --------------------------------------------------------


def test_literal_union(surface):
    desc = _wrap(
        inputs=[{"name": "color",
                  "schema": {"kind": "literal_union",
                              "members": ["red", "green", "blue"]}}],
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "Literal['red', 'green', 'blue']" in src
    assert "from typing import" in src and "Literal" in src


# --- Optional slot --------------------------------------------------------


def test_optional_slot_adds_pipe_none_and_default(surface):
    desc = _wrap(
        inputs=[
            {"name": "q", "schema": {"kind": "str"}},
            {"name": "config", "schema": {"kind": "str"}, "optional": True},
        ],
    )
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    # The optional input gets `| None` annotation and `= None` default.
    assert "config: str | None=None" in src or "config: str | None = None" in src


# --- field descriptions ---------------------------------------------------


def test_field_description_becomes_Field_kwarg():
    schemas_block = {
        "Answer": {
            "kind": "model",
            "fields": {
                "verdict": {"type": "str", "description": "The final verdict."},
            },
        }
    }
    src = render_schemas(schemas_block)
    assert "Field(description='The final verdict.')" in src


def test_field_description_with_optional():
    schemas_block = {
        "Answer": {
            "kind": "model",
            "fields": {
                "score": {"type": "float", "optional": True,
                           "description": "0.0–1.0 confidence."},
            },
        }
    }
    src = render_schemas(schemas_block)
    # Field(default=None, description='...')
    assert "default=None" in src
    assert "0.0–1.0" in src or "0.0–1.0" in src


def test_field_description_with_collection():
    schemas_block = {
        "Answer": {
            "kind": "model",
            "fields": {
                "tags": {"type": "list[str]", "description": "Free-form tags."},
            },
        }
    }
    src = render_schemas(schemas_block)
    assert "default_factory=list" in src
    assert "description='Free-form tags.'" in src


# --- Optional[T] in schema fields ----------------------------------------


def test_optional_T_type_string_in_schema_fields():
    """``Optional[float]`` should become ``float | None``."""
    schemas_block = {
        "Answer": {
            "kind": "model",
            "fields": {"score": {"type": "Optional[float]"}},
        }
    }
    src = render_schemas(schemas_block)
    assert "score: float | None" in src
