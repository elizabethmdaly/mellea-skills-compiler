"""Tests for v0.3 modality patterns at the renderer level.

Covers:
- per-call ``streaming: true`` (RFC §3.4 G8.2)
- skill-level ``async: true`` transitive transform (plan §11.9)
- image/document input slots (RFC §3.4 G8.4)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import render_descriptor


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _baseline_v03_descriptor() -> dict[str, Any]:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "mod-tests",
                  "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "q", "schema": {"kind": "str"}}],
        "outputs": [{"name": "r", "schema": {"kind": "str"}}],
        "schemas": {},
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
             "args": {"description": {"value": "answer"}}}
        ],
    }


# --- Streaming ------------------------------------------------------------


def test_streaming_call_emits_with_stream_join(surface):
    desc = _baseline_v03_descriptor()
    desc["pipeline"][0]["streaming"] = True
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "with session.instruct" in src
    assert ".stream()" in src
    assert "''.join(" in src
    # The node id is still bound.
    assert "ans = " in src


def test_non_streaming_call_unchanged(surface):
    """When streaming is omitted/false, the call should look exactly like v0.2."""
    desc = _baseline_v03_descriptor()
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert ".stream()" not in src
    assert "ans = session.instruct(" in src


# --- Async ----------------------------------------------------------------


def test_async_pipeline_emits_async_def(surface):
    desc = _baseline_v03_descriptor()
    desc["skill"]["async"] = True
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "async def run_pipeline" in src


def test_async_pipeline_awaits_call(surface):
    """Every Mellea call site becomes ``await <call>``."""
    desc = _baseline_v03_descriptor()
    desc["skill"]["async"] = True
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "await session.instruct(" in src


def test_async_pipeline_compiles(surface):
    """The rendered async pipeline must parse cleanly."""
    desc = _baseline_v03_descriptor()
    desc["skill"]["async"] = True
    result = render_descriptor(desc, surface)
    # ast.parse will catch syntax errors.
    ast.parse(result.pipeline_py)


def test_async_streaming_uses_async_with(surface):
    """Streaming + async → ``async with`` for the stream context manager."""
    desc = _baseline_v03_descriptor()
    desc["skill"]["async"] = True
    desc["pipeline"][0]["streaming"] = True
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "async with" in src


def test_sync_default_does_not_emit_await(surface):
    """v0.2 / v0.3 without async: no await anywhere."""
    desc = _baseline_v03_descriptor()
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "await" not in src
    assert "async def" not in src


# --- Image/document inputs ------------------------------------------------


def test_image_input_annotation_is_bytes_or_path(surface):
    desc = _baseline_v03_descriptor()
    desc["inputs"] = [
        {"name": "photo", "schema": {"kind": "image", "format_hint": "any"}}
    ]
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "photo: bytes | Path" in src
    assert "from pathlib import Path" in src


def test_document_input_annotation_is_bytes_or_path(surface):
    desc = _baseline_v03_descriptor()
    desc["inputs"] = [
        {"name": "doc", "schema": {"kind": "document", "format_hint": "pdf"}}
    ]
    result = render_descriptor(desc, surface)
    src = result.pipeline_py
    assert "doc: bytes | Path" in src
