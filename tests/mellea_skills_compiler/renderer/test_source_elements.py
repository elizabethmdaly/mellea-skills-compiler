"""Tests for ``source_elements: ["E14", ...]`` metadata propagation (RFC §3.6.2).

The renderer ignores ``source_elements`` for code generation but propagates
them to a render-result map so Step 6's ``mapping_report.md`` builder can
trace each pipeline node back to its inventory.json element ids.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import render_descriptor


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _descriptor_with_source_elements(elements: list[str]) -> dict[str, Any]:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "src-elem-test",
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
             "args": {"description": {"value": "x"}},
             "source_elements": elements}
        ],
    }


def test_source_elements_in_render_result_map(surface):
    desc = _descriptor_with_source_elements(["E14", "E15"])
    result = render_descriptor(desc, surface)
    assert "ans" in result.source_elements_map
    assert result.source_elements_map["ans"] == ["E14", "E15"]


def test_source_elements_do_not_change_emitted_code(surface):
    """Adding source_elements must not affect rendered Python source."""
    a = render_descriptor(_descriptor_with_source_elements([]), surface)
    b = render_descriptor(_descriptor_with_source_elements(["E14"]), surface)
    # Strip source_elements metadata from the descriptor to confirm only the
    # source_map differs.
    assert a.pipeline_py == b.pipeline_py


def test_source_elements_omitted_yields_empty_map(surface):
    desc = _descriptor_with_source_elements([])
    # An explicitly empty list should not appear in the map.
    desc["pipeline"][0].pop("source_elements")
    result = render_descriptor(desc, surface)
    assert result.source_elements_map == {}
