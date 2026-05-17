"""Golden regression: the hand-authored ``security-engineer`` descriptor.

The simplest of the three goldens — single-call pipeline emitting a
``SecurityResponse``. Acts as a smoke check that descriptors without any
composition still render cleanly through the renderer core.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture
def rendered(load_descriptor, render_and_smoke):
    return render_and_smoke(load_descriptor("security-engineer"))


def test_render_descriptor_does_not_raise(rendered) -> None:
    assert rendered.pipeline_src
    assert "def run_pipeline(" in rendered.pipeline_src


def test_schemas_module_exposes_response_schema(rendered) -> None:
    assert hasattr(rendered.schemas_module, "SecurityResponse")
    assert hasattr(rendered.schemas_module, "ResponseKind")
    assert hasattr(rendered.schemas_module, "Severity")
    assert hasattr(rendered.schemas_module, "ComplianceFramework")
    assert hasattr(rendered.schemas_module, "IncidentAction")


def test_pipeline_run_pipeline_signature(rendered) -> None:
    fn = getattr(rendered.pipeline_module, "run_pipeline", None)
    assert callable(fn)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["request"]
    assert sig.parameters["request"].annotation == "str"
    assert sig.return_annotation == "SecurityResponse"


def test_state_assigned_at_module_level(rendered) -> None:
    src = rendered.pipeline_src
    fn_idx = src.find("def run_pipeline")
    assert "backend = OpenAIBackend(" in src[:fn_idx]
    assert "session = MelleaSession(" in src[:fn_idx]


def test_no_composition_no_for_loops(rendered) -> None:
    """security-engineer is a single-call pipeline; no composition -> no
    for-loops or ThreadPoolExecutor or if-chains driven by descriptor branches."""
    src = rendered.pipeline_src
    for_lines = [
        ln for ln in src.splitlines()
        if ln.strip().startswith("for ") and ln.strip().endswith(":")
    ]
    assert not for_lines, f"unexpected for-loop emitted: {for_lines}"
    assert "ThreadPoolExecutor" not in src


def test_respond_node_is_returned(rendered) -> None:
    """`respond` captures `#/outputs/response`."""
    assert "return respond" in rendered.pipeline_src
