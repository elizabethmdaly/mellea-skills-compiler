"""Golden regression: the hand-authored ``security-review`` descriptor.

Shape: 2 sequential call nodes wrapping a single ``map`` over the
``detect_and_enumerate`` candidates list. Output is a model_ref to
``SecurityReviewReport``.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture
def rendered(load_descriptor, render_and_smoke):
    return render_and_smoke(load_descriptor("security-review"))


def test_render_descriptor_does_not_raise(rendered) -> None:
    assert rendered.pipeline_src
    assert "def run_pipeline(" in rendered.pipeline_src


def test_schemas_module_exposes_top_level_output(rendered) -> None:
    assert hasattr(rendered.schemas_module, "SecurityReviewReport")
    assert hasattr(rendered.schemas_module, "ConfirmedFinding")
    assert hasattr(rendered.schemas_module, "Severity")  # enum
    assert hasattr(rendered.schemas_module, "ConfidenceLevel")
    assert hasattr(rendered.schemas_module, "PotentialFindings")


def test_pipeline_run_pipeline_signature(rendered) -> None:
    fn = getattr(rendered.pipeline_module, "run_pipeline", None)
    assert callable(fn)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["code"]
    assert sig.parameters["code"].annotation == "str"
    assert sig.return_annotation == "SecurityReviewReport"


def test_state_assigned_at_module_level(rendered) -> None:
    src = rendered.pipeline_src
    fn_idx = src.find("def run_pipeline")
    assert "backend = OpenAIBackend(" in src[:fn_idx]
    assert "session = MelleaSession(" in src[:fn_idx]


def test_map_emits_for_loop(rendered) -> None:
    """One ``map`` operator -> at least one ``for ... in ...:`` line."""
    src = rendered.pipeline_src
    for_lines = [ln for ln in src.splitlines() if ln.strip().startswith("for ") and ln.strip().endswith(":")]
    assert len(for_lines) >= 1, f"expected >=1 for-loop, got {for_lines}"


def test_report_capture_node_is_returned(rendered) -> None:
    """`report` is the node that captures ``#/outputs/report``."""
    assert "return report" in rendered.pipeline_src
