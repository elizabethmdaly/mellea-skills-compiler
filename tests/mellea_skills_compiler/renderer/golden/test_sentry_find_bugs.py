"""Golden regression: the hand-authored ``sentry-find-bugs`` descriptor.

The descriptor exercises the ``map`` operator at three different depths plus
two top-level call nodes; if any of the assertions here fail, a renderer
regression has broken the most-used composition operator in the corpus.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture
def rendered(load_descriptor, render_and_smoke):
    return render_and_smoke(load_descriptor("sentry-find-bugs"))


def test_render_descriptor_does_not_raise(rendered) -> None:
    """The render call returned a RenderResult with non-empty pipeline_py."""
    assert rendered.pipeline_src
    assert "def run_pipeline(" in rendered.pipeline_src


def test_schemas_module_exposes_findings_report(rendered) -> None:
    """Top-level output schema must be a class in schemas.py."""
    assert hasattr(rendered.schemas_module, "FindingsReport")
    assert hasattr(rendered.schemas_module, "IssueReport")
    assert hasattr(rendered.schemas_module, "Severity")  # enum
    assert hasattr(rendered.schemas_module, "ModifiedFiles")


def test_pipeline_run_pipeline_signature(rendered) -> None:
    """``run_pipeline(diff: str) -> FindingsReport`` per the descriptor."""
    fn = getattr(rendered.pipeline_module, "run_pipeline", None)
    assert callable(fn)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["diff"]
    # PEP 563 stringised under `from __future__ import annotations`.
    assert sig.parameters["diff"].annotation == "str"
    assert sig.return_annotation == "FindingsReport"


def test_state_assigned_at_module_level(rendered) -> None:
    """`backend` and `session` must be assigned before the function."""
    src = rendered.pipeline_src
    fn_idx = src.find("def run_pipeline")
    assert "backend = OpenAIBackend(" in src[:fn_idx]
    assert "session = MelleaSession(" in src[:fn_idx]


def test_three_map_operators_emit_for_loops(rendered) -> None:
    """sentry-find-bugs descriptor uses ``map`` three times; we should see
    three corresponding `for ... in ...:` lines in the rendered pipeline."""
    src = rendered.pipeline_src
    # Each map emits `for <item_id> in <iterable>:`.
    for_lines = [ln for ln in src.splitlines() if ln.strip().startswith("for ") and ln.strip().endswith(":")]
    # at least 3 (one per descriptor map node).
    assert len(for_lines) >= 3, f"expected >=3 for-loops in rendered source, got {for_lines}"


def test_audit_capture_node_is_returned(rendered) -> None:
    """`audit` is the node whose ``captures`` declares ``#/outputs/findings``."""
    assert "return audit" in rendered.pipeline_src
