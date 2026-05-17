"""Tests for the ``branch`` composition operator.

Empirically load-bearing per Phase 0.3c zero-ref D″ runs: the LLM
consistently selects ``branch`` for NDA-review-style skills, so a correct
implementation here is the dispatcher most real descriptors depend on.
"""

from __future__ import annotations

import ast

import pytest

import mellea_skills_compiler.renderer.operators  # noqa: F401 — registers operators
from mellea_skills_compiler.renderer import RendererError, get_operator
from tests.mellea_skills_compiler.renderer.operators._helpers import (
    assert_compiles,
    bind_context,
    make_render_body,
    make_test_context,
)


def _render(node: dict):
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("branch")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def _two_case_branch() -> dict:
    return {
        "id": "triage",
        "kind": "composition",
        "operator": "branch",
        "on": {"ref": "classify", "select": "verdict"},
        "cases": {
            "compliant": [{"id": "short_path", "kind": "call", "symbol": "m.short"}],
            "non_compliant": [{"id": "long_path", "kind": "call", "symbol": "m.long"}],
        },
    }


def test_happy_path_two_cases() -> None:
    _, stmts = _render(_two_case_branch())
    assert len(stmts) == 1
    assert isinstance(stmts[0], ast.If)
    src = assert_compiles(stmts)
    assert "if classify.verdict == 'compliant':" in src
    assert "elif classify.verdict == 'non_compliant':" in src


def test_default_case_emitted_as_else() -> None:
    node = _two_case_branch()
    node["cases"]["_default"] = [{"id": "fallback", "kind": "call", "symbol": "m.fb"}]
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "else:" in src
    assert "fallback = " in src


def test_branch_result_bound_in_each_case() -> None:
    _, stmts = _render(_two_case_branch())
    src = assert_compiles(stmts)
    # The branch's id must be assigned in every arm.
    assert "triage = short_path" in src
    assert "triage = long_path" in src


def test_missing_required_field_raises() -> None:
    for missing in ("on", "cases"):
        node = _two_case_branch()
        del node[missing]
        with pytest.raises(RendererError, match=f"missing required field '{missing}'"):
            _render(node)


def test_empty_cases_raises() -> None:
    node = _two_case_branch()
    node["cases"] = {}
    with pytest.raises(RendererError, match="'cases' must be non-empty"):
        _render(node)


def test_case_body_not_list_raises() -> None:
    node = _two_case_branch()
    node["cases"]["compliant"] = "nope"
    with pytest.raises(RendererError, match="body must be a list"):
        _render(node)


def test_empty_case_body_raises() -> None:
    node = _two_case_branch()
    node["cases"]["compliant"] = []
    with pytest.raises(RendererError, match="must contain at least one node"):
        _render(node)


def test_case_last_node_without_id_raises() -> None:
    node = _two_case_branch()
    node["cases"]["compliant"] = [{"kind": "call", "symbol": "m.f"}]
    with pytest.raises(RendererError, match="must have an 'id'"):
        _render(node)


def test_no_default_emits_runtime_raise() -> None:
    """When the descriptor omits _default the rendered else clause raises."""
    _, stmts = _render(_two_case_branch())
    src = assert_compiles(stmts)
    # Closed-world dispatch — runtime fails loud on unexpected values.
    assert "raise RuntimeError" in src


def test_multi_case_preserves_descriptor_order() -> None:
    node = {
        "id": "level",
        "kind": "composition",
        "operator": "branch",
        "on": {"ref": "verdict"},
        "cases": {
            "critical": [{"id": "p1", "kind": "call", "symbol": "m.p1"}],
            "high": [{"id": "p2", "kind": "call", "symbol": "m.p2"}],
            "medium": [{"id": "p3", "kind": "call", "symbol": "m.p3"}],
            "low": [{"id": "p4", "kind": "call", "symbol": "m.p4"}],
        },
    }
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    # All four arms appear in declared order.
    assert src.index("'critical'") < src.index("'high'") < src.index("'medium'") < src.index("'low'")


def test_source_map_anchors_on_if() -> None:
    ctx, stmts = _render(_two_case_branch())
    assert ctx.source_map.has_node("triage")
    recorded = [stmt for nid, stmt, _ in ctx.source_map.entries if nid == "triage"]
    assert any(isinstance(s, ast.If) for s in recorded)


def test_only_default_case_emits_unconditional_block() -> None:
    node = {
        "id": "passthrough",
        "kind": "composition",
        "operator": "branch",
        "on": {"ref": "anything"},
        "cases": {"_default": [{"id": "single", "kind": "call", "symbol": "m.s"}]},
    }
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    # No 'if' chain — the body is emitted directly.
    assert "if " not in src or "passthrough = single" in src
