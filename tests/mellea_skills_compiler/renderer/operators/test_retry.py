"""Tests for the ``retry_with_feedback`` composition operator."""

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
    op = get_operator("retry_with_feedback")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def _minimal_retry(**overrides) -> dict:
    node = {
        "id": "verified",
        "kind": "composition",
        "operator": "retry_with_feedback",
        "body": [{"id": "draft", "kind": "call", "symbol": "m.draft"}],
        "max_attempts": 3,
        "failure_check": {"ref": "is_valid"},
    }
    node.update(overrides)
    return node


def test_happy_path() -> None:
    _, stmts = _render(_minimal_retry())
    assert len(stmts) == 2
    for_stmt, bind = stmts
    assert isinstance(for_stmt, ast.For)
    assert isinstance(bind, ast.Assign)
    src = assert_compiles(stmts)
    assert "for _attempt in range(3):" in src
    assert "if is_valid:" in src
    assert "break" in src
    assert "else:" in src
    assert "raise RuntimeError" in src
    assert "verified = draft" in src


def test_missing_required_field_raises() -> None:
    for missing in ("body", "max_attempts", "failure_check"):
        node = _minimal_retry()
        del node[missing]
        with pytest.raises(RendererError, match=f"missing required field '{missing}'"):
            _render(node)


def test_zero_max_attempts_raises() -> None:
    node = _minimal_retry(max_attempts=0)
    with pytest.raises(RendererError, match="'max_attempts' must be > 0"):
        _render(node)


def test_negative_max_attempts_raises() -> None:
    node = _minimal_retry(max_attempts=-1)
    with pytest.raises(RendererError, match="'max_attempts' must be > 0"):
        _render(node)


def test_max_attempts_must_be_int_not_bool() -> None:
    node = _minimal_retry(max_attempts=True)
    with pytest.raises(RendererError, match="'max_attempts' must be int, got bool"):
        _render(node)


def test_empty_body_raises() -> None:
    node = _minimal_retry()
    node["body"] = []
    with pytest.raises(RendererError, match="must contain at least one node"):
        _render(node)


def test_last_body_node_without_id_raises() -> None:
    node = _minimal_retry()
    node["body"] = [{"kind": "call", "symbol": "m.f"}]
    with pytest.raises(RendererError, match="must have an 'id'"):
        _render(node)


def test_source_map_anchors_on_for_loop() -> None:
    ctx, stmts = _render(_minimal_retry())
    assert ctx.source_map.has_node("verified")
    recorded = [stmt for nid, stmt, _ in ctx.source_map.entries if nid == "verified"]
    assert any(isinstance(s, ast.For) for s in recorded)


def test_multi_node_body() -> None:
    node = _minimal_retry()
    node["body"] = [
        {"id": "draft", "kind": "call", "symbol": "m.draft"},
        {"id": "polish", "kind": "call", "symbol": "m.polish"},
    ]
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "draft = " in src and "polish = " in src
    assert "verified = polish" in src


def test_failure_check_supports_select_ref() -> None:
    node = _minimal_retry(failure_check={"ref": "result", "select": "is_complete"})
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "if result.is_complete:" in src
