"""Tests for the ``sequential`` composition operator."""

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
    op = get_operator("sequential")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def test_happy_path_emits_body_in_order() -> None:
    node = {
        "id": "seq",
        "kind": "composition",
        "operator": "sequential",
        "body": [
            {"id": "a", "kind": "call", "symbol": "mod.fn_a"},
            {"id": "b", "kind": "call", "symbol": "mod.fn_b"},
            {"id": "c", "kind": "call", "symbol": "mod.fn_c"},
        ],
    }
    _, stmts = _render(node)
    assert len(stmts) == 3
    assert all(isinstance(s, ast.Assign) for s in stmts)
    src = assert_compiles(stmts)
    assert src.index("a = ") < src.index("b = ") < src.index("c = ")


def test_missing_body_raises_renderer_error() -> None:
    node = {"id": "seq", "kind": "composition", "operator": "sequential"}
    with pytest.raises(RendererError, match="missing required field 'body'"):
        _render(node)


def test_body_not_list_raises_renderer_error() -> None:
    node = {"id": "seq", "kind": "composition", "operator": "sequential", "body": "not a list"}
    with pytest.raises(RendererError, match="must be list"):
        _render(node)


def test_empty_body_emits_nothing_no_source_entry() -> None:
    node = {"id": "seq", "kind": "composition", "operator": "sequential", "body": []}
    ctx, stmts = _render(node)
    assert stmts == []
    assert not ctx.source_map.has_node("seq")


def test_source_map_records_first_statement() -> None:
    node = {
        "id": "seq",
        "kind": "composition",
        "operator": "sequential",
        "body": [{"id": "x", "kind": "call", "symbol": "m.f"}],
    }
    ctx, stmts = _render(node)
    assert ctx.source_map.has_node("seq")
    # The recorded stmt should be the first emitted one.
    recorded_stmts = [stmt for nid, stmt, _ in ctx.source_map.entries if nid == "seq"]
    assert recorded_stmts and recorded_stmts[0] is stmts[0]


def test_nested_sequential_flattens_through_render_body() -> None:
    node = {
        "id": "outer",
        "kind": "composition",
        "operator": "sequential",
        "body": [
            {"id": "first", "kind": "call", "symbol": "m.f"},
            {
                "id": "inner",
                "kind": "composition",
                "operator": "sequential",
                "body": [
                    {"id": "second", "kind": "call", "symbol": "m.g"},
                    {"id": "third", "kind": "call", "symbol": "m.h"},
                ],
            },
            {"id": "fourth", "kind": "call", "symbol": "m.i"},
        ],
    }
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    for ident in ("first", "second", "third", "fourth"):
        assert f"{ident} = " in src
    assert src.index("first") < src.index("second") < src.index("third") < src.index("fourth")
