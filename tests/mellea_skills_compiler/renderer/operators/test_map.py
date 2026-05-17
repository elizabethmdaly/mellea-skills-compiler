"""Tests for the ``map`` composition operator."""

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
    op = get_operator("map")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def _minimal_map(*, collect: str = "list", **overrides) -> dict:
    node = {
        "id": "per_item",
        "kind": "composition",
        "operator": "map",
        "over": {"ref": "items"},
        "body": [{"id": "process", "kind": "call", "symbol": "m.process"}],
        "collect": collect,
    }
    node.update(overrides)
    return node


def test_happy_path_list_collect() -> None:
    _, stmts = _render(_minimal_map())
    assert len(stmts) == 2
    init, for_stmt = stmts
    assert isinstance(init, ast.Assign)
    assert isinstance(init.value, ast.List)  # per_item = []
    assert isinstance(for_stmt, ast.For)
    src = assert_compiles(stmts)
    assert "per_item = []" in src
    assert "for _item in items:" in src
    assert "per_item.append(process)" in src


def test_dict_by_key_collect() -> None:
    node = _minimal_map(collect="dict_by_key", key_field="name")
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "per_item = {}" in src
    assert "per_item[process.name] = process" in src


def test_missing_required_field_raises(monkeypatch_each_field=None) -> None:
    for missing in ("over", "body", "collect"):
        node = _minimal_map()
        del node[missing]
        with pytest.raises(RendererError, match=f"missing required field '{missing}'"):
            _render(node)


def test_invalid_collect_raises() -> None:
    node = _minimal_map(collect="frobnicate")
    with pytest.raises(RendererError, match="'collect' must be one of"):
        _render(node)


def test_dict_by_key_requires_key_field() -> None:
    node = _minimal_map(collect="dict_by_key")
    with pytest.raises(RendererError, match="'key_field' is required"):
        _render(node)


def test_empty_body_raises() -> None:
    node = _minimal_map()
    node["body"] = []
    with pytest.raises(RendererError, match="must contain at least one node"):
        _render(node)


def test_last_body_node_without_id_raises() -> None:
    node = _minimal_map()
    node["body"] = [{"kind": "call", "symbol": "m.f"}]  # no id
    with pytest.raises(RendererError, match="must have an 'id'"):
        _render(node)


def test_custom_item_id() -> None:
    node = _minimal_map()
    node["item_id"] = "row"
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "for row in items:" in src


def test_invalid_item_id_raises() -> None:
    node = _minimal_map()
    node["item_id"] = "not an identifier"
    with pytest.raises(RendererError, match="'item_id' must be a valid Python identifier"):
        _render(node)


def test_source_map_anchors_on_for_loop() -> None:
    ctx, stmts = _render(_minimal_map())
    assert ctx.source_map.has_node("per_item")
    recorded_stmts = [stmt for nid, stmt, _ in ctx.source_map.entries if nid == "per_item"]
    assert any(isinstance(s, ast.For) for s in recorded_stmts)


def test_resolves_ref_with_select() -> None:
    node = _minimal_map()
    node["over"] = {"ref": "document", "select": "items.subitems"}
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "for _item in document.items.subitems:" in src


def test_register_id_called_for_accumulator() -> None:
    ctx, _ = _render(_minimal_map())
    assert ctx.ids.get("per_item") == "accumulator"
    assert ctx.ids.get("_item") == "loop_var"
