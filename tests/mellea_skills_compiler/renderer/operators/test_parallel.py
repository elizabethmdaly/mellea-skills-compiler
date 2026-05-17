"""Tests for the ``parallel`` composition operator."""

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
    op = get_operator("parallel")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def _two_branch_parallel() -> dict:
    return {
        "id": "fan_out",
        "kind": "composition",
        "operator": "parallel",
        "branches": [
            [{"id": "summarise", "kind": "call", "symbol": "m.summarise"}],
            [{"id": "classify", "kind": "call", "symbol": "m.classify"}],
        ],
    }


def test_happy_path_two_branches_dict_collect() -> None:
    _, stmts = _render(_two_branch_parallel())
    # 2 FunctionDefs + 1 With.
    assert len(stmts) == 3
    assert isinstance(stmts[0], ast.FunctionDef)
    assert isinstance(stmts[1], ast.FunctionDef)
    assert isinstance(stmts[2], ast.With)
    src = assert_compiles(stmts)
    assert "ThreadPoolExecutor(max_workers=2)" in src
    assert "fan_out = " in src
    # dict_by_branch_id default — keys are the branch last-ids.
    assert "'summarise'" in src and "'classify'" in src


def test_list_collect() -> None:
    node = _two_branch_parallel()
    node["collect"] = "list"
    _, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "fan_out = [" in src or "fan_out = [_f" in src


def test_missing_branches_raises() -> None:
    node = _two_branch_parallel()
    del node["branches"]
    with pytest.raises(RendererError, match="missing required field 'branches'"):
        _render(node)


def test_empty_branches_raises() -> None:
    node = _two_branch_parallel()
    node["branches"] = []
    with pytest.raises(RendererError, match="'branches' must be non-empty"):
        _render(node)


def test_branch_not_list_raises() -> None:
    node = _two_branch_parallel()
    node["branches"] = [[{"id": "ok", "kind": "call", "symbol": "m.f"}], "nope"]
    with pytest.raises(RendererError, match=r"branches\[1\] must be a list"):
        _render(node)


def test_empty_branch_raises() -> None:
    node = _two_branch_parallel()
    node["branches"] = [[]]
    with pytest.raises(RendererError, match="must contain at least one node"):
        _render(node)


def test_branch_last_node_without_id_raises() -> None:
    node = _two_branch_parallel()
    node["branches"] = [[{"kind": "call", "symbol": "m.f"}]]
    with pytest.raises(RendererError, match="must have an 'id'"):
        _render(node)


def test_invalid_collect_raises() -> None:
    node = _two_branch_parallel()
    node["collect"] = "frobnicate"
    with pytest.raises(RendererError, match="'collect' must be one of"):
        _render(node)


def test_import_concurrent_futures_registered() -> None:
    ctx, _ = _render(_two_branch_parallel())
    assert "concurrent.futures" in ctx.imports


def test_source_map_anchors_on_with() -> None:
    ctx, stmts = _render(_two_branch_parallel())
    assert ctx.source_map.has_node("fan_out")
    recorded = [stmt for nid, stmt, _ in ctx.source_map.entries if nid == "fan_out"]
    assert any(isinstance(s, ast.With) for s in recorded)


def test_three_branches_emits_three_functions() -> None:
    node = _two_branch_parallel()
    node["branches"].append([{"id": "extract", "kind": "call", "symbol": "m.extract"}])
    _, stmts = _render(node)
    fn_defs = [s for s in stmts if isinstance(s, ast.FunctionDef)]
    assert len(fn_defs) == 3
    names = {fn.name for fn in fn_defs}
    assert names == {"_fan_out_branch_0", "_fan_out_branch_1", "_fan_out_branch_2"}
    src = assert_compiles(stmts)
    assert "ThreadPoolExecutor(max_workers=3)" in src
