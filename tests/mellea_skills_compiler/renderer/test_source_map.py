"""Tests for the renderer's source-map data structure."""

from __future__ import annotations

import ast

import pytest

from mellea_skills_compiler.renderer.source_map import (
    DESCRIPTOR_NODE_ATTR,
    SourceMap,
    SourceMapEntry,
)


def _build_module(stmts: list[ast.stmt]) -> ast.Module:
    """Wrap a list of stmts in a Module and finalise locations."""
    m = ast.Module(body=stmts, type_ignores=[])
    ast.fix_missing_locations(m)
    return m


# --- Tagging --------------------------------------------------------------


def test_tag_stmt_attaches_attribute() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    out = SourceMap.tag_stmt(stmt, "node-a")
    assert out is stmt
    assert getattr(stmt, DESCRIPTOR_NODE_ATTR) == "node-a"


def test_tag_stmt_with_path() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    SourceMap.tag_stmt(stmt, "node-a", "pipeline[0]")
    assert SourceMap.get_tag(stmt) == "node-a"
    assert SourceMap.get_path(stmt) == "pipeline[0]"


def test_get_tag_returns_none_when_untagged() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    assert SourceMap.get_tag(stmt) is None
    assert SourceMap.get_path(stmt) == ""


# --- build_from_module / lookup ------------------------------------------


def test_build_from_module_records_single_assign() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    SourceMap.tag_stmt(stmt, "node-a")
    mod = _build_module([stmt])
    sm = SourceMap()
    sm.build_from_module(mod)
    assert sm.lookup(1) == "node-a"
    assert len(sm.entries) == 1


def test_lookup_returns_none_when_line_unmapped() -> None:
    sm = SourceMap()
    assert sm.lookup(42) is None


def test_inner_stmt_wins_on_overlapping_lines() -> None:
    """When a tagged for-loop and an inner tagged Assign overlap, lookup(inner_line)
    must return the INNER descriptor id (deepest scope = repair target)."""
    inner = ast.Assign(targets=[ast.Name(id="y")], value=ast.Constant(value=2))
    SourceMap.tag_stmt(inner, "inner-node")
    for_stmt = ast.For(
        target=ast.Name(id="i"),
        iter=ast.Call(func=ast.Name(id="range"), args=[ast.Constant(value=3)], keywords=[]),
        body=[inner],
        orelse=[],
    )
    SourceMap.tag_stmt(for_stmt, "outer-node")
    mod = _build_module([for_stmt])
    sm = SourceMap()
    sm.build_from_module(mod)

    # Reparse to find actual line numbers (since fix_missing_locations puts
    # everything on line 1; the source-map's build pass reparses internally).
    src = ast.unparse(mod)
    lines = src.splitlines()
    for_line = next(i + 1 for i, ln in enumerate(lines) if ln.startswith("for "))
    inner_line = next(i + 1 for i, ln in enumerate(lines) if "y = 2" in ln)
    assert for_line != inner_line  # sanity

    # The inner Assign's line resolves to the inner node id.
    assert sm.lookup(inner_line) == "inner-node"
    # The `for` line itself isn't covered by the inner Assign → outer wins.
    assert sm.lookup(for_line) == "outer-node"


def test_to_json_roundtrips_entry_data() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    SourceMap.tag_stmt(stmt, "node-a", "pipeline[0]")
    mod = _build_module([stmt])
    sm = SourceMap()
    sm.build_from_module(mod)
    payload = sm.to_json()
    assert payload == [
        {
            "descriptor_node_id": "node-a",
            "start_line": 1,
            "end_line": 1,
            "descriptor_path": "pipeline[0]",
        }
    ]


def test_lookup_entry_returns_full_entry() -> None:
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    SourceMap.tag_stmt(stmt, "node-z", "pipeline[2]")
    mod = _build_module([stmt])
    sm = SourceMap()
    sm.build_from_module(mod)
    entry = sm.lookup_entry(1)
    assert isinstance(entry, SourceMapEntry)
    assert entry.descriptor_node_id == "node-z"
    assert entry.descriptor_path == "pipeline[2]"
    assert entry.contains(1)


def test_record_kwarg_form_tags_stmt() -> None:
    """Operators (P2.B) call source_map.record(stmt=, node_id=) — exercise that path."""
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    sm = SourceMap()
    sm.record(stmt=stmt, node_id="op-anchor")
    mod = _build_module([stmt])
    sm.build_from_module(mod)
    assert sm.lookup(1) == "op-anchor"


def test_record_line_form_adds_entry_directly() -> None:
    sm = SourceMap()
    sm.record(line=5, node_id="manual-line")
    assert sm.lookup(5) == "manual-line"


def test_record_requires_node_id() -> None:
    sm = SourceMap()
    with pytest.raises(ValueError):
        sm.record(line=1, node_id="")


def test_untagged_stmt_is_not_recorded() -> None:
    """Imports etc. are intentionally NOT tagged; they shouldn't appear in the map."""
    stmt = ast.Assign(targets=[ast.Name(id="x")], value=ast.Constant(value=1))
    # no tag_stmt call
    mod = _build_module([stmt])
    sm = SourceMap()
    sm.build_from_module(mod)
    assert sm.entries == []
    assert sm.lookup(1) is None
