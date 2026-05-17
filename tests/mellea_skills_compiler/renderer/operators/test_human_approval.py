"""Tests for the ``human_approval`` composition operator (v0.3 RFC §3.3).

The operator emits a callback-aware prompt-and-branch block; tests cover:
- default stdin-based prompt
- ``_approval_cb`` import from ``runtime_overrides`` (callback hook)
- approved/rejected branch dispatch
- empty rejected branch handling
"""
from __future__ import annotations

import ast

import pytest

import mellea_skills_compiler.renderer.operators  # noqa: F401
from mellea_skills_compiler.renderer import RendererError, get_operator
from tests.mellea_skills_compiler.renderer.operators._helpers import (
    assert_compiles,
    bind_context,
    make_render_body,
    make_test_context,
)


def _human_approval_node(*, branches: dict | None = None, prompt: dict | None = None) -> dict:
    return {
        "id": "gate",
        "kind": "composition",
        "operator": "human_approval",
        "prompt": prompt or {"value": "Run the loop?"},
        "branches": branches or {
            "approved": [{"id": "ok", "kind": "call", "symbol": "ok"}],
            "rejected": [{"id": "abort", "kind": "call", "symbol": "abort"}],
        },
    }


def _render(node: dict):
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("human_approval")
    stmts = op.render(node, ctx, make_render_body())
    return ctx, stmts


def test_default_emits_stdin_fallback():
    ctx, stmts = _render(_human_approval_node())
    src = assert_compiles(stmts)
    assert "input('> ')" in src
    assert "'y', 'yes'" in src or "('y', 'yes')" in src


def test_callback_hook_imports_from_runtime_overrides():
    """A test harness can override the prompt by providing _human_approval_callback."""
    ctx, stmts = _render(_human_approval_node())
    src = assert_compiles(stmts)
    assert "from .runtime_overrides import _human_approval_callback as _approval_cb" in src
    assert "_approval_cb = None" in src  # the except-ImportError fallback


def test_callback_branch_dispatches_when_present():
    ctx, stmts = _render(_human_approval_node())
    src = assert_compiles(stmts)
    # When callback is not None, it should be invoked with the prompt.
    assert "if _approval_cb is not None" in src
    assert "_approved = _approval_cb(" in src


def test_approved_branch_dispatch():
    ctx, stmts = _render(_human_approval_node())
    src = assert_compiles(stmts)
    # if _approved: <approved>; else: <rejected>
    assert "if _approved:" in src
    assert "gate = ok" in src
    assert "gate = abort" in src


def test_rejected_branch_can_be_empty():
    """An empty rejected branch should bind gate = None gracefully."""
    node = _human_approval_node(branches={
        "approved": [{"id": "ok", "kind": "call", "symbol": "ok"}],
        "rejected": [],
    })
    ctx, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "gate = None" in src


def test_prompt_value_propagates():
    """A simple {"value": "..."} prompt becomes a string literal in print()."""
    node = _human_approval_node(prompt={"value": "Approve?"})
    ctx, stmts = _render(node)
    src = assert_compiles(stmts)
    assert "Approve?" in src


def test_missing_branches_rejected():
    node = {
        "id": "gate",
        "kind": "composition",
        "operator": "human_approval",
        "prompt": {"value": "y?"},
        "branches": {"approved": []},  # missing 'rejected'
    }
    with pytest.raises(RendererError, match="must define both 'approved' and 'rejected'"):
        _render(node)


def test_missing_prompt_rejected():
    node = {
        "id": "gate",
        "kind": "composition",
        "operator": "human_approval",
        "branches": {"approved": [], "rejected": []},
    }
    with pytest.raises(RendererError, match="missing required field 'prompt'"):
        _render(node)
