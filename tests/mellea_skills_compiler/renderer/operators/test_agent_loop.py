"""Tests for the ``agent_loop`` composition operator (v0.3 RFC §3.3).

Each test:
- builds an agent_loop node with one body element
- renders it via the operator
- unparses to source, asserts on shape (substring + ast parse)
- runs ``assert_compiles`` so we know the result is valid Python.
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


def _agent_loop_node(
    *,
    termination: dict | None = None,
    max_iterations: int = 5,
    initial_state=None,
    early_field: str | None = None,
    body: list[dict] | None = None,
) -> dict:
    """Build a synthetic agent_loop node for testing."""
    node: dict = {
        "id": "loop",
        "kind": "composition",
        "operator": "agent_loop",
        "max_iterations": max_iterations,
        "state_schema": {"ref": "#/schemas/LoopState"},
        "termination_condition": termination or {
            "predicate": "field_truthy",
            "field": "done",
        },
        "body": body or [{"id": "step", "kind": "call", "symbol": "stub"}],
    }
    if initial_state is not None:
        node["initial_state"] = initial_state
    if early_field is not None:
        node["early_termination_field"] = early_field
    return node


def test_field_truthy_predicate_emits_state_field_check():
    node = _agent_loop_node()
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    # The while loop and termination check
    assert "while iteration < 5:" in src
    assert "if state.done:" in src
    assert "break" in src
    assert "loop = state" in src


def test_field_equals_predicate():
    node = _agent_loop_node(
        termination={"predicate": "field_equals", "field": "status",
                     "value": "complete"}
    )
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "state.status == 'complete'" in src


def test_field_in_predicate():
    node = _agent_loop_node(
        termination={"predicate": "field_in", "field": "status",
                     "values": ["done", "abort"]}
    )
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "state.status in ['done', 'abort']" in src


def test_negate_flips_predicate():
    node = _agent_loop_node(
        termination={"predicate": "field_truthy", "field": "done", "negate": True}
    )
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "if not state.done:" in src


def test_max_iterations_drives_loop_bound():
    node = _agent_loop_node(max_iterations=42)
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "iteration < 42" in src
    assert "exhausted 42 attempts" in src


def test_max_iterations_zero_rejected():
    node = _agent_loop_node(max_iterations=0)
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    with pytest.raises(RendererError, match="must be > 0"):
        op.render(node, ctx, make_render_body())


def test_invalid_predicate_rejected_in_renderer():
    """Defence in depth: even if validator misses, renderer catches."""
    node = _agent_loop_node(termination={"predicate": "magic", "field": "x"})
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    with pytest.raises(RendererError, match="not in"):
        op.render(node, ctx, make_render_body())


def test_body_must_have_id_on_last_node():
    node = _agent_loop_node(body=[{"kind": "call", "symbol": "stub"}])
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    with pytest.raises(RendererError, match="last body node must have an 'id'"):
        op.render(node, ctx, make_render_body())


def test_initial_state_threads_into_loop():
    node = _agent_loop_node(initial_state={"ref": "seed"})
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "state = seed" in src


def test_state_threaded_through_body_each_iteration():
    """The body's last node id becomes the new state for the next iteration."""
    node = _agent_loop_node(body=[
        {"id": "step1", "kind": "call", "symbol": "a"},
        {"id": "step2", "kind": "call", "symbol": "b"},
    ])
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    # state is bound from the LAST body node (step2), not the first.
    assert "state = step2" in src
    assert "step1 = _call('a')" in src
    assert "step2 = _call('b')" in src


def test_early_termination_field_default_is_terminate_early():
    node = _agent_loop_node()
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "state.terminate_early" in src


def test_early_termination_field_can_be_customised():
    node = _agent_loop_node(early_field="halt_now")
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    assert "state.halt_now" in src
    assert "state.terminate_early" not in src


def test_iteration_counter_increments():
    node = _agent_loop_node()
    ctx = make_test_context()
    bind_context(ctx)
    op = get_operator("agent_loop")
    stmts = op.render(node, ctx, make_render_body())
    src = assert_compiles(stmts)
    # +1 increment per iteration
    assert "iteration += 1" in src
    # Init at 0
    assert "iteration = 0" in src
