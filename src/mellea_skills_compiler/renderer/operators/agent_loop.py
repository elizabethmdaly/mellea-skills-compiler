"""``agent_loop`` composition operator (descriptor v0.3 RFC §3.3 / plan §11.2).

A bounded iterative agent: a body of pipeline nodes is re-executed each
iteration with a typed ``state`` instance threaded through. A closed-vocab
termination predicate breaks the loop early; ``max_iterations`` caps it.

Required descriptor fields:
    max_iterations: int (>0)
    state_schema:   {"ref": "#/schemas/<Name>"}
    termination_condition: TerminationPredicate (one of field_truthy /
                                                 field_equals / field_in)
    body:           list[node] — re-executed each iteration

Optional:
    initial_state:         {ref, value} — first-iteration state (else None)
    tools_ref:             list[str] — dependency ids of kind=tool (renderer
                                       ignores; validator enforces resolution)
    early_termination_field: str (default "terminate_early")

Emitted shape (synchronous)::

    state = <initial_state_expr_or_None>
    iteration = 0
    while iteration < <max_iterations>:
        <body...>
        state = <last_body_id>
        iteration += 1
        if state.terminate_early:
            break
        if <termination_check>:
            break
    else:
        raise RuntimeError("agent_loop <id> exhausted <max_iterations> attempts")
    <node.id> = state

The ``while ... else`` is intentional: Python only runs the ``else`` when
the loop exits without ``break``, which is exactly the "exhausted" condition
we want to surface.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Callable

from mellea_skills_compiler.renderer import OperatorRenderer, RendererError
from mellea_skills_compiler.renderer.operators._support import (
    record_source,
    register_id_if_supported,
    require_field,
    resolve_ref,
)

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


_VALID_PREDICATES = ("field_truthy", "field_equals", "field_in")


class AgentLoopRenderer(OperatorRenderer):
    """Render an ``agent_loop`` composition node."""

    operator_name = "agent_loop"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        max_iter = require_field(node, "max_iterations", expected_type=int)
        if isinstance(max_iter, bool):
            raise RendererError(
                f"agent_loop node {node_id!r}: 'max_iterations' must be int, got bool"
            )
        if max_iter <= 0:
            raise RendererError(
                f"agent_loop node {node_id!r}: 'max_iterations' must be > 0, got {max_iter}"
            )
        body = require_field(node, "body", expected_type=list)
        if not body:
            raise RendererError(
                f"agent_loop node {node_id!r}: 'body' must contain at least one node"
            )
        last = body[-1]
        if not isinstance(last, dict) or "id" not in last:
            raise RendererError(
                f"agent_loop node {node_id!r}: last body node must have an 'id' "
                f"so the iteration's state can be threaded"
            )
        last_id = last["id"]

        state_schema = require_field(node, "state_schema", expected_type=dict)
        termination = require_field(node, "termination_condition", expected_type=dict)
        predicate = termination.get("predicate")
        if predicate not in _VALID_PREDICATES:
            # Defence-in-depth — R-SEM-AGENT-LOOP-FIELDS should have caught this.
            raise RendererError(
                f"agent_loop node {node_id!r}: termination_condition.predicate "
                f"{predicate!r} not in {_VALID_PREDICATES}"
            )

        early_field = node.get("early_termination_field", "terminate_early")
        if not isinstance(early_field, str) or not early_field:
            raise RendererError(
                f"agent_loop node {node_id!r}: 'early_termination_field' must be a non-empty string"
            )

        register_id_if_supported(ctx, node_id, kind="agent_loop_result")
        register_id_if_supported(ctx, "state", kind="loop_var")
        register_id_if_supported(ctx, "iteration", kind="loop_var")

        # Schema ref is used for type hinting; ensure it's registered.
        if "ref" in state_schema:
            schema_name = state_schema["ref"].split("/")[-1]
            register = getattr(ctx, "register_schema_use", None)
            if register is not None:
                register(schema_name)

        # 1. Initial state: state = <initial_state> if present, else None.
        initial = node.get("initial_state")
        if initial is None:
            init_expr: ast.expr = ast.Constant(value=None)
        else:
            init_expr = resolve_ref(ctx, initial, owner_id=node_id, field="initial_state")
        init_state_stmt = ast.Assign(
            targets=[ast.Name(id="state", ctx=ast.Store())],
            value=init_expr,
        )

        # 2. iteration = 0
        init_iter_stmt = ast.Assign(
            targets=[ast.Name(id="iteration", ctx=ast.Store())],
            value=ast.Constant(value=0),
        )

        # 3. while iteration < max_iterations: body; state = <last_id>; iteration += 1; checks
        body_stmts = render_body(body)
        thread_state = ast.Assign(
            targets=[ast.Name(id="state", ctx=ast.Store())],
            value=ast.Name(id=last_id, ctx=ast.Load()),
        )
        bump_iter = ast.AugAssign(
            target=ast.Name(id="iteration", ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        # Early-termination check on state.<early_field>
        early_check = ast.If(
            test=_state_getattr_or_index(early_field),
            body=[ast.Break()],
            orelse=[],
        )
        # Closed-vocab predicate check on state
        term_check_expr = _build_predicate_expr(termination)
        term_check = ast.If(test=term_check_expr, body=[ast.Break()], orelse=[])

        while_test = ast.Compare(
            left=ast.Name(id="iteration", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=max_iter)],
        )
        while_body: list[ast.stmt] = [
            *body_stmts,
            thread_state,
            bump_iter,
            early_check,
            term_check,
        ]
        # else clause: only runs if we never broke -> exhausted.
        exhaustion = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="RuntimeError"),
                args=[
                    ast.Constant(
                        value=f"agent_loop {node_id!r} exhausted {max_iter} attempts"
                    )
                ],
                keywords=[],
            ),
            cause=None,
        )
        while_stmt = ast.While(
            test=while_test,
            body=while_body,
            orelse=[exhaustion],
        )

        # 4. <node.id> = state
        bind_stmt = ast.Assign(
            targets=[ast.Name(id=node_id, ctx=ast.Store())],
            value=ast.Name(id="state", ctx=ast.Load()),
        )

        record_source(ctx, node, while_stmt)
        return [init_state_stmt, init_iter_stmt, while_stmt, bind_stmt]


# --- Predicate helpers -----------------------------------------------------


def _state_getattr_or_index(field_name: str) -> ast.expr:
    """Build ``state.<field>`` for predicate evaluation.

    We use attribute access (not subscript) because the descriptor's
    ``state_schema`` is a Pydantic model; field access is the idiomatic
    Python form and Pydantic supports it.
    """
    return ast.Attribute(
        value=ast.Name(id="state", ctx=ast.Load()),
        attr=field_name,
        ctx=ast.Load(),
    )


def _build_predicate_expr(termination: dict[str, Any]) -> ast.expr:
    """Build a Python AST expression that evaluates the termination predicate.

    Supports the closed-vocab predicates (RFC §3.3):
      - ``field_truthy``  → ``bool(state.<field>)`` (with ``negate``)
      - ``field_equals``  → ``state.<field> == <value>``
      - ``field_in``      → ``state.<field> in <values>``
    """
    predicate = termination["predicate"]
    field_name = termination.get("field")
    if not isinstance(field_name, str) or not field_name:
        raise RendererError(
            f"agent_loop termination_condition missing 'field' for predicate {predicate!r}"
        )
    negate = bool(termination.get("negate", False))
    target = _state_getattr_or_index(field_name)

    if predicate == "field_truthy":
        # `if state.<field>: break` (with optional negate)
        expr: ast.expr = target
    elif predicate == "field_equals":
        if "value" not in termination:
            raise RendererError(
                "agent_loop termination_condition.field_equals missing 'value'"
            )
        expr = ast.Compare(
            left=target,
            ops=[ast.Eq()],
            comparators=[_literal_to_ast(termination["value"])],
        )
    elif predicate == "field_in":
        values = termination.get("values")
        if not isinstance(values, list):
            raise RendererError(
                "agent_loop termination_condition.field_in 'values' must be a list"
            )
        expr = ast.Compare(
            left=target,
            ops=[ast.In()],
            comparators=[
                ast.List(
                    elts=[_literal_to_ast(v) for v in values],
                    ctx=ast.Load(),
                )
            ],
        )
    else:
        raise RendererError(
            f"agent_loop unknown predicate {predicate!r} (expected one of {_VALID_PREDICATES})"
        )

    if negate:
        expr = ast.UnaryOp(op=ast.Not(), operand=expr)
    return expr


def _literal_to_ast(value: Any) -> ast.expr:
    """Lower a JSON literal value to an AST constant/collection node."""
    if isinstance(value, list):
        return ast.List(elts=[_literal_to_ast(v) for v in value], ctx=ast.Load())
    if isinstance(value, dict):
        return ast.Dict(
            keys=[ast.Constant(value=k) for k in value.keys()],
            values=[_literal_to_ast(v) for v in value.values()],
        )
    return ast.Constant(value=value)
