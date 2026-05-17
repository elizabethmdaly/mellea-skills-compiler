"""``branch`` composition operator (plan §4.5).

Conditional dispatch. The Phase 0.3c zero-ref D″ runs showed the LLM
consistently selects ``branch`` for NDA-review-style skills, so a correct
implementation of this operator is empirically load-bearing.

Required descriptor fields (schema strawman §"Composition node"):
    on:    {"ref": "<id>", "select"?: "<dot.path>"}
    cases: dict[str, list[node]]
            — keys are case values; the special key "_default" is the
              ``else`` branch.

The operator's ``id`` is bound in **every** branch to the last-emitted id of
that branch, so downstream nodes can reference the branch's result via
``{ref: "<branch_id>"}`` regardless of which case fired (plan §4.5: "All
case branches' last-emitted ids are accessible as ``<branch_node_id>``").

Emitted shape (cases: {"a": [...], "b": [...], "_default": [...]})::

    if <on> == "a":
        <body_a...>
        <node.id> = <last_a_id>
    elif <on> == "b":
        <body_b...>
        <node.id> = <last_b_id>
    else:
        <body_default...>
        <node.id> = <last_default_id>

If no ``_default`` is provided we emit an explicit ``raise RendererError`` in
the else block — the descriptor author opted into a closed-world dispatch
and the runtime should fail fast on an unexpected case value (plan §10.x:
descriptor errors must be loud).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Callable

from mellea_skills_compiler.renderer import OperatorRenderer, RendererError
from mellea_skills_compiler.renderer.operators._support import (
    record_source,
    register_id_if_supported,
    require_field,
    resolve_ref,
)

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


_DEFAULT_KEY = "_default"


class BranchRenderer(OperatorRenderer):
    """Render a ``branch`` composition node."""

    operator_name = "branch"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        on = require_field(node, "on", expected_type=dict)
        cases = require_field(node, "cases", expected_type=dict)

        if not cases:
            raise RendererError(f"branch node {node_id!r}: 'cases' must be non-empty")

        # Preserve the descriptor's case ordering so the rendered if/elif chain
        # matches the author's intent. dict iteration is insertion-ordered in
        # Python 3.7+, which is the contract this relies on.
        ordered_cases: list[tuple[str, list[dict]]] = []
        default_body: list[dict] | None = None
        for case_value, case_body in cases.items():
            if not isinstance(case_body, list):
                raise RendererError(
                    f"branch node {node_id!r}: case {case_value!r} body must be a list, "
                    f"got {type(case_body).__name__}"
                )
            if case_value == _DEFAULT_KEY:
                default_body = case_body
                continue
            ordered_cases.append((case_value, case_body))

        if not ordered_cases and default_body is None:
            raise RendererError(
                f"branch node {node_id!r}: at least one non-default case or a _default is required"
            )

        register_id_if_supported(ctx, node_id, kind="branch_result")
        on_expr = resolve_ref(ctx, on, owner_id=node_id, field="on")

        # Build the if/elif/else chain bottom-up so each `orelse` slot can be
        # populated correctly.
        else_body = self._render_case_body(default_body, node_id, render_body) if default_body is not None else self._closed_world_else(node_id, on_expr)

        if_chain: ast.If | None = None
        for case_value, case_body in reversed(ordered_cases):
            test = ast.Compare(
                left=on_expr,
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=case_value)],
            )
            then_body = self._render_case_body(case_body, node_id, render_body)
            new_if = ast.If(test=test, body=then_body, orelse=else_body)
            else_body = [new_if]
            if_chain = new_if

        if if_chain is None:
            # Degenerate: only a default. Emit it as an unconditional block.
            record_source(ctx, node, else_body[0])
            return else_body

        record_source(ctx, node, if_chain)
        # else_body now contains exactly [if_chain]
        return [if_chain]

    @staticmethod
    def _render_case_body(
        case_body: list[dict],
        branch_id: str,
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        # Validate shape BEFORE delegating to render_body, so a missing 'id'
        # surfaces as a RendererError rather than as a KeyError inside the
        # body's own lowering.
        if not case_body:
            raise RendererError(
                f"branch node {branch_id!r}: case body must contain at least one node"
            )
        last_node = case_body[-1]
        if not isinstance(last_node, dict) or "id" not in last_node:
            raise RendererError(
                f"branch node {branch_id!r}: each case's last node must have an 'id' "
                f"so the branch's result variable can be bound"
            )
        last_id = last_node["id"]
        stmts = render_body(case_body)
        bind = ast.Assign(
            targets=[ast.Name(id=branch_id, ctx=ast.Store())],
            value=ast.Name(id=last_id, ctx=ast.Load()),
        )
        return [*stmts, bind]

    @staticmethod
    def _closed_world_else(branch_id: str, on_expr: ast.expr) -> list[ast.stmt]:
        """Else-block when the descriptor has no ``_default``.

        Emits ``raise RendererError(f"unexpected branch value: {<on>!r}")`` —
        we deliberately raise an error tied to the renderer's exception type
        so pipeline runtime failures of this kind are diagnosable.
        """
        raise_stmt = ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                args=[
                    ast.JoinedStr(
                        values=[
                            ast.Constant(value=f"branch {branch_id!r}: unexpected value "),
                            ast.FormattedValue(value=on_expr, conversion=114),  # !r
                        ]
                    )
                ],
                keywords=[],
            ),
            cause=None,
        )
        return [raise_stmt]
