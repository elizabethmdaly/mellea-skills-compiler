"""``retry_with_feedback`` composition operator (plan §4.5).

Bounded retry loop with feedback.

Required descriptor fields (schema strawman §"Composition node"):
    body:          list[node]  — re-executed each attempt
    max_attempts:  int (> 0)
    failure_check: ref-shape dict resolving to a callable / boolean
                   expression that returns truthy when the body's last
                   output **passes** verification.

The operator's ``id`` is bound to the last body node's id on the successful
attempt; downstream nodes reference the retry's result by that ``id``.

If all attempts fail the rendered code raises ``RuntimeError`` — we use the
stdlib type rather than introducing a new exception class because compiled
pipelines should not depend on the renderer's exception hierarchy at
runtime.

Emitted shape::

    for _attempt in range(<max_attempts>):
        <body...>
        if <failure_check>:
            break
    else:
        raise RuntimeError(
            f"retry_with_feedback {<node.id>!r} exhausted <max_attempts> attempts"
        )
    <node.id> = <last_body_id>

Note the ``for ... else`` construct: Python's ``else`` on a for-loop only
executes when the loop completes without ``break``, which is exactly the
"all attempts failed" condition we want to surface.
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


class RetryRenderer(OperatorRenderer):
    """Render a ``retry_with_feedback`` composition node."""

    operator_name = "retry_with_feedback"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        body = require_field(node, "body", expected_type=list)
        max_attempts = require_field(node, "max_attempts", expected_type=int)
        failure_check = require_field(node, "failure_check", expected_type=dict)

        if isinstance(max_attempts, bool):
            # bool is a subclass of int — reject explicitly.
            raise RendererError(
                f"retry_with_feedback node {node_id!r}: 'max_attempts' must be int, got bool"
            )
        if max_attempts <= 0:
            raise RendererError(
                f"retry_with_feedback node {node_id!r}: 'max_attempts' must be > 0, got {max_attempts}"
            )
        if not body:
            raise RendererError(
                f"retry_with_feedback node {node_id!r}: 'body' must contain at least one node"
            )
        last_node = body[-1]
        if not isinstance(last_node, dict) or "id" not in last_node:
            raise RendererError(
                f"retry_with_feedback node {node_id!r}: last body node must have an 'id' "
                f"so the retry's result variable can be bound"
            )
        last_id = last_node["id"]

        register_id_if_supported(ctx, node_id, kind="retry_result")
        register_id_if_supported(ctx, "_attempt", kind="loop_var")

        body_stmts = render_body(body)
        check_expr = resolve_ref(ctx, failure_check, owner_id=node_id, field="failure_check")

        # if <check>: break
        break_stmt = ast.If(test=check_expr, body=[ast.Break()], orelse=[])

        # for _attempt in range(max_attempts): <body...>; <break_stmt>
        # else: raise RuntimeError(...)
        for_stmt = ast.For(
            target=ast.Name(id="_attempt", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=max_attempts)],
                keywords=[],
            ),
            body=[*body_stmts, break_stmt],
            orelse=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                        args=[
                            ast.Constant(
                                value=(
                                    f"retry_with_feedback {node_id!r} exhausted "
                                    f"{max_attempts} attempts"
                                )
                            )
                        ],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
        )

        # <node.id> = <last_body_id>   (after the loop, on the success path)
        bind_stmt = ast.Assign(
            targets=[ast.Name(id=node_id, ctx=ast.Store())],
            value=ast.Name(id=last_id, ctx=ast.Load()),
        )

        record_source(ctx, node, for_stmt)
        return [for_stmt, bind_stmt]
