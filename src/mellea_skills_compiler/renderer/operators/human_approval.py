"""``human_approval`` composition operator (descriptor v0.3 RFC §3.3, plan §11.2).

Emits a prompt-and-branch block whose default behaviour is to read from
stdin, but which honours a ``_human_approval_callback`` import from a
sibling ``runtime_overrides`` module if present — so test harnesses can
short-circuit the prompt without monkey-patching stdin.

Required descriptor fields:
    prompt:   ArgValue (typically ``{"value": "..."}`` or ``{"template": "..."}``)
    branches: {"approved": [...nodes...], "rejected": [...nodes...]}

Optional:
    default_branch: "approved" | "rejected"  — used when the harness
                                                short-circuits without a
                                                decision.

Emitted shape::

    try:
        from .runtime_overrides import _human_approval_callback as _approval_cb
    except ImportError:
        _approval_cb = None

    if _approval_cb is not None:
        _approved = _approval_cb(<prompt_expr>)
    else:
        print(<prompt_expr>)
        _approved = input("> ").strip().lower() in ("y", "yes")

    if _approved:
        <approved_body>
        <node.id> = <last_approved_id>
    else:
        <rejected_body>
        <node.id> = <last_rejected_id>

Empty branches degrade gracefully: an empty branch body emits ``pass`` and
binds ``<node.id> = None``.
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


class HumanApprovalRenderer(OperatorRenderer):
    """Render a ``human_approval`` composition node."""

    operator_name = "human_approval"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        prompt = require_field(node, "prompt", expected_type=dict)
        branches = require_field(node, "branches", expected_type=dict)
        if "approved" not in branches or "rejected" not in branches:
            raise RendererError(
                f"human_approval node {node_id!r}: 'branches' must define both "
                f"'approved' and 'rejected'"
            )
        approved_body = branches["approved"]
        rejected_body = branches["rejected"]
        if not isinstance(approved_body, list) or not isinstance(rejected_body, list):
            raise RendererError(
                f"human_approval node {node_id!r}: branches must be lists"
            )

        register_id_if_supported(ctx, node_id, kind="human_approval_result")
        register_id_if_supported(ctx, "_approval_cb", kind="callback")
        register_id_if_supported(ctx, "_approved", kind="approval_flag")

        prompt_expr = resolve_ref(ctx, prompt, owner_id=node_id, field="prompt")

        # 1. try/except import of the runtime override callback.
        callback_import = ast.Try(
            body=[
                ast.ImportFrom(
                    module="runtime_overrides",
                    names=[
                        ast.alias(
                            name="_human_approval_callback",
                            asname="_approval_cb",
                        )
                    ],
                    level=1,
                )
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="ImportError"),
                    name=None,
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id="_approval_cb", ctx=ast.Store())],
                            value=ast.Constant(value=None),
                        )
                    ],
                )
            ],
            orelse=[],
            finalbody=[],
        )

        # 2. Decide whether to call the callback or fall back to stdin.
        callback_branch = ast.Assign(
            targets=[ast.Name(id="_approved", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="_approval_cb"),
                args=[prompt_expr],
                keywords=[],
            ),
        )
        stdin_branch = [
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print"),
                    args=[prompt_expr],
                    keywords=[],
                )
            ),
            ast.Assign(
                targets=[ast.Name(id="_approved", ctx=ast.Store())],
                value=ast.Compare(
                    left=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id="input"),
                                        args=[ast.Constant(value="> ")],
                                        keywords=[],
                                    ),
                                    attr="strip",
                                ),
                                args=[],
                                keywords=[],
                            ),
                            attr="lower",
                        ),
                        args=[],
                        keywords=[],
                    ),
                    ops=[ast.In()],
                    comparators=[
                        ast.Tuple(
                            elts=[
                                ast.Constant(value="y"),
                                ast.Constant(value="yes"),
                            ],
                            ctx=ast.Load(),
                        )
                    ],
                ),
            ),
        ]
        decide_stmt = ast.If(
            test=ast.Compare(
                left=ast.Name(id="_approval_cb", ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[callback_branch],
            orelse=stdin_branch,
        )

        # 3. The approved/rejected branch dispatch.
        approved_stmts = self._render_branch(approved_body, render_body, node_id)
        rejected_stmts = self._render_branch(rejected_body, render_body, node_id)
        dispatch_stmt = ast.If(
            test=ast.Name(id="_approved", ctx=ast.Load()),
            body=approved_stmts,
            orelse=rejected_stmts,
        )

        record_source(ctx, node, decide_stmt)
        return [callback_import, decide_stmt, dispatch_stmt]

    @staticmethod
    def _render_branch(
        body: list[dict],
        render_body: Callable[[list[dict]], list[ast.stmt]],
        node_id: str,
    ) -> list[ast.stmt]:
        """Render a branch body + bind ``<node_id>`` to its last node's id.

        Empty branches emit ``<node_id> = None`` so the dispatch is symmetric.
        """
        if not body:
            return [
                ast.Assign(
                    targets=[ast.Name(id=node_id, ctx=ast.Store())],
                    value=ast.Constant(value=None),
                )
            ]
        last = body[-1]
        if not isinstance(last, dict) or "id" not in last:
            raise RendererError(
                f"human_approval node {node_id!r}: each branch's last node must have an 'id'"
            )
        stmts = render_body(body)
        bind = ast.Assign(
            targets=[ast.Name(id=node_id, ctx=ast.Store())],
            value=ast.Name(id=last["id"], ctx=ast.Load()),
        )
        return [*stmts, bind]
