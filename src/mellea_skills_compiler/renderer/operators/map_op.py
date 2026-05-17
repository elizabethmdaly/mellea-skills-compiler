"""``map`` composition operator (plan §4.5).

Iterates a body of nodes over a list reference and collects per-iteration
results. Re-implementation of the Phase 0.2 spike's ``emit_map``
(``scripts/render_descriptor_spike.py``), promoted to the plug-in framework
and made strict about required fields.

Required descriptor fields (schema strawman §"Composition node"):
    over:    {"ref": "<id>", "select"?: "<dot.path>"}
    body:    list[node]   — executed once per item
    collect: "list" | "dict_by_key"

Optional fields:
    item_id:   loop-variable name (default ``"_item"``)
    key_field: required when ``collect == "dict_by_key"``; selects the key
               attribute on the last body node's result.

The operator's own ``id`` becomes the accumulator variable name in the
rendered code; downstream nodes reference the map's result by that ``id``.

Emitted shape (``collect: list``)::

    <node.id> = []
    for <item_id> in <over>:
        <body...>
        <node.id>.append(<last_body_id>)

Emitted shape (``collect: dict_by_key``)::

    <node.id> = {}
    for <item_id> in <over>:
        <body...>
        <node.id>[<last_body_id>.<key_field>] = <last_body_id>
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


_VALID_COLLECT = ("list", "dict_by_key")


class MapRenderer(OperatorRenderer):
    """Render a ``map`` composition node."""

    operator_name = "map"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        over = require_field(node, "over", expected_type=dict)
        body = require_field(node, "body", expected_type=list)
        collect = require_field(node, "collect", expected_type=str)

        if collect not in _VALID_COLLECT:
            raise RendererError(
                f"map node {node_id!r}: 'collect' must be one of {_VALID_COLLECT}, "
                f"got {collect!r}"
            )
        if not body:
            raise RendererError(f"map node {node_id!r}: 'body' must contain at least one node")

        item_id = node.get("item_id", "_item")
        if not isinstance(item_id, str) or not item_id.isidentifier():
            raise RendererError(
                f"map node {node_id!r}: 'item_id' must be a valid Python identifier, got {item_id!r}"
            )

        register_id_if_supported(ctx, item_id, kind="loop_var")
        register_id_if_supported(ctx, node_id, kind="accumulator")

        # 1. Accumulator initialisation
        if collect == "dict_by_key":
            accumulator_init: ast.expr = ast.Dict(keys=[], values=[])
        else:
            accumulator_init = ast.List(elts=[], ctx=ast.Load())
        init_stmt = ast.Assign(
            targets=[ast.Name(id=node_id, ctx=ast.Store())],
            value=accumulator_init,
        )

        # 2. Per-item body — validate the last node's shape BEFORE delegating
        # to render_body, so a missing 'id' surfaces as a RendererError rather
        # than as a downstream KeyError inside the body's own lowering.
        last_body_node = body[-1]
        if not isinstance(last_body_node, dict) or "id" not in last_body_node:
            raise RendererError(
                f"map node {node_id!r}: last body node must have an 'id' so the accumulator can capture it"
            )
        last_id = last_body_node["id"]
        body_stmts = render_body(body)

        if collect == "dict_by_key":
            key_field = node.get("key_field")
            if not isinstance(key_field, str) or not key_field:
                raise RendererError(
                    f"map node {node_id!r}: 'key_field' is required when collect == 'dict_by_key'"
                )
            append_stmt: ast.stmt = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=node_id, ctx=ast.Load()),
                        slice=ast.Attribute(
                            value=ast.Name(id=last_id, ctx=ast.Load()),
                            attr=key_field,
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Name(id=last_id, ctx=ast.Load()),
            )
        else:
            append_stmt = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=node_id, ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=last_id, ctx=ast.Load())],
                    keywords=[],
                )
            )

        # 3. for-loop
        iter_expr = resolve_ref(ctx, over, owner_id=node_id, field="over")
        for_stmt = ast.For(
            target=ast.Name(id=item_id, ctx=ast.Store()),
            iter=iter_expr,
            body=[*body_stmts, append_stmt],
            orelse=[],
        )

        # The for-loop is the operator's anchor for the source map.
        record_source(ctx, node, for_stmt)
        return [init_stmt, for_stmt]
