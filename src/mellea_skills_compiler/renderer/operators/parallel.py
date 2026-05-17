"""``parallel`` composition operator (plan §4.5).

Concurrent execution of multiple branches.

Required descriptor fields (schema strawman §"Composition node"):
    branches: list[list[node]]   — array of branch-arrays; each inner list is
                                   one parallel branch's body.

Optional:
    collect: "dict_by_branch_id" (default) | "list"

Implementation note — we use :class:`concurrent.futures.ThreadPoolExecutor`
rather than asyncio because the synchronous Mellea backend calls block on
network I/O; threads release the GIL during socket reads and give us real
parallelism without forcing the entire descriptor surface to become async.
When/if Mellea grows a native async surface (plan §12 Q6), this operator is
the single point we revisit.

Emitted shape (collect: dict_by_branch_id, the default)::

    def _<node.id>_branch_<i>():
        <branch_i_body...>
        return <last_branch_i_id>

    with concurrent.futures.ThreadPoolExecutor(max_workers=<n_branches>) as _pool:
        _futures = {
            "<last_branch_0_id>": _pool.submit(_<node.id>_branch_0),
            "<last_branch_1_id>": _pool.submit(_<node.id>_branch_1),
            ...
        }
        <node.id> = {k: f.result() for k, f in _futures.items()}

Emitted shape (collect: list)::

    ... same as above, then:
        <node.id> = [f.result() for f in _futures]    # ordered by submission

The per-branch nested-function pattern is necessary because the body may
contain its own local variable assignments (the body's last-node ``id``); we
need a fresh scope to avoid the threads mutating the outer pipeline's
namespace.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Callable

from mellea_skills_compiler.renderer import OperatorRenderer, RendererError
from mellea_skills_compiler.renderer.operators._support import (
    add_import_if_supported,
    record_source,
    register_id_if_supported,
    require_field,
)

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


_VALID_COLLECT = ("dict_by_branch_id", "list")


class ParallelRenderer(OperatorRenderer):
    """Render a ``parallel`` composition node."""

    operator_name = "parallel"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        node_id = require_field(node, "id", expected_type=str)
        branches = require_field(node, "branches", expected_type=list)
        if not branches:
            raise RendererError(f"parallel node {node_id!r}: 'branches' must be non-empty")

        collect = node.get("collect", "dict_by_branch_id")
        if collect not in _VALID_COLLECT:
            raise RendererError(
                f"parallel node {node_id!r}: 'collect' must be one of {_VALID_COLLECT}, got {collect!r}"
            )

        add_import_if_supported(ctx, "concurrent.futures", "ThreadPoolExecutor")
        register_id_if_supported(ctx, node_id, kind="parallel_result")

        # 1. Build a nested function per branch.
        func_defs: list[ast.stmt] = []
        branch_descriptors: list[tuple[str, str]] = []  # (func_name, last_body_id)
        for i, branch in enumerate(branches):
            if not isinstance(branch, list):
                raise RendererError(
                    f"parallel node {node_id!r}: branches[{i}] must be a list, got {type(branch).__name__}"
                )
            if not branch:
                raise RendererError(
                    f"parallel node {node_id!r}: branches[{i}] must contain at least one node"
                )
            last = branch[-1]
            if not isinstance(last, dict) or "id" not in last:
                raise RendererError(
                    f"parallel node {node_id!r}: branches[{i}]'s last node must have an 'id'"
                )
            last_id = last["id"]
            func_name = f"_{node_id}_branch_{i}"
            register_id_if_supported(ctx, func_name, kind="branch_fn")

            body_stmts = render_body(branch)
            body_stmts.append(ast.Return(value=ast.Name(id=last_id, ctx=ast.Load())))
            func_defs.append(
                ast.FunctionDef(
                    name=func_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=body_stmts,
                    decorator_list=[],
                    returns=None,
                    type_params=[],
                )
            )
            branch_descriptors.append((func_name, last_id))

        # 2. with ThreadPoolExecutor(...) as _pool:
        pool_var = f"_{node_id}_pool"
        futures_var = f"_{node_id}_futures"

        executor_ctor = ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="concurrent", ctx=ast.Load()),
                    attr="futures",
                    ctx=ast.Load(),
                ),
                attr="ThreadPoolExecutor",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[ast.keyword(arg="max_workers", value=ast.Constant(value=len(branches)))],
        )

        # _futures = {key: _pool.submit(fn), ...}
        if collect == "dict_by_branch_id":
            futures_init = ast.Assign(
                targets=[ast.Name(id=futures_var, ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(value=last_id) for _, last_id in branch_descriptors],
                    values=[
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=pool_var, ctx=ast.Load()),
                                attr="submit",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id=func_name, ctx=ast.Load())],
                            keywords=[],
                        )
                        for func_name, _ in branch_descriptors
                    ],
                ),
            )
            # <node.id> = {k: f.result() for k, f in _futures.items()}
            collect_stmt = ast.Assign(
                targets=[ast.Name(id=node_id, ctx=ast.Store())],
                value=ast.DictComp(
                    key=ast.Name(id="_k", ctx=ast.Load()),
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="_f", ctx=ast.Load()),
                            attr="result",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Tuple(
                                elts=[
                                    ast.Name(id="_k", ctx=ast.Store()),
                                    ast.Name(id="_f", ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            ),
                            iter=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id=futures_var, ctx=ast.Load()),
                                    attr="items",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            ),
                            ifs=[],
                            is_async=0,
                        )
                    ],
                ),
            )
        else:  # collect == "list"
            futures_init = ast.Assign(
                targets=[ast.Name(id=futures_var, ctx=ast.Store())],
                value=ast.List(
                    elts=[
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=pool_var, ctx=ast.Load()),
                                attr="submit",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id=func_name, ctx=ast.Load())],
                            keywords=[],
                        )
                        for func_name, _ in branch_descriptors
                    ],
                    ctx=ast.Load(),
                ),
            )
            collect_stmt = ast.Assign(
                targets=[ast.Name(id=node_id, ctx=ast.Store())],
                value=ast.ListComp(
                    elt=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="_f", ctx=ast.Load()),
                            attr="result",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id="_f", ctx=ast.Store()),
                            iter=ast.Name(id=futures_var, ctx=ast.Load()),
                            ifs=[],
                            is_async=0,
                        )
                    ],
                ),
            )

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=executor_ctor,
                    optional_vars=ast.Name(id=pool_var, ctx=ast.Store()),
                )
            ],
            body=[futures_init, collect_stmt],
        )

        record_source(ctx, node, with_stmt)
        return [*func_defs, with_stmt]
