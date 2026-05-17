"""Shared helpers for composition operator implementations.

Kept module-private (leading underscore) — not part of the operator interface.
P2.A's renderer core does not depend on these; they exist solely to keep the
five operator modules small and consistent.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from mellea_skills_compiler.renderer import RendererError

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


def record_source(ctx: "RenderContext", node: dict, stmt: ast.stmt) -> None:
    """Bind a source-map entry to the operator's anchor statement.

    Source-line numbers are not known at AST-build time (the unparser assigns
    them later), so we bind by ``stmt`` identity where possible. We tolerate
    either the documented ``record(line=..., node_id=...)`` signature or a
    ``record(stmt=..., node_id=...)`` variant — the precise shape is fixed by
    P2.A and may have refined by integration time. A missing or stub
    :attr:`source_map` is treated as a no-op so operator unit tests can run
    without P2.A's core present.
    """
    sm = getattr(ctx, "source_map", None)
    if sm is None:
        return
    record = getattr(sm, "record", None)
    if record is None:
        return
    try:
        record(stmt=stmt, node_id=node["id"])
        return
    except TypeError:
        pass
    record(line=getattr(stmt, "lineno", 0), node_id=node["id"])


def resolve_ref(ctx: "RenderContext", ref_dict: dict, *, owner_id: str, field: str) -> ast.expr:
    """Resolve a ``{ref, select}`` / ``{schema_ref}`` / etc. dict via the context.

    Operators must not implement their own ref resolution (see plan §5: the
    renderer's ref-resolution rules are owned by the core). If P2.A's
    :class:`RenderContext` does not expose ``resolve_ref`` yet (early
    integration), fall back to a minimal ``ref`` → ``Name`` lowering so unit
    tests can exercise operator shape without the full core. The fallback is
    documented in the integration handoff; operator behaviour in production
    is determined by P2.A's :meth:`resolve_ref`.
    """
    resolver = getattr(ctx, "resolve_ref", None)
    if resolver is not None:
        return resolver(ref_dict)

    # Minimal in-tree fallback so operator tests do not require P2.A's core.
    if not isinstance(ref_dict, dict):
        raise RendererError(
            f"{owner_id!r}: {field!r} must be a ref-shaped dict, got {type(ref_dict).__name__}"
        )
    if "ref" in ref_dict:
        expr: ast.expr = ast.Name(id=ref_dict["ref"], ctx=ast.Load())
        select = ref_dict.get("select")
        if select:
            for piece in str(select).split("."):
                expr = ast.Attribute(value=expr, attr=piece, ctx=ast.Load())
        return expr
    if "schema_ref" in ref_dict:
        return ast.Name(id=ref_dict["schema_ref"].split("/")[-1], ctx=ast.Load())
    if "value" in ref_dict:
        return ast.Constant(value=ref_dict["value"])
    raise RendererError(
        f"{owner_id!r}: {field!r} has unsupported ref shape: keys={sorted(ref_dict)}"
    )


def require_field(node: dict, field: str, *, expected_type: type | tuple[type, ...] | None = None) -> object:
    """Validate that ``node`` has ``field`` (and optionally its type).

    Centralised so every operator produces a consistent :class:`RendererError`
    message format for downstream tooling.
    """
    if field not in node:
        raise RendererError(
            f"{node.get('operator', '?')} node {node.get('id')!r} missing required field {field!r}"
        )
    value = node[field]
    if expected_type is not None and not isinstance(value, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " | ".join(t.__name__ for t in expected_type)
        )
        raise RendererError(
            f"{node.get('operator', '?')} node {node.get('id')!r}: "
            f"{field!r} must be {type_name}, got {type(value).__name__}"
        )
    return value


def register_id_if_supported(ctx: "RenderContext", identifier: str, kind: str) -> None:
    """Best-effort wrapper around ``ctx.register_id``.

    Silently no-ops if the context doesn't expose ``register_id`` — keeps
    operator unit tests independent of P2.A's core implementation.
    """
    register = getattr(ctx, "register_id", None)
    if register is None:
        return
    register(identifier, kind)


def add_import_if_supported(ctx: "RenderContext", module: str, name: str) -> None:
    """Best-effort wrapper around ``ctx.add_import``."""
    add = getattr(ctx, "add_import", None)
    if add is None:
        return
    add(module, name)
