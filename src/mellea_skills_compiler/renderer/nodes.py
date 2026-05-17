"""Signature-driven node emission.

This module is the workhorse: argument lowering, symbol resolution against
the introspected surface, call-AST construction, and state-block emission.

Architectural commitment (plan §5.4): NO per-symbol logic. Every behaviour
here is driven by the descriptor shape and by the surface JSON's signature
data. If you find yourself reaching for `if symbol == "X":`, stop and reach
for a signature/feature flag in the surface or descriptor instead.

The one renderer-side convention preserved here is the
`format=Schema` -> `Schema.model_validate_json(<call>.value)` wrap, called
out in Phase 0.2's findings.md as the cleanest way to surface typed model
outputs without forcing every descriptor to spell out the parse step.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.source_map import SourceMap

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


def _err(message: str, *, descriptor_path: str = "", symbol: str = "") -> RendererError:
    """Build a RendererError with optional locational suffixes.

    The repair loop (§7.5) parses these suffixes out of the message; keep
    them stable in shape: ``... (at descriptor path: X) (symbol: Y)``.
    """
    full = message
    if descriptor_path:
        full += f" (at descriptor path: {descriptor_path})"
    if symbol:
        full += f" (symbol: {symbol})"
    err = RendererError(full)
    err.descriptor_path = descriptor_path  # type: ignore[attr-defined]
    err.symbol = symbol  # type: ignore[attr-defined]
    return err


# --- Symbol resolution ----------------------------------------------------


def split_symbol(symbol: str, surface: dict[str, Any]) -> tuple[str, list[str]]:
    """Split `pkg.mod.Class.method` into (module_name, [attr, sub_attr, ...]).

    Strategy: walk back from the longest possible module prefix and stop at
    the first one that exists in surface["modules"]. This handles arbitrary
    nesting depths without per-package hard-coding.
    """
    if not symbol:
        raise _err("empty symbol", symbol=symbol)
    parts = symbol.split(".")
    modules = surface.get("modules", {})
    # Try progressively shorter prefixes; longest first.
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in modules:
            return candidate, parts[i:]
    raise _err(
        f"no module prefix of {symbol!r} resolves in surface",
        symbol=symbol,
    )


def resolve_symbol(symbol: str, surface: dict[str, Any]) -> dict[str, Any]:
    """Resolve `symbol` to its surface entry (a SurfaceSymbol or SurfaceMember dict).

    Raises RendererError if any step fails.
    """
    module_name, attr_chain = split_symbol(symbol, surface)
    module = surface["modules"].get(module_name)
    if not module:
        raise _err(
            f"module {module_name!r} not in surface",
            symbol=symbol,
        )
    if not attr_chain:
        raise _err(f"symbol {symbol!r} has no attribute chain", symbol=symbol)
    head = attr_chain[0]
    sym = module.get("symbols", {}).get(head)
    if sym is None:
        raise _err(
            f"symbol {head!r} not in module {module_name!r}",
            symbol=symbol,
        )
    # Walk into members for further attrs (e.g. Class.method).
    current = sym
    for piece in attr_chain[1:]:
        members = current.get("members", {}) if isinstance(current, dict) else {}
        if piece not in members:
            raise _err(
                f"member {piece!r} not found on {symbol!r}",
                symbol=symbol,
            )
        current = members[piece]
    return current


def symbol_resolves(symbol: str, surface: dict[str, Any]) -> bool:
    try:
        resolve_symbol(symbol, surface)
        return True
    except RendererError:
        return False


# --- Type lowering --------------------------------------------------------


def lower_type(type_str: str) -> ast.expr:
    """Lower a descriptor `type` expression into a Python annotation AST node.

    Handles primitives, `list[T]`, `dict[K, V]`, and bare schema references.
    """
    s = type_str.strip()
    if s in ("str", "int", "float", "bool", "bytes", "None"):
        return ast.Name(id=s)
    if s == "Any":
        return ast.Name(id="Any")
    if s.startswith("list[") and s.endswith("]"):
        return ast.Subscript(
            value=ast.Name(id="list"),
            slice=lower_type(s[len("list[") : -1]),
        )
    if s.startswith("dict[") and s.endswith("]"):
        inside = s[len("dict[") : -1]
        depth = 0
        split = -1
        for i, ch in enumerate(inside):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "," and depth == 0:
                split = i
                break
        if split == -1:
            raise RendererError(f"malformed dict type: {type_str!r}")
        return ast.Subscript(
            value=ast.Name(id="dict"),
            slice=ast.Tuple(
                elts=[lower_type(inside[:split].strip()), lower_type(inside[split + 1 :].strip())],
                ctx=ast.Load(),
            ),
        )
    # Bare name — schema reference, treated as a Python identifier.
    return ast.Name(id=s)


# --- Template lowering ----------------------------------------------------


def lower_template(template: str) -> ast.expr:
    """Convert a `{var.attr}`-style template into a Python f-string AST."""
    parts: list[ast.expr] = []
    i = 0
    n = len(template)
    while i < n:
        if template[i] == "{" and i + 1 < n and template[i + 1] == "{":
            parts.append(ast.Constant(value="{"))
            i += 2
            continue
        if template[i] == "}" and i + 1 < n and template[i + 1] == "}":
            parts.append(ast.Constant(value="}"))
            i += 2
            continue
        if template[i] == "{":
            end = template.find("}", i + 1)
            if end == -1:
                raise RendererError(f"unclosed brace in template: {template!r}")
            expr_src = template[i + 1 : end]
            try:
                value_ast = ast.parse(expr_src, mode="eval").body
            except SyntaxError as e:
                raise RendererError(
                    f"invalid expression inside template: {expr_src!r}"
                ) from e
            parts.append(ast.FormattedValue(value=value_ast, conversion=-1))
            i = end + 1
            continue
        if template[i] == "}":
            raise RendererError(
                f"unmatched closing brace in template: {template!r}"
            )
        # Constant run
        j = i
        while j < n and template[j] not in "{}":
            j += 1
        if j > i:
            parts.append(ast.Constant(value=template[i:j]))
        i = j
    return ast.JoinedStr(values=parts)


# --- Argument lowering (the heart of the renderer) ------------------------


# Argument-kind dispatch table. Each handler returns an ast.expr. Adding a
# new descriptor argument kind = adding a handler here; no symbol-specific
# branching needed because we're dispatching on argument *shape*, not on
# *what we're passing the value to*.

def _lower_value(value: dict, ctx: "RenderContext") -> ast.expr:
    v = value["value"]
    return _python_literal_to_ast(v)


def _python_literal_to_ast(v: Any) -> ast.expr:
    if isinstance(v, list):
        return ast.List(elts=[_python_literal_to_ast(x) for x in v], ctx=ast.Load())
    if isinstance(v, dict):
        return ast.Dict(
            keys=[ast.Constant(value=k) for k in v.keys()],
            values=[_python_literal_to_ast(x) for x in v.values()],
        )
    if isinstance(v, tuple):
        return ast.Tuple(
            elts=[_python_literal_to_ast(x) for x in v],
            ctx=ast.Load(),
        )
    return ast.Constant(value=v)


def _lower_template_arg(value: dict, ctx: "RenderContext") -> ast.expr:
    return lower_template(value["template"])


def _lower_ref(value: dict, ctx: "RenderContext") -> ast.expr:
    ref = value["ref"]
    node: ast.expr = ast.Name(id=ref)
    select = value.get("select")
    if select:
        for piece in select.split("."):
            node = ast.Attribute(value=node, attr=piece)
    return node


def _lower_schema_ref(value: dict, ctx: "RenderContext") -> ast.expr:
    ref = value["schema_ref"]
    name = ref.split("/")[-1]
    ctx.register_schema_use(name)
    return ast.Name(id=name)


def _lower_env(value: dict, ctx: "RenderContext") -> ast.expr:
    ctx.add_os_import()
    return ast.Subscript(
        value=ast.Attribute(value=ast.Name(id="os"), attr="environ"),
        slice=ast.Constant(value=value["env"]),
    )


def _lower_symbol(value: dict, ctx: "RenderContext") -> ast.expr:
    return build_call(
        symbol=value["symbol"],
        bound_to=value.get("bound_to"),
        args=value.get("args", {}),
        ctx=ctx,
    )


_ARG_HANDLERS: dict[str, Any] = {
    "value": _lower_value,
    "template": _lower_template_arg,
    "ref": _lower_ref,
    "schema_ref": _lower_schema_ref,
    "env": _lower_env,
    "symbol": _lower_symbol,
}


def lower_arg(value: Any, ctx: "RenderContext") -> ast.expr:
    """Lower a descriptor argument value into an AST expression.

    Recurses into lists; dispatches dicts by their identifying key.
    Raises RendererError on unknown shapes (no silent defaults).
    """
    if isinstance(value, list):
        return ast.List(elts=[lower_arg(v, ctx) for v in value], ctx=ast.Load())
    if not isinstance(value, dict):
        # Bare literal (number, string, bool, None) — accept as a pass-through.
        return ast.Constant(value=value)
    # Dispatch by the first recognised key (descriptor v0.1 makes these mutually exclusive).
    for key, handler in _ARG_HANDLERS.items():
        if key in value:
            return handler(value, ctx)
    raise RendererError(f"unknown argument value shape: {value!r}")


# --- Call construction ----------------------------------------------------


def build_call(
    symbol: str,
    bound_to: dict[str, Any] | None,
    args: dict[str, Any],
    ctx: "RenderContext",
) -> ast.Call:
    """Construct an ast.Call for a symbol invocation.

    Resolves `symbol` against the surface first (raising RendererError on
    miss). Builds the function expression either as a bound-method
    attribute (when `bound_to` is present) or as an imported free reference.
    """
    # Resolve to verify existence and to fail-fast with a useful message.
    resolve_symbol(symbol, ctx.surface)

    func: ast.expr
    if bound_to is not None:
        # Bound-method dispatch: <bound_var>.<method_name>(...)
        # `symbol`'s last component is the method name.
        method_name = symbol.rsplit(".", 1)[-1]
        func = ast.Attribute(value=lower_arg(bound_to, ctx), attr=method_name)
    else:
        module_name, attr_chain = split_symbol(symbol, ctx.surface)
        top = attr_chain[0]
        ctx.add_import(module_name, top)
        node: ast.expr = ast.Name(id=top)
        for piece in attr_chain[1:]:
            node = ast.Attribute(value=node, attr=piece)
        func = node

    kwargs: list[ast.keyword] = []
    for k, v in args.items():
        kwargs.append(ast.keyword(arg=k, value=lower_arg(v, ctx)))
    return ast.Call(func=func, args=[], keywords=kwargs)


# --- Call-node + state emission -------------------------------------------


def _format_arg_schema_name(args: dict[str, Any]) -> str | None:
    """If a call-node has a `format` kwarg with a schema_ref, return the schema name.

    This is the trigger for the `format=Schema -> Schema.model_validate_json(...)`
    renderer convention (Phase 0.2 findings.md item 1). Recognised by argument
    shape only — not by symbol — so it works for any future Mellea call that
    accepts `format=` for typed model output.
    """
    fmt = args.get("format")
    if not isinstance(fmt, dict):
        return None
    if "schema_ref" not in fmt:
        return None
    return fmt["schema_ref"].split("/")[-1]


def emit_call_node(node: dict[str, Any], ctx: "RenderContext", descriptor_path: str = "") -> ast.stmt:
    """Emit a single call descriptor node as an assignment statement.

    Applies the ``format=Schema`` parse wrap when applicable.
    Records the descriptor node id in the source map.

    Descriptor v0.3 additions:
      - ``streaming: true`` → emits the ``with ... .stream() as s: result =
        "".join(s)`` pattern instead of a direct assignment.
      - Skill-level ``async: true`` (read from ``ctx.is_async``) → wraps the
        call expression in ``await`` per plan §11.9.
      - ``source_elements: ["E14", ...]`` → propagated to the source map for
        Step 6's ``mapping_report.md`` builder; renderer otherwise ignores.

    Returns a single ``ast.stmt``. Callers that may need a multi-statement
    emission (e.g. the ``format=Schema`` two-step thunk/parse split for
    downstream-referenced results) should use :func:`emit_call_node_stmts`,
    which delegates here for the single-stmt cases and otherwise returns the
    two statements directly.
    """
    call = build_call(
        symbol=node["symbol"],
        bound_to=node.get("bound_to"),
        args=node.get("args", {}),
        ctx=ctx,
    )
    ctx.register_id(node["id"], "pipeline")
    is_async = getattr(ctx, "is_async", False)
    streaming = bool(node.get("streaming", False))

    schema_name = _format_arg_schema_name(node.get("args", {}))

    if streaming:
        # Streaming form (plan §11.9 / RFC §3.4 G8.2):
        #   with <call>.stream() as _s:
        #       <id> = "".join(_s)
        # Async streaming: assume an async context manager (``async with``)
        # because Mellea's streaming surface returns an async iterator under
        # async skills.
        stream_call = ast.Call(
            func=ast.Attribute(value=call, attr="stream"),
            args=[],
            keywords=[],
        )
        sm_var = f"_{node['id']}_stream"
        join_expr: ast.expr = ast.Call(
            func=ast.Attribute(value=ast.Constant(value=""), attr="join"),
            args=[ast.Name(id=sm_var, ctx=ast.Load())],
            keywords=[],
        )
        # Apply format wrap to the joined string when applicable.
        if schema_name:
            ctx.register_schema_use(schema_name)
            join_expr = ast.Call(
                func=ast.Attribute(value=ast.Name(id=schema_name), attr="model_validate_json"),
                args=[join_expr],
                keywords=[],
            )
        assign_inside = ast.Assign(
            targets=[ast.Name(id=node["id"])],
            value=join_expr,
        )
        with_cls = ast.AsyncWith if is_async else ast.With
        stmt: ast.stmt = with_cls(
            items=[
                ast.withitem(
                    context_expr=stream_call,
                    optional_vars=ast.Name(id=sm_var, ctx=ast.Store()),
                )
            ],
            body=[assign_inside],
        )
        _record_source_elements(ctx, node, stmt, descriptor_path)
        return SourceMap.tag_stmt(stmt, node["id"], descriptor_path)

    # Non-streaming form.
    rhs: ast.expr = call
    if is_async:
        rhs = ast.Await(value=rhs)
    if schema_name:
        # Wrap: <id> = <Schema>.model_validate_json(<call>.value)
        wrapped: ast.expr = ast.Call(
            func=ast.Attribute(value=ast.Name(id=schema_name), attr="model_validate_json"),
            args=[ast.Attribute(value=rhs, attr="value")],
            keywords=[],
        )
        ctx.register_schema_use(schema_name)
        rhs = wrapped
    stmt = ast.Assign(targets=[ast.Name(id=node["id"])], value=rhs)
    _record_source_elements(ctx, node, stmt, descriptor_path)
    return SourceMap.tag_stmt(stmt, node["id"], descriptor_path)


def emit_call_node_stmts(
    node: dict[str, Any],
    ctx: "RenderContext",
    descriptor_path: str = "",
) -> list[ast.stmt]:
    """Emit a call descriptor node, returning one or two statements.

    When the call has ``format={schema_ref: ...}``, is **not streaming**,
    and the node's ``id`` is referenced downstream
    (``ctx.referenced_ids``), emit the safer two-statement form::

        <id>_thunk = session.instruct(..., format=<Schema>)
        <id> = <Schema>.model_validate_json(<id>_thunk.value)

    This makes the binding visible to downstream nodes (and to the return
    statement) the **parsed Pydantic model**, not the raw thunk — closing
    the gap that the ``instruct-result-parse-before-access`` lint catches
    post-render. The lint exempts the
    ``<Schema>.model_validate_json(<id>_thunk.value)`` call when its first
    argument is the attribute access ``<id>_thunk.value`` for a Name bound
    earlier in the same scope from an ``m.instruct`` call carrying a
    ``format=`` kwarg.

    Falls back to a single statement (delegating to :func:`emit_call_node`)
    in every other case — no ``format=Schema``, format is a primitive /
    plain Python type instead of a ``schema_ref``, the result is not
    referenced anywhere downstream, or the call is streaming (which already
    binds the parsed model directly without a thunk in scope).
    """
    args = node.get("args", {}) or {}
    schema_name = _format_arg_schema_name(args)
    streaming = bool(node.get("streaming", False))
    node_id = node.get("id")
    referenced = bool(node_id and node_id in getattr(ctx, "referenced_ids", set()))

    if not (schema_name and referenced) or streaming:
        return [emit_call_node(node, ctx, descriptor_path)]

    # Two-statement form. We rebuild the underlying call here rather than
    # try to peel apart emit_call_node's wrap, because the wrap is built
    # inline (Schema.model_validate_json(<call>.value)) and reusing it
    # would entail unwrapping AST nodes. This stays consistent with
    # emit_call_node by routing through the same build_call + register
    # path (so imports, schema uses, source-map tagging stay aligned).
    call = build_call(
        symbol=node["symbol"],
        bound_to=node.get("bound_to"),
        args=args,
        ctx=ctx,
    )
    ctx.register_id(node_id, "pipeline")  # type: ignore[arg-type]
    ctx.register_schema_use(schema_name)

    is_async = getattr(ctx, "is_async", False)
    rhs: ast.expr = call
    if is_async:
        rhs = ast.Await(value=rhs)

    thunk_name = f"{node_id}_thunk"
    thunk_assign: ast.stmt = ast.Assign(
        targets=[ast.Name(id=thunk_name)],
        value=rhs,
    )
    parse_call = ast.Call(
        func=ast.Attribute(value=ast.Name(id=schema_name), attr="model_validate_json"),
        args=[ast.Attribute(value=ast.Name(id=thunk_name), attr="value")],
        keywords=[],
    )
    parsed_assign: ast.stmt = ast.Assign(
        targets=[ast.Name(id=node_id)],  # type: ignore[arg-type]
        value=parse_call,
    )

    # Tag both stmts in the source map for traceability; the descriptor
    # node maps to a 2-line span, and the tag-anchored first stmt is the
    # repair-loop entry point.
    _record_source_elements(ctx, node, thunk_assign, descriptor_path)
    SourceMap.tag_stmt(thunk_assign, node_id, descriptor_path)  # type: ignore[arg-type]
    SourceMap.tag_stmt(parsed_assign, node_id, descriptor_path)  # type: ignore[arg-type]
    return [thunk_assign, parsed_assign]


def _record_source_elements(
    ctx: "RenderContext", node: dict[str, Any], stmt: ast.stmt, descriptor_path: str
) -> None:
    """Propagate ``source_elements: ["E14", ...]`` into the source map.

    Step 6's ``mapping_report.md`` builder consumes these from the source-map
    entries. The renderer itself ignores them — they're metadata only.
    """
    elems = node.get("source_elements")
    if not elems:
        return
    setattr(stmt, "_mellea_source_elements", list(elems))
    recorder = getattr(ctx, "record_source_elements", None)
    if recorder is None:
        return
    try:
        recorder(node.get("id", ""), list(elems), descriptor_path)
    except Exception:  # noqa: BLE001 — keep render robust to recorder bugs
        pass


def emit_state(state: list[dict[str, Any]], ctx: "RenderContext") -> list[ast.stmt]:
    """Emit module-level state assignments. One per `state[]` entry."""
    stmts: list[ast.stmt] = []
    for idx, entry in enumerate(state):
        call = build_call(
            symbol=entry["symbol"],
            bound_to=entry.get("bound_to"),
            args=entry.get("args", {}),
            ctx=ctx,
        )
        ctx.register_id(entry["id"], "state")
        stmt: ast.stmt = ast.Assign(targets=[ast.Name(id=entry["id"])], value=call)
        SourceMap.tag_stmt(stmt, entry["id"], f"state[{idx}]")
        stmts.append(stmt)
    return stmts
