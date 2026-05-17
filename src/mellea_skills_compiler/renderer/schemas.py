"""Schema renderer — descriptor ``schemas`` block → ``schemas.py`` source.

This module owns the second of P2.C's supporting-artefact emitters (plan §5.2,
table row ``schemas.py``). It consumes a descriptor's ``schemas`` block (a
``dict[str, dict]`` keyed by Python class name) and emits a complete Python
source string containing one ``Enum`` subclass per ``kind: "enum"`` entry and
one ``BaseModel`` subclass per ``kind: "model"`` entry.

Design decisions worth knowing about:

- AST + ``ast.unparse``, never string templating. Operator/renderer P2.A took
  the same call; this module mirrors it so downstream Black/Ruff formatters
  see syntactically clean input.
- Enums are emitted **before** models so forward-referenced enum types resolve
  without ``ForwardRef`` gymnastics. (``from __future__ import annotations`` is
  also emitted, so this is belt + braces.)
- Collection fields (``list[T]`` / ``dict[K, V]``) default to
  ``Field(default_factory=...)`` — never ``= []`` or ``= {}``. The legacy
  compiler hit the mutable-default footgun more than once.
- Optional fields render as ``T | None = None`` (PEP 604), matching the
  existing emitted schemas in ``examples/sentry-find-bugs/.../schemas.py``.
- Pure: same descriptor in → byte-identical Python source out. No
  ``datetime.now()``, no set iteration, no environment lookups.

The supported v0.1 field-type vocabulary (per ``descriptor-schema-v0.md``):

  ============= ================================== =================================
  Descriptor    Annotation                         Default
  ============= ================================== =================================
  ``str``       ``str``                            none
  ``int``       ``int``                            none
  ``float``     ``float``                          none
  ``bool``      ``bool``                           none
  ``list[T]``   ``list[T]``                        ``Field(default_factory=list)``
  ``dict[K,V]`` ``dict[K, V]``                     ``Field(default_factory=dict)``
  ``enum_ref``  bare enum class name (from ``ref``) none
  ``model_ref`` bare model class name (from ``ref``) none
  ``optional``  ``<above> | None``                 ``None``
  ============= ================================== =================================
"""

from __future__ import annotations

import ast
from typing import Any

from mellea_skills_compiler.renderer import RendererError

__all__ = ["render_schemas"]


# --- Identifier helpers ------------------------------------------------------


def _safe_member_ident(value: str) -> str:
    """Lower a free-form enum member value to a SCREAMING_SNAKE_CASE identifier.

    The Pydantic convention for ``str``-backed Enums is SCREAMING_SNAKE_CASE
    for the Python identifier with the descriptor-supplied string as the
    runtime value::

        class Severity(str, Enum):
            CRITICAL = "Critical"

    Strategy: replace any non-alphanumeric / non-underscore run with ``_``,
    upper-case the result, then prefix with ``_`` if it starts with a digit.
    The runtime *value* is never altered.

    Raises:
        RendererError: if ``value`` is empty (would produce no identifier).
    """
    if not value:
        raise RendererError("enum member value cannot be empty")
    chars: list[str] = []
    for ch in value:
        chars.append(ch if (ch.isalnum() or ch == "_") else "_")
    out = "".join(chars).upper()
    if out[0].isdigit():
        out = "_" + out
    return out


# --- Type lowering -----------------------------------------------------------


def _split_top_level_comma(inside: str) -> tuple[str, str]:
    """Split a ``K, V`` payload at the top-level comma, respecting brackets."""
    depth = 0
    for i, ch in enumerate(inside):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            return inside[:i].strip(), inside[i + 1 :].strip()
    raise RendererError(f"malformed parameterised type: missing comma in {inside!r}")


def _lower_type_expr(
    type_str: str, *, known_schema_names: set[str] | None = None
) -> ast.expr:
    """Lower a v0.1 type expression string into a Python annotation AST node.

    Accepts ``str`` / ``int`` / ``float`` / ``bool``, parameterised
    ``list[...]`` / ``dict[..., ...]``, ``Optional[T]`` (v0.3 RFC §3.5
    permissiveness), and bare schema-name references (treated as
    ``ast.Name`` so the caller can resolve to a model/enum class).

    When ``known_schema_names`` is supplied, bare identifiers that do not
    appear in that set are rejected with :class:`RendererError`. This is the
    chokepoint pair for the validator's ``R-SEM-FIELD-TYPE-KNOWN`` rule —
    descriptors should already be rejected at validation time, but the
    renderer enforces the invariant for callers that bypass the validator.
    """
    s = type_str.strip()
    if not s:
        raise RendererError("type expression cannot be empty")
    if s in ("str", "int", "float", "bool"):
        return ast.Name(id=s)
    if s == "Any":
        return ast.Name(id="Any")
    if s.startswith("Optional[") and s.endswith("]"):
        # v0.3: Optional[T] → T | None
        inner = s[len("Optional[") : -1]
        return ast.BinOp(
            left=_lower_type_expr(inner, known_schema_names=known_schema_names),
            op=ast.BitOr(),
            right=ast.Constant(value=None),
        )
    if s.startswith("list[") and s.endswith("]"):
        inner = s[len("list[") : -1]
        return ast.Subscript(
            value=ast.Name(id="list"),
            slice=_lower_type_expr(inner, known_schema_names=known_schema_names),
            ctx=ast.Load(),
        )
    if s.startswith("dict[") and s.endswith("]"):
        k, v = _split_top_level_comma(s[len("dict[") : -1])
        return ast.Subscript(
            value=ast.Name(id="dict"),
            slice=ast.Tuple(
                elts=[
                    _lower_type_expr(k, known_schema_names=known_schema_names),
                    _lower_type_expr(v, known_schema_names=known_schema_names),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
    # Bare identifier — caller-resolved schema reference. Validate it's a
    # legal Python identifier so we fail fast on malformed descriptors.
    if not s.isidentifier():
        raise RendererError(f"unrecognised type expression: {type_str!r}")
    if known_schema_names is not None and s not in known_schema_names:
        raise RendererError(
            f"unrecognised type expression: {type_str!r} "
            f"(not in the known type vocabulary and not a declared schema name)"
        )
    return ast.Name(id=s)


def _is_collection_type(type_str: str) -> tuple[bool, str | None]:
    """Return (is_collection, factory_name) — ``"list"``/``"dict"`` or None."""
    s = type_str.strip()
    if s.startswith("list[") and s.endswith("]"):
        return True, "list"
    if s.startswith("dict[") and s.endswith("]"):
        return True, "dict"
    return False, None


def _lower_field_annotation(
    field: dict[str, Any], *, known_schema_names: set[str] | None = None
) -> ast.expr:
    """Lower a descriptor field definition into an annotation AST node.

    Handles ``enum_ref`` / ``model_ref`` (extract the bare class name from the
    ``ref`` JSON-pointer) and primitives/collections (delegated to
    :func:`_lower_type_expr`). Wraps in ``T | None`` if ``optional: true``.
    """
    kind = field.get("type")
    if kind is None:
        raise RendererError(f"field missing required 'type': {field!r}")

    node: ast.expr
    if kind == "enum_ref":
        ref = field.get("ref")
        if not ref:
            raise RendererError(f"enum_ref field missing 'ref': {field!r}")
        node = ast.Name(id=ref.split("/")[-1])
    elif kind == "model_ref":
        ref = field.get("ref")
        if not ref:
            raise RendererError(f"model_ref field missing 'ref': {field!r}")
        node = ast.Name(id=ref.split("/")[-1])
    elif isinstance(kind, str):
        node = _lower_type_expr(kind, known_schema_names=known_schema_names)
    else:
        raise RendererError(f"field 'type' must be a string, got {type(kind).__name__}")

    if field.get("optional"):
        node = ast.BinOp(left=node, op=ast.BitOr(), right=ast.Constant(value=None))
    return node


def _lower_field_default(field: dict[str, Any]) -> ast.expr | None:
    """Compute the default-value expression for a descriptor field, if any.

    Rules:
    - v0.3: ``description`` on a field → ``Field(description=...)`` (with
      ``default=None`` if optional + ``default_factory=...`` if collection).
    - ``optional: true`` → ``None``.
    - Collection types → ``Field(default_factory=list|dict)``.
    - Everything else → no default (required field).

    Optional + collection is unusual but legal; we treat ``optional`` as
    winning (default ``None``) since that's what the descriptor explicitly
    asks for.
    """
    description = field.get("description")
    optional = bool(field.get("optional"))
    kind = field.get("type")
    is_coll, factory = (False, None)
    if isinstance(kind, str):
        is_coll, factory = _is_collection_type(kind)

    has_description = isinstance(description, str) and description != ""

    if has_description:
        # Compose a Field(...) with whichever of {default, default_factory,
        # description} apply.
        kwargs: list[ast.keyword] = []
        if optional:
            kwargs.append(ast.keyword(arg="default", value=ast.Constant(value=None)))
        elif is_coll:
            kwargs.append(
                ast.keyword(arg="default_factory", value=ast.Name(id=factory))  # type: ignore[arg-type]
            )
        kwargs.append(ast.keyword(arg="description", value=ast.Constant(value=description)))
        return ast.Call(func=ast.Name(id="Field"), args=[], keywords=kwargs)

    if optional:
        return ast.Constant(value=None)
    if is_coll:
        return ast.Call(
            func=ast.Name(id="Field"),
            args=[],
            keywords=[
                ast.keyword(
                    arg="default_factory",
                    value=ast.Name(id=factory),  # type: ignore[arg-type]
                )
            ],
        )
    return None


# --- Class emission ----------------------------------------------------------


def _emit_enum(name: str, members: list[str]) -> ast.ClassDef:
    """Emit ``class <name>(str, Enum):`` with one ``MEMBER = "value"`` per entry.

    Member values are passed through verbatim; member identifiers are derived
    from the value by :func:`_safe_member_ident`. Distinct members that
    collide on the derived identifier are flagged as descriptor errors.
    """
    if not members:
        raise RendererError(f"enum {name!r} has no members")
    body: list[ast.stmt] = []
    seen_idents: set[str] = set()
    for value in members:
        ident = _safe_member_ident(value)
        if ident in seen_idents:
            raise RendererError(
                f"enum {name!r}: members collide on Python identifier {ident!r}"
            )
        seen_idents.add(ident)
        body.append(
            ast.Assign(
                targets=[ast.Name(id=ident)],
                value=ast.Constant(value=value),
            )
        )
    return ast.ClassDef(
        name=name,
        bases=[ast.Name(id="str"), ast.Name(id="Enum")],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )


def _emit_model(
    name: str,
    fields: dict[str, Any],
    *,
    known_schema_names: set[str] | None = None,
) -> ast.ClassDef:
    """Emit ``class <name>(BaseModel):`` with one annotated assignment per field.

    Field iteration order is the input ``fields`` dict's insertion order —
    Python dicts have preserved that since 3.7 and descriptor JSON parses
    insertion-ordered, so output is deterministic by construction.
    """
    body: list[ast.stmt] = []
    for fname, fdef in fields.items():
        if not isinstance(fdef, dict):
            raise RendererError(
                f"model {name!r} field {fname!r}: definition must be a dict, "
                f"got {type(fdef).__name__}"
            )
        annotation = _lower_field_annotation(
            fdef, known_schema_names=known_schema_names
        )
        default = _lower_field_default(fdef)
        body.append(
            ast.AnnAssign(
                target=ast.Name(id=fname),
                annotation=annotation,
                value=default,
                simple=1,
            )
        )
    if not body:
        # A model with no fields is legal but Python requires a non-empty
        # class body; emit `pass`.
        body.append(ast.Pass())
    return ast.ClassDef(
        name=name,
        bases=[ast.Name(id="BaseModel")],
        keywords=[],
        body=body,
        decorator_list=[],
        type_params=[],
    )


# --- Top-level entry point ---------------------------------------------------


def render_schemas(schemas_block: dict[str, Any]) -> str:
    """Render a descriptor's ``schemas`` block to ``schemas.py`` source.

    Args:
        schemas_block: The descriptor's ``schemas`` key — a dict mapping a
            Python class name to a schema definition (``kind: "enum"`` with
            ``members``, or ``kind: "model"`` with ``fields``). May be empty,
            in which case only the imports are emitted.

    Returns:
        A complete Python source string ending in a single newline. Parses,
        ``py_compile``-s, and imports cleanly.

    Raises:
        RendererError: on any structural problem in the descriptor (unknown
            ``kind``, missing required fields, unrecognised type, etc.).
    """
    if not isinstance(schemas_block, dict):
        raise RendererError(
            f"schemas block must be a dict, got {type(schemas_block).__name__}"
        )

    body: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        ),
        ast.ImportFrom(
            module="enum",
            names=[ast.alias(name="Enum")],
            level=0,
        ),
        ast.ImportFrom(
            module="pydantic",
            names=[ast.alias(name="BaseModel"), ast.alias(name="Field")],
            level=0,
        ),
    ]

    # Known schema names — passed into the type-lowering pass so bare
    # identifiers that aren't declared schemas are rejected. Pairs with
    # the validator's R-SEM-FIELD-TYPE-KNOWN rule.
    known_schema_names: set[str] = {sname for sname in schemas_block.keys()}

    # Enums first so model fields can reference them without forward refs.
    # Within each pass we iterate the descriptor's dict in insertion order —
    # the JSON parser preserves it, and Python dicts have done likewise since
    # 3.7, so output is deterministic.
    for sname, sdef in schemas_block.items():
        if not isinstance(sdef, dict):
            raise RendererError(
                f"schema {sname!r}: definition must be a dict, "
                f"got {type(sdef).__name__}"
            )
        kind = sdef.get("kind")
        if kind == "enum":
            members = sdef.get("members")
            if not isinstance(members, list):
                raise RendererError(
                    f"enum {sname!r}: 'members' must be a list, got "
                    f"{type(members).__name__}"
                )
            body.append(_emit_enum(sname, members))
        elif kind == "model":
            pass  # second pass
        else:
            raise RendererError(f"schema {sname!r}: unknown kind {kind!r}")

    for sname, sdef in schemas_block.items():
        if sdef.get("kind") == "model":
            fields = sdef.get("fields", {})
            if not isinstance(fields, dict):
                raise RendererError(
                    f"model {sname!r}: 'fields' must be a dict, got "
                    f"{type(fields).__name__}"
                )
            body.append(
                _emit_model(
                    sname,
                    fields,
                    known_schema_names=known_schema_names,
                )
            )

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"
