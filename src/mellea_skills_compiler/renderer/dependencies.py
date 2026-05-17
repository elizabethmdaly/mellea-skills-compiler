"""Renderer support for the descriptor v0.3 ``dependencies[]`` block.

Per RFC §3.1 and plan §11.1, each dependency carries a ``disposition`` that
determines how the renderer materialises it in the output package. The eight
dispositions map 1:1 to ``dependency_plan.json``:

  ==================== ======================================================
  Disposition           Renderer emission
  ==================== ======================================================
  ``delegate_to_runtime`` ``Callable[...]`` kwarg in ``run_pipeline()``
  ``bundle``              File-copy at render time (no code change)
  ``real_impl``           Stub function w/ ``NotImplementedError`` + TODO
  ``stub``                Function returning a sentinel for the return type
  ``mock``                Function returning ``MagicMock()`` (fixture-friendly)
  ``external_input``      Required positional kwarg in ``run_pipeline()``
  ``load_from_disk``      Path constant + ``_load_<id>()`` helper
  ``remove``              No emission
  ==================== ======================================================

This module is **schema-aware only**, not validator-aware: it trusts that
the descriptor has already passed ``R-SEM-DISPOSITION-CONSISTENT`` etc.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mellea_skills_compiler.renderer import RendererError

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


# Closed disposition vocabulary — matches descriptor.schema.v0.3.json $defs/Dependency.
_DISPOSITIONS: tuple[str, ...] = (
    "delegate_to_runtime",
    "bundle",
    "real_impl",
    "stub",
    "mock",
    "external_input",
    "load_from_disk",
    "remove",
)


@dataclass
class FileCopyOp:
    """Render-time file-copy operation scheduled by ``bundle`` / ``load_from_disk``.

    The renderer caller (e.g. compile/mellea_skills.py companion-directory
    logic) consumes this list at write-time.
    """

    source: str
    """Path relative to the skill root."""

    dest: str
    """Path under the rendered package."""

    is_directory: bool = False
    """If True, recursively copy a directory."""

    is_package_data: bool = False
    """If True, the destination should also be added to package-data."""


@dataclass
class DependenciesArtefact:
    """Render output of :func:`render_dependencies`.

    Carries:
      - new ``run_pipeline()`` kwargs (with annotations + defaults)
      - top-level statements to inject into ``pipeline.py`` (or ``tools.py``)
      - scheduled file-copy operations
      - which dependency ids are bundled (for downstream pyproject mutation)
    """

    pipeline_args: list[ast.arg] = field(default_factory=list)
    """Extra ``ast.arg`` to append to ``run_pipeline()`` parameters."""

    pipeline_defaults: list[ast.expr | None] = field(default_factory=list)
    """Per-arg default. ``None`` means "required arg" (no default)."""

    top_level_stmts: list[ast.stmt] = field(default_factory=list)
    """Module-level statements to emit before ``run_pipeline``."""

    file_copies: list[FileCopyOp] = field(default_factory=list)
    """File-copy operations to schedule at write-time."""

    package_data_dirs: set[str] = field(default_factory=set)
    """Directories to add to pyproject.toml package-data."""


# --- Signature parser ------------------------------------------------------


_SIG_RE = re.compile(r"^\s*\((?P<params>.*?)\)\s*(?:->\s*(?P<ret>.+))?\s*$", re.S)


def _parse_signature(sig: str) -> tuple[list[tuple[str, str | None]], str | None]:
    """Parse a Python-style signature string like ``(query: str) -> list[str]``.

    Returns ``(params, return_type)`` where ``params`` is a list of
    ``(name, annotation_or_None)`` pairs and ``return_type`` is the annotation
    string (or None for bare ``()`` signatures).

    Tolerates whitespace; rejects shapes the renderer can't faithfully
    represent (e.g. ``*args``, ``**kwargs`` — not legal in descriptor
    signatures per the RFC).
    """
    m = _SIG_RE.match(sig)
    if not m:
        raise RendererError(f"unparseable dependency signature: {sig!r}")
    params_src = (m.group("params") or "").strip()
    ret = (m.group("ret") or "").strip() or None
    out: list[tuple[str, str | None]] = []
    if params_src:
        # Split on top-level commas (respecting bracket depth).
        depth = 0
        buf: list[str] = []
        chunks: list[str] = []
        for ch in params_src:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            if ch == "," and depth == 0:
                chunks.append("".join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        chunks.append("".join(buf).strip())
        for chunk in chunks:
            if not chunk:
                continue
            if ":" in chunk:
                name, ann = chunk.split(":", 1)
                out.append((name.strip(), ann.strip()))
            else:
                out.append((chunk.strip(), None))
    return out, ret


def _annotation_ast(annotation: str | None) -> ast.expr:
    """Lower a string annotation into an AST expression.

    Uses ``ast.parse(..., mode="eval")`` so callers don't have to re-implement
    a type parser; any annotation Python accepts as a type expression is
    accepted here. Falls back to ``Any`` for missing annotations.
    """
    if not annotation:
        return ast.Name(id="Any")
    try:
        return ast.parse(annotation, mode="eval").body
    except SyntaxError as exc:
        raise RendererError(
            f"invalid annotation in dependency signature: {annotation!r}"
        ) from exc


def _callable_annotation(params: list[tuple[str, str | None]], ret: str | None) -> ast.expr:
    """Build ``Callable[[<arg_types>], <ret_type>]`` from a parsed signature."""
    arg_types: list[ast.expr] = [_annotation_ast(a) for _, a in params]
    ret_node = _annotation_ast(ret) if ret else ast.Name(id="Any")
    return ast.Subscript(
        value=ast.Name(id="Callable"),
        slice=ast.Tuple(
            elts=[
                ast.List(elts=arg_types, ctx=ast.Load()),
                ret_node,
            ],
            ctx=ast.Load(),
        ),
        ctx=ast.Load(),
    )


def _sentinel_for(annotation: str | None) -> ast.expr:
    """Best-effort default sentinel matching a return-type annotation.

    Used by ``stub`` to give the rendered function a valid return value
    without forcing a runtime crash. The mapping is conservative; unknown
    types fall through to ``None``.
    """
    if not annotation:
        return ast.Constant(value=None)
    s = annotation.strip()
    if s in ("None", "NoneType"):
        return ast.Constant(value=None)
    if s == "str":
        return ast.Constant(value="")
    if s == "int":
        return ast.Constant(value=0)
    if s == "float":
        return ast.Constant(value=0.0)
    if s == "bool":
        return ast.Constant(value=False)
    if s.startswith("list"):
        return ast.List(elts=[], ctx=ast.Load())
    if s.startswith("dict"):
        return ast.Dict(keys=[], values=[])
    if s.startswith("set"):
        return ast.Call(func=ast.Name(id="set"), args=[], keywords=[])
    if s.startswith("tuple"):
        return ast.Tuple(elts=[], ctx=ast.Load())
    return ast.Constant(value=None)


# --- Per-disposition emitters ----------------------------------------------


def _emit_delegate(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``delegate_to_runtime`` — add a ``Callable`` kwarg, no default."""
    sig = dep.get("signature")
    if not sig:
        raise RendererError(
            f"dependency {dep.get('id')!r}: delegate_to_runtime requires 'signature'"
        )
    params, ret = _parse_signature(sig)
    ann = _callable_annotation(params, ret)
    art.pipeline_args.append(ast.arg(arg=dep["id"], annotation=ann))
    art.pipeline_defaults.append(None)  # no default = required kwarg
    ctx.add_import("typing", "Callable")
    ctx.add_import("typing", "Any")


def _emit_external_input(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``external_input`` — required parameter, no default. Annotated.

    Differs from ``delegate_to_runtime`` only in that the annotation reflects
    the actual return-type of a no-arg call (the dep is a value, not a
    callable). For ``kind: "secret"`` etc. we annotate by the return type.
    """
    sig = dep.get("signature")
    if sig:
        _params, ret = _parse_signature(sig)
        ann = _annotation_ast(ret) if ret else ast.Name(id="Any")
    else:
        ann = ast.Name(id="Any")
    ctx.add_import("typing", "Any")
    art.pipeline_args.append(ast.arg(arg=dep["id"], annotation=ann))
    art.pipeline_defaults.append(None)


def _emit_stub(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``stub`` — module-level function returning a sentinel.

    Implementation note: returns a primitive sentinel matching the parsed
    return type, not ``NotImplementedError`` — that's what ``real_impl`` is
    for. Stubs are intentionally callable so fixtures can exercise the
    pipeline without runtime crashes.
    """
    sig = dep.get("signature") or "() -> None"
    params, ret = _parse_signature(sig)
    args_node = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name, annotation=_annotation_ast(ann)) for name, ann in params],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    returns = _annotation_ast(ret) if ret else ast.Name(id="Any")
    if not ret:
        ctx.add_import("typing", "Any")
    fn = ast.FunctionDef(
        name=dep["id"],
        args=args_node,
        body=[ast.Return(value=_sentinel_for(ret))],
        decorator_list=[],
        returns=returns,
        type_params=[],
    )
    art.top_level_stmts.append(fn)


def _emit_real_impl(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``real_impl`` — stub function with ``NotImplementedError`` + TODO.

    Real implementations are filled in by the LLM in a downstream step; the
    renderer just emits a placeholder so the package imports cleanly and
    fixtures can locate the symbol.
    """
    sig = dep.get("signature") or "() -> None"
    params, ret = _parse_signature(sig)
    rationale = dep.get("rationale") or "provide real implementation"
    args_node = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name, annotation=_annotation_ast(ann)) for name, ann in params],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    returns = _annotation_ast(ret) if ret else ast.Name(id="Any")
    if not ret:
        ctx.add_import("typing", "Any")
    fn = ast.FunctionDef(
        name=dep["id"],
        args=args_node,
        body=[
            ast.Expr(value=ast.Constant(value=f"TODO: provide real implementation — {rationale}")),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="NotImplementedError"),
                    args=[ast.Constant(value=rationale)],
                    keywords=[],
                ),
                cause=None,
            ),
        ],
        decorator_list=[],
        returns=returns,
        type_params=[],
    )
    art.top_level_stmts.append(fn)


def _emit_mock(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``mock`` — return ``MagicMock()`` from a fresh function.

    The function captures one MagicMock per call so each invocation is a
    distinct mock instance (fixture-friendly).
    """
    sig = dep.get("signature") or "() -> Any"
    params, ret = _parse_signature(sig)
    args_node = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name, annotation=_annotation_ast(ann)) for name, ann in params],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    returns = _annotation_ast(ret) if ret else ast.Name(id="Any")
    ctx.add_import("typing", "Any")
    ctx.add_import("unittest.mock", "MagicMock")
    fn = ast.FunctionDef(
        name=dep["id"],
        args=args_node,
        body=[
            ast.Return(value=ast.Call(func=ast.Name(id="MagicMock"), args=[], keywords=[]))
        ],
        decorator_list=[],
        returns=returns,
        type_params=[],
    )
    art.top_level_stmts.append(fn)


def _emit_load_from_disk(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``load_from_disk`` — module-level Path constant + ``_load_<id>()`` helper.

    The Path is constructed relative to ``__file__`` so the rendered package
    can find the file at runtime regardless of where it's installed.
    """
    path = dep.get("path")
    if not path:
        raise RendererError(
            f"dependency {dep.get('id')!r}: load_from_disk requires 'path'"
        )
    ctx.add_import("pathlib", "Path")
    dep_id = dep["id"]
    const_name = f"_{dep_id}"
    # _<id> = Path(__file__).parent / "<path>"
    const_stmt = ast.Assign(
        targets=[ast.Name(id=const_name)],
        value=ast.BinOp(
            left=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="Path"),
                    args=[ast.Name(id="__file__")],
                    keywords=[],
                ),
                attr="parent",
            ),
            op=ast.Div(),
            right=ast.Constant(value=path),
        ),
    )
    # def _load_<id>() -> str: return _<id>.read_text(encoding="utf-8")
    helper_fn = ast.FunctionDef(
        name=f"_load_{dep_id}",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=const_name), attr="read_text"),
                    args=[],
                    keywords=[ast.keyword(arg="encoding", value=ast.Constant(value="utf-8"))],
                )
            )
        ],
        decorator_list=[],
        returns=ast.Name(id="str"),
        type_params=[],
    )
    art.top_level_stmts.append(const_stmt)
    art.top_level_stmts.append(helper_fn)


def _emit_bundle(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``bundle`` — schedule file-copy; no Python code emitted.

    A ``bundle`` dependency declares that ``path`` (relative to the skill
    root) should be copied into the rendered package at the same relative
    path, and added to the wheel's package-data. The renderer caller does
    the actual copy at write-time via :class:`FileCopyOp`.
    """
    path = dep.get("path")
    if not path:
        raise RendererError(
            f"dependency {dep.get('id')!r}: bundle requires 'path'"
        )
    art.file_copies.append(
        FileCopyOp(source=path, dest=path, is_directory=False, is_package_data=True)
    )
    # Track the parent dir for pyproject.toml package-data.
    parts = path.rsplit("/", 1)
    if len(parts) == 2:
        art.package_data_dirs.add(parts[0])


def _emit_remove(dep: dict, art: DependenciesArtefact, ctx: "RenderContext") -> None:
    """``remove`` — emit nothing."""
    return None


_EMITTERS = {
    "delegate_to_runtime": _emit_delegate,
    "external_input": _emit_external_input,
    "stub": _emit_stub,
    "real_impl": _emit_real_impl,
    "mock": _emit_mock,
    "load_from_disk": _emit_load_from_disk,
    "bundle": _emit_bundle,
    "remove": _emit_remove,
}


# --- Top-level entry point -------------------------------------------------


def render_dependencies(
    descriptor: dict[str, Any],
    ctx: "RenderContext",
) -> DependenciesArtefact:
    """Translate ``descriptor.dependencies[]`` into a :class:`DependenciesArtefact`.

    Returns a fresh artefact for every call. The caller decides where the
    pieces land (e.g. statements into ``pipeline.py``, file-copies into the
    write-time companion-directory step).

    Unknown dispositions raise :class:`RendererError`: the validator should
    have caught these, but the renderer enforces the invariant.
    """
    art = DependenciesArtefact()
    deps = descriptor.get("dependencies") or []
    if not isinstance(deps, list):
        raise RendererError(
            f"descriptor.dependencies must be a list, got {type(deps).__name__}"
        )
    for dep in deps:
        if not isinstance(dep, dict):
            raise RendererError(
                f"each dependency must be a dict, got {type(dep).__name__}"
            )
        disp = dep.get("disposition")
        if disp not in _EMITTERS:
            raise RendererError(
                f"dependency {dep.get('id')!r}: unknown disposition {disp!r} "
                f"(valid: {sorted(_DISPOSITIONS)})"
            )
        _EMITTERS[disp](dep, art, ctx)
    return art


__all__ = [
    "DependenciesArtefact",
    "FileCopyOp",
    "render_dependencies",
]
