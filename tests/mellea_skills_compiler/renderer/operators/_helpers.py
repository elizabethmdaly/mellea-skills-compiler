"""Test helpers for the composition operator unit suites.

P2.A owns the real :class:`RenderContext` and :class:`SourceMap`; until that
lands these stand-ins implement the documented contract surface so the
operator suites can run in isolation. The shapes here intentionally mirror
the contract spec (P2.A's classes will be drop-in replacements).
"""

from __future__ import annotations

import ast
import py_compile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mellea_skills_compiler.renderer import get_operator


@dataclass
class _StubSourceMap:
    """Records (node_id, stmt) tuples; supports both signature shapes."""

    entries: list[tuple[str, ast.stmt | None, int | None]] = field(default_factory=list)

    def record(self, *, node_id: str, stmt: ast.stmt | None = None, line: int | None = None) -> None:
        self.entries.append((node_id, stmt, line))

    def has_node(self, node_id: str) -> bool:
        return any(nid == node_id for nid, _, _ in self.entries)


@dataclass
class _StubContext:
    """Minimal stand-in for P2.A's :class:`RenderContext`.

    Implements ``add_import`` / ``register_id`` / ``source_map`` and a
    ``resolve_ref`` that mirrors the spike's lowering rules for tests. The
    full :class:`RenderContext` may add fields by the time of integration;
    this stub keeps the operator suites independent of those refinements.
    """

    imports: dict[str, set[str]] = field(default_factory=dict)
    ids: dict[str, str] = field(default_factory=dict)
    source_map: _StubSourceMap = field(default_factory=_StubSourceMap)

    def add_import(self, module: str, name: str) -> None:
        self.imports.setdefault(module, set()).add(name)

    def register_id(self, identifier: str, kind: str) -> None:
        self.ids[identifier] = kind

    def resolve_ref(self, ref_dict: dict) -> ast.expr:
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
        raise AssertionError(f"test helper cannot resolve: {ref_dict!r}")


def make_test_context() -> _StubContext:
    """Construct a fresh test context for a single operator render."""
    return _StubContext()


def make_render_body(
    *,
    fake_call: Callable[[dict], ast.stmt] | None = None,
) -> Callable[[list[dict]], list[ast.stmt]]:
    """Produce a ``render_body`` callable for operator tests.

    The default lowering treats every node as either:

    - ``kind == "call"`` → ``<id> = _call(<symbol>)`` (a placeholder call).
    - ``kind == "composition"`` → recurse through the registered operator.

    Tests can override the call lowering by passing ``fake_call``.
    """

    def _default_call(node: dict) -> ast.stmt:
        target = ast.Name(id=node["id"], ctx=ast.Store())
        value: ast.expr = ast.Call(
            func=ast.Name(id="_call", ctx=ast.Load()),
            args=[ast.Constant(value=node.get("symbol", "stub"))],
            keywords=[],
        )
        return ast.Assign(targets=[target], value=value)

    lower_call = fake_call or _default_call

    def render_body(body: list[dict]) -> list[ast.stmt]:
        stmts: list[ast.stmt] = []
        for child in body:
            kind = child.get("kind", "call")
            if kind == "call":
                stmts.append(lower_call(child))
            elif kind == "composition":
                op = get_operator(child["operator"])
                # The inner operator gets the same render_body callback so
                # nested composition works.
                ctx = _CURRENT_CTX["ctx"]
                stmts.extend(op.render(child, ctx, render_body))
            else:
                raise AssertionError(f"unknown kind in test render_body: {kind!r}")
        return stmts

    return render_body


_CURRENT_CTX: dict[str, Any] = {"ctx": None}


def bind_context(ctx: _StubContext) -> None:
    """Bind the current context for nested-render_body recursion in tests."""
    _CURRENT_CTX["ctx"] = ctx


def assert_compiles(stmts: list[ast.stmt]) -> str:
    """Wrap statements in a module, unparse, and ``py_compile``-check them.

    Returns the unparsed source for further assertions. Wraps stmts in a
    ``def _run(): ...`` so things like ``break`` (legal only inside a loop —
    we use ``for ... else``) and bare ``return`` survive validation.
    """
    # Wrap in a function so ``return``/``break`` are legal at the outer scope.
    func = ast.FunctionDef(
        name="_run",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=stmts if stmts else [ast.Pass()],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    src = ast.unparse(module)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = Path(f.name)
    try:
        py_compile.compile(str(path), doraise=True)
    finally:
        path.unlink(missing_ok=True)
    return src
