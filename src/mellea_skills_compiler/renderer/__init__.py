"""Descriptor renderer — composition operator plug-in framework.

Phase 2.B (this stream) ships the 5 composition operators as plug-in modules under
``operators/``. Phase 2.A owns ``core.py`` / ``nodes.py`` / ``source_map.py``
(the renderer core, the per-node dispatch, and the source-map sink); this
module deliberately ships **only** the public surface that operators depend on:

- :class:`OperatorRenderer` — the ABC P2.A's pipeline walker invokes per node.
- :func:`register_operator` / :func:`get_operator` — the registry used both by
  this package's ``operators/__init__.py`` (to self-register on import) and by
  the renderer core (to dispatch by ``node["operator"]``).
- :class:`RendererError` — the canonical error raised by operators and by core
  on malformed descriptor input.

The :class:`RenderContext` and :class:`SourceMap` types are owned by P2.A and
imported lazily under :data:`typing.TYPE_CHECKING` so this package can be
imported without P2.A's core present (and vice versa).
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # P2.A owns these concrete types; we only need them at type-check time.
    from mellea_skills_compiler.renderer.core import RenderContext  # noqa: F401


class RendererError(Exception):
    """Raised by operators (and by the renderer core) on malformed descriptor input.

    Operators MUST raise this — not ``ValueError`` / ``KeyError`` / ``TypeError`` —
    so the repair loop and CI surfaces can catch a single, well-known exception
    type.
    """


class OperatorRenderer(ABC):
    """Plug-in interface for composition operators (plan §4.5).

    Concrete subclasses are instantiated once and registered via
    :func:`register_operator` at import time. The renderer core looks the
    operator up by :attr:`operator_name` (the value of ``node["operator"]`` in
    the descriptor) and invokes :meth:`render`.

    Implementations must be **stateless** — all per-call state lives on
    ``ctx`` / ``node`` / ``render_body``.
    """

    #: The descriptor ``operator`` key this renderer handles. Class variable;
    #: set by each subclass.
    operator_name: str

    @abstractmethod
    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        """Render a composition node into a list of Python AST statements.

        Args:
            node: The descriptor composition node. Always has ``id``, ``kind``,
                ``operator``. Operator-specific required fields are listed in
                the schema strawman (``descriptor-schema-v0.md``).
            ctx: P2.A's render context (imports, identifier registry, ref
                resolver, source map).
            render_body: Callback supplied by the renderer core; converts a
                child-node array into AST statements. Operators recurse via
                this callback rather than calling each other directly, so the
                core retains a single point of node-kind dispatch.

        Returns:
            A list of ``ast.stmt`` nodes to be inserted into the enclosing
            function body. The list is opaque to the core; operators may emit
            any combination of ``Assign``, ``For``, ``If``, ``With``, etc.

        Raises:
            RendererError: if any operator-specific required field is missing
                or malformed.
        """


# ---- Registry --------------------------------------------------------------

_REGISTRY: dict[str, OperatorRenderer] = {}


def register_operator(renderer: OperatorRenderer) -> None:
    """Register an operator renderer instance under its ``operator_name``.

    Idempotent for the *same* instance (re-registering the identical object
    is a no-op, useful when the operators package is imported twice via
    different paths). Re-registering a *different* instance under an
    existing name is a programming error and raises :class:`RendererError`.
    """
    name = renderer.operator_name
    if not name:
        raise RendererError(f"operator renderer {renderer!r} has empty operator_name")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not renderer:
        raise RendererError(
            f"operator {name!r} already registered (existing={existing!r}, new={renderer!r})"
        )
    _REGISTRY[name] = renderer


def get_operator(name: str) -> OperatorRenderer:
    """Look up an operator renderer by name.

    Raises:
        RendererError: if no operator is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise RendererError(
            f"unknown composition operator: {name!r} "
            f"(registered: {sorted(_REGISTRY)})"
        ) from exc


def registered_operator_names() -> list[str]:
    """Return the sorted list of registered operator names. Test-only helper."""
    return sorted(_REGISTRY)


__all__ = [
    "OperatorRenderer",
    "RenderContext",
    "RenderResult",
    "RendererError",
    "SourceMap",
    "SourceMapEntry",
    "get_operator",
    "register_operator",
    "registered_operator_names",
    "render_descriptor",
]


def __getattr__(name: str):
    """Lazy re-export of P2.A's renderer-core symbols.

    Avoids an import cycle: ``core.py`` and ``nodes.py`` import
    ``RendererError`` / ``OperatorRenderer`` / registry helpers from this
    package's top-level, so we must NOT import them eagerly here. The lazy
    ``__getattr__`` (PEP 562) lets callers write
    ``from mellea_skills_compiler.renderer import render_descriptor`` without
    forcing an import-time cycle through ``core``.
    """
    if name in {"render_descriptor", "RenderContext", "RenderResult", "ResolvedRef"}:
        from mellea_skills_compiler.renderer import core as _core

        return getattr(_core, name)
    if name in {"SourceMap", "SourceMapEntry"}:
        from mellea_skills_compiler.renderer import source_map as _sm

        return getattr(_sm, name)
    raise AttributeError(f"module 'mellea_skills_compiler.renderer' has no attribute {name!r}")
