"""Typed shapes for the introspected Mellea surface.

These are TypedDicts (not dataclasses) because the surface is round-tripped
through JSON for cache fingerprinting (P1.C) and prompt assembly. TypedDicts
give static-analysis benefit without imposing a JSON-to-object conversion
step at the boundary.

The one exception is `CoverageRow`: it never round-trips through JSON; it is a
transient result of `assess_coverage()`. A dataclass is the more ergonomic
shape there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


# --- Surface shapes ---------------------------------------------------------


class DecoratorInfo(TypedDict, total=False):
    """Information about Mellea-specific decorator wrappers."""

    wrapped: bool
    wrapped_qualname: str | None
    wrapped_signature: str | None
    markers: list[str]


class SurfaceMember(TypedDict, total=False):
    """A member defined on a class or enum.

    `kind` distinguishes ``method`` / ``function`` / ``classmethod`` /
    ``staticmethod`` / ``callable`` / ``enum_member`` / ``attribute``.
    """

    kind: str
    signature: str
    type_hints: dict[str, str]
    doc: str | None
    decorator: DecoratorInfo
    value: str
    repr: str


class SurfaceSymbol(TypedDict, total=False):
    """A top-level public symbol in a Mellea module.

    `kind` is one of ``class``, ``enum``, ``function``, ``callable``,
    ``constant``. `defined_in` records where the symbol's `__module__`
    attribute points; this is the canonical-location signal for the
    re-export deduplication index (see `Surface.canonical_definitions`).
    """

    kind: str
    defined_in: str | None
    bases: list[str]
    doc: str | None
    members: dict[str, SurfaceMember]
    constructor_signature: str
    signature: str
    type_hints: dict[str, str]
    decorator: DecoratorInfo
    repr: str


class SurfaceModule(TypedDict, total=False):
    """One Mellea module's worth of public surface."""

    doc: str | None
    symbols: dict[str, SurfaceSymbol]
    import_error: str
    traceback: str


class Surface(TypedDict, total=False):
    """The complete introspected Mellea surface.

    Top-level fields:

    - ``mellea_version``: installed Mellea version string.
    - ``python_version``: interpreter version that produced the surface.
    - ``modules``: ``{module_name: SurfaceModule}``.
    - ``canonical_definitions``: ``{symbol_qualname: defining_module}``.
      Re-export deduplication index — when the same symbol shows up under
      multiple module paths (e.g. ``Backend`` in ``mellea.core.backend`` and
      ``mellea.stdlib.session``), this map records the canonical defining
      module so downstream consumers can resolve "where is this defined" in
      O(1) without scanning every module.
    - ``surface_fingerprint``: sha256 of the canonical-form JSON minus the
      fingerprint field itself. Used by the cache layer to detect "did the
      surface actually change, or was it just regenerated".
    """

    mellea_version: str
    python_version: str
    modules: dict[str, SurfaceModule]
    canonical_definitions: dict[str, str]
    surface_fingerprint: str


# --- Coverage shape ---------------------------------------------------------


CoverageStatus = Literal["found", "missing", "empty", "import_error"]


@dataclass
class CoverageRow:
    """One row of the llms.txt → introspected-surface cross-reference table."""

    url: str
    expected_module: str
    status: str  # CoverageStatus
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "expected_module": self.expected_module,
            "status": self.status,
            "note": self.note,
        }
