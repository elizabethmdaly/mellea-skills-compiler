"""Module walker + per-symbol introspection.

Promoted from the Phase 0.1 throwaway spike at `scripts/introspect_mellea_spike.py`.
The spike achieved 100% coverage of ``llms.txt``-listed modules at the Phase 0
deadline; this module preserves that bar.

Stdlib only — no third-party introspection libraries.

Design points carried over from the spike:

* Module walker filtered by underscore convention + ``__all__``.
* Per-symbol capture: ``fqn``, ``kind``, signature (``inspect.signature``),
  type hints (``typing.get_type_hints``), first paragraph of docstring.
* Class member walking includes only members defined on the class itself
  (skipping inherited members), with an exception for ``enum`` members.
* Decorator-aware unwrapping via ``__wrapped__`` plus Mellea-specific marker
  attributes (``_mellea_generative``, ``_mellea_mify``, ``_mellea_hook``,
  ``_is_mellea_plugin``).

Improvements over the spike:

* ``canonical_definitions`` map for O(1) re-export deduplication.
* Stable JSON serialisation (``sort_keys=True``, ``indent=2``) so cross-version
  diffs are reviewable.
* ``surface_fingerprint`` sha256 over the canonical body so the cache layer
  (P1.C) can distinguish "regenerated" from "changed".
"""

from __future__ import annotations

import enum
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import pkgutil
import re
import sys
import textwrap
import traceback
import typing
from pathlib import Path
from typing import Any

from mellea_skills_compiler.mellea_surface.types import (
    DecoratorInfo,
    Surface,
    SurfaceMember,
    SurfaceModule,
    SurfaceSymbol,
)


# --- Mellea-specific knobs --------------------------------------------------

#: Marker attributes left by known Mellea decorators (``@generative``, etc.).
#: Detection of these is the one place introspection is Mellea-aware; the set
#: is bounded and version-pinned.
DECORATOR_MARKERS: tuple[str, ...] = (
    "_mellea_generative",
    "_mellea_mify",
    "_mellea_hook",
    "_is_mellea_plugin",
)

#: Members never skipped by the underscore filter on classes.
_CLASS_DUNDER_KEEP: frozenset[str] = frozenset({"__init__", "__call__"})


# --- Pure helpers (testable without importing mellea) -----------------------


def first_paragraph(doc: str | None) -> str | None:
    """First paragraph of a docstring with surrounding whitespace stripped."""
    if not doc:
        return None
    text = textwrap.dedent(doc).strip()
    return text.split("\n\n", 1)[0].strip() or None


# Object reprs frequently embed memory addresses (e.g. "at 0x109fa99d0") and
# those change across Python process invocations. The surface JSON must be
# byte-stable across runs for cache fingerprinting (P1.C); we redact those
# addresses to a stable placeholder before serialising.
_OBJECT_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _redact_addresses(text: str) -> str:
    """Replace ``at 0xABCDEF`` substrings with a stable placeholder.

    Many Mellea default arguments are sentinel objects whose ``repr()`` ends
    in ``<...Strategy object at 0x109fa99d0>``. The hex suffix changes per
    process; redacting it preserves identity (the class path is still
    visible) while making the serialised surface byte-identical across runs.
    """
    return _OBJECT_ADDRESS_RE.sub(" at 0xADDR", text)


def stringify(annot: Any) -> str:
    """JSON-friendly ``repr()`` that never raises, with addresses redacted."""
    try:
        text = annot if isinstance(annot, str) else repr(annot)
    except Exception:  # pragma: no cover - paranoid guard
        return "<unrepresentable>"
    return _redact_addresses(text)


def safe_signature(obj: Any) -> str | None:
    """``str(inspect.signature(obj))`` with addresses redacted; None if it
    can't be obtained.
    """
    try:
        text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None
    return _redact_addresses(text)


def safe_type_hints(obj: Any) -> dict[str, str]:
    """Stringified ``typing.get_type_hints`` map, or empty dict on failure."""
    try:
        hints = typing.get_type_hints(obj, include_extras=True)
    except Exception:
        return {}
    return {k: stringify(v) for k, v in hints.items()}


def detect_decorator(obj: Any) -> DecoratorInfo | None:
    """Detect known Mellea decorator wraps.

    Returns ``None`` when no wrapping or marker is found. Otherwise returns a
    dict suitable to splice into a ``SurfaceMember`` / ``SurfaceSymbol`` under
    the ``decorator`` key.
    """
    info: DecoratorInfo = {}
    wrapped = getattr(obj, "__wrapped__", None)
    if wrapped is not None:
        info["wrapped"] = True
        info["wrapped_qualname"] = getattr(wrapped, "__qualname__", None)
        sig = safe_signature(wrapped)
        if sig is not None:
            info["wrapped_signature"] = sig
    for marker in DECORATOR_MARKERS:
        if hasattr(obj, marker):
            info.setdefault("markers", []).append(marker)
    return info or None


def classify_member(member: Any) -> str:
    """Classify a class/enum member into a stable kind string."""
    if isinstance(member, enum.Enum):
        return "enum_member"
    if inspect.ismethod(member):
        return "classmethod"
    if isinstance(inspect.getattr_static(type(member), "__func__", None), staticmethod):
        return "staticmethod"
    if inspect.isfunction(member):
        # Unbound function on a class == instance method.
        return "method"
    if inspect.isbuiltin(member) or callable(member):
        return "callable"
    return "attribute"


def classify_top_level(obj: Any) -> str:
    """Classify a module-level symbol into a stable kind string."""
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            return "enum"
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if inspect.isbuiltin(obj) or callable(obj):
        return "callable"
    return "constant"


# --- Class / module walkers -------------------------------------------------


def class_members(cls: type) -> dict[str, SurfaceMember]:
    """Walk a class and capture members defined directly on it.

    Inherited members are skipped (with ``enum.Enum`` subclasses as the
    documented exception, since their members live on the metaclass).
    """
    members: dict[str, SurfaceMember] = {}
    is_enum = isinstance(cls, type(enum.Enum))
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") and name not in _CLASS_DUNDER_KEEP:
            continue
        # Only members defined on this class itself, with an exception for
        # enum members (which are surfaced via the metaclass, not __dict__).
        try:
            if name not in cls.__dict__ and not is_enum:
                continue
        except Exception:
            pass

        kind = classify_member(member)
        entry: SurfaceMember = {"kind": kind}
        if kind in ("method", "function", "classmethod", "staticmethod"):
            sig = safe_signature(member)
            if sig is not None:
                entry["signature"] = sig
            hints = safe_type_hints(member)
            if hints:
                entry["type_hints"] = hints
            entry["doc"] = first_paragraph(inspect.getdoc(member))
            deco = detect_decorator(member)
            if deco is not None:
                entry["decorator"] = deco
        elif kind == "enum_member":
            entry["value"] = repr(member.value)
        else:
            entry["repr"] = stringify(member)
        members[name] = entry
    return members


def public_names(mod: Any, namespace_prefix: str = "mellea") -> list[str]:
    """Return public names exposed by `mod`.

    Honours ``__all__`` if defined; otherwise falls back to ``dir()`` filtered
    by leading underscore and by ``__module__`` starting with
    ``namespace_prefix`` (to skip the std-lib re-exports that leak through
    every module's namespace). The prefix is a parameter so unit tests can
    exercise the walker against a synthetic fixture package.
    """
    explicit = getattr(mod, "__all__", None)
    if explicit is not None:
        return [n for n in explicit if not n.startswith("_")]
    return [
        n
        for n in dir(mod)
        if not n.startswith("_")
        and getattr(getattr(mod, n, None), "__module__", "").startswith(
            (namespace_prefix, "")
        )
    ]


def walk_modules(root_package: str = "mellea") -> list[str]:
    """Walk ``root_package`` and return all importable public sub-module names.

    Returns ``[root_package, ...sub-modules]`` in stable order. Private/dunder
    modules (any path segment starting with ``_``) are skipped.
    """
    root = importlib.import_module(root_package)
    modules = [root_package]
    if hasattr(root, "__path__") and root.__path__:
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{root_package}."):
            if any(part.startswith("_") for part in info.name.split(".")):
                continue
            modules.append(info.name)
    return modules


def introspect_module(
    mod_name: str, namespace_prefix: str = "mellea"
) -> SurfaceModule:
    """Introspect a single module by dotted name.

    Returns a SurfaceModule. On import failure, the returned dict carries
    ``import_error`` and ``traceback`` keys instead of ``symbols``.

    Symbols whose ``__module__`` does not start with ``namespace_prefix`` are
    skipped — they're considered re-exports from outside the target package
    and would inflate the surface with stdlib / third-party noise.
    """
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return {
            "import_error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=2),
        }

    out: SurfaceModule = {
        "doc": first_paragraph(inspect.getdoc(mod)),
        "symbols": {},
    }
    for name in public_names(mod, namespace_prefix=namespace_prefix):
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        # Skip re-exports from outside the target namespace.
        obj_mod = getattr(obj, "__module__", None)
        if obj_mod and not obj_mod.startswith(namespace_prefix):
            continue

        kind = classify_top_level(obj)
        entry: SurfaceSymbol = {"kind": kind, "defined_in": obj_mod}
        if kind in ("class", "enum"):
            entry["bases"] = [
                f"{b.__module__}.{b.__qualname__}"
                for b in obj.__bases__
                if b is not object
            ]
            entry["doc"] = first_paragraph(inspect.getdoc(obj))
            entry["members"] = class_members(obj)
            init_sig = safe_signature(obj)
            if init_sig is not None:
                entry["constructor_signature"] = init_sig
        elif kind in ("function", "callable"):
            sig = safe_signature(obj)
            if sig is not None:
                entry["signature"] = sig
            hints = safe_type_hints(obj)
            if hints:
                entry["type_hints"] = hints
            entry["doc"] = first_paragraph(inspect.getdoc(obj))
            deco = detect_decorator(obj)
            if deco is not None:
                entry["decorator"] = deco
        else:
            entry["repr"] = stringify(obj)
        out["symbols"][name] = entry  # type: ignore[index]
    return out


# --- Top-level surface assembly --------------------------------------------


def _resolve_mellea_version() -> str:
    """Best-effort Mellea version string."""
    try:
        return importlib.metadata.version("mellea")
    except Exception:
        try:
            mod = importlib.import_module("mellea")
        except Exception:
            return "unknown"
        return getattr(mod, "__version__", "unknown")


def _build_canonical_definitions(
    modules: dict[str, SurfaceModule],
    namespace_prefix: str = "mellea",
) -> dict[str, str]:
    """Deduplicate re-exports.

    A symbol like ``Backend`` may appear under both ``mellea.core.backend`` and
    ``mellea.stdlib.session``. The ``defined_in`` field captured per symbol
    gives the canonical-location signal; this function inverts it into an
    O(1) lookup ``{symbol_name: defining_module}``. Where the defining module
    is itself unknown or non-namespace, the entry is skipped.
    """
    canonical: dict[str, str] = {}
    for mod_name, mod in modules.items():
        symbols = mod.get("symbols") or {}
        for sym_name, sym in symbols.items():
            defined_in = sym.get("defined_in")
            if not defined_in or not defined_in.startswith(namespace_prefix):
                continue
            # Record the first sighting only; subsequent appearances are
            # re-exports. Sorted module iteration order is deterministic
            # (Python preserves dict insertion order; walk_modules sorts via
            # pkgutil.walk_packages which is stable).
            canonical.setdefault(sym_name, defined_in)
    return canonical


def _fingerprint(payload: dict[str, Any]) -> str:
    """sha256 hex digest over a stably-serialised payload."""
    body = json.dumps(payload, sort_keys=True, default=stringify).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def build_surface(root_package: str = "mellea") -> Surface:
    """Walk ``root_package`` and return the full introspected surface.

    Output ordering is deterministic; serialising the result with
    ``write_surface`` produces byte-identical output across runs whenever the
    inputs (Mellea version, Python version, imported state) are identical.
    """
    surface: Surface = {
        "mellea_version": _resolve_mellea_version(),
        "python_version": sys.version.split()[0],
        "modules": {},
    }
    for mod_name in walk_modules(root_package):
        surface["modules"][mod_name] = introspect_module(
            mod_name, namespace_prefix=root_package
        )

    surface["canonical_definitions"] = _build_canonical_definitions(
        surface["modules"], namespace_prefix=root_package
    )

    # Fingerprint is computed over everything except the fingerprint field
    # itself; this lets the cache layer (P1.C) distinguish "actually changed"
    # from "regenerated with same inputs".
    surface["surface_fingerprint"] = _fingerprint(dict(surface))
    return surface


# --- Serialisation ----------------------------------------------------------


def serialise_surface(surface: Surface) -> str:
    """Canonical-form JSON: sorted keys, two-space indent, trailing newline."""
    body = json.dumps(surface, indent=2, sort_keys=True, default=stringify)
    return body + "\n"


def write_surface(surface: Surface, out_path: Path | str) -> Path:
    """Write a Surface to disk in canonical form."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialise_surface(surface), encoding="utf-8")
    return path


def load_surface(path: Path | str) -> Surface:
    """Load a Surface from a JSON file emitted by ``write_surface``."""
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)
