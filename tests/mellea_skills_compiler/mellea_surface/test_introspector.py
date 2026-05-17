"""Unit tests for the introspection walker.

These tests deliberately use small in-memory fixture modules rather than
importing all of mellea — the heavy mellea-walk is covered by
``test_integration.py`` (one slow test).
"""

from __future__ import annotations

import enum
import importlib
import re
import sys
import textwrap
import types
from functools import wraps
from pathlib import Path

import pytest

from mellea_skills_compiler.mellea_surface.introspector import (
    DECORATOR_MARKERS,
    _build_canonical_definitions,
    _fingerprint,
    _redact_addresses,
    build_surface,
    class_members,
    classify_member,
    classify_top_level,
    detect_decorator,
    first_paragraph,
    introspect_module,
    load_surface,
    public_names,
    safe_signature,
    safe_type_hints,
    serialise_surface,
    stringify,
    walk_modules,
    write_surface,
)


# --- Synthetic in-memory module fixture ------------------------------------
#
# Used for the "do these helpers do the right thing on small synthetic
# inputs" tests. Class/function objects are built once and re-used.


class _Greeter:
    """A greeter class.

    Long-form docs that should not survive the first-paragraph filter.
    """

    CONSTANT: int = 42

    def __init__(self, name: str) -> None:
        """Store the name."""
        self.name = name

    def hello(self, greeting: str = "hi") -> str:
        """Return a greeting."""
        return f"{greeting}, {self.name}"

    @classmethod
    def make(cls, name: str) -> "_Greeter":
        """Construct."""
        return cls(name)

    @staticmethod
    def static_method() -> int:
        """Return a constant."""
        return 7

    def _private(self) -> None:
        """Should be skipped by underscore filter."""


class _Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


# --- On-disk package fixture (for walk_modules / build_surface tests) -------


@pytest.fixture
def ondisk_package(tmp_path: Path, monkeypatch):
    """Create a small on-disk Python package and put it on sys.path.

    Layout::

        msctmpfixture/
            __init__.py
            sub.py

    Returns the dotted package name. Cleans up sys.modules + sys.path
    on teardown.
    """
    pkg_name = "msctmpfixture"
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        textwrap.dedent(
            '''\
            """Fixture package docstring.

            With trailing paragraph that should be dropped.
            """

            from __future__ import annotations

            import enum


            class Greeter:
                """A greeter class.

                Long-form docs that should not survive first-paragraph.
                """

                CONSTANT: int = 42

                def __init__(self, name: str) -> None:
                    """Store the name."""
                    self.name = name

                def hello(self, greeting: str = "hi") -> str:
                    """Return a greeting."""
                    return f"{greeting}, {self.name}"


            class Color(enum.Enum):
                RED = "red"
                BLUE = "blue"


            def helper(x: int, y: int = 1) -> int:
                """Add two ints."""
                return x + y


            __all__ = ["Greeter", "Color", "helper"]
            '''
        ),
        encoding="utf-8",
    )
    (pkg_dir / "sub.py").write_text(
        textwrap.dedent(
            '''\
            """Sub doc."""


            def nested(z: str) -> str:
                """Echo."""
                return z


            __all__ = ["nested"]
            '''
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    yield pkg_name

    for mod_name in list(sys.modules):
        if mod_name == pkg_name or mod_name.startswith(f"{pkg_name}."):
            del sys.modules[mod_name]


# --- Pure-helper tests ------------------------------------------------------


class TestFirstParagraph:
    def test_none_returns_none(self):
        assert first_paragraph(None) is None

    def test_empty_returns_none(self):
        assert first_paragraph("") is None
        assert first_paragraph("   \n  ") is None

    def test_takes_first_paragraph(self):
        doc = "Headline.\n\nLong-form details below."
        assert first_paragraph(doc) == "Headline."

    def test_dedents_and_strips(self):
        doc = "    Indented headline.\n    \n    Body."
        assert first_paragraph(doc) == "Indented headline."

    def test_single_paragraph(self):
        assert first_paragraph("Only this.") == "Only this."


class TestRedactAddresses:
    def test_redacts_hex_address(self):
        text = "<MyClass object at 0x109fa99d0>"
        assert _redact_addresses(text) == "<MyClass object at 0xADDR>"

    def test_redacts_multiple(self):
        text = "a at 0x1 b at 0xABCD c"
        assert _redact_addresses(text) == "a at 0xADDR b at 0xADDR c"

    def test_passthrough_when_no_address(self):
        assert _redact_addresses("plain text") == "plain text"


class TestStringify:
    def test_string_passthrough(self):
        assert stringify("foo") == "foo"

    def test_non_string_uses_repr(self):
        assert stringify(7) == "7"
        assert stringify(int) == repr(int)

    def test_address_redacted_in_repr(self):
        class Marker:
            pass

        out = stringify(Marker())
        assert "at 0xADDR" in out
        # Hex digits beyond "ADDR" should not appear:
        assert not re.search(r"at 0x[0-9a-f]+", out)


class TestSafeSignature:
    def test_normal_function(self):
        def f(x: int, y: str = "z") -> int:
            return x

        sig = safe_signature(f)
        assert sig is not None
        assert "x" in sig and "y" in sig

    def test_uninspectable_returns_none(self):
        # Plain int has no signature.
        assert safe_signature(7) is None


class TestSafeTypeHints:
    def test_function_with_annotations(self):
        def f(x: int, y: str = "z") -> bool:
            return True

        hints = safe_type_hints(f)
        assert hints == {"x": repr(int), "y": repr(str), "return": repr(bool)}

    def test_no_annotations_returns_empty(self):
        def g(x, y):
            return x

        assert safe_type_hints(g) == {}


class TestDetectDecorator:
    def test_plain_function_returns_none(self):
        def f():
            pass

        assert detect_decorator(f) is None

    def test_wrapped_function_records_wrapped(self):
        def inner(x: int) -> int:
            """inner doc."""
            return x

        @wraps(inner)
        def outer(*args, **kwargs):
            return inner(*args, **kwargs)

        info = detect_decorator(outer)
        assert info is not None
        assert info["wrapped"] is True
        # `inspect.signature` stringifies forward-ref / future-annotations
        # annotations as quoted strings on Python 3.12+. Either form is accepted.
        assert info["wrapped_signature"] in {"(x: int) -> int", "(x: 'int') -> 'int'"}

    def test_marker_is_captured(self):
        def f():
            pass

        f._mellea_generative = True  # type: ignore[attr-defined]
        info = detect_decorator(f)
        assert info is not None
        assert info["markers"] == ["_mellea_generative"]

    def test_all_known_markers(self):
        for marker in DECORATOR_MARKERS:
            def f():
                pass

            setattr(f, marker, True)
            info = detect_decorator(f)
            assert info is not None
            assert marker in info["markers"]


class TestClassifyMember:
    def test_method(self):
        class C:
            def m(self):
                pass

        assert classify_member(C.__dict__["m"]) == "method"

    def test_staticmethod_resolved_via_class(self):
        class C:
            @staticmethod
            def s():
                pass

        # Accessing via class returns the underlying function in CPython.
        resolved = getattr(C, "s")
        assert classify_member(resolved) in ("staticmethod", "method")

    def test_classmethod(self):
        class C:
            @classmethod
            def cm(cls):
                pass

        resolved = getattr(C, "cm")
        assert classify_member(resolved) == "classmethod"

    def test_enum_member(self):
        class E(enum.Enum):
            A = 1

        assert classify_member(E.A) == "enum_member"

    def test_attribute(self):
        assert classify_member(42) == "attribute"


class TestClassifyTopLevel:
    def test_function(self):
        def f():
            pass

        assert classify_top_level(f) == "function"

    def test_class(self):
        class C:
            pass

        assert classify_top_level(C) == "class"

    def test_enum(self):
        class E(enum.Enum):
            A = 1

        assert classify_top_level(E) == "enum"

    def test_constant(self):
        assert classify_top_level(42) == "constant"
        assert classify_top_level("hi") == "constant"


# --- Class member walker tests ---------------------------------------------


class TestClassMembers:
    def test_skips_inherited_members(self):
        members = class_members(_Greeter)
        assert "__init__" in members
        assert "hello" in members
        assert "make" in members
        assert "static_method" in members
        assert "CONSTANT" in members
        # Inherited from object should NOT appear:
        assert "__class__" not in members
        # Private should NOT appear:
        assert "_private" not in members

    def test_enum_members_appear(self):
        members = class_members(_Color)
        assert "RED" in members
        assert "BLUE" in members
        assert members["RED"]["kind"] == "enum_member"
        assert members["RED"]["value"] == repr("red")

    def test_method_captures_signature_and_doc(self):
        members = class_members(_Greeter)
        hello = members["hello"]
        assert hello["kind"] == "method"
        assert "greeting" in hello["signature"]
        assert hello["doc"] == "Return a greeting."

    def test_constant_attribute_captured(self):
        members = class_members(_Greeter)
        assert members["CONSTANT"]["kind"] == "attribute"
        assert members["CONSTANT"]["repr"] == "42"


class TestPublicNames:
    def test_respects_dunder_all(self):
        mod = types.ModuleType("m_dunder_all")
        mod.__all__ = ["a", "b"]
        mod.a = 1
        mod.b = 2
        mod.c = 3  # hidden — not in __all__
        assert public_names(mod) == ["a", "b"]

    def test_falls_back_to_dir_when_no_all(self):
        # Build a synthetic module whose function `f` has the right __module__.
        mod = types.ModuleType("m_dir_fallback")

        def f():
            pass

        f.__module__ = "m_dir_fallback"
        mod.f = f
        # The default namespace_prefix is "mellea", but the empty-string
        # prefix tuple-trick means objects whose __module__ is "" (i.e. local
        # constants like ints) survive. f's __module__ is "m_dir_fallback",
        # which doesn't start with "mellea" — so we pass an override.
        names = public_names(mod, namespace_prefix="m_dir_fallback")
        assert "f" in names


# --- Module walker tests (on-disk fixture) ---------------------------------


class TestWalkModules:
    def test_walks_fixture_package(self, ondisk_package: str):
        mods = walk_modules(ondisk_package)
        assert mods == [ondisk_package, f"{ondisk_package}.sub"]


class TestIntrospectModule:
    def test_introspects_fixture_root(self, ondisk_package: str):
        result = introspect_module(
            ondisk_package, namespace_prefix=ondisk_package
        )
        assert "symbols" in result
        assert "Greeter" in result["symbols"]
        assert "Color" in result["symbols"]
        assert "helper" in result["symbols"]

    def test_class_entry_has_constructor_and_members(self, ondisk_package: str):
        result = introspect_module(
            ondisk_package, namespace_prefix=ondisk_package
        )
        greeter = result["symbols"]["Greeter"]
        assert greeter["kind"] == "class"
        assert greeter["defined_in"] == ondisk_package
        assert "constructor_signature" in greeter
        assert "members" in greeter and "hello" in greeter["members"]

    def test_function_entry_has_signature(self, ondisk_package: str):
        result = introspect_module(
            ondisk_package, namespace_prefix=ondisk_package
        )
        helper = result["symbols"]["helper"]
        assert helper["kind"] == "function"
        assert "signature" in helper
        assert helper["doc"] == "Add two ints."

    def test_enum_top_level_classified_as_enum(self, ondisk_package: str):
        result = introspect_module(
            ondisk_package, namespace_prefix=ondisk_package
        )
        color = result["symbols"]["Color"]
        assert color["kind"] == "enum"
        assert "RED" in color["members"]

    def test_first_paragraph_of_module_doc(self, ondisk_package: str):
        result = introspect_module(
            ondisk_package, namespace_prefix=ondisk_package
        )
        assert result["doc"] == "Fixture package docstring."

    def test_import_error_reported(self):
        result = introspect_module("definitely_not_a_module_xyz_9999")
        assert "import_error" in result
        assert "symbols" not in result


# --- Surface assembly + serialisation --------------------------------------


class TestBuildSurface:
    def test_against_fixture_package(self, ondisk_package: str):
        surface = build_surface(root_package=ondisk_package)
        assert "mellea_version" in surface
        assert "python_version" in surface
        assert "modules" in surface
        assert "canonical_definitions" in surface
        assert "surface_fingerprint" in surface

        assert ondisk_package in surface["modules"]
        assert f"{ondisk_package}.sub" in surface["modules"]

        # Canonical definitions should contain Greeter pointing at the root.
        assert surface["canonical_definitions"].get("Greeter") == ondisk_package
        # And `nested` pointing at the sub.
        assert (
            surface["canonical_definitions"].get("nested")
            == f"{ondisk_package}.sub"
        )

    def test_fingerprint_is_sha256(self, ondisk_package: str):
        surface = build_surface(root_package=ondisk_package)
        fp = surface["surface_fingerprint"]
        assert isinstance(fp, str)
        assert len(fp) == 64
        int(fp, 16)  # hex parseable


class TestCanonicalDefinitions:
    def test_picks_first_defining_module(self):
        modules = {
            "mellea.a": {"symbols": {"Foo": {"defined_in": "mellea.core.foo"}}},
            "mellea.b": {"symbols": {"Foo": {"defined_in": "mellea.core.foo"}}},
        }
        canonical = _build_canonical_definitions(modules)  # type: ignore[arg-type]
        assert canonical == {"Foo": "mellea.core.foo"}

    def test_skips_non_namespace_defining_modules(self):
        modules = {
            "mellea.a": {"symbols": {"Foo": {"defined_in": "third_party.x"}}},
        }
        canonical = _build_canonical_definitions(modules)  # type: ignore[arg-type]
        assert canonical == {}

    def test_custom_namespace_prefix(self):
        modules = {
            "pkg.a": {"symbols": {"Foo": {"defined_in": "pkg.core.foo"}}},
        }
        canonical = _build_canonical_definitions(modules, namespace_prefix="pkg")  # type: ignore[arg-type]
        assert canonical == {"Foo": "pkg.core.foo"}


class TestFingerprint:
    def test_deterministic(self):
        a = {"x": 1, "y": [1, 2, 3]}
        b = {"y": [1, 2, 3], "x": 1}
        assert _fingerprint(a) == _fingerprint(b)

    def test_change_detected(self):
        a = {"x": 1}
        b = {"x": 2}
        assert _fingerprint(a) != _fingerprint(b)


class TestSerialiseAndLoad:
    def test_roundtrip(self, tmp_path: Path, ondisk_package: str):
        surface = build_surface(root_package=ondisk_package)
        out_path = tmp_path / "surface.json"
        write_surface(surface, out_path)
        loaded = load_surface(out_path)
        assert loaded == surface

    def test_byte_stable(self, ondisk_package: str):
        s1 = build_surface(root_package=ondisk_package)
        s2 = build_surface(root_package=ondisk_package)
        assert serialise_surface(s1) == serialise_surface(s2)
        assert s1["surface_fingerprint"] == s2["surface_fingerprint"]

    def test_serialise_has_indent_and_trailing_newline(self, ondisk_package: str):
        surface = build_surface(root_package=ondisk_package)
        text = serialise_surface(surface)
        assert text.endswith("\n")
        # Two-space indent => first nested entry starts with "  ".
        assert '\n  "' in text
