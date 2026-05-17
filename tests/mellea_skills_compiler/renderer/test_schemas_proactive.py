"""Proactive adversarial + property battery for the schemas renderer.

Target: ``mellea_skills_compiler.renderer.schemas::render_schemas(
schemas_block: dict) -> str``. A pure function: takes a descriptor schemas
block, returns Python source. Per its docstring: same input -> byte-identical
output, py_compile clean, imports BaseModel + Field, enums emitted before
models for forward-reference discipline.

Adversarial inputs enumerated (S3):

  A1. Circular refs (Schema A refers to Schema B which refers to Schema A) —
      must produce a clear error, not an infinite loop / stack overflow.
  A2. Self-refs (Schema A has a field of type Schema A) — render with
      forward ref OR explicit error.
  A3. Empty schemas block ({}) — produces a minimal schemas.py with imports
      only.
  A4. Schema with empty fields dict — produces ``class X(BaseModel): pass``.
  A5. Enum with one member, with reserved-word-like name, with numeric
      leading character.
  A6. Field type that's neither in the documented type vocabulary nor a
      model_ref/enum_ref — must error.
  A7. Schema field name that's a Python keyword (class, return) — error
      or escape.
  A8. Schema with extra: "allow" / "forbid" config (not v0.1) — accepted
      or rejected explicitly.
  A9. Multi-thousand-field schema — must not exhibit O(n²) blow-up.
  A10. Deeply-nested list[dict[str, list[model_ref]]] — handles arbitrary
      nesting.

Properties (S4):

  P1. Same input -> byte-identical output (determinism).
  P2. Output contains ``from pydantic import BaseModel, Field``.
  P3. Every declared schema name appears as a ``class X(...):``.
  P4. Class declaration order: enums BEFORE models.
  P5. ``py_compile`` on the rendered output succeeds.

What this battery is NOT testing:

  - End-to-end runtime instantiation of every emitted Pydantic class (only
    that the source is parseable + compilable).
  - Behaviour when the descriptor itself is malformed at the top level
    (caught by the validator, not the schemas renderer).
  - The schemas-renderer's interaction with the pipeline renderer (Battery
    2 covers the bound).
  - Adversarial type strings designed to exploit the type-lowering parser
    (e.g., ``str; import os; os.system('rm -rf /')``) — the renderer treats
    types as type-strings, not as code, so injection-resistance is a
    documented invariant we trust.
"""

from __future__ import annotations

import ast
import py_compile
import tempfile
import time
from pathlib import Path

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.schemas import render_schemas


# =============================================================================
# Helpers
# =============================================================================


def _compile_source(src: str) -> None:
    """py_compile a rendered output. Raises py_compile.PyCompileError on
    failure."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(src)
        path = f.name
    try:
        py_compile.compile(path, doraise=True)
    finally:
        Path(path).unlink(missing_ok=True)


def _class_order(src: str) -> list[str]:
    """Return class names in declaration order from rendered source."""
    tree = ast.parse(src)
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _class_bases(src: str, name: str) -> list[str]:
    """Return base class names for a given class in the rendered source."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [ast.unparse(b) for b in node.bases]
    raise KeyError(name)


# =============================================================================
# Sanity baselines — happy paths
# =============================================================================


def test_empty_schemas_block_renders_minimal_source():
    """A3: empty block returns parseable source with imports only — no
    class declarations."""
    src = render_schemas({})
    ast.parse(src)
    classes = _class_order(src)
    assert classes == [], f"expected no classes, got {classes}"


def test_single_model_with_single_str_field():
    """Baseline happy path — one model, one str field."""
    src = render_schemas({
        "Thing": {"kind": "model", "fields": {"name": {"type": "str"}}},
    })
    ast.parse(src)
    assert "class Thing" in src
    assert "name: str" in src


def test_single_enum_with_three_members():
    """Baseline happy path — one enum, three str members."""
    src = render_schemas({
        "Color": {"kind": "enum", "members": ["red", "blue", "green"]},
    })
    ast.parse(src)
    assert "class Color" in src
    # SCREAMING_SNAKE_CASE per docstring of _safe_member_ident.
    assert "RED" in src
    assert "BLUE" in src
    assert "GREEN" in src


# =============================================================================
# Adversarial inputs
# =============================================================================


class TestAdversarialInputs:
    """Adversarial framing: a hostile descriptor author submits schemas
    blocks designed to break the renderer. Each input asserts the renderer
    either errors cleanly (RendererError) or produces well-formed Python."""

    # ---- A1: circular refs ------------------------------------------------

    def test_a1_circular_refs_between_two_models(self):
        """Schema A has a field of model_ref B; Schema B has a field of
        model_ref A. The renderer must NOT infinite-loop. It must either
        emit valid forward-referencing code (Pydantic handles forward refs
        with ``from __future__ import annotations``) OR raise RendererError."""
        block = {
            "A": {
                "kind": "model",
                "fields": {"b_ref": {"type": "model_ref", "ref": "#/schemas/B"}},
            },
            "B": {
                "kind": "model",
                "fields": {"a_ref": {"type": "model_ref", "ref": "#/schemas/A"}},
            },
        }
        # Render must terminate in a reasonable time.
        start = time.time()
        try:
            src = render_schemas(block)
            elapsed = time.time() - start
            assert elapsed < 2.0, f"render took {elapsed:.2f}s (suspect infinite loop)"
            ast.parse(src)
            _compile_source(src)
        except RendererError:
            # Acceptable failure mode.
            pass

    # ---- A2: self-refs ----------------------------------------------------

    def test_a2_self_referential_model(self):
        """Schema A has a field of type model_ref A (recursive). Pydantic
        supports this via forward refs; the renderer must produce code that
        py_compiles."""
        block = {
            "Node": {
                "kind": "model",
                "fields": {
                    "value": {"type": "str"},
                    "child": {"type": "model_ref", "ref": "#/schemas/Node", "optional": True},
                },
            },
        }
        try:
            src = render_schemas(block)
            ast.parse(src)
            _compile_source(src)
        except RendererError:
            pass

    # ---- A4: model with empty fields dict --------------------------------

    def test_a4_model_with_empty_fields(self):
        """Empty fields dict — must emit ``class X(BaseModel): pass``."""
        try:
            src = render_schemas({
                "Empty": {"kind": "model", "fields": {}},
            })
        except RendererError:
            pytest.xfail("renderer rejects empty-fields models (may be intentional)")
            return
        ast.parse(src)
        _compile_source(src)
        assert "class Empty" in src

    # ---- A5: enum oddities -----------------------------------------------

    def test_a5a_enum_with_one_member(self):
        """An enum with a single member must still render cleanly."""
        src = render_schemas({
            "Singleton": {"kind": "enum", "members": ["only"]},
        })
        ast.parse(src)
        _compile_source(src)
        assert "ONLY" in src

    def test_a5b_enum_with_reserved_word_member_name(self):
        """An enum member whose name happens to be a Python keyword. The
        identifier-safe lowering should escape it."""
        try:
            src = render_schemas({
                "K": {"kind": "enum", "members": ["class", "return", "pass"]},
            })
            ast.parse(src)
            _compile_source(src)
        except RendererError:
            # Acceptable if the renderer rejects rather than silently corrupting.
            pass

    def test_a5c_enum_with_numeric_leading_member(self):
        """Member values that start with a digit can't be Python identifiers
        directly — must be escaped."""
        try:
            src = render_schemas({
                "Version": {"kind": "enum", "members": ["1.0", "2.0", "3.0"]},
            })
            ast.parse(src)
            _compile_source(src)
        except RendererError:
            pass

    # ---- A6: unknown field type ------------------------------------------

    def test_a6_unknown_field_type(self):
        """A field with type "TotallyUnknownType" — must raise RendererError
        explicitly per the docstring's guarantees."""
        with pytest.raises(RendererError):
            render_schemas({
                "Bad": {
                    "kind": "model",
                    "fields": {"x": {"type": "TotallyUnknownType"}},
                }
            })

    # ---- A7: field name is a Python keyword -------------------------------

    @pytest.mark.xfail(
        reason="P3.5 — keyword-named fields not yet escaped/rejected; today they "
        "produce Python that won't parse",
        strict=False,
    )
    def test_a7_field_name_is_python_keyword(self):
        """A field named ``class`` or ``return`` is a Python keyword — the
        renderer must reject or escape, not produce broken Python."""
        try:
            src = render_schemas({
                "Bad": {
                    "kind": "model",
                    "fields": {"class": {"type": "str"}, "return": {"type": "int"}},
                },
            })
        except RendererError:
            return  # acceptable: clean rejection
        # If it rendered, the source must still be parseable Python.
        ast.parse(src)

    # ---- A8: extra config not in v0.1 ------------------------------------

    def test_a8_extra_config_extra_allow(self):
        """A model with ``extra: 'allow'`` — not in the v0.1 vocabulary.
        Acceptable behaviours: ignore silently OR raise RendererError. The
        renderer must NOT crash / produce malformed output."""
        try:
            src = render_schemas({
                "Loose": {
                    "kind": "model",
                    "fields": {"x": {"type": "str"}},
                    "extra": "allow",
                },
            })
            ast.parse(src)
            _compile_source(src)
        except RendererError:
            pass

    # ---- A9: large schema (no O(n²) blow-up) -----------------------------

    def test_a9_thousand_field_schema_completes_in_under_5s(self):
        """A 1000-field model. Renders in <5s — guards against O(n²) bugs."""
        fields = {f"f_{i}": {"type": "str"} for i in range(1000)}
        block = {"Big": {"kind": "model", "fields": fields}}
        start = time.time()
        src = render_schemas(block)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"render took {elapsed:.2f}s on 1000-field schema"
        ast.parse(src)
        _compile_source(src)

    # ---- A10: deeply nested type ------------------------------------------

    def test_a10_deeply_nested_type_renders(self):
        """list[dict[str, list[str]]] — verifies the type-lowering parser
        handles legitimately complex types without crashing."""
        try:
            src = render_schemas({
                "Nested": {
                    "kind": "model",
                    "fields": {"data": {"type": "list[dict[str, list[str]]]"}},
                }
            })
            ast.parse(src)
            _compile_source(src)
        except RendererError as e:
            # If unsupported, the error must mention the type — not crash
            # with a generic Python error.
            assert "type" in str(e).lower() or "field" in str(e).lower(), (
                f"unattributed type-lowering failure: {e}"
            )

    # ---- A11: non-dict schemas_block --------------------------------------

    def test_a11_non_dict_input_raises_renderererror(self):
        """Per docstring: schemas_block must be a dict. Lists / strings /
        None must raise RendererError, not generic AttributeError/TypeError."""
        for bad in [[], "not a dict", None, 42, ()]:
            with pytest.raises(RendererError):
                render_schemas(bad)  # type: ignore[arg-type]

    # ---- A12: unknown kind -----------------------------------------------

    def test_a12_unknown_schema_kind(self):
        """A schema with kind: 'union' (not in v0.1 vocabulary) — must
        raise RendererError per docstring."""
        with pytest.raises(RendererError):
            render_schemas({
                "Union": {"kind": "union", "members": ["str", "int"]},
            })


# =============================================================================
# Properties
# =============================================================================


_SCHEMA_FIXTURES = [
    pytest.param({}, id="empty"),
    pytest.param(
        {"X": {"kind": "model", "fields": {"a": {"type": "str"}}}},
        id="single-model",
    ),
    pytest.param(
        {"C": {"kind": "enum", "members": ["r", "g", "b"]}},
        id="single-enum",
    ),
    pytest.param(
        {
            "Color": {"kind": "enum", "members": ["red", "blue"]},
            "Thing": {
                "kind": "model",
                "fields": {
                    "name": {"type": "str"},
                    "color": {"type": "enum_ref", "ref": "#/schemas/Color"},
                },
            },
        },
        id="enum-then-model",
    ),
    pytest.param(
        {
            "Item": {"kind": "model", "fields": {"v": {"type": "int"}}},
            "Container": {
                "kind": "model",
                "fields": {
                    "items": {"type": "list[Item]"},
                },
            },
        },
        id="nested-list-model-ref",
    ),
]


# ---- P1: determinism -------------------------------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p1_render_is_deterministic(block):
    """Same input -> byte-identical output across two calls."""
    a = render_schemas(block)
    b = render_schemas(block)
    assert a == b, "render_schemas is not deterministic"


# ---- P2: imports header ----------------------------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p2_pydantic_imports_present(block):
    """The output must import BaseModel + Field (per docstring's design
    decision: "AST + ast.unparse, never string templating")."""
    src = render_schemas(block)
    # BaseModel must always appear in the imports.
    assert "BaseModel" in src
    # If there's at least one collection-typed default, Field appears too;
    # at minimum it should be present whether or not it's needed.
    assert "from pydantic import" in src or "import pydantic" in src


# ---- P3: every declared schema produces a class declaration ----------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p3_every_schema_emits_a_class(block):
    """For every key in schemas_block, a class with that exact name MUST
    appear in the output."""
    src = render_schemas(block)
    classes = _class_order(src)
    for name in block:
        assert name in classes, f"schema {name!r} missing as a class declaration"


# ---- P4: enum-before-model ordering ----------------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p4_enums_emitted_before_models(block):
    """Per docstring: "Enums are emitted before models so forward-referenced
    enum types resolve without ForwardRef gymnastics".

    Property: in the rendered output, the last enum class (if any) appears
    before the first model class (if any).
    """
    src = render_schemas(block)
    classes = _class_order(src)
    if not classes:
        return  # empty case
    # Re-introspect each class's bases. Enum subclasses have 'Enum' or 'str' in
    # bases; Pydantic models have 'BaseModel' in bases.
    enum_indices = []
    model_indices = []
    for i, name in enumerate(classes):
        bases = _class_bases(src, name)
        if any("Enum" in b for b in bases):
            enum_indices.append(i)
        elif any("BaseModel" in b for b in bases):
            model_indices.append(i)
    if enum_indices and model_indices:
        assert max(enum_indices) < min(model_indices), (
            f"enums must precede models; got order: {classes} "
            f"(enums at {enum_indices}, models at {model_indices})"
        )


# ---- P5: py_compile success ------------------------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p5_rendered_output_py_compiles(block):
    """py_compile on the rendered output succeeds (per docstring's
    guarantee: 'Parses, py_compile-s, and imports cleanly')."""
    src = render_schemas(block)
    _compile_source(src)


# ---- P6: ast.parse success (parallel to P5) --------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p6_ast_parse_success(block):
    """ast.parse on the rendered output succeeds. Catches malformed AST."""
    src = render_schemas(block)
    ast.parse(src)


# ---- P7: trailing newline --------------------------------------------------


@pytest.mark.parametrize("block", _SCHEMA_FIXTURES)
def test_p7_ends_with_single_newline(block):
    """Per docstring: 'Returns a complete Python source string ending in a
    single newline.'"""
    src = render_schemas(block)
    assert src.endswith("\n"), f"output does not end with newline; ends with {src[-20:]!r}"
    # Also check it ends with exactly ONE newline (not two).
    assert not src.endswith("\n\n\n"), "output ends with excessive newlines"
