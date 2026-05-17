"""Tests for :mod:`mellea_skills_compiler.renderer.schemas`.

Strategy: most tests parse the emitted source with :mod:`ast` and inspect the
resulting tree (structural assertions), and a handful additionally execute
the module in a fresh namespace to confirm the Pydantic + Enum imports
actually wire up (runtime assertions). The latter exercises a fixture from
``melleafy-handoff/`` so the test surface tracks real descriptor evolution.
"""

from __future__ import annotations

import ast
import json
import py_compile
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.schemas import render_schemas

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTRY_DESCRIPTOR = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "sentry-find-bugs.descriptor.json"
)


# --- helpers ----------------------------------------------------------------


def _parse(src: str) -> ast.Module:
    return ast.parse(src)


def _classdefs(src: str) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in _parse(src).body if isinstance(n, ast.ClassDef)}


def _compile_to_tempfile(src: str) -> Path:
    """Write source to a temp file and py_compile it; return the source path."""
    fd = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    fd.write(src)
    fd.close()
    py_compile.compile(fd.name, doraise=True)
    return Path(fd.name)


def _exec(src: str) -> dict[str, object]:
    """Execute source in a fresh namespace; return the resulting globals."""
    ns: dict[str, object] = {}
    exec(compile(src, "<rendered>", "exec"), ns)
    return ns


# --- imports / empty input --------------------------------------------------


def test_render_empty_schemas_block_emits_only_imports():
    src = render_schemas({})
    mod = _parse(src)
    # Three ImportFrom nodes: __future__, enum, pydantic.
    imports = [n for n in mod.body if isinstance(n, ast.ImportFrom)]
    assert [imp.module for imp in imports] == ["__future__", "enum", "pydantic"]
    assert [n for n in mod.body if isinstance(n, ast.ClassDef)] == []


def test_emitted_source_compiles_cleanly():
    """Real fixture round-trip — must parse + py_compile + import."""
    descriptor = json.loads(SENTRY_DESCRIPTOR.read_text())
    src = render_schemas(descriptor["schemas"])
    # py_compile catches syntax errors that ast.parse might not.
    _compile_to_tempfile(src)
    # Execute it too — confirms imports resolve and class bodies are legal.
    ns = _exec(src)
    assert "FindingsReport" in ns
    assert "Severity" in ns


# --- enum emission ----------------------------------------------------------


def test_enum_class_emitted_as_str_enum_subclass():
    src = render_schemas(
        {"Color": {"kind": "enum", "members": ["red", "green", "blue"]}}
    )
    cls = _classdefs(src)["Color"]
    base_names = [b.id for b in cls.bases]
    assert base_names == ["str", "Enum"]


def test_enum_members_use_screaming_snake_case_idents_and_preserve_values():
    """Pydantic str-enum convention: SCREAMING_SNAKE ident, raw value preserved."""
    src = render_schemas(
        {"Severity": {"kind": "enum", "members": ["Critical", "High", "Medium", "Low"]}}
    )
    cls = _classdefs(src)["Severity"]
    idents = [t.id for stmt in cls.body for t in getattr(stmt, "targets", [])]
    assert idents == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    values = [stmt.value.value for stmt in cls.body if isinstance(stmt, ast.Assign)]
    assert values == ["Critical", "High", "Medium", "Low"]


def test_enum_non_identifier_member_is_sanitised():
    """Members like ``"Authorization/IDOR"`` produce a legal Python ident."""
    src = render_schemas(
        {"Cat": {"kind": "enum", "members": ["Authorization/IDOR", "Race conditions"]}}
    )
    cls = _classdefs(src)["Cat"]
    idents = [stmt.targets[0].id for stmt in cls.body]
    assert idents == ["AUTHORIZATION_IDOR", "RACE_CONDITIONS"]
    # But the *value* should be untouched.
    values = [stmt.value.value for stmt in cls.body]
    assert values == ["Authorization/IDOR", "Race conditions"]


def test_enum_member_collision_raises():
    with pytest.raises(RendererError, match="collide"):
        render_schemas(
            {"Bad": {"kind": "enum", "members": ["foo bar", "foo/bar"]}}
        )


def test_enum_empty_members_raises():
    with pytest.raises(RendererError, match="no members"):
        render_schemas({"Empty": {"kind": "enum", "members": []}})


# --- model emission ---------------------------------------------------------


def test_model_class_emitted_as_basemodel_subclass():
    src = render_schemas(
        {"M": {"kind": "model", "fields": {"x": {"type": "str"}}}}
    )
    cls = _classdefs(src)["M"]
    assert [b.id for b in cls.bases] == ["BaseModel"]


def test_model_field_primitive_types_emit_as_bare_names():
    src = render_schemas(
        {
            "M": {
                "kind": "model",
                "fields": {
                    "a": {"type": "str"},
                    "b": {"type": "int"},
                    "c": {"type": "float"},
                    "d": {"type": "bool"},
                },
            }
        }
    )
    cls = _classdefs(src)["M"]
    annot_names = [stmt.annotation.id for stmt in cls.body]
    assert annot_names == ["str", "int", "float", "bool"]
    # None of these have defaults.
    assert all(stmt.value is None for stmt in cls.body)


def test_list_field_uses_default_factory_list():
    src = render_schemas(
        {"M": {"kind": "model", "fields": {"xs": {"type": "list[str]"}}}}
    )
    cls = _classdefs(src)["M"]
    field_stmt = cls.body[0]
    # Annotation is list[str]
    assert isinstance(field_stmt.annotation, ast.Subscript)
    assert field_stmt.annotation.value.id == "list"
    assert field_stmt.annotation.slice.id == "str"
    # Default is Field(default_factory=list)
    assert isinstance(field_stmt.value, ast.Call)
    assert field_stmt.value.func.id == "Field"
    kw = field_stmt.value.keywords[0]
    assert kw.arg == "default_factory"
    assert kw.value.id == "list"


def test_dict_field_uses_default_factory_dict():
    src = render_schemas(
        {"M": {"kind": "model", "fields": {"m": {"type": "dict[str, int]"}}}}
    )
    cls = _classdefs(src)["M"]
    field_stmt = cls.body[0]
    # Annotation is dict[str, int]
    assert isinstance(field_stmt.annotation, ast.Subscript)
    assert field_stmt.annotation.value.id == "dict"
    assert isinstance(field_stmt.annotation.slice, ast.Tuple)
    assert [e.id for e in field_stmt.annotation.slice.elts] == ["str", "int"]
    # default_factory=dict
    kw = field_stmt.value.keywords[0]
    assert kw.arg == "default_factory"
    assert kw.value.id == "dict"


def test_enum_ref_field_resolves_to_bare_class_name():
    src = render_schemas(
        {
            "Severity": {"kind": "enum", "members": ["high", "low"]},
            "M": {
                "kind": "model",
                "fields": {
                    "sev": {"type": "enum_ref", "ref": "#/schemas/Severity"},
                },
            },
        }
    )
    m = _classdefs(src)["M"]
    assert m.body[0].annotation.id == "Severity"
    # No default.
    assert m.body[0].value is None


def test_model_ref_field_resolves_to_bare_class_name():
    src = render_schemas(
        {
            "Inner": {"kind": "model", "fields": {"x": {"type": "str"}}},
            "Outer": {
                "kind": "model",
                "fields": {
                    "inner": {"type": "model_ref", "ref": "#/schemas/Inner"},
                },
            },
        }
    )
    outer = _classdefs(src)["Outer"]
    assert outer.body[0].annotation.id == "Inner"


def test_optional_field_emits_union_with_none_and_default_none():
    src = render_schemas(
        {
            "M": {
                "kind": "model",
                "fields": {"name": {"type": "str", "optional": True}},
            }
        }
    )
    field_stmt = _classdefs(src)["M"].body[0]
    # Annotation is `str | None`
    assert isinstance(field_stmt.annotation, ast.BinOp)
    assert isinstance(field_stmt.annotation.op, ast.BitOr)
    assert field_stmt.annotation.left.id == "str"
    assert field_stmt.annotation.right.value is None
    # Default is None constant
    assert isinstance(field_stmt.value, ast.Constant)
    assert field_stmt.value.value is None


def test_optional_enum_ref_field():
    src = render_schemas(
        {
            "Sev": {"kind": "enum", "members": ["a", "b"]},
            "M": {
                "kind": "model",
                "fields": {
                    "s": {"type": "enum_ref", "ref": "#/schemas/Sev", "optional": True}
                },
            },
        }
    )
    f = _classdefs(src)["M"].body[0]
    assert isinstance(f.annotation, ast.BinOp)
    assert f.annotation.left.id == "Sev"
    assert f.value.value is None


def test_empty_model_emits_pass():
    src = render_schemas({"M": {"kind": "model", "fields": {}}})
    cls = _classdefs(src)["M"]
    assert len(cls.body) == 1
    assert isinstance(cls.body[0], ast.Pass)


def test_nested_list_of_model_ref_by_bare_name():
    """``list[IssueReport]`` appears in real descriptors — must resolve."""
    src = render_schemas(
        {
            "IssueReport": {"kind": "model", "fields": {"x": {"type": "str"}}},
            "Report": {
                "kind": "model",
                "fields": {"issues": {"type": "list[IssueReport]"}},
            },
        }
    )
    f = _classdefs(src)["Report"].body[0]
    assert isinstance(f.annotation, ast.Subscript)
    assert f.annotation.value.id == "list"
    assert f.annotation.slice.id == "IssueReport"


# --- ordering: enums before models ------------------------------------------


def test_enums_emitted_before_models():
    src = render_schemas(
        {
            "ModelFirst": {"kind": "model", "fields": {"x": {"type": "str"}}},
            "EnumLast": {"kind": "enum", "members": ["a"]},
        }
    )
    classes = [n.name for n in _parse(src).body if isinstance(n, ast.ClassDef)]
    assert classes.index("EnumLast") < classes.index("ModelFirst")


# --- determinism ------------------------------------------------------------


def test_render_is_byte_identical_on_repeat():
    descriptor = json.loads(SENTRY_DESCRIPTOR.read_text())
    a = render_schemas(descriptor["schemas"])
    b = render_schemas(descriptor["schemas"])
    assert a == b


# --- error paths -------------------------------------------------------------


def test_unknown_schema_kind_raises():
    with pytest.raises(RendererError, match="unknown kind"):
        render_schemas({"X": {"kind": "interface", "members": []}})


def test_unrecognised_type_raises():
    with pytest.raises(RendererError, match="unrecognised"):
        render_schemas(
            {"M": {"kind": "model", "fields": {"x": {"type": "complex<int>"}}}}
        )


def test_enum_ref_without_ref_raises():
    with pytest.raises(RendererError, match="enum_ref"):
        render_schemas(
            {"M": {"kind": "model", "fields": {"x": {"type": "enum_ref"}}}}
        )


def test_non_dict_schemas_block_raises():
    with pytest.raises(RendererError, match="must be a dict"):
        render_schemas([])  # type: ignore[arg-type]
