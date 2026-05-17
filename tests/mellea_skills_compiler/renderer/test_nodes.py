"""Tests for signature-driven node emission: argument lowering, symbol resolution, state setup."""

from __future__ import annotations

import ast

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.core import RenderContext
from mellea_skills_compiler.renderer.nodes import (
    build_call,
    emit_call_node,
    emit_state,
    lower_arg,
    lower_template,
    lower_type,
    resolve_symbol,
    split_symbol,
    symbol_resolves,
)


# --- Surface fixtures (minimal) ------------------------------------------


def _make_surface() -> dict:
    """A small in-memory surface; mirrors the real surface_0.5.0 shape."""
    return {
        "mellea_version": "test",
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {
                    "MelleaSession": {
                        "kind": "class",
                        "constructor_signature": "(self, backend, ctx=None)",
                        "members": {
                            "instruct": {
                                "kind": "method",
                                "signature": "(self, description, *, format=None)",
                            },
                            "achat": {"kind": "method", "signature": "(self, content)"},
                        },
                    },
                },
            },
            "mellea.backends.openai": {
                "symbols": {
                    "OpenAIBackend": {
                        "kind": "class",
                        "constructor_signature": "(self, model_id)",
                        "members": {},
                    },
                },
            },
            "mellea.stdlib.requirements": {
                "symbols": {
                    "Requirement": {
                        "kind": "class",
                        "constructor_signature": "(self, description)",
                        "members": {},
                    },
                },
            },
        },
    }


def _make_ctx(descriptor: dict | None = None) -> RenderContext:
    return RenderContext(descriptor=descriptor or {}, surface=_make_surface())


# --- split_symbol / resolve_symbol ---------------------------------------


def test_split_symbol_class_method() -> None:
    mod, attrs = split_symbol("mellea.stdlib.session.MelleaSession.instruct", _make_surface())
    assert mod == "mellea.stdlib.session"
    assert attrs == ["MelleaSession", "instruct"]


def test_split_symbol_class_constructor() -> None:
    mod, attrs = split_symbol("mellea.backends.openai.OpenAIBackend", _make_surface())
    assert mod == "mellea.backends.openai"
    assert attrs == ["OpenAIBackend"]


def test_split_symbol_unknown_module_raises() -> None:
    with pytest.raises(RendererError, match="no module prefix"):
        split_symbol("not.a.real.module.Symbol", _make_surface())


def test_resolve_symbol_method_returns_member_dict() -> None:
    res = resolve_symbol("mellea.stdlib.session.MelleaSession.instruct", _make_surface())
    assert res["kind"] == "method"


def test_resolve_symbol_unknown_member_raises() -> None:
    with pytest.raises(RendererError, match="member 'nope' not found"):
        resolve_symbol("mellea.stdlib.session.MelleaSession.nope", _make_surface())


def test_resolve_symbol_unknown_top_level_symbol_raises() -> None:
    with pytest.raises(RendererError, match="symbol 'Mystery' not in module"):
        resolve_symbol("mellea.stdlib.session.Mystery", _make_surface())


def test_symbol_resolves_returns_false_on_unknown() -> None:
    assert symbol_resolves("mellea.stdlib.session.Mystery", _make_surface()) is False


def test_symbol_resolves_returns_true_on_known() -> None:
    assert symbol_resolves("mellea.stdlib.session.MelleaSession", _make_surface()) is True


# --- lower_type -----------------------------------------------------------


def _unparse(node: ast.expr) -> str:
    """Wrap a bare expr in a module for unparse."""
    expr = ast.Expression(body=node)
    ast.fix_missing_locations(expr)
    return ast.unparse(expr)


def test_lower_type_primitives() -> None:
    assert _unparse(lower_type("str")) == "str"
    assert _unparse(lower_type("int")) == "int"
    assert _unparse(lower_type("bool")) == "bool"


def test_lower_type_list() -> None:
    assert _unparse(lower_type("list[str]")) == "list[str]"


def test_lower_type_dict() -> None:
    assert _unparse(lower_type("dict[str, int]")) == "dict[str, int]"


def test_lower_type_nested() -> None:
    assert _unparse(lower_type("list[dict[str, int]]")) == "list[dict[str, int]]"


def test_lower_type_schema_ref_passthrough() -> None:
    # Bare identifier is treated as a schema reference (Python name).
    assert _unparse(lower_type("FindingsReport")) == "FindingsReport"


def test_lower_type_malformed_dict_raises() -> None:
    with pytest.raises(RendererError):
        lower_type("dict[str]")


# --- lower_template -------------------------------------------------------


def test_lower_template_plain_string() -> None:
    # `ast.unparse` of a JoinedStr with one Constant produces an f-string.
    assert _unparse(lower_template("hello world")) == "f'hello world'"


def test_lower_template_var_interpolation() -> None:
    assert _unparse(lower_template("hi {name}")) == "f'hi {name}'"


def test_lower_template_attribute_interpolation() -> None:
    assert _unparse(lower_template("file {x.path}")) == "f'file {x.path}'"


def test_lower_template_escaped_braces() -> None:
    out = _unparse(lower_template("a {{ b }} c"))
    # Doubled braces become literal { and }; the result is an f-string with
    # constants. The unparsed form will contain doubled braces.
    assert "{{" in out and "}}" in out


def test_lower_template_unclosed_brace_raises() -> None:
    with pytest.raises(RendererError, match="unclosed brace"):
        lower_template("hi {name")


# --- lower_arg ------------------------------------------------------------


def test_lower_arg_value_string() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"value": "hello"}, ctx)) == "'hello'"


def test_lower_arg_value_int() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"value": 42}, ctx)) == "42"


def test_lower_arg_value_list_of_primitives() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"value": [1, 2, 3]}, ctx)) == "[1, 2, 3]"


def test_lower_arg_value_dict_literal() -> None:
    ctx = _make_ctx()
    out = _unparse(lower_arg({"value": {"a": 1, "b": 2}}, ctx))
    assert out == "{'a': 1, 'b': 2}"


def test_lower_arg_template_becomes_fstring() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"template": "x={x}"}, ctx)) == "f'x={x}'"


def test_lower_arg_ref_plain() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"ref": "session"}, ctx)) == "session"


def test_lower_arg_ref_with_select_dotted() -> None:
    ctx = _make_ctx()
    assert _unparse(lower_arg({"ref": "step", "select": "result.files"}, ctx)) == "step.result.files"


def test_lower_arg_schema_ref_extracts_name() -> None:
    ctx = _make_ctx({"schemas": {"Finding": {"kind": "model", "fields": {}}}})
    assert _unparse(lower_arg({"schema_ref": "#/schemas/Finding"}, ctx)) == "Finding"
    # And it registers the use:
    assert "Finding" in ctx.schema_uses


def test_lower_arg_env_emits_environ_lookup() -> None:
    ctx = _make_ctx()
    out = _unparse(lower_arg({"env": "MODEL_ID"}, ctx))
    assert out == "os.environ['MODEL_ID']"
    assert ctx.needs_os is True


def test_lower_arg_nested_symbol_builds_call() -> None:
    ctx = _make_ctx()
    out = _unparse(
        lower_arg(
            {
                "symbol": "mellea.stdlib.requirements.Requirement",
                "args": {"description": {"value": "must be valid"}},
            },
            ctx,
        )
    )
    assert out == "Requirement(description='must be valid')"
    assert "Requirement" in ctx.imports.get("mellea.stdlib.requirements", set())


def test_lower_arg_list_of_mixed_kinds() -> None:
    ctx = _make_ctx()
    out = _unparse(
        lower_arg(
            [
                {"value": "literal"},
                {"ref": "x"},
            ],
            ctx,
        )
    )
    assert out == "['literal', x]"


def test_lower_arg_unknown_shape_raises() -> None:
    ctx = _make_ctx()
    with pytest.raises(RendererError, match="unknown argument value shape"):
        lower_arg({"weirdo": "?"}, ctx)


# --- build_call -----------------------------------------------------------


def test_build_call_free_function_form_imports_top_symbol() -> None:
    ctx = _make_ctx()
    call = build_call(
        symbol="mellea.backends.openai.OpenAIBackend",
        bound_to=None,
        args={"model_id": {"env": "MODEL_ID"}},
        ctx=ctx,
    )
    out = _unparse(call)
    assert out == "OpenAIBackend(model_id=os.environ['MODEL_ID'])"
    assert "OpenAIBackend" in ctx.imports.get("mellea.backends.openai", set())


def test_build_call_bound_method_uses_bound_to_attr() -> None:
    ctx = _make_ctx()
    call = build_call(
        symbol="mellea.stdlib.session.MelleaSession.instruct",
        bound_to={"ref": "session"},
        args={"description": {"value": "hi"}},
        ctx=ctx,
    )
    out = _unparse(call)
    assert out == "session.instruct(description='hi')"
    # bound-method form does NOT import MelleaSession (it's used via the bound var)
    assert "MelleaSession" not in ctx.imports.get("mellea.stdlib.session", set())


def test_build_call_unknown_symbol_raises() -> None:
    ctx = _make_ctx()
    with pytest.raises(RendererError, match="symbol 'Mystery' not in module"):
        build_call(
            symbol="mellea.stdlib.session.Mystery",
            bound_to=None,
            args={},
            ctx=ctx,
        )


# --- emit_call_node + format=Schema wrap ----------------------------------


def test_emit_call_node_plain_no_format() -> None:
    ctx = _make_ctx({"schemas": {}})
    node = {
        "id": "step",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"value": "hi"}},
    }
    stmt = emit_call_node(node, ctx)
    ast.fix_missing_locations(stmt)
    out = ast.unparse(stmt)
    assert out == "step = session.instruct(description='hi')"
    assert "step" in ctx.pipeline_ids


def test_emit_call_node_with_format_schema_wraps_model_validate_json() -> None:
    ctx = _make_ctx({"schemas": {"Finding": {"kind": "model", "fields": {}}}})
    node = {
        "id": "step",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {
            "description": {"value": "hi"},
            "format": {"schema_ref": "#/schemas/Finding"},
        },
    }
    stmt = emit_call_node(node, ctx)
    ast.fix_missing_locations(stmt)
    out = ast.unparse(stmt)
    assert out == "step = Finding.model_validate_json(session.instruct(description='hi', format=Finding).value)"
    assert "Finding" in ctx.schema_uses


def test_emit_call_node_tags_source_map() -> None:
    """The emitted stmt must carry the descriptor-node id attribute."""
    from mellea_skills_compiler.renderer.source_map import SourceMap

    ctx = _make_ctx()
    node = {
        "id": "step",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"value": "hi"}},
    }
    stmt = emit_call_node(node, ctx, "pipeline[0]")
    assert SourceMap.get_tag(stmt) == "step"
    assert SourceMap.get_path(stmt) == "pipeline[0]"


# --- emit_state -----------------------------------------------------------


def test_emit_state_emits_module_level_assignments() -> None:
    ctx = _make_ctx()
    state = [
        {
            "id": "backend",
            "symbol": "mellea.backends.openai.OpenAIBackend",
            "args": {"model_id": {"env": "MODEL_ID"}},
        },
        {
            "id": "session",
            "symbol": "mellea.stdlib.session.MelleaSession",
            "args": {"backend": {"ref": "backend"}},
        },
    ]
    stmts = emit_state(state, ctx)
    assert len(stmts) == 2
    for s in stmts:
        ast.fix_missing_locations(s)
    assert ast.unparse(stmts[0]) == "backend = OpenAIBackend(model_id=os.environ['MODEL_ID'])"
    assert ast.unparse(stmts[1]) == "session = MelleaSession(backend=backend)"
    assert ctx.state_ids == {"backend", "session"}


def test_emit_state_empty_list_returns_empty() -> None:
    ctx = _make_ctx()
    assert emit_state([], ctx) == []
