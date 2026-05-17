"""Tests for the ``format=Schema`` thunk/parse split in emit_call_node_stmts.

Closes the renderer-side gap that the ``instruct-result-parse-before-access``
lint catches post-render: every ``m.instruct(..., format=Schema)`` call whose
result is referenced downstream must emit a two-statement form so the binding
visible to subsequent nodes is the parsed Pydantic model, not the raw thunk::

    <id>_thunk = session.instruct(..., format=Schema)
    <id> = Schema.model_validate_json(<id>_thunk.value)

Unreferenced results, primitive ``format=`` arguments, and ``format``-less
calls keep the prior single-statement emission for byte-stable backward
compatibility with the v0.1/v0.2 goldens.
"""

from __future__ import annotations

import ast

from mellea_skills_compiler.renderer.core import RenderContext, render_descriptor
from mellea_skills_compiler.renderer.nodes import emit_call_node_stmts


def _make_surface() -> dict:
    """Minimal in-memory surface with MelleaSession, OpenAIBackend, Requirement."""
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


def _unparse_stmts(stmts: list[ast.stmt]) -> str:
    return "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in stmts)


def _instruct_node(
    node_id: str = "step",
    *,
    with_schema: bool = True,
    format_value: object | None = None,
) -> dict:
    args: dict = {"description": {"value": "hi"}}
    if with_schema:
        args["format"] = {"schema_ref": "#/schemas/Finding"}
    elif format_value is not None:
        args["format"] = format_value
    return {
        "id": node_id,
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": args,
    }


# --- Direct emit_call_node_stmts cases -----------------------------------


def test_emit_parse_step_when_format_and_referenced() -> None:
    """``format={schema_ref}`` + downstream-ref → two-statement thunk/parse."""
    ctx = _make_ctx({"schemas": {"Finding": {"kind": "model", "fields": {}}}})
    ctx.referenced_ids = {"step"}
    stmts = emit_call_node_stmts(_instruct_node("step"), ctx)
    out = _unparse_stmts(stmts)
    assert len(stmts) == 2
    assert "step_thunk = session.instruct(description='hi', format=Finding)" in out
    assert "step = Finding.model_validate_json(step_thunk.value)" in out
    assert "Finding" in ctx.schema_uses
    assert "step" in ctx.pipeline_ids


def test_no_parse_step_when_no_format() -> None:
    """No format kwarg → single stmt, unchanged from emit_call_node."""
    ctx = _make_ctx({"schemas": {}})
    ctx.referenced_ids = {"step"}
    node = {
        "id": "step",
        "kind": "call",
        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        "bound_to": {"ref": "session"},
        "args": {"description": {"value": "hi"}},
    }
    stmts = emit_call_node_stmts(node, ctx)
    out = _unparse_stmts(stmts)
    assert len(stmts) == 1
    assert out == "step = session.instruct(description='hi')"
    assert "step_thunk" not in out


def test_no_parse_step_when_format_is_primitive() -> None:
    """``format={value: 'str'}`` lowers to a literal — not a schema_ref."""
    ctx = _make_ctx({"schemas": {}})
    ctx.referenced_ids = {"step"}
    node = _instruct_node("step", with_schema=False, format_value={"value": "str"})
    stmts = emit_call_node_stmts(node, ctx)
    out = _unparse_stmts(stmts)
    assert len(stmts) == 1
    assert "step_thunk" not in out
    assert "model_validate_json" not in out
    assert "step = session.instruct(" in out


def test_no_parse_step_when_result_unreferenced() -> None:
    """``format={schema_ref}`` but no downstream ref → keep inline wrap."""
    ctx = _make_ctx({"schemas": {"Finding": {"kind": "model", "fields": {}}}})
    # referenced_ids deliberately empty.
    stmts = emit_call_node_stmts(_instruct_node("step"), ctx)
    out = _unparse_stmts(stmts)
    assert len(stmts) == 1
    assert out == (
        "step = Finding.model_validate_json("
        "session.instruct(description='hi', format=Finding).value)"
    )
    assert "step_thunk" not in out


def test_parse_step_for_nested_models() -> None:
    """A schema whose declared fields nest other models still triggers the
    two-statement form. The renderer only cares about the format schema_ref
    name; nested-field shape introspection is out of scope here."""
    schemas = {
        "Inner": {"kind": "model", "fields": {"x": {"type": "int"}}},
        "Outer": {"kind": "model", "fields": {"inner": {"type": "Inner"}}},
    }
    ctx = _make_ctx({"schemas": schemas})
    ctx.referenced_ids = {"outer_step"}
    node = _instruct_node("outer_step")
    node["args"]["format"] = {"schema_ref": "#/schemas/Outer"}
    stmts = emit_call_node_stmts(node, ctx)
    out = _unparse_stmts(stmts)
    assert "outer_step_thunk = session.instruct(description='hi', format=Outer)" in out
    assert "outer_step = Outer.model_validate_json(outer_step_thunk.value)" in out


def test_downstream_ref_uses_parsed_value() -> None:
    """When a downstream node carries ``{"ref": "<id>"}``, the rendered
    expression must reference the parsed variable (``<id>``), not the
    raw thunk (``<id>_thunk``)."""
    surface = _make_surface()
    descriptor = {
        "descriptor_version": "0.2",
        "inputs": [{"name": "diff", "schema": {"kind": "str"}}],
        "outputs": [
            {"name": "out", "schema": {"kind": "model_ref", "ref": "#/schemas/Finding"}}
        ],
        "schemas": {"Finding": {"kind": "model", "fields": {"summary": {"type": "str"}}}},
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"value": "x"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "step",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"value": "hi"},
                    "format": {"schema_ref": "#/schemas/Finding"},
                },
            },
            {
                "id": "req",
                "kind": "call",
                "symbol": "mellea.stdlib.requirements.Requirement",
                "args": {"description": {"ref": "step", "select": "summary"}},
            },
        ],
    }
    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    assert "step_thunk = session.instruct(description='hi', format=Finding)" in src
    assert "step = Finding.model_validate_json(step_thunk.value)" in src
    assert "req = Requirement(description=step.summary)" in src
    assert "step_thunk.summary" not in src


# --- End-to-end against the lint -----------------------------------------


def test_against_failing_skill_pattern_passes_lint(tmp_path) -> None:
    """Synthesize a descriptor in the shape of nil-contract-analysis-style
    skills: two ``m.instruct(..., format=Schema)`` calls where the first
    binding is read by a template inside the second. Render the package,
    run ``lint_instruct_result_parse_before_access`` against the output,
    confirm verdict=pass."""
    from mellea_skills_compiler.compile.lints import (
        lint_instruct_result_parse_before_access,
    )

    surface = _make_surface()
    descriptor = {
        "descriptor_version": "0.2",
        "inputs": [{"name": "diff", "schema": {"kind": "str"}}],
        "outputs": [
            {"name": "report", "schema": {"kind": "model_ref", "ref": "#/schemas/Report"}}
        ],
        "schemas": {
            "TriageResult": {"kind": "model", "fields": {"items": {"type": "list[str]"}}},
            "Report": {"kind": "model", "fields": {"verdict": {"type": "str"}}},
        },
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"value": "x"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "triage",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"value": "do triage"},
                    "format": {"schema_ref": "#/schemas/TriageResult"},
                },
            },
            {
                "id": "report",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    # Downstream-reference to ``triage`` via a select. The
                    # pre-scan walks all dicts so this counts.
                    "description": {"ref": "triage", "select": "items"},
                    "format": {"schema_ref": "#/schemas/Report"},
                },
            },
        ],
    }

    result = render_descriptor(descriptor, surface)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")
    (pkg / "slots.py").write_text("", encoding="utf-8")
    (pkg / "constrained_slots.py").write_text("", encoding="utf-8")

    # Sanity: both calls split into the two-statement form.
    src = result.pipeline_py
    assert "triage_thunk = session.instruct(" in src
    assert "triage = TriageResult.model_validate_json(triage_thunk.value)" in src
    assert "report_thunk = session.instruct(" in src
    assert "report = Report.model_validate_json(report_thunk.value)" in src

    lint_result = lint_instruct_result_parse_before_access(pkg)
    assert lint_result.verdict == "pass", (
        f"lint failed:\n{src}\nfailures={lint_result.failures}"
    )


def test_unreferenced_format_call_still_passes_lint(tmp_path) -> None:
    """An unreferenced ``format=Schema`` call keeps the inline wrap, which
    binds no raw thunk Name — so the lint must also pass there
    (regression check on backward-compat path)."""
    from mellea_skills_compiler.compile.lints import (
        lint_instruct_result_parse_before_access,
    )

    surface = _make_surface()
    descriptor = {
        "descriptor_version": "0.2",
        "inputs": [{"name": "diff", "schema": {"kind": "str"}}],
        "outputs": [
            {"name": "out", "schema": {"kind": "model_ref", "ref": "#/schemas/Finding"}}
        ],
        "schemas": {"Finding": {"kind": "model", "fields": {"summary": {"type": "str"}}}},
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"value": "x"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "step",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"value": "hi"},
                    "format": {"schema_ref": "#/schemas/Finding"},
                },
            }
        ],
    }
    # ``step`` is the function's return value (last call id with format=) → it
    # IS referenced by the return statement. So this descriptor exercises the
    # two-statement form for the return-capture id rule.
    result = render_descriptor(descriptor, surface)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")
    (pkg / "slots.py").write_text("", encoding="utf-8")
    (pkg / "constrained_slots.py").write_text("", encoding="utf-8")
    lint_result = lint_instruct_result_parse_before_access(pkg)
    assert lint_result.verdict == "pass"


# --- _collect_downstream_referenced_ids unit checks -----------------------


def test_collect_referenced_ids_walks_nested_refs() -> None:
    """The ref pre-scan must recurse into args, operator bodies, and dicts."""
    from mellea_skills_compiler.renderer.core import (
        _collect_downstream_referenced_ids,
    )

    descriptor = {
        "outputs": [{"name": "out"}],
        "pipeline": [
            {"id": "a", "kind": "call", "symbol": "x"},
            {
                "id": "loop",
                "kind": "composition",
                "operator": "map",
                "over": {"ref": "a"},
                "body": [
                    {
                        "id": "inner",
                        "kind": "call",
                        "symbol": "y",
                        "args": {"x": {"ref": "loop"}},
                    }
                ],
            },
        ],
    }
    refs = _collect_downstream_referenced_ids(descriptor)
    assert "a" in refs  # referenced by loop.over
    assert "loop" in refs  # referenced by inner.args.x
    # Plus the last-call-id fallback (no captures declared, no output name match):
    assert "inner" in refs


def test_collect_referenced_ids_handles_captures() -> None:
    """If a node declares ``captures: {slot: 'outputs/<name>'}``, that node id
    must be in the referenced set (the return statement reads it)."""
    from mellea_skills_compiler.renderer.core import (
        _collect_downstream_referenced_ids,
    )

    descriptor = {
        "outputs": [{"name": "report"}],
        "pipeline": [
            {
                "id": "build_report",
                "kind": "call",
                "symbol": "x",
                "captures": {"result": "outputs/report"},
            },
            {"id": "side_effect", "kind": "call", "symbol": "y"},
        ],
    }
    refs = _collect_downstream_referenced_ids(descriptor)
    assert "build_report" in refs
    # The non-capture last call is NOT the return target.
    assert "side_effect" not in refs
