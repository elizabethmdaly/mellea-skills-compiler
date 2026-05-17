"""Tests for the per-rule semantic checks.

These rules are the Python-level second pass that JSON Schema can't
express: symbol-against-surface resolution, ref/schema_ref resolution,
composition-operator required-field cross-check, and (v0.2 only)
``on_parse_failure.default`` shape match.
"""
from __future__ import annotations

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.semantic_rules import (
    R_FMT_DEFAULT,
    R_OPERATOR,
    R_REF,
    R_SCHEMA_REF,
    R_SYMBOL,
    _symbol_in_surface,
)


# A tiny in-memory surface that mimics the real surface JSON layout.
SURFACE = {
    "mellea_version": "0.5.0",
    "modules": {
        "mellea.stdlib.session": {
            "symbols": {
                "MelleaSession": {
                    "kind": "class",
                    "members": {
                        "instruct": {"kind": "method"},
                        "chat": {"kind": "method"},
                    },
                },
            }
        },
        "mellea.backends.openai": {
            "symbols": {
                "OpenAIBackend": {
                    "kind": "class",
                    "members": {"__init__": {"kind": "method"}},
                }
            }
        },
        "mellea.stdlib.requirements": {
            "symbols": {
                "Requirement": {
                    "kind": "class",
                    "members": {"__init__": {"kind": "method"}},
                }
            }
        },
    },
}


def _base_descriptor() -> dict:
    return {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [],
    }


# --- Symbol-in-surface unit -------------------------------------------------- #


def test_symbol_in_surface_class():
    assert _symbol_in_surface("mellea.stdlib.session.MelleaSession", SURFACE)


def test_symbol_in_surface_method():
    assert _symbol_in_surface("mellea.stdlib.session.MelleaSession.instruct", SURFACE)


def test_symbol_in_surface_unknown_module():
    assert not _symbol_in_surface("mellea.backends.nope.OpenAIBackend", SURFACE)


def test_symbol_in_surface_unknown_method():
    assert not _symbol_in_surface(
        "mellea.stdlib.session.MelleaSession.does_not_exist", SURFACE
    )


def test_symbol_in_surface_no_modules_dict():
    assert not _symbol_in_surface("any.thing", {})


# --- Symbol resolution end-to-end ------------------------------------------- #


def test_unknown_state_symbol_reports_error():
    desc = _base_descriptor()
    desc["state"][0]["symbol"] = "mellea.backends.nope.OpenAIBackend"
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    assert not report.valid
    sym_errors = [e for e in report.errors if e.rule == R_SYMBOL]
    assert sym_errors
    assert sym_errors[0].path == "/state/0/symbol"


def test_unknown_call_symbol_reports_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.fakemethod",
            "bound_to": {"ref": "session"},
            "args": {},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    sym_errors = [e for e in report.errors if e.rule == R_SYMBOL]
    assert sym_errors
    assert sym_errors[0].path == "/pipeline/0/symbol"


def test_known_symbols_pass():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"description": {"value": "hi"}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    assert report.valid, report.errors


def test_no_surface_skips_symbol_resolution():
    desc = _base_descriptor()
    desc["state"][0]["symbol"] = "nothing.like.this"
    report = validate(desc, schema_version="0.2", surface=None)
    sym_errors = [e for e in report.errors if e.rule == R_SYMBOL]
    assert not sym_errors


# --- Ref resolution --------------------------------------------------------- #


def test_unknown_ref_reports_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"prefix": {"ref": "nope_doesnt_exist"}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    ref_errors = [e for e in report.errors if e.rule == R_REF]
    assert ref_errors
    assert "nope_doesnt_exist" in ref_errors[0].message


def test_ref_to_input_is_allowed():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"description": {"template": "{doc}"}, "ctx": {"ref": "doc"}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    ref_errors = [e for e in report.errors if e.rule == R_REF]
    assert not ref_errors


def test_ref_to_state_id_is_allowed():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    assert report.valid


def test_forward_ref_is_not_allowed():
    """A ref must point at a *prior* node, not a later one."""
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "first",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"input": {"ref": "second"}},
        },
        {
            "id": "second",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {},
        },
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    ref_errors = [e for e in report.errors if e.rule == R_REF]
    assert ref_errors, "forward refs must be rejected"


def test_map_body_can_ref_item_id():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "extract",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {},
        },
        {
            "id": "m",
            "kind": "composition",
            "operator": "map",
            "over": {"ref": "extract"},
            "item_id": "file_path",
            "collect": "list",
            "body": [
                {
                    "id": "inner",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "bound_to": {"ref": "session"},
                    "args": {"description": {"template": "{file_path}"},
                              "ctx": {"ref": "file_path"}},
                }
            ],
        },
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    ref_errors = [e for e in report.errors if e.rule == R_REF]
    assert not ref_errors, ref_errors


def test_map_body_cannot_ref_unknown_outer():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "m",
            "kind": "composition",
            "operator": "map",
            "over": {"ref": "doc"},
            "item_id": "f",
            "collect": "list",
            "body": [
                {
                    "id": "inner",
                    "kind": "call",
                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                    "bound_to": {"ref": "session"},
                    "args": {"ctx": {"ref": "no_such_outer"}},
                }
            ],
        },
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    ref_errors = [e for e in report.errors if e.rule == R_REF]
    assert ref_errors


# --- Schema_ref resolution -------------------------------------------------- #


def test_undeclared_schema_ref_reports_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/DoesNotExist"}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    sr_errors = [e for e in report.errors if e.rule == R_SCHEMA_REF]
    assert sr_errors
    assert sr_errors[0].path == "/pipeline/0/args/format/schema_ref"


def test_declared_schema_ref_is_accepted():
    desc = _base_descriptor()
    desc["schemas"] = {
        "Foo": {"kind": "model", "fields": {"name": {"type": "str"}}}
    }
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/Foo"}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    sr_errors = [e for e in report.errors if e.rule == R_SCHEMA_REF]
    assert not sr_errors


# --- Composition required fields -------------------------------------------- #


def test_filter_missing_predicate_reports_operator_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "f",
            "kind": "composition",
            "operator": "filter",
            "over": {"ref": "doc"},
            "item_id": "x",
            # predicate missing
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    op_errors = [e for e in report.errors if e.rule == R_OPERATOR]
    assert any("predicate" in e.path for e in op_errors)


def test_branch_missing_cases_reports_operator_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "br",
            "kind": "composition",
            "operator": "branch",
            "on": {"ref": "doc"},
            # cases missing
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    op_errors = [e for e in report.errors if e.rule == R_OPERATOR]
    assert any("cases" in e.path for e in op_errors)


# --- on_parse_failure default-shape (v0.2 only) ----------------------------- #


def test_on_parse_failure_default_shape_ok():
    desc = _base_descriptor()
    desc["schemas"] = {
        "ModifiedFiles": {"kind": "model", "fields": {"files": {"type": "list[str]"}}}
    }
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/ModifiedFiles"}},
            "on_parse_failure": {"strategy": "default", "default": {"files": []}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    fd_errors = [e for e in report.errors if e.rule == R_FMT_DEFAULT]
    assert not fd_errors, fd_errors


def test_on_parse_failure_default_has_extra_fields():
    desc = _base_descriptor()
    desc["schemas"] = {
        "ModifiedFiles": {"kind": "model", "fields": {"files": {"type": "list[str]"}}}
    }
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/ModifiedFiles"}},
            "on_parse_failure": {
                "strategy": "default",
                "default": {"files": [], "garbage_field": 1},
            },
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    fd_errors = [e for e in report.errors if e.rule == R_FMT_DEFAULT]
    assert fd_errors


def test_on_parse_failure_default_without_format_is_an_error():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {},
            "on_parse_failure": {"strategy": "default", "default": {"x": 1}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    fd_errors = [e for e in report.errors if e.rule == R_FMT_DEFAULT]
    assert fd_errors


def test_on_parse_failure_default_missing_required_is_warning():
    desc = _base_descriptor()
    desc["schemas"] = {
        "ModifiedFiles": {
            "kind": "model",
            "fields": {"files": {"type": "list[str]"}, "name": {"type": "str"}},
        }
    }
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"format": {"schema_ref": "#/schemas/ModifiedFiles"}},
            "on_parse_failure": {"strategy": "default", "default": {"files": []}},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    warnings = [e for e in report.errors if e.severity == "warning"]
    assert warnings, "missing-required should be a warning, not blocking"
    # And no error-severity entries from R_FMT_DEFAULT.
    fd_errors = [
        e for e in report.errors if e.rule == R_FMT_DEFAULT and e.severity == "error"
    ]
    assert not fd_errors


def test_on_parse_failure_raise_strategy_passes_no_default_check():
    desc = _base_descriptor()
    desc["pipeline"] = [
        {
            "id": "c",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {},
            "on_parse_failure": {"strategy": "raise"},
        }
    ]
    report = validate(desc, schema_version="0.2", surface=SURFACE)
    fd_errors = [e for e in report.errors if e.rule == R_FMT_DEFAULT]
    assert not fd_errors


def test_v01_skips_on_parse_failure_check():
    """v0.1 doesn't allow on_parse_failure at all; the rule is v0.2-only."""
    desc = _base_descriptor()
    desc["descriptor_version"] = "0.1"
    # remove v0.2-specific things to focus the assertion
    report = validate(desc, schema_version="0.1", surface=SURFACE)
    fd_errors = [e for e in report.errors if e.rule == R_FMT_DEFAULT]
    assert not fd_errors
