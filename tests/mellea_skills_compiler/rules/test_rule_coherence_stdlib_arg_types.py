"""Coherence audits for the ``stdlib-arg-types`` static lint.

Catches the runtime-AttributeError class observed in real compiles:
  * tech-contract (2026-05-19): `grounding_context=context` where
    `context: NegotiationContext` (a pydantic model).
  * gpai-code-of-practice (2026-05-20): same `'str' object has no
    attribute 'items'` class — a string ending up where a dict is
    expected inside Mellea's instruct chain.

Narrow MVP scope: `grounding_context=` kwarg on session-method calls
(`instruct`/`chat`/`act` family). The audit guards both the catch-the-
bug direction AND the don't-over-fire direction (ambiguous names,
missing kwarg, dict-typed param).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_stdlib_arg_types
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "stdlib-arg-types"


def _make_pkg(tmp: Path, code: str) -> Path:
    pkg = tmp / "pkg_mellea"
    pkg.mkdir()
    (pkg / "intermediate").mkdir()
    (pkg / "pipeline.py").write_text(code, encoding="utf-8")
    return pkg


def test_string_literal_grounding_context_fails():
    """C-STRING-LITERAL-GROUNDING-CONTEXT-FAILS: a literal string passed
    to grounding_context= must be flagged."""
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "def run_pipeline(x: str) -> str:\n"
        "    return m.instruct(description='hi', grounding_context='oops').value\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "fail", (
            "C-STRING-LITERAL-GROUNDING-CONTEXT-FAILS: lint accepted a "
            "string literal as grounding_context. The runtime would "
            "crash with `'str' object has no attribute 'items'`."
        )
        assert "non-dict literal" in result.failures[0].message


def test_dict_literal_grounding_context_passes():
    """C-DICT-LITERAL-GROUNDING-CONTEXT-PASSES: dict literal is the
    canonical shape and must not be flagged."""
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "def run_pipeline(x: str) -> str:\n"
        "    return m.instruct(description='hi', grounding_context={}).value\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "pass", (
            f"C-DICT-LITERAL-GROUNDING-CONTEXT-PASSES failed — lint "
            f"flagged the canonical empty-dict shape. Failures: "
            f"{[f.message for f in result.failures]}"
        )


def test_non_dict_param_grounding_context_fails():
    """C-NON-DICT-PARAM-GROUNDING-CONTEXT-FAILS: tech-contract pattern
    — pass a function parameter typed as a non-dict class.
    """
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "class NegotiationContext:\n"
        "    pass\n"
        "def run_pipeline(ctx: NegotiationContext) -> str:\n"
        "    return m.instruct(description='hi', grounding_context=ctx).value\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "fail", (
            "C-NON-DICT-PARAM-GROUNDING-CONTEXT-FAILS: lint accepted a "
            "non-dict-annotated parameter as grounding_context. The "
            "exact tech-contract pattern from 2026-05-19."
        )
        msg = result.failures[0].message
        assert "ctx" in msg


def test_dict_param_grounding_context_passes():
    """C-DICT-PARAM-GROUNDING-CONTEXT-PASSES: dict-annotated parameter
    must pass — symmetric guard against over-firing."""
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "def run_pipeline(ctx: dict[str, str]) -> str:\n"
        "    return m.instruct(description='hi', grounding_context=ctx).value\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "pass", (
            f"C-DICT-PARAM-GROUNDING-CONTEXT-PASSES failed — lint "
            f"flagged a dict-annotated parameter. Failures: "
            f"{[f.message for f in result.failures]}"
        )


def test_ambiguous_name_not_flagged():
    """C-AMBIGUOUS-NAME-NOT-FLAGGED: a Name without a visible annotation
    in the function scope is ambiguous — we can't statically prove the
    type. Must not flag, otherwise the lint false-positives on
    legitimate code where the dict is built outside the function."""
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "SOME_GLOBAL_DICT = {'key': 'value'}\n"
        "def run_pipeline(x: str) -> str:\n"
        # Name without annotation in the visible function scope.
        "    ctx = SOME_GLOBAL_DICT\n"
        "    return m.instruct(description='hi', grounding_context=ctx).value\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "pass", (
            f"C-AMBIGUOUS-NAME-NOT-FLAGGED failed — lint flagged an "
            f"ambiguous Name (no local annotation). Failures: "
            f"{[f.message for f in result.failures]}"
        )


def test_no_grounding_context_not_flagged():
    """C-NO-GROUNDING-CONTEXT-NOT-FLAGGED: calls that don't pass the
    kwarg at all must not be flagged by this lint."""
    code = (
        "from mellea.stdlib.session import start_session\n"
        "m = start_session(backend_name='ollama', model_id='granite4.1:8b')\n"
        "def run_pipeline(x: str) -> str:\n"
        "    return m.instruct(description='hi').value\n"  # no grounding_context
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_stdlib_arg_types(pkg)
        assert result.verdict == "pass", (
            f"C-NO-GROUNDING-CONTEXT-NOT-FLAGGED failed — lint fired "
            f"on a call that omits grounding_context entirely. That's "
            f"outside scope — a different lint may cover it later. "
            f"Failures: {[f.message for f in result.failures]}"
        )


def test_declared_severity_matches_central_table():
    """C-DECLARED-SEVERITY-MATCHES."""
    from mellea_skills_compiler.compile.lints import (
        LintSeverity,
        _LINT_SEVERITY,
    )
    rule = get_rule(_RULE_ID)
    declared = rule["validation"]["severity"]
    actual_enum = _LINT_SEVERITY.get(_RULE_ID)
    assert actual_enum is not None
    actual = (
        actual_enum.value if isinstance(actual_enum, LintSeverity) else actual_enum
    )
    assert declared == actual
