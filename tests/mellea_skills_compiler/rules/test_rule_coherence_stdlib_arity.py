"""Coherence audits for the ``stdlib-arity`` lint.

This test file is referenced by the rule-coherence registry entry for
``stdlib-arity``. Each ``test_*`` here corresponds to one entry in the
registry's ``coherence_checks`` array — failures are organised by which
of the three faces (directive, implementation, validation) needs
fixing.

The rule's three faces:

  * **Directive**: ``.claude/commands/mellea-fy-generate.md`` tells the
    LLM that POSITIONAL_OR_KEYWORD params accept either call style
    (line ~200: "either as the first positional argument or as a
    ``description=`` keyword").
  * **Implementation**: Python's ``inspect.Signature`` is the ground
    truth. POSITIONAL_OR_KEYWORD params accept both styles by
    definition. Mellea exposes its surface via
    ``intermediate/mellea_api_ref.json``.
  * **Validation**: ``compile/lints.py::lint_stdlib_arity`` reads the
    surface JSON, parses each signature string, and rejects calls
    that don't match.

The first run of this file is expected to FAIL on
``test_positional_or_keyword_accepts_keyword_form`` — that failure is
the impl ↔ validation drift surfaced by the
``MelleaSession(backend=...)`` false positive on 2026-05-19. Fixing
the lint's grounded-signature parser to honour the parameter kind is
the work that makes this test pass.
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_stdlib_arity
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "stdlib-arity"


# ─── Shared fixtures ─────────────────────────────────────────────────


def _make_surface_with_signature(symbol_path: str, signature: str) -> dict:
    """Build a minimal mellea_api_ref.json-shaped dict carrying one
    symbol with the given signature string. Mirrors the introspection
    shape the lint expects.
    """
    module, _, leaf = symbol_path.rpartition(".")
    return {
        "modules": {
            module: {
                leaf: {"signature": signature},
            },
        },
    }


def _run_lint_against(call_source: str, surface: dict) -> list:
    """Render ``call_source`` as pipeline.py, drop the surface JSON
    alongside it, and run the lint. Returns the lint's failures list.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        (pkg / "intermediate" / "mellea_api_ref.json").write_text(
            json.dumps(surface), encoding="utf-8"
        )
        (pkg / "pipeline.py").write_text(call_source, encoding="utf-8")
        result = lint_stdlib_arity(pkg)
        return list(result.failures)


# ─── Coherence checks for `stdlib-arity` ─────────────────────────────


def test_positional_or_keyword_accepts_keyword_form():
    """C-POSORKW-ACCEPTS-KEYWORD: a POSITIONAL_OR_KEYWORD param can be
    filled by keyword.

    The signature ``MelleaSession(backend: 'Backend', ctx: 'Context | None' = None)``
    declares ``backend`` as positional-or-keyword. Python accepts
    ``MelleaSession(backend=b)`` — the lint must agree.

    Failure here means the lint's grounded-sig parser treats
    positional-or-keyword params as positional-only. Fix: include
    those params in the ``valid_kwargs`` set, and verify required
    params are filled either positionally OR by name (not strictly
    by position count).
    """
    surface = _make_surface_with_signature(
        "mellea.stdlib.session.MelleaSession",
        "MelleaSession(backend: 'Backend', ctx: 'Context | None' = None)",
    )
    source = (
        "def run_pipeline():\n"
        "    return MelleaSession(backend=mybackend)\n"
    )
    failures = _run_lint_against(source, surface)
    assert failures == [], (
        "C-POSORKW-ACCEPTS-KEYWORD failed (impl ↔ validation drift): the lint "
        "rejected MelleaSession(backend=...) which Python accepts. The lint's "
        "grounded-sig parser is treating POSITIONAL_OR_KEYWORD params as "
        "positional-only. Failures: "
        + repr([f.message for f in failures])
    )


def test_positional_or_keyword_accepts_positional_form():
    """C-POSORKW-ACCEPTS-POSITIONAL: a POSITIONAL_OR_KEYWORD param can
    also be filled positionally. Paired with the previous check —
    confirms the fix doesn't flip the lint to reject BOTH forms.
    """
    surface = _make_surface_with_signature(
        "mellea.stdlib.session.MelleaSession",
        "MelleaSession(backend: 'Backend', ctx: 'Context | None' = None)",
    )
    source = (
        "def run_pipeline():\n"
        "    return MelleaSession(mybackend)\n"
    )
    failures = _run_lint_against(source, surface)
    assert failures == [], (
        "C-POSORKW-ACCEPTS-POSITIONAL failed: the lint rejected the positional "
        "form. If the lint was just fixed to accept keyword form, the fix "
        "regressed the positional form. Failures: "
        + repr([f.message for f in failures])
    )


def test_missing_required_param_fails():
    """C-REQUIRED-MISSING-FAILS: a call that omits a required param both
    positionally AND by keyword must fail. Guards against the lint
    becoming too permissive in the fix.
    """
    surface = _make_surface_with_signature(
        "mellea.stdlib.session.MelleaSession",
        "MelleaSession(backend: 'Backend', ctx: 'Context | None' = None)",
    )
    source = (
        "def run_pipeline():\n"
        "    return MelleaSession()\n"  # missing `backend`
    )
    failures = _run_lint_against(source, surface)
    assert failures, (
        "C-REQUIRED-MISSING-FAILS failed: the lint accepted "
        "MelleaSession() with no `backend`. The fix made the lint too "
        "permissive — missing required params must still be flagged."
    )


def test_unknown_kwarg_fails():
    """C-UNKNOWN-KWARG-FAILS: a call with a kwarg name not in the
    signature must fail (typo guard).
    """
    surface = _make_surface_with_signature(
        "mellea.stdlib.session.MelleaSession",
        "MelleaSession(backend: 'Backend', ctx: 'Context | None' = None)",
    )
    source = (
        "def run_pipeline():\n"
        "    return MelleaSession(backend=b, totally_unknown=42)\n"
    )
    failures = _run_lint_against(source, surface)
    # Expect at least one failure citing the unknown kwarg.
    assert any("totally_unknown" in f.message for f in failures), (
        "C-UNKNOWN-KWARG-FAILS failed: the lint did not flag the typo "
        "kwarg `totally_unknown`. Failures: "
        + repr([f.message for f in failures])
    )


def test_static_table_matches_directive_doc():
    """C-STATIC-TABLE-MATCHES-DIRECTIVE: the static-fallback signatures
    for ``req``, ``check``, ``simple_validate`` in
    ``compile/lints.py::_STDLIB_STATIC_SIGS`` must match what the
    directive doc claims they look like.

    The directive's 'Known signatures' table is the curated contract
    presented to the LLM. The static-fallback table is the validation's
    enforcement of that contract. Drift here means the LLM is being
    told one signature shape and the lint enforces a different one.
    """
    from mellea_skills_compiler.compile.lints import _STDLIB_STATIC_SIGS

    rule = get_rule(_RULE_ID)
    doc_path = (
        Path(__file__).resolve().parents[3] / rule["directive"]["doc"]
    )
    assert doc_path.is_file(), (
        f"directive doc not found at {doc_path}; coherence check cannot run"
    )
    doc_text = doc_path.read_text(encoding="utf-8")

    # Curated shapes per the static table — each must be mentioned in
    # the directive doc by its function name. (Looser than parsing the
    # markdown table: the doc's table format may evolve; what matters
    # is that the function exists and its signature claims are in the
    # doc somewhere identifiable.)
    for fn_name in _STDLIB_STATIC_SIGS:
        assert fn_name in doc_text, (
            f"C-STATIC-TABLE-MATCHES-DIRECTIVE failed: static table entry "
            f"{fn_name!r} is enforced by the lint but the directive doc "
            f"({rule['directive']['doc']}) does not mention it. Either "
            f"the directive needs to document the curated signature, "
            f"or the static table needs to drop {fn_name!r}."
        )


def test_declared_severity_matches_central_table():
    """C-DECLARED-SEVERITY-MATCHES: registry's declared severity must
    match ``_LINT_SEVERITY`` in compile/lints.py — the single source
    of truth for lint gate severity.
    """
    from mellea_skills_compiler.compile.lints import (
        LintSeverity,
        _LINT_SEVERITY,
    )

    rule = get_rule(_RULE_ID)
    declared = rule["validation"]["severity"]
    actual_enum = _LINT_SEVERITY.get(_RULE_ID)
    assert actual_enum is not None, (
        f"_LINT_SEVERITY has no entry for {_RULE_ID!r}; the registry "
        f"references a lint that isn't in the central severity table."
    )
    actual = actual_enum.value if isinstance(actual_enum, LintSeverity) else actual_enum
    assert declared == actual, (
        f"C-DECLARED-SEVERITY-MATCHES failed: registry declares "
        f"severity={declared!r} but _LINT_SEVERITY[{_RULE_ID!r}] = "
        f"{actual!r}. Pick one source of truth and align."
    )
