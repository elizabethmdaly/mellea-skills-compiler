"""Coherence audits for the ``runtime-defaults-bound`` static lint.

Three-place wiring (CLI → claude_directives writer → lint) makes this
audit important: any one of the three drifting silently is a hard
failure for users.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_runtime_defaults_bound
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "runtime-defaults-bound"


def _make_pkg(
    tmp: Path,
    *,
    config_py: str | None,
    directive: dict | None,
) -> Path:
    pkg = tmp / "pkg_mellea"
    pkg.mkdir()
    (pkg / "intermediate").mkdir()
    if config_py is not None:
        (pkg / "config.py").write_text(config_py, encoding="utf-8")
    if directive is not None:
        (pkg / "intermediate" / "runtime_directive.json").write_text(
            json.dumps(directive), encoding="utf-8"
        )
    return pkg


def test_directive_covers_c8_backend_rule():
    """C-DIRECTIVE-COVERS-C8: the directive doc must mention the C8
    backend rule by name and explain it.
    """
    rule = get_rule(_RULE_ID)
    doc = (
        Path(__file__).resolve().parents[3] / rule["directive"]["doc"]
    ).read_text(encoding="utf-8")
    # Both anchors: the rule name AND the runtime-defaults file reference.
    assert "C8" in doc, (
        f"C-DIRECTIVE-COVERS-C8 failed: directive doc lacks the 'C8' "
        f"anchor. The lint references it but the doc doesn't."
    )
    assert "runtime_defaults" in doc or "BACKEND" in doc, (
        f"C-DIRECTIVE-COVERS-C8 failed: directive doc lacks a mention "
        f"of `runtime_defaults.json` or `BACKEND` — the LLM has no "
        f"explanation of what to actually do."
    )


def test_matching_config_passes():
    """C-CONFIG-MATCHES-PASSES: matching BACKEND/MODEL_ID values pass."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            config_py='BACKEND = "ollama"\nMODEL_ID = "granite4.1:8b"\n',
            directive={"backend": "ollama", "model_id": "granite4.1:8b"},
        )
        result = lint_runtime_defaults_bound(pkg)
        assert result.verdict == "pass", (
            f"C-CONFIG-MATCHES-PASSES failed: matching values were "
            f"rejected by the lint. Failures: "
            f"{[f.message for f in result.failures]}"
        )


def test_mismatch_fails_with_clear_message():
    """C-CONFIG-MISMATCH-FAILS: divergence between config.py and the
    directive must fail with a precise actual-vs-expected message."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            config_py='BACKEND = "openai"\nMODEL_ID = "granite4.1:8b"\n',
            directive={"backend": "ollama", "model_id": "granite4.1:8b"},
        )
        result = lint_runtime_defaults_bound(pkg)
        assert result.verdict == "fail", (
            "C-CONFIG-MISMATCH-FAILS failed: openai-vs-ollama drift "
            "was not flagged."
        )
        msg = result.failures[0].message
        assert "openai" in msg and "ollama" in msg, (
            f"C-CONFIG-MISMATCH-FAILS failed: failure message must "
            f"name BOTH the actual (openai) and expected (ollama) "
            f"values. Got: {msg!r}"
        )


def test_missing_directive_skips():
    """C-MISSING-DIRECTIVE-SKIPS: absent runtime_directive.json → skip."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            config_py='BACKEND = "ollama"\nMODEL_ID = "granite4.1:8b"\n',
            directive=None,
        )
        result = lint_runtime_defaults_bound(pkg)
        assert result.verdict == "skipped", (
            f"C-MISSING-DIRECTIVE-SKIPS failed: expected skip, got "
            f"{result.verdict}. The directive doc states this case "
            f"must skip rather than fail."
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
