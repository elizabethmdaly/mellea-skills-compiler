"""Coherence audits for the ``pipeline-entry-canonical`` static lint.

Three-face check for: `pipeline.py` must define `run_pipeline`, and
`melleafy.json:entry_signature` (when present) must start with
`run_pipeline(`. The lint hard-fails when `pipeline.py` is absent.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_pipeline_entry_canonical
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "pipeline-entry-canonical"


def _make_pkg(tmp: Path, files: dict[str, str]) -> Path:
    pkg = tmp / "pkg_mellea"
    pkg.mkdir()
    (pkg / "intermediate").mkdir()
    for relpath, content in files.items():
        target = pkg / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return pkg


def test_directive_documents_rule_3_2():
    """C-DIRECTIVE-DOCUMENTS-RULE-3-2: the directive doc must mention
    `Rule 3-2` AND `run_pipeline` to establish the canonical-entry contract.
    """
    rule = get_rule(_RULE_ID)
    doc = (
        Path(__file__).resolve().parents[3] / rule["directive"]["doc"]
    ).read_text(encoding="utf-8")
    # Both anchors must be present — the rule has a stable name AND a
    # mention of the entry-point name that the LLM needs to emit.
    assert "Rule 3-2" in doc and "run_pipeline" in doc, (
        f"C-DIRECTIVE-DOCUMENTS-RULE-3-2 failed: directive doc "
        f"{rule['directive']['doc']!r} is missing either the 'Rule 3-2' "
        f"anchor or the canonical name 'run_pipeline'. The LLM has no "
        f"unambiguous reference for what this lint enforces."
    )


def test_missing_run_pipeline_fails():
    """C-MISSING-RUN_PIPELINE-FAILS: `pipeline.py` without `run_pipeline`
    must fail the lint."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            {
                "pipeline.py": "def run_phase_2():\n    return 'wrong'\n",
                "melleafy.json": json.dumps(
                    {"entry_signature": "run_phase_2(x: str) -> str"}
                ),
            },
        )
        result = lint_pipeline_entry_canonical(pkg)
        assert result.verdict == "fail", (
            "C-MISSING-RUN_PIPELINE-FAILS failed: a pipeline.py "
            "missing `run_pipeline` was accepted by the lint."
        )


def test_absent_pipeline_py_fails():
    """C-MISSING-PIPELINE-PY-FAILS: an absent pipeline.py must fail
    (not skip). Directive explicitly mandates this for catching
    descriptor-mode renderer rejections and similar."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            {"melleafy.json": json.dumps({"entry_signature": "run_pipeline()"})},
        )
        result = lint_pipeline_entry_canonical(pkg)
        assert result.verdict == "fail", (
            "C-MISSING-PIPELINE-PY-FAILS failed: an absent pipeline.py "
            "must surface as a lint failure, not a silent skip. The "
            "directive's empirical justification: this is how "
            "descriptor-mode renderer rejections surface as lint "
            "failures rather than silent misses."
        )


def test_wrong_manifest_entry_fails():
    """C-WRONG-MANIFEST-ENTRY-FAILS: when melleafy.json:entry_signature
    starts with the wrong name, the lint must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(
            Path(tmp),
            {
                "pipeline.py": "def run_pipeline(x: str) -> str:\n    return x\n",
                "melleafy.json": json.dumps(
                    {"entry_signature": "run_assessment(x: str) -> str"}
                ),
            },
        )
        result = lint_pipeline_entry_canonical(pkg)
        assert result.verdict == "fail", (
            "C-WRONG-MANIFEST-ENTRY-FAILS failed: melleafy.json named "
            "`run_assessment` as the entry but the lint did not catch "
            "the manifest-side drift. The contract is two-sided."
        )


def test_declared_severity_matches_central_table():
    """C-DECLARED-SEVERITY-MATCHES: registry severity must match
    _LINT_SEVERITY[pipeline-entry-canonical]."""
    from mellea_skills_compiler.compile.lints import (
        LintSeverity,
        _LINT_SEVERITY,
    )
    rule = get_rule(_RULE_ID)
    declared = rule["validation"]["severity"]
    actual_enum = _LINT_SEVERITY.get(_RULE_ID)
    assert actual_enum is not None, (
        f"_LINT_SEVERITY has no entry for {_RULE_ID!r}"
    )
    actual = (
        actual_enum.value if isinstance(actual_enum, LintSeverity) else actual_enum
    )
    assert declared == actual, (
        f"C-DECLARED-SEVERITY-MATCHES failed: registry says {declared!r} "
        f"but _LINT_SEVERITY[{_RULE_ID!r}] = {actual!r}."
    )
