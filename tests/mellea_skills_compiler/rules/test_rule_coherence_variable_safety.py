"""Coherence audits for the ``variable-safety`` static lint.

This rule fired on the gdpr-breach-sentinel-oliver-schmidt-prietz
compile on 2026-05-19 — sub-check A (uninit-in-except). The check is
correct and the validate doc explains it; what's potentially missing
is preemptive guidance in generate.md so the LLM doesn't emit the
buggy pattern in the first place.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_variable_safety
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "variable-safety"


def _make_pkg(tmp: Path, code: str) -> Path:
    pkg = tmp / "pkg_mellea"
    pkg.mkdir()
    (pkg / "intermediate").mkdir()
    (pkg / "pipeline.py").write_text(code, encoding="utf-8")
    return pkg


def test_validate_doc_explains_subchecks():
    """C-VALIDATE-DOC-EXPLAINS-LINT: validate.md describes both
    sub-checks A and B with concrete examples.
    """
    rule = get_rule(_RULE_ID)
    doc = (
        Path(__file__).resolve().parents[3] / rule["directive"]["doc"]
    ).read_text(encoding="utf-8")
    assert "variable-safety" in doc, (
        f"C-VALIDATE-DOC-EXPLAINS-LINT failed: rule name absent from "
        f"{rule['directive']['doc']!r}."
    )
    assert "Sub-check A" in doc or "sub-check A" in doc, (
        "C-VALIDATE-DOC-EXPLAINS-LINT failed: validate.md does not "
        "explicitly describe sub-check A (uninit-in-except)."
    )


def test_generate_doc_covers_bind_before_try_pattern():
    """C-GENERATE-DOC-COVERS-PATTERN: generate.md should preemptively
    teach the bind-before-try pattern so the LLM doesn't emit the
    unsafe code that the lint then catches.

    Initially expected to fail — the gdpr-breach compile shows the
    LLM emits the unsafe pattern, suggesting generate.md doesn't
    cover it. The failure surfaces the directive ↔ implementation
    drift: validate.md describes the WHAT (catch this), but
    generate.md doesn't cover the HOW (write it this way).
    """
    repo_root = Path(__file__).resolve().parents[3]
    generate_md = repo_root / ".claude" / "commands" / "mellea-fy-generate.md"
    assert generate_md.is_file(), "mellea-fy-generate.md missing"
    text = generate_md.read_text(encoding="utf-8").lower()
    # Look for the canonical mention. Two acceptable signals:
    #   (a) explicit pattern: "initialise <name> before the try" /
    #       "bind ... before the try"
    #   (b) lint-by-name + bind hint: "variable-safety" with a nearby
    #       "before the try" or "init"
    pattern_mentioned = (
        "bind " in text and "before the try" in text
    ) or (
        "initialise" in text and "before" in text and "try" in text
    ) or (
        "variable-safety" in text and "before" in text
    )
    if not pattern_mentioned:
        # Honour the registry's `xfail_until` deadline (v0.3 discipline
        # — prevents the xfail from ossifying into a permanent "this is
        # expected to fail" annotation).
        from mellea_skills_compiler.rules import check_xfail_deadline

        deadline_reason = check_xfail_deadline(
            _RULE_ID, "C-GENERATE-DOC-COVERS-PATTERN"
        )
        if deadline_reason:
            pytest.xfail(deadline_reason)
        pytest.fail(
            "mellea-fy-generate.md does not preemptively teach the "
            "bind-before-try pattern (no mention of `variable-safety` "
            "with `before the try`, no explicit `bind X = None before "
            "try` phrasing). validate.md describes what the lint "
            "catches, but the LLM is generating from generate.md and "
            "doesn't get told the safe pattern up front — this is the "
            "drift that produced the gdpr-breach failure. The "
            "registry's xfail_until deadline has expired or is unset; "
            "add a Variable-safety patterns section to "
            "mellea-fy-generate.md per Task #43, or extend the "
            "registry deadline with explicit justification."
        )


def test_uninit_in_except_fails():
    """C-UNINIT-IN-EXCEPT-FAILS: the exact pattern from gdpr-breach."""
    code = (
        "def _safe_parse(thunk):\n"
        "    try:\n"
        "        raw = thunk.value\n"
        "        return raw\n"
        "    except Exception:\n"
        "        return raw  # raw might be unbound here\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_variable_safety(pkg)
        assert result.verdict == "fail", (
            "C-UNINIT-IN-EXCEPT-FAILS failed: the canonical bug "
            "pattern (assign-in-try, reference-in-except) was not "
            "flagged."
        )


def test_bind_before_try_passes():
    """C-BIND-BEFORE-TRY-PASSES: the correct pattern (bind None
    before the try) must pass the lint."""
    code = (
        "def _safe_parse(thunk):\n"
        "    raw = None\n"
        "    try:\n"
        "        raw = thunk.value\n"
        "        return raw\n"
        "    except Exception:\n"
        "        return raw  # safe — bound to None above\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_variable_safety(pkg)
        assert result.verdict == "pass", (
            f"C-BIND-BEFORE-TRY-PASSES failed: the correct bind-"
            f"before-try pattern was flagged. Failures: "
            f"{[f.message for f in result.failures]}"
        )


def test_cond_bind_uncond_return_fails():
    """C-COND-BIND-UNCOND-RETURN-FAILS: the exact gdpr-breach-sentinel
    pattern — `assemble_dashboard` bound only in branch 1, returned
    unconditionally — must fail sub-check C.
    """
    code = (
        "def run_pipeline(trigger):\n"
        "    if trigger == 'breach':\n"
        "        assemble_dashboard = build_dashboard()\n"
        "        assessment_branch = assemble_dashboard\n"
        "    elif trigger == 'query':\n"
        "        gdpr_query_response = build_query_response()\n"
        "        assessment_branch = gdpr_query_response\n"
        "    elif trigger == 'not_applicable':\n"
        "        not_applicable_response = build_not_applicable()\n"
        "        assessment_branch = not_applicable_response\n"
        "    else:\n"
        "        raise RuntimeError('unexpected trigger')\n"
        "    return assemble_dashboard\n"  # only branch 1 binds; crash on 2/3
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_variable_safety(pkg)
        assert result.verdict == "fail", (
            "C-COND-BIND-UNCOND-RETURN-FAILS: lint accepted the gdpr-"
            "breach pattern. Sub-check C is missing or under-firing."
        )
        msg = result.failures[0].message
        assert "assemble_dashboard" in msg, (
            f"Expected the failure message to name the conditionally-"
            f"bound `assemble_dashboard`. Got: {msg!r}"
        )


def test_all_branches_bind_passes():
    """C-ALL-BRANCHES-BIND-PASSES: when every falling-through branch of
    an if/elif/else binds the returned name, sub-check C must NOT
    fire. Guards against the new check becoming over-eager.
    """
    code = (
        "def f(cond):\n"
        "    if cond == 'a':\n"
        "        x = 1\n"
        "    elif cond == 'b':\n"
        "        x = 2\n"
        "    else:\n"
        "        x = 3\n"
        "    return x\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_variable_safety(pkg)
        assert result.verdict == "pass", (
            f"C-ALL-BRANCHES-BIND-PASSES: lint flagged a return where "
            f"every branch binds the name. Sub-check C over-fired. "
            f"Failures: {[f.message for f in result.failures]}"
        )


def test_raising_branch_ignored():
    """C-RAISING-BRANCH-IGNORED: this is the EXACT structure of the
    gdpr-breach pipeline, but with the unifying name `assessment_branch`
    returned (the one-line fix). The else-branch raises rather than
    binds; the other 3 branches all bind `assessment_branch` — so
    `return assessment_branch` is safe and must pass.
    """
    code = (
        "def run_pipeline(trigger):\n"
        "    if trigger == 'breach':\n"
        "        assessment_branch = 'breach_result'\n"
        "    elif trigger == 'query':\n"
        "        assessment_branch = 'query_result'\n"
        "    elif trigger == 'not_applicable':\n"
        "        assessment_branch = 'na_result'\n"
        "    else:\n"
        "        raise RuntimeError('unexpected trigger')\n"
        "    return assessment_branch\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = _make_pkg(Path(tmp), code)
        result = lint_variable_safety(pkg)
        assert result.verdict == "pass", (
            f"C-RAISING-BRANCH-IGNORED: lint flagged a return where 3 "
            f"of 4 branches bind the name and the 4th raises. A raising "
            f"branch doesn't reach the return, so the binding is "
            f"guaranteed. Sub-check C is mishandling raising branches. "
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
