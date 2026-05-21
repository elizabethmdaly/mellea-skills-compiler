"""Coherence audits for the ``r-sem-bundled-path-exists`` semantic rule.

Validates `bundled_resources[].source_dir` paths in the descriptor:
relative-shape always; on-disk-existence when skill_root is supplied.
Implements Rule OUT-6: bundled assets co-located under the skill root.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.descriptor.semantic_rules import R_BUNDLED_PATH_EXISTS
from mellea_skills_compiler.descriptor.validator import validate


def _descriptor_with_bundle(source_dir: str) -> dict:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "ex", "classification": {"primary_axis": "AGENT"}},
        "inputs": [],
        "outputs": [],
        "schemas": {},
        "state": [{"id": "s", "symbol": "mellea.stdlib.session.start_session"}],
        "pipeline": [{
            "kind": "call",
            "id": "c0",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "args": {},
        }],
        "bundled_resources": [
            {"source_dir": source_dir, "package_dir": "references"}
        ],
    }


def _has_error(errors, rule_id: str) -> bool:
    return any(e.rule == rule_id for e in errors)


def test_relative_path_passes_shape_check():
    """C-RELATIVE-PATH-PASSES — shape-only check (no skill_root).

    Without skill_root, the rule may emit a WARNING-severity "cannot
    be verified" entry (telemetry). The contract is no ERROR-severity
    entry on a well-shaped relative path.
    """
    descriptor = _descriptor_with_bundle("references")
    rep = validate(
        descriptor, schema_version="0.3", surface=None, skill_root=None
    )
    error_severity = [
        e for e in rep.errors
        if e.rule == R_BUNDLED_PATH_EXISTS and e.severity == "error"
    ]
    assert not error_severity, (
        f"well-shaped relative path must not fire an ERROR-severity "
        f"R-SEM-BUNDLED-PATH-EXISTS; got: "
        f"{[(e.rule, e.severity, e.message) for e in error_severity]}"
    )


def test_absolute_path_fails():
    """C-ABSOLUTE-PATH-FAILS."""
    descriptor = _descriptor_with_bundle("/etc/passwd")
    rep = validate(
        descriptor, schema_version="0.3", surface=None, skill_root=None
    )
    assert _has_error(rep.errors, R_BUNDLED_PATH_EXISTS), (
        f"absolute source_dir must fire R-SEM-BUNDLED-PATH-EXISTS; got: "
        f"{[(e.rule, e.message) for e in rep.errors]}"
    )


def test_traversal_path_fails():
    """C-TRAVERSAL-PATH-FAILS."""
    descriptor = _descriptor_with_bundle("../escape")
    rep = validate(
        descriptor, schema_version="0.3", surface=None, skill_root=None
    )
    assert _has_error(rep.errors, R_BUNDLED_PATH_EXISTS), (
        f"traversal source_dir must fire R-SEM-BUNDLED-PATH-EXISTS; "
        f"got: {[(e.rule, e.message) for e in rep.errors]}"
    )


def test_missing_path_fails_with_skill_root():
    """C-MISSING-ON-DISK-FAILS-WITH-SKILL-ROOT."""
    with tempfile.TemporaryDirectory() as tmp:
        # Don't create `references/` under tmp — the path is missing.
        descriptor = _descriptor_with_bundle("references")
        rep = validate(
            descriptor,
            schema_version="0.3",
            surface=None,
            skill_root=tmp,
        )
        assert _has_error(rep.errors, R_BUNDLED_PATH_EXISTS), (
            f"missing on-disk path must fire R-SEM-BUNDLED-PATH-EXISTS "
            f"when skill_root is supplied; got: "
            f"{[(e.rule, e.message) for e in rep.errors]}"
        )
