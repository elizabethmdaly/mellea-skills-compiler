"""Coherence audits for the ``r-sem-disposition-consistent`` semantic rule.

The rule enforces that descriptor dependencies declare the supporting
fields their (kind, disposition) pair requires:

  * ``bundle`` / ``load_from_disk`` → ``path`` required
  * ``kind=='tool'`` + disposition in {delegate_to_runtime, real_impl,
    stub, mock, external_input} → ``signature`` required, and the
    signature string must be parseable
  * ``kind=='tool'`` + ``disposition=='real_impl'`` + ``symbol`` set →
    symbol must resolve in the introspected Mellea surface

These tests exercise the validation face directly via the private
``_check_dispositions`` callable and assert the rule's behaviour matches
the registry's documented stance. C-SCHEMA-ENFORCES-SIG-WHEN-TOOL-NEEDS
and C-DIRECTIVE-DOC-EXISTS are ``xfail(strict=True)`` until the
structural-fix and directive-doc work lands; their registry
``xfail_until`` deadlines mirror the decorator.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from mellea_skills_compiler.descriptor.semantic_rules import (
    R_DISPOSITION_CONSISTENT,
    _check_dispositions,
)
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "r-sem-disposition-consistent"


def _descriptor_with_dep(dep: dict) -> dict:
    """Wrap a single dependency in a minimal descriptor envelope.

    ``_check_dispositions`` only reads ``descriptor['dependencies']``, so
    everything else can be elided.
    """
    return {"dependencies": [dep]}


def _has_error(errors, rule_id: str, *, path_suffix: str | None = None) -> bool:
    """Return True iff any ValidationError matches the rule (and path)."""
    for err in errors:
        if err.rule != rule_id:
            continue
        if path_suffix is not None and not err.path.endswith(path_suffix):
            continue
        return True
    return False


# ─── Coherence checks ────────────────────────────────────────────────


def test_tool_needs_signature_without_signature_fails():
    """C-TOOL-NEEDS-SIG-WITHOUT-FAILS — the original failure."""
    dep = {"id": "install_skill_dep", "kind": "tool", "disposition": "stub"}
    errors = _check_dispositions(
        descriptor=_descriptor_with_dep(dep), surface=None
    )
    assert _has_error(
        errors, R_DISPOSITION_CONSISTENT, path_suffix="/signature"
    ), (
        f"tool/stub without `signature` must fire R-SEM-DISPOSITION-CONSISTENT "
        f"on /signature; got {errors!r}"
    )


def test_tool_needs_signature_with_signature_passes():
    """C-TOOL-NEEDS-SIG-WITH-PASSES."""
    dep = {
        "id": "install_skill_dep",
        "kind": "tool",
        "disposition": "stub",
        "signature": "(skill_id: str) -> dict[str, Any]",
    }
    errors = _check_dispositions(
        descriptor=_descriptor_with_dep(dep), surface=None
    )
    assert not _has_error(errors, R_DISPOSITION_CONSISTENT), (
        f"tool/stub WITH a parseable signature must not fire the rule; "
        f"got {errors!r}"
    )


def test_bundle_without_path_fails():
    """C-BUNDLE-WITHOUT-PATH-FAILS."""
    dep = {"id": "schema_pack", "kind": "file", "disposition": "bundle"}
    errors = _check_dispositions(
        descriptor=_descriptor_with_dep(dep), surface=None
    )
    assert _has_error(
        errors, R_DISPOSITION_CONSISTENT, path_suffix="/path"
    ), (
        f"bundle without `path` must fire R-SEM-DISPOSITION-CONSISTENT on "
        f"/path; got {errors!r}"
    )


def test_unparseable_signature_fails():
    """C-UNPARSEABLE-SIG-FAILS."""
    dep = {
        "id": "install_skill_dep",
        "kind": "tool",
        "disposition": "stub",
        "signature": "not a real signature string",
    }
    errors = _check_dispositions(
        descriptor=_descriptor_with_dep(dep), surface=None
    )
    assert _has_error(
        errors, R_DISPOSITION_CONSISTENT, path_suffix="/signature"
    ), (
        f"tool/stub with unparseable signature must fire the rule; "
        f"got {errors!r}"
    )


def test_non_tool_kind_without_signature_passes():
    """C-NON-TOOL-NO-SIG-PASSES."""
    for kind in ("file", "secret", "data"):
        dep = {
            "id": f"{kind}_dep",
            "kind": kind,
            "disposition": "stub",
        }
        errors = _check_dispositions(
            descriptor=_descriptor_with_dep(dep), surface=None
        )
        assert not _has_error(errors, R_DISPOSITION_CONSISTENT), (
            f"kind={kind!r} with disposition=stub must not require a "
            f"signature (signature rule is gated on kind=='tool'); "
            f"got {errors!r}"
        )


def test_schema_structurally_requires_signature_for_tool_needs_signature():
    """C-SCHEMA-ENFORCES-SIG-WHEN-TOOL-NEEDS."""
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = (
        repo_root
        / "src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    dep_schema = schema["$defs"]["Dependency"]
    validator = jsonschema.Draft202012Validator(dep_schema)
    bad_dep = {"id": "x", "kind": "tool", "disposition": "stub"}
    errors = list(validator.iter_errors(bad_dep))
    assert errors, (
        "schema must structurally reject a tool/stub dep without "
        "`signature` (currently only the semantic rule catches this)"
    )


def test_directive_doc_section_exists():
    """C-DIRECTIVE-DOC-EXISTS."""
    rule = get_rule(_RULE_ID)
    doc_pointer = rule["directive"]["doc"]
    section = rule["directive"]["section"]
    assert doc_pointer, "directive.doc must not be null"
    assert section, "directive.section must not be null"
    repo_root = Path(__file__).resolve().parents[3]
    doc_path = repo_root / doc_pointer
    assert doc_path.is_file(), f"directive doc {doc_pointer!r} does not exist"
    doc_text = doc_path.read_text(encoding="utf-8")
    assert section in doc_text, (
        f"directive section {section!r} not found in {doc_pointer!r}"
    )
