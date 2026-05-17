"""Tests for the schema-version dispatch helper (Phase 3.5.B).

The validator routes incoming descriptors to a frozen JSON Schema file based
on the descriptor's ``descriptor_version`` field. These tests pin the routing
behaviour:

- explicit ``schema_version`` argument wins;
- otherwise, the descriptor's declared version is used;
- otherwise, the default ``"0.2"`` is used;
- unsupported versions raise ``ValueError``.
"""
from __future__ import annotations

import pytest

from mellea_skills_compiler.descriptor import (
    SUPPORTED_SCHEMA_VERSIONS,
    load_schema,
    resolve_schema_version,
    validate,
)


def test_supported_versions_include_v03():
    assert "0.3" in SUPPORTED_SCHEMA_VERSIONS


def test_load_schema_v03():
    schema = load_schema("0.3")
    assert schema["$id"].endswith("descriptor.v0.3.json")
    assert schema["properties"]["descriptor_version"]["const"] == "0.3"


def test_resolve_explicit_version_wins():
    desc = {"descriptor_version": "0.2"}
    assert resolve_schema_version(desc, "0.3") == "0.3"


def test_resolve_descriptor_version_used_when_unspecified():
    assert resolve_schema_version({"descriptor_version": "0.3"}) == "0.3"
    assert resolve_schema_version({"descriptor_version": "0.2"}) == "0.2"
    assert resolve_schema_version({"descriptor_version": "0.1"}) == "0.1"


def test_resolve_default_when_no_descriptor_version():
    assert resolve_schema_version({}) == "0.2"


def test_resolve_unsupported_version_raises():
    with pytest.raises(ValueError):
        resolve_schema_version({"descriptor_version": "9.9"})
    with pytest.raises(ValueError):
        resolve_schema_version({}, requested="9.9")


def test_validate_routes_to_v03():
    desc = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }
    report = validate(desc, schema_version="0.3")
    assert report.valid, [(e.path, e.message) for e in report.errors]
    assert report.schema_version == "0.3"


def test_v02_descriptors_unchanged_against_v02():
    """Pin: a valid v0.2 descriptor still routes through v0.2 unchanged."""
    desc = {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }
    report = validate(desc, schema_version="0.2")
    assert report.valid
    assert report.schema_version == "0.2"


def test_v02_classification_string_rejected_under_v03():
    """v0.2 string classification is not accepted by v0.3 schema (which
    requires the structured classification block)."""
    desc = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},  # v0.2 shape
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }
    report = validate(desc, schema_version="0.3")
    assert not report.valid
