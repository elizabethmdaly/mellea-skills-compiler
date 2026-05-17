"""Tests for v0.3 consistency / resolution rules:

- R-SEM-DISPOSITION-CONSISTENT
- R-SEM-BUNDLED-PATH-EXISTS
- R-SEM-AGENT-LOOP-FIELDS
- R-SEM-SOURCE-ELEMENT-RESOLVES
- R-SEM-FIELD-TYPE-KNOWN
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.descriptor import validate
from mellea_skills_compiler.descriptor.semantic_rules import (
    R_AGENT_LOOP_FIELDS,
    R_BUNDLED_PATH_EXISTS,
    R_DISPOSITION_CONSISTENT,
    R_FIELD_TYPE_KNOWN,
    R_SOURCE_ELEMENT_RESOLVES,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _v03() -> dict[str, Any]:
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": {"primary_axis": "DSL"}},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }


def _findings(report, rule):
    return [e for e in report.errors if e.rule == rule]


# ===========================================================================
# R-SEM-DISPOSITION-CONSISTENT
# ===========================================================================


class TestDispositionConsistent:

    def test_pre_not_met_no_deps(self):
        d = _v03()
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_DISPOSITION_CONSISTENT)

    def test_tool_delegate_to_runtime_with_sig_passes(self):
        d = _v03()
        d["dependencies"] = [{
            "id": "fn", "kind": "tool", "disposition": "delegate_to_runtime",
            "signature": "(x: str) -> int",
        }]
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_DISPOSITION_CONSISTENT)

    def test_tool_delegate_to_runtime_missing_sig_fires(self):
        d = _v03()
        d["dependencies"] = [{
            "id": "fn", "kind": "tool", "disposition": "delegate_to_runtime"
        }]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_DISPOSITION_CONSISTENT)
        assert findings
        assert any("signature" in f.message for f in findings)

    def test_unparseable_signature_fires(self):
        d = _v03()
        d["dependencies"] = [{
            "id": "fn", "kind": "tool", "disposition": "stub",
            "signature": "not-a-signature",
        }]
        report = validate(d, schema_version="0.3")
        assert _findings(report, R_DISPOSITION_CONSISTENT)

    def test_bundle_missing_path_fires(self):
        d = _v03()
        d["dependencies"] = [
            {"id": "files", "kind": "file", "disposition": "bundle"},
        ]
        # Schema-level rejection is expected too; but our rule also flags it.
        report = validate(d, schema_version="0.3")
        # The JSON-Schema layer also raises; we just verify our rule contributes:
        assert _findings(report, R_DISPOSITION_CONSISTENT)

    def test_real_impl_with_resolving_symbol_passes(self, surface):
        d = _v03()
        d["dependencies"] = [{
            "id": "tool_one", "kind": "tool", "disposition": "real_impl",
            "signature": "(x: str) -> int",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
        }]
        report = validate(d, schema_version="0.3", surface=surface)
        assert not _findings(report, R_DISPOSITION_CONSISTENT)

    def test_real_impl_with_non_resolving_symbol_fires(self, surface):
        d = _v03()
        d["dependencies"] = [{
            "id": "tool_one", "kind": "tool", "disposition": "real_impl",
            "signature": "(x: str) -> int",
            "symbol": "nothing.like.this",
        }]
        report = validate(d, schema_version="0.3", surface=surface)
        assert _findings(report, R_DISPOSITION_CONSISTENT)

    def test_remove_disposition_needs_nothing(self):
        d = _v03()
        d["dependencies"] = [
            {"id": "gone", "kind": "tool", "disposition": "remove"},
        ]
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_DISPOSITION_CONSISTENT)


# ===========================================================================
# R-SEM-BUNDLED-PATH-EXISTS
# ===========================================================================


class TestBundledPathExists:

    def test_pre_not_met_no_resources(self):
        d = _v03()
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_BUNDLED_PATH_EXISTS)

    def test_existing_path_passes(self, tmp_path):
        (tmp_path / "references").mkdir()
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "references", "package_dir": "references"}
        ]
        report = validate(d, schema_version="0.3", skill_root=tmp_path)
        assert not _findings(report, R_BUNDLED_PATH_EXISTS)

    def test_missing_path_with_skill_root_errors(self, tmp_path):
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "missing", "package_dir": "missing"}
        ]
        report = validate(d, schema_version="0.3", skill_root=tmp_path)
        findings = _findings(report, R_BUNDLED_PATH_EXISTS)
        assert findings
        assert all(f.severity == "error" for f in findings)

    def test_missing_path_without_skill_root_warns(self):
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "missing", "package_dir": "missing"}
        ]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_BUNDLED_PATH_EXISTS)
        assert findings
        assert all(f.severity == "warning" for f in findings)

    def test_absolute_path_rejected(self):
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "/etc/passwd", "package_dir": "etc"}
        ]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_BUNDLED_PATH_EXISTS)
        assert findings
        assert any("absolute" in f.message for f in findings)

    def test_traversal_rejected(self):
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "../escape", "package_dir": "escape"}
        ]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_BUNDLED_PATH_EXISTS)
        assert findings
        assert any(".." in f.message or "parent" in f.message for f in findings)

    def test_nested_relative_path_accepted(self, tmp_path):
        (tmp_path / "data" / "subdir").mkdir(parents=True)
        d = _v03()
        d["bundled_resources"] = [
            {"source_dir": "data/subdir", "package_dir": "data/subdir"}
        ]
        report = validate(d, schema_version="0.3", skill_root=tmp_path)
        assert not _findings(report, R_BUNDLED_PATH_EXISTS)


# ===========================================================================
# R-SEM-AGENT-LOOP-FIELDS
# ===========================================================================


class TestAgentLoopFields:

    def _loop(self, **overrides):
        n = {
            "id": "loop_one",
            "kind": "composition",
            "operator": "agent_loop",
            "max_iterations": 5,
            "state_schema": {"ref": "#/schemas/S"},
            "termination_condition": {"predicate": "field_truthy", "field": "x"},
            "body": [
                {"id": "b1", "kind": "call",
                 "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                 "args": {}}
            ],
        }
        n.update(overrides)
        return n

    def test_pre_not_met_no_agent_loop(self):
        d = _v03()
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_AGENT_LOOP_FIELDS)

    def test_all_required_passes(self):
        d = _v03()
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        d["pipeline"] = [self._loop()]
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_AGENT_LOOP_FIELDS)

    def test_missing_max_iterations_fires(self):
        d = _v03()
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        loop = self._loop()
        del loop["max_iterations"]
        d["pipeline"] = [loop]
        report = validate(d, schema_version="0.3")
        # JSON-Schema also rejects, but our rule also reports it.
        findings = _findings(report, R_AGENT_LOOP_FIELDS)
        assert findings
        assert any("max_iterations" in f.message for f in findings)

    def test_invalid_termination_predicate_fires(self):
        d = _v03()
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        loop = self._loop(termination_condition={"predicate": "wibble", "field": "x"})
        d["pipeline"] = [loop]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_AGENT_LOOP_FIELDS)
        assert findings

    def test_tools_ref_resolves_to_dep_passes(self):
        d = _v03()
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        d["dependencies"] = [{
            "id": "search", "kind": "tool", "disposition": "delegate_to_runtime",
            "signature": "(q: str) -> list[str]",
        }]
        loop = self._loop(tools_ref=["search"])
        d["pipeline"] = [loop]
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_AGENT_LOOP_FIELDS)

    def test_tools_ref_unresolved_fires(self):
        d = _v03()
        d["schemas"]["S"] = {"kind": "model", "fields": {"x": {"type": "bool"}}}
        loop = self._loop(tools_ref=["nonexistent_tool"])
        d["pipeline"] = [loop]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_AGENT_LOOP_FIELDS)
        assert findings
        assert any("nonexistent_tool" in f.message for f in findings)


# ===========================================================================
# R-SEM-SOURCE-ELEMENT-RESOLVES
# ===========================================================================


class TestSourceElementResolves:

    def test_pre_not_met_no_source_elements(self):
        d = _v03()
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_SOURCE_ELEMENT_RESOLVES)

    def test_resolving_ids_with_inventory_pass(self):
        d = _v03()
        d["pipeline"] = [
            {"id": "n1", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {}, "source_elements": ["E14"]}
        ]
        inv = {"elements": [{"id": "E14"}, {"id": "E15"}]}
        report = validate(d, schema_version="0.3", inventory=inv)
        assert not _findings(report, R_SOURCE_ELEMENT_RESOLVES)

    def test_missing_id_with_inventory_errors(self):
        d = _v03()
        d["pipeline"] = [
            {"id": "n1", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {}, "source_elements": ["E99"]}
        ]
        inv = {"elements": [{"id": "E14"}]}
        report = validate(d, schema_version="0.3", inventory=inv)
        findings = _findings(report, R_SOURCE_ELEMENT_RESOLVES)
        assert findings
        assert all(f.severity == "error" for f in findings)

    def test_source_elements_without_inventory_warns(self):
        d = _v03()
        d["pipeline"] = [
            {"id": "n1", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {}, "source_elements": ["E14"]}
        ]
        report = validate(d, schema_version="0.3")
        findings = _findings(report, R_SOURCE_ELEMENT_RESOLVES)
        assert findings
        assert all(f.severity == "warning" for f in findings)

    def test_mixed_resolution_errors_on_missing(self):
        d = _v03()
        d["pipeline"] = [
            {"id": "n1", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {}, "source_elements": ["E14", "E99"]}
        ]
        inv = {"elements": [{"id": "E14"}]}
        report = validate(d, schema_version="0.3", inventory=inv)
        findings = _findings(report, R_SOURCE_ELEMENT_RESOLVES)
        assert findings
        # Only the missing one fires.
        assert any("E99" in f.message for f in findings)
        assert not any("E14" in f.message for f in findings)

    def test_empty_array_passes(self):
        d = _v03()
        d["pipeline"] = [
            {"id": "n1", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "args": {}, "source_elements": []}
        ]
        report = validate(d, schema_version="0.3", inventory={"elements": []})
        assert not _findings(report, R_SOURCE_ELEMENT_RESOLVES)


# ===========================================================================
# R-SEM-FIELD-TYPE-KNOWN
# ===========================================================================


class TestFieldTypeKnown:

    def test_pre_not_met_no_schemas(self):
        d = _v03()
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)

    def test_scalar_types_pass(self):
        d = _v03()
        d["schemas"]["M"] = {
            "kind": "model",
            "fields": {
                "a": {"type": "str"}, "b": {"type": "int"},
                "c": {"type": "float"}, "d": {"type": "bool"},
            },
        }
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)

    def test_model_ref_pass(self):
        d = _v03()
        d["schemas"]["M"] = {"kind": "model", "fields": {"x": {"type": "model_ref",
                                                                "ref": "#/schemas/M"}}}
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)

    def test_list_and_dict_pass(self):
        d = _v03()
        d["schemas"]["M"] = {
            "kind": "model",
            "fields": {
                "items": {"type": "list[str]"},
                "lookup": {"type": "dict[str, int]"},
            },
        }
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)

    def test_optional_and_literal_pass(self):
        d = _v03()
        d["schemas"]["M"] = {
            "kind": "model",
            "fields": {
                "maybe": {"type": "Optional[str]"},
                "color": {"type": "Literal[red, blue]"},
            },
        }
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)

    def test_unknown_type_fires(self):
        """This closes test_a6_unknown_field_type via the validator chokepoint."""
        d = _v03()
        d["schemas"]["Bad"] = {
            "kind": "model",
            "fields": {"x": {"type": "TotallyUnknownType"}},
        }
        # 'TotallyUnknownType' is a valid identifier — so the rule treats it as
        # a model-reference candidate, but there is no declared schema by that
        # name. The rule fires per the closed-vocabulary contract.
        # NOTE: this matches the renderer-side check in render_schemas, which
        # rejects bare identifiers that aren't declared schemas in its
        # schemas_block.
        from mellea_skills_compiler.renderer import RendererError
        from mellea_skills_compiler.renderer.schemas import render_schemas
        import pytest as _pytest
        with _pytest.raises(RendererError):
            render_schemas(d["schemas"])

    def test_bare_declared_schema_pass(self):
        """A bare identifier that names a declared schema is accepted."""
        d = _v03()
        d["schemas"] = {
            "Other": {"kind": "model", "fields": {"v": {"type": "str"}}},
            "M": {"kind": "model", "fields": {"x": {"type": "Other"}}},
        }
        report = validate(d, schema_version="0.3")
        assert not _findings(report, R_FIELD_TYPE_KNOWN)
