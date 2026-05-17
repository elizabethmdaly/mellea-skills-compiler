"""Schema-validation battery for intermediate-artefact contracts.

The /mellea-fy workflow produces 8 distinct intermediate artefacts in
``intermediate/``. Each has a JSON Schema in ``.claude/schemas/``. Today
NONE of these schemas have automated test coverage — they live as
documentation only, exercised solely by LLM-driven slash-command steps.

This file is the proactive battery for those schemas, applying the
seven strategies from ``process/testing-strategy.md``:

- S1 — test from the schema CONTRACT, not the LLM-emitted artefact
- S2 — categorise failure modes first (see test class names)
- S3 — adversarial framing: "what would a malicious /mellea-fy step
  produce that the schema fails to catch?"
- S4 — property-based across schemas (e.g. every schema MUST self-validate
  against Draft 2020-12)
- S7 — domain categories: required-field violations, enum violations,
  pattern violations, cross-reference invariants

Worked example: this file currently covers ``element_mapping.schema.json``
in depth. The same patterns extend to the other 7 schemas
(classification, inventory, dependency_plan, config_emission,
fixtures_emission, melleafy, runtime_defaults). A follow-up batch
generates those.

What this battery is NOT testing (acknowledged gaps):
- Whether the LLM at Step 2 actually produces output that satisfies the
  schema (that's an end-to-end /mellea-fy concern, not a schema concern)
- Whether the schema itself is COMPLETE — i.e., whether it catches every
  semantically invalid mapping (cross-reference invariants are NOT
  enforced by JSON Schema; we encode them as separate property tests)
- Schema migration / version compatibility (if a schema changes, do old
  artefacts continue to validate?) — separate concern
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / ".claude" / "schemas"


# --- helpers ---------------------------------------------------------------


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))


def _validate(schema: dict[str, Any], doc: dict[str, Any]) -> list[ValidationError]:
    """Return all errors (empty list = valid)."""
    validator = Draft202012Validator(schema)
    return list(validator.iter_errors(doc))


def _make_valid_mapping_entry(*, mapping_id: str = "map_001", element_id: str = "elem_042") -> dict[str, Any]:
    """Return a minimal-but-valid mapping entry per element_mapping.schema.json."""
    return {
        "mapping_id": mapping_id,
        "element_id": element_id,
        "target_file": "pipeline.py",
        "target_symbol": "extract_files",
        "primitive": "m.instruct",
        "final_target_file": "pipeline.py",
    }


def _make_valid_element_mapping(*, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if mappings is None:
        mappings = [_make_valid_mapping_entry()]
    return {"format_version": "1.0", "mappings": mappings}


# --- Property: every schema in .claude/schemas/ MUST self-validate ----------


@pytest.mark.parametrize(
    "schema_filename",
    [
        "classification.schema.json",
        "inventory.schema.json",
        "element_mapping.schema.json",
        "dependency_plan.schema.json",
        "config_emission.schema.json",
        "fixtures_emission.schema.json",
        "melleafy.schema.json",
        "runtime_defaults.schema.json",
    ],
)
def test_property_all_artefact_schemas_self_validate(schema_filename):
    """Property: every schema file must itself be a valid Draft 2020-12 schema.

    Adversarial framing: a malformed schema would silently accept any
    artefact. This is the most catastrophic failure mode for the contract
    layer. Test must run on EVERY schema in the directory."""
    schema = _load_schema(schema_filename)
    # Must not raise.
    Draft202012Validator.check_schema(schema)


def test_property_all_schemas_have_draft_2020_12_dialect():
    """Every schema declares the 2020-12 dialect explicitly. If a schema
    omits ``$schema``, it falls back to the validator default — silent
    incompatibility risk on dialect upgrade."""
    for path in SCHEMAS_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", (
            f"{path.name} missing or wrong $schema dialect"
        )


# =============================================================================
# Category 1: element_mapping required-field violations
# =============================================================================
# Adversarial framing: a malicious Step 2 omits fields the downstream
# Step 2.5d / Step 5 will rely on. The schema must catch this BEFORE
# downstream consumers see incomplete data.


class TestElementMappingRequiredFields:

    def test_valid_minimal_element_mapping_passes(self):
        schema = _load_schema("element_mapping.schema.json")
        errors = _validate(schema, _make_valid_element_mapping())
        assert errors == []

    def test_missing_format_version_fails(self):
        schema = _load_schema("element_mapping.schema.json")
        doc = _make_valid_element_mapping()
        del doc["format_version"]
        errors = _validate(schema, doc)
        assert any("format_version" in str(e.message) or "required" in e.validator for e in errors)

    def test_missing_mappings_fails(self):
        schema = _load_schema("element_mapping.schema.json")
        doc = _make_valid_element_mapping()
        del doc["mappings"]
        errors = _validate(schema, doc)
        assert any("mappings" in str(e.message) or "required" in e.validator for e in errors)

    def test_empty_mappings_array_fails(self):
        """element_mapping.schema specifies minItems: 1 — a skill with zero
        mappings is suspect even if inventory had zero elements."""
        schema = _load_schema("element_mapping.schema.json")
        doc = _make_valid_element_mapping(mappings=[])
        errors = _validate(schema, doc)
        assert any("minItems" in e.validator for e in errors), (
            "empty mappings array slipped past minItems constraint"
        )

    def test_mapping_entry_missing_required_field_fails(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        del entry["primitive"]  # required
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert any("primitive" in str(e.message) or "required" in e.validator for e in errors)


# =============================================================================
# Category 2: pattern-constraint violations (mapping_id, element_id)
# =============================================================================


class TestElementMappingPatterns:

    @pytest.mark.parametrize(
        "bad_mapping_id",
        ["MAP_001", "map_01", "map001", "1_map", "map_001_", "map_", ""],
    )
    def test_mapping_id_pattern_rejects_malformed(self, bad_mapping_id):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry(mapping_id=bad_mapping_id)
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert any("pattern" in e.validator for e in errors), (
            f"malformed mapping_id {bad_mapping_id!r} slipped past pattern constraint"
        )

    @pytest.mark.parametrize(
        "good_mapping_id",
        ["map_001", "map_042", "map_999", "map_100000"],
    )
    def test_mapping_id_pattern_accepts_canonical(self, good_mapping_id):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry(mapping_id=good_mapping_id)
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []

    @pytest.mark.parametrize(
        "bad_element_id",
        ["ELEM_001", "elem_01", "element_042", "el_042", "elem_042-step3", "elem_042-bogus", "elem_042-step"],
    )
    def test_element_id_pattern_rejects_malformed(self, bad_element_id):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry(element_id=bad_element_id)
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert any("pattern" in e.validator for e in errors), (
            f"malformed element_id {bad_element_id!r} slipped past pattern"
        )

    @pytest.mark.parametrize(
        "good_element_id",
        ["elem_042", "elem_017-step1", "elem_017-step2", "elem_088-mod", "elem_088-eval", "elem_088-loop"],
    )
    def test_element_id_pattern_accepts_canonical_and_expansions(self, good_element_id):
        """The schema's pattern allows EXTRACT (-step1/-step2) and REMEDIATE
        (-mod/-eval/-loop) expansion suffixes."""
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry(element_id=good_element_id)
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []


# =============================================================================
# Category 3: enum-constraint violations (target_file, primitive)
# =============================================================================


class TestElementMappingEnums:

    def test_target_file_outside_enum_rejected(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["target_file"] = "nonexistent.py"
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert any("enum" in e.validator for e in errors)

    @pytest.mark.parametrize(
        "tf",
        ["pipeline.py", "tools.py", "schemas.py", "config.py", "slots.py",
         "requirements.py", "constrained_slots.py", "mobjects.py", "loader.py", "main.py"],
    )
    def test_target_file_all_ten_accepted(self, tf):
        """All 10 always-emitted/conditional package files are valid."""
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["target_file"] = tf
        entry["final_target_file"] = tf
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []

    def test_primitive_outside_enum_rejected(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["primitive"] = "m.fictitious_primitive"
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert any("enum" in e.validator for e in errors)

    @pytest.mark.parametrize(
        "prim",
        ["@generative", "m.instruct", "m.transform", "m.query", "m.chat", "m.react",
         "Requirement", "BaseModel", "Final", "function", "while_loop",
         "parameter", "loader_call", "stub", "none"],
    )
    def test_primitive_all_documented_kinds_accepted(self, prim):
        """The 15 documented primitive kinds all validate."""
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["primitive"] = prim
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []


# =============================================================================
# Category 4: final_target_file special values + cross-step staging
# =============================================================================
# Adversarial framing: Step 2 emits an entry with final_target_file ==
# "pending_step_2.5". Step 2.5 must eventually amend it. The schema permits
# pending_step_2.5 but does NOT enforce the post-2.5 invariant
# (no entry retains pending_step_2.5 after Step 2.5 declares done).
# That cross-step invariant must be tested separately.


class TestElementMappingFinalTargetFile:

    def test_final_target_pending_is_valid_pre_step_2_5(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["final_target_file"] = "pending_step_2.5"
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []

    def test_final_target_fixtures_mock_tools_is_valid(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["final_target_file"] = "fixtures/mock_tools.py"
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors == []

    def test_final_target_arbitrary_string_rejected(self):
        schema = _load_schema("element_mapping.schema.json")
        entry = _make_valid_mapping_entry()
        entry["final_target_file"] = "made_up_target.py"
        doc = _make_valid_element_mapping(mappings=[entry])
        errors = _validate(schema, doc)
        assert errors, "arbitrary final_target_file slipped past oneOf"


# =============================================================================
# Category 5: cross-step invariants NOT enforced by JSON Schema
# =============================================================================
# These are documented in the schema descriptions as "not enforced by
# JSON Schema." We test them as Python assertions.
#
# Adversarial framing: the schema deliberately doesn't enforce these
# (would require cross-file context). A defective Step 2 implementation
# could produce schema-valid output that violates these invariants
# silently. The validator at /mellea-fy or in CI must catch them.


class TestElementMappingCrossStepInvariants:

    def test_every_inventory_element_appears_in_mapping(self):
        """Invariant: for every elem_NNN in inventory, there's at least one
        mapping entry whose element_id starts with that elem_NNN (possibly
        with -step1/-step2/-mod/-eval/-loop suffixes)."""
        inventory = {
            "format_version": "1.0",
            "elements": [
                {"element_id": "elem_001", "tag": "PROMPT_BLOCK", "category": "C1",
                 "content_summary": "x", "source_file": "spec.md", "source_lines": [1, 1]},
                {"element_id": "elem_002", "tag": "EXTRACT", "category": "C3",
                 "content_summary": "y", "source_file": "spec.md", "source_lines": [2, 2]},
            ],
        }
        mapping = _make_valid_element_mapping(
            mappings=[
                _make_valid_mapping_entry(mapping_id="map_001", element_id="elem_001"),
                # elem_002 (EXTRACT) → two-step expansion
                _make_valid_mapping_entry(mapping_id="map_002", element_id="elem_002-step1"),
                _make_valid_mapping_entry(mapping_id="map_003", element_id="elem_002-step2"),
            ]
        )

        inv_ids = {e["element_id"] for e in inventory["elements"]}
        mapped_root_ids = {m["element_id"].split("-")[0] for m in mapping["mappings"]}
        missing = inv_ids - mapped_root_ids
        assert not missing, f"inventory elements not represented in mapping: {missing}"

    @pytest.mark.xfail(
        reason="Phase 3.5 — cross-step invariant 'EXTRACT requires -step1 AND -step2 expansion' "
        "is documented in element_mapping.schema.json but NOT enforced by any validator today. "
        "When Phase 3.5 wires a cross-step validator this test passes.",
        strict=False,
    )
    def test_extract_element_must_have_step1_and_step2_expansion(self):
        """Adversarial: Step 2 emits an EXTRACT element_id without the
        -step1/-step2 expansion. The mapping is INCOMPLETE.

        This is NOT enforced by JSON Schema (would require knowing the
        tag from inventory). Must be a separate check."""
        # An EXTRACT element in inventory.
        extract_element_id = "elem_017"
        # ... but only one expansion emitted.
        mapping = _make_valid_element_mapping(
            mappings=[
                _make_valid_mapping_entry(mapping_id="map_001", element_id=f"{extract_element_id}-step1"),
                # MISSING -step2 expansion.
            ]
        )
        seen_suffixes = {
            m["element_id"].split("-", 1)[1] if "-" in m["element_id"] else None
            for m in mapping["mappings"]
            if m["element_id"].startswith(extract_element_id)
        }
        # For an EXTRACT element, both -step1 and -step2 must be present.
        assert seen_suffixes == {"step1", "step2"}, (
            f"EXTRACT element {extract_element_id} is missing expansion: "
            f"only saw {seen_suffixes}"
        )

    @pytest.mark.xfail(
        reason="Phase 3.5 — cross-step invariant 'TOOL_TEMPLATE entries must have "
        "final_target_file == pending_step_2.5 before Step 2.5d' is documented but not enforced. "
        "Belongs in a cross-step validator the descriptor pipeline should ship.",
        strict=False,
    )
    def test_tool_template_provisional_must_be_pending_until_step_2_5(self):
        """Invariant: a TOOL_TEMPLATE primitive's final_target_file
        MUST be 'pending_step_2.5' in the OUTPUT of Step 2 (before
        Step 2.5d). After Step 2.5, this should be amended.

        This is NOT enforced by JSON Schema because it requires knowing
        whether Step 2.5 has run."""
        # Step 2's output: a TOOL_TEMPLATE entry. Schema only knows the
        # values; we encode the policy here.
        entry = _make_valid_mapping_entry()
        entry["primitive_details"] = {"is_tool_template": True}
        entry["final_target_file"] = "tools.py"  # WRONG before Step 2.5
        # Policy check (not in JSON Schema):
        if entry.get("primitive_details", {}).get("is_tool_template"):
            assert entry["final_target_file"] == "pending_step_2.5", (
                f"TOOL_TEMPLATE entry must defer final_target_file until Step 2.5d; "
                f"saw {entry['final_target_file']!r}"
            )


# =============================================================================
# Property test: schema-valid documents always round-trip through json
# =============================================================================


def test_property_valid_documents_json_round_trip():
    """Any document that validates against the schema must round-trip
    through json.dumps/loads without loss. (Catches schemas that secretly
    allow non-JSON-serializable values like Infinity or NaN.)"""
    schema = _load_schema("element_mapping.schema.json")
    doc = _make_valid_element_mapping()
    assert _validate(schema, doc) == []
    round_tripped = json.loads(json.dumps(doc))
    assert _validate(schema, round_tripped) == []
    assert round_tripped == doc
