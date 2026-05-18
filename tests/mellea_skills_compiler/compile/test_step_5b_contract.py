"""Tests for the Step 5b in-session emission-validation contract entry.

Step 5b is an opportunistic structural-lint gate inserted between Step 5
(body emission) and Step 6 (artefacts). It is documented at
`.claude/commands/mellea-fy-validate-emissions.md`, registered in the
pipeline contract at `pipeline_contract.PIPELINE_CONTRACT`, and emits
`intermediate/step_5b_report.json` (schema at
`src/mellea_skills_compiler/descriptor/schemas/step_5b_report.schema.json`).

These tests guard:

  1. The contract entry exists and ordering is correct.
  2. Its declared inputs cover the Step 5 emissions + the canonical
     Mellea surface (which Claude reasons over).
  3. The output schema file is on disk and a parseable JSON Schema.
  4. Minimal/representative payloads validate (pass / fixed / fail).
  5. Missing required fields are rejected.
  6. ``verify_contract()`` is clean on the real contract after the
     addition.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from mellea_skills_compiler.compile.pipeline_contract import (
    PIPELINE_CONTRACT,
    topological_order,
    verify_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = (
    REPO_ROOT
    / "src"
    / "mellea_skills_compiler"
    / "descriptor"
    / "schemas"
    / "step_5b_report.schema.json"
)


def _find_step(step_id: str):
    for step in PIPELINE_CONTRACT:
        if step.step_id == step_id:
            return step
    return None


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Contract-entry shape
# ---------------------------------------------------------------------------


class TestStep5bContractEntry:

    def test_pipeline_contract_includes_step_5b(self):
        """Step 5b must appear in PIPELINE_CONTRACT with its canonical id."""
        step = _find_step("step_5b_validate_emissions")
        assert step is not None, (
            "Expected a `step_5b_validate_emissions` entry in PIPELINE_CONTRACT"
        )
        assert step.step_label.startswith("Step 5b"), (
            "Step 5b label should start with 'Step 5b' for human-readable CLI "
            f"output; got {step.step_label!r}"
        )
        assert step.slash_command == "mellea-fy-validate-emissions.md"

    def test_step_5b_inputs_include_step_5_outputs(self):
        """Inputs should cover the Step 5 Python emissions plus the
        canonical Mellea API surface (which the in-session lint reasons
        over). We do not require every conditional file — only the
        recurring ones (`pipeline.py`, `requirements.py`, `slots.py`,
        `tools.py`, `constrained_slots.py`, `loader.py`) and
        `intermediate/mellea_api_ref.json`."""
        step = _find_step("step_5b_validate_emissions")
        input_names = {a.name for a in step.inputs}
        # Mellea-stdlib call sites live primarily in these files.
        must_have = {
            "pipeline.py",
            "requirements.py",
            "slots.py",
            "tools.py",
            "constrained_slots.py",
            "loader.py",
            "intermediate/mellea_api_ref.json",
        }
        missing = must_have - input_names
        assert not missing, (
            f"Step 5b is missing expected inputs: {sorted(missing)}; "
            f"got {sorted(input_names)}"
        )

    def test_step_5b_output_is_step_5b_report(self):
        step = _find_step("step_5b_validate_emissions")
        output_names = {a.name for a in step.outputs}
        assert "intermediate/step_5b_report.json" in output_names, (
            "Step 5b must produce `intermediate/step_5b_report.json` "
            "(the schema-bound structured evidence file)."
        )
        # Schema path should resolve.
        report = next(
            a for a in step.outputs if a.name == "intermediate/step_5b_report.json"
        )
        assert report.schema_path, (
            "step_5b_report.json must declare a schema_path so the "
            "contract verifier can locate the schema file."
        )

    def test_step_5b_ordering_between_step_5_and_step_6(self):
        """Topological order MUST place Step 5b after Step 5 and before
        Step 6. Independent steps could appear interleaved, but Step 5b
        consumes Step 5 outputs and Step 6 reads pipeline.py (which
        Step 5b may have edited in place)."""
        order = topological_order(PIPELINE_CONTRACT)
        assert "step_5b_validate_emissions" in order
        idx_5b = order.index("step_5b_validate_emissions")
        idx_5 = order.index("step_5_generate")
        idx_6 = order.index("step_6_artefacts")
        assert idx_5 < idx_5b, (
            f"step_5_generate (idx={idx_5}) must precede "
            f"step_5b_validate_emissions (idx={idx_5b}); order: {order}"
        )
        assert idx_5b < idx_6, (
            f"step_5b_validate_emissions (idx={idx_5b}) must precede "
            f"step_6_artefacts (idx={idx_6}); order: {order}"
        )


class TestStep5bReportSchema:

    def test_step_5b_output_schema_exists(self):
        """The schema file must be present on disk and parse as JSON."""
        assert SCHEMA_PATH.exists(), (
            f"Expected schema at {SCHEMA_PATH}; missing"
        )
        # Parses as JSON.
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        # Looks like a JSON Schema document.
        assert "type" in data or "$ref" in data
        # jsonschema can validate the schema's metaschema.
        jsonschema.Draft7Validator.check_schema(data)

    def test_step_5b_report_schema_validates_minimal_pass_report(self):
        schema = _load_schema()
        report = {
            "format_version": "1.0",
            "verdict": "pass",
            "files_checked": ["pipeline.py", "requirements.py"],
            "fixes_applied": [],
        }
        jsonschema.validate(report, schema)

    def test_step_5b_report_schema_validates_fixed_report(self):
        schema = _load_schema()
        report = {
            "format_version": "1.0",
            "verdict": "fixed",
            "files_checked": ["pipeline.py", "requirements.py", "slots.py"],
            "call_sites_enumerated": 23,
            "fixes_applied": [
                {
                    "file": "requirements.py",
                    "line": 30,
                    "issue": "stdlib-arity",
                    "fix_summary": (
                        "rewrote check(req) (1 positional arg) to "
                        "req(description) factory call"
                    ),
                },
                {
                    "file": "slots.py",
                    "line": 4,
                    "issue": "import-soundness",
                    "fix_summary": (
                        "rewrote `from mellea.backends.tools import ...` "
                        "to `from mellea.stdlib.tools import ...`"
                    ),
                },
            ],
            "remaining_issues": [],
        }
        jsonschema.validate(report, schema)

    def test_step_5b_report_schema_validates_fail_report(self):
        schema = _load_schema()
        report = {
            "format_version": "1.0",
            "verdict": "fail",
            "files_checked": ["pipeline.py"],
            "fixes_applied": [],
            "remaining_issues": [
                {
                    "file": "pipeline.py",
                    "line": 51,
                    "issue": "stdlib-arity",
                    "reason_unfixable": (
                        "canonical signature not in mellea_api_ref.json; "
                        "Step 7 will be authoritative"
                    ),
                }
            ],
        }
        jsonschema.validate(report, schema)

    def test_step_5b_report_schema_rejects_missing_required_fields(self):
        """Missing `format_version` → ValidationError. The schema is the
        contract; the wrapper-side / downstream consumer relies on every
        required field being present."""
        schema = _load_schema()
        broken_report = {
            # missing "format_version"
            "verdict": "pass",
            "files_checked": [],
            "fixes_applied": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken_report, schema)

    def test_step_5b_report_schema_rejects_invalid_verdict(self):
        """A verdict outside the enum must be rejected — the gate has
        exactly three legal states (pass / fixed / fail)."""
        schema = _load_schema()
        report = {
            "format_version": "1.0",
            "verdict": "ok",  # not in enum
            "files_checked": [],
            "fixes_applied": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(report, schema)

    def test_step_5b_report_schema_rejects_fix_entry_missing_field(self):
        """Each `fixes_applied` entry has its own required-field set; a
        missing `fix_summary` (or `file`/`line`/`issue`) must be
        rejected."""
        schema = _load_schema()
        report = {
            "format_version": "1.0",
            "verdict": "fixed",
            "files_checked": ["requirements.py"],
            "fixes_applied": [
                {
                    "file": "requirements.py",
                    "line": 4,
                    "issue": "stdlib-arity",
                    # missing "fix_summary"
                }
            ],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(report, schema)


class TestStep5bVerifierIntegration:

    def test_pipeline_contract_verifies_after_step_5b_added(self):
        """verify_contract() must return ok=True against the real
        contract after the Step 5b addition. Warnings are tolerated
        (existing slash-command-doc-drift findings on Step 1 / Step 3 are
        independent of Step 5b)."""
        result = verify_contract(PIPELINE_CONTRACT, repo_root=REPO_ROOT)
        if result.errors:
            pytest.fail(
                "real PIPELINE_CONTRACT (with Step 5b) failed verification:\n"
                + "\n".join(result.errors)
            )
        assert result.ok is True
