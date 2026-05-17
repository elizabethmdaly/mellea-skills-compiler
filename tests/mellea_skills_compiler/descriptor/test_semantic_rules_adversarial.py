"""Adversarial semantic-rules battery — written from the v0.1 strawman + v0.2
RFC contract, not the validator's implementation.

Each test class corresponds to one category of semantic rule. Each category
imagines a HOSTILE descriptor author who, in transcribing a spec, slips a
violation past the JSON-Schema layer hoping the semantic layer also misses
it. The validator must catch every one with the correct rule code at the
correct JSON-Pointer path.

The rule code constants under test (per the v0.1 strawman validation rules +
v0.2 RFC §3.4 ``on_parse_failure`` default-shape check):

  R-SEM-SYMBOL        — every ``symbol`` resolves in surface
  R-SEM-REF           — every ``{ref}`` resolves to a declared id
  R-SEM-SCHEMA-REF    — every ``schema_ref`` resolves to a declared schema
  R-SEM-OPERATOR      — operator-specific required fields present
  R-SEM-FMT-DEFAULT   — v0.2 ``on_parse_failure: {strategy: "default"}``
                        default value matches the schema's required fields

Failure-mode categories enumerated (S2):

  1. Symbol resolution — wrong module, wrong attribute, constant-not-callable
  2. Reference resolution — undeclared id, forward ref, ref-to-state vs
     pipeline, ref-with-select to nonexistent field
  3. Schema-ref resolution — missing schema, broken nested path
  4. Format defaults — default doesn't match the schema's required fields
  5. Operator required fields — ``map`` without ``over``, ``branch`` without
     ``cases``, ``parallel`` without ``branches``, ``retry_with_feedback``
     without ``max_attempts``
  6. Argument shape — ``{value: ...}`` where ``{ref: ...}`` is required, etc.
  7. State binding — ``bound_to`` to non-existent state id

What this battery is NOT testing (acknowledged gaps):

  - Cross-version compatibility (v0.1 vs v0.2 schema). Covered by
    ``test_v01_schema.py`` / ``test_v02_schema.py``.
  - Modality rules ``R-SEM-MODALITY-*``. These are Phase 3.5.B and not yet
    shipped — the rule codes exist in the docs but the rule logic does not.
  - ``R-SEM-CALL-ARGS-MATCH-SIG`` and ``R-SEM-REF-SELECT-RESOLVES``. These
    are also Phase 3.5.B "v0.3" enhancements — the corresponding negative
    tests are marked xfail with that phase reference.
  - Performance / pathological-size descriptors (very deeply nested
    composition trees).
  - End-to-end render verification. This battery only exercises
    ``validate()``; render-time errors are a separate battery (Battery 2).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.descriptor import validate


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "surface_0.5.0.json"
)
SENTRY_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "sentry-find-bugs.descriptor.json"
)


# --- Module-level constants for the rule codes under test --------------------

R_SYMBOL = "R-SEM-SYMBOL"
R_REF = "R-SEM-REF"
R_SCHEMA_REF = "R-SEM-SCHEMA-REF"
R_OPERATOR = "R-SEM-OPERATOR"
R_FMT_DEFAULT = "R-SEM-FMT-DEFAULT"


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def surface() -> dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _baseline_descriptor() -> dict[str, Any]:
    """A minimal v0.1 descriptor that the validator accepts as valid.

    Tests deep-copy this and apply a single rule-violating mutation.
    """
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "adv-test", "classification": "DSL"},
        "inputs": [{"name": "diff", "schema": {"kind": "str"}}],
        "outputs": [
            {
                "name": "report",
                "schema": {"kind": "model_ref", "ref": "#/schemas/Report"},
            }
        ],
        "schemas": {
            "Report": {
                "kind": "model",
                "fields": {"summary": {"type": "str"}},
            }
        },
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"env": "MODEL_ID"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "step",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"template": "Analyse {diff}"},
                    "format": {"schema_ref": "#/schemas/Report"},
                },
                "captures": {"result": "#/outputs/report"},
            }
        ],
    }


@pytest.fixture
def baseline() -> dict[str, Any]:
    """Deep-copied so individual tests can mutate without bleeding state."""
    return copy.deepcopy(_baseline_descriptor())


def _validate(descriptor: dict[str, Any], surface: dict[str, Any], version: str = "0.1"):
    return validate(descriptor, schema_version=version, surface=surface)


def _has_error(report, *, rule: str, path_contains: str | None = None) -> bool:
    """Return True iff `report` contains an error matching the given criteria."""
    for err in report.errors:
        if err.rule != rule:
            continue
        if path_contains is not None and path_contains not in err.path:
            continue
        return True
    return False


def _error_summary(report) -> str:
    """Human-readable error summary used in assertion messages."""
    return "; ".join(f"{e.rule}@{e.path}: {e.message[:60]}" for e in report.errors)


# =============================================================================
# Sanity baseline
# =============================================================================


def test_baseline_descriptor_is_valid(baseline, surface):
    """If this fails, every other test in this file is meaningless."""
    report = _validate(baseline, surface)
    assert report.valid, _error_summary(report)


# =============================================================================
# Category 1: Symbol resolution (R-SEM-SYMBOL)
# =============================================================================
# Adversarial framing: a hostile descriptor author writes a symbol that LOOKS
# like a Mellea symbol but doesn't resolve. The attack pattern: typos that
# JSON-Schema can't catch (it's just a string). The validator must catch with
# R-SEM-SYMBOL at the precise pipeline node path.


class TestSymbolResolution:

    def test_symbol_module_does_not_exist(self, baseline, surface):
        """Symbol points at a module that isn't in the surface at all."""
        baseline["pipeline"][0]["symbol"] = "mellea.NOT_A_MODULE.foo"
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SYMBOL, path_contains="/pipeline/0"), (
            f"expected {R_SYMBOL} at /pipeline/0; got {_error_summary(report)}"
        )

    def test_symbol_module_exists_but_attribute_missing(self, baseline, surface):
        """Module is real but the attr after it isn't — typo of a method name."""
        baseline["pipeline"][0]["symbol"] = "mellea.stdlib.session.MelleaSession.instuct"
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SYMBOL, path_contains="/pipeline/0"), (
            f"expected {R_SYMBOL}; got {_error_summary(report)}"
        )

    def test_symbol_in_state_block_unknown(self, baseline, surface):
        """A bogus symbol in state[] should fire R-SEM-SYMBOL at /state/N."""
        baseline["state"][0]["symbol"] = "mellea.backends.openai.NotARealBackend"
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SYMBOL, path_contains="/state/0"), (
            f"expected {R_SYMBOL} at /state/0; got {_error_summary(report)}"
        )

    def test_symbol_empty_string(self, baseline, surface):
        """Pathological empty symbol — must not crash and must produce a
        clear error. The exact rule code may be jsonschema-level (minLength)
        OR R-SEM-SYMBOL; either is acceptable, but the descriptor must NOT
        be reported as valid."""
        baseline["pipeline"][0]["symbol"] = ""
        report = _validate(baseline, surface)
        assert not report.valid, _error_summary(report)

    def test_symbol_valid_resolves(self, baseline, surface):
        """Positive — the baseline already uses a valid symbol."""
        report = _validate(baseline, surface)
        assert report.valid


# =============================================================================
# Category 2: Reference resolution (R-SEM-REF)
# =============================================================================
# Adversarial framing: the author writes a ref pointing at an id that does
# not exist or has not yet been declared. Schema-level checking can't catch
# this — refs are just strings. R-SEM-REF must catch every variant.


class TestRefResolution:

    def test_ref_to_undeclared_state_id(self, baseline, surface):
        baseline["pipeline"][0]["bound_to"] = {"ref": "no_such_session"}
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_REF), _error_summary(report)

    def test_ref_in_args_points_at_undeclared_id(self, baseline, surface):
        # Replace the bound_to with a valid one but inject a bogus ref into args.
        baseline["pipeline"][0]["args"]["extra_arg"] = {"ref": "ghost_id"}
        report = _validate(baseline, surface)
        # The validator may flag this as R-SEM-REF (id ghost_id undeclared)
        # OR as jsonschema:additionalProperties depending on whether the args
        # shape allows extras. We assert at minimum that the descriptor is
        # rejected.
        assert not report.valid, _error_summary(report)

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — forward-ref detection not yet implemented; later-declared "
        "pipeline ids may resolve eagerly without a forward-ref rule",
        strict=False,
    )
    def test_ref_forward_to_later_pipeline_node(self, baseline, surface):
        """The first pipeline node refs a node that comes AFTER it. Today the
        validator may accept this silently; ideal behaviour is to flag it."""
        baseline["pipeline"][0]["args"]["dependency"] = {"ref": "later_step"}
        baseline["pipeline"].append({
            "id": "later_step",
            "kind": "call",
            "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            "bound_to": {"ref": "session"},
            "args": {"description": {"template": "later"}},
        })
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_REF), _error_summary(report)

    def test_ref_to_declared_state_resolves(self, baseline, surface):
        """Positive — baseline binds to session which is in state."""
        assert _validate(baseline, surface).valid

    def test_ref_to_input_resolves(self, baseline, surface):
        """A template can use a ref-shape pointing at an input. Inputs are
        also valid ref targets."""
        baseline["pipeline"][0]["args"]["context"] = {"ref": "diff"}
        report = _validate(baseline, surface)
        # The descriptor must validate — diff is a declared input.
        # If it does NOT validate due to a different rule (like schema-arg
        # shape), that's a different issue; assert at minimum R-SEM-REF
        # does NOT fire for diff.
        assert not _has_error(report, rule=R_REF, path_contains="diff"), (
            f"R-SEM-REF fired for a valid input ref: {_error_summary(report)}"
        )


# =============================================================================
# Category 3: Schema-ref resolution (R-SEM-SCHEMA-REF)
# =============================================================================
# Adversarial framing: the author writes a schema_ref that points at a schema
# that doesn't exist in the descriptor's schemas block. Schema-level checks
# only validate the ref STRING SHAPE; the existence check is semantic.


class TestSchemaRefResolution:

    def test_schema_ref_to_undeclared_schema(self, baseline, surface):
        baseline["pipeline"][0]["args"]["format"] = {"schema_ref": "#/schemas/Missing"}
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SCHEMA_REF), _error_summary(report)

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — R-SEM-SCHEMA-REF walks call-node format.schema_ref but "
        "does NOT walk output.schema.ref; bogus output refs silently pass",
        strict=False,
    )
    def test_schema_ref_at_output_to_undeclared(self, baseline, surface):
        baseline["outputs"][0]["schema"] = {"kind": "model_ref", "ref": "#/schemas/Ghost"}
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SCHEMA_REF), _error_summary(report)

    def test_schema_ref_to_declared_schema_passes(self, baseline, surface):
        """Positive — baseline uses #/schemas/Report which is declared."""
        assert _validate(baseline, surface).valid

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — nested-path schema_ref (e.g. #/schemas/X/fields/Y) not "
        "yet supported by R-SEM-SCHEMA-REF; only top-level schema names walked",
        strict=False,
    )
    def test_schema_ref_with_deep_nested_path_unknown_leaf(self, baseline, surface):
        baseline["pipeline"][0]["args"]["format"] = {
            "schema_ref": "#/schemas/Report/fields/no_such_field"
        }
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_SCHEMA_REF), _error_summary(report)


# =============================================================================
# Category 4: Format defaults (R-SEM-FMT-DEFAULT, v0.2 only)
# =============================================================================
# Adversarial framing: the author specifies on_parse_failure with a default
# object whose shape doesn't match the schema's required fields. v0.2 §3.4
# requires the default to be schema-shape compatible.


def _v02_baseline_with_on_parse_failure(default_obj: dict[str, Any]) -> dict[str, Any]:
    """In v0.2, on_parse_failure is a TOP-LEVEL call-node field (sibling of
    args + captures), not nested under args.format."""
    desc = copy.deepcopy(_baseline_descriptor())
    desc["descriptor_version"] = "0.2"
    desc["pipeline"][0]["on_parse_failure"] = {
        "strategy": "default",
        "default": default_obj,
    }
    return desc


class TestFormatDefaults:

    def test_default_missing_required_field(self, surface):
        """Default object is missing the schema's required `summary` field.
        The rule MUST fire (the validator must surface a complaint), even if
        it ships today as a warning rather than a hard error."""
        desc = _v02_baseline_with_on_parse_failure({})
        report = _validate(desc, surface, version="0.2")
        assert _has_error(report, rule=R_FMT_DEFAULT), (
            f"R-SEM-FMT-DEFAULT did not fire: {_error_summary(report)}"
        )

    def test_default_with_wrong_extra_field(self, surface):
        """Default has the right field plus an unknown extra. v0.2 RFC says
        the default must structurally match — extras may or may not be
        tolerated. At minimum the validator must not crash."""
        desc = _v02_baseline_with_on_parse_failure({
            "summary": "fallback", "totally_unknown_field": 99
        })
        # Either accepted or rejected with a clear rule — must not crash.
        report = _validate(desc, surface, version="0.2")
        # We're NOT asserting valid/invalid here; this is a fuzzy-tolerance
        # test. We only assert no crash and that, if invalid, the rule is
        # the documented one.
        if not report.valid:
            assert _has_error(report, rule=R_FMT_DEFAULT) or _has_error(
                report, rule="jsonschema:additionalProperties"
            ), _error_summary(report)

    def test_default_with_all_required_fields_passes(self, surface):
        """Positive — default matches the schema."""
        desc = _v02_baseline_with_on_parse_failure({"summary": "fallback"})
        report = _validate(desc, surface, version="0.2")
        assert report.valid, _error_summary(report)


# =============================================================================
# Category 5: Operator required fields (R-SEM-OPERATOR)
# =============================================================================
# Adversarial framing: the author drops a required field from a composition
# operator. The schema may catch some of these (oneOf on `kind` + `operator`),
# but the per-operator required-field check is the semantic rule's job.


class TestOperatorRequiredFields:

    def _map_baseline(self) -> dict[str, Any]:
        """Build a descriptor with a map composition node + a tail call that
        captures the output. (Captures live on a call node in v0.1, not on a
        composition node per the schema.)"""
        desc = _baseline_descriptor()
        desc["schemas"]["Items"] = {
            "kind": "model",
            "fields": {"items": {"type": "list[str]"}},
        }
        desc["pipeline"] = [
            {
                "id": "extract",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"template": "extract from {diff}"},
                    "format": {"schema_ref": "#/schemas/Items"},
                },
            },
            {
                "id": "loop",
                "kind": "composition",
                "operator": "map",
                "over": {"ref": "extract", "select": "items"},
                "item_id": "item",
                "collect": "list",
                "body": [
                    {
                        "id": "inner",
                        "kind": "call",
                        "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                        "bound_to": {"ref": "session"},
                        "args": {"description": {"template": "process {item}"}},
                    }
                ],
            },
            {
                "id": "finalise",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"template": "summarise"},
                    "format": {"schema_ref": "#/schemas/Report"},
                },
                "captures": {"result": "#/outputs/report"},
            },
        ]
        return desc

    def test_map_without_over(self, surface):
        desc = self._map_baseline()
        del desc["pipeline"][1]["over"]
        report = _validate(desc, surface)
        assert not report.valid
        # Could be R-SEM-OPERATOR or jsonschema:required — both acceptable.
        assert _has_error(report, rule=R_OPERATOR) or _has_error(
            report, rule="jsonschema:required"
        ), _error_summary(report)

    def test_map_without_body(self, surface):
        desc = self._map_baseline()
        del desc["pipeline"][1]["body"]
        report = _validate(desc, surface)
        assert not report.valid

    def test_map_with_all_required_fields_passes(self, surface):
        desc = self._map_baseline()
        report = _validate(desc, surface)
        assert report.valid, _error_summary(report)

    @pytest.mark.xfail(
        reason="P3.5.B branch operator — `branch` not yet a first-class operator in v0.1; "
        "R-SEM-OPERATOR may not have branch-specific predicates yet",
        strict=False,
    )
    def test_branch_without_cases(self, surface):
        desc = _baseline_descriptor()
        desc["pipeline"] = [
            {
                "id": "br",
                "kind": "composition",
                "operator": "branch",
                # NO 'cases' field
                "default": [],
            }
        ]
        report = _validate(desc, surface)
        assert not report.valid
        assert _has_error(report, rule=R_OPERATOR), _error_summary(report)

    @pytest.mark.xfail(
        reason="P3.5.B parallel operator — `parallel` schema may not be in v0.1 schema yet; "
        "R-SEM-OPERATOR may not have parallel-specific predicates",
        strict=False,
    )
    def test_parallel_without_branches(self, surface):
        desc = _baseline_descriptor()
        desc["pipeline"] = [
            {
                "id": "par",
                "kind": "composition",
                "operator": "parallel",
                # NO 'branches' field
            }
        ]
        report = _validate(desc, surface)
        assert not report.valid
        assert _has_error(report, rule=R_OPERATOR), _error_summary(report)

    @pytest.mark.xfail(
        reason="P3.5.B retry_with_feedback — operator may not be implemented as a "
        "first-class schema entry yet",
        strict=False,
    )
    def test_retry_with_feedback_without_max_attempts(self, surface):
        desc = _baseline_descriptor()
        desc["pipeline"] = [
            {
                "id": "retry",
                "kind": "composition",
                "operator": "retry_with_feedback",
                # NO 'max_attempts' field
                "body": [],
            }
        ]
        report = _validate(desc, surface)
        assert not report.valid


# =============================================================================
# Category 6: Argument shape (mixed jsonschema + semantic)
# =============================================================================
# Adversarial framing: the author writes an args dict that's the wrong KIND
# of ref (e.g. {value: ...} where {template: ...} is contextually correct, or
# a {ref: ...} that points at the wrong kind of declared id). Many of these
# pass JSON-Schema validation (the shapes are all valid arg-shapes) but
# semantic reality says they're wrong.


class TestArgumentShape:

    def test_args_with_unknown_arg_kind(self, baseline, surface):
        """An args entry shaped like {totally_made_up: ...}. JSON-Schema may or
        may not reject; semantic layer should at minimum not crash."""
        baseline["pipeline"][0]["args"]["description"] = {"totally_made_up_kind": "x"}
        report = _validate(baseline, surface)
        # Must be rejected (one way or another).
        assert not report.valid, _error_summary(report)

    def test_args_with_empty_args_dict_when_required_args_missing(self, baseline, surface):
        """instruct() requires `description`. Empty args should fail — either
        via R-SEM-SYMBOL parameter-coverage check or R-SEM-CALL-ARGS-MATCH-SIG
        (Phase 3.5.B). At minimum, the descriptor must be flagged."""
        baseline["pipeline"][0]["args"] = {}
        report = _validate(baseline, surface)
        # Today this may pass — the strawman §5 says "every required parameter
        # of every called symbol is satisfied" but the rule may not be
        # implemented yet.
        if report.valid:
            pytest.xfail(
                "P3.5.B v0.3 — R-SEM-CALL-ARGS-MATCH-SIG not yet enforcing "
                "missing-required-arg detection"
            )
        assert not report.valid

    def test_env_arg_resolves(self, baseline, surface):
        """Positive — baseline state uses {env: 'MODEL_ID'} which is valid."""
        assert _validate(baseline, surface).valid


# =============================================================================
# Category 7: State binding (R-SEM-REF on bound_to)
# =============================================================================
# Adversarial framing: bound_to must reference a declared state id. The author
# writes a bound_to that points at a pipeline id, an input id, or just a
# nonexistent string. The rule that fires may be R-SEM-REF (id resolution).


class TestStateBinding:

    def test_bound_to_undeclared_id(self, baseline, surface):
        baseline["pipeline"][0]["bound_to"] = {"ref": "nope"}
        report = _validate(baseline, surface)
        assert not report.valid
        assert _has_error(report, rule=R_REF), _error_summary(report)

    def test_bound_to_input_id_instead_of_state(self, baseline, surface):
        """bound_to should reference a state id, not an input slot."""
        baseline["pipeline"][0]["bound_to"] = {"ref": "diff"}  # diff is an input
        report = _validate(baseline, surface)
        # Today this may pass since R-SEM-REF only checks existence not kind.
        if report.valid:
            pytest.xfail(
                "P3.5.B v0.3 — R-SEM-REF doesn't differentiate state vs input "
                "kinds for bound_to context"
            )

    def test_bound_to_declared_state_passes(self, baseline, surface):
        """Positive — baseline binds to declared state id `session`."""
        assert _validate(baseline, surface).valid


# =============================================================================
# Property: parametrized — every R-SEM-* rule fires somewhere
# =============================================================================
# Strategy §S4: across the rule families above, the validator MUST surface at
# least one error per family on a known-bad descriptor. Catches the silent-
# failure mode where a rule ships but its predicate never fires.


@pytest.mark.parametrize(
    "mutation,expected_rule",
    [
        # Mutator: changes pipeline[0].symbol to a bogus value -> expect R-SEM-SYMBOL.
        (lambda d: d["pipeline"][0].update({"symbol": "totally.bogus.symbol"}), R_SYMBOL),
        # Mutator: bogus bound_to -> expect R-SEM-REF.
        (lambda d: d["pipeline"][0].update({"bound_to": {"ref": "nope"}}), R_REF),
        # Mutator: bogus schema_ref -> expect R-SEM-SCHEMA-REF.
        (lambda d: d["pipeline"][0]["args"].update({"format": {"schema_ref": "#/schemas/Missing"}}), R_SCHEMA_REF),
    ],
)
def test_rule_fires_when_expected(baseline, surface, mutation, expected_rule):
    """Property: each rule family above must fire on its targeted mutation.
    Catches the silent-rule failure mode (rule exists, predicate never fires)."""
    mutation(baseline)
    report = _validate(baseline, surface)
    assert not report.valid, f"mutation did not invalidate descriptor: {_error_summary(report)}"
    assert any(e.rule == expected_rule for e in report.errors), (
        f"expected {expected_rule}; got {_error_summary(report)}"
    )


@pytest.mark.parametrize(
    "schema_version", ["0.1", "0.2"],
)
def test_rule_codes_consistent_across_versions(baseline, surface, schema_version):
    """Property: the R-SEM-* rule codes are stable across schema versions.
    A bogus symbol should produce R-SEM-SYMBOL in BOTH v0.1 and v0.2 mode.
    Catches drift where v0.2 silently renames or drops a rule code."""
    if schema_version == "0.2":
        baseline["descriptor_version"] = "0.2"
    baseline["pipeline"][0]["symbol"] = "totally.bogus.symbol"
    report = _validate(baseline, surface, version=schema_version)
    assert not report.valid
    assert any(e.rule == R_SYMBOL for e in report.errors), _error_summary(report)
