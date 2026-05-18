"""Pipeline contract registry — single source of truth for stage I/O.

Each entry declares what a pipeline step consumes and produces, plus the
schema (if any) that governs each artefact. The registry powers:

* A dependency graph (this module's :func:`build_dependency_graph`,
  :func:`to_mermaid`, :func:`topological_order`).
* A static checker (:func:`verify_contract`) — see the CLI subcommand
  ``mellea-skills contract verify``.
* Future: runtime validation at step boundaries (P3 follow-up) and
  misalignment-shape tests (P4).

The contract is the canonical declaration. Slash-command markdown docs
and per-artefact JSON schemas must agree with it. The verifier flags
disagreements.

How to update the contract when a slash command changes
-------------------------------------------------------

1.  Update the matching :class:`StepContract` entry in
    :data:`PIPELINE_CONTRACT` below — add/remove
    :class:`ArtefactRef` entries to match the slash command's declared
    inputs/outputs.
2.  If a new schema was introduced, add it to ``.claude/schemas/`` (or
    ``src/mellea_skills_compiler/descriptor/schemas/``) and reference it
    via ``schema_path``.
3.  Run ``mellea-skills contract verify`` — the static checker confirms
    schemas resolve on disk and producer/consumer references stay aligned.
4.  Add / update tests under
    ``tests/mellea_skills_compiler/compile/test_pipeline_contract.py``.

Step-boundary edge cases (acknowledged)
---------------------------------------

* The orchestrator's :data:`_STEP_ARTEFACTS` (in ``mellea_skills.py``)
  treats Step 1a + 1b as a single resume step keyed on
  ``intermediate/inventory.json``. The contract preserves the slash
  command's two-phase logical view but pairs them under a single
  step entry (``step_1_inventory``) keyed on the artefact that is
  actually persisted on disk.
* Step 2 emits two artefacts (``element_mapping.json`` and
  ``expected_signature.json``). The latter is governed by a schema in
  ``src/mellea_skills_compiler/descriptor/schemas/`` rather than
  ``.claude/schemas/``.
* The two wrapper-side synthesizers in
  :mod:`mellea_skills_compiler.compile.writer_renderer`
  (``synthesize_config_emission_from_directive`` and
  ``synthesize_fixtures_emission_from_existing``) are modelled as
  separate ``step_4b_synthesize_*`` entries with no matching
  slash command — the verifier's slash-command cross-check skips them
  via the ``slash_command`` attribute on :class:`StepContract`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


# ---------------------------------------------------------------------------
# Core data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtefactRef:
    """A named artefact plus an optional schema reference.

    ``name`` is the canonical artefact identifier (e.g.
    ``"intermediate/classification.json"``, ``"spec_md"``,
    ``"<package>/pipeline.py"``). It should match what the producer
    actually writes and what the consumer actually reads.

    ``format`` is one of: ``"json"``, ``"markdown"``, ``"python"``,
    ``"yaml"``, ``"binary"``.

    ``schema_path`` is repo-relative path to the schema file. ``None``
    for schema-less artefacts (raw spec, markdown reports, etc).

    ``kind`` distinguishes:

      * ``"intermediate"``: lives in ``<package>/intermediate/``
      * ``"rendered"``: lives in ``<package>/`` (e.g. ``config.py``,
        ``pipeline.py``)
      * ``"input"``: consumed-but-not-produced-by-pipeline (e.g.
        ``spec.md``)
      * ``"package_meta"``: lives at skill-root (e.g. ``pyproject.toml``)
    """

    name: str
    format: Literal["json", "markdown", "python", "yaml", "binary"]
    schema_path: str | None = None
    kind: Literal[
        "intermediate", "rendered", "input", "package_meta"
    ] = "intermediate"


@dataclass(frozen=True)
class StepContract:
    """Declarative description of a single pipeline step.

    ``step_id`` is a stable identifier (e.g. ``"step_0_classify"``).
    ``step_label`` is the human-readable name printed by the CLI.
    ``slash_command`` (optional) names the slash-command file under
    ``.claude/commands/`` that documents this step — the verifier uses
    it for the textual cross-check. Synthesizer / wrapper-side steps
    leave it as ``None`` to skip the check.
    """

    step_id: str
    step_label: str
    inputs: tuple[ArtefactRef, ...]
    outputs: tuple[ArtefactRef, ...]
    notes: str = ""
    slash_command: str | None = None


# ---------------------------------------------------------------------------
# The canonical pipeline contract
# ---------------------------------------------------------------------------
#
# Steps are listed in declaration order (matches the documented
# 10-step workflow in `.claude/commands/mellea-fy.md`). The
# :func:`topological_order` helper recomputes the order from
# producer/consumer edges; the declaration order doubles as a
# tiebreaker for independent steps.

PIPELINE_CONTRACT: tuple[StepContract, ...] = (
    StepContract(
        step_id="step_pre_wrapper_prep",
        step_label=(
            "Pre-step (wrapper) — runtime directive + Mellea API ref + docs index"
        ),
        # No slash-command — the wrapper writes these before the
        # `/mellea-fy` session starts. Documented in
        # `mellea-fy-deps.md` §2.5e and §2.5f, and in
        # `claude_directives.py::write_runtime_directive`.
        slash_command=None,
        inputs=(
            ArtefactRef(name="spec_md", format="markdown", kind="input"),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/runtime_directive.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/mellea_api_ref.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/mellea_doc_index.json",
                format="json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Deterministic wrapper preparation. `runtime_directive.json` "
            "is written by `claude_directives.write_runtime_directive`; "
            "`mellea_api_ref.json` and `mellea_doc_index.json` are "
            "written by `grounding.write_mellea_api_ref` / "
            "`grounding.write_mellea_doc_index`. Slash command Step 2.5 "
            "only VERIFIES these files exist."
        ),
    ),
    StepContract(
        step_id="step_0_classify",
        step_label="Step 0 — 5-axis classification",
        slash_command="mellea-fy-classify.md",
        inputs=(
            ArtefactRef(name="spec_md", format="markdown", kind="input"),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/classification.json",
                format="json",
                schema_path=".claude/schemas/classification.schema.json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Five-axis archetype detection. Drives entry-point shape "
            "(Step 3), category defaults (Step 2.5), and modality "
            "validation (Step 7)."
        ),
    ),
    StepContract(
        step_id="step_1_inventory",
        step_label="Step 1a + 1b — file inventory + element tagging",
        slash_command="mellea-fy-inventory.md",
        inputs=(
            ArtefactRef(name="spec_md", format="markdown", kind="input"),
            ArtefactRef(
                name="intermediate/classification.json",
                format="json",
                schema_path=".claude/schemas/classification.schema.json",
                kind="intermediate",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/inventory.json",
                format="json",
                schema_path=".claude/schemas/inventory.schema.json",
                kind="intermediate",
            ),
            # Coverage report + step_1b_trace are advisory side artefacts;
            # they have no schema but are documented in mellea-fy-inventory.md.
            ArtefactRef(
                name="intermediate/coverage_report.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/step_1b_trace.json",
                format="json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Two-pass element tagging — 17 tags + 9 C-categories. "
            "Coverage target: ≥95% of significant source lines."
        ),
    ),
    StepContract(
        step_id="step_2_map",
        step_label="Step 2 — element-to-Mellea-primitive mapping",
        slash_command="mellea-fy-map.md",
        inputs=(
            ArtefactRef(
                name="intermediate/inventory.json",
                format="json",
                schema_path=".claude/schemas/inventory.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/classification.json",
                format="json",
                schema_path=".claude/schemas/classification.schema.json",
                kind="intermediate",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/expected_signature.json",
                format="json",
                schema_path=(
                    "src/mellea_skills_compiler/descriptor/"
                    "schemas/expected_signature.schema.json"
                ),
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping_judgment_calls.json",
                format="json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Tag → primitive routing (Tag-to-primitive table). "
            "TOOL_TEMPLATE entries land as provisional "
            "`pending_step_2.5`. P3.5.D emits "
            "`expected_signature.json` here."
        ),
    ),
    StepContract(
        step_id="step_2_5_deps",
        step_label="Step 2.5 — dependency audit, elicitation, API refs",
        slash_command="mellea-fy-deps.md",
        inputs=(
            ArtefactRef(
                name="intermediate/inventory.json",
                format="json",
                schema_path=".claude/schemas/inventory.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/dependency_plan.json",
                format="json",
                schema_path=".claude/schemas/dependency_plan.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping_amendments.json",
                format="json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Deterministic — no LLM. Sub-steps 2.5a-d (audit, "
            "elicitation, commit, amend). 2.5e/2.5f are slash-command "
            "*verification* steps; the underlying "
            "`mellea_api_ref.json` and `mellea_doc_index.json` "
            "artefacts are written by the wrapper in `step_pre_wrapper_prep`."
        ),
    ),
    StepContract(
        step_id="step_3_skeleton",
        step_label="Step 3 — skeleton emission + companion-dir mirror",
        slash_command="mellea-fy-generate.md",
        inputs=(
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping_amendments.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/dependency_plan.json",
                format="json",
                schema_path=".claude/schemas/dependency_plan.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/classification.json",
                format="json",
                schema_path=".claude/schemas/classification.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/expected_signature.json",
                format="json",
                schema_path=(
                    "src/mellea_skills_compiler/descriptor/"
                    "schemas/expected_signature.schema.json"
                ),
                kind="intermediate",
            ),
        ),
        outputs=(
            # Skeleton Python files — populated in Step 5.
            ArtefactRef(name="pipeline.py", format="python", kind="rendered"),
            ArtefactRef(name="schemas.py", format="python", kind="rendered"),
            ArtefactRef(name="main.py", format="python", kind="rendered"),
            ArtefactRef(name="__init__.py", format="python", kind="rendered"),
            ArtefactRef(name="__main__.py", format="python", kind="rendered"),
            # Skeleton manifest — finalised in Step 6.
            ArtefactRef(
                name="melleafy.json",
                format="json",
                schema_path=".claude/schemas/melleafy.schema.json",
                kind="rendered",
            ),
            # Package-root metadata (lives at skill root, not in <package>/).
            ArtefactRef(
                name="pyproject.toml",
                format="yaml",  # TOML closest to YAML in this enum.
                kind="package_meta",
            ),
        ),
        notes=(
            "Locks the `run_pipeline` signature (Rule 3-1, 3-2, 3-3). "
            "Companion directories (`scripts/`, `references/`, "
            "`assets/`) are mirrored deterministically by the wrapper "
            "before this step begins."
        ),
    ),
    StepContract(
        step_id="step_4_fixtures",
        step_label="Step 4 — fixtures emission + writer",
        slash_command="mellea-fy-fixtures.md",
        inputs=(
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/expected_signature.json",
                format="json",
                schema_path=(
                    "src/mellea_skills_compiler/descriptor/"
                    "schemas/expected_signature.schema.json"
                ),
                kind="intermediate",
            ),
            # Reads pipeline.py to learn the locked signature when
            # expected_signature.json is absent (legacy pre-P3.5.D path).
            ArtefactRef(name="pipeline.py", format="python", kind="rendered"),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/fixtures_emission.json",
                format="json",
                schema_path=".claude/schemas/fixtures_emission.schema.json",
                kind="intermediate",
            ),
            # The writer renders fixtures/ from the JSON above. We
            # model fixtures/__init__.py as the canonical evidence
            # that the writer ran.
            ArtefactRef(
                name="fixtures/__init__.py",
                format="python",
                kind="rendered",
            ),
        ),
        notes=(
            "Rule 4-1 — model emits JSON only; deterministic writer "
            "renders fixtures/*.py."
        ),
    ),
    StepContract(
        step_id="step_4b_synthesize_fixtures_emission",
        step_label=(
            "Step 4b (wrapper) — synthesize fixtures_emission.json from "
            "existing fixtures/"
        ),
        # No slash-command — this is a writer-renderer fallback.
        slash_command=None,
        inputs=(
            ArtefactRef(
                name="fixtures/__init__.py",
                format="python",
                kind="rendered",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/fixtures_emission.json",
                format="json",
                schema_path=".claude/schemas/fixtures_emission.schema.json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Last-resort wrapper fallback. AST-parses surviving fixture "
            "files and reverse-engineers a schema-compliant emission "
            "before the writer's wipe-then-render runs. See "
            "`writer_renderer.synthesize_fixtures_emission_from_existing`."
        ),
    ),
    StepContract(
        step_id="step_5_generate",
        step_label="Step 5 — body / descriptor emission",
        slash_command="mellea-fy-generate.md",
        inputs=(
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/dependency_plan.json",
                format="json",
                schema_path=".claude/schemas/dependency_plan.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/mellea_api_ref.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/expected_signature.json",
                format="json",
                schema_path=(
                    "src/mellea_skills_compiler/descriptor/"
                    "schemas/expected_signature.schema.json"
                ),
                kind="intermediate",
            ),
            ArtefactRef(
                name="fixtures/__init__.py",
                format="python",
                kind="rendered",
            ),
            ArtefactRef(
                name="intermediate/runtime_directive.json",
                format="json",
                kind="intermediate",
            ),
        ),
        outputs=(
            # config.py is rendered by the writer from config_emission.json,
            # not by the LLM directly.
            ArtefactRef(
                name="intermediate/config_emission.json",
                format="json",
                schema_path=".claude/schemas/config_emission.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(name="config.py", format="python", kind="rendered"),
            # Step 3 emitted the skeletons for pipeline.py / schemas.py /
            # main.py / __init__.py / __main__.py; Step 5 populates the
            # bodies in place. We deliberately do NOT re-declare those as
            # outputs here — that would create a cycle in the graph
            # (Step 5 reading the locked signature from `pipeline.py` and
            # then producing `pipeline.py` is an in-place mutation, not a
            # distinct artefact). The contract treats Step 3 as the
            # canonical producer; downstream readers (Step 6, Step 7)
            # transitively consume the populated form.
            #
            # Step 5 IS the canonical producer of the conditional files
            # that were not present in the Step 3 skeleton.
            ArtefactRef(name="requirements.py", format="python", kind="rendered"),
            ArtefactRef(name="slots.py", format="python", kind="rendered"),
            ArtefactRef(name="tools.py", format="python", kind="rendered"),
            ArtefactRef(
                name="constrained_slots.py", format="python", kind="rendered"
            ),
            ArtefactRef(name="mobjects.py", format="python", kind="rendered"),
            ArtefactRef(name="loader.py", format="python", kind="rendered"),
        ),
        notes=(
            "Default path: free-form Python emission. With "
            "`--use-descriptor`: emits "
            "`intermediate/<package>.descriptor.json` and the renderer "
            "produces the same files. Step 3's skeleton files "
            "(`pipeline.py`, `schemas.py`, `main.py`, `__init__.py`, "
            "`__main__.py`) are populated in place — not re-emitted as "
            "fresh artefacts here. The contract therefore lists only "
            "Step 5's *newly emitted* files plus the `config_emission.json` "
            "IR consumed by the writer."
        ),
    ),
    StepContract(
        step_id="step_4b_synthesize_config_emission",
        step_label=(
            "Step 5b (wrapper) — synthesize config_emission.json from "
            "runtime_directive.json"
        ),
        slash_command=None,
        inputs=(
            ArtefactRef(
                name="intermediate/runtime_directive.json",
                format="json",
                kind="intermediate",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/config_emission.json",
                format="json",
                schema_path=".claude/schemas/config_emission.schema.json",
                kind="intermediate",
            ),
        ),
        notes=(
            "Wrapper fallback. Idempotent; short-circuits when the LLM "
            "already emitted `config_emission.json`. Minimal C8-only "
            "payload so the `runtime-defaults-bound` lint passes. "
            "See `writer_renderer.synthesize_config_emission_from_directive`."
        ),
    ),
    StepContract(
        step_id="step_5b_validate_emissions",
        step_label="Step 5b — In-session validate Step 5 emissions",
        slash_command="mellea-fy-validate-emissions.md",
        inputs=(
            ArtefactRef(name="pipeline.py", format="python", kind="rendered"),
            ArtefactRef(name="requirements.py", format="python", kind="rendered"),
            ArtefactRef(name="slots.py", format="python", kind="rendered"),
            ArtefactRef(name="tools.py", format="python", kind="rendered"),
            ArtefactRef(
                name="constrained_slots.py", format="python", kind="rendered"
            ),
            ArtefactRef(name="loader.py", format="python", kind="rendered"),
            ArtefactRef(
                name="intermediate/mellea_api_ref.json",
                format="json",
                kind="intermediate",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/step_5b_report.json",
                format="json",
                schema_path=(
                    "src/mellea_skills_compiler/descriptor/"
                    "schemas/step_5b_report.schema.json"
                ),
                kind="intermediate",
            ),
        ),
        notes=(
            "In-session structural-lint gate between Step 5 and Step 6. "
            "Applies a 9-lint subset (stdlib-arity, import-soundness, "
            "instruct-has-description, instruct-result-parse-before-access, "
            "format-annotation, validator-soundness, "
            "generative-forbidden-params, generative-call-passes-session, "
            "variable-safety) using the canonical Mellea surface at "
            "`intermediate/mellea_api_ref.json`. Best-effort in-place "
            "repair via the Edit tool; never halts the pipeline. The "
            "wrapper-side Step 7 lint suite (`mellea-fy-validate`) remains "
            "the authoritative gate — Step 5b is purely a faster-feedback "
            "layer for the empirically dominant `check(...)` arity / "
            "import-soundness defects. Deeper follow-up tracked as F9 "
            "(extending the descriptor IR to cover these files)."
        ),
    ),
    StepContract(
        step_id="step_6_artefacts",
        step_label="Step 6 — supporting artifacts (reports, manifests)",
        slash_command="mellea-fy-artifacts.md",
        inputs=(
            ArtefactRef(
                name="intermediate/classification.json",
                format="json",
                schema_path=".claude/schemas/classification.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/inventory.json",
                format="json",
                schema_path=".claude/schemas/inventory.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping.json",
                format="json",
                schema_path=".claude/schemas/element_mapping.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/element_mapping_amendments.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/dependency_plan.json",
                format="json",
                schema_path=".claude/schemas/dependency_plan.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(name="pipeline.py", format="python", kind="rendered"),
        ),
        outputs=(
            ArtefactRef(
                name="melleafy.json",
                format="json",
                schema_path=".claude/schemas/melleafy.schema.json",
                kind="rendered",
            ),
            ArtefactRef(name="mapping_report.md", format="markdown", kind="rendered"),
            ArtefactRef(name="README.md", format="markdown", kind="rendered"),
            ArtefactRef(name="SETUP.md", format="markdown", kind="rendered"),
            ArtefactRef(name="SKILL.md", format="markdown", kind="rendered"),
            ArtefactRef(
                name="dependencies.yaml",
                format="yaml",
                kind="rendered",
            ),
        ),
        notes=(
            "Finalises `melleafy.json:categories_resolved` "
            "deterministically from `dependency_plan.json`. SKILL.md "
            "is generated only for non-`.md` source runtimes."
        ),
    ),
    StepContract(
        step_id="step_7_validate",
        step_label="Step 7 — static validation (25 lints)",
        slash_command="mellea-fy-validate.md",
        inputs=(
            ArtefactRef(name="pipeline.py", format="python", kind="rendered"),
            ArtefactRef(name="config.py", format="python", kind="rendered"),
            ArtefactRef(
                name="melleafy.json",
                format="json",
                schema_path=".claude/schemas/melleafy.schema.json",
                kind="rendered",
            ),
            ArtefactRef(
                name="intermediate/dependency_plan.json",
                format="json",
                schema_path=".claude/schemas/dependency_plan.schema.json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/runtime_directive.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/mellea_api_ref.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="intermediate/mellea_doc_index.json",
                format="json",
                kind="intermediate",
            ),
            ArtefactRef(
                name="fixtures/__init__.py",
                format="python",
                kind="rendered",
            ),
        ),
        outputs=(
            ArtefactRef(
                name="intermediate/step_7_report.json",
                format="json",
                kind="intermediate",
            ),
        ),
        notes=(
            "25 lints across 3 tiers (parseability, structural, "
            "cross-artifact). Severity model: `error` blocks compile; "
            "`warning` / `info` surface as findings."
        ),
    ),
)


# Terminal outputs — these are unconsumed-by-design (final reports +
# rendered Python files that end users invoke, not pipeline-internal
# consumers). The verifier suppresses unconsumed-output warnings for
# this allowlist.
_TERMINAL_OUTPUTS: frozenset[str] = frozenset(
    {
        # Final reports
        "intermediate/step_5b_report.json",
        "intermediate/step_7_report.json",
        "intermediate/coverage_report.json",
        "intermediate/step_1b_trace.json",
        "intermediate/element_mapping_judgment_calls.json",
        # User-facing docs
        "mapping_report.md",
        "README.md",
        "SETUP.md",
        "SKILL.md",
        # Package metadata
        "pyproject.toml",
        "dependencies.yaml",
        # Rendered Python — the end-user / runtime consumes these, not
        # any other *pipeline* step. Lints inspect them inside
        # `step_7_validate`, but several conditional files (slots.py,
        # tools.py, etc.) are not necessarily required-input to lint.
        "main.py",
        "__init__.py",
        "__main__.py",
        "schemas.py",
        "requirements.py",
        "slots.py",
        "tools.py",
        "constrained_slots.py",
        "mobjects.py",
        "loader.py",
        # Emission IR consumed by the *renderer* (deterministic writers
        # at `.claude/melleafy/writers/*`), not by any later pipeline
        # step. Listed here so the verifier doesn't flag them as
        # accidentally unconsumed.
        "intermediate/fixtures_emission.json",
        "intermediate/config_emission.json",
    }
)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _materialize(
    contract: Iterable[StepContract],
) -> tuple[StepContract, ...]:
    """Coerce an iterable to a tuple so we can iterate twice."""
    return tuple(contract)


def build_dependency_graph(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> dict[str, list[str]]:
    """Return adjacency map: producer_step_id -> [consumer_step_id, ...].

    Two steps are connected if any of producer's outputs appears in
    consumer's inputs (matched by ``ArtefactRef.name``). Consumer lists
    are de-duplicated and ordered to match the contract's declaration
    order, so the graph is deterministic.
    """
    steps = _materialize(contract)
    by_id = {s.step_id: s for s in steps}
    declared_order = {s.step_id: idx for idx, s in enumerate(steps)}

    adjacency: dict[str, list[str]] = {s.step_id: [] for s in steps}
    for producer in steps:
        produced_names = {a.name for a in producer.outputs}
        if not produced_names:
            continue
        consumers: set[str] = set()
        for consumer in steps:
            if consumer.step_id == producer.step_id:
                continue
            consumed_names = {a.name for a in consumer.inputs}
            if produced_names & consumed_names:
                consumers.add(consumer.step_id)
        adjacency[producer.step_id] = sorted(
            consumers, key=lambda sid: declared_order[sid]
        )
    # Touch `by_id` to ensure no orphans remain in adjacency keys.
    assert set(adjacency.keys()) == set(by_id.keys())
    return adjacency


def topological_order(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> list[str]:
    """Return step_ids in dependency order. Raises if a cycle exists.

    Implementation: Kahn's algorithm. Independent steps preserve their
    declaration order (the ready queue is a deque populated in
    declaration order; we pop from the front).
    """
    from collections import deque

    steps = _materialize(contract)
    declared_order = {s.step_id: idx for idx, s in enumerate(steps)}
    adjacency = build_dependency_graph(steps)
    in_degree: dict[str, int] = {sid: 0 for sid in declared_order}
    for producer_id, consumers in adjacency.items():
        for consumer_id in consumers:
            in_degree[consumer_id] += 1

    ready = deque(
        sid for sid, _ in sorted(declared_order.items(), key=lambda kv: kv[1])
        if in_degree[sid] == 0
    )
    order: list[str] = []
    while ready:
        sid = ready.popleft()
        order.append(sid)
        for consumer_id in adjacency[sid]:
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                # Insert in declaration order — preserves stability for
                # independent steps.
                # We keep `ready` sorted by re-inserting in the right
                # spot (small N — pipeline has ~10 steps; bisect ok).
                insert_at = len(ready)
                target_order = declared_order[consumer_id]
                for i, existing in enumerate(ready):
                    if target_order < declared_order[existing]:
                        insert_at = i
                        break
                ready.insert(insert_at, consumer_id)
    if len(order) != len(declared_order):
        unresolved = set(declared_order) - set(order)
        raise ValueError(
            "pipeline contract has a cycle — these step_ids could not be "
            f"topologically ordered: {sorted(unresolved)}"
        )
    return order


def to_mermaid(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> str:
    """Render the contract as a Mermaid ``graph TD`` block.

    Nodes are step_ids labelled with the step_label. Edges are
    producer -> consumer labelled with the artefact name (truncated if
    long). A small legend follows the diagram.
    """
    steps = _materialize(contract)
    by_id = {s.step_id: s for s in steps}
    adjacency = build_dependency_graph(steps)

    lines: list[str] = ["graph TD"]
    # Nodes
    for s in steps:
        label = _mermaid_escape(s.step_label)
        lines.append(f"    {s.step_id}[\"{label}\"]")

    # Edges — for each producer/consumer pair, group all shared
    # artefacts onto a single labelled edge.
    for producer in steps:
        produced = {a.name: a for a in producer.outputs}
        for consumer_id in adjacency[producer.step_id]:
            consumer = by_id[consumer_id]
            consumed_names = {a.name for a in consumer.inputs}
            shared = sorted(set(produced) & consumed_names)
            if not shared:
                continue
            label = ", ".join(_truncate(name, 28) for name in shared)
            label = _mermaid_escape(label)
            lines.append(
                f"    {producer.step_id} -- \"{label}\" --> {consumer_id}"
            )

    # Legend
    lines.append("")
    lines.append("    %% Legend:")
    lines.append("    %%   nodes  = pipeline steps (step_id from PIPELINE_CONTRACT)")
    lines.append(
        "    %%   edges  = labelled with the artefact name(s) the consumer reads"
    )
    return "\n".join(lines)


def _truncate(name: str, limit: int) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _mermaid_escape(text: str) -> str:
    # Mermaid double-quoted labels handle most characters, but escape
    # internal double quotes defensively.
    return text.replace('"', '\\"')


def find_orphan_inputs(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> list[tuple[str, ArtefactRef]]:
    """Inputs (excluding ``kind == "input"``) that no step produces.

    Returns a list of ``(consumer_step_id, ArtefactRef)`` tuples in
    declaration order. Used by :func:`verify_contract`.
    """
    steps = _materialize(contract)
    produced: set[str] = set()
    for s in steps:
        for a in s.outputs:
            produced.add(a.name)

    orphans: list[tuple[str, ArtefactRef]] = []
    for consumer in steps:
        for art in consumer.inputs:
            if art.kind == "input":
                continue
            if art.name not in produced:
                orphans.append((consumer.step_id, art))
    return orphans


def find_unconsumed_outputs(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
    *,
    terminal_allowlist: Iterable[str] = _TERMINAL_OUTPUTS,
) -> list[tuple[str, ArtefactRef]]:
    """Outputs no step consumes, filtered against a terminal-output allowlist.

    Returns ``(producer_step_id, ArtefactRef)`` tuples in declaration
    order. Outputs whose name is in ``terminal_allowlist`` are treated
    as intentionally terminal (final reports, user-facing docs,
    rendered Python files) and are NOT returned.
    """
    steps = _materialize(contract)
    consumed: set[str] = set()
    for s in steps:
        for a in s.inputs:
            consumed.add(a.name)

    allow = frozenset(terminal_allowlist)
    unconsumed: list[tuple[str, ArtefactRef]] = []
    for producer in steps:
        for art in producer.outputs:
            if art.name in consumed:
                continue
            if art.name in allow:
                continue
            unconsumed.append((producer.step_id, art))
    return unconsumed


# ---------------------------------------------------------------------------
# Static checker
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Result of a contract verification pass."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _repo_root_from_module() -> Path:
    # This file lives at
    # ``src/mellea_skills_compiler/compile/pipeline_contract.py`` — three
    # parents up from the file gets us to the repo root.
    return Path(__file__).resolve().parents[3]


def verify_contract(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
    *,
    repo_root: Path | None = None,
) -> VerifyResult:
    """Run static checks against the contract.

    Checks (errors halt; warnings surface as findings):

    1.  No cycles (topological sort succeeds).
    2.  Every consumer's input (``kind != "input"``) has a producer in
        the contract.
    3.  Every output that is consumed by ANOTHER step has the SAME
        schema reference on both sides (no schema drift).
    4.  Every referenced ``schema_path`` file exists on disk under
        ``repo_root``.
    5.  No duplicate step_ids; no duplicate output artefact names
        within the same step.
    6.  The terminal-output allowlist covers every unconsumed output
        (warning if not — the developer probably wants to either
        document the artefact as terminal or wire it into a consumer).
    7.  For each step's slash-command markdown (if present at
        ``.claude/commands/<slash_command>``), confirm the declared
        outputs appear textually in the markdown — light textual
        cross-check.

    Returns a :class:`VerifyResult`. ``ok=False`` if any errors.
    """
    repo_root = repo_root or _repo_root_from_module()
    steps = _materialize(contract)
    errors: list[str] = []
    warnings: list[str] = []

    # Check 5a — no duplicate step_ids.
    seen_ids: set[str] = set()
    for s in steps:
        if s.step_id in seen_ids:
            errors.append(f"[duplicate-step-id] step_id={s.step_id!r} declared twice")
        seen_ids.add(s.step_id)

    # Check 5b — no duplicate output names within a step.
    for s in steps:
        seen_outputs: set[str] = set()
        for art in s.outputs:
            if art.name in seen_outputs:
                errors.append(
                    f"[duplicate-output] step={s.step_id!r} declares output "
                    f"{art.name!r} more than once"
                )
            seen_outputs.add(art.name)

    # Check 1 — no cycles.
    try:
        topological_order(steps)
    except ValueError as exc:
        errors.append(f"[cycle] {exc}")

    # Check 2 — orphan inputs.
    for step_id, art in find_orphan_inputs(steps):
        errors.append(
            f"[orphan-input] step={step_id!r} consumes {art.name!r} but no "
            "other step produces it (and its kind is not 'input')"
        )

    # Check 3 — schema drift between producer and consumer.
    # Build a map of artefact_name -> {(step_id, role, schema_path)}.
    schema_decls: dict[str, list[tuple[str, str, str | None]]] = {}
    for s in steps:
        for art in s.outputs:
            schema_decls.setdefault(art.name, []).append(
                (s.step_id, "output", art.schema_path)
            )
        for art in s.inputs:
            schema_decls.setdefault(art.name, []).append(
                (s.step_id, "input", art.schema_path)
            )

    for name, decls in schema_decls.items():
        schema_paths = {
            sp for _step, _role, sp in decls if sp is not None
        }
        if len(schema_paths) > 1:
            errors.append(
                f"[schema-drift] artefact {name!r} declared with multiple "
                f"schema_path values: {sorted(schema_paths)}; producer and "
                "all consumers must agree"
            )

    # Check 4 — schema_path files exist on disk.
    referenced_paths: set[str] = set()
    for s in steps:
        for art in (*s.inputs, *s.outputs):
            if art.schema_path:
                referenced_paths.add(art.schema_path)
    for rel in sorted(referenced_paths):
        target = repo_root / rel
        if not target.exists():
            errors.append(
                f"[missing-schema-file] schema_path={rel!r} (referenced in the "
                f"contract) does not exist under repo_root={repo_root}"
            )

    # Check 6 — terminal-output allowlist coverage.
    leftovers = find_unconsumed_outputs(steps)
    for step_id, art in leftovers:
        warnings.append(
            f"[unconsumed-output] step={step_id!r} produces {art.name!r} but "
            "no later step consumes it AND it is not in the "
            "_TERMINAL_OUTPUTS allowlist; consider either wiring it into "
            "a consumer or adding it to the allowlist"
        )

    # Check 7 — slash-command markdown cross-check.
    for s in steps:
        if not s.slash_command:
            continue
        md_path = repo_root / ".claude" / "commands" / s.slash_command
        if not md_path.exists():
            warnings.append(
                f"[missing-slash-command] step={s.step_id!r} references "
                f"{s.slash_command!r} but file is absent at {md_path}"
            )
            continue
        try:
            md_text = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(
                f"[slash-command-read-error] step={s.step_id!r} cannot read "
                f"{md_path}: {exc}"
            )
            continue
        for art in s.outputs:
            # We match the basename of the artefact in the markdown.
            # Slash-command docs typically refer to artefacts by
            # filename (e.g. ``classification.json``), not full
            # `intermediate/...` paths.
            needle = art.name.rsplit("/", 1)[-1]
            if needle not in md_text:
                warnings.append(
                    f"[slash-command-doc-drift] step={s.step_id!r} declares "
                    f"output {art.name!r} but the basename "
                    f"{needle!r} does not appear in "
                    f".claude/commands/{s.slash_command}"
                )

    return VerifyResult(ok=not errors, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Rendering helpers used by the CLI
# ---------------------------------------------------------------------------


def render_markdown_report(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> str:
    """Render a Markdown report: Mermaid diagram + per-step I/O tables.

    Useful for paste-into-design-doc workflows. The CLI's
    ``contract show`` subcommand prints this verbatim.
    """
    steps = _materialize(contract)
    out: list[str] = ["# Pipeline contract"]
    out.append("")
    out.append("```mermaid")
    out.append(to_mermaid(steps))
    out.append("```")
    out.append("")
    out.append("## Topological order")
    out.append("")
    for idx, sid in enumerate(topological_order(steps), 1):
        out.append(f"{idx}. `{sid}`")
    out.append("")
    out.append("## Per-step I/O")
    out.append("")
    for s in steps:
        out.append(f"### `{s.step_id}` — {s.step_label}")
        out.append("")
        if s.slash_command:
            out.append(f"Slash command: `.claude/commands/{s.slash_command}`")
            out.append("")
        if s.notes:
            out.append(s.notes)
            out.append("")
        out.append("| Direction | Name | Kind | Schema |")
        out.append("|---|---|---|---|")
        for art in s.inputs:
            out.append(
                f"| input | `{art.name}` | {art.kind} | "
                f"{('`' + art.schema_path + '`') if art.schema_path else '—'} |"
            )
        for art in s.outputs:
            out.append(
                f"| output | `{art.name}` | {art.kind} | "
                f"{('`' + art.schema_path + '`') if art.schema_path else '—'} |"
            )
        out.append("")
    return "\n".join(out)


def render_contract_json(
    contract: Iterable[StepContract] = PIPELINE_CONTRACT,
) -> str:
    """Machine-readable JSON dump of the contract + computed graph."""
    steps = _materialize(contract)
    payload = {
        "format_version": "1.0",
        "steps": [
            {
                "step_id": s.step_id,
                "step_label": s.step_label,
                "slash_command": s.slash_command,
                "notes": s.notes,
                "inputs": [
                    {
                        "name": a.name,
                        "format": a.format,
                        "schema_path": a.schema_path,
                        "kind": a.kind,
                    }
                    for a in s.inputs
                ],
                "outputs": [
                    {
                        "name": a.name,
                        "format": a.format,
                        "schema_path": a.schema_path,
                        "kind": a.kind,
                    }
                    for a in s.outputs
                ],
            }
            for s in steps
        ],
        "topological_order": topological_order(steps),
        "dependency_graph": build_dependency_graph(steps),
    }
    return json.dumps(payload, indent=2)


__all__ = [
    "ArtefactRef",
    "StepContract",
    "VerifyResult",
    "PIPELINE_CONTRACT",
    "build_dependency_graph",
    "topological_order",
    "to_mermaid",
    "find_orphan_inputs",
    "find_unconsumed_outputs",
    "verify_contract",
    "render_markdown_report",
    "render_contract_json",
]
