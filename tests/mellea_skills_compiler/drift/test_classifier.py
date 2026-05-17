"""Unit tests for the three-class drift classifier (Phase 3.5 §14.3).

Independent of corpus_compare.py — exercises classifier.py + report.py
directly via in-memory fixture trees in ``tmp_path``.

Coverage targets per Phase 3.5 plan §14.3:

* ``DriftClass.NONE``       — completely unchanged.
* ``DriftClass.COSMETIC``   — whitespace / import order / transient renames.
* ``DriftClass.STRUCTURAL`` — operator picks, schema arity, control-flow.
* ``DriftClass.SEMANTIC``   — prompt-template text, symbol swaps, schema
                              field renames.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from mellea_skills_compiler.drift import (
    DriftClass,
    SkillDriftVerdict,
    classify_skill_drift,
    render_drift_report,
)


# --- Fixture helpers ------------------------------------------------------


def _make_pkg(
    root: Path,
    *,
    pipeline_py: str | None = None,
    schemas_py: str | None = None,
    melleafy_json: dict | None = None,
) -> Path:
    """Build a fake ``<skill>_mellea/`` directory under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    if pipeline_py is not None:
        (root / "pipeline.py").write_text(pipeline_py, encoding="utf-8")
    if schemas_py is not None:
        (root / "schemas.py").write_text(schemas_py, encoding="utf-8")
    if melleafy_json is not None:
        (root / "melleafy.json").write_text(
            json.dumps(melleafy_json, indent=2), encoding="utf-8"
        )
    return root


PIPELINE_BASE = textwrap.dedent(
    '''
    """Module docstring."""

    from mellea import start_session
    from .schemas import Finding


    def run_pipeline(diff: str) -> Finding:
        """Run the pipeline."""
        with start_session() as m:
            step_1 = m.instruct("Find security issues in the supplied diff and report each.")
            step_2 = m.instruct("Validate that each finding has a CVE reference or a CWE id.")
        return step_1
    '''
).strip() + "\n"


SCHEMAS_BASE = textwrap.dedent(
    """
    from pydantic import BaseModel


    class Finding(BaseModel):
        title: str
        severity: str
        cwe: str | None
    """
).strip() + "\n"


DESCRIPTOR_BASE = {
    "schema_version": "0.2.0",
    "name": "demo",
    "inputs": [{"name": "diff", "schema": {"type": "string"}}],
    "outputs": [{"name": "finding", "schema": {"$ref": "#/schemas/Finding"}}],
    "pipeline": {
        "operator": "sequential",
        "body": [
            {"kind": "call", "symbol": "instruct"},
            {"kind": "call", "symbol": "instruct"},
        ],
    },
}


# --- NONE-class tests -----------------------------------------------------


def test_no_drift_when_byte_identical(tmp_path: Path) -> None:
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=DESCRIPTOR_BASE,
    )
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=DESCRIPTOR_BASE,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.NONE
    # NONE-class still records the "AST identical" observation.
    assert any("AST identical" in obs for obs in verdict.pipeline_observations)


def test_no_drift_handles_missing_descriptors_consistently(tmp_path: Path) -> None:
    """If neither side has melleafy.json the classifier shouldn't crash —
    the descriptor diff is treated as 'both empty == hash match'."""
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    curr = _make_pkg(tmp_path / "curr", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.NONE


# --- COSMETIC tests -------------------------------------------------------


def test_cosmetic_whitespace_only_change_with_descriptor_hash_delta(
    tmp_path: Path,
) -> None:
    """Whitespace changes alone normalise to identical ASTs.  To trigger
    *cosmetic* (rather than NONE) at least one artefact must show a non-
    normalised diff — here, the descriptor canonical hash changes via a
    benign metadata field.  This mirrors the real-world case: the renderer
    re-emits whitespace + a fresh ``generated_at`` stamp on every compile,
    even when the skill's meaning didn't change."""
    prev_desc = dict(DESCRIPTOR_BASE)
    curr_desc = {**DESCRIPTOR_BASE, "generated_at": "2026-05-16T12:00:00Z"}
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=prev_desc,
    )
    bumpy = (
        PIPELINE_BASE.replace("def run_pipeline", "\n\ndef run_pipeline")
        + "\n# trailing\n"
    )
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=bumpy,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=curr_desc,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.COSMETIC


def test_cosmetic_transient_id_rename_with_descriptor_hash_delta(
    tmp_path: Path,
) -> None:
    """Renaming ``step_1`` → ``step_99`` is a pure transient-name rewrite
    the normaliser collapses to the sentinel.  Combined with a benign
    descriptor metadata bump, this exercises the cosmetic class — same
    semantic shape, different bytes on disk."""
    prev_desc = dict(DESCRIPTOR_BASE)
    curr_desc = {**DESCRIPTOR_BASE, "generated_at": "2026-05-16T12:00:00Z"}
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=prev_desc,
    )
    renamed = PIPELINE_BASE.replace("step_1", "step_99").replace("step_2", "step_100")
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=renamed,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=curr_desc,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.COSMETIC


def test_cosmetic_import_reorder_and_docstring_delta(tmp_path: Path) -> None:
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    # Swap import order, change the module docstring text, change the
    # function docstring — all stripped by the normaliser.
    cosmetic = (
        textwrap.dedent(
            '''
            """An updated module docstring."""

            from .schemas import Finding
            from mellea import start_session


            def run_pipeline(diff: str) -> Finding:
                """A reworded function docstring."""
                with start_session() as m:
                    step_1 = m.instruct("Find security issues in the supplied diff and report each.")
                    step_2 = m.instruct("Validate that each finding has a CVE reference or a CWE id.")
                return step_1
            '''
        ).strip()
        + "\n"
    )
    curr = _make_pkg(tmp_path / "curr", pipeline_py=cosmetic, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.COSMETIC


def test_cosmetic_when_descriptor_hash_differs_but_pipeline_unchanged(
    tmp_path: Path,
) -> None:
    """A whitespace/key-order change in melleafy.json that produces a different
    canonical hash but no operator/symbol/arity change should still classify
    as cosmetic — the descriptor changed shape, the *meaning* didn't."""
    prev_desc = dict(DESCRIPTOR_BASE)
    curr_desc = dict(DESCRIPTOR_BASE)
    # Add a metadata key the diff treats as non-load-bearing for drift.
    curr_desc = {**curr_desc, "generated_at": "2026-05-16T12:00:00Z"}
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=prev_desc,
    )
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=curr_desc,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.COSMETIC


# --- STRUCTURAL tests -----------------------------------------------------


def test_structural_schema_field_count_change(tmp_path: Path) -> None:
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    extra_field = SCHEMAS_BASE.replace(
        "cwe: str | None\n", "cwe: str | None\n    impact: str\n"
    )
    curr = _make_pkg(tmp_path / "curr", pipeline_py=PIPELINE_BASE, schemas_py=extra_field)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("Finding" in obs for obs in verdict.schemas_observations)


def test_structural_different_operator_pick_in_descriptor(tmp_path: Path) -> None:
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=DESCRIPTOR_BASE,
    )
    flipped = dict(DESCRIPTOR_BASE)
    flipped["pipeline"] = {
        "operator": "branch",  # ← was sequential
        "body": [
            {"kind": "call", "symbol": "instruct"},
            {"kind": "call", "symbol": "instruct"},
        ],
    }
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=flipped,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("operator bag" in obs for obs in verdict.descriptor_observations)


def test_structural_control_flow_shape_change(tmp_path: Path) -> None:
    """Adding a for-loop in pipeline.py without changing verbs or templates
    is structural — the control-flow shape diverged."""
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    with_loop = textwrap.dedent(
        '''
        from mellea import start_session
        from .schemas import Finding


        def run_pipeline(diff: str) -> Finding:
            with start_session() as m:
                step_1 = m.instruct("Find security issues in the supplied diff and report each.")
                for piece in diff.split():
                    step_2 = m.instruct("Validate that each finding has a CVE reference or a CWE id.")
            return step_1
        '''
    ).strip() + "\n"
    curr = _make_pkg(tmp_path / "curr", pipeline_py=with_loop, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("control-flow" in obs for obs in verdict.pipeline_observations)


def test_structural_inputs_arity_change(tmp_path: Path) -> None:
    prev = _make_pkg(
        tmp_path / "prev",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=DESCRIPTOR_BASE,
    )
    extra_input = dict(DESCRIPTOR_BASE)
    extra_input["inputs"] = [
        {"name": "diff", "schema": {"type": "string"}},
        {"name": "context", "schema": {"type": "string"}},
    ]
    curr = _make_pkg(
        tmp_path / "curr",
        pipeline_py=PIPELINE_BASE,
        schemas_py=SCHEMAS_BASE,
        melleafy_json=extra_input,
    )
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("inputs arity" in obs for obs in verdict.descriptor_observations)


# --- SEMANTIC tests -------------------------------------------------------


def test_semantic_prompt_template_text_rewrite(tmp_path: Path) -> None:
    """Materially changing a prompt template text fires semantic drift."""
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    reworded = PIPELINE_BASE.replace(
        "Find security issues in the supplied diff and report each.",
        "Hunt for memory-safety bugs across the entire codebase and grade impact.",
    ).replace(
        "Validate that each finding has a CVE reference or a CWE id.",
        "Reformat the bug list as a json document containing severity rankings.",
    )
    curr = _make_pkg(tmp_path / "curr", pipeline_py=reworded, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.SEMANTIC
    assert any(
        "[semantic]" in obs and "fuzzy similarity" in obs
        for obs in verdict.pipeline_observations
    )


def test_semantic_symbol_swap_instruct_to_aact(tmp_path: Path) -> None:
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    swapped = PIPELINE_BASE.replace("m.instruct", "m.aact")
    curr = _make_pkg(tmp_path / "curr", pipeline_py=swapped, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.SEMANTIC
    assert any(
        "[semantic]" in obs and "semantic-symbol" in obs
        for obs in verdict.pipeline_observations
    )


def test_semantic_schema_field_rename_same_arity(tmp_path: Path) -> None:
    """Same field count, different names → semantic drift (meaning changed)."""
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    renamed = SCHEMAS_BASE.replace("title: str", "name: str").replace(
        "severity: str", "criticality: str"
    )
    curr = _make_pkg(tmp_path / "curr", pipeline_py=PIPELINE_BASE, schemas_py=renamed)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.SEMANTIC
    assert any(
        "[semantic]" in obs and "renamed" in obs
        for obs in verdict.schemas_observations
    )


def test_semantic_prompt_template_introduced_from_empty(tmp_path: Path) -> None:
    """Going from no templates to having templates is treated as semantic
    drift (one side has none, the other does)."""
    no_templates = textwrap.dedent(
        '''
        def run_pipeline(diff):
            return diff
        '''
    ).strip() + "\n"
    prev = _make_pkg(tmp_path / "prev", pipeline_py=no_templates, schemas_py=SCHEMAS_BASE)
    curr = _make_pkg(tmp_path / "curr", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    assert verdict.drift_class == DriftClass.SEMANTIC


# --- Edge / robustness tests ---------------------------------------------


def test_missing_previous_directory_is_flagged_structural(tmp_path: Path) -> None:
    curr = _make_pkg(tmp_path / "curr", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", tmp_path / "missing-prev", curr)
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("previous package missing" in n for n in verdict.notes)


def test_missing_current_directory_is_flagged_structural(tmp_path: Path) -> None:
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, tmp_path / "missing-curr")
    assert verdict.drift_class == DriftClass.STRUCTURAL
    assert any("current package missing" in n for n in verdict.notes)


def test_neither_side_present_returns_none_with_note(tmp_path: Path) -> None:
    verdict = classify_skill_drift(
        "demo", tmp_path / "a", tmp_path / "b"
    )
    assert verdict.drift_class == DriftClass.NONE
    assert verdict.notes  # non-empty: explains why


def test_malformed_pipeline_doesnt_crash(tmp_path: Path) -> None:
    prev = _make_pkg(tmp_path / "prev", pipeline_py=PIPELINE_BASE, schemas_py=SCHEMAS_BASE)
    bad = "def broken(:\n    pass\n"  # SyntaxError
    curr = _make_pkg(tmp_path / "curr", pipeline_py=bad, schemas_py=SCHEMAS_BASE)
    verdict = classify_skill_drift("demo", prev, curr)
    # Just assert it didn't crash and produced *some* non-NONE verdict
    # (the syntax-error hash diverges from a clean parse).
    assert verdict.drift_class in (
        DriftClass.COSMETIC,
        DriftClass.STRUCTURAL,
        DriftClass.SEMANTIC,
    )


# --- Report rendering ----------------------------------------------------


def test_render_drift_report_contains_summary_and_table(tmp_path: Path) -> None:
    verdicts = [
        SkillDriftVerdict(skill="alpha", drift_class=DriftClass.NONE),
        SkillDriftVerdict(
            skill="beta",
            drift_class=DriftClass.COSMETIC,
            pipeline_observations=["[whitespace only]"],
        ),
        SkillDriftVerdict(
            skill="gamma",
            drift_class=DriftClass.STRUCTURAL,
            schemas_observations=["[structural] field count delta"],
        ),
        SkillDriftVerdict(
            skill="delta",
            drift_class=DriftClass.SEMANTIC,
            pipeline_observations=["[semantic] prompt rewrite"],
        ),
    ]
    md = render_drift_report(
        verdicts,
        mellea_version="0.6.0",
        recompile_trigger="Mellea 0.5 → 0.6 bump",
        previous_ref="git rev abc123",
    )
    # Top-level metadata
    assert "Mellea version" in md and "0.6.0" in md
    assert "abc123" in md
    # Class-summary table headers
    assert "Drift class summary" in md
    # Per-skill table rows
    assert "| alpha | none |" in md
    assert "| beta | cosmetic |" in md
    assert "| gamma | structural |" in md
    assert "| delta | semantic |" in md
    # Per-skill detail section appears for non-cosmetic where notes exist
    assert "gamma" in md and "delta" in md
    # Decision authority section
    assert "Decision authority" in md
    assert "structural-drift.md" in md


def test_render_drift_report_hard_stop_when_semantic_exceeds_5pct(
    tmp_path: Path,
) -> None:
    """§14.4: >5% semantic drift triggers the hard-stop callout."""
    verdicts = [
        SkillDriftVerdict(skill=f"s{i}", drift_class=DriftClass.NONE) for i in range(9)
    ] + [SkillDriftVerdict(skill="bad", drift_class=DriftClass.SEMANTIC)]
    md = render_drift_report(verdicts)
    # 1/10 = 10% > 5%
    assert "HARD STOP" in md
