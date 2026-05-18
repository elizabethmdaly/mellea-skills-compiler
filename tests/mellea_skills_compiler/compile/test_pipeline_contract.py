"""Tests for the pipeline contract registry + static checker.

Covers:

* Phase 1 (graph): topological order, adjacency, Mermaid rendering, orphan
  detection on the *real* contract.
* Phase 2 (verifier): cycle detection, schema-drift detection, missing-file
  detection, duplicate detection, slash-command cross-check, CLI exit code.

The synthetic-failure tests build a small "bad" contract instance and run
the helpers directly — they never mutate the real
:data:`PIPELINE_CONTRACT`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.compile.pipeline_contract import (
    PIPELINE_CONTRACT,
    ArtefactRef,
    StepContract,
    VerifyResult,
    build_dependency_graph,
    find_orphan_inputs,
    find_unconsumed_outputs,
    render_contract_json,
    render_markdown_report,
    to_mermaid,
    topological_order,
    verify_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Phase 1 — graph helpers against the REAL contract
# ---------------------------------------------------------------------------


class TestRealContractGraph:
    """Phase 1 — the graph helpers behave sanely against PIPELINE_CONTRACT."""

    def test_contract_loads_with_expected_step_count(self):
        # Pipeline has at minimum: wrapper-prep + 8 documented slash-command
        # steps (0, 1, 2, 2.5, 3, 4, 5, 6, 7) + 2 synthesizer steps = 11.
        assert len(PIPELINE_CONTRACT) >= 10
        # No empty step ids.
        for step in PIPELINE_CONTRACT:
            assert step.step_id, f"empty step_id at index {step}"
            assert step.step_label, f"empty step_label for {step.step_id}"

    def test_topological_order_is_well_defined(self):
        order = topological_order(PIPELINE_CONTRACT)
        # Every declared step appears exactly once.
        assert len(order) == len(PIPELINE_CONTRACT)
        assert len(set(order)) == len(order)

    def test_topological_order_includes_all_step_ids(self):
        order = set(topological_order(PIPELINE_CONTRACT))
        declared = {s.step_id for s in PIPELINE_CONTRACT}
        assert order == declared

    def test_topological_order_respects_declared_dependencies(self):
        """A consumer must appear after every step that produces one of
        its inputs."""
        order = topological_order(PIPELINE_CONTRACT)
        pos = {sid: idx for idx, sid in enumerate(order)}
        graph = build_dependency_graph(PIPELINE_CONTRACT)
        for producer, consumers in graph.items():
            for consumer in consumers:
                assert pos[producer] < pos[consumer], (
                    f"{producer} must come before {consumer} but order is "
                    f"{order}"
                )

    def test_topological_order_raises_on_synthetic_cycle(self):
        a = StepContract(
            step_id="a",
            step_label="A",
            inputs=(ArtefactRef(name="from_b", format="json"),),
            outputs=(ArtefactRef(name="from_a", format="json"),),
        )
        b = StepContract(
            step_id="b",
            step_label="B",
            inputs=(ArtefactRef(name="from_a", format="json"),),
            outputs=(ArtefactRef(name="from_b", format="json"),),
        )
        with pytest.raises(ValueError, match="cycle"):
            topological_order((a, b))

    def test_build_dependency_graph_links_classify_to_inventory(self):
        graph = build_dependency_graph(PIPELINE_CONTRACT)
        assert "step_1_inventory" in graph["step_0_classify"], (
            "classification.json must flow from Step 0 to Step 1"
        )

    def test_build_dependency_graph_links_element_mapping_to_descriptor_emission(self):
        graph = build_dependency_graph(PIPELINE_CONTRACT)
        assert "step_5_generate" in graph["step_2_map"], (
            "element_mapping.json (Step 2) must flow into Step 5 descriptor "
            "emission"
        )

    def test_to_mermaid_includes_all_step_ids(self):
        diagram = to_mermaid(PIPELINE_CONTRACT)
        for step in PIPELINE_CONTRACT:
            assert step.step_id in diagram, (
                f"{step.step_id} missing from Mermaid output"
            )

    def test_to_mermaid_includes_edge_labels(self):
        diagram = to_mermaid(PIPELINE_CONTRACT)
        # At least one labelled edge.
        assert " -- \"" in diagram
        assert "classification" in diagram  # classification.json appears

    def test_to_mermaid_is_valid_mermaid_syntax(self):
        """Heuristic: starts with ``graph``, every non-empty / non-comment
        line is either a node-def or an edge-def."""
        diagram = to_mermaid(PIPELINE_CONTRACT)
        lines = diagram.splitlines()
        assert lines, "empty mermaid output"
        assert lines[0].strip().startswith("graph TD"), (
            f"Mermaid must declare a `graph TD` block, got {lines[0]!r}"
        )
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%%"):
                # comment / legend
                continue
            # Node defs match `<id>[...]`; edge defs contain ` --> `.
            assert "[" in stripped or "-->" in stripped, (
                f"unexpected Mermaid line: {stripped!r}"
            )

    def test_find_orphan_inputs_on_real_contract(self):
        """The real contract MUST have zero orphan inputs (excluding
        kind=='input'). If this fails, the wrapper or a slash command is
        consuming something nothing else produces."""
        orphans = find_orphan_inputs(PIPELINE_CONTRACT)
        assert not orphans, (
            f"unexpected orphan inputs in real contract: "
            f"{[(sid, art.name) for sid, art in orphans]}"
        )


# ---------------------------------------------------------------------------
# Phase 2 — verifier against the REAL contract
# ---------------------------------------------------------------------------


class TestVerifyRealContract:

    def test_verify_passes_on_real_contract(self):
        result = verify_contract(PIPELINE_CONTRACT, repo_root=REPO_ROOT)
        assert isinstance(result, VerifyResult)
        if result.errors:
            pytest.fail(
                "real PIPELINE_CONTRACT failed verification:\n"
                + "\n".join(result.errors)
            )
        assert result.ok is True

    def test_verify_slash_command_doc_cross_check_passes_for_real_docs(self):
        """Each real slash command should textually mention at least one of
        its declared outputs. We tolerate the existing two-output drift
        warnings on `step_1b_trace.json` / `__main__.py` (documented in
        the analysis doc) — those are explicitly the kind of finding we
        WANT the verifier to surface.

        This test guards against *regression* in slash-command doc
        coverage: if a slash-command's PRIMARY output stops appearing in
        its doc, that's a real problem."""
        result = verify_contract(PIPELINE_CONTRACT, repo_root=REPO_ROOT)
        drift_msgs = [
            w for w in result.warnings if "slash-command-doc-drift" in w
        ]
        # Each step's primary output (the one schema-governed in
        # `.claude/schemas/`) must be referenced in its slash command.
        # Build the set of primary outputs that ARE missing.
        for step in PIPELINE_CONTRACT:
            if not step.slash_command:
                continue
            md_path = REPO_ROOT / ".claude" / "commands" / step.slash_command
            md_text = md_path.read_text(encoding="utf-8")
            primary_outputs = [
                a for a in step.outputs
                if a.schema_path and a.schema_path.startswith(".claude/schemas/")
            ]
            for art in primary_outputs:
                basename = art.name.rsplit("/", 1)[-1]
                assert basename in md_text, (
                    f"primary output {basename!r} of {step.step_id!r} not "
                    f"mentioned in its slash command {step.slash_command}"
                )
        # Sanity: drift_msgs are still a list (the existing known drift may
        # or may not be present, but the structure should be a list).
        assert isinstance(drift_msgs, list)


# ---------------------------------------------------------------------------
# Phase 2 — synthetic-failure tests
# ---------------------------------------------------------------------------


def _trivial_step(
    step_id: str,
    *,
    inputs: tuple[ArtefactRef, ...] = (),
    outputs: tuple[ArtefactRef, ...] = (),
    slash_command: str | None = None,
) -> StepContract:
    return StepContract(
        step_id=step_id,
        step_label=step_id,
        inputs=inputs,
        outputs=outputs,
        slash_command=slash_command,
    )


class TestVerifySyntheticFailures:

    def test_verify_detects_synthetic_cycle(self, tmp_path):
        a = _trivial_step(
            "a",
            inputs=(ArtefactRef(name="from_b", format="json"),),
            outputs=(ArtefactRef(name="from_a", format="json"),),
        )
        b = _trivial_step(
            "b",
            inputs=(ArtefactRef(name="from_a", format="json"),),
            outputs=(ArtefactRef(name="from_b", format="json"),),
        )
        result = verify_contract((a, b), repo_root=tmp_path)
        assert not result.ok
        assert any("[cycle]" in e for e in result.errors)

    def test_verify_detects_missing_schema_file(self, tmp_path):
        step = _trivial_step(
            "alpha",
            outputs=(
                ArtefactRef(
                    name="alpha_out.json",
                    format="json",
                    schema_path="nonexistent/dir/missing.schema.json",
                ),
            ),
        )
        result = verify_contract((step,), repo_root=tmp_path)
        assert not result.ok
        assert any("[missing-schema-file]" in e for e in result.errors)

    def test_verify_detects_schema_drift_between_producer_and_consumer(
        self, tmp_path
    ):
        # Drop two schema files on disk to keep check 4 happy.
        sa = tmp_path / "a.schema.json"
        sa.write_text("{}", encoding="utf-8")
        sb = tmp_path / "b.schema.json"
        sb.write_text("{}", encoding="utf-8")

        producer = _trivial_step(
            "producer",
            outputs=(
                ArtefactRef(
                    name="drifty.json",
                    format="json",
                    schema_path="a.schema.json",
                ),
            ),
        )
        consumer = _trivial_step(
            "consumer",
            inputs=(
                ArtefactRef(
                    name="drifty.json",
                    format="json",
                    schema_path="b.schema.json",
                ),
            ),
            outputs=(ArtefactRef(name="end_out.json", format="json"),),
        )
        result = verify_contract((producer, consumer), repo_root=tmp_path)
        assert not result.ok
        assert any("[schema-drift]" in e for e in result.errors)

    def test_verify_detects_orphan_input(self, tmp_path):
        consumer = _trivial_step(
            "lonely",
            inputs=(ArtefactRef(name="never_produced.json", format="json"),),
            outputs=(ArtefactRef(name="lonely_out.json", format="json"),),
        )
        result = verify_contract((consumer,), repo_root=tmp_path)
        assert not result.ok
        assert any("[orphan-input]" in e for e in result.errors)

    def test_verify_does_not_complain_about_kind_input(self, tmp_path):
        consumer = _trivial_step(
            "head",
            inputs=(ArtefactRef(name="spec_md", format="markdown", kind="input"),),
            outputs=(ArtefactRef(name="head_out.json", format="json"),),
        )
        result = verify_contract((consumer,), repo_root=tmp_path)
        # No orphan error — `spec_md` is kind=input.
        assert not any("[orphan-input]" in e for e in result.errors)

    def test_verify_detects_duplicate_step_id(self, tmp_path):
        a1 = _trivial_step("dup")
        a2 = _trivial_step("dup")
        result = verify_contract((a1, a2), repo_root=tmp_path)
        assert not result.ok
        assert any("[duplicate-step-id]" in e for e in result.errors)

    def test_verify_detects_duplicate_output_name_within_step(self, tmp_path):
        step = _trivial_step(
            "twice",
            outputs=(
                ArtefactRef(name="dupe.json", format="json"),
                ArtefactRef(name="dupe.json", format="json"),
            ),
        )
        result = verify_contract((step,), repo_root=tmp_path)
        assert not result.ok
        assert any("[duplicate-output]" in e for e in result.errors)

    def test_verify_terminal_output_allowlist_suppresses_warning(self, tmp_path):
        # Build a contract whose final output is `mapping_report.md` (in
        # the allowlist) — verifier should emit no unconsumed-output
        # warning for it.
        step = _trivial_step(
            "producer",
            outputs=(
                ArtefactRef(
                    name="mapping_report.md", format="markdown", kind="rendered"
                ),
            ),
        )
        result = verify_contract((step,), repo_root=tmp_path)
        assert not any(
            "unconsumed-output" in w and "mapping_report.md" in w
            for w in result.warnings
        )

    def test_verify_terminal_output_allowlist_warns_on_unknown_terminal(
        self, tmp_path
    ):
        # Same scenario but with an artefact NOT in the allowlist.
        step = _trivial_step(
            "producer",
            outputs=(
                ArtefactRef(
                    name="some_orphan_terminal.json",
                    format="json",
                    kind="intermediate",
                ),
            ),
        )
        result = verify_contract((step,), repo_root=tmp_path)
        assert any(
            "[unconsumed-output]" in w
            and "some_orphan_terminal.json" in w
            for w in result.warnings
        )

    def test_verify_slash_command_cross_check_warns_on_missing_mention(
        self, tmp_path
    ):
        # Lay down a slash-command markdown that does NOT mention the
        # declared output basename.
        commands_dir = tmp_path / ".claude" / "commands"
        commands_dir.mkdir(parents=True)
        (commands_dir / "synthetic.md").write_text(
            "Lorem ipsum — references nothing relevant.", encoding="utf-8"
        )
        step = _trivial_step(
            "syn",
            outputs=(ArtefactRef(name="secret_payload.json", format="json"),),
            slash_command="synthetic.md",
        )
        result = verify_contract((step,), repo_root=tmp_path)
        assert any(
            "[slash-command-doc-drift]" in w
            and "secret_payload.json" in w
            for w in result.warnings
        )


# ---------------------------------------------------------------------------
# Render helpers + CLI exit-code smoke test
# ---------------------------------------------------------------------------


class TestRendering:

    def test_render_markdown_report_includes_mermaid_block(self):
        out = render_markdown_report(PIPELINE_CONTRACT)
        assert "```mermaid" in out
        assert "graph TD" in out
        assert "## Topological order" in out
        assert "## Per-step I/O" in out

    def test_render_contract_json_is_well_formed(self):
        import json

        payload = json.loads(render_contract_json(PIPELINE_CONTRACT))
        assert payload["format_version"] == "1.0"
        assert "steps" in payload
        assert "topological_order" in payload
        assert "dependency_graph" in payload
        assert len(payload["steps"]) == len(PIPELINE_CONTRACT)

    def test_find_unconsumed_outputs_on_real_contract_returns_list(self):
        # We don't assert exact content — the real contract may evolve.
        # The shape must be a list and every entry must be a (str,
        # ArtefactRef) tuple.
        leftovers = find_unconsumed_outputs(PIPELINE_CONTRACT)
        for entry in leftovers:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            step_id, art = entry
            assert isinstance(step_id, str)
            assert isinstance(art, ArtefactRef)


class TestCLIExitCode:

    def test_verify_exit_code_via_cli_subcommand_passes_on_real_contract(
        self, monkeypatch
    ):
        """Smoke-test the CLI sub: invoke `contract verify` through the
        typer test runner and confirm exit code 0 on the real contract.
        """
        from typer.testing import CliRunner

        from mellea_skills_compiler.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["contract", "verify"])
        assert result.exit_code == 0, (
            f"contract verify failed against real contract:\n{result.output}"
        )

    def test_contract_show_order_lists_all_step_ids(self):
        from typer.testing import CliRunner

        from mellea_skills_compiler.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["contract", "show", "--order"])
        assert result.exit_code == 0
        listed = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        declared = {s.step_id for s in PIPELINE_CONTRACT}
        assert listed == declared

    def test_contract_show_json_is_machine_readable(self):
        import json as _json
        from typer.testing import CliRunner

        from mellea_skills_compiler.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["contract", "show", "--json"])
        assert result.exit_code == 0
        payload = _json.loads(result.stdout)
        assert payload["format_version"] == "1.0"
        assert len(payload["steps"]) == len(PIPELINE_CONTRACT)
