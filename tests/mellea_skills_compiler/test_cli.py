"""Tests for the top-level `mellea-skills` Typer CLI in
`mellea_skills_compiler.cli`.

Phase 3.5.A scope: the `--use-descriptor` flag on the `compile` command now
routes through the SAME `mellea_skills.compile` subprocess-spawn path the
legacy free-form Python emission flow uses — only the slash-command argv
gains a `--use-descriptor` suffix. The orchestrator at
`.claude/commands/mellea-fy.md` parses the flag out of `$ARGUMENTS` and
forwards it to `/mellea-fy-generate`, which switches Step 5 to descriptor IR
emission + render. The legacy free-form flow remains the default until the
Phase 5 flag flip.

These tests cover:
- the `--use-descriptor` flag is wired into the `compile` Typer command
- both routes invoke `mellea_skills.compile`; the descriptor route passes
  `use_descriptor=True`, the legacy route passes `use_descriptor=False`
- `--repair-mode` is forwarded through unchanged in both modes
- spec-path resolution accepts spec.md / SKILL.md / direct .md / fallback
  to first .md in directory; errors on directories with no .md files
- exit codes (0 on success; non-zero on failure / spec-resolution error)

The in-process `compile_via_descriptor` entrypoint is preserved for unit
tests and `scripts/batch_descriptor_test.py` / `scripts/corpus_compare.py`
one-shot mode, but is no longer reached from `mellea-skills compile`.

The `@pytest.mark.live` test at the bottom exercises the real pipeline
against `skills/sentry-find-bugs/`. It is skipped unless `--run-live` is
passed; see the module-level conftest hooks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from mellea_skills_compiler.cli import app


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- Test infrastructure ---------------------------------------------------


@pytest.fixture
def compile_mock(monkeypatch):
    """Stub the shared `mellea_skills.compile` subprocess-spawn path so tests
    don't actually attempt to spawn the `claude` CLI / install mellea / etc.

    Phase 3.5.A: this is now the single entrypoint both legacy and descriptor
    routes go through. The descriptor branch only differs by passing
    `use_descriptor=True` as a kwarg.
    """
    from mellea_skills_compiler.compile import mellea_skills as _ms

    mock = MagicMock(return_value=None)
    monkeypatch.setattr(_ms, "compile", mock)
    return mock


def _spec_text(filename: str) -> str:
    return f"---\nname: test-skill\n---\n# Test skill from {filename}\n"


@pytest.fixture
def skill_with_spec(tmp_path):
    """A minimal skill directory containing `spec.md`."""
    skill = tmp_path / "test-skill"
    skill.mkdir()
    (skill / "spec.md").write_text(_spec_text("spec.md"))
    return skill


@pytest.fixture
def skill_with_SKILL(tmp_path):
    """A minimal skill directory containing `SKILL.md` (preferred over spec.md)."""
    skill = tmp_path / "test-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(_spec_text("SKILL.md"))
    return skill


@pytest.fixture
def skill_with_arbitrary_md(tmp_path):
    """A workspace directory containing a non-canonical `.md` file."""
    skill = tmp_path / "test-skill"
    skill.mkdir()
    (skill / "agent.md").write_text(_spec_text("agent.md"))
    return skill


# ---- Flag wiring ----------------------------------------------------------


class TestUseDescriptorFlagWiring:
    """The `--use-descriptor` flag must appear on the `compile` command."""

    def test_help_lists_use_descriptor_flag(self):
        runner = CliRunner()
        result = runner.invoke(app, ["compile", "--help"])
        assert result.exit_code == 0
        assert "--use-descriptor" in result.stdout


# ---- Default routing (legacy free-form Python emission) -------------------


class TestDefaultRoutingIsLegacy:
    """Without `--use-descriptor`, the CLI must call the shared
    `mellea_skills.compile` WITHOUT the `use_descriptor` flag set. This is
    the contract that protects users until the Phase 5 flip.
    """

    def test_without_use_descriptor_calls_legacy_compile(
        self, skill_with_spec, compile_mock
    ):
        runner = CliRunner()
        result = runner.invoke(app, ["compile", str(skill_with_spec), "--no-run"])
        assert result.exit_code == 0, result.stdout
        compile_mock.assert_called_once()
        # Legacy route does not set use_descriptor; the kwarg may be absent
        # or False (both are legacy behaviour).
        assert compile_mock.call_args.kwargs.get("use_descriptor", False) is False


# ---- Descriptor routing ---------------------------------------------------


class TestDescriptorRouting:
    """With `--use-descriptor`, the CLI must call the shared spawn path with
    `use_descriptor=True`, so the slash-command orchestrator routes Step 5
    through descriptor IR emission + render.
    """

    def test_cli_with_use_descriptor_sets_flag(self, skill_with_spec, compile_mock):
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_spec), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 0, result.stdout
        compile_mock.assert_called_once()
        assert compile_mock.call_args.kwargs.get("use_descriptor") is True

    def test_cli_handles_skill_with_spec_md(self, skill_with_spec, compile_mock):
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_spec), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 0
        # The shared spawn path receives the original spec_path (a directory
        # or .md file); legacy and descriptor share the same plumbing.
        assert compile_mock.call_args.args[0] == skill_with_spec

    def test_cli_handles_skill_with_SKILL_md(self, skill_with_SKILL, compile_mock):
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_SKILL), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 0
        assert compile_mock.call_args.args[0] == skill_with_SKILL

    def test_cli_accepts_direct_spec_md_path(self, skill_with_spec, compile_mock):
        """Legacy resolver accepts a direct `.md` file. Descriptor mode
        must keep that contract.
        """
        spec_md = skill_with_spec / "spec.md"
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(spec_md), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 0
        assert compile_mock.call_args.args[0] == spec_md

    def test_cli_accepts_workspace_with_arbitrary_md(
        self, skill_with_arbitrary_md, compile_mock
    ):
        """Phase 3.5.A §2.1: a workspace dir with a non-canonical .md file
        (no spec.md / SKILL.md) must be accepted; the resolver falls back
        to the first .md file alphabetically.
        """
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compile", str(skill_with_arbitrary_md), "--use-descriptor", "--no-run"],
        )
        assert result.exit_code == 0, result.stdout
        assert compile_mock.call_args.args[0] == skill_with_arbitrary_md

    def test_cli_errors_when_skill_has_no_md_file(self, tmp_path, compile_mock):
        empty_skill = tmp_path / "empty-skill"
        empty_skill.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(empty_skill), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 1
        compile_mock.assert_not_called()


# ---- Repair mode ----------------------------------------------------------


class TestRepairModeRouting:
    """`--repair-mode` paired with `--use-descriptor` must forward
    `repair_mode=True` through to the shared spawn path; the orchestrator
    then invokes ``./mellea-fy-repair`` instead of ``./mellea-fy``.
    """

    def test_repair_mode_forwards_to_spawn(self, skill_with_spec, compile_mock):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(skill_with_spec),
                "--use-descriptor",
                "--repair-mode",
                "--no-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        compile_mock.assert_called_once()
        assert compile_mock.call_args.kwargs.get("use_descriptor") is True
        assert compile_mock.call_args.kwargs.get("repair_mode") is True


# ---- D1: repair-on-lint-failure default resolution ------------------------


class TestRepairOnLintFailureDefault:
    """D1 — when ``--use-descriptor`` is set, the wrapper defaults
    ``--repair-on-lint-failure`` to True (the descriptor IR contracts
    make repair prompts precise enough that the loop is productive).
    When ``--use-descriptor`` is NOT set, the flag retains its legacy
    False default. The user can override either default with the
    explicit ``--repair-on-lint-failure`` / ``--no-repair-on-lint-failure``
    toggle.
    """

    def test_use_descriptor_enables_repair_by_default(
        self, skill_with_spec, compile_mock
    ):
        """``--use-descriptor`` (no explicit flag) → repair_on_lint_failure=True."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compile", str(skill_with_spec), "--use-descriptor", "--no-run"],
        )
        assert result.exit_code == 0, result.stdout
        assert (
            compile_mock.call_args.kwargs.get("repair_on_lint_failure") is True
        )

    def test_use_descriptor_with_explicit_no_repair_opts_out(
        self, skill_with_spec, compile_mock
    ):
        """``--use-descriptor --no-repair-on-lint-failure`` → False (user opt-out).

        The default kicks in only when the flag is omitted; an explicit
        ``--no-repair-on-lint-failure`` must override the descriptor-mode
        default so users can restore legacy halting behaviour.
        """
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(skill_with_spec),
                "--use-descriptor",
                "--no-repair-on-lint-failure",
                "--no-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (
            compile_mock.call_args.kwargs.get("repair_on_lint_failure") is False
        )

    def test_use_descriptor_with_explicit_repair_stays_true(
        self, skill_with_spec, compile_mock
    ):
        """``--use-descriptor --repair-on-lint-failure`` → True (user-explicit)."""
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(skill_with_spec),
                "--use-descriptor",
                "--repair-on-lint-failure",
                "--no-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (
            compile_mock.call_args.kwargs.get("repair_on_lint_failure") is True
        )

    def test_legacy_mode_keeps_repair_off_by_default(
        self, skill_with_spec, compile_mock
    ):
        """Without ``--use-descriptor`` and no explicit flag, repair stays
        off — the legacy free-form emission flow doesn't benefit from
        auto-repair the way the descriptor IR does (weaker contracts =
        less precise repair prompts).
        """
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_spec), "--no-run"]
        )
        assert result.exit_code == 0, result.stdout
        assert (
            compile_mock.call_args.kwargs.get("repair_on_lint_failure")
            is False
        )

    def test_legacy_mode_with_explicit_repair_still_works(
        self, skill_with_spec, compile_mock
    ):
        """Without ``--use-descriptor`` but with explicit
        ``--repair-on-lint-failure``, the user opt-in is preserved (the
        D1 default change must not eat the explicit-True case).
        """
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(skill_with_spec),
                "--repair-on-lint-failure",
                "--no-run",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert (
            compile_mock.call_args.kwargs.get("repair_on_lint_failure") is True
        )


# ---- Exit codes -----------------------------------------------------------


class TestExitCodes:
    def test_cli_exit_code_zero_on_success(self, skill_with_spec, compile_mock):
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_spec), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 0

    def test_cli_exit_code_nonzero_when_compile_raises(
        self, skill_with_spec, compile_mock
    ):
        """When the shared spawn path raises (subprocess failure, lint
        failure, smoke failure), the CLI must propagate a non-zero exit.
        """
        compile_mock.side_effect = Exception("synthetic spawn failure")
        runner = CliRunner()
        result = runner.invoke(
            app, ["compile", str(skill_with_spec), "--use-descriptor", "--no-run"]
        )
        assert result.exit_code == 1


# ---- Path resolver ---------------------------------------------------------


class TestSpecPathResolver:
    """Phase 3.5.A §2.1: the path resolver must accept a `.md` file by path,
    a workspace directory containing `spec.md` / `SKILL.md`, and a workspace
    directory containing only non-canonical `.md` files. It must reject a
    directory with no `.md` files.
    """

    def test_resolver_accepts_md_file(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        spec = tmp_path / "agent.md"
        spec.write_text("# spec")
        result = resolve_spec_path(spec)
        assert result.spec_file == spec
        assert result.workspace_dir == tmp_path
        assert result.is_workspace is False

    def test_resolver_accepts_workspace_with_spec_md(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        workspace = tmp_path / "skill"
        workspace.mkdir()
        spec = workspace / "spec.md"
        spec.write_text("# spec")
        result = resolve_spec_path(workspace)
        assert result.spec_file == spec
        assert result.workspace_dir == workspace
        assert result.is_workspace is True

    def test_resolver_accepts_workspace_with_SKILL_md(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        workspace = tmp_path / "skill"
        workspace.mkdir()
        skill_md = workspace / "SKILL.md"
        skill_md.write_text("# spec")
        result = resolve_spec_path(workspace)
        assert result.spec_file == skill_md

    def test_resolver_prefers_SKILL_md_over_spec_md(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        workspace = tmp_path / "skill"
        workspace.mkdir()
        (workspace / "spec.md").write_text("# spec")
        (workspace / "SKILL.md").write_text("# skill")
        result = resolve_spec_path(workspace)
        assert result.spec_file.name == "SKILL.md"

    def test_resolver_falls_back_to_arbitrary_md(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        workspace = tmp_path / "skill"
        workspace.mkdir()
        (workspace / "agent.md").write_text("# agent")
        result = resolve_spec_path(workspace)
        assert result.spec_file.name == "agent.md"
        assert result.is_workspace is True

    def test_resolver_rejects_workspace_with_no_md(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import (
            SpecPathResolutionError,
            resolve_spec_path,
        )

        workspace = tmp_path / "skill"
        workspace.mkdir()
        (workspace / "README.txt").write_text("not markdown")
        with pytest.raises(SpecPathResolutionError):
            resolve_spec_path(workspace)

    def test_resolver_rejects_missing_path(self, tmp_path):
        from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

        with pytest.raises(FileNotFoundError):
            resolve_spec_path(tmp_path / "does-not-exist")


# ---- Live test ------------------------------------------------------------


@pytest.mark.live
def test_cli_live_against_sentry_find_bugs(tmp_path):
    """Live end-to-end run of `mellea-skills compile --use-descriptor` against
    the `skills/sentry-find-bugs/` skill. Requires:
        - `.venv-spike` Python with mellea + anthropic installed
        - LiteLLM credentials at `/tmp/anthropic-litellm.env`
        - the skill exists at `skills/sentry-find-bugs/`

    Skipped unless `--run-live` is passed (see top-level conftest).
    """
    skill = REPO_ROOT / "skills" / "sentry-find-bugs"
    if not skill.exists():
        pytest.skip(f"skill not present at {skill}")
    if not (skill / "spec.md").exists():
        pytest.skip(f"spec.md missing in {skill}")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compile",
            str(skill),
            "--use-descriptor",
            "--no-run",
        ],
    )
    assert result.exit_code == 0, result.output
