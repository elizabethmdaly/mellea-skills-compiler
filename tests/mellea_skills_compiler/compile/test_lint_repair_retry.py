"""Unit tests for Fix A — post-session lint-failure retry.

Covers:

* :func:`_bake_repair_prescriptions` — reads ``step_7_report.json``,
  builds F1 prescriptions via :mod:`compile.repair_prompt`, persists
  to ``intermediate/repair_prescriptions.md``.
* :func:`_build_lint_repair_system_prompt` — augments the base system
  prompt for a repair-mode re-spawn (names failing lints, points to
  prescription file).
* :func:`_collect_error_failures` — severity filter helper.
* :func:`_spawn_claude_with_lint_repair` — the wrapper-side retry loop
  composed around :func:`_spawn_claude`.
* :func:`compile` — verifies the opt-in flag dispatches to the new
  helper; default behaviour stays single-shot.
* Repair-mode argv shape — uses ``./mellea-fy-repair`` not
  ``./mellea-fy``; propagates ``--use-descriptor``.

Tests must NEVER spawn a real ``claude`` subprocess. The spawn callable
is mocked in every test that touches it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mellea_skills_compiler.compile import mellea_skills
from mellea_skills_compiler.compile.mellea_skills import (
    _bake_repair_prescriptions,
    _build_claude_argv,
    _build_lint_repair_system_prompt,
    _collect_error_failures,
    _lint_repair_retry_loop,
    _spawn_claude_with_lint_repair,
)


# ─── Helpers ────────────────────────────────────────────────────────────


def _write_step_7_report(
    package_dir: Path,
    *,
    overall_verdict: str = "fail",
    lints: list[dict] | None = None,
) -> Path:
    """Write a synthetic ``step_7_report.json`` and return its path."""
    intermediate = package_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": "1.1",
        "overall_verdict": overall_verdict,
        "lints": lints or [],
    }
    out = intermediate / "step_7_report.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    return out


def _stdlib_arity_failure_lint() -> dict:
    """Synthesize a lint entry matching the real nda-review failure."""
    return {
        "lint_id": "stdlib-arity",
        "verdict": "fail",
        "severity": "error",
        "effective_severity": "error",
        "failures": [
            {
                "file": "requirements.py",
                "line": 30,
                "column": 32,
                "message": (
                    "`check(...)` called with 1 positional argument(s); "
                    "signature requires between 2 and 2 positional arg(s). "
                    "Consult intermediate/mellea_api_ref.json:.modules."
                ),
                "rule_ref": "Rule 4-4 (stdlib-arity)",
            }
        ],
    }


# ─── _bake_repair_prescriptions ─────────────────────────────────────────


class TestBakeRepairPrescriptions:
    def test_writes_file_when_templates_match(self, tmp_path):
        package = tmp_path / "weather_mellea"
        report_path = _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        out = _bake_repair_prescriptions(package, lint_report_path=report_path)

        assert out is not None
        assert out == package / "intermediate" / "repair_prescriptions.md"
        assert out.exists()
        contents = out.read_text(encoding="utf-8")
        # Section header is always present.
        assert "Per-lint fix prescriptions" in contents
        # Template-specific guidance for stdlib-arity references the
        # failing file and the canonical-signature lookup.
        assert "requirements.py" in contents

    def test_returns_none_when_no_templates(self, tmp_path):
        """Error-severity failure but no template registered for the lint id."""
        package = tmp_path / "weather_mellea"
        # ``no-watsonx`` is a real lint id but has no registered template
        # AND is WARNING severity — but we synthesize it as error here so
        # the *only* reason for None is "no template registered". This
        # confirms the no-template path returns None cleanly.
        report_path = _write_step_7_report(
            package,
            lints=[
                {
                    "lint_id": "lint-with-no-template",
                    "verdict": "fail",
                    "severity": "error",
                    "effective_severity": "error",
                    "failures": [
                        {
                            "file": "pipeline.py",
                            "line": 12,
                            "column": None,
                            "message": "unmapped failure",
                            "rule_ref": None,
                        }
                    ],
                }
            ],
        )

        out = _bake_repair_prescriptions(package, lint_report_path=report_path)

        assert out is None
        # No prescription file should have been written.
        assert not (
            package / "intermediate" / "repair_prescriptions.md"
        ).exists()

    def test_skips_warnings(self, tmp_path):
        """Only WARNING-severity failures present — no prescription baked."""
        package = tmp_path / "weather_mellea"
        warning_lint = dict(_stdlib_arity_failure_lint())
        warning_lint["severity"] = "warning"
        warning_lint["effective_severity"] = "warning"
        report_path = _write_step_7_report(
            package,
            overall_verdict="pass",
            lints=[warning_lint],
        )

        out = _bake_repair_prescriptions(package, lint_report_path=report_path)

        assert out is None
        assert not (
            package / "intermediate" / "repair_prescriptions.md"
        ).exists()

    def test_returns_none_when_report_missing(self, tmp_path):
        """Defensive: missing report file → None, no crash."""
        package = tmp_path / "weather_mellea"
        package.mkdir()
        missing = package / "intermediate" / "step_7_report.json"

        out = _bake_repair_prescriptions(package, lint_report_path=missing)

        assert out is None


# ─── _collect_error_failures ────────────────────────────────────────────


class TestCollectErrorFailures:
    def test_collects_error_severity_only(self):
        report = {
            "lints": [
                _stdlib_arity_failure_lint(),
                {
                    "lint_id": "warn-lint",
                    "verdict": "fail",
                    "severity": "warning",
                    "effective_severity": "warning",
                    "failures": [
                        {
                            "file": "x.py",
                            "line": 1,
                            "column": None,
                            "message": "advisory",
                            "rule_ref": None,
                        }
                    ],
                },
            ]
        }

        out = _collect_error_failures(report)

        assert len(out) == 1
        lint_id, locs = out[0]
        assert lint_id == "stdlib-arity"
        assert locs == ["requirements.py:30"]

    def test_handles_empty_report(self):
        assert _collect_error_failures({}) == []
        assert _collect_error_failures({"lints": []}) == []


# ─── _build_lint_repair_system_prompt ───────────────────────────────────


class TestBuildLintRepairSystemPrompt:
    def test_includes_base_prompt_and_failures(self, tmp_path):
        prescription = (
            tmp_path / "weather_mellea" / "intermediate" / "repair_prescriptions.md"
        )
        prescription.parent.mkdir(parents=True)
        prescription.write_text("dummy")
        lint_report = prescription.parent / "step_7_report.json"
        lint_report.write_text("{}")

        prompt = _build_lint_repair_system_prompt(
            "BASE_PROMPT",
            failing_lints=[("stdlib-arity", ["requirements.py:30"])],
            prescriptions_path=prescription,
            lint_report_path=lint_report,
        )

        assert "BASE_PROMPT" in prompt
        assert "WRAPPER-SIDE LINT REPAIR" in prompt
        assert "stdlib-arity" in prompt
        assert "requirements.py:30" in prompt
        assert "DO NOT redo the full pipeline" in prompt
        # Prescription path is referenced (relative to package_dir).
        assert "repair_prescriptions.md" in prompt

    def test_handles_no_prescriptions_path(self, tmp_path):
        prompt = _build_lint_repair_system_prompt(
            "BASE",
            failing_lints=[("foo", [])],
            prescriptions_path=None,
            lint_report_path=tmp_path / "step_7_report.json",
        )

        assert "BASE" in prompt
        assert "No structured prescription templates matched" in prompt
        assert "step_7_report.json" in prompt

    def test_includes_descriptor_render_failure_reason(self, tmp_path):
        prompt = _build_lint_repair_system_prompt(
            "BASE",
            failing_lints=[("pipeline-entry-canonical", ["pipeline.py"])],
            prescriptions_path=None,
            lint_report_path=tmp_path / "step_7_report.json",
            descriptor_render_failure_reason=(
                "render_descriptor raised: RendererError: no module prefix of "
                "'config.SKILL_DESCRIPTION' resolves in surface"
            ),
        )

        assert "DESCRIPTOR RENDER FAILED" in prompt
        assert "config.SKILL_DESCRIPTION" in prompt
        # Should direct Claude to fix the descriptor JSON, not pipeline.py
        assert "descriptor JSON" in prompt
        # Lint list is still surfaced as a downstream consequence
        assert "pipeline-entry-canonical" in prompt


# ─── _spawn_claude_with_lint_repair ─────────────────────────────────────


class TestSpawnClaudeWithLintRepair:
    def _make_initial_argv(self, *, use_descriptor: bool = False) -> list[str]:
        return _build_claude_argv(
            spec_path=Path("/tmp/spec.md"),
            model="claude-fake",
            system_prompt="BASE",
            compile_settings_path=None,
            repair_mode=False,
            use_descriptor=use_descriptor,
        )

    def _common_kwargs(self, tmp_path, package):
        intermediate = package / "intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
        return dict(
            spec_path=tmp_path / "spec.md",
            model="claude-fake",
            base_system_prompt="BASE_PROMPT",
            compile_settings_path=None,
            use_descriptor=False,
            subprocess_env={},
            intermediate_dir=intermediate,
            package_dir=package,
            timeout=60,
            processing=MagicMock(),
        )

    def test_passes_through_when_lints_clean(self, tmp_path):
        """No repair spawn happens when lints pass on the first try."""
        package = tmp_path / "weather_mellea"
        spawn = MagicMock()
        render = MagicMock()
        lints = MagicMock(
            return_value=SimpleNamespace(overall_verdict="pass")
        )

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        assert spawn.call_count == 1
        assert render.call_count == 1
        assert lints.call_count == 1

    def test_invokes_repair_on_error_severity(self, tmp_path):
        """When lints fail with error severity, repair spawn fires + file written."""
        package = tmp_path / "weather_mellea"
        intermediate = package / "intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)

        # Write a failing report so prescription baking has input.
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        # First lint call returns fail; second returns pass (the repair fixed it).
        lint_call_count = {"n": 0}

        def fake_lints(pkg):
            lint_call_count["n"] += 1
            if lint_call_count["n"] == 1:
                return SimpleNamespace(overall_verdict="fail")
            return SimpleNamespace(overall_verdict="pass")

        spawn = MagicMock()
        render = MagicMock()

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        # 1 initial + 1 repair = 2 spawn calls.
        assert spawn.call_count == 2

        # Inspect the repair (second) spawn's argv: it must use the
        # repair slash command.
        second_call = spawn.call_args_list[1]
        repair_argv = second_call.kwargs["claude_argv"]
        slash_token = next(t for t in repair_argv if t.startswith("'./"))
        assert slash_token.startswith("'./mellea-fy-repair")

        # Prescription file must have been written between spawns.
        prescription = intermediate / "repair_prescriptions.md"
        assert prescription.exists()
        assert "stdlib-arity" in prescription.read_text() or \
            "requirements.py" in prescription.read_text()

    def test_exits_on_lint_pass_between_rounds(self, tmp_path):
        """First repair round actually fixes the failures → loop exits."""
        package = tmp_path / "weather_mellea"
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        call_n = {"n": 0}

        def fake_lints(pkg):
            call_n["n"] += 1
            # 1st: fail (triggers repair); 2nd: pass (loop exits).
            return SimpleNamespace(
                overall_verdict="fail" if call_n["n"] == 1 else "pass"
            )

        spawn = MagicMock()
        render = MagicMock()

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
            max_repair_rounds=2,
        )

        # 1 initial + 1 repair = 2 spawns; loop exited after pass.
        assert spawn.call_count == 2
        # 2 lint runs (one after initial spawn, one after repair spawn).
        assert call_n["n"] == 2

    def test_terminates_after_max_rounds(self, tmp_path):
        """Spawn always produces errors → loops exactly max_rounds + 1 times."""
        package = tmp_path / "weather_mellea"
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        lints = MagicMock(
            return_value=SimpleNamespace(overall_verdict="fail")
        )
        spawn = MagicMock()
        render = MagicMock()

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
            max_repair_rounds=2,
        )

        # 1 initial + 2 repair = 3 spawns; helper does NOT raise.
        assert spawn.call_count == 3

    def test_passes_use_descriptor_through_to_repair(self, tmp_path):
        """Initial argv has --use-descriptor → repair argv also has it."""
        package = tmp_path / "weather_mellea"
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        call_n = {"n": 0}

        def fake_lints(pkg):
            call_n["n"] += 1
            return SimpleNamespace(
                overall_verdict="fail" if call_n["n"] == 1 else "pass"
            )

        spawn = MagicMock()
        render = MagicMock()

        kwargs = self._common_kwargs(tmp_path, package)
        kwargs["use_descriptor"] = True
        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(use_descriptor=True),
            **kwargs,
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        repair_call = spawn.call_args_list[1]
        repair_argv = repair_call.kwargs["claude_argv"]
        slash_token = next(t for t in repair_argv if t.startswith("'./"))
        assert "--use-descriptor" in slash_token
        assert "./mellea-fy-repair" in slash_token

    def test_no_prescriptions_when_no_templates_still_spawns_repair(self, tmp_path):
        """No template match → no prescription file but repair still fires."""
        package = tmp_path / "weather_mellea"
        # Use a lint id with no template registered.
        _write_step_7_report(
            package,
            lints=[
                {
                    "lint_id": "lint-with-no-template",
                    "verdict": "fail",
                    "severity": "error",
                    "effective_severity": "error",
                    "failures": [
                        {
                            "file": "pipeline.py",
                            "line": 1,
                            "column": None,
                            "message": "no template",
                            "rule_ref": None,
                        }
                    ],
                }
            ],
        )

        call_n = {"n": 0}

        def fake_lints(pkg):
            call_n["n"] += 1
            return SimpleNamespace(
                overall_verdict="fail" if call_n["n"] == 1 else "pass"
            )

        spawn = MagicMock()
        render = MagicMock()

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        # 1 initial + 1 repair (still fires even without prescriptions).
        assert spawn.call_count == 2
        prescription = package / "intermediate" / "repair_prescriptions.md"
        assert not prescription.exists()


# ─── _lint_repair_retry_loop (direct invocation) ───────────────────────


class TestLintRepairRetryLoop:
    """The extracted retry loop must work when called in isolation.

    The composition refactor splits :func:`_spawn_claude_with_lint_repair`
    into Layer 1 (initial spawn via :func:`_spawn_claude`) and Layer 2
    (this retry loop). The loop assumes the initial spawn has ALREADY
    happened; the caller controls Layer 1 separately so it can run a
    resume-aware spawn instead.
    """

    def _common_kwargs(self, tmp_path, package):
        intermediate = package / "intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
        return dict(
            spec_path=tmp_path / "spec.md",
            model="claude-fake",
            base_system_prompt="BASE_PROMPT",
            compile_settings_path=None,
            use_descriptor=False,
            subprocess_env={},
            intermediate_dir=intermediate,
            package_dir=package,
            timeout=60,
            processing=MagicMock(),
        )

    def test_lint_repair_retry_loop_runs_in_isolation(self, tmp_path):
        """Call the loop directly (no Layer-1 spawn) with a failing lint
        state → confirms a repair spawn happens."""
        package = tmp_path / "weather_mellea"
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        # 1st lint: fail (triggers repair); 2nd: pass.
        call_n = {"n": 0}

        def fake_lints(pkg):
            call_n["n"] += 1
            return SimpleNamespace(
                overall_verdict="fail" if call_n["n"] == 1 else "pass"
            )

        spawn = MagicMock()
        render = MagicMock()

        _lint_repair_retry_loop(
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        # No initial spawn — only the repair re-spawn.
        assert spawn.call_count == 1
        # Render and lint both ran twice (post-spawn render/lint then
        # post-repair render/lint).
        assert render.call_count == 2
        assert call_n["n"] == 2

        # The (single) repair spawn must use the repair slash command.
        repair_argv = spawn.call_args_list[0].kwargs["claude_argv"]
        slash_token = next(t for t in repair_argv if t.startswith("'./"))
        assert slash_token.startswith("'./mellea-fy-repair")

    def test_lint_repair_retry_loop_skips_when_lints_pass_immediately(
        self, tmp_path
    ):
        """Clean lint state → no repair spawn, loop exits cleanly."""
        package = tmp_path / "weather_mellea"
        spawn = MagicMock()
        render = MagicMock()
        lints = MagicMock(
            return_value=SimpleNamespace(overall_verdict="pass")
        )

        _lint_repair_retry_loop(
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        # Render + lint each ran once; NO repair spawn.
        assert render.call_count == 1
        assert lints.call_count == 1
        assert spawn.call_count == 0


class TestSpawnClaudeWithLintRepairBackwardCompat:
    """The wrapper :func:`_spawn_claude_with_lint_repair` still does
    spawn-then-retry-loop, preserving the pre-refactor contract.
    """

    def _common_kwargs(self, tmp_path, package):
        intermediate = package / "intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
        return dict(
            spec_path=tmp_path / "spec.md",
            model="claude-fake",
            base_system_prompt="BASE_PROMPT",
            compile_settings_path=None,
            use_descriptor=False,
            subprocess_env={},
            intermediate_dir=intermediate,
            package_dir=package,
            timeout=60,
            processing=MagicMock(),
        )

    def _make_initial_argv(self) -> list[str]:
        return _build_claude_argv(
            spec_path=Path("/tmp/spec.md"),
            model="claude-fake",
            system_prompt="BASE",
            compile_settings_path=None,
            repair_mode=False,
            use_descriptor=False,
        )

    def test_spawn_claude_with_lint_repair_still_works_as_before(self, tmp_path):
        """One initial spawn + one repair spawn when lints fail then pass."""
        package = tmp_path / "weather_mellea"
        _write_step_7_report(
            package, lints=[_stdlib_arity_failure_lint()]
        )

        call_n = {"n": 0}

        def fake_lints(pkg):
            call_n["n"] += 1
            return SimpleNamespace(
                overall_verdict="fail" if call_n["n"] == 1 else "pass"
            )

        spawn = MagicMock()
        render = MagicMock()

        _spawn_claude_with_lint_repair(
            initial_claude_argv=self._make_initial_argv(),
            **self._common_kwargs(tmp_path, package),
            run_post_session_lints=fake_lints,
            run_writer_renderer=render,
            spawn_fn=spawn,
        )

        # 1 initial spawn (the legacy single-shot path) + 1 repair spawn.
        assert spawn.call_count == 2
        # First spawn uses ./mellea-fy (not repair); second uses repair.
        initial_argv = spawn.call_args_list[0].kwargs["claude_argv"]
        initial_slash = next(t for t in initial_argv if t.startswith("'./"))
        assert initial_slash.startswith("'./mellea-fy ")
        repair_argv = spawn.call_args_list[1].kwargs["claude_argv"]
        repair_slash = next(t for t in repair_argv if t.startswith("'./"))
        assert repair_slash.startswith("'./mellea-fy-repair")


# ─── Repair argv shape ─────────────────────────────────────────────────


class TestRepairArgvShape:
    def test_uses_mellea_fy_repair_slash_command(self):
        argv = _build_claude_argv(
            spec_path=Path("/tmp/spec.md"),
            model="claude-fake",
            system_prompt="WRAPPER REPAIR PROMPT",
            compile_settings_path=None,
            repair_mode=True,
            use_descriptor=False,
        )
        slash_token = next(t for t in argv if t.startswith("'./"))
        assert slash_token.startswith("'./mellea-fy-repair")
        assert "./mellea-fy " not in slash_token  # not the legacy command

    def test_initial_argv_uses_mellea_fy(self):
        """Sanity: repair_mode=False stays on ./mellea-fy."""
        argv = _build_claude_argv(
            spec_path=Path("/tmp/spec.md"),
            model="claude-fake",
            system_prompt="BASE",
            compile_settings_path=None,
            repair_mode=False,
            use_descriptor=False,
        )
        slash_token = next(t for t in argv if t.startswith("'./"))
        assert slash_token.startswith("'./mellea-fy ")
        assert "mellea-fy-repair" not in slash_token


# ─── compile() dispatch via CLI flag ───────────────────────────────────


class TestCompileFlagDispatch:
    """The opt-in flag dispatches to the new spawn helper; default is unchanged."""

    @staticmethod
    def _patch_compile_env(monkeypatch):
        """Stub the side-effects compile() does before/around the dispatch.

        Patches the Anthropic client, the proxy server socket, and the
        ``subprocess.call`` used to clear the terminal. Returns nothing
        — used purely for its monkeypatching effects.
        """
        monkeypatch.setattr(
            "mellea_skills_compiler.compile.mellea_skills.Anthropic",
            lambda: MagicMock(
                models=MagicMock(
                    list=MagicMock(
                        return_value=[MagicMock(id="claude-sonnet-fake")]
                    )
                )
            ),
        )

        def _fake_server(*_a, **_k):
            server = MagicMock()
            server.server_address = ("127.0.0.1", 9999)
            return server

        monkeypatch.setattr(
            "mellea_skills_compiler.compile.mellea_skills.socketserver.ThreadingTCPServer",
            _fake_server,
        )
        monkeypatch.setattr(
            "mellea_skills_compiler.compile.mellea_skills.subprocess.call",
            lambda *a, **k: 0,
        )

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._lint_repair_retry_loop")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_compile_with_neither_flag_runs_legacy_spawn_only(
        self, mock_resume, mock_retry_loop, mock_spawn, tmp_path, monkeypatch
    ):
        """Baseline: no flags → only the legacy single spawn."""
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        mock_spawn.side_effect = RuntimeError("STOP_AFTER_SPAWN")
        mock_retry_loop.side_effect = RuntimeError("STOP_AFTER_RETRY_LOOP")
        mock_resume.side_effect = RuntimeError("STOP_AFTER_RESUME")

        self._patch_compile_env(monkeypatch)

        with pytest.raises(Exception):
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=False,
                repair_on_lint_failure=False,
            )

        assert mock_spawn.called, "neither flag → legacy _spawn_claude must fire"
        assert not mock_resume.called, (
            "neither flag → resume helper must NOT fire"
        )
        assert not mock_retry_loop.called, (
            "neither flag → lint-repair retry loop must NOT fire"
        )

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._lint_repair_retry_loop")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_compile_with_only_repair_flag_skips_resume_layer(
        self, mock_resume, mock_retry_loop, mock_spawn, tmp_path, monkeypatch
    ):
        """Repair-only: legacy spawn for Layer 1, retry loop for Layer 2."""
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        # Allow Layer 1 to proceed; raise inside Layer 2 to stop early.
        mock_spawn.return_value = None
        mock_retry_loop.side_effect = RuntimeError("STOP_AFTER_RETRY_LOOP")
        mock_resume.side_effect = RuntimeError("STOP_AFTER_RESUME")

        self._patch_compile_env(monkeypatch)

        with pytest.raises(Exception):
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=False,
                repair_on_lint_failure=True,
            )

        assert mock_spawn.called, "repair only → Layer 1 legacy spawn must fire"
        assert mock_retry_loop.called, (
            "repair only → Layer 2 lint-repair retry loop must fire"
        )
        assert not mock_resume.called, (
            "repair only → resume helper must NOT fire"
        )

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._lint_repair_retry_loop")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_compile_with_only_resume_flag_skips_lint_repair_layer(
        self, mock_resume, mock_retry_loop, mock_spawn, tmp_path, monkeypatch
    ):
        """Resume-only: resume helper for Layer 1, NO retry loop for Layer 2."""
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        mock_spawn.side_effect = RuntimeError("STOP_AFTER_SPAWN")
        mock_retry_loop.side_effect = RuntimeError("STOP_AFTER_RETRY_LOOP")
        mock_resume.side_effect = RuntimeError("STOP_AFTER_RESUME")

        self._patch_compile_env(monkeypatch)

        with pytest.raises(Exception):
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=True,
                repair_on_lint_failure=False,
            )

        assert mock_resume.called, "resume only → resume helper must fire"
        assert not mock_spawn.called, (
            "resume only → legacy _spawn_claude must NOT fire (resume "
            "helper internally spawns Claude)"
        )
        assert not mock_retry_loop.called, (
            "resume only → lint-repair retry loop must NOT fire"
        )

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._lint_repair_retry_loop")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_compile_with_both_flags_invokes_both_layers(
        self, mock_resume, mock_retry_loop, mock_spawn, tmp_path, monkeypatch
    ):
        """Both flags → resume helper THEN lint-repair retry loop."""
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        # Resume helper proceeds; retry loop raises so we stop here.
        call_order: list[str] = []

        def _resume(**_kw):
            call_order.append("resume")

        def _retry(**_kw):
            call_order.append("retry")
            raise RuntimeError("STOP_AFTER_RETRY_LOOP")

        mock_resume.side_effect = _resume
        mock_retry_loop.side_effect = _retry
        mock_spawn.side_effect = RuntimeError("STOP_AFTER_SPAWN")

        self._patch_compile_env(monkeypatch)

        with pytest.raises(Exception):
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=True,
                repair_on_lint_failure=True,
            )

        assert mock_resume.called, "both flags → resume helper must fire"
        assert mock_retry_loop.called, (
            "both flags → lint-repair retry loop must fire"
        )
        assert not mock_spawn.called, (
            "both flags → legacy _spawn_claude must NOT fire directly "
            "(resume helper handles Layer 1)"
        )
        # Resume runs FIRST (Layer 1), then retry loop (Layer 2).
        assert call_order == ["resume", "retry"], (
            f"expected resume then retry, got {call_order}"
        )
