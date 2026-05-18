"""Unit tests for F6 — pipeline resume on early turn-end.

Covers:

* :func:`_detect_completed_steps` — artefact-based step-completion detection
  used to decide whether Claude's session ended with the pipeline finished
  or mid-way.
* :func:`_build_resume_system_prompt` — augments the base system prompt
  with a "resume from Step N+1" directive listing what's already done.
* :func:`_spawn_claude_with_resume` — the loop wrapping :func:`_spawn_claude`
  that detects the gap and re-invokes Claude with the resume directive.
* :func:`compile` — verifies the opt-in flag (``resume_on_early_end``)
  dispatches to the new path; default behaviour stays single-shot.
* :func:`build_system_prompt` — anti-narration directive present + stable.

The tests must NEVER spawn a real ``claude`` subprocess. ``_spawn_claude``
is monkey-patched in every test that touches it.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mellea_skills_compiler.compile import mellea_skills
from mellea_skills_compiler.compile.claude_directives import build_system_prompt
from mellea_skills_compiler.compile.mellea_skills import (
    _build_resume_system_prompt,
    _detect_completed_steps,
    _spawn_claude_with_resume,
    _STEP_ARTEFACTS,
)


def _write_all_step_artefacts(package_dir: Path) -> None:
    """Synthesize a package directory with every step's canonical artefact.

    Used by tests that need to simulate "the previous spawn produced
    everything". Writes the FIRST candidate path for each step (the
    canonical one); for steps whose canonical path is a glob, writes a
    concrete filename matching the glob.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "intermediate").mkdir(exist_ok=True)
    (package_dir / "intermediate" / "classification.json").write_text("{}")
    (package_dir / "intermediate" / "inventory.json").write_text("{}")
    (package_dir / "intermediate" / "element_mapping.json").write_text("{}")
    (package_dir / "intermediate" / "dependency_plan.json").write_text("{}")
    (package_dir / "intermediate" / "fixtures_emission.json").write_text("{}")
    (package_dir / "intermediate" / "descriptor_emission.json").write_text("{}")
    (package_dir / "melleafy.json").write_text("{}")


class TestDetectCompletedSteps:
    """Per-step artefact existence is the resume signal."""

    def test_detect_completed_steps_finds_all_seven_artefacts(self, tmp_path):
        package = tmp_path / "weather_mellea"
        _write_all_step_artefacts(package)

        completed = _detect_completed_steps(package)

        expected = [key for key, _ in _STEP_ARTEFACTS]
        assert completed == expected
        assert len(completed) == 7

    def test_detect_completed_steps_finds_partial(self, tmp_path):
        # Mirror the empirical dpia-sentinel failure: only Step 0 produced.
        package = tmp_path / "dpia_sentinel_mellea"
        (package / "intermediate").mkdir(parents=True)
        (package / "intermediate" / "classification.json").write_text("{}")

        completed = _detect_completed_steps(package)

        assert completed == ["step_0_classify"]

    def test_detect_completed_steps_finds_descriptor_emission(self, tmp_path):
        # Descriptor mode writes the canonical filename
        # ``intermediate/descriptor_emission.json``. Originally the
        # detector matched a stale glob ``*.descriptor.json`` (a name
        # pattern that never existed on disk), so descriptor-mode F6
        # detection silently fell through to the pipeline.py fallback
        # (which is wrapper-rendered POST-session and therefore never
        # present at F6 detection time). This test pins the canonical
        # filename so the regression cannot recur.
        package = tmp_path / "weather_mellea"
        (package / "intermediate").mkdir(parents=True)
        (package / "intermediate" / "descriptor_emission.json").write_text("{}")

        completed = _detect_completed_steps(package)

        assert "step_5_descriptor" in completed

    def test_detect_completed_steps_returns_empty_when_nothing_done(self, tmp_path):
        # Claude died before Step 0 — package_dir may not even exist yet.
        package = tmp_path / "weather_mellea"
        package.mkdir()

        completed = _detect_completed_steps(package)

        assert completed == []

    def test_detect_completed_steps_handles_missing_package_dir(self, tmp_path):
        # Defensive: the detector should not raise when called against a
        # nonexistent directory (early-failure path before package_dir
        # was created at all).
        package = tmp_path / "never_created_mellea"

        completed = _detect_completed_steps(package)

        assert completed == []

    def test_step_5_alternative_path_via_pipeline_py(self, tmp_path):
        # In legacy mode Step 5 emits per-element code into `pipeline.py`
        # at the package root. The detector treats pipeline.py existence
        # as Step 5 evidence (descriptor mode uses the *.descriptor.json
        # glob — covered by test_detect_completed_steps_finds_descriptor_glob).
        package = tmp_path / "weather_mellea"
        package.mkdir(parents=True)
        (package / "pipeline.py").write_text("# stub\n")

        completed = _detect_completed_steps(package)

        assert "step_5_descriptor" in completed

    def test_step_5_NOT_signalled_by_config_emission(self, tmp_path):
        # Regression: the writer-renderer synthesizes config_emission.json
        # from runtime_directive.json when the LLM didn't write it. So its
        # presence does NOT prove Step 5 ran. The empirical dpia-sentinel
        # failure had config_emission.json AND only Step 0 actually done.
        package = tmp_path / "weather_mellea"
        (package / "intermediate").mkdir(parents=True)
        (package / "intermediate" / "config_emission.json").write_text("{}")

        completed = _detect_completed_steps(package)

        assert "step_5_descriptor" not in completed


class TestResumeSystemPrompt:
    """The resume directive must explicitly list completed steps and tell
    Claude to start at the next one with an immediate tool call."""

    def test_resume_system_prompt_includes_completed_steps(self):
        base = "BASE_PROMPT"
        prompt = _build_resume_system_prompt(
            base, ["step_0_classify", "step_1a_inventory"]
        )

        assert "BASE_PROMPT" in prompt
        assert "Step 0" in prompt
        assert "Step 1a" in prompt
        # Resume directive headline
        assert "PIPELINE RESUME" in prompt
        # Anti-narration directive
        assert "tool call" in prompt.lower()
        assert "forbidden" in prompt.lower()

    def test_resume_system_prompt_names_next_step(self):
        prompt = _build_resume_system_prompt(
            "BASE", ["step_0_classify", "step_1a_inventory"]
        )
        # The first incomplete step is step_2_map.
        assert "Step 2" in prompt

    def test_resume_system_prompt_handles_empty_completed(self):
        prompt = _build_resume_system_prompt("BASE", [])

        assert "BASE" in prompt
        assert "PIPELINE RESUME" in prompt
        # Empty case still steers to Step 0.
        assert "Step 0" in prompt


class TestSpawnClaudeWithResume:
    """The resume loop must respect max_resumes and stop early when done."""

    def _make_kwargs(self, tmp_path, package_dir):
        intermediate = package_dir / "intermediate"
        intermediate.mkdir(parents=True, exist_ok=True)
        return dict(
            spec_path=tmp_path / "spec.md",
            model="claude-fake",
            base_system_prompt="BASE_PROMPT",
            compile_settings_path=None,
            repair_mode=False,
            use_descriptor=False,
            subprocess_env={},
            intermediate_dir=intermediate,
            package_dir=package_dir,
            timeout=60,
            processing=MagicMock(),
        )

    def test_resume_loop_calls_spawn_once_when_complete(self, tmp_path):
        package = tmp_path / "weather_mellea"
        spawn = MagicMock()

        # On the first spawn, populate ALL step artefacts — so the loop
        # detects completion and returns without a second spawn.
        def fake_spawn(**_kwargs):
            _write_all_step_artefacts(package)

        spawn.side_effect = fake_spawn

        _spawn_claude_with_resume(
            **self._make_kwargs(tmp_path, package),
            max_resumes=3,
            spawn_fn=spawn,
        )

        assert spawn.call_count == 1

    def test_resume_loop_resumes_when_step_5_missing_but_step_6_present(
        self, tmp_path
    ):
        """Regression for the nil-contract overnight failure: Claude wrote
        ``melleafy.json`` (Step 6) while skipping ``descriptor_emission.json``
        (Step 5). Originally F6's termination check only required Step 6,
        so it returned silently — the wrapper then crashed at smoke check
        with the misleading "pipeline.py missing" error. F6 must now insist
        on BOTH Step 5 and Step 6 before declaring the pipeline complete.
        """
        package = tmp_path / "nil_contract_mellea"
        spawn = MagicMock()
        call_state = {"count": 0}

        def fake_spawn(**_kwargs):
            call_state["count"] += 1
            (package / "intermediate").mkdir(parents=True, exist_ok=True)
            if call_state["count"] == 1:
                # First spawn: Claude produces Step 0..4 + Step 6 but
                # skips Step 5 entirely (the empirical nil-contract bug).
                (package / "intermediate" / "classification.json").write_text("{}")
                (package / "intermediate" / "inventory.json").write_text("{}")
                (package / "intermediate" / "element_mapping.json").write_text(
                    "{}"
                )
                (package / "intermediate" / "dependency_plan.json").write_text(
                    "{}"
                )
                (package / "intermediate" / "fixtures_emission.json").write_text(
                    "{}"
                )
                (package / "melleafy.json").write_text("{}")
            else:
                # Resume round writes the missing descriptor.
                (package / "intermediate" / "descriptor_emission.json").write_text(
                    "{}"
                )

        spawn.side_effect = fake_spawn

        _spawn_claude_with_resume(
            **self._make_kwargs(tmp_path, package),
            max_resumes=3,
            spawn_fn=spawn,
        )

        # Must have re-spawned once because Step 5 was missing.
        assert spawn.call_count == 2

    def test_resume_loop_calls_spawn_twice_on_early_end(self, tmp_path):
        package = tmp_path / "weather_mellea"
        spawn = MagicMock()
        call_state = {"count": 0}

        def fake_spawn(**_kwargs):
            call_state["count"] += 1
            if call_state["count"] == 1:
                # First spawn: only Step 0 done (mimics dpia-sentinel).
                (package / "intermediate").mkdir(parents=True, exist_ok=True)
                (package / "intermediate" / "classification.json").write_text("{}")
            else:
                # Second spawn (the resume): produce the rest.
                _write_all_step_artefacts(package)

        spawn.side_effect = fake_spawn

        _spawn_claude_with_resume(
            **self._make_kwargs(tmp_path, package),
            max_resumes=3,
            spawn_fn=spawn,
        )

        assert spawn.call_count == 2

    def test_resume_loop_respects_max_resumes(self, tmp_path):
        package = tmp_path / "weather_mellea"
        spawn = MagicMock()

        # Every spawn produces only Step 0 — never makes progress.
        # With the forward-progress check, the loop should abort after
        # the SECOND spawn (round 1 produced step_0; round 2 produced
        # the same; no new artefacts → bail).
        def fake_spawn(**_kwargs):
            (package / "intermediate").mkdir(parents=True, exist_ok=True)
            (package / "intermediate" / "classification.json").write_text("{}")

        spawn.side_effect = fake_spawn

        _spawn_claude_with_resume(
            **self._make_kwargs(tmp_path, package),
            max_resumes=3,
            spawn_fn=spawn,
        )

        # First spawn + at least one resume attempt; the forward-progress
        # check bails after the second-spawn no-op. Either way, the loop
        # must NOT exceed (1 initial + max_resumes) total spawns.
        assert spawn.call_count <= 1 + 3
        # And it must not have raised: the helper logs and returns.

    def test_resume_loop_no_artefacts_at_all_still_terminates(self, tmp_path):
        # Claude crashed before producing anything. Loop still terminates
        # after the initial spawn and the first resume attempt finds
        # nothing new (the second spawn also produces nothing → bail).
        package = tmp_path / "weather_mellea"
        spawn = MagicMock(side_effect=lambda **_k: None)

        _spawn_claude_with_resume(
            **self._make_kwargs(tmp_path, package),
            max_resumes=3,
            spawn_fn=spawn,
        )

        # Initial + at most max_resumes additional invocations.
        assert spawn.call_count <= 1 + 3


class TestCompileFlagDispatch:
    """The opt-in flag must dispatch to the resume path; default is unchanged."""

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_resume_disabled_by_default(
        self, mock_resume, mock_spawn, tmp_path, monkeypatch
    ):
        """Default ``compile(...)`` invocation must NOT call the resume path.

        Behaviour must be byte-identical to today — same single spawn call.
        """
        # Stub out everything compile() does before the spawn so we
        # exercise only the dispatch site.
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        # Make compile()'s post-spawn writer/lint chain a no-op by raising
        # an early benign error after the spawn dispatch — we only care
        # which spawn function was called.
        mock_spawn.side_effect = RuntimeError("STOP_AFTER_SPAWN")
        mock_resume.side_effect = RuntimeError("STOP_AFTER_SPAWN")

        # Patch out the Anthropic client.models.list() call.
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

        # Patch out the proxy server (avoid actually binding a socket).
        # Return a MagicMock() *instance* so attribute access (e.g.
        # `.server_address[1]`) succeeds. Passing the raw MagicMock class
        # constructs a Mock (no auto-spec) on which attribute access raises.
        def _fake_server(*_args, **_kwargs):
            server = MagicMock()
            server.server_address = ("127.0.0.1", 9999)
            return server

        monkeypatch.setattr(
            "mellea_skills_compiler.compile.mellea_skills.socketserver.ThreadingTCPServer",
            _fake_server,
        )

        # Avoid the subprocess.call("clear") side effect.
        monkeypatch.setattr(
            "mellea_skills_compiler.compile.mellea_skills.subprocess.call",
            lambda *a, **k: 0,
        )

        with pytest.raises(Exception):  # STOP_AFTER_SPAWN propagates
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=False,
            )

        assert mock_spawn.called, "default should call _spawn_claude"
        assert not mock_resume.called, (
            "default must NOT route through _spawn_claude_with_resume"
        )

    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude")
    @patch("mellea_skills_compiler.compile.mellea_skills._spawn_claude_with_resume")
    def test_resume_enabled_uses_new_path(
        self, mock_resume, mock_spawn, tmp_path, monkeypatch
    ):
        spec_md = tmp_path / "spec.md"
        spec_md.write_text("---\nname: weather\n---\nbody\n")

        mock_spawn.side_effect = RuntimeError("STOP_AFTER_SPAWN")
        mock_resume.side_effect = RuntimeError("STOP_AFTER_SPAWN")

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

        def _fake_server(*_args, **_kwargs):
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

        with pytest.raises(Exception):
            mellea_skills.compile(
                spec_md,
                model="claude-sonnet-fake",
                resume_on_early_end=True,
            )

        assert mock_resume.called, (
            "--resume-on-early-end must route through "
            "_spawn_claude_with_resume"
        )
        assert not mock_spawn.called, (
            "resume helper should be the ONLY spawn entry point in the "
            "opt-in mode (it calls _spawn_claude internally, but compile() "
            "should not call _spawn_claude directly)"
        )


class TestAntiNarrationDirective:
    """The base system prompt must carry the anti-narration directive."""

    def test_system_prompt_includes_anti_narration_directive(self):
        prompt = build_system_prompt("ollama", "granite4.1:8b", "src")
        # Must mention the failure mode (Proceeding to...).
        assert "Proceeding to" in prompt
        # Must instruct immediate tool invocation.
        assert "IMMEDIATELY" in prompt
        # Must mention step boundary semantics.
        assert "step boundary" in prompt.lower() or "step boundaries" in prompt.lower()

    def test_system_prompt_byte_stable_with_directive(self):
        # Same inputs → same prompt every time (no random ordering /
        # dict-order non-determinism).
        a = build_system_prompt("ollama", "granite4.1:8b", "src")
        b = build_system_prompt("ollama", "granite4.1:8b", "src")
        assert a == b
