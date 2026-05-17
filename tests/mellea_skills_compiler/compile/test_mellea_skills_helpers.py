"""Unit tests for the deterministic pre-mellea-fy helpers in
mellea_skills_compiler.compile.mellea_skills.

These cover the small private helpers (`derive_package_name`,
`mirror_companion_dirs`, `resolve_runtime_defaults`,
`write_runtime_directive`, `build_system_prompt`) that handle the
plumbing around the LLM-driven compile() orchestrator. None of these
require Claude Code, an LLM, network access, or a mellea install.
"""

import json
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.claude_directives import (
    build_system_prompt,
    derive_package_name,
    mirror_companion_dirs,
    resolve_runtime_defaults,
    write_runtime_directive,
)
from mellea_skills_compiler.compile.mellea_skills import _tolerant_name_extract


class TestDerivePackageName:
    """Rule OUT-2: lowercase, hyphens/spaces -> underscores, append `_mellea`."""

    def test_md_spec_uses_frontmatter_name(self):
        result = derive_package_name(Path("/x/y/spec.md"), {"name": "weather"})
        assert result == "weather_mellea"

    def test_md_spec_falls_back_to_parent_dir_name_when_no_frontmatter_name(self):
        result = derive_package_name(Path("/x/weather/spec.md"), {})
        assert result == "weather_mellea"

    def test_md_spec_falls_back_to_parent_dir_name_when_frontmatter_is_none(self):
        result = derive_package_name(Path("/x/weather/spec.md"), None)
        assert result == "weather_mellea"

    def test_dir_input_uses_directory_name(self, tmp_path):
        skill_dir = tmp_path / "crewai-job-posting"
        skill_dir.mkdir()
        result = derive_package_name(skill_dir, None)
        assert result == "crewai_job_posting_mellea"

    def test_dir_input_prefers_frontmatter_name_over_dir_name(self, tmp_path):
        """When dir name and frontmatter name diverge, frontmatter wins.

        This is the bug found by the eval-corpus overnight batch:
        directories like ``0f596e5feb22/`` had spec.md frontmatter
        ``name: coding-agent``. The wrapper used to emit
        ``0f596e5feb22_mellea/`` while the LLM (which reads frontmatter)
        wrote intermediate JSON to ``coding_agent_mellea/``. They must
        agree — frontmatter is the source of truth per Rule OUT-2.
        """
        skill_dir = tmp_path / "0f596e5feb22"
        skill_dir.mkdir()
        result = derive_package_name(skill_dir, {"name": "coding-agent"})
        assert result == "coding_agent_mellea"

    def test_dir_input_falls_back_to_dir_when_frontmatter_lacks_name(self, tmp_path):
        skill_dir = tmp_path / "my-crewai-skill"
        skill_dir.mkdir()
        result = derive_package_name(skill_dir, {"description": "no name field"})
        assert result == "my_crewai_skill_mellea"

    def test_normalises_uppercase(self):
        result = derive_package_name(Path("/x/y/spec.md"), {"name": "WeatherSkill"})
        assert result == "weatherskill_mellea"

    def test_normalises_spaces(self):
        result = derive_package_name(Path("/x/y/spec.md"), {"name": "weather skill"})
        assert result == "weather_skill_mellea"

    def test_collapses_double_underscores(self):
        result = derive_package_name(Path("/x/y/spec.md"), {"name": "weather--skill"})
        assert result == "weather_skill_mellea"
        assert "__" not in result

    def test_strips_leading_trailing_underscores(self):
        result = derive_package_name(Path("/x/y/spec.md"), {"name": "_weather_"})
        assert result == "weather_mellea"

    def test_falls_back_to_skill_for_empty_input(self):
        # Empty frontmatter name AND empty parent dir name -> safe fallback "skill".
        # Path("/spec.md").parent.name == "" so the helper's `.strip("_") or "skill"`
        # fallback kicks in.
        result = derive_package_name(Path("/spec.md"), {"name": ""})
        assert result == "skill_mellea"


class TestTolerantNameExtract:
    """Fallback name extractor used when strict YAML parsing fails.

    Real-world specs sometimes have unquoted colons / parens inside their
    ``description:`` value (e.g. "Roles: provider (Anbieter), deployer
    (Betreiber)"). yaml.safe_load chokes on those — but the slash command
    is more permissive and the wrapper must agree on the package name.
    This helper does a regex extraction of just the ``name:`` field so
    the wrapper's package-directory choice still matches the LLM's.
    """

    def test_extracts_name_from_clean_frontmatter(self, tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("---\nname: weather\ndescription: clean\n---\nbody\n")
        assert _tolerant_name_extract(spec) == {"name": "weather"}

    def test_extracts_name_when_yaml_would_fail(self, tmp_path):
        # description contains unquoted "Roles:" which breaks strict YAML.
        spec = tmp_path / "spec.md"
        spec.write_text(
            "---\n"
            "name: eu-ai-act-classification\n"
            "description: Classify an AI system. Roles: provider (Anbieter), deployer.\n"
            "---\n"
            "body\n"
        )
        assert _tolerant_name_extract(spec) == {"name": "eu-ai-act-classification"}

    def test_strips_quotes_around_name_value(self, tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("---\nname: 'quoted-name'\n---\nbody\n")
        assert _tolerant_name_extract(spec) == {"name": "quoted-name"}
        spec.write_text('---\nname: "double-quoted"\n---\nbody\n')
        assert _tolerant_name_extract(spec) == {"name": "double-quoted"}

    def test_returns_empty_when_no_frontmatter_block(self, tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("# just a body\n\nNo frontmatter here.\n")
        assert _tolerant_name_extract(spec) == {}

    def test_returns_empty_when_frontmatter_lacks_name(self, tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("---\ndescription: no name field\n---\nbody\n")
        assert _tolerant_name_extract(spec) == {}

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _tolerant_name_extract(tmp_path / "nope.md") == {}


class TestMirrorCompanionDirs:
    """Rule OUT-6: mirror scripts/, references/, assets/ from skill -> package."""

    def test_mirrors_present_directories(self, tmp_path):
        skill = tmp_path / "skill"
        package = tmp_path / "skill" / "skill_mellea"
        (skill / "scripts" / "bash").mkdir(parents=True)
        (skill / "scripts" / "bash" / "x.sh").write_text("hello")

        mirrored = mirror_companion_dirs(skill, package)

        assert (package / "scripts" / "bash" / "x.sh").exists()
        assert (package / "scripts" / "bash" / "x.sh").read_text() == "hello"
        assert mirrored == ["scripts"]

    def test_mirrors_multiple_companion_dirs(self, tmp_path):
        skill = tmp_path / "skill"
        package = tmp_path / "skill" / "skill_mellea"
        for name in ("scripts", "references", "assets"):
            (skill / name).mkdir(parents=True)
            (skill / name / "file.txt").write_text(f"contents of {name}")

        mirrored = mirror_companion_dirs(skill, package)

        for name in ("scripts", "references", "assets"):
            assert (package / name / "file.txt").exists()
            assert (package / name / "file.txt").read_text() == f"contents of {name}"
            assert name in mirrored
        assert len(mirrored) == 3

    def test_skips_absent_directories(self, tmp_path):
        skill = tmp_path / "skill"
        package = tmp_path / "skill" / "skill_mellea"
        (skill / "scripts").mkdir(parents=True)
        (skill / "scripts" / "x.sh").write_text("hi")

        mirrored = mirror_companion_dirs(skill, package)

        assert mirrored == ["scripts"]
        assert (package / "scripts").exists()
        assert not (package / "references").exists()
        assert not (package / "assets").exists()

    def test_idempotent_on_second_call(self, tmp_path):
        skill = tmp_path / "skill"
        package = tmp_path / "skill" / "skill_mellea"
        (skill / "scripts").mkdir(parents=True)
        (skill / "scripts" / "x.sh").write_text("first")

        first = mirror_companion_dirs(skill, package)
        # Second call must not raise (relies on dirs_exist_ok=True).
        second = mirror_companion_dirs(skill, package)

        assert first == second == ["scripts"]
        assert (package / "scripts" / "x.sh").read_text() == "first"

    def test_creates_package_dir_if_missing(self, tmp_path):
        skill = tmp_path / "skill"
        skill.mkdir()
        package = tmp_path / "nonexistent_parent" / "skill_mellea"
        assert not package.exists()

        mirrored = mirror_companion_dirs(skill, package)

        assert package.exists()
        assert package.is_dir()
        assert mirrored == []

    def test_returns_empty_when_no_companion_dirs(self, tmp_path):
        skill = tmp_path / "skill"
        skill.mkdir()
        package = tmp_path / "skill" / "skill_mellea"

        mirrored = mirror_companion_dirs(skill, package)

        assert mirrored == []


class TestResolveRuntimeDefaults:
    """Precedence: CLI override > defaults file > built-in fallback."""

    def _write_defaults(self, tmp_path: Path, payload: str) -> None:
        defaults_dir = tmp_path / ".claude" / "data"
        defaults_dir.mkdir(parents=True)
        (defaults_dir / "runtime_defaults.json").write_text(payload)

    def test_uses_defaults_file_when_no_override(self, tmp_path, monkeypatch):
        self._write_defaults(
            tmp_path,
            json.dumps({"backend": "ollama", "model_id": "granite4.1:8b"}),
        )
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults(None, None)

        assert backend == "ollama"
        assert model_id == "granite4.1:8b"
        assert "defaults file" in source

    def test_cli_backend_override_wins(self, tmp_path, monkeypatch):
        self._write_defaults(
            tmp_path,
            json.dumps({"backend": "ollama", "model_id": "granite4.1:8b"}),
        )
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults("vllm", None)

        assert backend == "vllm"
        assert model_id == "granite4.1:8b"
        assert "command-line override" in source

    def test_cli_both_overrides_win(self, tmp_path, monkeypatch):
        self._write_defaults(
            tmp_path,
            json.dumps({"backend": "ollama", "model_id": "granite4.1:8b"}),
        )
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults("vllm", "mistral-7b")

        assert backend == "vllm"
        assert model_id == "mistral-7b"
        assert "command-line override" in source

    def test_falls_back_to_builtin_when_file_missing(self, tmp_path, monkeypatch):
        # No .claude/data/ written under tmp_path.
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults(None, None)

        assert backend == "ollama"
        assert model_id == "granite4.1:8b"
        assert "built-in fallback" in source

    def test_falls_back_to_builtin_when_file_malformed(self, tmp_path, monkeypatch):
        self._write_defaults(tmp_path, "{not valid json")
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults(None, None)

        # On JSON decode error the helper logs a warning and keeps the
        # built-in fallback values; source remains the built-in fallback
        # string because the try/except short-circuits before the source
        # is updated.
        assert backend == "ollama"
        assert model_id == "granite4.1:8b"
        assert "built-in fallback" in source

    def test_falls_back_to_builtin_when_file_missing_keys(self, tmp_path, monkeypatch):
        # Empty JSON object: file is readable but contains no keys.
        # `data.get("backend", file_backend)` returns the built-in fallback
        # for both keys; source is updated to "defaults file (...)" because
        # the read succeeded.
        self._write_defaults(tmp_path, json.dumps({}))
        monkeypatch.chdir(tmp_path)

        backend, model_id, source = resolve_runtime_defaults(None, None)

        assert backend == "ollama"
        assert model_id == "granite4.1:8b"
        assert "defaults file" in source


class TestWriteRuntimeDirective:
    """Persist the chosen backend/model so the post-compile lint can check drift."""

    def test_writes_json_with_required_fields(self, tmp_path):
        intermediate = tmp_path / "intermediate"

        path = write_runtime_directive(
            intermediate, "ollama", "granite4.1:8b", "defaults file (...)"
        )

        assert path == intermediate / "runtime_directive.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["format_version"] == "1.0"
        assert data["backend"] == "ollama"
        assert data["model_id"] == "granite4.1:8b"
        assert data["source"] == "defaults file (...)"

    def test_creates_intermediate_dir_if_missing(self, tmp_path):
        intermediate = tmp_path / "deep" / "nested" / "intermediate"
        assert not intermediate.exists()

        write_runtime_directive(intermediate, "ollama", "granite4.1:8b", "src")

        assert intermediate.exists()
        assert (intermediate / "runtime_directive.json").exists()

    def test_returns_path_to_written_file(self, tmp_path):
        intermediate = tmp_path / "intermediate"

        path = write_runtime_directive(intermediate, "ollama", "granite4.1:8b", "src")

        assert isinstance(path, Path)
        assert path.exists()
        assert path == intermediate / "runtime_directive.json"

    def test_overwrites_existing_directive(self, tmp_path):
        intermediate = tmp_path / "intermediate"

        write_runtime_directive(intermediate, "ollama", "granite4.1:8b", "first")
        path = write_runtime_directive(intermediate, "vllm", "mistral-7b", "second")

        data = json.loads(path.read_text())
        assert data["backend"] == "vllm"
        assert data["model_id"] == "mistral-7b"
        assert data["source"] == "second"


class TestBuildSystemPrompt:
    """Assemble the instruction string passed to the mellea-fy slash command."""

    def test_includes_backend_value(self):
        prompt = build_system_prompt("ollama", "granite4.1:8b", "defaults file")
        assert "ollama" in prompt

    def test_includes_model_id_value(self):
        prompt = build_system_prompt("ollama", "granite4.1:8b", "defaults file")
        assert "granite4.1:8b" in prompt

    def test_includes_source(self):
        prompt = build_system_prompt(
            "ollama", "granite4.1:8b", "defaults file (xyz.json)"
        )
        assert "defaults file (xyz.json)" in prompt

    def test_includes_autonomous_run_directive(self):
        prompt = build_system_prompt("ollama", "granite4.1:8b", "src")
        assert "Run the complete 10-step pipeline" in prompt

    def test_quotes_values_with_repr(self):
        # A backend with embedded single quotes should be safely repr'd
        # rather than splatted in bare. The repr of "o'l'l'ama" wraps it
        # in double quotes (Python's repr picks the safer quote style).
        prompt = build_system_prompt("o'l'l'ama", "granite4.1:8b", "src")
        assert repr("o'l'l'ama") in prompt
        assert repr("granite4.1:8b") in prompt
