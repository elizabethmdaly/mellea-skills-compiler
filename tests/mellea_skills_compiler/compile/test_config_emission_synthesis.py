"""Regression tests for `synthesize_config_emission_from_directive`.

Background — the bug this pins:
    The deterministic writer at `.claude/melleafy/writers/config_writer.py`
    renders `config.py` from `<package>/intermediate/config_emission.json`.
    When the LLM (running the slash command in a sandboxed subprocess) failed
    to emit `config_emission.json` at all, the writer reported
    ``missing-emission``, no `config.py` was written, and the Step 7
    `runtime-defaults-bound` lint hard-failed on the missing file. Multiple
    overnight-batch skills hit this failure mode purely because of LLM
    omission — even though the authoritative BACKEND/MODEL_ID values were
    already on disk in `intermediate/runtime_directive.json` (which the
    wrapper writes BEFORE the Claude session starts).

    The fix synthesises a minimal schema-compliant `config_emission.json`
    from `runtime_directive.json` whenever the LLM did not emit one. The
    writer then renders `config.py` deterministically from the synthesized
    JSON, and the lint passes — same as if the LLM had emitted the IR
    itself.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_runtime_defaults_bound
from mellea_skills_compiler.compile.writer_renderer import (
    synthesize_config_emission_from_directive,
)
from mellea_skills_compiler.toolkit.logging import configure_logger


def _write_directive(intermediate: Path, *, backend: str, model_id: str) -> Path:
    """Helper: write a minimal runtime_directive.json to ``intermediate``."""
    intermediate.mkdir(parents=True, exist_ok=True)
    path = intermediate / "runtime_directive.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "backend": backend,
                "model_id": model_id,
                "source": "test fixture",
            }
        )
    )
    return path


class TestSynthesizeConfigEmissionFromDirective:
    """Pin the fallback semantics for the config_emission.json synthesizer."""

    def test_synthesizes_when_missing(self, tmp_path: Path) -> None:
        """No emission on disk + directive present → synthesise from directive."""
        intermediate = tmp_path / "intermediate"
        _write_directive(intermediate, backend="ollama", model_id="granite4.1:8b")

        result = synthesize_config_emission_from_directive(intermediate)

        assert result is not None
        constants = {c["name"]: c["value"] for c in result["constants"]}
        assert constants["BACKEND"] == "ollama"
        assert constants["MODEL_ID"] == "granite4.1:8b"

    def test_passthrough_when_present(self, tmp_path: Path) -> None:
        """Existing emission must be returned verbatim and never overwritten."""
        intermediate = tmp_path / "intermediate"
        _write_directive(intermediate, backend="ollama", model_id="granite4.1:8b")

        existing = {
            "constants": [
                {
                    "name": "PERSONA",
                    "value": "I am a custom emission preserved by the test.",
                    "type": "str",
                    "category": "C1",
                },
            ]
        }
        emission_path = intermediate / "config_emission.json"
        emission_path.write_text(json.dumps(existing))
        # Capture the on-disk text so we can prove the file is NOT rewritten.
        original_text = emission_path.read_text()

        result = synthesize_config_emission_from_directive(intermediate)

        assert result == existing
        assert emission_path.read_text() == original_text

    def test_returns_none_when_directive_missing(self, tmp_path: Path) -> None:
        """Neither file present → return None without raising."""
        intermediate = tmp_path / "intermediate"
        intermediate.mkdir()

        result = synthesize_config_emission_from_directive(intermediate)

        assert result is None
        assert not (intermediate / "config_emission.json").exists()

    def test_returns_none_when_intermediate_dir_missing(self, tmp_path: Path) -> None:
        """Intermediate dir does not exist at all → return None without raising.

        This branch matters because `render_writers` calls the synthesizer
        unconditionally; a pre-Step-3 invocation against an empty package
        must not blow up.
        """
        intermediate = tmp_path / "does-not-exist"

        result = synthesize_config_emission_from_directive(intermediate)

        assert result is None

    def test_returns_none_when_directive_malformed_json(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Directive exists but is not valid JSON → warn and return None.

        This is an explicitly tested edge case so we do not mask a corrupted
        directive by silently writing a bogus config_emission. The
        ``runtime-defaults-bound`` lint will then fail on the genuinely
        missing artefact rather than on a fabricated one.
        """
        intermediate = tmp_path / "intermediate"
        intermediate.mkdir()
        (intermediate / "runtime_directive.json").write_text("not-json{")
        configure_logger()

        with caplog.at_level(logging.WARNING):
            result = synthesize_config_emission_from_directive(intermediate)

        assert result is None
        assert not (intermediate / "config_emission.json").exists()
        assert any("unreadable" in r.message for r in caplog.records)

    def test_returns_none_when_directive_missing_required_fields(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Directive present but BACKEND/MODEL_ID absent → warn and return None."""
        intermediate = tmp_path / "intermediate"
        intermediate.mkdir()
        (intermediate / "runtime_directive.json").write_text(json.dumps({"foo": "bar"}))
        configure_logger()

        with caplog.at_level(logging.WARNING):
            result = synthesize_config_emission_from_directive(intermediate)

        assert result is None
        assert not (intermediate / "config_emission.json").exists()

    def test_writes_to_disk(self, tmp_path: Path) -> None:
        """After synthesis, the file exists on disk and reloads as valid JSON."""
        intermediate = tmp_path / "intermediate"
        _write_directive(intermediate, backend="openai", model_id="gpt-4o-mini")

        synthesize_config_emission_from_directive(intermediate)

        emission_path = intermediate / "config_emission.json"
        assert emission_path.exists()
        loaded = json.loads(emission_path.read_text())
        constants = {c["name"]: c["value"] for c in loaded["constants"]}
        assert constants["BACKEND"] == "openai"
        assert constants["MODEL_ID"] == "gpt-4o-mini"

    def test_log_message_on_synthesis(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Synthesizer logs an unambiguous message when it materialises an emission."""
        intermediate = tmp_path / "intermediate"
        _write_directive(intermediate, backend="ollama", model_id="granite4.1:8b")
        # The renderer's logger uses the project's `configure_logger`; ensure it
        # is materialised before caplog hooks the handler so messages surface.
        configure_logger()

        with caplog.at_level(logging.WARNING):
            synthesize_config_emission_from_directive(intermediate)

        joined = "\n".join(rec.message for rec in caplog.records)
        assert "Synthesized config_emission.json" in joined
        assert "runtime_directive.json" in joined

    def test_synthesized_emission_conforms_to_schema(self, tmp_path: Path) -> None:
        """The synthesized payload must validate against config_emission.schema.json."""
        intermediate = tmp_path / "intermediate"
        _write_directive(intermediate, backend="ollama", model_id="granite4.1:8b")
        result = synthesize_config_emission_from_directive(intermediate)

        # Locate the schema next to the writers — same resolution pattern the
        # writer-renderer uses.
        import mellea_skills_compiler

        compiler_pkg_dir = Path(mellea_skills_compiler.__file__).resolve().parent
        from mellea_skills_compiler.compile.mellea_skills import (
            _resolve_writers_repo_root,
        )

        repo_root = _resolve_writers_repo_root(compiler_pkg_dir)
        schema_path = repo_root / ".claude" / "schemas" / "config_emission.schema.json"
        schema = json.loads(schema_path.read_text())

        import jsonschema

        jsonschema.Draft202012Validator(schema).validate(result)


class TestSynthesizedPassesRuntimeDefaultsBoundLint:
    """End-to-end: synthesize → render config.py → lint passes."""

    def test_synthesized_passes_runtime_defaults_bound_lint(
        self, tmp_path: Path
    ) -> None:
        """The full deterministic path produces a config.py that the
        ``runtime-defaults-bound`` lint accepts.

        Setup mirrors what the wrapper does after a Claude session:
          1. The wrapper has already written runtime_directive.json.
          2. The LLM did NOT emit config_emission.json (the failure mode).
          3. The wrapper calls the synthesizer (now part of render_writers).
          4. The deterministic config_writer renders config.py.
          5. lint_runtime_defaults_bound passes because BACKEND/MODEL_ID in
             config.py match the directive verbatim.
        """
        package_dir = tmp_path / "demo_skill_mellea"
        intermediate = package_dir / "intermediate"
        _write_directive(intermediate, backend="ollama", model_id="granite4.1:8b")

        # Step 1: synthesise. After this, the writer is guaranteed an emission.
        result = synthesize_config_emission_from_directive(intermediate)
        assert result is not None

        # Step 2: render config.py via the deterministic writer (same code
        # path render_writers takes, exercised in isolation here to keep the
        # test independent of the full render_writers + WriterSpec machinery).
        import importlib.util
        import mellea_skills_compiler
        from mellea_skills_compiler.compile.mellea_skills import (
            _resolve_writers_repo_root,
        )

        compiler_pkg_dir = Path(mellea_skills_compiler.__file__).resolve().parent
        repo_root = _resolve_writers_repo_root(compiler_pkg_dir)
        writer_path = (
            repo_root / ".claude" / "melleafy" / "writers" / "config_writer.py"
        )
        spec = importlib.util.spec_from_file_location("config_writer", writer_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config_path = package_dir / "config.py"
        config_path.write_text(module.render(result))

        # Step 3: the lint must pass on the rendered config.py.
        lint_result = lint_runtime_defaults_bound(package_dir)
        assert lint_result.verdict == "pass", (
            f"lint_runtime_defaults_bound expected pass, got "
            f"{lint_result.verdict}: failures={lint_result.failures}"
        )
