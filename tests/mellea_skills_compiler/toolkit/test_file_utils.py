"""Unit tests for mellea_skills_compiler.toolkit.file_utils module."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.toolkit.file_utils import (
    load_skill_pipeline,
    parse_spec_file,
)


class TestParseSkillMd:
    """Test cases for parse_spec_file function."""

    def test_parse_with_valid_frontmatter(self):
        """Test parsing a SKILL.md with valid YAML frontmatter."""
        content = """---
name: test-skill
description: A test skill
allowed-tools: Bash, Read, Write
---

This is the body of the skill."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)

            assert result["frontmatter"]["name"] == "test-skill"
            assert result["frontmatter"]["description"] == "A test skill"
            assert result["frontmatter"]["allowed-tools"] == ["Bash", "Read", "Write"]
            assert result["body"] == "This is the body of the skill."
            assert str(path) in result["path"]
        finally:
            path.unlink()

    def test_parse_without_frontmatter(self):
        """Test parsing a file without frontmatter."""
        content = "This is just markdown content without frontmatter."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)

            assert result["frontmatter"] == {}
            assert result["body"] == content
            assert str(path) in result["path"]
        finally:
            path.unlink()

    def test_parse_allowed_tools_as_comma_separated_string(self):
        """Test parsing allowed-tools as comma-separated string."""
        content = """---
allowed-tools: "Bash, Read, Write, Glob"
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            assert result["frontmatter"]["allowed-tools"] == [
                "Bash",
                "Read",
                "Write",
                "Glob",
            ]
        finally:
            path.unlink()

    def test_parse_allowed_tools_as_space_separated_string(self):
        """Test parsing allowed-tools as space-separated string without commas."""
        content = """---
allowed-tools: Bash Read Write
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            assert result["frontmatter"]["allowed-tools"] == ["Bash", "Read", "Write"]
        finally:
            path.unlink()

    def test_parse_allowed_tools_as_list(self):
        """Test parsing allowed-tools as YAML list."""
        content = """---
allowed-tools:
  - Bash
  - Read
  - Write
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            assert result["frontmatter"]["allowed-tools"] == ["Bash", "Read", "Write"]
        finally:
            path.unlink()

    def test_parse_openclaw_requires_bins(self):
        """Test that openclaw.requires.bins are added to allowed-tools."""
        content = """---
allowed-tools:
  - Bash
metadata:
  openclaw:
    requires:
      bins:
        - git
        - docker
      anyBins:
        - kubectl
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            tools = result["frontmatter"]["allowed-tools"]
            assert "Bash" in tools
            assert "git" in tools
            assert "docker" in tools
            assert "kubectl" in tools
        finally:
            path.unlink()

    def test_parse_openclaw_no_duplicates(self):
        """Test that openclaw tools don't create duplicates."""
        content = """---
allowed-tools:
  - git
  - docker
metadata:
  openclaw:
    requires:
      bins:
        - git
        - docker
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            tools = result["frontmatter"]["allowed-tools"]
            assert tools.count("git") == 1
            assert tools.count("docker") == 1
        finally:
            path.unlink()

    def test_parse_empty_frontmatter(self):
        """Test parsing with empty frontmatter section."""
        content = """Just a body."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            result = parse_spec_file(path)
            # Empty frontmatter is parsed as None by yaml.safe_load, which becomes {}
            assert result["frontmatter"] == {} or result["frontmatter"] is None
            assert result["body"] == "Just a body."
        finally:
            path.unlink()

    def test_parse_malformed_yaml(self):
        """Test that malformed YAML in frontmatter raises an error."""
        content = """---
name: test
invalid yaml: [unclosed bracket
---

Body content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(Exception):  # yaml.YAMLError
                parse_spec_file(path)
        finally:
            path.unlink()


def _materialise_pipeline_package(
    parent: Path,
    package_name: str,
    pipeline_source: str,
    melleafy_json: dict | None = None,
) -> Path:
    """Build a synthetic skill package on disk and return its directory."""
    pkg_dir = parent / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "pipeline.py").write_text(pipeline_source)
    if melleafy_json is not None:
        (pkg_dir / "melleafy.json").write_text(json.dumps(melleafy_json))
    return pkg_dir


class TestLoadSkillPipeline:
    """Tests for `load_skill_pipeline` entry-point discovery.

    Regression target: pre-fix behaviour picked the first `run_*` attribute
    in `dir()` ordering, which is alphabetical. A pipeline.py defining
    `run_phase_2_gap_analysis`, `run_phase_3_roadmap`, and `run_pipeline`
    would yield `run_phase_2_gap_analysis` — wrong. Smoke-check then called
    that phase function with entry-point kwargs and crashed with TypeError.

    Post-fix behaviour:
      Tier 1: read `melleafy.json:entry_signature`, parse the leading name,
              look it up on the imported module.
      Tier 2: fall back to `run_pipeline` by name (the documented canonical).
      Tier 3: alphabetical `run_*` scan (legacy compat).
    """

    def _restore_sys_modules(self, package_name: str) -> None:
        """Drop the imported module from sys.modules so re-imports reload cleanly."""
        for mod_name in list(sys.modules):
            if mod_name == package_name or mod_name.startswith(package_name + "."):
                sys.modules.pop(mod_name, None)

    def test_uses_melleafy_entry_signature_when_present(self):
        """Tier 1: melleafy.json:entry_signature names `run_pipeline` even when
        a phase helper sorts alphabetically first."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkg_name = "skill_tier1"
            pipeline = (
                "def run_phase_2_gap_analysis(**kwargs):\n"
                "    return ('phase_2', kwargs)\n"
                "def run_pipeline(**kwargs):\n"
                "    return ('pipeline', kwargs)\n"
            )
            pkg_dir = _materialise_pipeline_package(
                tmp_path,
                pkg_name,
                pipeline,
                melleafy_json={
                    "manifest_version": "1.1.0",
                    "entry_signature": "run_pipeline(session_id: str) -> dict",
                    "package_name": pkg_name,
                },
            )
            try:
                fn = load_skill_pipeline(pkg_dir)
                tag, _ = fn()
                assert tag == "pipeline", (
                    f"Expected run_pipeline; got function returning tag={tag!r}"
                )
                assert fn.__name__ == "run_pipeline"
            finally:
                self._restore_sys_modules(pkg_name)

    def test_falls_back_to_run_pipeline_by_name_when_manifest_absent(self):
        """Tier 2: no melleafy.json → loader prefers `run_pipeline` by convention."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkg_name = "skill_tier2"
            pipeline = (
                "def run_phase_2_gap_analysis(**kwargs):\n"
                "    return ('phase_2', kwargs)\n"
                "def run_pipeline(**kwargs):\n"
                "    return ('pipeline', kwargs)\n"
            )
            pkg_dir = _materialise_pipeline_package(tmp_path, pkg_name, pipeline)
            try:
                fn = load_skill_pipeline(pkg_dir)
                assert fn.__name__ == "run_pipeline", (
                    f"Without melleafy.json, run_pipeline should win over "
                    f"alphabetically-earlier run_phase_2; got fn.__name__="
                    f"{fn.__name__}"
                )
            finally:
                self._restore_sys_modules(pkg_name)

    def test_falls_back_to_alphabetical_when_no_run_pipeline(self):
        """Tier 3: legacy package without run_pipeline → first run_* alphabetically.

        This preserves backward compat with packages compiled before
        `run_pipeline` was canonicalised. The lint catches new packages;
        the loader keeps loading old ones.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkg_name = "skill_tier3"
            pipeline = (
                "def run_alpha(**kwargs):\n"
                "    return ('alpha', kwargs)\n"
                "def run_zeta(**kwargs):\n"
                "    return ('zeta', kwargs)\n"
            )
            pkg_dir = _materialise_pipeline_package(tmp_path, pkg_name, pipeline)
            try:
                fn = load_skill_pipeline(pkg_dir)
                assert fn.__name__ == "run_alpha", (
                    f"Tier 3 fallback should pick alphabetically-first run_*; "
                    f"got fn.__name__={fn.__name__}"
                )
            finally:
                self._restore_sys_modules(pkg_name)

    def test_malformed_melleafy_falls_through_to_run_pipeline(self):
        """A broken melleafy.json should not block discovery — Tier 2 takes over."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkg_name = "skill_malformed"
            pipeline = (
                "def run_pipeline(**kwargs):\n"
                "    return ('pipeline', kwargs)\n"
            )
            pkg_dir = _materialise_pipeline_package(tmp_path, pkg_name, pipeline)
            (pkg_dir / "melleafy.json").write_text("{not valid json")
            try:
                fn = load_skill_pipeline(pkg_dir)
                assert fn.__name__ == "run_pipeline"
            finally:
                self._restore_sys_modules(pkg_name)

    def test_manifest_entry_not_callable_falls_through(self):
        """If melleafy.json names a function that doesn't exist locally, fall through.

        Edge case: someone hand-edits the manifest to name a function that
        isn't actually defined in pipeline.py. The loader should NOT raise
        — it should silently fall through to Tier 2 (run_pipeline by name)
        so a coherent package still loads. The lint catches the manifest
        mismatch at compile time.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pkg_name = "skill_phantom"
            pipeline = (
                "def run_pipeline(**kwargs):\n"
                "    return ('pipeline', kwargs)\n"
            )
            pkg_dir = _materialise_pipeline_package(
                tmp_path,
                pkg_name,
                pipeline,
                melleafy_json={
                    "manifest_version": "1.1.0",
                    "entry_signature": "run_does_not_exist(**kwargs) -> dict",
                    "package_name": pkg_name,
                },
            )
            try:
                fn = load_skill_pipeline(pkg_dir)
                assert fn.__name__ == "run_pipeline", (
                    f"Phantom entry name should fall through to Tier 2; "
                    f"got fn.__name__={fn.__name__}"
                )
            finally:
                self._restore_sys_modules(pkg_name)
