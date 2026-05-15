"""Regression tests for emission schema validation in writer_renderer.

Background — the bug this pins:
    The writers (`config_writer.py`, `fixtures_writer.py`) dict-accessed the
    emission JSON without validating it against the `.claude/schemas/*.schema.json`
    contracts. When the LLM emitted JSON with the wrong key names (e.g.
    `fixture_id` instead of `id`, plus extra unexpected fields like
    `module_name`, `coverage`, `run_pipeline_params`) the writer crashed mid-
    rendering with a bare `KeyError: 'id'`. The compile then reported
    `writer.write() raised: 'id'` — useless for diagnosis — and Step 7's
    `fixtures-loader-contract` lint correctly hard-failed on the absent
    rendered output.

    The fix validates each emission JSON against its declared schema BEFORE
    invoking the writer. Schema violations surface as a precise
    `schema-invalid` status with a list of JSON paths and reasons, the writer
    is never invoked on bad input, and on-disk artifacts stay untouched
    rather than half-rendered.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.writer_renderer import (
    WriterSpec,
    render_writers,
)


# A minimal valid emission for the fixtures schema. Five fixtures meets the
# minItems=5 constraint and exercises happy-path validation.
_VALID_FIXTURES_EMISSION = {
    "fixtures": [
        {
            "id": f"case_{i}",
            "description": f"Fixture exercise number {i}",
            "inputs": {"x": i, "y": "value"},
        }
        for i in range(5)
    ]
}


# A valid config emission (the writer's own happy path).
_VALID_CONFIG_EMISSION = {
    "constants": [
        {"name": "AGENT_NAME", "value": "test-agent", "type": "str", "category": "C1"},
        {"name": "LOOP_BUDGET", "value": 3, "type": "int"},
    ]
}


# Stub writer modules — we don't care what the writer does for these tests;
# we only need the renderer to dispatch (or refuse to dispatch, on schema
# error) correctly. So we plant minimal valid writers in a tmp .claude tree.
_STUB_CONFIG_WRITER = textwrap.dedent(
    '''\
    """Stub config writer used in schema-validation tests."""
    def render(emission):
        return "# stub config\\n"
    '''
)

_STUB_FIXTURES_WRITER = textwrap.dedent(
    '''\
    """Stub fixtures writer used in schema-validation tests."""
    from pathlib import Path
    def write(emission, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        init = out_dir / "__init__.py"
        init.write_text("ALL_FIXTURES = []\\n")
        return [init]
    '''
)


def _make_synthetic_repo(tmp: Path) -> Path:
    """Build a `.claude/schemas/` and `.claude/melleafy/writers/` tree.

    Copies the real production schemas (so the tests validate against the
    same JSON as production) and plants stub writers so the renderer can
    dispatch without depending on the real writers' behaviour.
    """
    schemas_dir = tmp / ".claude" / "schemas"
    writers_dir = tmp / ".claude" / "melleafy" / "writers"
    schemas_dir.mkdir(parents=True)
    writers_dir.mkdir(parents=True)

    # Copy real schemas from the project root.
    project_root = Path(__file__).resolve().parents[3]
    src_schemas = project_root / ".claude" / "schemas"
    for name in ("config_emission.schema.json", "fixtures_emission.schema.json"):
        (schemas_dir / name).write_text((src_schemas / name).read_text())

    (writers_dir / "config_writer.py").write_text(_STUB_CONFIG_WRITER)
    (writers_dir / "fixtures_writer.py").write_text(_STUB_FIXTURES_WRITER)
    return tmp


def _build_specs(repo_root: Path) -> list[WriterSpec]:
    """Construct WriterSpecs that point at the synthetic repo's writers + schemas."""
    schemas_dir = repo_root / ".claude" / "schemas"
    writers_dir = repo_root / ".claude" / "melleafy" / "writers"
    return [
        WriterSpec(
            name="config.py",
            emission_relpath="intermediate/config_emission.json",
            output_relpath="config.py",
            writer_path=writers_dir / "config_writer.py",
            schema_path=schemas_dir / "config_emission.schema.json",
        ),
        WriterSpec(
            name="fixtures/",
            emission_relpath="intermediate/fixtures_emission.json",
            output_relpath="fixtures",
            writer_path=writers_dir / "fixtures_writer.py",
            output_kind="directory",
            schema_path=schemas_dir / "fixtures_emission.schema.json",
        ),
    ]


def _write_emission(pkg: Path, relpath: str, data: dict) -> None:
    p = pkg / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


# ─── TestSchemaValidationHappyPath ───


class TestSchemaValidationHappyPath:
    """Valid emission JSON should pass validation and reach the writer."""

    def test_valid_fixtures_emission_dispatches_to_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(
                pkg, "intermediate/fixtures_emission.json", _VALID_FIXTURES_EMISSION
            )

            specs = _build_specs(repo)
            # Only run the fixtures spec for this happy-path check.
            results = render_writers(pkg, [specs[1]], enforce=True)

            assert results[0].status == "match", (
                f"Valid fixtures emission should dispatch + render; got "
                f"{results[0].status}: {results[0].detail}"
            )
            assert (pkg / "fixtures" / "__init__.py").exists()

    def test_valid_config_emission_dispatches_to_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(
                pkg, "intermediate/config_emission.json", _VALID_CONFIG_EMISSION
            )

            specs = _build_specs(repo)
            results = render_writers(pkg, [specs[0]], enforce=True)

            assert results[0].status == "match"
            assert (pkg / "config.py").exists()


# ─── TestSchemaValidationCatchesViolations ───


class TestSchemaValidationCatchesViolations:
    """Schema-invalid emissions should fail at validation, not at the writer.

    The regression target: risk-and-issues-manager-scott-margetts emitted
    fixtures with `fixture_id` instead of `id`, plus extra `module_name`,
    `coverage`, `run_pipeline_params`, `format_version` fields. The writer
    crashed with `KeyError: 'id'`. With schema validation, this should now
    fail BEFORE the writer is invoked, with a precise error message naming
    every violation.
    """

    def test_missing_required_id_is_caught(self):
        """Fixture entry missing the required `id` field."""
        bad = {
            "fixtures": [
                {"description": "x", "inputs": {}},  # no `id`
                *_VALID_FIXTURES_EMISSION["fixtures"][1:],
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(pkg, "intermediate/fixtures_emission.json", bad)
            specs = _build_specs(repo)
            results = render_writers(pkg, [specs[1]], enforce=True)

            assert results[0].status == "schema-invalid", (
                f"Missing required 'id' must fail validation; got {results[0].status}"
            )
            assert "'id' is a required property" in (results[0].detail or "")
            assert not (pkg / "fixtures" / "__init__.py").exists(), (
                "Writer must NOT run when schema is invalid"
            )

    def test_renamed_key_fixture_id_is_caught(self):
        """The exact risk-and-issues-manager regression: `fixture_id` instead of `id`."""
        bad = {
            "fixtures": [
                {"fixture_id": "x", "description": "x", "inputs": {}},
                *_VALID_FIXTURES_EMISSION["fixtures"][1:],
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(pkg, "intermediate/fixtures_emission.json", bad)
            specs = _build_specs(repo)
            results = render_writers(pkg, [specs[1]], enforce=True)

            assert results[0].status == "schema-invalid"
            detail = results[0].detail or ""
            assert "'id' is a required property" in detail
            assert "fixture_id" in detail, (
                f"Error message should name the unexpected key; got: {detail}"
            )

    def test_extra_top_level_keys_are_caught(self):
        """LLM-emitted extras (`coverage`, `run_pipeline_params`, `format_version`)."""
        bad = {
            **_VALID_FIXTURES_EMISSION,
            "coverage": {},
            "run_pipeline_params": [],
            "format_version": "1.0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(pkg, "intermediate/fixtures_emission.json", bad)
            specs = _build_specs(repo)
            results = render_writers(pkg, [specs[1]], enforce=True)

            assert results[0].status == "schema-invalid"
            detail = results[0].detail or ""
            for unexpected in ("coverage", "run_pipeline_params", "format_version"):
                assert unexpected in detail, (
                    f"Error should name {unexpected!r}; got: {detail}"
                )

    def test_violation_message_includes_json_path(self):
        """Validation errors must be navigable: include $.fixtures[0].<...> paths."""
        bad = {
            "fixtures": [
                _VALID_FIXTURES_EMISSION["fixtures"][0],
                {"description": "missing id and inputs"},  # at $.fixtures[1]
                *_VALID_FIXTURES_EMISSION["fixtures"][2:],
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(pkg, "intermediate/fixtures_emission.json", bad)
            specs = _build_specs(repo)
            results = render_writers(pkg, [specs[1]], enforce=True)

            assert results[0].status == "schema-invalid"
            detail = results[0].detail or ""
            assert "$.fixtures[1]" in detail, (
                f"Error should include JSON path; got: {detail}"
            )

    def test_writer_not_invoked_when_schema_invalid(self):
        """Schema-invalid emission must NOT touch on-disk fixtures/ — pre-fix it
        was half-rendered after KeyError partway through."""
        bad = {"fixtures": [{"fixture_id": "x"}] * 5}  # totally wrong shape
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            # Plant a sentinel file in fixtures/ to detect if writer wiped it.
            (pkg / "fixtures").mkdir()
            (pkg / "fixtures" / "sentinel.txt").write_text("must not be wiped")
            _write_emission(pkg, "intermediate/fixtures_emission.json", bad)
            specs = _build_specs(repo)
            render_writers(pkg, [specs[1]], enforce=True)

            assert (pkg / "fixtures" / "sentinel.txt").exists(), (
                "Schema-invalid path must not wipe pre-existing fixtures/ "
                "contents — the dispatcher returns BEFORE the enforce-mode wipe."
            )


# ─── TestSchemaValidationGracefulFallback ───


class TestSchemaValidationGracefulFallback:
    """Edge cases where schema validation should not abort the renderer."""

    def test_no_schema_path_skips_validation(self):
        """A WriterSpec with schema_path=None preserves pre-validation behaviour."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            # Emit junk that would fail the real schema.
            _write_emission(
                pkg, "intermediate/fixtures_emission.json", {"fixtures": []}
            )
            writers_dir = repo / ".claude" / "melleafy" / "writers"
            spec_no_schema = WriterSpec(
                name="fixtures/",
                emission_relpath="intermediate/fixtures_emission.json",
                output_relpath="fixtures",
                writer_path=writers_dir / "fixtures_writer.py",
                output_kind="directory",
                schema_path=None,  # ← no schema → no validation
            )
            results = render_writers(pkg, [spec_no_schema], enforce=True)
            # Stub writer renders an empty fixtures/ — should succeed.
            assert results[0].status == "match"

    def test_missing_schema_file_skips_validation(self):
        """Pointing at a nonexistent schema falls through to writer dispatch."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_synthetic_repo(Path(tmp) / "repo")
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            _write_emission(
                pkg,
                "intermediate/fixtures_emission.json",
                _VALID_FIXTURES_EMISSION,
            )
            writers_dir = repo / ".claude" / "melleafy" / "writers"
            spec_missing_schema = WriterSpec(
                name="fixtures/",
                emission_relpath="intermediate/fixtures_emission.json",
                output_relpath="fixtures",
                writer_path=writers_dir / "fixtures_writer.py",
                output_kind="directory",
                schema_path=repo / ".claude" / "schemas" / "does_not_exist.schema.json",
            )
            results = render_writers(pkg, [spec_missing_schema], enforce=True)
            assert results[0].status == "match", (
                f"Missing schema file should be a soft skip; got {results[0].status}"
            )
