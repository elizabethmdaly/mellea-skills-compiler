"""Unit tests for mellea_skills_compiler.compile.lints module.

Tests the three Step 7 structural lints (fixtures-loader-contract,
bundled-asset-path-resolution, runtime-defaults-bound) plus the run_lints
runner that emits intermediate/step_7_report.json.

All tests construct synthetic skill packages on disk in tempfile.TemporaryDirectory
contexts; no network, Ollama, or slash-command invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

from mellea_skills_compiler.compile.lints import (
    lint_bundled_asset_path_resolution,
    lint_complex_schema_needs_strategy_or_fallback,
    lint_fixture_signature_bound,
    lint_fixtures_loader_contract,
    lint_generative_call_passes_session,
    lint_generative_forbidden_params,
    lint_import_soundness,
    lint_instruct_has_description,
    lint_instruct_result_parse_before_access,
    lint_melleafy_json_consistency,
    lint_no_watsonx,
    lint_optional_extraction_guidance,
    lint_pipeline_entry_canonical,
    lint_prefix_persona,
    lint_run_pipeline_params_typed,
    lint_runtime_defaults_bound,
    lint_session_boundary,
    lint_stdlib_arity,
    lint_validator_soundness,
    run_lints,
)


def _make_package(root: Path, files: Dict[str, str]) -> Path:
    """Materialise a synthetic skill package under `root`.

    `files` maps relative paths (POSIX-style) to file contents. Parent
    directories are created as needed. Returns the package directory (root).
    """
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


# ─── TestFixturesLoaderContract ───


class TestFixturesLoaderContract:
    """Test cases for lint_fixtures_loader_contract."""

    def test_passes_with_all_fixtures_export(self):
        """fixtures/__init__.py exporting ALL_FIXTURES = [] should pass."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"fixtures/__init__.py": "ALL_FIXTURES = []\n"},
            )
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "pass", (
                f"Expected pass for ALL_FIXTURES export, got {result.verdict}: "
                f"{[f.message for f in result.failures]}"
            )
            assert result.failures == []
            assert result.files_checked == 1

    def test_passes_with_fixtures_export(self):
        """fixtures/__init__.py exporting FIXTURES = [] (alt shape) should pass."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"fixtures/__init__.py": "FIXTURES = []\n"},
            )
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "pass", (
                f"Expected pass for FIXTURES export, got {result.verdict}: "
                f"{[f.message for f in result.failures]}"
            )

    def test_passes_with_annotated_assignment(self):
        """fixtures/__init__.py with annotated ALL_FIXTURES: list = [] should pass."""
        content = (
            "from typing import Callable\n"
            "ALL_FIXTURES: list[Callable] = []\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"fixtures/__init__.py": content})
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "pass", (
                f"Annotated ALL_FIXTURES assignment should be recognised; got "
                f"{result.verdict} with failures {[f.message for f in result.failures]}"
            )

    def test_fails_with_input_only_module(self):
        """fixtures/__init__.py with only INPUT = {...} (crewai drift) should fail."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"fixtures/__init__.py": 'INPUT = {"foo": "bar"}\n'},
            )
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "fail", (
                "INPUT-only fixtures module should fail the loader contract"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "ALL_FIXTURES" in msg, (
                f"Failure message should name ALL_FIXTURES; got: {msg}"
            )
            assert "FIXTURES" in msg, (
                f"Failure message should also name FIXTURES; got: {msg}"
            )

    def test_fails_with_pytest_test_function(self):
        """fixtures/__init__.py with only def test_x(): ... (security-review drift)."""
        content = (
            "def test_something():\n"
            "    assert True\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"fixtures/__init__.py": content})
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "fail", (
                "pytest-style fixtures module without ALL_FIXTURES/FIXTURES should fail"
            )
            assert len(result.failures) == 1

    def test_fails_when_fixtures_init_missing(self):
        """fixtures/ exists with no __init__.py — verdict fail (mandatory file)."""
        # NOTE: previously this was asserted as `skipped`. That semantics was
        # wrong — fixtures/__init__.py is a mandatory output of fixtures_writer.py,
        # and treating its absence as a skip masked the out-of-tree writer-renderer
        # bug fixed alongside this test. See the docstring on
        # lint_fixtures_loader_contract for the full rationale.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / "fixtures").mkdir()
            # Some other file but no __init__.py
            (pkg / "fixtures" / "data.txt").write_text("not python\n")
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "fail", (
                f"Missing fixtures/__init__.py should fail (mandatory file), "
                f"got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "fixtures/__init__.py not found" in msg, (
                f"Failure message should name the missing mandatory file; got: {msg}"
            )
            assert "fixtures_writer" in msg, (
                f"Failure message should reference the writer responsible; got: {msg}"
            )

    def test_fails_with_syntax_error(self):
        """fixtures/__init__.py with a Python syntax error should fail with line info."""
        broken = "def broken(:\n    pass\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"fixtures/__init__.py": broken})
            result = lint_fixtures_loader_contract(pkg)
            assert result.verdict == "fail", (
                f"SyntaxError in fixtures/__init__.py should fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.line is not None, (
                "SyntaxError failure should carry a line number"
            )
            assert "SyntaxError" in failure.message


# ─── TestBundledAssetPathResolution ───


class TestBundledAssetPathResolution:
    """Test cases for lint_bundled_asset_path_resolution."""

    def test_passes_with_path_dunder_file_chain(self):
        """Path(__file__).parent / 'scripts' / 'bash' / 'x.sh' should pass."""
        content = (
            "from pathlib import Path\n"
            "p = Path(__file__).parent / 'scripts' / 'bash' / 'x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "pass", (
                f"Path(__file__).parent chain should pass; got failures: "
                f"{[f.message for f in result.failures]}"
            )

    def test_passes_with_path_dunder_file_compound_string(self):
        """Path(__file__).parent / 'scripts/bash/x.sh' (compound string) should pass."""
        content = (
            "from pathlib import Path\n"
            "p = Path(__file__).parent / 'scripts/bash/x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "pass", (
                f"Path(__file__).parent / compound bundled string should pass; "
                f"failures: {[f.message for f in result.failures]}"
            )

    def test_passes_with_no_bundled_paths(self):
        """tools.py with no scripts/references/assets references should pass."""
        content = (
            "from pathlib import Path\n"
            "p = Path('/tmp') / 'whatever.txt'\n"
            "q = 'no bundled dirs here'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "pass", (
                f"Module with no bundled-dir references should pass; "
                f"failures: {[f.message for f in result.failures]}"
            )

    def test_fails_with_repo_root_join_compound_string(self):
        """Path(repo_root) / 'scripts/bash/x.sh' should fail."""
        content = (
            "from pathlib import Path\n"
            "repo_root = '/tmp/whatever'\n"
            "p = Path(repo_root) / 'scripts/bash/x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "fail", (
                f"Path(repo_root) / 'scripts/...' should fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "Path(__file__).parent" in msg, (
                f"Failure message should advise Path(__file__).parent; got: {msg}"
            )

    def test_fails_with_repo_root_join_chain(self):
        """Path(repo_root) / 'scripts' / 'bash' should fail."""
        content = (
            "from pathlib import Path\n"
            "repo_root = '/tmp/whatever'\n"
            "p = Path(repo_root) / 'scripts' / 'bash'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "fail", (
                f"Path(repo_root) / 'scripts' / 'bash' chain should fail; "
                f"got {result.verdict}"
            )
            assert len(result.failures) >= 1

    def test_fails_with_os_path_join_non_file_rooted(self):
        """os.path.join(repo_root, 'scripts', 'bash') should fail."""
        content = (
            "import os\n"
            "repo_root = '/tmp/whatever'\n"
            "p = os.path.join(repo_root, 'scripts', 'bash')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "fail", (
                f"os.path.join(repo_root, 'scripts', ...) should fail; "
                f"got {result.verdict}"
            )
            assert len(result.failures) == 1

    def test_passes_with_os_path_join_dunder_file(self):
        """os.path.join(os.path.dirname(__file__), 'scripts') should pass."""
        content = (
            "import os\n"
            "p = os.path.join(os.path.dirname(__file__), 'scripts')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "pass", (
                f"os.path.join with __file__-rooted base should pass; "
                f"failures: {[f.message for f in result.failures]}"
            )

    def test_skips_fixtures_directory(self):
        """Files under fixtures/ are excluded — bad path resolution there shouldn't fail."""
        content = (
            "from pathlib import Path\n"
            "repo_root = '/tmp/whatever'\n"
            "p = Path(repo_root) / 'scripts/x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"fixtures/test_x.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "pass", (
                f"fixtures/ should be excluded from this lint's scope; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )
            assert result.files_checked == 0, (
                "fixtures/ files should not be counted as checked"
            )

    def test_dedups_failures_in_chain(self):
        """A chained Path(repo_root) / 'scripts' / 'bash' / 'x.sh' produces ONE failure."""
        content = (
            "from pathlib import Path\n"
            "repo_root = '/tmp/whatever'\n"
            "p = Path(repo_root) / 'scripts' / 'bash' / 'x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"tools.py": content})
            result = lint_bundled_asset_path_resolution(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1, (
                f"Chained BinOp(Div) should dedupe to a single failure; got "
                f"{len(result.failures)}: {[f.message for f in result.failures]}"
            )


# ─── TestRuntimeDefaultsBound ───


def _write_directive(pkg: Path, backend: str, model_id: str) -> None:
    """Write intermediate/runtime_directive.json to the synthetic package."""
    intermediate = pkg / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    (intermediate / "runtime_directive.json").write_text(
        json.dumps({"backend": backend, "model_id": model_id})
    )


class TestRuntimeDefaultsBound:
    """Test cases for lint_runtime_defaults_bound."""

    def test_passes_when_config_matches_directive(self):
        """config.py with matching BACKEND/MODEL_ID should pass."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "pass", (
                f"Matching config and directive should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )

    def test_fails_when_backend_diverges(self):
        """config.py BACKEND != directive backend → fail with actionable message."""
        config = (
            'BACKEND = "watsonx"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "fail", (
                f"Backend divergence should fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "watsonx" in msg, f"Message should name actual value 'watsonx': {msg}"
            assert "ollama" in msg, f"Message should name expected value 'ollama': {msg}"
            assert ".claude/data/runtime_defaults.json" in msg, (
                f"Message should reference .claude/data/runtime_defaults.json: {msg}"
            )

    def test_fails_when_model_id_diverges(self):
        """config.py MODEL_ID != directive model_id → fail."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite3.3:2b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "fail", (
                f"Model ID divergence should fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "granite3.3:2b" in msg
            assert "granite4.1:8b" in msg

    def test_fails_when_backend_constant_missing(self):
        """config.py without BACKEND → fail with a clear message naming BACKEND."""
        config = 'MODEL_ID = "granite4.1:8b"\n'
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "fail", (
                f"Missing BACKEND constant should fail; got {result.verdict}"
            )
            # At least one failure should mention BACKEND missing
            backend_msgs = [f.message for f in result.failures if "BACKEND" in f.message]
            assert backend_msgs, (
                f"Expected a failure naming missing BACKEND; got: "
                f"{[f.message for f in result.failures]}"
            )
            assert any("does not define" in m for m in backend_msgs), (
                f"Message should say config.py 'does not define' BACKEND; got: {backend_msgs}"
            )

    def test_skipped_when_directive_missing(self):
        """No intermediate/runtime_directive.json → skipped with 'older pipeline' reason."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "skipped", (
                f"Missing directive should skip; got {result.verdict}"
            )
            assert result.skipped_reason is not None
            assert "older pipeline" in result.skipped_reason, (
                f"Skip reason should mention 'older pipeline'; got: {result.skipped_reason}"
            )

    def test_fails_when_config_missing(self):
        """No config.py → fail (config.py is a mandatory writer output)."""
        # NOTE: previously this was asserted as `skipped`. That semantics was
        # wrong — config.py is a mandatory output of config_writer.py (Step 3,
        # ENFORCE mode), and treating its absence as a skip masked the
        # out-of-tree writer-renderer bug fixed alongside this test. See the
        # docstring on lint_runtime_defaults_bound for the full rationale.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "fail", (
                f"Missing config.py should fail (mandatory file), "
                f"got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "config.py not found" in msg, (
                f"Failure message should name the missing mandatory file; got: {msg}"
            )
            assert "config_writer" in msg, (
                f"Failure message should reference the writer responsible; got: {msg}"
            )

    def test_skipped_when_directive_malformed_json(self):
        """Malformed JSON in runtime_directive.json → skipped with parse-error reason."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            (pkg / "intermediate").mkdir(parents=True, exist_ok=True)
            (pkg / "intermediate" / "runtime_directive.json").write_text(
                "{not valid json"
            )
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "skipped", (
                f"Malformed directive JSON should skip; got {result.verdict}"
            )
            assert result.skipped_reason is not None
            assert (
                "could not read" in result.skipped_reason
                or "runtime_directive.json" in result.skipped_reason
            ), (
                f"Skip reason should cite the parse failure; got: {result.skipped_reason}"
            )

    def test_handles_annotated_assignment(self):
        """config.py using BACKEND: Final[str] = 'ollama' (annotated) should validate."""
        config = (
            "from typing import Final\n"
            'BACKEND: Final[str] = "ollama"\n'
            'MODEL_ID: Final[str] = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "pass", (
                f"Annotated BACKEND/MODEL_ID assignments should be recognised; "
                f"got {result.verdict} with failures "
                f"{[f.message for f in result.failures]}"
            )

    def test_handles_plain_assignment(self):
        """config.py using BACKEND = 'ollama' (no annotation) should validate."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "pass", (
                f"Plain BACKEND/MODEL_ID assignments should be recognised; "
                f"got {result.verdict} with failures "
                f"{[f.message for f in result.failures]}"
            )


# ─── TestInstructHasDescription ───


class TestInstructHasDescription:
    """Test cases for lint_instruct_has_description (KB12).

    Mellea 0.5.0 `MelleaSession.instruct(description, *, ...)` requires a
    description as the first positional-or-keyword argument. Calls that
    omit it crash at runtime with `TypeError`. The lint hard-fails on any
    `m.instruct(...)` callsite missing both a positional first arg and a
    `description=` kwarg.
    """

    def test_passes_with_positional_description(self):
        """m.instruct("desc", format=Model, ...) — positional description passes."""
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import JurisdictionRoute\n"
            "\n"
            "def run_pipeline():\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        route_thunk = m.instruct(\n"
            "            'Classify the query.',\n"
            "            format=JurisdictionRoute,\n"
            "        )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "pass", (
                f"Positional description should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )
            assert result.files_checked == 1

    def test_passes_with_description_kwarg(self):
        """m.instruct(description="desc", format=Model, ...) — keyword form passes."""
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import JurisdictionRoute\n"
            "\n"
            "def run_pipeline():\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        route_thunk = m.instruct(\n"
            "            description='Classify the query.',\n"
            "            format=JurisdictionRoute,\n"
            "        )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "pass", (
                f"description= keyword should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )

    def test_fails_with_format_only_call(self):
        """m.instruct(format=Model, ...) — no description — should fail (KB12 / nis2-navigator)."""
        pipeline = (
            "from mellea import start_session\n"
            "from mellea.backends.model_options import ModelOption\n"
            "from .schemas import JurisdictionRoute\n"
            "from .config import AGENT_DESCRIPTION\n"
            "\n"
            "def run_pipeline():\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        route_thunk = m.instruct(\n"
            "            format=JurisdictionRoute,\n"
            "            grounding_context={'query': 'x'},\n"
            "            model_options={ModelOption.SYSTEM_PROMPT: AGENT_DESCRIPTION},\n"
            "        )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "fail", (
                f"Format-only call must fail (would crash at runtime); "
                f"got {result.verdict}"
            )
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.file == "pipeline.py"
            assert failure.line is not None, (
                "Failure should report the call's line number"
            )
            assert "description" in failure.message.lower()
            assert "TypeError" in failure.message or "missing" in failure.message

    def test_fails_with_bare_call_no_args(self):
        """m.instruct() — no args at all — should fail (sanity check)."""
        pipeline = (
            "from mellea import start_session\n"
            "\n"
            "def run_pipeline():\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        m.instruct()\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1

    def test_passes_when_no_pipeline_files_present(self):
        """A package with no pipeline.py/slots.py/constrained_slots.py should pass trivially."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "print('hi')\n"})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "pass"
            assert result.files_checked == 0

    def test_checks_slots_and_constrained_slots(self):
        """Lint should walk slots.py and constrained_slots.py too — multi-file failures."""
        bad_call = (
            "from mellea import start_session\n"
            "from .schemas import S\n"
            "\n"
            "def f():\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        m.instruct(format=S)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "slots.py": bad_call,
                    "constrained_slots.py": bad_call,
                },
            )
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "fail", (
                f"Expected fail across both files; got {result.verdict}"
            )
            assert len(result.failures) == 2
            files = sorted(f.file for f in result.failures)
            assert files == ["constrained_slots.py", "slots.py"]

    def test_skips_unparseable_files(self):
        """Pipeline.py with a SyntaxError should not crash this lint (parseable owns reporting)."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": "def broken(:\n    pass\n"},
            )
            result = lint_instruct_has_description(pkg)
            # Skipping unparseable files means no failures recorded here.
            assert result.verdict == "pass"

    def test_ignores_non_m_instruct_calls(self):
        """A call to `other.instruct(format=X)` or `m.other(format=X)` is not in scope."""
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import S\n"
            "\n"
            "def f():\n"
            "    other = None\n"
            "    other.instruct(format=S)\n"  # not m.instruct — ignore
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        m.transform(format=S)\n"  # not .instruct — ignore
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_has_description(pkg)
            assert result.verdict == "pass", (
                f"Non-m.instruct calls should be ignored; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )


# ─── TestPipelineEntryCanonical ───


def _melleafy_json(entry_signature: str) -> str:
    """Build a minimal melleafy.json with a given entry_signature."""
    return json.dumps(
        {
            "manifest_version": "1.1.0",
            "entry_signature": entry_signature,
            "package_name": "test_pkg",
        }
    )


class TestPipelineEntryCanonical:
    """Test cases for lint_pipeline_entry_canonical.

    Regression target: `nis2-navigator` pipeline.py defined `run_pipeline`,
    `run_phase_2_gap_analysis`, and `run_phase_3_roadmap` as public top-level
    functions. The smoke-check loader used `dir()` ordering and picked
    `run_phase_2_gap_analysis` (alphabetically first), crashing at fixture
    run with a TypeError. This lint enforces that `run_pipeline` exists and
    the manifest agrees.
    """

    def test_passes_with_run_pipeline_only(self):
        """A pipeline.py defining only `run_pipeline` should pass."""
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "pass", (
                f"Expected pass; got {result.verdict} with failures "
                f"{[f.message for f in result.failures]}"
            )
            assert result.files_checked == 1

    def test_passes_with_run_pipeline_plus_public_helpers(self):
        """Multiple public run_* functions are allowed as long as run_pipeline exists.

        The loader uses melleafy.json:entry_signature as authoritative, so
        naming collisions among public helpers are not a contract violation.
        This lint only requires the canonical entry to exist.
        """
        pipeline = (
            "def run_phase_2(session_id: str) -> dict:\n    return {}\n"
            "def run_pipeline(session_id: str) -> dict:\n    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "pass", (
                f"Helper functions named run_* alongside run_pipeline should "
                f"be allowed; got {result.verdict} with failures "
                f"{[f.message for f in result.failures]}"
            )

    def test_fails_without_run_pipeline(self):
        """A pipeline.py with no `run_pipeline` def must fail."""
        pipeline = (
            "def run_phase_2(session_id: str) -> dict:\n    return {}\n"
            "def run_phase_3(session_id: str) -> dict:\n    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "fail", (
                f"Missing run_pipeline must fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "run_pipeline" in msg
            assert "run_phase_2" in msg and "run_phase_3" in msg, (
                f"Failure should enumerate the found run_* helpers; got: {msg}"
            )

    def test_passes_with_matching_melleafy_entry_signature(self):
        """pipeline.py + melleafy.json:entry_signature naming run_pipeline should pass."""
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        manifest = _melleafy_json(
            "run_pipeline(x: str) -> str"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "melleafy.json": manifest},
            )
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "pass", (
                f"Matching entry_signature should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )

    def test_fails_with_mismatched_melleafy_entry_signature(self):
        """melleafy.json naming a non-canonical function must fail."""
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        manifest = _melleafy_json("run_phase_2(x: str) -> str")
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "melleafy.json": manifest},
            )
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "fail", (
                f"Mismatched entry_signature must fail; got {result.verdict}"
            )
            assert any(
                "melleafy.json" in f.file and "run_phase_2" in f.message
                for f in result.failures
            ), (
                f"Failure should name melleafy.json and the bad entry name; got: "
                f"{[(f.file, f.message) for f in result.failures]}"
            )

    def test_passes_when_melleafy_unreadable(self):
        """A malformed melleafy.json should not block pass when pipeline.py is correct.

        The loader falls back gracefully on a missing/malformed manifest; the
        lint mirrors that by silently ignoring the manifest-side check.
        """
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "melleafy.json": "{not valid json"},
            )
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "pass", (
                f"Malformed melleafy.json should not fail this lint when "
                f"pipeline.py has run_pipeline; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )

    def test_skipped_when_pipeline_absent(self):
        """No pipeline.py → skipped (other lints own that report)."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "print('hi')\n"})
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "skipped"
            assert result.skipped_reason is not None

    def test_fails_on_pipeline_syntax_error(self):
        """A SyntaxError in pipeline.py should fail with line info."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"pipeline.py": "def broken(:\n    pass\n"}
            )
            result = lint_pipeline_entry_canonical(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.line is not None
            assert "SyntaxError" in failure.message


# ─── TestFixtureSignatureBound ───


class TestFixtureSignatureBound:
    """Test cases for lint_fixture_signature_bound.

    Regression target: same `nis2-navigator` smoke-check crash —
    `run_phase_2_gap_analysis(**inputs)` failed because `inputs` contained
    `regulatory_updates`, which the phase function didn't declare. Even with
    the loader fix routing to `run_pipeline`, fixtures must still agree on
    the entry parameter set or `run_pipeline(**inputs)` will TypeError.
    """

    def test_passes_when_inputs_match_params(self):
        """Fixture inputs keys equal to run_pipeline params → pass."""
        pipeline = (
            "def run_pipeline(session_id: str, regulatory_updates: str = '') -> dict:\n"
            "    return {}\n"
        )
        fixture = (
            "def make_x():\n"
            "    inputs = {'session_id': 's1', 'regulatory_updates': 'x'}\n"
            "    return inputs, 'x', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "fixtures/x.py": fixture,
                    "fixtures/__init__.py": "ALL_FIXTURES = []\n",
                },
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "pass", (
                f"Matching inputs should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )
            assert result.files_checked == 1

    def test_passes_when_inputs_strict_subset(self):
        """Fixture using only required params (omitting defaulted ones) → pass."""
        pipeline = (
            "def run_pipeline(session_id: str, regulatory_updates: str = '') -> dict:\n"
            "    return {}\n"
        )
        fixture = (
            "def make_x():\n"
            "    inputs = {'session_id': 's1'}\n"
            "    return inputs, 'x', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "fixtures/x.py": fixture},
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "pass"

    def test_fails_when_extra_key(self):
        """Fixture with a key not in run_pipeline params → fail naming the key.

        This is the exact nis2-navigator failure shape, modulo the loader
        bug also being fixed. Here the regression assertion is that even
        with the loader picking the right function, an out-of-sync fixture
        is caught at compile time.
        """
        pipeline = "def run_pipeline(session_id: str) -> dict:\n    return {}\n"
        fixture = (
            "def make_x():\n"
            "    inputs = {'session_id': 's1', 'regulatory_updates': 'oops'}\n"
            "    return inputs, 'x', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "fixtures/x.py": fixture},
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "fail", (
                f"Extra fixture key must fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.file == "fixtures/x.py"
            assert "regulatory_updates" in failure.message
            assert "session_id" in failure.message, (
                f"Failure should enumerate valid params; got: {failure.message}"
            )

    def test_passes_with_var_keyword_pipeline(self):
        """run_pipeline(**kwargs) accepts anything — no fixture keys can fail."""
        pipeline = "def run_pipeline(**kwargs) -> dict:\n    return {}\n"
        fixture = (
            "def make_x():\n"
            "    inputs = {'foo': 1, 'bar': 2, 'whatever': 3}\n"
            "    return inputs, 'x', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "fixtures/x.py": fixture},
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "pass"

    def test_skipped_when_pipeline_absent(self):
        """No pipeline.py → skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"fixtures/__init__.py": "ALL_FIXTURES = []\n"}
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "skipped"

    def test_skipped_when_run_pipeline_absent(self):
        """pipeline.py exists but has no run_pipeline → skipped.

        pipeline-entry-canonical owns reporting this; we shouldn't double-fail.
        """
        pipeline = "def run_phase_2(session_id: str) -> dict:\n    return {}\n"
        fixture = (
            "def make_x():\n"
            "    inputs = {'wrong_key': 's1'}\n"
            "    return inputs, 'x', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "fixtures/x.py": fixture},
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "skipped"
            assert "run_pipeline" in (result.skipped_reason or "")

    def test_skipped_when_fixtures_absent(self):
        """pipeline.py with run_pipeline but no fixtures/ → skipped."""
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "skipped"

    def test_skips_fixture_with_dynamic_inputs(self):
        """Inputs built dynamically (not a dict literal) → silently skip that fixture."""
        pipeline = "def run_pipeline(x: str) -> str:\n    return x\n"
        fixture = (
            "def make_x():\n"
            "    inputs = build_inputs()\n"  # non-Dict assignment
            "    return inputs, 'x', 'desc'\n"
            "def build_inputs():\n"
            "    return {'x': 1}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "fixtures/x.py": fixture},
            )
            result = lint_fixture_signature_bound(pkg)
            # Cannot statically verify — passes (no failure raised).
            assert result.verdict == "pass"
            assert result.failures == []

    def test_reports_multiple_fixtures_independently(self):
        """Multiple fixtures: only the bad one fails; good ones don't pollute."""
        pipeline = "def run_pipeline(session_id: str) -> dict:\n    return {}\n"
        good = (
            "def make_good():\n"
            "    inputs = {'session_id': 's1'}\n"
            "    return inputs, 'good', 'desc'\n"
        )
        bad = (
            "def make_bad():\n"
            "    inputs = {'session_id': 's2', 'extra': 'oops'}\n"
            "    return inputs, 'bad', 'desc'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "fixtures/good.py": good,
                    "fixtures/bad.py": bad,
                },
            )
            result = lint_fixture_signature_bound(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.file == "fixtures/bad.py", (
                f"Only the bad fixture should be reported; got: {failure.file}"
            )
            assert "extra" in failure.message


# ─── TestGenerativeForbiddenParams ───


class TestGenerativeForbiddenParams:
    """Test cases for lint_generative_forbidden_params (KB6 definition-side).

    Regression target: risk-and-issues-manager-scott-margetts compiled with
    6 `@generative` functions each declaring `m` as their first parameter.
    The compiled package crashed at import with:
        ValueError: cannot create a generative stub with disallowed
        parameter names: ['m']
    Every Step 7 LLM-side lint reported pass; this lint now catches it
    deterministically at compile time.
    """

    def test_passes_with_clean_signature(self):
        """@generative def f(text: str) -> str — no forbidden params, passes."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def classify(text: str) -> str:\n"
            "    \"\"\"Set result to one of: a, b, c.\"\"\"\n"
            "    ...\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"slots.py": slots})
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "pass", (
                f"Clean signature should pass; got {result.verdict} with "
                f"failures {[f.message for f in result.failures]}"
            )

    def test_fails_with_m_parameter(self):
        """@generative def f(m, text) — `m` is forbidden, must fail."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def classify(m, text: str) -> str:\n"
            "    ...\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"slots.py": slots})
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.file == "slots.py"
            assert failure.line is not None
            assert "`m`" in failure.message or "['m']" in failure.message or "m." in failure.message

    def test_fails_with_multiple_forbidden_params(self):
        """Multiple forbidden names in one signature all appear in the failure."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def slot(m, context, backend, text: str) -> str:\n"
            "    ...\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"slots.py": slots})
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "m" in msg and "context" in msg and "backend" in msg

    def test_walks_multiple_files(self):
        """Failures across slots.py + constrained_slots.py both reported."""
        bad = (
            "from mellea import generative\n"
            "@generative\n"
            "def slot(m, text: str) -> str:\n"
            "    ...\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"slots.py": bad, "constrained_slots.py": bad},
            )
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 2
            files = sorted(f.file for f in result.failures)
            assert files == ["constrained_slots.py", "slots.py"]

    def test_ignores_non_generative_functions(self):
        """A regular function with `m` as a parameter is NOT in scope."""
        slots = (
            "def helper(m, text: str) -> str:\n"
            "    return text\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"slots.py": slots})
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "pass"

    def test_uses_grounded_forbidden_list_when_available(self):
        """If intermediate/mellea_api_ref.json provides forbidden_param_names,
        the lint uses that list rather than the static fallback."""
        api_ref = json.dumps(
            {
                "format_version": "1.0",
                "mellea_version": "0.5.0",
                "grounding_unavailable": False,
                "modules": {},
                "forbidden_param_names": ["forbidden_custom", "m"],
                "compatibility": [],
            }
        )
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def slot(forbidden_custom: str) -> str:\n"
            "    ...\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"slots.py": slots, "intermediate/mellea_api_ref.json": api_ref},
            )
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "fail", (
                f"Custom forbidden name should be caught via grounding; got "
                f"{result.verdict} with failures {[f.message for f in result.failures]}"
            )
            assert "forbidden_custom" in result.failures[0].message

    def test_passes_when_no_pipeline_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "print(1)\n"})
            result = lint_generative_forbidden_params(pkg)
            assert result.verdict == "pass"


# ─── TestGenerativeCallPassesSession ───


class TestGenerativeCallPassesSession:
    """Test cases for lint_generative_call_passes_session (KB6 callsite half).

    Regression target: eu-ai-act-classification-werner-plutat had:
        with start_session(BACKEND, MODEL_ID):
            raw_profile = extract_ai_system_profile_raw(system_description=...)
    No `as m` binding; no positional m passed to the @generative call.
    Smoke-check crashed with TypeError at fixture-run time.
    """

    def test_passes_with_positional_session(self):
        """slot(m, ...) — positional first arg, valid."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def classify(text: str) -> str:\n"
            "    ...\n"
        )
        pipeline = (
            "from mellea import start_session\n"
            "from .slots import classify\n"
            "def run_pipeline(x: str) -> str:\n"
            "    with start_session('ollama', 'granite4.1:8b') as m:\n"
            "        return classify(m, text=x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"slots.py": slots, "pipeline.py": pipeline}
            )
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "pass", (
                f"Positional m should pass; got {result.verdict} with "
                f"failures {[f.message for f in result.failures]}"
            )

    def test_passes_with_m_keyword(self):
        """slot(m=session, ...) — keyword m=, valid."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def classify(text: str) -> str:\n"
            "    ...\n"
        )
        pipeline = (
            "from mellea import start_session\n"
            "from .slots import classify\n"
            "def f(x):\n"
            "    with start_session('ollama', 'm') as session:\n"
            "        return classify(m=session, text=x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"slots.py": slots, "pipeline.py": pipeline}
            )
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "pass"

    def test_passes_with_context_and_backend_kwargs(self):
        """slot(context=..., backend=...) — both required kwargs, valid."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def classify(text: str) -> str:\n"
            "    ...\n"
        )
        pipeline = (
            "from .slots import classify\n"
            "def f(x):\n"
            "    return classify(context=ctx, backend=be, text=x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"slots.py": slots, "pipeline.py": pipeline}
            )
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "pass"

    def test_fails_with_no_session_passed(self):
        """slot(text=x) — no positional m, no m=, no context/backend pair → fail.

        Exact eu-ai-act-classification regression.
        """
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def extract_profile(system_description: str) -> str:\n"
            "    ...\n"
        )
        pipeline = (
            "from mellea import start_session\n"
            "from .slots import extract_profile\n"
            "def run_pipeline(x: str) -> str:\n"
            "    with start_session('ollama', 'granite4.1:8b'):\n"
            "        return extract_profile(system_description=x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"slots.py": slots, "pipeline.py": pipeline}
            )
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            failure = result.failures[0]
            assert failure.file == "pipeline.py"
            assert "extract_profile" in failure.message
            assert "MelleaSession" in failure.message or "m=" in failure.message

    def test_fails_when_only_context_kwarg_present(self):
        """`context=` alone is not enough — both context AND backend required."""
        slots = (
            "from mellea import generative\n"
            "@generative\n"
            "def slot(text: str) -> str:\n"
            "    ...\n"
        )
        pipeline = (
            "from .slots import slot\n"
            "def f(x):\n"
            "    return slot(context=ctx, text=x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"slots.py": slots, "pipeline.py": pipeline}
            )
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "fail"

    def test_ignores_calls_to_non_generative_functions(self):
        """A call to a regular helper that happens to share a name is not in scope."""
        # No slots.py — so no functions are @generative; ANY call is ignored.
        pipeline = (
            "def f(x):\n"
            "    return some_helper(text=x)  # not @generative, ignored\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "pass"

    def test_trivial_pass_when_no_generative_defs(self):
        """No @generative defs in slots.py → no calls to check, trivially passes."""
        slots = "def helper(x):\n    return x\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"slots.py": slots})
            result = lint_generative_call_passes_session(pkg)
            assert result.verdict == "pass"


# ─── TestValidatorSoundness ───


class TestValidatorSoundness:
    """Test cases for lint_validator_soundness (KB3 sub-check A + KB4 sub-check B)."""

    def test_passes_with_simple_validate_wrapper(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req, simple_validate\n"
            "LENGTH_REQ = req(\n"
            "    'Keep response under 100 words',\n"
            "    validation_fn=simple_validate(lambda x: len(x.split()) < 100),\n"
            ")\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass", (
                f"simple_validate wrapper should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )
            assert result.files_checked == 1

    def test_passes_with_named_context_function(self):
        requirements_py = (
            "from mellea import Context, ValidationResult\n"
            "from mellea.stdlib.requirements import req\n"
            "\n"
            "def _check_length(ctx: Context, result) -> ValidationResult:\n"
            "    return ValidationResult(len(str(result).split()) < 100)\n"
            "\n"
            "LENGTH_REQ = req('Keep response under 100 words', validation_fn=_check_length)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass"

    def test_fails_with_raw_lambda(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req\n"
            "LENGTH_REQ = req(\n"
            "    'Keep response under 100 words',\n"
            "    validation_fn=lambda x: len(x.split()) < 100,\n"
            ")\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            assert "simple_validate" in result.failures[0].message.lower()
            assert "lambda" in result.failures[0].message.lower()

    def test_fails_with_named_function_missing_context_annotation(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req\n"
            "\n"
            "def _check(result):\n"
            "    return result is not None\n"
            "\n"
            "REQ = req('Non-empty', validation_fn=_check)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert "_check" in result.failures[0].message
            assert "Context" in result.failures[0].message

    def test_fails_with_named_function_wrong_annotation(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req\n"
            "\n"
            "def _check(value: str, result) -> bool:\n"
            "    return bool(value)\n"
            "\n"
            "REQ = req('Non-empty', validation_fn=_check)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert "_check" in result.failures[0].message

    def test_fails_with_vacuous_lambda_true(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req, simple_validate\n"
            "ALWAYS_PASS = req(\n"
            "    'Always passes',\n"
            "    validation_fn=simple_validate(lambda x: True),\n"
            ")\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert "vacuous" in result.failures[0].message.lower()

    def test_passes_with_simple_validate_real_predicate(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req, simple_validate\n"
            "LENGTH_REQ = req(\n"
            "    'Bound length',\n"
            "    validation_fn=simple_validate(lambda x: len(str(x)) > 0),\n"
            ")\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass"

    def test_walks_multiple_files(self):
        bad = (
            "from mellea.stdlib.requirements import req\n"
            "REQ = req('x', validation_fn=lambda x: True)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "requirements.py": bad,
                    "slots.py": bad,
                    "constrained_slots.py": bad,
                    "pipeline.py": bad,
                },
            )
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 4

    def test_passes_when_no_validator_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "print('hi')\n"})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass"
            assert result.files_checked == 0

    def test_skips_unparseable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"requirements.py": "def broken(:\n    pass\n"},
            )
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass"

    def test_ignores_imported_named_functions(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req\n"
            "from .validators import external_check\n"
            "REQ = req('x', validation_fn=external_check)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "pass"

    def test_collects_both_subchecks_in_one_file(self):
        requirements_py = (
            "from mellea.stdlib.requirements import req, simple_validate\n"
            "R1 = req('a', validation_fn=lambda x: x is not None)\n"
            "R2 = req('b', validation_fn=simple_validate(lambda y: True))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": requirements_py})
            result = lint_validator_soundness(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 2


# ─── TestSessionBoundary ───


class TestSessionBoundary:
    """Test cases for lint_session_boundary (KB5)."""

    def test_passes_with_single_format_type(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import TriageVerdict\n"
            "def run_pipeline(t: str) -> dict:\n"
            "    with start_session('ollama', 'm') as m:\n"
            "        v1 = m.instruct('Triage', format=TriageVerdict)\n"
            "        v2 = m.instruct('Refine', format=TriageVerdict)\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_session_boundary(pkg).verdict == "pass"

    def test_fails_with_two_format_types_in_one_session(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import A, B\n"
            "def run_pipeline(t: str) -> dict:\n"
            "    with start_session('o','m') as m:\n"
            "        x = m.instruct('one', format=A)\n"
            "        y = m.instruct('two', format=B)\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_session_boundary(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 1
            assert "A" in result.failures[0].message
            assert "B" in result.failures[0].message

    def test_passes_with_two_sessions_each_one_format(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import A, B\n"
            "def f(x):\n"
            "    with start_session('o','m') as m:\n"
            "        a = m.instruct('1', format=A)\n"
            "    with start_session('o','m') as m:\n"
            "        b = m.instruct('2', format=B)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_session_boundary(pkg).verdict == "pass"

    def test_passes_with_no_format_kwarg(self):
        pipeline = (
            "from mellea import start_session\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct('Answer')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_session_boundary(pkg).verdict == "pass"

    def test_fails_with_three_distinct_format_types(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import A, B, C\n"
            "def f(x):\n"
            "    with start_session('o','m') as m:\n"
            "        a = m.instruct('1', format=A)\n"
            "        b = m.instruct('2', format=B)\n"
            "        c = m.instruct('3', format=C)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_session_boundary(pkg)
            assert result.verdict == "fail"
            msg = result.failures[0].message
            assert "A" in msg and "B" in msg and "C" in msg

    def test_checks_slots_and_constrained_slots(self):
        bad = (
            "from mellea import start_session\n"
            "from .schemas import A, B\n"
            "def helper():\n"
            "    with start_session('o','m') as m:\n"
            "        m.instruct('x', format=A)\n"
            "        m.instruct('y', format=B)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"slots.py": bad, "constrained_slots.py": bad},
            )
            result = lint_session_boundary(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 2

    def test_skips_unparseable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": "def broken(:\n    pass\n"})
            assert lint_session_boundary(pkg).verdict == "pass"

    def test_passes_when_no_pipeline_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "x = 1\n"})
            result = lint_session_boundary(pkg)
            assert result.verdict == "pass"
            assert result.files_checked == 0


# ─── TestPrefixPersona ───


class TestPrefixPersona:
    """Test cases for lint_prefix_persona (KB7)."""

    def test_passes_with_string_literal_prefix(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Result\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            '        r = m.instruct(q, format=Result, prefix=\'{"value":\')\n'
            "    return r\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_prefix_persona(pkg).verdict == "pass"

    def test_fails_with_config_constant_via_assignment(self):
        config = 'PREFIX_TEXT = "You are a helpful agent."\n'
        pipeline = (
            "from mellea import start_session\n"
            "from .config import PREFIX_TEXT\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct(q, prefix=PREFIX_TEXT)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"config.py": config, "pipeline.py": pipeline}
            )
            result = lint_prefix_persona(pkg)
            assert result.verdict == "fail"
            assert "PREFIX_TEXT" in result.failures[0].message
            assert "SYSTEM_PROMPT" in result.failures[0].message

    def test_passes_with_system_prompt_pattern(self):
        config = 'PREFIX_TEXT = "persona"\n'
        pipeline = (
            "from mellea import start_session\n"
            "from mellea.backends.model_options import ModelOption\n"
            "from .config import PREFIX_TEXT\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct(q, model_options={ModelOption.SYSTEM_PROMPT: PREFIX_TEXT})\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"config.py": config, "pipeline.py": pipeline}
            )
            assert lint_prefix_persona(pkg).verdict == "pass"

    def test_fails_with_config_import_no_config_py(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .config import PERSONA\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct(q, prefix=PERSONA)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_prefix_persona(pkg)
            assert result.verdict == "fail"
            assert "PERSONA" in result.failures[0].message

    def test_fails_with_bare_name(self):
        pipeline = (
            "from mellea import start_session\n"
            "def run_pipeline(q):\n"
            "    PREFIX_TEXT = 'persona'\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct(q, prefix=PREFIX_TEXT)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_prefix_persona(pkg).verdict == "fail"

    def test_passes_with_no_prefix_kwarg(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import R\n"
            "def run_pipeline(q):\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct(q, format=R)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_prefix_persona(pkg).verdict == "pass"


# ─── TestNoWatsonx ───


class TestNoWatsonx:
    """Test cases for lint_no_watsonx (KB8)."""

    def test_passes_with_supported_backend(self):
        config = 'BACKEND = "ollama"\n'
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            assert lint_no_watsonx(pkg).verdict == "pass"

    def test_fails_with_watsonx_in_config(self):
        config = 'BACKEND = "watsonx"\n'
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            result = lint_no_watsonx(pkg)
            assert result.verdict == "fail"
            assert result.failures[0].file == "config.py"
            assert result.failures[0].line == 1

    def test_fails_with_watsonx_inline_in_pipeline(self):
        pipeline = (
            "from mellea import start_session\n"
            "def run_pipeline(q):\n"
            "    with start_session(backend='watsonx') as m:\n"
            "        return m.instruct(q)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_no_watsonx(pkg).verdict == "fail"

    def test_fails_across_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "config.py": 'BACKEND = "watsonx"\n',
                    "slots.py": "# legacy backend was watsonx\n",
                },
            )
            result = lint_no_watsonx(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 2

    def test_passes_when_no_files_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"README.md": "# pkg\n"})
            assert lint_no_watsonx(pkg).verdict == "pass"


# ─── TestOptionalExtractionGuidance ───


class TestOptionalExtractionGuidance:
    """Test cases for lint_optional_extraction_guidance (KB11)."""

    def test_passes_with_extract_keyword(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel, Field\n"
            "class BookingIntent(BaseModel):\n"
            "    destination: str\n"
            "    departure_date: Optional[str] = Field(\n"
            "        default=None,\n"
            "        description=\"Extract the departure date.\",\n"
            "    )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            assert lint_optional_extraction_guidance(pkg).verdict == "pass"

    def test_passes_with_do_not_ask(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel, Field\n"
            "class SearchIntent(BaseModel):\n"
            "    query: str\n"
            "    max_results: Optional[int] = Field(\n"
            "        default=None,\n"
            "        description=\"Limit; do not ask the user.\",\n"
            "    )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            assert lint_optional_extraction_guidance(pkg).verdict == "pass"

    def test_passes_with_if_the(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel, Field\n"
            "class ComplianceSchema(BaseModel):\n"
            "    contract_context: Optional[str] = Field(\n"
            "        default=None,\n"
            "        description=\"Contract context if the user provided one.\",\n"
            "    )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            assert lint_optional_extraction_guidance(pkg).verdict == "pass"

    def test_fails_with_no_field_at_all(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel\n"
            "class BookingIntent(BaseModel):\n"
            "    destination: str\n"
            "    departure_date: Optional[str]\n"
            "    return_date: Optional[str] = None\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            result = lint_optional_extraction_guidance(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 2

    def test_fails_with_field_no_extraction_keyword(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel, Field\n"
            "class BookingIntent(BaseModel):\n"
            "    departure_date: Optional[str] = Field(\n"
            "        default=None, description=\"The departure date.\",\n"
            "    )\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            result = lint_optional_extraction_guidance(pkg)
            assert result.verdict == "fail"
            assert "departure_date" in result.failures[0].message

    def test_fails_with_field_no_description(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel, Field\n"
            "class SearchIntent(BaseModel):\n"
            "    max_results: Optional[int] = Field(default=None)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            assert lint_optional_extraction_guidance(pkg).verdict == "fail"

    def test_ignores_non_schema_intent_classes(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel\n"
            "class RoleAssessmentResult(BaseModel):\n"
            "    confidence: Optional[float] = None\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": schemas})
            assert lint_optional_extraction_guidance(pkg).verdict == "pass"

    def test_pulls_in_format_referenced_class(self):
        schemas = (
            "from typing import Optional\n"
            "from pydantic import BaseModel\n"
            "class RoleAssessment(BaseModel):\n"
            "    confidence: Optional[float] = None\n"
        )
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import RoleAssessment\n"
            "def run_pipeline(x: str) -> str:\n"
            "    with start_session('o','m') as m:\n"
            "        return m.instruct('classify', format=RoleAssessment)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"schemas.py": schemas, "pipeline.py": pipeline},
            )
            result = lint_optional_extraction_guidance(pkg)
            assert result.verdict == "fail"
            assert "RoleAssessment" in result.failures[0].message

    def test_skipped_when_schemas_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": "x = 1\n"})
            assert lint_optional_extraction_guidance(pkg).verdict == "skipped"

    def test_skipped_when_schemas_unparseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"schemas.py": "class :(:\n    pass\n"})
            assert lint_optional_extraction_guidance(pkg).verdict == "skipped"


# ─── TestRunPipelineParamsTyped ───


class TestRunPipelineParamsTyped:
    """Test cases for lint_run_pipeline_params_typed (Rule 3-1)."""

    def test_passes_with_all_params_annotated(self):
        pipeline = (
            "def run_pipeline(ticket: str, priority: str = 'normal') -> dict:\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_run_pipeline_params_typed(pkg).verdict == "pass"

    def test_passes_with_no_params(self):
        pipeline = "def run_pipeline() -> dict:\n    return {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_run_pipeline_params_typed(pkg).verdict == "pass"

    def test_fails_with_one_bare_param(self):
        pipeline = (
            "def run_pipeline(ticket, priority: str = 'normal') -> dict:\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_run_pipeline_params_typed(pkg)
            assert result.verdict == "fail"
            assert "ticket" in result.failures[0].message

    def test_fails_with_all_params_bare(self):
        pipeline = "def run_pipeline(t, u, q) -> dict:\n    return {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_run_pipeline_params_typed(pkg)
            assert result.verdict == "fail"
            assert len(result.failures) == 3

    def test_fails_with_kwonly_bare(self):
        pipeline = "def run_pipeline(q: str, *, flag) -> dict:\n    return {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_run_pipeline_params_typed(pkg)
            assert result.verdict == "fail"
            assert "flag" in result.failures[0].message

    def test_ignores_variadic_args(self):
        pipeline = (
            "def run_pipeline(query: str, *args, **kwargs) -> dict:\n"
            "    return {}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_run_pipeline_params_typed(pkg).verdict == "pass"

    def test_skipped_when_pipeline_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": "x=1\n"})
            assert lint_run_pipeline_params_typed(pkg).verdict == "skipped"

    def test_skipped_when_run_pipeline_absent(self):
        pipeline = "def run_phase_2(x) -> dict:\n    return {}\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_run_pipeline_params_typed(pkg).verdict == "skipped"


# ─── TestImportSoundness ───


def _api_ref(modules: list, grounding_unavailable: bool = False) -> str:
    return json.dumps(
        {
            "format_version": "1.0",
            "mellea_version": "0.5.0",
            "grounding_unavailable": grounding_unavailable,
            "modules": {m: {} for m in modules},
            "forbidden_param_names": [],
            "compatibility": [],
        }
    )


class TestImportSoundness:
    """Test cases for lint_import_soundness (Rule 4-2)."""

    def test_passes_with_known_paths(self):
        api_ref = _api_ref(
            ["mellea.backends.model_options", "mellea.stdlib.requirements"]
        )
        pipeline = (
            "from mellea.backends.model_options import ModelOption\n"
            "from mellea.stdlib.requirements import req\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_import_soundness(pkg).verdict == "pass"

    def test_fails_on_shortened_path(self):
        api_ref = _api_ref(["mellea.backends.model_options"])
        pipeline = "from mellea.model_options import ModelOption\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            result = lint_import_soundness(pkg)
            assert result.verdict == "fail"
            assert "mellea.model_options" in result.failures[0].message

    def test_fails_on_hallucinated_submodule(self):
        api_ref = _api_ref(["mellea.stdlib.requirements"])
        pipeline = "from mellea.stdlib.strategies import X\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_import_soundness(pkg).verdict == "fail"

    def test_fails_on_bare_import_mellea_submodule(self):
        api_ref = _api_ref(["mellea.backends.model_options"])
        pipeline = "import mellea.not_a_real_module\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_import_soundness(pkg).verdict == "fail"

    def test_skipped_when_api_ref_absent(self):
        pipeline = "from mellea.anything import X\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert lint_import_soundness(pkg).verdict == "skipped"

    def test_skipped_when_grounding_unavailable(self):
        api_ref = _api_ref([], grounding_unavailable=True)
        pipeline = "from mellea.bogus import X\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_import_soundness(pkg).verdict == "skipped"

    def test_skipped_when_malformed_api_ref(self):
        pipeline = "from mellea.foo import X\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": "{not valid",
                },
            )
            assert lint_import_soundness(pkg).verdict == "skipped"

    def test_ignores_non_mellea_imports(self):
        api_ref = _api_ref(["mellea.backends.model_options"])
        pipeline = (
            "import pydantic\n"
            "from anthropic import Anthropic\n"
            "from mellea.backends.model_options import ModelOption\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_import_soundness(pkg).verdict == "pass"

    def test_skips_fixtures_dir(self):
        api_ref = _api_ref(["mellea.backends.model_options"])
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "fixtures/x.py": "from mellea.bogus import X\n",
                    "intermediate/mellea_api_ref.json": api_ref,
                    "pipeline.py": (
                        "from mellea.backends.model_options import ModelOption\n"
                    ),
                },
            )
            assert lint_import_soundness(pkg).verdict == "pass"


# ─── TestStdlibArity ───


class TestStdlibArity:
    """Test cases for lint_stdlib_arity (Rule 4-4)."""

    def test_passes_with_correct_calls(self):
        code = (
            "from mellea.stdlib.requirements import req, check, simple_validate\n"
            "R = req('Keep response under 100 words')\n"
            "R2 = req('words', validation_fn=simple_validate(lambda x: True))\n"
            "result = check(R, 'output')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": code})
            assert lint_stdlib_arity(pkg).verdict == "pass"

    def test_fails_when_req_called_with_zero_positional(self):
        code = "from mellea.stdlib.requirements import req\nBAD = req()\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": code})
            result = lint_stdlib_arity(pkg)
            assert result.verdict == "fail"
            assert "0 positional" in result.failures[0].message

    def test_fails_when_simple_validate_called_with_zero_args(self):
        code = (
            "from mellea.stdlib.requirements import req, simple_validate\n"
            "R = req('x', validation_fn=simple_validate())\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": code})
            assert lint_stdlib_arity(pkg).verdict == "fail"

    def test_fails_when_req_uses_unrecognised_kwarg(self):
        code = (
            "from mellea.stdlib.requirements import req\n"
            "R = req('text', validator=lambda x: True)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": code})
            result = lint_stdlib_arity(pkg)
            assert result.verdict == "fail"
            assert "validator" in result.failures[0].message

    def test_passes_check_with_two_positional(self):
        code = (
            "from mellea.stdlib.requirements import req, check\n"
            "R = req('words')\n"
            "ok = check(R, 'output text')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"requirements.py": code})
            assert lint_stdlib_arity(pkg).verdict == "pass"

    def test_ignores_unknown_callees(self):
        code = "def foo(*a, **k): pass\nfoo()\nfoo(1, 2, 3, weird=42)\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"main.py": code})
            assert lint_stdlib_arity(pkg).verdict == "pass"

    def test_passes_when_grounding_unavailable(self):
        api_ref = _api_ref([], grounding_unavailable=True)
        code = "from mellea.stdlib.requirements import req\nR = req('hello')\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "requirements.py": code,
                    "intermediate/mellea_api_ref.json": api_ref,
                },
            )
            assert lint_stdlib_arity(pkg).verdict == "pass"

    def test_skips_fixtures_dir(self):
        code = "from mellea.stdlib.requirements import req\nBAD = req()\n"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"fixtures/x.py": code})
            assert lint_stdlib_arity(pkg).verdict == "pass"


# ─── TestMelleafyJsonConsistency ───


_DELETE = object()


def _full_melleafy_json(**overrides) -> str:
    base = {
        "manifest_version": "1.1.0",
        "entry_signature": "run_pipeline(query: str) -> str",
        "package_name": "test_pkg",
        "source_runtime": "openclaw",
        "modality": "text",
        "categories_resolved": {"c6_tools": 0},
        "declared_env_vars": [],
        "pipeline_parameters": ["query"],
    }
    for k, v in overrides.items():
        if v is _DELETE:
            base.pop(k, None)
        else:
            base[k] = v
    return json.dumps(base)


class TestMelleafyJsonConsistency:
    """Test cases for lint_melleafy_json_consistency (Rule 5-1 sub-check A)."""

    def test_passes_with_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"melleafy.json": _full_melleafy_json()}
            )
            result = lint_melleafy_json_consistency(pkg)
            assert result.verdict == "pass"
            assert result.failures == []

    def test_passes_with_only_hard_fields_with_warning(self):
        manifest = json.dumps(
            {
                "manifest_version": "1.1.0",
                "entry_signature": "run_pipeline() -> None",
                "package_name": "p",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"melleafy.json": manifest})
            result = lint_melleafy_json_consistency(pkg)
            assert result.verdict == "pass"
            assert any("WARN" in f.message for f in result.failures)

    def test_fails_when_manifest_version_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(manifest_version=_DELETE)},
            )
            result = lint_melleafy_json_consistency(pkg)
            assert result.verdict == "fail"

    def test_fails_when_entry_signature_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(entry_signature=_DELETE)},
            )
            assert lint_melleafy_json_consistency(pkg).verdict == "fail"

    def test_fails_when_package_name_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(package_name=_DELETE)},
            )
            assert lint_melleafy_json_consistency(pkg).verdict == "fail"

    def test_fails_when_manifest_version_below_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(manifest_version="1.0.0")},
            )
            result = lint_melleafy_json_consistency(pkg)
            assert result.verdict == "fail"

    def test_passes_when_manifest_version_above_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(manifest_version="2.0.0")},
            )
            assert lint_melleafy_json_consistency(pkg).verdict == "pass"

    def test_fails_when_manifest_version_not_semver(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"melleafy.json": _full_melleafy_json(manifest_version="abc")},
            )
            assert lint_melleafy_json_consistency(pkg).verdict == "fail"

    def test_fails_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"melleafy.json": "{not valid"})
            assert lint_melleafy_json_consistency(pkg).verdict == "fail"

    def test_skipped_when_manifest_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": "pass\n"})
            assert lint_melleafy_json_consistency(pkg).verdict == "skipped"


# ─── TestInstructResultParseBeforeAccess ───


class TestInstructResultParseBeforeAccess:
    """Test cases for lint_instruct_result_parse_before_access (KB1)."""

    _HELPERS = (
        "def _parse_instruct_result(thunk, model_class):\n"
        "    return model_class.model_validate_json(thunk.value)\n"
        "def _safe_parse_with_fallback(thunk, model_class, **fb):\n"
        "    try: return model_class.model_validate_json(thunk.value)\n"
        "    except Exception: return model_class(**fb)\n"
    )

    def test_passes_with_parse_instruct_result(self):
        pipeline = self._HELPERS + (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    intent = _parse_instruct_result(thunk, Intent)\n"
            "    return intent.query_type\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_passes_with_safe_parse_with_fallback(self):
        pipeline = self._HELPERS + (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    intent = _safe_parse_with_fallback(thunk, Intent, query_type='x')\n"
            "    return intent.query_type\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_passes_with_model_validate_json_value(self):
        pipeline = self._HELPERS + (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    parsed = Intent.model_validate_json(thunk.value)\n"
            "    return parsed.query_type\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_fails_with_direct_field_access(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    return thunk.query_type\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_result_parse_before_access(pkg)
            assert result.verdict == "fail"
            assert "thunk" in result.failures[0].message

    def test_fails_with_model_dump_call(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        t = m.instruct('go', format=Intent)\n"
            "    return t.model_dump()\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_instruct_result_parse_before_access(pkg)
            assert result.verdict == "fail"

    def test_fails_with_parsed_repr(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        t = m.instruct('go', format=Intent)\n"
            "    x = t.parsed_repr\n"
            "    return x\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "fail"
            )

    def test_passes_with_access_on_reassigned_parsed_var(self):
        pipeline = self._HELPERS + (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    parsed = _safe_parse_with_fallback(thunk, Intent, query_type='x')\n"
            "    return parsed.query_type + parsed.location\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_untracks_thunk_after_rebind(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    thunk = {'query_type': 'x'}\n"
            "    return thunk['query_type']\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_multi_function_scope_isolation(self):
        pipeline = self._HELPERS + (
            "from mellea import start_session\n"
            "from .schemas import Intent\n"
            "def fn_a():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Intent)\n"
            "    parsed = _parse_instruct_result(thunk, Intent)\n"
            "    return parsed.query_type\n"
            "def fn_b(thunk):\n"
            "    return thunk.query_type\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_walks_slots_and_constrained_slots(self):
        bad = (
            "from mellea import start_session\n"
            "from .schemas import S\n"
            "def f():\n"
            "    with start_session('o','m') as m:\n"
            "        t = m.instruct('go', format=S)\n"
            "    return t.x\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"slots.py": bad, "constrained_slots.py": bad},
            )
            result = lint_instruct_result_parse_before_access(pkg)
            assert result.verdict == "fail"

    def test_skips_unparseable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"pipeline.py": "def broken(:\n    pass\n"}
            )
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )

    def test_ignores_instruct_without_format_kwarg(self):
        pipeline = (
            "from mellea import start_session\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('classify the query')\n"
            "    return thunk.value\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            assert (
                lint_instruct_result_parse_before_access(pkg).verdict == "pass"
            )


# ─── TestComplexSchemaNeedsStrategyOrFallback ───


class TestComplexSchemaNeedsStrategyOrFallback:
    """Test cases for lint_complex_schema_needs_strategy_or_fallback (KB2)."""

    _SIMPLE_SCHEMA = (
        "from pydantic import BaseModel\n"
        "class Tiny(BaseModel):\n"
        "    a: str\n"
        "    b: str\n"
    )

    _COMPLEX_BY_FIELD_COUNT = (
        "from pydantic import BaseModel\n"
        "class Big(BaseModel):\n"
        "    a: str\n    b: str\n    c: str\n    d: str\n    e: str\n"
    )

    _COMPLEX_BY_LIST = (
        "from pydantic import BaseModel\n"
        "class WithList(BaseModel):\n"
        "    items: list[str]\n"
        "    name: str\n"
    )

    def test_passes_with_simple_schema_no_mitigation(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Tiny\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Tiny)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"pipeline.py": pipeline, "schemas.py": self._SIMPLE_SCHEMA},
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "pass"
            )

    def test_passes_with_strategy_kwarg(self):
        pipeline = (
            "from mellea import start_session\n"
            "from mellea.stdlib.sampling import RepairTemplateStrategy\n"
            "from .schemas import Big\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Big, "
            "strategy=RepairTemplateStrategy(loop_budget=3))\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "schemas.py": self._COMPLEX_BY_FIELD_COUNT,
                },
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "pass"
            )

    def test_passes_with_safe_parse_with_fallback_followup(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Big\n"
            "def _safe_parse_with_fallback(t, m, **fb): ...\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Big)\n"
            "    parsed = _safe_parse_with_fallback(thunk, Big, a='', b='',"
            " c='', d='', e='')\n"
            "    return parsed\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "schemas.py": self._COMPLEX_BY_FIELD_COUNT,
                },
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "pass"
            )

    def test_fails_with_complex_schema_no_mitigation(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Big\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Big)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "schemas.py": self._COMPLEX_BY_FIELD_COUNT,
                },
            )
            result = lint_complex_schema_needs_strategy_or_fallback(pkg)
            assert result.verdict == "fail"
            assert "Big" in result.failures[0].message

    def test_fails_with_list_field_no_mitigation(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import WithList\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=WithList)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "schemas.py": self._COMPLEX_BY_LIST,
                },
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "fail"
            )

    def test_passes_with_unknown_schema_name(self):
        pipeline = (
            "from mellea import start_session\n"
            "from external import Ghost\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Ghost)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "pipeline.py": pipeline,
                    "schemas.py": self._COMPLEX_BY_FIELD_COUNT,
                },
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "pass"
            )

    def test_passes_when_schemas_py_absent(self):
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Big\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Big)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"pipeline.py": pipeline})
            result = lint_complex_schema_needs_strategy_or_fallback(pkg)
            assert result.verdict == "pass"
            assert result.files_checked == 0

    def test_4_field_schema_is_not_complex(self):
        four_fields = (
            "from pydantic import BaseModel\n"
            "class Quad(BaseModel):\n"
            "    a: str\n    b: str\n    c: str\n    d: str\n"
        )
        pipeline = (
            "from mellea import start_session\n"
            "from .schemas import Quad\n"
            "def run_pipeline():\n"
            "    with start_session('o','m') as m:\n"
            "        thunk = m.instruct('go', format=Quad)\n"
            "    return thunk\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp), {"pipeline.py": pipeline, "schemas.py": four_fields}
            )
            assert (
                lint_complex_schema_needs_strategy_or_fallback(pkg).verdict
                == "pass"
            )


# ─── TestRunLints ───


class TestRunLints:
    """Integration tests for the run_lints orchestrator."""

    def test_overall_pass_when_all_pass(self):
        """A compliant synthetic package should yield overall_verdict == 'pass'."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite4.1:8b"\n'
        )
        tools = (
            "from pathlib import Path\n"
            "p = Path(__file__).parent / 'scripts' / 'bash' / 'x.sh'\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {
                    "fixtures/__init__.py": "ALL_FIXTURES = []\n",
                    "config.py": config,
                    "tools.py": tools,
                },
            )
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = run_lints(pkg)
            assert result.overall_verdict == "pass", (
                f"Compliant package should pass overall; got {result.overall_verdict} "
                f"with lints={[(r.lint_id, r.verdict) for r in result.lints]}"
            )
            assert result.failed is False

    def test_overall_fail_on_any_lint_fail(self):
        """A package with one failing lint should yield overall 'fail'."""
        # fixtures/__init__.py is wrong shape — fixtures-loader-contract fails.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"fixtures/__init__.py": 'INPUT = {"x": 1}\n'},
            )
            result = run_lints(pkg)
            assert result.overall_verdict == "fail", (
                f"Any failing lint should fail overall; got {result.overall_verdict} "
                f"with lints={[(r.lint_id, r.verdict) for r in result.lints]}"
            )
            assert result.failed is True

    def test_writes_step_7_report(self):
        """run_lints must write a parseable intermediate/step_7_report.json."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(
                Path(tmp),
                {"fixtures/__init__.py": "ALL_FIXTURES = []\n"},
            )
            run_lints(pkg)
            report_path = pkg / "intermediate" / "step_7_report.json"
            assert report_path.exists(), (
                f"step_7_report.json should be written at {report_path}"
            )
            data = json.loads(report_path.read_text())
            assert "format_version" in data, (
                f"Report should carry format_version; keys: {list(data.keys())}"
            )
            assert "overall_verdict" in data
            assert "lints" in data and isinstance(data["lints"], list)
            for lint_entry in data["lints"]:
                assert "lint_id" in lint_entry, (
                    f"Each lint entry needs lint_id; got {lint_entry}"
                )
                assert "verdict" in lint_entry
                assert "failures" in lint_entry
                assert isinstance(lint_entry["failures"], list)

    def test_overall_fail_on_missing_mandatory_writer_outputs(self):
        """An incomplete package missing config.py + fixtures/ must fail overall.

        Regression test for the out-of-tree writer-renderer bug: previously, both
        lints reported `skipped` for the missing-file case, the Step-7 aggregator
        counted `skipped` as `pass`, and incomplete packages shipped with a
        falsely-green verdict. Both lints now fail loudly so the aggregator
        propagates the fail upward.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            # No config.py, no fixtures/__init__.py — but a runtime directive
            # exists, so the runtime-defaults-bound lint actually proceeds past
            # its directive-missing skip branch and reaches the config.py check.
            _write_directive(pkg, "ollama", "granite4.1:8b")
            result = run_lints(pkg)
            assert result.overall_verdict == "fail", (
                f"Package missing both mandatory writer outputs must fail "
                f"overall; got {result.overall_verdict} with lints="
                f"{[(r.lint_id, r.verdict) for r in result.lints]}"
            )
            verdicts = {r.lint_id: r.verdict for r in result.lints}
            assert verdicts.get("fixtures-loader-contract") == "fail", (
                f"fixtures-loader-contract should fail on missing __init__.py; "
                f"verdicts={verdicts}"
            )
            assert verdicts.get("runtime-defaults-bound") == "fail", (
                f"runtime-defaults-bound should fail on missing config.py; "
                f"verdicts={verdicts}"
            )

    def test_intermediate_dir_created(self):
        """intermediate/ should be created if it didn't already exist."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)  # no intermediate/ to start
            assert not (pkg / "intermediate").exists()
            run_lints(pkg)
            assert (pkg / "intermediate").is_dir(), (
                "run_lints should create intermediate/ when absent"
            )
            assert (pkg / "intermediate" / "step_7_report.json").exists()
