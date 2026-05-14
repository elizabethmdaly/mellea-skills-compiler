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
    lint_fixtures_loader_contract,
    lint_runtime_defaults_bound,
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
            'MODEL_ID = "granite3.3:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite3.3:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "pass", (
                f"Matching config and directive should pass; got {result.verdict} "
                f"with failures {[f.message for f in result.failures]}"
            )

    def test_fails_when_backend_diverges(self):
        """config.py BACKEND != directive backend → fail with actionable message."""
        config = (
            'BACKEND = "watsonx"\n'
            'MODEL_ID = "granite3.3:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
            _write_directive(pkg, "ollama", "granite3.3:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "fail", (
                f"Model ID divergence should fail; got {result.verdict}"
            )
            assert len(result.failures) == 1
            msg = result.failures[0].message
            assert "granite3.3:2b" in msg
            assert "granite3.3:8b" in msg

    def test_fails_when_backend_constant_missing(self):
        """config.py without BACKEND → fail with a clear message naming BACKEND."""
        config = 'MODEL_ID = "granite3.3:8b"\n'
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
            'MODEL_ID = "granite3.3:8b"\n'
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
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
            'MODEL_ID = "granite3.3:8b"\n'
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
            'MODEL_ID: Final[str] = "granite3.3:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
            'MODEL_ID = "granite3.3:8b"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_package(Path(tmp), {"config.py": config})
            _write_directive(pkg, "ollama", "granite3.3:8b")
            result = lint_runtime_defaults_bound(pkg)
            assert result.verdict == "pass", (
                f"Plain BACKEND/MODEL_ID assignments should be recognised; "
                f"got {result.verdict} with failures "
                f"{[f.message for f in result.failures]}"
            )


# ─── TestRunLints ───


class TestRunLints:
    """Integration tests for the run_lints orchestrator."""

    def test_overall_pass_when_all_pass(self):
        """A compliant synthetic package should yield overall_verdict == 'pass'."""
        config = (
            'BACKEND = "ollama"\n'
            'MODEL_ID = "granite3.3:8b"\n'
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
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
            _write_directive(pkg, "ollama", "granite3.3:8b")
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
