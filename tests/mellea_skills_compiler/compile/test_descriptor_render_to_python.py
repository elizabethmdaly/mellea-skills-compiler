"""Tests for ``writer_renderer.render_descriptor_to_python``.

The wrapper's post-session writer flow renders ``config.py`` and
``fixtures/`` from their respective emission IRs. Before this fix it did
NOT render ``pipeline.py`` / ``schemas.py`` from
``intermediate/descriptor_emission.json``, despite the slash command
telling Claude "the wrapper renders those". Three skills in the May 17
overnight batch shipped descriptor_emission.json on disk without a
rendered pipeline.py, the Step 7 lint reported skipped (=pass), and the
downstream smoke check crashed.

These tests pin the new function's contract and the ``compile()`` flow's
post-session invocation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from mellea_skills_compiler.compile.writer_renderer import (
    render_descriptor_to_python,
)


# Path to the bundled hand-authored descriptor we ship as the
# 1-shot example. It is schema-valid v0.1 (no v0.3 bundled resources)
# so it renders without a skill_root anchor.
_BUNDLED_DESCRIPTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "mellea_skills_compiler"
    / "compile"
    / "data"
    / "sentry-find-bugs.descriptor.json"
)


def _load_bundled_descriptor() -> dict:
    return json.loads(_BUNDLED_DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def _make_pkg_with_descriptor(tmp: Path, descriptor: dict | None = None) -> Path:
    """Materialise a synthetic package with descriptor_emission.json on disk."""
    pkg = tmp / "pkg_mellea"
    pkg.mkdir()
    (pkg / "intermediate").mkdir()
    if descriptor is not None:
        (pkg / "intermediate" / "descriptor_emission.json").write_text(
            json.dumps(descriptor), encoding="utf-8"
        )
    return pkg


# ─── Render flow happy path ───


class TestRenderDescriptorToPython:
    def test_renders_when_emission_present(self):
        """descriptor_emission.json on disk → pipeline.py + schemas.py written."""
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            result = render_descriptor_to_python(pkg)
            assert result is not None
            assert result["verdict"] == "rendered", (
                f"Expected rendered; got {result}"
            )
            assert sorted(result["files_written"]) == ["pipeline.py", "schemas.py"]
            assert (pkg / "pipeline.py").exists()
            assert (pkg / "schemas.py").exists()
            assert "def run_pipeline" in (pkg / "pipeline.py").read_text()

    def test_skips_when_emission_absent(self):
        """No descriptor_emission.json → verdict=skipped, no files written.

        In legacy mode Claude writes pipeline.py directly; the absence of
        descriptor_emission.json is the signal that the wrapper has
        nothing to render. The lint then validates whatever ended up on
        disk (or hard-fails on its absence).
        """
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor=None)
            result = render_descriptor_to_python(pkg)
            assert result is not None
            assert result["verdict"] == "skipped"
            assert result["files_written"] == []
            assert result["reason"] == "descriptor_emission_absent"
            assert not (pkg / "pipeline.py").exists()
            assert not (pkg / "schemas.py").exists()

    def test_overwrites_existing_files(self):
        """Pre-existing pipeline.py / schemas.py are overwritten.

        Contract: in --use-descriptor mode the wrapper is the source of
        truth, so a Claude-written pipeline.py is overwritten with the
        deterministic descriptor-rendered version.
        """
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            (pkg / "pipeline.py").write_text(
                "# stale free-form Claude output\n"
                "def run_pipeline(x):\n"
                "    return x\n",
                encoding="utf-8",
            )
            (pkg / "schemas.py").write_text(
                "# stale\n", encoding="utf-8"
            )
            result = render_descriptor_to_python(pkg)
            assert result["verdict"] == "rendered"
            pipeline_text = (pkg / "pipeline.py").read_text()
            assert "stale" not in pipeline_text, (
                "Wrapper-rendered pipeline.py should overwrite stale "
                "Claude-written content"
            )
            assert "def run_pipeline" in pipeline_text
            schemas_text = (pkg / "schemas.py").read_text()
            assert "stale" not in schemas_text

    def test_schema_gate_rejects_invented_arg_value_shape(self):
        """gpai-code-of-practice failure pattern: the LLM emitted an
        invented ``args`` value of shape ``{'<name>': '#/outputs/<name>'}``
        (extrapolated from the legal ``captures`` shape into the illegal
        ``args`` location). The descriptor IR schema's ``ArgValue`` is a
        closed ``oneOf`` over six exact shapes; this is not one of them.

        Before this gate existed, the renderer failed at AST-lowering
        time with a generic ``RendererError: unknown argument value
        shape: ...`` message. With the gate, the failure now happens
        before the renderer is invoked, with a precise JSON path naming
        the violation location.
        """
        descriptor = _load_bundled_descriptor()
        # Inject the invented shape into a call node's args. The
        # bundled descriptor uses v0.1; ArgValue's closed ``oneOf`` is
        # consistent across v0.1/v0.2/v0.3, so this triggers the same
        # schema rejection regardless of declared version.
        target = None
        for entry in descriptor.get("pipeline", []):
            if entry.get("kind") == "call" and "args" in entry:
                target = entry
                break
        assert target is not None, (
            "bundled descriptor missing a call node with args; this "
            "test depends on the fixture's shape"
        )
        target["args"]["__invented__"] = {
            "privacy_notice_delivery": "#/outputs/privacy_notice_delivery"
        }

        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            result = render_descriptor_to_python(pkg)

            assert result is not None
            assert result["verdict"] == "failed", (
                "Schema-invalid descriptor must fail at the gate, not reach "
                "the renderer. Got: " + repr(result)
            )
            assert "schema_errors" in result, (
                "Schema-gate failures must carry structured errors so D1 "
                "auto-repair can feed them back to the LLM"
            )
            assert result["schema_errors"], (
                "schema_errors should be non-empty when verdict=failed via "
                "the schema gate"
            )
            # Reason string names the schema file.
            assert "descriptor.schema" in result["reason"]
            # The violating pipeline node should be referenced in at
            # least one error path. jsonschema reports oneOf failures at
            # the top of the failing union, so we check for the node
            # path itself (e.g. ``/pipeline/0``) rather than the leaf
            # location of the invented key.
            assert any(
                "/pipeline/" in e["path"] for e in result["schema_errors"]
            ), (
                "Expected at least one schema error pointing into the "
                "pipeline subtree; got paths: "
                + repr([e["path"] for e in result["schema_errors"]])
            )
            # Files NOT written — renderer must not have been invoked.
            assert not (pkg / "pipeline.py").exists()
            assert not (pkg / "schemas.py").exists()

    def test_schema_gate_rejects_missing_required_field(self):
        """A descriptor missing a top-level required field (e.g.
        ``pipeline``) is rejected by the gate. Catches a different class
        of structural mistake than the gpai case — exercising the same
        gate for breadth coverage.
        """
        descriptor = _load_bundled_descriptor()
        # Strip a required field. The schema declares pipeline as required.
        descriptor.pop("pipeline", None)

        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            result = render_descriptor_to_python(pkg)

            assert result["verdict"] == "failed"
            assert result["schema_errors"], (
                "missing-required-field violation must surface as "
                "structured schema errors"
            )

    def test_schema_gate_passes_valid_descriptor_through(self):
        """A schema-valid descriptor must still render normally — the
        gate is transparent in the happy path.
        """
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            result = render_descriptor_to_python(pkg)

            assert result["verdict"] == "rendered", (
                "The schema gate must not interfere with valid descriptors. "
                f"Got: {result}"
            )
            assert "schema_errors" not in result
            assert (pkg / "pipeline.py").exists()

    def test_idempotent(self):
        """Running render twice produces byte-identical output."""
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            assert render_descriptor_to_python(pkg)["verdict"] == "rendered"
            first_pipeline = (pkg / "pipeline.py").read_text()
            first_schemas = (pkg / "schemas.py").read_text()

            assert render_descriptor_to_python(pkg)["verdict"] == "rendered"
            second_pipeline = (pkg / "pipeline.py").read_text()
            second_schemas = (pkg / "schemas.py").read_text()

            assert first_pipeline == second_pipeline
            assert first_schemas == second_schemas

    def test_handles_renderer_error(self):
        """Renderer raises (unknown operator) → verdict=failed; no exception leaks.

        The wrapper-side function MUST swallow renderer errors and return a
        structured failure summary so the surrounding compile flow can log
        and continue — the post-session lint will then hard-fail on the
        missing pipeline.py via ``pipeline-entry-canonical`` (Bug 1 fix).
        """
        bad_descriptor: dict = {
            "descriptor_version": "0.1",
            "skill": {"name": "x"},
            "pipeline": [
                {
                    "id": "n1",
                    "kind": "composition",
                    "operator": "NOT_A_REAL_OPERATOR",
                    "children": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), bad_descriptor)
            result = render_descriptor_to_python(pkg)
            assert result is not None
            assert result["verdict"] == "failed", (
                f"Expected failed verdict on malformed descriptor; got {result}"
            )
            assert result["reason"] is not None
            assert result["files_written"] == []
            # No files were written when the render failed.
            assert not (pkg / "pipeline.py").exists()
            assert not (pkg / "schemas.py").exists()

    def test_handles_unreadable_descriptor_json(self):
        """descriptor_emission.json with invalid JSON → verdict=failed."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor=None)
            (pkg / "intermediate" / "descriptor_emission.json").write_text(
                "{not valid json", encoding="utf-8"
            )
            result = render_descriptor_to_python(pkg)
            assert result["verdict"] == "failed"
            assert "unreadable" in result["reason"]

    def test_returns_failed_when_surface_unavailable(self):
        """No surface JSON → verdict=failed, no files written.

        Defensive check: when ``surface=None`` is passed AND the bundled
        surface cannot be loaded (mocked), we return a failed summary.
        """
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            with mock.patch(
                "mellea_skills_compiler.compile.writer_renderer._load_default_surface",
                return_value=None,
            ):
                result = render_descriptor_to_python(pkg)
            assert result["verdict"] == "failed"
            assert "surface" in result["reason"].lower()

    def test_explicit_surface_bypasses_bundled_loader(self):
        """When surface is passed explicitly, no bundled load is attempted.

        Allows callers (tests, alternate Mellea versions) to inject their
        own surface without forcing the importlib.resources lookup.
        """
        descriptor = _load_bundled_descriptor()
        # Load the same default the bundled loader would have produced;
        # explicit pass-through must produce a successful render.
        from importlib.resources import files as _resource_files
        surface_path = Path(
            str(
                _resource_files("mellea_skills_compiler.mellea_surface")
                / "data"
                / "surface_0.5.0.json"
            )
        )
        surface = json.loads(surface_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            # Sentinel: if we try to load the bundled default, the test
            # fails. The explicit ``surface=`` path must bypass it.
            with mock.patch(
                "mellea_skills_compiler.compile.writer_renderer._load_default_surface",
                side_effect=AssertionError(
                    "should not load bundled surface when explicit one passed"
                ),
            ):
                result = render_descriptor_to_python(pkg, surface=surface)
            assert result["verdict"] == "rendered"


# ─── compile() flow integration ───


class TestCompileFlowInvokesDescriptorRender:
    """When ``compile(..., use_descriptor=True)`` succeeds through the spawn
    step and a ``*_mellea/`` directory exists with descriptor_emission.json,
    the post-session block must invoke ``render_descriptor_to_python``.
    Legacy ``use_descriptor=False`` runs must NOT invoke it.

    We exercise the wiring directly via the import path the post-session
    block uses, plus a behavioural test against a synthetic package.
    """

    def test_render_descriptor_function_imported_in_compile_module(self):
        """The compile module references render_descriptor_to_python.

        Soft pin: any future refactor that drops the wiring must also
        update this test (or risk re-introducing the May 17 regression).
        """
        from mellea_skills_compiler.compile import mellea_skills

        source = Path(mellea_skills.__file__).read_text(encoding="utf-8")
        assert "render_descriptor_to_python" in source, (
            "compile() must reference render_descriptor_to_python so the "
            "wrapper renders pipeline.py / schemas.py from "
            "descriptor_emission.json under --use-descriptor."
        )

    def test_descriptor_render_runs_against_real_package(self):
        """End-to-end: a package with descriptor_emission.json gets pipeline.py
        rendered when render_descriptor_to_python runs (the same callable
        the post-session block invokes).
        """
        descriptor = _load_bundled_descriptor()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _make_pkg_with_descriptor(Path(tmp), descriptor)
            # Simulate what compile() does post-session under use_descriptor=True.
            result = render_descriptor_to_python(pkg)
            assert result["verdict"] == "rendered"
            assert (pkg / "pipeline.py").exists()
            assert "def run_pipeline" in (pkg / "pipeline.py").read_text()
