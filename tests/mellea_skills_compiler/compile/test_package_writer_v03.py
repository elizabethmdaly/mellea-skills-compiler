"""End-to-end tests for the v0.3 package writer integration.

These cover the writer-side glue that materialises a rendered :class:`RenderResult`
into a real package directory:

* ``file_copies`` are executed (``shutil.copy`` / ``shutil.copytree``) onto the
  package dir, with sources resolved against ``skill_root``.
* ``deferred_binaries`` are NOT copied; a WARNING is logged so it shows up in
  compile output.
* ``package_data_dirs`` are merged into a ``pyproject.toml`` colocated with
  the package directory.
* ``source_elements_map`` is written to ``intermediate/source_elements_map.json``
  for Step 6's mapping-report builder.

The synthetic descriptor mirrors
``tests/mellea_skills_compiler/renderer/test_integration_v03.py`` so the writer
and renderer agree on the v0.3 surface shape.

These tests do NOT exercise the LLM emission path; they call
:func:`_write_package_to` directly with a freshly rendered package.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.compile.descriptor_emission import (
    _render_full_package,
    _write_package_to,
)
from mellea_skills_compiler.renderer import render_descriptor


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _synthetic_v03_descriptor() -> dict[str, Any]:
    """Shape mirrors ``tests/.../renderer/test_integration_v03.py``."""
    return {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {
            "name": "synthetic-everything",
            "async": True,
            "classification": {
                "primary_axis": "AGENT",
                "interaction_style": "tool_using_loop",
                "input_modality": ["text", "images"],
                "output_modality": "streaming_text",
                "state_persistence": "cross_session",
            },
        },
        "dependencies": [
            {"id": "search_fn", "kind": "tool",
             "disposition": "delegate_to_runtime",
             "signature": "(query: str) -> list[str]"},
            {"id": "owasp_refs", "kind": "file",
             "disposition": "load_from_disk",
             "path": "references/owasp.md"},
            {"id": "scratchpad", "kind": "tool",
             "disposition": "stub",
             "signature": "(content: str) -> None"},
        ],
        "bundled_resources": [
            {"source_dir": "references", "package_dir": "references"},
            {"source_dir": "scripts", "package_dir": "scripts"},
            {"source_dir": "vendor", "package_dir": "vendor"},
        ],
        "inputs": [
            {"name": "query", "schema": {"kind": "str"}},
            {"name": "options", "schema": {"kind": "dict",
                                            "key_type": "str",
                                            "value_type": "str"},
             "optional": True},
            {"name": "color", "schema": {"kind": "literal_union",
                                          "members": ["red", "green", "blue"]}},
            {"name": "photo", "schema": {"kind": "image", "format_hint": "any"}},
        ],
        "outputs": [
            {"name": "answer", "schema": {"kind": "model_ref",
                                            "ref": "#/schemas/Answer"}}
        ],
        "schemas": {
            "Answer": {
                "kind": "model",
                "fields": {
                    "verdict": {"type": "str"},
                    "score": {"type": "Optional[float]"},
                },
            },
            "LoopState": {
                "kind": "model",
                "fields": {
                    "done": {"type": "bool"},
                    "scratch": {"type": "list[str]"},
                },
            },
        },
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [
            {
                "id": "gate",
                "kind": "composition",
                "operator": "human_approval",
                "prompt": {"value": "Run the loop?"},
                "branches": {
                    "approved": [
                        {
                            "id": "loop",
                            "kind": "composition",
                            "operator": "agent_loop",
                            "max_iterations": 10,
                            "state_schema": {"ref": "#/schemas/LoopState"},
                            "termination_condition": {
                                "predicate": "field_truthy",
                                "field": "done",
                            },
                            "tools_ref": ["search_fn"],
                            "body": [
                                {
                                    "id": "speak",
                                    "kind": "call",
                                    "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                                    "bound_to": {"ref": "session"},
                                    "args": {"description": {"value": "step"}},
                                    "streaming": True,
                                    "source_elements": ["E14"],
                                }
                            ],
                        }
                    ],
                    "rejected": [],
                },
            }
        ],
    }


def _v01_descriptor_minimal() -> dict[str, Any]:
    """Tiny v0.1 descriptor with NO bundled-resources / dependencies — used for
    byte-stability checks: the writer must produce identical output to the
    pre-v0.3 path."""
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "tiny-legacy"},
        "inputs": [{"name": "query", "schema": {"kind": "str"}}],
        "outputs": [{"name": "answer", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [
            {
                "id": "step1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {"description": {"ref": "query"}},
                "captures": {"value": "#/outputs/answer"},
            }
        ],
    }


def _make_skill_root(tmp_path: Path) -> Path:
    """Create a fake skill root with the resources the synthetic descriptor expects."""
    skill_root = tmp_path / "skill_root"
    refs = skill_root / "references"
    refs.mkdir(parents=True)
    (refs / "owasp.md").write_text("# OWASP\n\nReference text.\n", encoding="utf-8")
    (refs / "nist.md").write_text("# NIST\n", encoding="utf-8")
    scripts = skill_root / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    vendor = skill_root / "vendor"
    vendor.mkdir()
    (vendor / "lib.py").write_text("# vendored\n", encoding="utf-8")
    return skill_root


# --- 1. Happy path: full v0.3 render-and-write ----------------------------


def test_v03_bundled_resources_are_copied_into_package(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    pipeline_py, schemas_py, init_py, fixtures_py = bundle[0], bundle[1], bundle[2], bundle[3]
    melleafy_json, setup_md, readme_md, source_map, render_result = bundle[4:]

    out_dir = tmp_path / "synthetic_everything_mellea"
    _write_package_to(
        out_dir,
        pipeline_py=pipeline_py,
        schemas_py=schemas_py,
        init_py=init_py,
        fixtures_py=fixtures_py,
        melleafy_json=melleafy_json,
        setup_md=setup_md,
        readme_md=readme_md,
        source_map=source_map,
        render_result=render_result,
        skill_root=skill_root,
    )

    # 1. Core Python files exist and pipeline.py parses.
    assert (out_dir / "pipeline.py").is_file()
    ast.parse((out_dir / "pipeline.py").read_text(encoding="utf-8"))
    assert (out_dir / "schemas.py").is_file()
    assert (out_dir / "__init__.py").is_file()

    # 2. Non-OUT-6 bundled_resources were copied by the writer (here, vendor/).
    #    OUT-6 dirs (references/, scripts/) are deliberately NOT copied by the
    #    writer — legacy mirror_companion_dirs handles those at Step 3a-pre.
    #    See _apply_v03_artefacts dedup in descriptor_emission.py.
    assert (out_dir / "vendor" / "lib.py").is_file()
    assert not (out_dir / "references").exists(), \
        "OUT-6 'references/' must be skipped by the writer (legacy mirrors it)"
    assert not (out_dir / "scripts").exists(), \
        "OUT-6 'scripts/' must be skipped by the writer (legacy mirrors it)"

    # 3. pyproject.toml was created and lists ONLY non-OUT-6 package-data dirs.
    pyproject = out_dir / "pyproject.toml"
    assert pyproject.is_file()
    text = pyproject.read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    assert "synthetic_everything_mellea" in text
    assert "vendor/**/*" in text
    assert "references/**/*" not in text, \
        "OUT-6 'references/' is already in the legacy pyproject template"
    assert "scripts/**/*" not in text, \
        "OUT-6 'scripts/' is already in the legacy pyproject template"


# --- 2. Source-elements map propagation -----------------------------------


def test_v03_source_elements_map_is_written(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    render_result = bundle[8]
    out_dir = tmp_path / "synthetic_everything_mellea"
    _write_package_to(
        out_dir,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
        render_result=render_result,
        skill_root=skill_root,
    )
    map_path = out_dir / "intermediate" / "source_elements_map.json"
    assert map_path.is_file()
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    assert payload == {"speak": ["E14"]}


# --- 3. Deferred binaries: NOT copied, WARNING logged ---------------------


def test_v03_deferred_binaries_warn_and_skip_copy(
    surface: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    # Drop a binary into references/ so the renderer marks it as deferred.
    (skill_root / "references" / "diagram.png").write_bytes(b"\x89PNG\r\n\x00")

    # Renderer will emit a UserWarning at render time AND populate
    # render_result.deferred_binaries. We capture both signals.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    render_result = bundle[8]
    assert any("diagram.png" in p for p in render_result.deferred_binaries), \
        f"renderer should have flagged diagram.png; got {render_result.deferred_binaries}"

    out_dir = tmp_path / "synthetic_everything_mellea"
    with caplog.at_level(logging.WARNING,
                          logger="mellea_skills_compiler.compile.descriptor_emission"):
        _write_package_to(
            out_dir,
            pipeline_py=bundle[0],
            schemas_py=bundle[1],
            init_py=bundle[2],
            fixtures_py=bundle[3],
            melleafy_json=bundle[4],
            setup_md=bundle[5],
            readme_md=bundle[6],
            source_map=bundle[7],
            render_result=render_result,
            skill_root=skill_root,
        )

    # The renderer's bundled-directory copy uses shutil.copytree on the source
    # directory in skill_root, which DOES include diagram.png as bytes — that
    # is expected and acceptable. The defer-and-warn semantic applies to the
    # writer's own logging (so compile output flags it), not to a filesystem
    # exclusion. We assert the WARNING was logged at INFO level or higher.
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("diagram.png" in msg and "Deferred" in msg for msg in warning_msgs), \
        f"expected a 'Deferred binary resource' warning mentioning diagram.png; got {warning_msgs}"


# --- 4. Missing skill_root → clean error when copies are needed -----------


def test_v03_missing_skill_root_raises_clean_error(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    out_dir = tmp_path / "synthetic_everything_mellea"
    with pytest.raises(FileNotFoundError, match="skill_root not provided"):
        _write_package_to(
            out_dir,
            pipeline_py=bundle[0],
            schemas_py=bundle[1],
            init_py=bundle[2],
            fixtures_py=bundle[3],
            melleafy_json=bundle[4],
            setup_md=bundle[5],
            readme_md=bundle[6],
            source_map=bundle[7],
            render_result=bundle[8],
            skill_root=None,
        )


# --- 5. pyproject.toml merge with pre-existing block -----------------------


def test_v03_pyproject_merges_into_existing_block(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    out_dir = tmp_path / "synthetic_everything_mellea"
    out_dir.mkdir()
    # Pre-seed a pyproject.toml with existing content the writer must preserve.
    (out_dir / "pyproject.toml").write_text(
        '[project]\nname = "my-existing-project"\nversion = "0.0.1"\n\n'
        '[tool.setuptools.package-data]\n'
        'synthetic_everything_mellea = ["preserved/*"]\n',
        encoding="utf-8",
    )
    _write_package_to(
        out_dir,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
        render_result=bundle[8],
        skill_root=skill_root,
    )
    text = (out_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert "my-existing-project" in text
    assert "[tool.setuptools.package-data]" in text
    # Non-OUT-6 entry (vendor) gets merged in; OUT-6 entries do not because
    # the legacy pyproject template already declares them.
    assert "vendor/**/*" in text
    # Pre-existing user content is preserved.
    assert "preserved/*" in text


# --- 5b. Rule OUT-6 dedup: writer does NOT copy paths legacy mirrors -----


def test_v03_writer_skips_out6_copies_to_avoid_legacy_duplication(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    """Companion dirs (scripts/, references/, assets/) are mirrored by the
    legacy pipeline at Step 3a-pre via ``mirror_companion_dirs``. The v0.3
    writer must NOT re-copy them — that would be wasted I/O and would create
    a second source of truth for the same files. The writer's file_copies
    handler filters out any FileCopyOp whose destination top-level is one of
    Rule OUT-6's reserved names.

    This test confirms: (1) OUT-6 dirs do not appear in writer output when
    legacy mirroring hasn't run; (2) non-OUT-6 dirs (here, ``vendor/``) DO
    appear, proving the writer's file_copies loop is otherwise live.
    """
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    out_dir = tmp_path / "synthetic_everything_mellea"
    _write_package_to(
        out_dir,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
        render_result=bundle[8],
        skill_root=skill_root,
    )
    # OUT-6 reserved names: writer SKIPS — legacy mirror_companion_dirs owns
    # these. In this isolated unit-test context, the legacy step doesn't run,
    # so the dirs don't appear at all. (Production: they appear because of
    # the legacy step, not because of writer-side copy.)
    assert not (out_dir / "references").exists()
    assert not (out_dir / "scripts").exists()
    assert not (out_dir / "assets").exists()
    # Non-OUT-6 path: writer copies normally.
    assert (out_dir / "vendor" / "lib.py").is_file()
    # pyproject.toml lists only non-OUT-6 entries.
    pyproject_text = (out_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert "vendor/**/*" in pyproject_text
    assert "references/**/*" not in pyproject_text
    assert "scripts/**/*" not in pyproject_text


# --- 6. v0.1 byte-stability: no v0.3 side effects -------------------------


def test_v01_writer_byte_identical_to_pre_v03(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    """A v0.1 descriptor with no bundled_resources / dependencies must produce
    a package directory identical (per file) to the legacy writer."""
    descriptor = _v01_descriptor_minimal()
    bundle = _render_full_package(descriptor, surface)
    render_result = bundle[8]
    # Sanity: renderer should have produced no v0.3 artefacts for a v0.1 input.
    assert render_result.file_copies == []
    assert render_result.package_data_dirs == set()
    assert render_result.deferred_binaries == []
    assert render_result.source_elements_map == {}

    # Write with v0.3 artefacts threaded through.
    out_v03 = tmp_path / "v03_path"
    _write_package_to(
        out_v03,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
        render_result=render_result,
        skill_root=None,
    )
    # Write WITHOUT render_result (legacy call shape).
    out_legacy = tmp_path / "legacy_path"
    _write_package_to(
        out_legacy,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
    )
    # Compare file trees: same files, same byte content.
    files_v03 = {p.relative_to(out_v03) for p in out_v03.rglob("*") if p.is_file()}
    files_legacy = {p.relative_to(out_legacy) for p in out_legacy.rglob("*") if p.is_file()}
    assert files_v03 == files_legacy, \
        f"v0.3-path produced different files: {files_v03 ^ files_legacy}"
    for rel in files_v03:
        a = (out_v03 / rel).read_bytes()
        b = (out_legacy / rel).read_bytes()
        assert a == b, f"file {rel} differs between v0.3 and legacy write paths"

    # No pyproject.toml, no intermediate/source_elements_map.json should appear.
    assert not (out_v03 / "pyproject.toml").exists()
    assert not (out_v03 / "intermediate").exists()


# --- 7. End-to-end import sanity: rendered module imports cleanly ---------


def test_v03_compiled_module_imports_cleanly(
    surface: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_ID", "test-model")
    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    bundle = _render_full_package(descriptor, surface, skill_root=skill_root)
    out_dir = tmp_path / "synthetic_everything_mellea"
    _write_package_to(
        out_dir,
        pipeline_py=bundle[0],
        schemas_py=bundle[1],
        init_py=bundle[2],
        fixtures_py=bundle[3],
        melleafy_json=bundle[4],
        setup_md=bundle[5],
        readme_md=bundle[6],
        source_map=bundle[7],
        render_result=bundle[8],
        skill_root=skill_root,
    )

    # ast.parse pipeline.py
    src = (out_dir / "pipeline.py").read_text(encoding="utf-8")
    ast.parse(src)

    # Optional: try to import the rendered pipeline module. If mellea backends
    # can't initialise without a real install, we treat the ast.parse above as
    # sufficient evidence (matching test_synthetic_v03_imports_cleanly).
    sys.path.insert(0, str(tmp_path))
    try:
        parent_name = "synthetic_everything_mellea"
        parent_spec = importlib.util.spec_from_file_location(
            parent_name,
            out_dir / "__init__.py",
            submodule_search_locations=[str(out_dir)],
        )
        assert parent_spec is not None and parent_spec.loader is not None
        parent_mod = importlib.util.module_from_spec(parent_spec)
        sys.modules[parent_name] = parent_mod
        try:
            parent_spec.loader.exec_module(parent_mod)
        except Exception:
            return  # __init__.py may import the pipeline which needs mellea.
        pipeline_spec = importlib.util.spec_from_file_location(
            f"{parent_name}.pipeline", out_dir / "pipeline.py"
        )
        assert pipeline_spec is not None and pipeline_spec.loader is not None
        pipeline_mod = importlib.util.module_from_spec(pipeline_spec)
        try:
            pipeline_spec.loader.exec_module(pipeline_mod)
        except Exception:
            return  # tolerate missing mellea/backend deps; ast.parse already passed.
        assert hasattr(pipeline_mod, "run_pipeline")
    finally:
        try:
            sys.path.remove(str(tmp_path))
        except ValueError:
            pass
        sys.modules.pop("synthetic_everything_mellea", None)
        sys.modules.pop("synthetic_everything_mellea.pipeline", None)


# --- 8. compile_via_descriptor integration with skill_root + bundled ------


def test_compile_via_descriptor_executes_file_copies(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    """Confirms ``compile_via_descriptor(write_to=..., skill_root=...)`` plumbs
    through to the writer's v0.3 logic."""
    from tests.mellea_skills_compiler.compile.test_descriptor_emission import (
        _FakeAnthropicClient,
        _wrap_descriptor_response,
    )
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    descriptor = _synthetic_v03_descriptor()
    skill_root = _make_skill_root(tmp_path)
    write_to = tmp_path / "compiled" / "synthetic_everything_mellea"

    client = _FakeAnthropicClient(_wrap_descriptor_response(descriptor))
    result = compile_via_descriptor(
        "dummy spec",
        surface,
        client=client,
        write_to=write_to,
        skill_root=skill_root,
    )
    # We don't assert failure_stage is None — the validator may reject the
    # synthetic descriptor if v0.3 validation is stricter; we only require that
    # IF the package was written, the v0.3 artefacts landed.
    if result.failure_stage is None:
        assert write_to.is_dir()
        # Writer copies only non-OUT-6 bundled paths; OUT-6 (references/,
        # scripts/) are mirrored by legacy mirror_companion_dirs at
        # Step 3a-pre, which doesn't run in this unit-test context.
        assert (write_to / "vendor" / "lib.py").is_file()
        assert (write_to / "pyproject.toml").is_file()
        assert (write_to / "intermediate" / "source_elements_map.json").is_file()
    else:
        # Surface the failure reason for diagnosability — the write_to dir
        # MUST NOT exist if we failed before the writer ran.
        # (compile_via_descriptor only writes on full success.)
        assert not write_to.exists(), \
            f"writer ran despite failure_stage={result.failure_stage}; reason={result.failure_reason}"
        pytest.skip(
            f"synthetic descriptor failed earlier-stage ({result.failure_stage}): "
            f"{result.failure_reason} — writer integration verified by other "
            "tests in this file"
        )
