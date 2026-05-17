"""End-to-end renderer tests: descriptor JSON -> pipeline.py source."""

from __future__ import annotations

import ast
import json
import py_compile
from pathlib import Path

import pytest

from mellea_skills_compiler.renderer import (
    RenderResult,
    RendererError,
    SourceMap,
    render_descriptor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTRY_DESC = REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "descriptors" / "sentry-find-bugs.descriptor.json"
SURFACE = REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(scope="module")
def real_surface() -> dict:
    return json.loads(SURFACE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentry_descriptor() -> dict:
    return json.loads(SENTRY_DESC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentry_result(sentry_descriptor: dict, real_surface: dict) -> RenderResult:
    return render_descriptor(sentry_descriptor, real_surface)


# --- Output shape ---------------------------------------------------------


def test_render_returns_render_result(sentry_result: RenderResult) -> None:
    assert isinstance(sentry_result, RenderResult)
    assert isinstance(sentry_result.pipeline_py, str)
    assert isinstance(sentry_result.source_map, SourceMap)


def test_rendered_pipeline_parses(sentry_result: RenderResult) -> None:
    """The rendered Python source must parse."""
    ast.parse(sentry_result.pipeline_py)


def test_rendered_pipeline_py_compiles(sentry_result: RenderResult, tmp_path: Path) -> None:
    """The rendered Python source must `py_compile` cleanly."""
    p = tmp_path / "pipeline.py"
    p.write_text(sentry_result.pipeline_py, encoding="utf-8")
    py_compile.compile(str(p), doraise=True)


def test_rendered_pipeline_imports_with_env(
    sentry_descriptor: dict, sentry_result: RenderResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end import-time check: combine P2.A's pipeline.py + P2.C's schemas.py
    into a package and import it. Must expose `run_pipeline(diff: str) -> FindingsReport`.

    Sets MODEL_ID/OPENAI_API_KEY in env so OpenAIBackend construction at module
    import time doesn't blow up on missing credentials. We do NOT call
    run_pipeline (that would require a live model); we only verify the
    function is importable and has the expected signature.
    """
    import importlib.util
    import inspect

    # P2.C's schemas emitter — emit schemas.py alongside our pipeline.py.
    from mellea_skills_compiler.renderer.schemas import render_schemas

    pkg = tmp_path / "rendered_skill"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "schemas.py").write_text(
        render_schemas(sentry_descriptor.get("schemas", {})), encoding="utf-8"
    )
    (pkg / "pipeline.py").write_text(sentry_result.pipeline_py, encoding="utf-8")

    monkeypatch.setenv("MODEL_ID", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    import sys as _sys

    # Load the package as a module via importlib.
    pkg_init_spec = importlib.util.spec_from_file_location(
        "rendered_skill_under_test", pkg / "__init__.py", submodule_search_locations=[str(pkg)]
    )
    assert pkg_init_spec and pkg_init_spec.loader
    pkg_mod = importlib.util.module_from_spec(pkg_init_spec)
    _sys.modules["rendered_skill_under_test"] = pkg_mod
    try:
        pkg_init_spec.loader.exec_module(pkg_mod)

        pipe_spec = importlib.util.spec_from_file_location(
            "rendered_skill_under_test.pipeline", pkg / "pipeline.py"
        )
        assert pipe_spec and pipe_spec.loader
        pipe_mod = importlib.util.module_from_spec(pipe_spec)
        _sys.modules["rendered_skill_under_test.pipeline"] = pipe_mod
        pipe_spec.loader.exec_module(pipe_mod)

        fn = getattr(pipe_mod, "run_pipeline", None)
        assert callable(fn), "run_pipeline must be defined in the rendered pipeline.py"
        sig = inspect.signature(fn)
        assert list(sig.parameters) == ["diff"]
        assert sig.parameters["diff"].annotation == "str"  # PEP 563 stringised
        assert sig.return_annotation == "FindingsReport"
    finally:
        _sys.modules.pop("rendered_skill_under_test.pipeline", None)
        _sys.modules.pop("rendered_skill_under_test", None)


def test_rendered_exposes_run_pipeline_function(sentry_result: RenderResult) -> None:
    """A `def run_pipeline(diff: str) -> FindingsReport` must be present."""
    src = sentry_result.pipeline_py
    assert "def run_pipeline(diff: str) -> FindingsReport:" in src


def test_rendered_imports_session_and_backend(sentry_result: RenderResult) -> None:
    src = sentry_result.pipeline_py
    assert "from mellea.stdlib.session import MelleaSession" in src
    assert "from mellea.backends.openai import OpenAIBackend" in src
    assert "from mellea.stdlib.requirements import Requirement" in src


def test_rendered_imports_schemas_from_relative_module(sentry_result: RenderResult) -> None:
    src = sentry_result.pipeline_py
    assert "from .schemas import" in src
    assert "FindingsReport" in src
    assert "ModifiedFiles" in src


def test_rendered_emits_format_schema_parse_wrap(sentry_result: RenderResult) -> None:
    """Every call with `format=Schema` is wrapped in `Schema.model_validate_json(...).value`."""
    src = sentry_result.pipeline_py
    assert "ModifiedFiles.model_validate_json(session.instruct(" in src
    assert ".value)" in src
    assert "FindingsReport.model_validate_json(session.instruct(" in src


def test_rendered_state_assignments_module_level(sentry_result: RenderResult) -> None:
    src = sentry_result.pipeline_py
    # State must appear before the function definition.
    fn_idx = src.find("def run_pipeline")
    assert "backend = OpenAIBackend(model_id=os.environ['MODEL_ID'])" in src[:fn_idx]
    assert "session = MelleaSession(backend=backend)" in src[:fn_idx]


def test_rendered_pipeline_returns_audit(sentry_result: RenderResult) -> None:
    """`audit` is the node that captures `findings` — the return must reference it."""
    src = sentry_result.pipeline_py
    assert "return audit" in src


# --- Source-map round-trip -----------------------------------------------


def test_source_map_lookup_returns_pipeline_node_id(sentry_result: RenderResult) -> None:
    """Pick a line inside `run_pipeline` body; lookup must return a descriptor id."""
    src = sentry_result.pipeline_py
    lines = src.splitlines()
    # Find a line containing the first call wrap.
    target_line_no = None
    for idx, line in enumerate(lines, start=1):
        if "extract_files = ModifiedFiles.model_validate_json" in line:
            target_line_no = idx
            break
    assert target_line_no is not None
    assert sentry_result.source_map.lookup(target_line_no) == "extract_files"


def test_source_map_lookup_inside_for_returns_inner_node_id(sentry_result: RenderResult) -> None:
    """A line inside the map's `for` body that's the attack_surface_step should
    resolve to that node id, NOT the outer composition's id."""
    src = sentry_result.pipeline_py
    lines = src.splitlines()
    for idx, line in enumerate(lines, start=1):
        if "attack_surface_step =" in line:
            assert sentry_result.source_map.lookup(idx) == "attack_surface_step"
            return
    pytest.fail("expected line containing 'attack_surface_step =' not found in rendered source")


def test_source_map_lookup_unknown_line_returns_none(sentry_result: RenderResult) -> None:
    """A line far past the file length yields None."""
    assert sentry_result.source_map.lookup(100_000) is None


def test_source_map_to_json_is_listy(sentry_result: RenderResult) -> None:
    payload = sentry_result.source_map.to_json()
    assert isinstance(payload, list)
    assert all("descriptor_node_id" in e for e in payload)
    # `audit`, `extract_files`, and the inner step ids should all be represented.
    ids = {e["descriptor_node_id"] for e in payload}
    assert {"extract_files", "audit", "attack_surface_step"} <= ids


# --- Error paths ---------------------------------------------------------


def test_unknown_symbol_raises_with_descriptor_path(real_surface: dict) -> None:
    bad = {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "broken", "classification": "DSL"},
        "inputs": [{"name": "x", "schema": {"kind": "str"}}],
        "outputs": [],
        "schemas": {},
        "state": [],
        "pipeline": [
            {
                "id": "broken_call",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.not_a_real_method",
                "bound_to": {"ref": "session"},
                "args": {},
            }
        ],
    }
    with pytest.raises(RendererError) as exc:
        render_descriptor(bad, real_surface)
    msg = str(exc.value)
    assert "not in surface" in msg or "not found" in msg
    assert "pipeline[0]" in msg
    assert "mellea.stdlib.session.MelleaSession.not_a_real_method" in msg


def test_unknown_composition_operator_raises(real_surface: dict) -> None:
    bad = {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "broken", "classification": "DSL"},
        "inputs": [],
        "outputs": [],
        "schemas": {},
        "state": [],
        "pipeline": [
            {
                "id": "weird",
                "kind": "composition",
                "operator": "no_such_operator",
                "body": [],
            }
        ],
    }
    with pytest.raises(RendererError, match="unknown composition operator"):
        render_descriptor(bad, real_surface)


def test_missing_kind_raises_with_path(real_surface: dict) -> None:
    bad = {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "broken", "classification": "DSL"},
        "inputs": [],
        "outputs": [],
        "schemas": {},
        "state": [],
        "pipeline": [{"id": "x"}],
    }
    with pytest.raises(RendererError, match="pipeline\\[0\\]"):
        render_descriptor(bad, real_surface)


# --- Trivial sequential-only render --------------------------------------


def _trivial_descriptor() -> dict:
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "trivial", "classification": "DSL"},
        "inputs": [{"name": "q", "schema": {"kind": "str"}}],
        "outputs": [{"name": "msg", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"env": "MODEL_ID"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "say_hi",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.achat",
                "bound_to": {"ref": "session"},
                "args": {"content": {"template": "Echo: {q}"}},
                "captures": {"result": "#/outputs/msg"},
            }
        ],
    }


def test_trivial_render_sequential_only(real_surface: dict) -> None:
    """End-to-end on a sequential-only descriptor (no operator needed beyond implicit)."""
    res = render_descriptor(_trivial_descriptor(), real_surface)
    src = res.pipeline_py
    assert "def run_pipeline(q: str)" in src
    assert "say_hi = session.achat(content=f'Echo: {q}')" in src
    assert "return say_hi" in src
    ast.parse(src)


def test_no_per_symbol_logic_in_codebase() -> None:
    """Architectural assertion (plan §5.4): no `MelleaSession` literal in source."""
    src_root = REPO_ROOT / "src" / "mellea_skills_compiler" / "renderer"
    for path in src_root.rglob("*.py"):
        # Skip __pycache__ and tests (we don't search them anyway).
        text = path.read_text(encoding="utf-8")
        assert "MelleaSession" not in text, (
            f"renderer source must contain no per-symbol literals; "
            f"`MelleaSession` found in {path.relative_to(REPO_ROOT)}"
        )


def test_render_result_is_dataclass_with_named_fields() -> None:
    """Defensive: RenderResult exposes the fields downstream callers depend on."""
    res = render_descriptor(_trivial_descriptor(), json.loads(SURFACE.read_text(encoding="utf-8")))
    assert hasattr(res, "pipeline_py")
    assert hasattr(res, "source_map")
