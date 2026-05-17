"""Tests for :mod:`mellea_skills_compiler.renderer.artefacts`.

One representative input per artefact emitter; emphasis on determinism,
required-field validation, and the descriptor → text contract.
"""

from __future__ import annotations

import ast
import json
import py_compile
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.artefacts import (
    canonical_descriptor_hash,
    render_fixtures,
    render_init,
    render_melleafy_json,
    render_readme_md,
    render_setup_md,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTRY_DESCRIPTOR = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "sentry-find-bugs.descriptor.json"
)


def _sentry() -> dict:
    return json.loads(SENTRY_DESCRIPTOR.read_text())


def _compile(src: str) -> Path:
    fd = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    fd.write(src)
    fd.close()
    py_compile.compile(fd.name, doraise=True)
    return Path(fd.name)


# --- canonical hash ---------------------------------------------------------


def test_canonical_hash_is_stable_across_key_order():
    a = {"x": 1, "y": [2, 3]}
    b = {"y": [2, 3], "x": 1}
    assert canonical_descriptor_hash(a) == canonical_descriptor_hash(b)


def test_canonical_hash_changes_when_content_changes():
    a = {"x": 1}
    b = {"x": 2}
    assert canonical_descriptor_hash(a) != canonical_descriptor_hash(b)


# --- __init__.py ------------------------------------------------------------


def test_render_init_reexports_run_pipeline():
    src = render_init(_sentry())
    _compile(src)
    tree = ast.parse(src)
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert any(
        imp.module == "pipeline"
        and imp.level == 1
        and any(a.name == "run_pipeline" for a in imp.names)
        for imp in imports
    )
    # __all__ exposes run_pipeline.
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
    all_assign = next(a for a in assigns if a.targets[0].id == "__all__")
    elts = [e.value for e in all_assign.value.elts]
    assert elts == ["run_pipeline"]


def test_render_init_requires_skill_name():
    with pytest.raises(RendererError, match="skill"):
        render_init({"skill": {}})


# --- fixtures.py ------------------------------------------------------------


def test_render_fixtures_emits_load_fixture_helper(tmp_path):
    src = render_fixtures(_sentry(), tmp_path / "fixtures")
    _compile(src)
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = [f.name for f in fns]
    assert "load_fixture" in names
    load = next(f for f in fns if f.name == "load_fixture")
    args = [a.arg for a in load.args.args]
    assert args == ["name"]


def test_render_fixtures_load_fixture_round_trip(tmp_path):
    """Execute the rendered module and exercise load_fixture against a JSON file."""
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    payload = {"diff": "hello"}
    (fixtures_dir / "smoke.json").write_text(json.dumps(payload))

    src = render_fixtures(_sentry(), fixtures_dir)
    # Write the rendered fixtures module next to fixtures_dir so the
    # ``Path(__file__).parent / "fixtures"`` resolution finds it.
    pkg_file = tmp_path / "fixtures_mod.py"
    pkg_file.write_text(src)

    ns: dict[str, object] = {"__file__": str(pkg_file)}
    exec(compile(src, str(pkg_file), "exec"), ns)
    load_fixture = ns["load_fixture"]
    assert load_fixture("smoke") == payload


def test_render_fixtures_load_fixture_missing_raises(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    src = render_fixtures(_sentry(), fixtures_dir)
    pkg_file = tmp_path / "fixtures_mod.py"
    pkg_file.write_text(src)
    ns: dict[str, object] = {"__file__": str(pkg_file)}
    exec(compile(src, str(pkg_file), "exec"), ns)
    with pytest.raises(FileNotFoundError, match="fixture not found"):
        ns["load_fixture"]("does-not-exist")  # type: ignore[operator]


# --- melleafy.json ----------------------------------------------------------


def test_render_melleafy_json_has_required_fields():
    src = render_melleafy_json(
        _sentry(),
        mellea_version="0.5.0",
        renderer_version="2.0.0",
        compiled_at="2026-05-16T12:00:00Z",
    )
    payload = json.loads(src)
    assert set(payload) == {
        "compiled_at",
        "descriptor_hash",
        "mellea_version",
        "renderer_version",
    }
    assert payload["mellea_version"] == "0.5.0"
    assert payload["renderer_version"] == "2.0.0"
    assert payload["compiled_at"] == "2026-05-16T12:00:00Z"
    # SHA-256 hex digest is 64 chars.
    assert len(payload["descriptor_hash"]) == 64
    int(payload["descriptor_hash"], 16)  # parses as hex


def test_render_melleafy_json_is_byte_identical_on_repeat():
    desc = _sentry()
    a = render_melleafy_json(
        desc, "0.5.0", "2.0.0", "2026-05-16T12:00:00Z"
    )
    b = render_melleafy_json(
        desc, "0.5.0", "2.0.0", "2026-05-16T12:00:00Z"
    )
    assert a == b


def test_render_melleafy_json_requires_timestamp():
    with pytest.raises(RendererError, match="compiled_at"):
        render_melleafy_json(_sentry(), "0.5.0", "2.0.0", "")


# --- SETUP.md ---------------------------------------------------------------


def test_render_setup_md_mentions_skill_and_env_vars():
    src = render_setup_md(_sentry())
    assert "find-bugs" in src
    assert "MODEL_ID" in src
    assert "OPENAI_API_KEY" in src
    # Module-import line uses snake_case form.
    assert "from find_bugs import run_pipeline" in src


def test_render_setup_md_requires_skill_name():
    with pytest.raises(RendererError, match="skill"):
        render_setup_md({})


# --- README.md --------------------------------------------------------------


def test_render_readme_md_includes_call_signature():
    src = render_readme_md(_sentry())
    assert "find-bugs" in src
    # Sentry has `diff: str` as the only input and `FindingsReport` as the
    # output.
    assert "run_pipeline(diff: str) -> FindingsReport" in src
    assert "diff=..." in src
    # Inputs table renders.
    assert "| `diff` |" in src


def test_render_readme_md_with_no_inputs():
    desc = {
        "skill": {"name": "noop", "classification": "DSL"},
        "inputs": [],
        "outputs": [],
    }
    src = render_readme_md(desc)
    assert "run_pipeline() -> Any" in src
    assert "(no declared inputs)" in src


def test_render_readme_md_handles_pydantic_model_kind():
    desc = {
        "skill": {"name": "thing"},
        "inputs": [
            {
                "name": "doc",
                "schema": {"kind": "pydantic_model", "ref": "#/schemas/Document"},
            }
        ],
        "outputs": [
            {
                "name": "out",
                "schema": {"kind": "model_ref", "ref": "#/schemas/Result"},
            }
        ],
    }
    src = render_readme_md(desc)
    assert "run_pipeline(doc: Document) -> Result" in src


def test_render_readme_md_includes_classification_when_present():
    src = render_readme_md(_sentry())
    assert "Classification" in src
    assert "DSL" in src


def test_render_readme_md_omits_classification_when_absent():
    desc = {
        "skill": {"name": "simple"},
        "inputs": [{"name": "x", "schema": {"kind": "str"}}],
        "outputs": [],
    }
    src = render_readme_md(desc)
    assert "Classification" not in src
