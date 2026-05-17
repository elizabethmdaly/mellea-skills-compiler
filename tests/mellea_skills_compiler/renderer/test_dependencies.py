"""Per-disposition renderer tests for the v0.3 ``dependencies[]`` block.

Each test exercises one disposition kind in isolation. The shape of the
emitted Python is asserted by ``ast`` walking + ``ast.unparse`` + substring
checks (golden-file style without committing brittle whole-source goldens).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import RendererError
from mellea_skills_compiler.renderer.core import RenderContext
from mellea_skills_compiler.renderer.dependencies import (
    DependenciesArtefact,
    FileCopyOp,
    render_dependencies,
)


from tests.mellea_skills_compiler.conftest import SURFACE_PATH


@pytest.fixture(scope="module")
def surface() -> dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _ctx(surface: dict, descriptor: dict | None = None) -> RenderContext:
    return RenderContext(descriptor=descriptor or {}, surface=surface)


def _unparse(stmts: list[ast.stmt]) -> str:
    mod = ast.Module(body=stmts, type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod)


# --- delegate_to_runtime --------------------------------------------------


def test_delegate_to_runtime_adds_callable_kwarg(surface):
    desc = {
        "dependencies": [
            {
                "id": "search_fn",
                "kind": "tool",
                "disposition": "delegate_to_runtime",
                "signature": "(query: str) -> list[str]",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    assert len(art.pipeline_args) == 1
    assert art.pipeline_args[0].arg == "search_fn"
    assert art.pipeline_defaults == [None]  # required kwarg
    assert art.top_level_stmts == []
    # The annotation is Callable[[str], list[str]] — confirm by unparse.
    ann_src = ast.unparse(art.pipeline_args[0].annotation)
    assert "Callable" in ann_src
    assert "str" in ann_src
    assert "list[str]" in ann_src


# --- external_input -------------------------------------------------------


def test_external_input_adds_required_kwarg(surface):
    desc = {
        "dependencies": [
            {
                "id": "api_key",
                "kind": "secret",
                "disposition": "external_input",
                "signature": "() -> str",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    assert len(art.pipeline_args) == 1
    assert art.pipeline_args[0].arg == "api_key"
    assert art.pipeline_defaults == [None]
    assert "str" == ast.unparse(art.pipeline_args[0].annotation)


# --- stub -----------------------------------------------------------------


def test_stub_emits_sentinel_function(surface):
    desc = {
        "dependencies": [
            {
                "id": "scratchpad",
                "kind": "tool",
                "disposition": "stub",
                "signature": "(content: str) -> list[str]",
            }
        ]
    }
    ctx = _ctx(surface, desc)
    art = render_dependencies(desc, ctx)
    assert len(art.top_level_stmts) == 1
    src = _unparse(art.top_level_stmts)
    assert "def scratchpad(content: str) -> list[str]:" in src
    assert "return []" in src
    assert art.pipeline_args == []


def test_stub_returns_empty_string_for_str(surface):
    desc = {
        "dependencies": [
            {"id": "noop", "kind": "tool", "disposition": "stub",
             "signature": "() -> str"}
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert "return ''" in src


# --- real_impl ------------------------------------------------------------


def test_real_impl_emits_not_implemented_stub(surface):
    desc = {
        "dependencies": [
            {
                "id": "search",
                "kind": "tool",
                "disposition": "real_impl",
                "signature": "(query: str) -> list[str]",
                "rationale": "Calls the external search API.",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert "def search(query: str) -> list[str]:" in src
    assert "TODO: provide real implementation" in src
    assert "NotImplementedError" in src


# --- mock -----------------------------------------------------------------


def test_mock_returns_magicmock(surface):
    desc = {
        "dependencies": [
            {
                "id": "send",
                "kind": "tool",
                "disposition": "mock",
                "signature": "(msg: str) -> None",
            }
        ]
    }
    ctx = _ctx(surface, desc)
    art = render_dependencies(desc, ctx)
    src = _unparse(art.top_level_stmts)
    assert "def send(msg: str) -> None:" in src
    assert "MagicMock()" in src
    assert "unittest.mock" in ctx.imports
    assert "MagicMock" in ctx.imports["unittest.mock"]


# --- load_from_disk -------------------------------------------------------


def _write_pkg_with_pipeline(tmp_path: Path, src: str) -> Path:
    """Write a minimal package at tmp_path containing the given pipeline.py.

    Returns the package directory. The pipeline.py is prefixed with the
    ``from pathlib import Path`` import the renderer would emit so the file
    is syntactically valid for the lint's ast.parse.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    full = "from pathlib import Path\n" + src + "\n"
    (pkg / "pipeline.py").write_text(full)
    return pkg


def test_load_from_disk_emits_path_constant_and_helper(surface):
    desc = {
        "dependencies": [
            {
                "id": "owasp",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            }
        ]
    }
    ctx = _ctx(surface, desc)
    art = render_dependencies(desc, ctx)
    src = _unparse(art.top_level_stmts)
    # _PATH suffix distinguishes the constant from any same-id symbol the
    # LLM might emit later; per-segment "/" path-join is the exact AST shape
    # the bundled-asset-path-resolution lint accepts.
    assert "_owasp_PATH = Path(__file__).parent / 'references' / 'owasp.md'" in src
    assert "def _load_owasp() -> str" in src
    assert "_owasp_PATH.read_text(encoding='utf-8')" in src
    assert "pathlib" in ctx.imports


def test_load_from_disk_emits_path_file_parent(surface):
    """The emitted path resolution MUST be rooted at Path(__file__).parent
    (Rule OUT-6) — never via Path(repo_root), an injected function arg,
    os.getcwd(), or a module-level _PKG_DIR constant."""
    desc = {
        "dependencies": [
            {
                "id": "owasp",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert "Path(__file__).parent" in src
    # Forbidden shapes — defense in depth against accidental regressions.
    assert "Path(repo_root)" not in src
    assert "os.getcwd" not in src
    assert "_PKG_DIR" not in src


def test_load_from_disk_emits_const_and_loader(surface):
    """Both the module-level path constant AND the loader helper must be
    present in the artefact (RFC §3.1: load_from_disk renders as
    'Path constant + _load_<id>() helper')."""
    desc = {
        "dependencies": [
            {
                "id": "owasp",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    assert len(art.top_level_stmts) == 2
    const_stmt, helper_stmt = art.top_level_stmts
    assert isinstance(const_stmt, ast.Assign)
    assert isinstance(const_stmt.targets[0], ast.Name)
    assert const_stmt.targets[0].id == "_owasp_PATH"
    assert isinstance(helper_stmt, ast.FunctionDef)
    assert helper_stmt.name == "_load_owasp"


def test_load_from_disk_lint_passes(surface, tmp_path):
    """The rendered artefact must satisfy lint_bundled_asset_path_resolution
    (Rule OUT-6 / Rule 2.5-2). The lint, not just substring checks, is the
    contract — assert verdict directly."""
    from mellea_skills_compiler.compile.lints import lint_bundled_asset_path_resolution

    desc = {
        "dependencies": [
            {
                "id": "owasp",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    pkg = _write_pkg_with_pipeline(tmp_path, src)
    result = lint_bundled_asset_path_resolution(pkg)
    assert result.verdict == "pass", [f.message for f in result.failures]


def test_load_from_disk_lint_passes_with_multiple_deps(surface, tmp_path):
    """Multiple load_from_disk deps spread across references/ and scripts/
    must all render lint-clean."""
    from mellea_skills_compiler.compile.lints import lint_bundled_asset_path_resolution

    desc = {
        "dependencies": [
            {
                "id": "owasp",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/owasp.md",
            },
            {
                "id": "playbook",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/playbooks/xss.md",
            },
            {
                "id": "linter",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "scripts/lint.sh",
            },
            {
                "id": "logo",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "assets/logo.png",
            },
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    pkg = _write_pkg_with_pipeline(tmp_path, src)
    result = lint_bundled_asset_path_resolution(pkg)
    assert result.verdict == "pass", [f.message for f in result.failures]


def test_load_from_disk_nested_paths_split_per_segment(surface):
    """Nested paths (e.g. references/sub/file.md) must split into per-segment
    path-joins; collapsing into a single 'sub/file.md' constant works for the
    lint today but loses the canonical OUT-6 shape and is harder to read."""
    desc = {
        "dependencies": [
            {
                "id": "nested",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/sub/file.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert (
        "_nested_PATH = Path(__file__).parent / 'references' / 'sub' / 'file.md'"
        in src
    )


def test_load_from_disk_synthetic_against_failing_skill(surface, tmp_path):
    """Synthesise a descriptor matching the shape that produced the
    gdpr-breach-sentinel failure (three references/ load_from_disk deps) and
    confirm the renderer's output is lint-clean — a regression test for the
    specific failure that motivated this fix."""
    from mellea_skills_compiler.compile.lints import lint_bundled_asset_path_resolution

    desc = {
        "dependencies": [
            {
                "id": "enisa_methodology",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/enisa-methodology.md",
            },
            {
                "id": "edpb_cases",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/edpb-cases.md",
            },
            {
                "id": "templates",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "references/templates.md",
            },
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    pkg = _write_pkg_with_pipeline(tmp_path, src)
    result = lint_bundled_asset_path_resolution(pkg)
    assert result.verdict == "pass", [f.message for f in result.failures]


def test_load_from_disk_tolerates_messy_path(surface):
    """Defensively normalise leading/trailing/duplicate slashes and Windows
    backslashes — the schema validator doesn't reject these, and a single
    malformed descriptor path shouldn't crash the renderer or emit a broken
    AST. Output must still be lint-clean."""
    desc = {
        "dependencies": [
            {
                "id": "msg",
                "kind": "file",
                "disposition": "load_from_disk",
                "path": "/references//owasp.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert "_msg_PATH = Path(__file__).parent / 'references' / 'owasp.md'" in src


def test_load_from_disk_empty_path_after_normalisation_raises(surface):
    """A path that normalises to nothing (e.g. "/", "//") is a hard error;
    silently emitting Path(__file__).parent alone would lose the file
    reference and the resulting pipeline would crash at runtime."""
    desc = {
        "dependencies": [
            {"id": "x", "kind": "file", "disposition": "load_from_disk", "path": "/"}
        ]
    }
    with pytest.raises(RendererError, match="must not be empty"):
        render_dependencies(desc, _ctx(surface, desc))


def test_load_from_disk_requires_path(surface):
    desc = {
        "dependencies": [
            {"id": "x", "kind": "file", "disposition": "load_from_disk"}
        ]
    }
    with pytest.raises(RendererError, match="load_from_disk requires 'path'"):
        render_dependencies(desc, _ctx(surface, desc))


# --- bundle ---------------------------------------------------------------


def test_bundle_schedules_file_copy(surface):
    desc = {
        "dependencies": [
            {
                "id": "ref",
                "kind": "file",
                "disposition": "bundle",
                "path": "references/owasp.md",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    assert art.top_level_stmts == []
    assert art.pipeline_args == []
    assert len(art.file_copies) == 1
    fc = art.file_copies[0]
    assert isinstance(fc, FileCopyOp)
    assert fc.source == "references/owasp.md"
    assert fc.dest == "references/owasp.md"
    assert fc.is_package_data
    assert "references" in art.package_data_dirs


# --- remove ---------------------------------------------------------------


def test_remove_emits_nothing(surface):
    desc = {
        "dependencies": [
            {"id": "obsolete", "kind": "tool", "disposition": "remove"}
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    assert art.pipeline_args == []
    assert art.top_level_stmts == []
    assert art.file_copies == []


# --- error paths ----------------------------------------------------------


def test_unknown_disposition_raises(surface):
    desc = {
        "dependencies": [
            {"id": "x", "kind": "tool", "disposition": "summon_a_demon"}
        ]
    }
    with pytest.raises(RendererError, match="unknown disposition"):
        render_dependencies(desc, _ctx(surface, desc))


def test_delegate_without_signature_raises(surface):
    desc = {
        "dependencies": [
            {"id": "x", "kind": "tool", "disposition": "delegate_to_runtime"}
        ]
    }
    with pytest.raises(RendererError, match="delegate_to_runtime requires 'signature'"):
        render_dependencies(desc, _ctx(surface, desc))


def test_empty_dependencies_is_a_no_op(surface):
    art = render_dependencies({}, _ctx(surface))
    assert art == DependenciesArtefact()


def test_signature_with_default_param(surface):
    """The signature parser handles complex annotation strings."""
    desc = {
        "dependencies": [
            {
                "id": "fetch",
                "kind": "tool",
                "disposition": "stub",
                "signature": "(url: str, headers: dict[str, str]) -> dict[str, str]",
            }
        ]
    }
    art = render_dependencies(desc, _ctx(surface, desc))
    src = _unparse(art.top_level_stmts)
    assert "headers: dict[str, str]" in src
    assert "return {}" in src
