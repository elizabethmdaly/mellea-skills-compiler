"""Shared fixtures for the adversarial operator suite.

Each adversarial JSON fixture exercises exactly one composition operator (or
one operator combination). The :func:`render_fixture` factory loads, renders,
py_compiles, and imports the rendered package — returning the same
:class:`RenderedPackage` shape used by the golden suite for assertion
consistency.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import RenderResult, render_descriptor
from mellea_skills_compiler.renderer.schemas import render_schemas

from tests.mellea_skills_compiler.conftest import SURFACE_PATH

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class RenderedPackage:
    pkg_dir: Path
    pipeline_module: Any
    schemas_module: Any
    render_result: RenderResult
    schemas_src: str
    pipeline_src: str
    descriptor: dict


@pytest.fixture(scope="session")
def load_surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _load_fixture_descriptor(fixture_name: str) -> dict:
    path = FIXTURES_DIR / f"{fixture_name}.descriptor.json"
    if not path.is_file():
        raise FileNotFoundError(f"adversarial fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def render_fixture(
    load_surface: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Return ``render(name)`` -> :class:`RenderedPackage` for an adversarial fixture."""

    def _do(fixture_name: str) -> RenderedPackage:
        descriptor = _load_fixture_descriptor(fixture_name)
        result = render_descriptor(descriptor, load_surface)
        schemas_src = render_schemas(descriptor.get("schemas", {}))

        pkg = tmp_path / "rendered_skill"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "schemas.py").write_text(schemas_src, encoding="utf-8")
        (pkg / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")

        import py_compile

        py_compile.compile(str(pkg / "pipeline.py"), doraise=True)
        py_compile.compile(str(pkg / "schemas.py"), doraise=True)

        monkeypatch.setenv("MODEL_ID", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

        unique = f"adv_{fixture_name.replace('-', '_')}_{abs(id(descriptor)) % 1_000_000}"
        pkg_spec = importlib.util.spec_from_file_location(
            unique,
            pkg / "__init__.py",
            submodule_search_locations=[str(pkg)],
        )
        assert pkg_spec and pkg_spec.loader
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        sys.modules[unique] = pkg_mod
        pkg_spec.loader.exec_module(pkg_mod)

        schemas_spec = importlib.util.spec_from_file_location(
            f"{unique}.schemas", pkg / "schemas.py"
        )
        assert schemas_spec and schemas_spec.loader
        schemas_mod = importlib.util.module_from_spec(schemas_spec)
        sys.modules[f"{unique}.schemas"] = schemas_mod
        schemas_spec.loader.exec_module(schemas_mod)

        pipe_spec = importlib.util.spec_from_file_location(
            f"{unique}.pipeline", pkg / "pipeline.py"
        )
        assert pipe_spec and pipe_spec.loader
        pipe_mod = importlib.util.module_from_spec(pipe_spec)
        sys.modules[f"{unique}.pipeline"] = pipe_mod
        pipe_spec.loader.exec_module(pipe_mod)

        return RenderedPackage(
            pkg_dir=pkg,
            pipeline_module=pipe_mod,
            schemas_module=schemas_mod,
            render_result=result,
            schemas_src=schemas_src,
            pipeline_src=result.pipeline_py,
            descriptor=descriptor,
        )

    return _do
