"""Shared fixtures + helpers for the Phase 2.E golden-file integration suite.

The three hand-authored Phase 0.2 descriptors (sentry-find-bugs, security-review,
security-engineer) are exercised end-to-end here: render -> compile -> import
-> signature assertion. Each test module pulls in :func:`render_and_smoke`
which packages the rendered ``pipeline.py`` + ``schemas.py`` into a tmp_path
directory and imports it under the dummy backend env (MODEL_ID / OPENAI_API_KEY
set to non-secret stubs so ``OpenAIBackend(...)`` constructs cleanly at
module-import time).

We intentionally do NOT invoke ``run_pipeline`` — that would require a live
model. The assertions cover render success, ``py_compile`` of pipeline.py,
import under the dummy backend, and the ``run_pipeline`` signature matching
the descriptor's declared output schema.
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

from tests.mellea_skills_compiler.conftest import (
    DESCRIPTOR_FIXTURE_DIR as DESCRIPTORS_DIR,
    SURFACE_PATH,
)


@dataclass
class RenderedPackage:
    """Bundle returned by :func:`render_and_smoke` for downstream assertions."""

    pkg_dir: Path
    pipeline_module: Any
    schemas_module: Any
    render_result: RenderResult
    schemas_src: str
    pipeline_src: str


@pytest.fixture(scope="session")
def load_surface() -> dict:
    """The introspected Mellea 0.5.0 surface — shared across the whole session."""
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def load_descriptor() -> Any:
    """Factory: ``load_descriptor("sentry-find-bugs")`` returns the parsed JSON."""

    def _load(name: str) -> dict:
        path = DESCRIPTORS_DIR / f"{name}.descriptor.json"
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def render_and_smoke(
    load_surface: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Render ``descriptor``, py_compile pipeline.py, import the package.

    Returns a :class:`RenderedPackage` carrying the imported modules + the
    raw rendered sources. The dummy backend env is set on the supplied
    ``monkeypatch`` so the rendered module-level ``OpenAIBackend(...)`` call
    does not blow up on missing credentials when we import it.

    A unique module name is generated per call (using ``id(descriptor)`` and
    the test's tmp_path basename) so multiple golden tests in the same
    process do not collide in ``sys.modules``.
    """

    def _do_render(descriptor: dict) -> RenderedPackage:
        result = render_descriptor(descriptor, load_surface)
        schemas_src = render_schemas(descriptor.get("schemas", {}))

        pkg = tmp_path / "rendered_skill"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "schemas.py").write_text(schemas_src, encoding="utf-8")
        (pkg / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")

        # py_compile separately — gives a clearer error than import would.
        import py_compile

        py_compile.compile(str(pkg / "pipeline.py"), doraise=True)
        py_compile.compile(str(pkg / "schemas.py"), doraise=True)

        monkeypatch.setenv("MODEL_ID", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

        # Unique module name per invocation so re-renders in the same session
        # don't collide.
        unique = f"golden_{tmp_path.name}_{abs(id(descriptor)) % 1_000_000}"

        pkg_spec = importlib.util.spec_from_file_location(
            unique,
            pkg / "__init__.py",
            submodule_search_locations=[str(pkg)],
        )
        assert pkg_spec and pkg_spec.loader
        pkg_mod = importlib.util.module_from_spec(pkg_spec)
        sys.modules[unique] = pkg_mod
        pkg_spec.loader.exec_module(pkg_mod)

        # Import schemas first so pipeline can resolve `.schemas` imports.
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
        )

    return _do_render
