"""Tests for ``@validate_call`` emission on exported entry-point functions.

The renderer decorates the top-level ``run_pipeline`` with Pydantic's
``@validate_call(config={"arbitrary_types_allowed": True})`` so that callers
can pass plain dicts where Pydantic models are typed and have them coerced
automatically at call time. This closes a class of bugs surfaced by
fixture-driven runs where ``session_state={"turn_number": 1, ...}`` (a dict)
reaches downstream code that calls ``session_state.model_copy(...)`` and
crashes with ``'dict' object has no attribute 'model_copy'``.

These tests cover:
- The decorator appears on the rendered ``run_pipeline``
- The ``from pydantic import validate_call`` import is emitted
- Internal helper functions (emitted by composition operators) are NOT
  decorated
- The decorator is emitted on ``async def`` entry points too
- Re-render is byte-stable (no duplicate decorator / import)
- End-to-end: a rendered package coerces a dict into the Pydantic model at
  call time
- v0.1 descriptors also get the decorator (the contract is the same)
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import RenderResult, render_descriptor
from mellea_skills_compiler.renderer.schemas import render_schemas

from tests.mellea_skills_compiler.conftest import (
    DESCRIPTOR_FIXTURE_DIR,
    SURFACE_PATH,
)

SENTRY_DESC = DESCRIPTOR_FIXTURE_DIR / "sentry-find-bugs.descriptor.json"


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentry_descriptor() -> dict:
    return json.loads(SENTRY_DESC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentry_result(sentry_descriptor: dict, surface: dict) -> RenderResult:
    return render_descriptor(sentry_descriptor, surface)


def _trivial_descriptor(*, descriptor_version: str = "0.1") -> dict[str, Any]:
    """Sequential-only descriptor used by several tests below."""
    return {
        "descriptor_version": descriptor_version,
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


def _async_descriptor() -> dict[str, Any]:
    """A v0.3 descriptor with ``skill.async = True``."""
    d = _trivial_descriptor(descriptor_version="0.3")
    d["skill"]["async"] = True
    # MelleaSession.achat is async-friendly per surface.
    return d


def _descriptor_with_parallel_helpers() -> dict[str, Any]:
    """A descriptor that emits internal helper functions via the ``parallel`` operator."""
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "with_helpers", "classification": "DSL"},
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
                "id": "fan_out",
                "kind": "composition",
                "operator": "parallel",
                "collect": "list",
                "branches": [
                    [
                        {
                            "id": "a_call",
                            "kind": "call",
                            "symbol": "mellea.stdlib.session.MelleaSession.achat",
                            "bound_to": {"ref": "session"},
                            "args": {"content": {"template": "A: {q}"}},
                        }
                    ],
                    [
                        {
                            "id": "b_call",
                            "kind": "call",
                            "symbol": "mellea.stdlib.session.MelleaSession.achat",
                            "bound_to": {"ref": "session"},
                            "args": {"content": {"template": "B: {q}"}},
                        }
                    ],
                ],
                "captures": {"result": "#/outputs/msg"},
            },
        ],
    }


# --- Decorator emission ---------------------------------------------------


def test_run_pipeline_emits_validate_call_decorator(sentry_result: RenderResult) -> None:
    """The rendered AST has ``@validate_call(...)`` on the ``run_pipeline`` FunctionDef."""
    tree = ast.parse(sentry_result.pipeline_py)
    fns = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "run_pipeline"
    ]
    assert len(fns) == 1, "exactly one run_pipeline must be emitted"
    decorators = fns[0].decorator_list
    assert len(decorators) == 1, f"expected 1 decorator, got {len(decorators)}"

    deco = decorators[0]
    # Shape: ``validate_call(config={'arbitrary_types_allowed': True})``
    assert isinstance(deco, ast.Call), "decorator must be a Call expression"
    assert isinstance(deco.func, ast.Name) and deco.func.id == "validate_call"
    assert not deco.args, "decorator takes no positional args"
    assert len(deco.keywords) == 1
    kw = deco.keywords[0]
    assert kw.arg == "config"
    assert isinstance(kw.value, ast.Dict)
    cfg = {
        k.value: v.value
        for k, v in zip(kw.value.keys, kw.value.values)
        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
    }
    assert cfg == {"arbitrary_types_allowed": True}


def test_run_pipeline_emits_validate_call_import(sentry_result: RenderResult) -> None:
    """``from pydantic import validate_call`` is at the top of the rendered module."""
    src = sentry_result.pipeline_py
    assert "from pydantic import validate_call" in src
    # And it must appear *before* the run_pipeline definition.
    import_idx = src.find("from pydantic import validate_call")
    fn_idx = src.find("def run_pipeline")
    assert import_idx >= 0 and fn_idx >= 0
    assert import_idx < fn_idx


def test_internal_helpers_not_decorated(surface: dict) -> None:
    """Helper functions emitted by composition operators are NOT decorated."""
    descriptor = _descriptor_with_parallel_helpers()
    result = render_descriptor(descriptor, surface)
    tree = ast.parse(result.pipeline_py)

    # All FunctionDef / AsyncFunctionDef nodes in the rendered module.
    fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # Sanity: at least one helper (parallel emits ``_fan_out_branch_*``).
    helpers = [f for f in fns if f.name != "run_pipeline"]
    assert helpers, "expected at least one helper from the parallel operator"
    for helper in helpers:
        assert helper.decorator_list == [], (
            f"internal helper {helper.name!r} must not be decorated; "
            f"got {[ast.unparse(d) for d in helper.decorator_list]}"
        )

    # And run_pipeline IS decorated.
    rp = [f for f in fns if f.name == "run_pipeline"]
    assert len(rp) == 1 and len(rp[0].decorator_list) == 1


def test_async_pipeline_emits_validate_call(surface: dict) -> None:
    """``skill.async == True`` → decorator is emitted on the ``async def``."""
    result = render_descriptor(_async_descriptor(), surface)
    tree = ast.parse(result.pipeline_py)
    fns = [
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_pipeline"
    ]
    assert len(fns) == 1, "expected an async def run_pipeline"
    assert len(fns[0].decorator_list) == 1
    deco = fns[0].decorator_list[0]
    assert isinstance(deco, ast.Call)
    assert isinstance(deco.func, ast.Name) and deco.func.id == "validate_call"
    # And the import is present.
    assert "from pydantic import validate_call" in result.pipeline_py


def test_validate_call_idempotent_on_rerender(surface: dict) -> None:
    """Re-render produces byte-identical output (no duplicate decorator/import)."""
    descriptor = _trivial_descriptor()
    a = render_descriptor(descriptor, surface).pipeline_py
    b = render_descriptor(descriptor, surface).pipeline_py
    assert a == b, "re-render must be byte-stable"
    # Exactly one occurrence of the import + decorator pattern.
    assert a.count("from pydantic import validate_call") == 1
    assert a.count("@validate_call(") == 1


def test_v01_descriptor_also_decorated(surface: dict) -> None:
    """v0.1 descriptors get the decorator too — the contract is the same.

    Decision: we apply the decorator unconditionally rather than gating on
    ``descriptor_version >= "0.3"``. The golden integration tests for v0.1
    descriptors (sentry-find-bugs, security-review, security-engineer) do
    not assert byte stability against an external snapshot; they only smoke
    test that the rendered source parses, py_compiles, imports under the
    dummy backend env, and exposes a ``run_pipeline`` with the declared
    signature. The decorator is invisible to all of those checks.
    """
    descriptor = _trivial_descriptor(descriptor_version="0.1")
    assert descriptor["descriptor_version"] == "0.1"
    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    assert "from pydantic import validate_call" in src
    assert "@validate_call(config={'arbitrary_types_allowed': True})" in src


# --- End-to-end coercion -------------------------------------------------


class _StubSession:
    """Records ``.achat(...)`` calls and returns a sentinel result."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def achat(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "ok"


def test_dict_input_coercion_end_to_end(
    surface: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Render a descriptor with a Pydantic-typed input; call with a dict; expect coercion.

    The descriptor declares an input typed against a schema; the rendered
    ``run_pipeline`` is decorated with ``@validate_call``. Calling
    ``run_pipeline(payload={"text": "hi"})`` should be coerced to the
    Pydantic model at call time — i.e., the value visible inside the
    function body is a model instance, not a dict.
    """
    descriptor = {
        "descriptor_version": "0.3",
        "mellea_version": "0.5.0",
        "skill": {"name": "coerce_check", "classification": "DSL"},
        "schemas": {
            "Payload": {
                "kind": "model",
                "fields": {"text": {"type": "str"}},
            }
        },
        "inputs": [{"name": "payload", "schema": {"ref": "#/schemas/Payload"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
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
                "id": "echo",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.achat",
                "bound_to": {"ref": "session"},
                "args": {"content": {"ref": "payload", "select": "text"}},
                "captures": {"result": "#/outputs/result"},
            }
        ],
    }
    result = render_descriptor(descriptor, surface)
    schemas_src = render_schemas(descriptor["schemas"])

    pkg = tmp_path / "coerce_skill"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "schemas.py").write_text(schemas_src, encoding="utf-8")
    (pkg / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")

    monkeypatch.setenv("MODEL_ID", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    unique = f"coerce_test_{abs(id(descriptor)) % 1_000_000}"
    pkg_spec = importlib.util.spec_from_file_location(
        unique,
        pkg / "__init__.py",
        submodule_search_locations=[str(pkg)],
    )
    assert pkg_spec and pkg_spec.loader
    pkg_mod = importlib.util.module_from_spec(pkg_spec)
    sys.modules[unique] = pkg_mod
    try:
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

        # Patch the module-level session with our stub so we can call
        # run_pipeline without a real model round-trip. Then call with a
        # dict — validate_call should coerce it to a Payload instance.
        stub = _StubSession()
        pipe_mod.session = stub  # type: ignore[attr-defined]

        # Call with a dict; @validate_call must coerce it to the Pydantic model.
        out = pipe_mod.run_pipeline(payload={"text": "hello world"})
        assert out == "ok"
        # The captured ``content`` should resolve to the dict's ``text``
        # field via attribute access — proving the dict was coerced into a
        # model (dicts would have raised AttributeError on ``.text``).
        assert stub.calls, "session.achat must have been invoked"
        assert stub.calls[0]["content"] == "hello world"
    finally:
        sys.modules.pop(f"{unique}.pipeline", None)
        sys.modules.pop(f"{unique}.schemas", None)
        sys.modules.pop(unique, None)
