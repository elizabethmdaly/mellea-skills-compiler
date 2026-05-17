"""Tests for the composition-operator registry (plan §4.5)."""

from __future__ import annotations

import pytest

# Importing the operators package self-registers all 5.
import mellea_skills_compiler.renderer.operators as _operators  # noqa: F401
from mellea_skills_compiler.renderer import (
    OperatorRenderer,
    RendererError,
    get_operator,
    register_operator,
    registered_operator_names,
)
from mellea_skills_compiler.renderer.operators import (
    BranchRenderer,
    MapRenderer,
    ParallelRenderer,
    RetryRenderer,
    SequentialRenderer,
)


EXPECTED_OPERATORS = {
    "sequential": SequentialRenderer,
    "map": MapRenderer,
    "branch": BranchRenderer,
    "parallel": ParallelRenderer,
    "retry_with_feedback": RetryRenderer,
}


def test_all_five_registered() -> None:
    assert set(registered_operator_names()) >= set(EXPECTED_OPERATORS)


@pytest.mark.parametrize(("name", "cls"), list(EXPECTED_OPERATORS.items()))
def test_lookup_by_name_returns_right_class(name: str, cls: type) -> None:
    renderer = get_operator(name)
    assert isinstance(renderer, cls)
    assert renderer.operator_name == name


def test_unknown_operator_raises_renderer_error() -> None:
    with pytest.raises(RendererError) as exc:
        get_operator("definitely_not_a_real_operator")
    assert "definitely_not_a_real_operator" in str(exc.value)


def test_registry_idempotent_for_same_instance() -> None:
    # Calling register_operator twice with the *same instance* must be a no-op.
    # (Importlib.reload would create new instances and is correctly rejected
    # as a name collision — that's tested by test_registry_rejects_different_instance.)
    existing = get_operator("sequential")
    register_operator(existing)  # must not raise
    assert get_operator("sequential") is existing


def test_registry_rejects_different_instance_under_existing_name() -> None:
    class _Conflict(OperatorRenderer):
        operator_name = "map"

        def render(self, node, ctx, render_body):  # pragma: no cover - never called
            return []

    with pytest.raises(RendererError):
        register_operator(_Conflict())


def test_registry_rejects_empty_name() -> None:
    class _Empty(OperatorRenderer):
        operator_name = ""

        def render(self, node, ctx, render_body):  # pragma: no cover - never called
            return []

    with pytest.raises(RendererError):
        register_operator(_Empty())


def test_operator_renderer_is_abstract() -> None:
    with pytest.raises(TypeError):
        OperatorRenderer()  # type: ignore[abstract]
