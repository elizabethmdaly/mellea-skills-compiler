"""Regression tests for the typed-exception dispatch in ``compile()``.

FU3 added a typed ``except (LintFailureError, SmokeCheckFailure):`` clause to
``compile/mellea_skills.py::compile``. Because ``compile()`` previously
contained a *lazy local* ``from ... import SmokeCheckFailure`` statement
deep inside a conditional block, Python's name-binding rules treated
``SmokeCheckFailure`` as a function-local variable for the WHOLE function.
That shadowed the module-level import FU3 added at the top of the file —
so any failure path that fired BEFORE the lazy import executed would land
at the typed-except clause and crash with::

    UnboundLocalError: cannot access local variable 'SmokeCheckFailure'
    where it is not associated with a value

The fix (2026-05-19) removed the lazy local import. The module-level import
at the top of ``mellea_skills.py`` is now the sole source.

These tests guard against the bug recurring by inspecting the AST of
``compile()`` and asserting:

  1. Neither ``LintFailureError`` nor ``SmokeCheckFailure`` is bound by a
     ``from ... import`` statement inside the function body.
  2. Both names are present as module-level imports in ``mellea_skills.py``.

If a future edit re-introduces a local import of either typed exception
inside ``compile()``, this test will fail with a precise pointer to the
file, line, and the shadowing import statement.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MELLEA_SKILLS_PY = (
    _REPO_ROOT
    / "src"
    / "mellea_skills_compiler"
    / "compile"
    / "mellea_skills.py"
)


_SHADOW_RISK_NAMES = frozenset({"LintFailureError", "SmokeCheckFailure"})


def _parse_mellea_skills() -> ast.Module:
    """Parse mellea_skills.py once; fail informatively if it's gone."""
    if not _MELLEA_SKILLS_PY.is_file():
        pytest.fail(
            f"mellea_skills.py not found at {_MELLEA_SKILLS_PY}; "
            f"this regression test cannot run. Investigate before "
            f"skipping — the file's existence is part of the package "
            f"contract."
        )
    return ast.parse(_MELLEA_SKILLS_PY.read_text(encoding="utf-8"))


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Find a top-level ``def <name>(...)``; fail if absent."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(
        f"Top-level function ``{name}`` not found in mellea_skills.py. "
        f"This regression test indexes against that function; either the "
        f"function was renamed (update this test) or removed (re-evaluate "
        f"the FU3 wiring)."
    )


def _local_typed_exception_imports(
    func: ast.FunctionDef,
) -> list[tuple[str, int]]:
    """Walk ``func`` and find every local ``from ... import X`` of a name in
    ``_SHADOW_RISK_NAMES``. Returns a list of ``(name, lineno)`` tuples.
    """
    hits: list[tuple[str, int]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name in _SHADOW_RISK_NAMES:
                    hits.append((bound_name, node.lineno))
    return hits


def test_compile_has_no_local_shadowing_imports_of_typed_exceptions():
    """``compile()`` must not lazy-import LintFailureError or SmokeCheckFailure.

    A local ``from ... import LintFailureError`` or
    ``from ... import SmokeCheckFailure`` inside ``compile()`` shadows the
    module-level imports the FU3 typed-except dispatch relies on. The
    consequence is an ``UnboundLocalError`` on any failure path that fires
    before the lazy import executes — observed in real compiles on
    2026-05-19.

    If this test fires, move the import to the top of ``mellea_skills.py``
    (it almost certainly already exists there for FU3 reasons; the local
    copy is the bug).
    """
    tree = _parse_mellea_skills()
    compile_fn = _find_function_def(tree, "compile")
    shadowing = _local_typed_exception_imports(compile_fn)
    if shadowing:
        formatted = "\n".join(
            f"  - {name} at {_MELLEA_SKILLS_PY}:{lineno}"
            for name, lineno in shadowing
        )
        pytest.fail(
            f"compile() contains local ``from ... import`` statements that "
            f"shadow the module-level typed-exception imports. Python's "
            f"name-binding rule treats these as function-local for the "
            f"entire function, breaking the outer typed-except dispatch. "
            f"Move the import to the top of mellea_skills.py and delete "
            f"the local copy.\n\nShadowing imports:\n{formatted}"
        )


def test_module_has_typed_exception_imports_at_module_scope():
    """Both typed exception names must be importable from mellea_skills.py.

    Sanity check that the module-level imports FU3 added haven't been
    inadvertently removed. The typed-except dispatch fails closed without
    them.
    """
    tree = _parse_mellea_skills()
    module_level_imports: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name in _SHADOW_RISK_NAMES:
                    module_level_imports[bound_name] = node.lineno

    missing = _SHADOW_RISK_NAMES - set(module_level_imports)
    assert not missing, (
        f"mellea_skills.py is missing module-level imports for the typed "
        f"exception names: {sorted(missing)}. The FU3 typed-except dispatch "
        f"in ``compile()`` references these names; without module-level "
        f"imports the dispatch crashes with NameError. Add ``from ... "
        f"import X`` at the top of mellea_skills.py for each missing name."
    )
