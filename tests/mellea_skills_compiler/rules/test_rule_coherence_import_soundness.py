"""Coherence audits for the ``import-soundness`` lint.

Three-face check for the rule registered as ``import-soundness``:

  * **Directive** (``.claude/commands/mellea-fy-generate.md`` line ~233):
    the LLM is told ``mellea_api_ref.json:.modules`` is authoritative
    — any path not present is invalid and must not be generated.
  * **Implementation**: the introspector walks the installed Mellea
    package via ``pkgutil.walk_packages`` and collects callables from
    a curated set of modules (``CORE_MODULES`` in ``compile/grounding.py``).
  * **Validation** (``compile/lints.py::lint_import_soundness``): reads
    ``mellea_api_ref.json:.modules`` and flags any ``mellea.*`` import
    whose path isn't a key in that dict.

The first run is expected to FAIL on
``test_every_mellea_backends_submodule_in_surface`` and
``test_real_mellea_backend_import_passes_lint`` — that failure is the
directive ↔ implementation drift surfaced by the gpai compile's
``from mellea.backends.ollama import ...`` false positive on 2026-05-19.
Fixing the introspector to enumerate ``mellea.backends.*`` makes both
checks pass.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import lint_import_soundness
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "import-soundness"


# ─── Shared helpers ──────────────────────────────────────────────────


def _enumerate_runtime_mellea_backends_submodules() -> set[str]:
    """Return every IMPORTABLE ``mellea.backends.<X>`` submodule per
    runtime introspection. This is the IMPLEMENTATION face — the
    ground truth the surface should match.

    Filters out submodules that ``pkgutil`` lists but that raise
    ``ImportError`` when actually imported. Some Mellea backends
    (huggingface, litellm, watsonx, bedrock) require optional
    extras (``mellea[hf]`` etc.); without the extras installed, the
    runtime cannot use them and the lint correctly flags imports of
    them. The surface should contain only what's actually usable in
    the current environment.
    """
    try:
        import mellea.backends as _backends
    except ImportError:  # pragma: no cover - mellea must be installed for tests
        pytest.skip("mellea package not installed; cannot audit import-soundness")
    discovered = {
        info.name
        for info in pkgutil.walk_packages(
            path=_backends.__path__, prefix="mellea.backends."
        )
    }
    importable: set[str] = set()
    for name in discovered:
        try:
            importlib.import_module(name)
        except ImportError:
            continue  # optional-extras backend not installed; skip
        importable.add(name)
    return importable


def _build_introspected_surface() -> set[str]:
    """Return the set of ``mellea.*`` module names that the actual
    introspector emits — the VALIDATION face's input.
    """
    # Local import to avoid carrying the compile-time import cost into
    # every test file in this directory.
    from mellea_skills_compiler.compile.grounding import _introspect_mellea

    api_ref = _introspect_mellea(referenced_modules=set())
    return set(api_ref.keys())


def _run_import_soundness_lint(import_statement: str, surface_modules: set[str]) -> list:
    """Render ``import_statement`` as a one-line ``pipeline.py``, drop a
    surface JSON containing the given module names, and run the lint.
    """
    surface = {
        "modules": {name: {} for name in surface_modules},
    }
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg_mellea"
        pkg.mkdir()
        (pkg / "intermediate").mkdir()
        (pkg / "intermediate" / "mellea_api_ref.json").write_text(
            json.dumps(surface), encoding="utf-8"
        )
        (pkg / "pipeline.py").write_text(
            f"{import_statement}\n", encoding="utf-8"
        )
        result = lint_import_soundness(pkg)
        return list(result.failures)


# ─── Coherence checks ────────────────────────────────────────────────


def test_every_mellea_backends_submodule_in_surface():
    """C-BACKENDS-NAMESPACE-COMPLETE: every importable
    ``mellea.backends.<X>`` submodule must appear in the introspected
    surface. Failure means the directive lies to the LLM: it tells
    the LLM "use mellea_api_ref.json as ground truth" but the artifact
    is incomplete relative to the runtime.
    """
    runtime_submodules = _enumerate_runtime_mellea_backends_submodules()
    if not runtime_submodules:
        pytest.skip("no mellea.backends.* submodules discovered at runtime")
    introspected = _build_introspected_surface()
    missing = runtime_submodules - introspected
    assert not missing, (
        f"C-BACKENDS-NAMESPACE-COMPLETE failed (directive ↔ impl drift): "
        f"{len(missing)} mellea.backends.* submodule(s) are importable at "
        f"runtime but NOT present in the introspected surface. The "
        f"directive (mellea-fy-generate.md Rule 5-2) tells the LLM to "
        f"treat the surface as authoritative — so missing modules cause "
        f"real `from mellea.backends.<X> import ...` imports to fail "
        f"the lint as false positives. Missing: {sorted(missing)}"
    )


def test_real_mellea_backend_import_passes_lint():
    """C-REAL-MELLEA-IMPORT-PASSES: a `from mellea.backends.ollama import
    OllamaModelBackend` import (a real, importable module) must not be
    flagged by the lint. Verifies the FULL chain — surface contains
    the module AND the lint correctly accepts an import of it.

    This test builds its surface from the actual introspector output
    rather than a hand-curated synthetic surface, so it audits the
    real production pipeline end-to-end.
    """
    # Sanity: only run if mellea.backends.ollama is actually importable.
    try:
        importlib.import_module("mellea.backends.ollama")
    except ImportError:
        pytest.skip("mellea.backends.ollama not available in this install")
    introspected = _build_introspected_surface()
    failures = _run_import_soundness_lint(
        "from mellea.backends.ollama import OllamaModelBackend",
        introspected,
    )
    assert failures == [], (
        f"C-REAL-MELLEA-IMPORT-PASSES failed (directive ↔ impl drift): "
        f"a real, importable Mellea module was flagged by import-soundness. "
        f"Either the introspector is missing this module (most likely) or "
        f"the lint is mis-parsing the import. Failure messages: "
        + repr([f.message for f in failures])
    )


def test_nonexistent_mellea_module_import_fails_lint():
    """C-FAKE-MELLEA-IMPORT-FAILS: a typo or hallucinated `mellea.*`
    import path must still be flagged. Guards against the fix
    becoming too permissive (e.g., accidentally accepting any
    `mellea.*` path without checking the surface).
    """
    # Use a synthetic surface that mimics a minimal real one.
    surface_modules = {"mellea.stdlib.session", "mellea.backends.ollama"}
    failures = _run_import_soundness_lint(
        "from mellea.totally.fake.module import SomethingFictional",
        surface_modules,
    )
    assert failures, (
        "C-FAKE-MELLEA-IMPORT-FAILS failed: a fictional mellea.* import "
        "was accepted by the lint. Typo guard regressed — the lint no "
        "longer catches hallucinated module names."
    )


def test_non_mellea_import_not_flagged():
    """C-NON-MELLEA-IMPORT-IGNORED: imports from non-mellea packages
    (stdlib, third-party) must not be flagged. The lint is scoped to
    the Mellea namespace; flagging unrelated imports would be a
    regression in the other direction.
    """
    surface_modules = {"mellea.stdlib.session"}  # whatever, not relevant here
    failures = _run_import_soundness_lint(
        "from collections import OrderedDict",
        surface_modules,
    )
    # Also check a non-`from` import.
    failures2 = _run_import_soundness_lint(
        "import json",
        surface_modules,
    )
    assert failures == [] and failures2 == [], (
        "C-NON-MELLEA-IMPORT-IGNORED failed: a non-mellea import was "
        "flagged. The lint is over-scoped — it should only check "
        "imports whose paths start with 'mellea.'."
    )


def test_declared_severity_matches_central_table():
    """C-DECLARED-SEVERITY-MATCHES: registry severity must match the
    central ``_LINT_SEVERITY`` table.
    """
    from mellea_skills_compiler.compile.lints import (
        LintSeverity,
        _LINT_SEVERITY,
    )

    rule = get_rule(_RULE_ID)
    declared = rule["validation"]["severity"]
    actual_enum = _LINT_SEVERITY.get(_RULE_ID)
    assert actual_enum is not None, (
        f"_LINT_SEVERITY has no entry for {_RULE_ID!r}"
    )
    actual = (
        actual_enum.value if isinstance(actual_enum, LintSeverity) else actual_enum
    )
    assert declared == actual, (
        f"C-DECLARED-SEVERITY-MATCHES failed: registry declares "
        f"{declared!r} but _LINT_SEVERITY[{_RULE_ID!r}] = {actual!r}."
    )
