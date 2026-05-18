"""Coverage test — every Mellea module the slash-command prose tells the LLM
to import MUST be reachable through CORE_MODULES or modality-driven expansion.

Why this test exists:
    Pre-fix, the api_ref grounding covered only 3 modules
    (model_options, requirements, sampling). The slash-command prose freely
    referenced `from mellea.stdlib.session import start_session` and similar
    surfaces — without grounding, the LLM fell back to training-data
    (Mellea 0.4) memory and emitted drift like treating `m.chat()` return
    as a string. Nothing in the test suite or pre-commit caught the
    prose/grounding mismatch; it only surfaced at runtime smoke-check on
    nis2-navigator.

    This test scans every `.claude/commands/mellea-fy-*.md` for
    `from mellea.X import …` and `mellea.X.Y` references and asserts each
    referenced module is either in `CORE_MODULES`, in `_MODALITY_MODULES`,
    or in `_TOOL_VARIANT_MODULES`. A new module reference in the prose
    that isn't grounded will fail this test loudly with a precise message.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.grounding import (
    CORE_MODULES,
    _MODALITY_MODULES,
    _TOOL_VARIANT_MODULES,
)


_COMMANDS_DIR = Path(__file__).resolve().parents[3] / ".claude" / "commands"


# Stale references that the prose still mentions for historical/illustrative
# reasons (e.g. as wrong-import examples in lint documentation). These are
# explicitly excluded from coverage. Add an explanation comment with each
# entry so future contributors know why.
_PROSE_REFERENCE_ALLOWLIST = frozenset({
    # Used in `mellea-fy-validate.md` as an EXAMPLE of a wrong import path
    # that the `parseable` lint would catch — the module does not actually
    # exist in mellea. The reference is purely illustrative.
    "mellea.stdlib.strategies",
    # Internal symbol the compiler pulls from genslot via a compat shim
    # (`_extract_forbidden_param_names`). Documented in deps.md as the source
    # for forbidden param names; the LLM does not import it directly.
    "mellea.stdlib.components.genslot._disallowed_param_names",
    # Used in `mellea-fy-validate-emissions.md` as an EXAMPLE of a
    # hallucinated `from mellea.backends.tools import ...` that the
    # Step 5b `import-soundness` rule catches (the canonical path is
    # `mellea.stdlib.tools`). Purely illustrative — the module does
    # not exist.
    "mellea.backends.tools",
})


# Captures any reference to a mellea submodule:
#  - `from mellea.X.Y import Z`
#  - `import mellea.X.Y`
#  - `mellea.X.Y.symbol` in prose (e.g. dotted reference like
#    `mellea.stdlib.requirements.Requirement` in explanation text)
_MELLEA_MODULE_REFERENCE_RE = re.compile(
    r"""
    (?:
        from\s+(mellea(?:\.[a-z_][a-z_0-9]*)+)\s+import     # from mellea.X.Y import
        |
        import\s+(mellea(?:\.[a-z_][a-z_0-9]*)+)            # import mellea.X.Y
        |
        \b(mellea\.[a-z_][a-z_0-9]*(?:\.[a-z_][a-z_0-9]*)+) # dotted reference mellea.X.Y[.Z]
    )
    """,
    re.VERBOSE,
)


def _collect_prose_module_refs() -> set[str]:
    """Walk `.claude/commands/mellea-fy-*.md` and collect mellea module refs."""
    refs: set[str] = set()
    for md_path in _COMMANDS_DIR.glob("mellea-fy-*.md"):
        text = md_path.read_text()
        for match in _MELLEA_MODULE_REFERENCE_RE.finditer(text):
            ref = next((g for g in match.groups() if g), None)
            if ref and ref != "mellea":
                refs.add(ref)
    return refs


def _grounded_modules() -> set[str]:
    """Return every module reachable through CORE + modality + variant expansions."""
    grounded = set(CORE_MODULES)
    for mods in _MODALITY_MODULES.values():
        grounded |= mods
    for mods in _TOOL_VARIANT_MODULES.values():
        grounded |= mods
    return grounded


def _normalise_reference(ref: str, grounded: set[str]) -> str:
    """Trim trailing symbols from a dotted reference to find a module match.

    `mellea.stdlib.requirements.Requirement` is a class reference under the
    `mellea.stdlib.requirements` module; the grounding indexes the module,
    not the class. Walk the dotted path from the end inward to find the
    longest prefix that is a known module.
    """
    parts = ref.split(".")
    for i in range(len(parts), 1, -1):
        candidate = ".".join(parts[:i])
        if candidate in grounded:
            return candidate
    return ref


class TestProseGroundingCoverage:
    """Every Mellea module the prose mentions MUST be grounded somewhere."""

    def test_all_prose_module_refs_are_grounded(self):
        """Scan slash-command prose; assert each referenced module is grounded."""
        prose_refs = _collect_prose_module_refs()
        grounded = _grounded_modules()

        uncovered: list[str] = []
        for ref in sorted(prose_refs):
            if ref in _PROSE_REFERENCE_ALLOWLIST:
                continue
            resolved = _normalise_reference(ref, grounded)
            if resolved in grounded:
                continue
            uncovered.append(ref)

        assert not uncovered, (
            "These mellea module references appear in slash-command prose but "
            "are NOT grounded via CORE_MODULES, _MODALITY_MODULES, or "
            "_TOOL_VARIANT_MODULES — meaning the LLM is told to import them "
            "without ever seeing their signatures in mellea_api_ref.json. Fix "
            "by adding the module to grounding.py:CORE_MODULES (universal) or "
            "to _MODALITY_MODULES[<modality>] (per-modality) or to "
            "_TOOL_VARIANT_MODULES[<variant>] (per-tool-variant). If the "
            "reference is intentionally illustrative (e.g. a wrong-import "
            "example for a lint), add it to "
            "_PROSE_REFERENCE_ALLOWLIST in this test file with a comment "
            "explaining why.\n\nUncovered references:\n  "
            + "\n  ".join(uncovered)
        )

    def test_prose_refs_actually_collected(self):
        """Sanity check — confirm the regex finds the well-known references.

        Without this, a regex regression could silently make the coverage
        test pass by collecting zero references.
        """
        prose_refs = _collect_prose_module_refs()
        assert "mellea.stdlib.requirements" in prose_refs, (
            "Regex regression: expected at least mellea.stdlib.requirements "
            "to appear in slash-command prose. Got: {}".format(sorted(prose_refs))
        )
        assert "mellea.backends.model_options" in prose_refs

    def test_allowlist_only_contains_currently_referenced_modules(self):
        """Prevent allowlist rot — every allowlist entry must appear in prose.

        If a module is removed from the prose, its allowlist entry should
        also be removed. This test fails cheaply when that drifts.
        """
        prose_refs = _collect_prose_module_refs()
        stale: list[str] = []
        for allowed in _PROSE_REFERENCE_ALLOWLIST:
            # Allow either the exact ref or a longer dotted form
            if not any(ref == allowed or ref.startswith(allowed + ".") for ref in prose_refs):
                stale.append(allowed)
        assert not stale, (
            f"These entries in _PROSE_REFERENCE_ALLOWLIST no longer appear "
            f"in slash-command prose and should be removed: {stale}"
        )
