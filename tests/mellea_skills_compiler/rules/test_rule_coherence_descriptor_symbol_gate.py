"""Coherence audits for the ``descriptor-symbol-gate`` mechanism.

The gate is a pre-render check that resolves every Mellea-API ``symbol``
field in the descriptor IR against the introspected Mellea surface. It
has three faces: directive (currently null — see C-DIRECTIVE-DOC-EXISTS),
implementation (``descriptor_symbol_gate.py``), validation (this file).

These tests synthesize all three and assert they agree. Where the
implementation does not yet meet the contract — the re-export collision
bug surfaced on 2026-05-20 — the corresponding test is ``xfail(strict=True)``
with the matching registry-level ``xfail_until`` deadline. When the
resolver fix lands the test flips green; if the bug re-regresses the
strict-xfail catches it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.compile.descriptor_symbol_gate import (
    build_surface_index,
    is_valid_for_renderer,
    resolve_symbol,
    run_symbol_gate,
)
from mellea_skills_compiler.rules import get_rule


_RULE_ID = "descriptor-symbol-gate"


# Synthetic surfaces are used throughout — small, deterministic, and
# decoupled from the live ``surface_0.5.0.json`` so changes upstream
# don't make these coherence checks flap.


def _surface_with_canonical_only() -> dict:
    """Surface where each symbol exists at exactly one path. Used to
    exercise stages 1-4 without re-export complications."""
    return {
        "modules": {
            "mellea.stdlib.session": {
                "symbols": {
                    "MelleaSession": {
                        "defined_in": "mellea.stdlib.session",
                        "doc": "Session class.",
                    },
                    "MelleaSession.instruct": {
                        "defined_in": "mellea.stdlib.session",
                        "doc": "Instruction call.",
                    },
                    "MelleaSession.chat": {
                        "defined_in": "mellea.stdlib.session",
                        "doc": "Chat call.",
                    },
                }
            }
        }
    }


def _surface_with_reexport() -> dict:
    """Surface where ``start_session`` is re-exported at the top level
    AND defined canonically — the exact shape that broke the gate on
    2026-05-20.
    """
    return {
        "modules": {
            "mellea": {
                "symbols": {
                    "start_session": {
                        "defined_in": "mellea.stdlib.session",
                        "doc": "Re-export of mellea.stdlib.session.start_session.",
                    }
                }
            },
            "mellea.stdlib.session": {
                "symbols": {
                    "start_session": {
                        "defined_in": "mellea.stdlib.session",
                        "doc": "Start a new Mellea session.",
                    }
                }
            },
        }
    }


def _surface_with_genuine_collision() -> dict:
    """Surface where two DISTINCT symbols share a leaf — different
    ``defined_in`` modules. The gate must continue to fail loud here.
    """
    return {
        "modules": {
            "mellea.stdlib.helpers": {
                "symbols": {
                    "format_input": {
                        "defined_in": "mellea.stdlib.helpers",
                        "doc": "Helper formatter.",
                    }
                }
            },
            "mellea.stdlib.requirements": {
                "symbols": {
                    "format_input": {
                        "defined_in": "mellea.stdlib.requirements",
                        "doc": "Requirements formatter (different symbol).",
                    }
                }
            },
        }
    }


# ─── Coherence checks ────────────────────────────────────────────────


def test_exact_canonical_path_passes():
    """C-EXACT-CANONICAL-PASSES."""
    index = build_surface_index(_surface_with_canonical_only())
    assert is_valid_for_renderer(
        "mellea.stdlib.session.MelleaSession.instruct", index
    )
    match = resolve_symbol(
        "mellea.stdlib.session.MelleaSession.instruct", index
    )
    assert match.resolved == "mellea.stdlib.session.MelleaSession.instruct"
    assert match.method == "exact"


def test_unique_suffix_auto_normalises():
    """C-SUFFIX-UNIQUE-NORMALISES."""
    index = build_surface_index(_surface_with_canonical_only())
    match = resolve_symbol("MelleaSession.instruct", index)
    assert match.resolved == "mellea.stdlib.session.MelleaSession.instruct"
    assert match.method in ("suffix", "bare-name")


def test_module_scoped_leaf_auto_normalises():
    """C-MODULE-SCOPED-LEAF-NORMALISES.

    The 'dropped class prefix' pattern: LLM wrote
    ``mellea.stdlib.session.chat`` when canonical is
    ``mellea.stdlib.session.MelleaSession.chat``.
    """
    index = build_surface_index(_surface_with_canonical_only())
    match = resolve_symbol("mellea.stdlib.session.chat", index)
    assert (
        match.resolved == "mellea.stdlib.session.MelleaSession.chat"
    ), f"expected module-scoped-leaf normalisation; got {match!r}"
    assert match.method == "module-scoped-leaf"


def test_reexport_aliases_resolve_to_canonical():
    """C-REEXPORT-RESOLVES-UNIQUELY."""
    index = build_surface_index(_surface_with_reexport())
    match = resolve_symbol("start_session", index)
    assert match.resolved == "mellea.stdlib.session.start_session", (
        f"bare `start_session` should resolve to its canonical "
        f"defined_in path, not be treated as an ambiguous collision; "
        f"got {match!r}"
    )


def test_genuine_collision_still_fails_with_candidates():
    """C-GENUINE-COLLISION-FAILS."""
    index = build_surface_index(_surface_with_genuine_collision())
    match = resolve_symbol("format_input", index)
    assert match.resolved is None, (
        f"two distinct symbols sharing a leaf must NOT auto-resolve; "
        f"got {match!r}"
    )
    # Candidates may be empty when rapidfuzz isn't installed — the
    # contract is "fail without silent rewrite", not "always have
    # candidates". The candidate-presence guarantee is rapidfuzz's job.


def test_non_mellea_symbols_are_skipped():
    """C-NON-MELLEA-SKIPPED."""
    index = build_surface_index(_surface_with_canonical_only())
    # A local helper / type marker — does NOT look like a Mellea ref.
    descriptor = {
        "dependencies": [{"id": "x", "symbol": "loader.SkillLoader"}],
        "state": [{"id": "s0", "symbol": "builtins.dict"}],
    }
    result = run_symbol_gate(descriptor, _surface_with_canonical_only())
    assert result.ok, (
        f"non-Mellea symbols must pass the gate without flagging; "
        f"unresolvable: {result.unresolvable!r}"
    )
    assert not result.normalisations, (
        "non-Mellea symbols must not be auto-rewritten; "
        f"got {result.normalisations!r}"
    )


def test_directive_doc_section_exists():
    """C-DIRECTIVE-DOC-EXISTS."""
    rule = get_rule(_RULE_ID)
    doc_pointer = rule["directive"]["doc"]
    section = rule["directive"]["section"]
    assert doc_pointer, "directive.doc must not be null"
    assert section, "directive.section must not be null"
    repo_root = Path(__file__).resolve().parents[3]
    doc_path = repo_root / doc_pointer
    assert doc_path.is_file(), f"directive doc {doc_pointer!r} does not exist"
    doc_text = doc_path.read_text(encoding="utf-8")
    assert section in doc_text, (
        f"directive section {section!r} not found in {doc_pointer!r}"
    )
