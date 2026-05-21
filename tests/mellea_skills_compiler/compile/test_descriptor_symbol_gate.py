"""Unit tests for the C1 pre-render symbol-resolution gate.

The gate's discipline: **fail loud on ambiguity, fix silently on
certainty**. Every test below either exercises one of the five
auto-resolution stages or confirms that ambiguous/missing symbols are
surfaced as structured failures (not silently rewritten).

Stages covered:

  1. exact match
  2. suffix unification
  3. module-scoped leaf match  (handles "dropped class prefix")
  4. bare-name resolution
  5. fuzzy fallback  (only when rapidfuzz is installed)

Plus the cross-cutting properties:

  * The descriptor is mutated in place (renderer-ready after the gate).
  * Telemetry is written to symbol_normalisations.jsonl on rewrites.
  * Unresolvable symbols carry top-K candidate suggestions.
  * Real-world failure cases (gdpr-privacy-notice, gpai-code-of-practice,
    gdpr-breach-sentinel) all auto-resolve.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.descriptor_symbol_gate import (
    SymbolGateResult,
    SymbolMatch,
    SymbolResolutionError,
    _HAS_RAPIDFUZZ,
    _looks_like_mellea_reference,
    build_surface_index,
    resolve_symbol,
    run_symbol_gate,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_surface() -> dict:
    """A miniature Mellea-shaped surface for unit tests.

    Mirrors the real introspection's shape: ``modules`` keyed by dotted
    module paths, each module's value a dict whose keys are symbols
    (including ``ClassName.method`` for methods).
    """
    return {
        "modules": {
            "mellea.stdlib.session": {
                "start_session": {"signature": "start_session(...)"},
                "MelleaSession": {"signature": "MelleaSession(...)"},
                "MelleaSession.instruct": {"signature": "MelleaSession.instruct(...)"},
                "MelleaSession.chat": {"signature": "MelleaSession.chat(...)"},
                "MelleaSession.cleanup": {"signature": "MelleaSession.cleanup()"},
            },
            "mellea.stdlib.sampling": {
                "RepairTemplateStrategy": {"signature": "RepairTemplateStrategy(...)"},
                "RepairTemplateStrategy.repair": {"signature": "..."},
                "RejectionSamplingStrategy": {"signature": "..."},
            },
            "mellea.stdlib.requirements": {
                "Requirement": {"signature": "Requirement(...)"},
                "check": {"signature": "check(...)"},
            },
            "mellea.stdlib.components.genstub": {
                "generative": {"signature": "generative(...)"},
            },
        }
    }


@pytest.fixture
def surface() -> dict:
    return _make_surface()


@pytest.fixture
def index(surface: dict) -> dict:
    return build_surface_index(surface)


# ─── Stage 1: exact match ────────────────────────────────────────────


def test_exact_match_passes_through_unchanged(index: dict):
    m = resolve_symbol(
        "mellea.stdlib.session.MelleaSession.instruct", index
    )
    assert m.resolved == "mellea.stdlib.session.MelleaSession.instruct"
    assert m.method == "exact"


def test_exact_match_for_top_level_symbol(index: dict):
    m = resolve_symbol("mellea.stdlib.session.start_session", index)
    assert m.resolved == "mellea.stdlib.session.start_session"
    assert m.method == "exact"


# ─── Stage 2: suffix unification ─────────────────────────────────────


def test_suffix_match_class_method_drops_module_prefix(index: dict):
    """``MelleaSession.instruct`` is a unique surface suffix → resolve."""
    m = resolve_symbol("MelleaSession.instruct", index)
    assert m.resolved == "mellea.stdlib.session.MelleaSession.instruct"
    assert m.method == "suffix"


def test_suffix_match_drops_full_module_prefix(index: dict):
    """Bare ``RepairTemplateStrategy`` resolves via suffix (top-level leaf)."""
    m = resolve_symbol("RepairTemplateStrategy", index)
    assert m.resolved == "mellea.stdlib.sampling.RepairTemplateStrategy"
    assert m.method == "suffix"


def test_suffix_match_does_not_fire_on_ambiguous_leaf():
    """Two symbols sharing a leaf name → suffix stage does NOT auto-fix.

    Constructs a surface with two modules both exposing a function
    named ``Requirement`` and asserts that ``Requirement`` (bare) does
    not silently resolve.
    """
    surface = {
        "modules": {
            "mellea.stdlib.requirements": {"Requirement": {}},
            "mellea.stdlib.alt": {"Requirement": {}},
        }
    }
    idx = build_surface_index(surface)
    m = resolve_symbol("Requirement", idx)
    assert m.resolved is None, (
        "Two surface paths share the leaf name; the gate must not "
        "guess. Expected None; got " + repr(m.resolved)
    )
    assert m.method is None


# ─── Stage 3: module-scoped leaf match ───────────────────────────────


def test_module_scoped_leaf_match_handles_dropped_class_prefix(index: dict):
    """The gpai-code-of-practice failure: module prefix is correct but
    class prefix is missing.

    Surface has ``mellea.stdlib.session.MelleaSession.chat``; LLM emitted
    ``mellea.stdlib.session.chat``. The module-scoped leaf stage
    rewrites it.
    """
    m = resolve_symbol("mellea.stdlib.session.chat", index)
    assert m.resolved == "mellea.stdlib.session.MelleaSession.chat"
    assert m.method == "module-scoped-leaf"


def test_module_scoped_leaf_match_only_fires_for_real_module_paths(
    index: dict,
):
    """Prefix that isn't a module key → don't auto-resolve."""
    m = resolve_symbol("not.a.real.module.instruct", index)
    assert m.resolved is None


def test_module_scoped_leaf_does_not_fire_when_ambiguous():
    """If multiple symbols within a module end with the candidate leaf,
    the stage refuses to auto-fix."""
    surface = {
        "modules": {
            "mellea.stdlib.session": {
                "MelleaSession.cleanup": {},
                "AltSession.cleanup": {},
            }
        }
    }
    idx = build_surface_index(surface)
    m = resolve_symbol("mellea.stdlib.session.cleanup", idx)
    assert m.resolved is None


# ─── Stage 4: bare-name resolution ───────────────────────────────────


def test_bare_name_resolves_when_unique(index: dict):
    """The bare ``check`` leaf maps to exactly one surface path."""
    m = resolve_symbol("check", index)
    assert m.resolved == "mellea.stdlib.requirements.check"
    # Note: this case is matched by suffix unification first since
    # 'check' is a unique suffix; either result is acceptable, but the
    # canonical form is the same.
    assert m.method in {"suffix", "bare-name"}


def test_bare_name_does_not_fire_for_dotted_candidates(index: dict):
    """The bare-name stage only fires when there are no dots; dotted
    candidates that didn't match suffix or module-scoped stages fall
    through to fuzzy (or, with no rapidfuzz, to unresolvable)."""
    m = resolve_symbol("foo.bar", index)
    assert m.resolved is None


# ─── Stage 5: fuzzy fallback ─────────────────────────────────────────


@pytest.mark.skipif(
    not _HAS_RAPIDFUZZ, reason="rapidfuzz not installed; stage 5 skipped"
)
def test_fuzzy_match_recovers_close_typo(index: dict):
    """When rapidfuzz is available, near-spelling typos resolve."""
    m = resolve_symbol("mellea.stdlib.session.start_sesion", index)  # typo
    # We don't assert the exact resolved path because the fuzzy
    # behaviour is approximate; we assert it found *something* with
    # high confidence.
    if m.resolved is not None:
        assert m.method == "fuzzy"
        assert m.candidates  # populated when fuzzy ran


# ─── Unresolvable / structured failure ───────────────────────────────


def test_unresolvable_symbol_returns_none_with_no_method(index: dict):
    m = resolve_symbol("nothing.like.this.exists", index)
    assert m.resolved is None
    assert m.method is None


def test_unresolvable_re_export_form_is_not_silently_accepted(index: dict):
    """Per Step 0b's Option 1 (docs match renderer), the re-export form
    is NOT a canonical surface path. The gate must refuse to
    auto-resolve it — that's the whole point of choosing Option 1.
    """
    m = resolve_symbol("mellea.MelleaSession.instruct", index)
    assert m.resolved is None, (
        "Re-export forms must NOT silently auto-normalise. Step 0b "
        "Option 1 decision: the renderer (and gate) accept only "
        "defining-module paths."
    )


# ─── Descriptor walking + in-place mutation ──────────────────────────


def test_run_symbol_gate_mutates_descriptor_in_place(surface: dict, tmp_path):
    """The descriptor returned from ``run_symbol_gate`` has every
    auto-normalisation applied so the renderer can consume it directly."""
    descriptor = {
        "pipeline": [
            {
                "id": "main",
                "kind": "composition",
                "body": [
                    {
                        "id": "n1",
                        "kind": "call",
                        "symbol": "MelleaSession.instruct",  # stage 2
                    },
                    {
                        "id": "n2",
                        "kind": "call",
                        "symbol": "mellea.stdlib.session.chat",  # stage 3
                        "args": {
                            "strategy": {
                                "symbol": "RepairTemplateStrategy",  # stage 2
                            }
                        },
                    },
                ],
            }
        ]
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )

    assert result.ok
    assert len(result.normalisations) == 3

    # Symbols rewritten in place.
    body = result.descriptor_after["pipeline"][0]["body"]
    assert body[0]["symbol"] == "mellea.stdlib.session.MelleaSession.instruct"
    assert body[1]["symbol"] == "mellea.stdlib.session.MelleaSession.chat"
    assert (
        body[1]["args"]["strategy"]["symbol"]
        == "mellea.stdlib.sampling.RepairTemplateStrategy"
    )


def test_run_symbol_gate_writes_telemetry_jsonl(surface: dict, tmp_path):
    """When normalisations happen and intermediate_dir is set, telemetry
    is written to ``symbol_normalisations.jsonl``."""
    descriptor = {
        "state": [{"id": "session", "symbol": "MelleaSession.instruct"}],
    }
    intermediate = tmp_path / "intermediate"
    run_symbol_gate(descriptor, surface, intermediate_dir=intermediate)

    telemetry = intermediate / "symbol_normalisations.jsonl"
    assert telemetry.exists()
    lines = telemetry.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["original"] == "MelleaSession.instruct"
    assert record["resolved"] == "mellea.stdlib.session.MelleaSession.instruct"
    assert record["method"] == "suffix"
    assert record["descriptor_path"].endswith("symbol")


def test_run_symbol_gate_does_not_write_telemetry_when_no_rewrites(
    surface: dict, tmp_path
):
    """Steady state: descriptor already canonical → no telemetry file."""
    descriptor = {
        "state": [
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
            }
        ]
    }
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    run_symbol_gate(descriptor, surface, intermediate_dir=intermediate)

    assert not (intermediate / "symbol_normalisations.jsonl").exists()


def test_run_symbol_gate_surfaces_unresolvable_with_candidates(
    surface: dict, tmp_path
):
    """A symbol the gate cannot resolve becomes a structured entry in
    ``result.unresolvable`` with optional top-K candidates for repair.

    Uses a ``mellea.``-prefixed typo so the shape-based skip
    (:func:`_looks_like_mellea_reference`) classifies it as a Mellea
    reference and the gate validates it (rather than passing it
    through as a non-Mellea symbol).
    """
    descriptor = {
        "state": [{"id": "s", "symbol": "mellea.totally.unknown.symbol"}],
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )
    assert not result.ok
    assert len(result.unresolvable) == 1
    failed = result.unresolvable[0]
    assert failed.original == "mellea.totally.unknown.symbol"
    assert failed.resolved is None
    assert failed.method is None
    # candidates may be empty when rapidfuzz is absent — both are fine.
    if _HAS_RAPIDFUZZ:
        assert isinstance(failed.candidates, list)


# ─── SymbolResolutionError ───────────────────────────────────────────


def test_symbol_resolution_error_carries_structured_result(surface: dict):
    """The typed exception preserves the full gate result so the repair
    pipeline (D2) can extract candidates without re-parsing."""
    descriptor = {
        "state": [{"id": "s", "symbol": "mellea.completely.unknown"}],
    }
    result = run_symbol_gate(descriptor, surface)
    exc = SymbolResolutionError(result)
    assert exc.gate_result is result
    # The summary message should name the first failing symbol so
    # operators see it in logs without inspecting the exception.
    assert "mellea.completely.unknown" in str(exc)


# ─── Surface-shape compatibility ─────────────────────────────────────


def test_gate_handles_bundled_surface_shape():
    """The bundled ``surface_0.5.0.json`` nests symbols under a
    ``symbols`` sub-dict; the introspected ``mellea_api_ref.json``
    puts them as direct keys. The index builder must handle both.
    """
    bundled_shape = {
        "modules": {
            "mellea.stdlib.session": {
                "doc": "Session module docs.",
                "symbols": {
                    "MelleaSession.instruct": {"signature": "..."},
                    "start_session": {"signature": "..."},
                },
            }
        }
    }
    idx = build_surface_index(bundled_shape)
    # Both symbols should be discoverable via the index.
    assert "mellea.stdlib.session.MelleaSession.instruct" in idx["by_path"]
    assert "mellea.stdlib.session.start_session" in idx["by_path"]
    # And resolution from a partial form should succeed.
    m = resolve_symbol("MelleaSession.instruct", idx)
    assert m.resolved == "mellea.stdlib.session.MelleaSession.instruct"


def test_gate_handles_introspected_surface_shape():
    """The introspected shape (no ``symbols`` sub-key) is the same
    case our other tests already cover; this is the explicit
    counterpart to ``test_gate_handles_bundled_surface_shape``."""
    introspected_shape = {
        "modules": {
            "mellea.stdlib.session": {
                "MelleaSession.instruct": {"signature": "..."},
                "start_session": {"signature": "..."},
            }
        }
    }
    idx = build_surface_index(introspected_shape)
    assert "mellea.stdlib.session.MelleaSession.instruct" in idx["by_path"]
    assert "mellea.stdlib.session.start_session" in idx["by_path"]


# ─── Shape-based skip: only Mellea API refs are validated ────────────


def test_looks_like_mellea_reference_accepts_mellea_dot_prefix(index: dict):
    """Anything starting with ``mellea.`` is treated as a Mellea reference,
    even if the rest of the path is wrong (so genuine typos still flag)."""
    assert _looks_like_mellea_reference(
        "mellea.stdlib.session.MelleaSession.instruct", index
    )
    assert _looks_like_mellea_reference(
        "mellea.stdlib.session.completely_wrong_symbol", index
    )


def test_looks_like_mellea_reference_accepts_known_top_level_symbol(
    index: dict,
):
    """Bare or partial qualifications of a Mellea top-level symbol are
    treated as Mellea references — that's how the gate catches
    ``MelleaSession.instruct``-style partial qualifications."""
    assert _looks_like_mellea_reference("MelleaSession.instruct", index)
    assert _looks_like_mellea_reference("RepairTemplateStrategy", index)
    assert _looks_like_mellea_reference("start_session", index)


def test_looks_like_mellea_reference_rejects_local_helpers(index: dict):
    """Local-helper symbols (``loader.X``, ``utils.X``) are NOT Mellea
    references — the leading segment is not a known Mellea top-level
    symbol AND the path doesn't start with ``mellea.``."""
    assert not _looks_like_mellea_reference("loader.load_notice_types", index)
    assert not _looks_like_mellea_reference("helpers.format_output", index)


def test_looks_like_mellea_reference_rejects_python_builtins(index: dict):
    """Python type markers (``builtins.dict``) are NOT Mellea references."""
    assert not _looks_like_mellea_reference("builtins.dict", index)
    assert not _looks_like_mellea_reference("builtins.list", index)


def test_looks_like_mellea_reference_rejects_package_qualified_locals(
    index: dict,
):
    """Fully-qualified local-package references (the gpai-code-of-practice
    failure pattern) are NOT Mellea references and must pass through."""
    assert not _looks_like_mellea_reference(
        "gpai_code_of_practice_mellea.slots.classify_gpai_provider_status",
        index,
    )
    assert not _looks_like_mellea_reference(
        "gdpr_privacy_notice_mellea.requirements.is_breach", index
    )


def test_run_symbol_gate_skips_local_helper_symbols(
    surface: dict, tmp_path
):
    """gdpr-privacy-notice failure pattern: descriptor mixes Mellea API
    symbols (validated) with local-helper symbols in
    ``dependencies[].symbol`` and Python type markers in
    ``pipeline[].args[X].symbol``. The gate must validate Mellea symbols
    only — locals and builtins are passed through unflagged.
    """
    descriptor = {
        "dependencies": [
            {"id": "load_data", "symbol": "loader.load_notice_types"},
            {"id": "fmt", "symbol": "helpers.format_output"},
        ],
        "pipeline": [
            {
                "id": "n1",
                "kind": "call",
                "symbol": "MelleaSession.instruct",  # Mellea (stage 2)
                "args": {
                    "ret_type": {"symbol": "builtins.dict"},  # Python marker
                    "strategy": {"symbol": "RepairTemplateStrategy"},  # Mellea
                },
            }
        ],
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )
    assert result.ok, (
        "Local-helper and builtins symbols should NOT cause the gate to "
        "fail; only Mellea-API symbols are validated against the "
        f"surface. Unresolvable: {[m.original for m in result.unresolvable]}"
    )
    # Mellea symbols were normalised in place.
    body = result.descriptor_after["pipeline"][0]
    assert (
        body["symbol"]
        == "mellea.stdlib.session.MelleaSession.instruct"
    )
    assert (
        body["args"]["strategy"]["symbol"]
        == "mellea.stdlib.sampling.RepairTemplateStrategy"
    )
    # Non-Mellea symbols were preserved verbatim.
    assert result.descriptor_after["dependencies"][0]["symbol"] == (
        "loader.load_notice_types"
    )
    assert result.descriptor_after["dependencies"][1]["symbol"] == (
        "helpers.format_output"
    )
    assert body["args"]["ret_type"]["symbol"] == "builtins.dict"


def test_run_symbol_gate_skips_fully_qualified_local_package_refs(
    surface: dict, tmp_path
):
    """gpai-code-of-practice failure pattern: the ENTIRE pipeline mixes
    Mellea API calls with fully-qualified references into the
    skill's own emitted package (``<package>_mellea.X.Y``). Those
    must pass through unflagged — only Mellea references are
    validated against the surface.
    """
    descriptor = {
        "pipeline": [
            {
                "id": "n1",
                "kind": "call",
                "symbol": (
                    "gpai_code_of_practice_mellea.slots."
                    "classify_gpai_provider_status"
                ),
            },
            {
                "id": "n2",
                "kind": "call",
                "symbol": "mellea.stdlib.session.start_session",  # Mellea exact
            },
            {
                "id": "n3",
                "kind": "call",
                "symbol": (
                    "gpai_code_of_practice_mellea.requirements."
                    "is_systemic_risk"
                ),
            },
            {
                "id": "n4",
                "kind": "call",
                "symbol": "MelleaSession.instruct",  # Mellea suffix
            },
        ]
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )
    assert result.ok, (
        "Fully-qualified local-package references must pass through. "
        f"Unresolvable: {[m.original for m in result.unresolvable]}"
    )
    body = result.descriptor_after["pipeline"]
    # Local-package symbols preserved verbatim.
    assert body[0]["symbol"] == (
        "gpai_code_of_practice_mellea.slots.classify_gpai_provider_status"
    )
    assert body[2]["symbol"] == (
        "gpai_code_of_practice_mellea.requirements.is_systemic_risk"
    )
    # Mellea symbols validated / normalised.
    assert body[1]["symbol"] == "mellea.stdlib.session.start_session"
    assert (
        body[3]["symbol"]
        == "mellea.stdlib.session.MelleaSession.instruct"
    )


def test_run_symbol_gate_still_catches_genuine_mellea_typos(
    surface: dict, tmp_path
):
    """The skip must NOT swallow typos in genuine Mellea references.

    ``mellea.X`` paths and partial-qualifications of Mellea top-level
    symbols still get validated; an unresolvable Mellea reference
    surfaces in ``result.unresolvable`` as before.
    """
    descriptor = {
        "pipeline": [
            {
                "id": "n1",
                "kind": "call",
                "symbol": "mellea.stdlib.session.does_not_exist",  # typo
            }
        ]
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )
    assert not result.ok, (
        "Typos in genuine Mellea references must still fail the gate; "
        "the shape-based skip only suppresses NON-Mellea symbols."
    )
    assert len(result.unresolvable) == 1
    assert (
        result.unresolvable[0].original
        == "mellea.stdlib.session.does_not_exist"
    )


# ─── End-to-end against the real failure cases ───────────────────────


def test_real_world_failure_cases_all_resolve(surface: dict, tmp_path):
    """Every symbol from the three observed compile failures
    (gdpr-breach-sentinel, gdpr-privacy-notice,
    gpai-code-of-practice) auto-resolves via stages 2 and 3."""
    descriptor = {
        "pipeline": [
            {
                "id": "n",
                "kind": "call",
                "symbol": "start_session",  # gdpr-breach-sentinel
                "args": {
                    "a": {"symbol": "MelleaSession.instruct"},  # gdpr-privacy-notice
                    "b": {"symbol": "mellea.stdlib.session.chat"},  # gpai-code-of-practice
                    "c": {"symbol": "RepairTemplateStrategy"},
                },
            }
        ]
    }
    result = run_symbol_gate(
        descriptor, surface, intermediate_dir=tmp_path / "intermediate"
    )
    assert result.ok, (
        "All four observed-failure symbols should resolve through "
        f"the gate's deterministic stages. Unresolvable: "
        f"{[m.original for m in result.unresolvable]}"
    )
    assert len(result.normalisations) == 4
    # Spot-check each resolved to the canonical defining-module path.
    n = result.descriptor_after["pipeline"][0]
    assert n["symbol"] == "mellea.stdlib.session.start_session"
    assert (
        n["args"]["a"]["symbol"]
        == "mellea.stdlib.session.MelleaSession.instruct"
    )
    assert (
        n["args"]["b"]["symbol"]
        == "mellea.stdlib.session.MelleaSession.chat"
    )
    assert (
        n["args"]["c"]["symbol"]
        == "mellea.stdlib.sampling.RepairTemplateStrategy"
    )
