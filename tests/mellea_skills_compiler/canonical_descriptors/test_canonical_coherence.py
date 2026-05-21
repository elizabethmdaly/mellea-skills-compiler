"""Coherence checks for the canonical-descriptor library.

The cornerstone discipline, scaled from per-rule example fragments
(registry.examples) to per-archetype whole descriptors:

* **Every canonical's inner ``descriptor`` MUST pass
  ``descriptor.validator.validate``** against the current schema +
  semantic rules. When schemas tighten, broken canonicals fail loudly
  and a maintainer must update them — the registry can't silently drift
  past the canonicals.
* **Every canonical's wrapper MUST validate against the wrapper
  schema** (`canonical_descriptors/schema.json`).
* **Every canonical's filename MUST encode its classification triple**
  so the selector + the filesystem agree on what each file represents.

When the library is empty (Phase-1 in-progress state), these tests pass
trivially — the contract holds vacuously.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import jsonschema
import pytest

from mellea_skills_compiler import canonical_descriptors
from mellea_skills_compiler.canonical_descriptors import load_all, select_canonical
from mellea_skills_compiler.descriptor.validator import validate


_PKG = canonical_descriptors.__package__


def _wrapper_schema() -> dict:
    text = resources.files(_PKG).joinpath("schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_library_loads_without_error():
    """Smoke: every JSON in canonical_descriptors/ parses cleanly."""
    wrappers = load_all()
    # Empty is acceptable during Phase 1 (no canonicals harvested yet).
    assert isinstance(wrappers, list)


def test_every_wrapper_validates_against_wrapper_schema():
    """Wrapper-level validation — metadata.classification triple +
    provenance + format_version."""
    wrappers = load_all()
    validator = jsonschema.Draft7Validator(_wrapper_schema())
    failures: list[str] = []
    for w in wrappers:
        errors = list(validator.iter_errors(w))
        if errors:
            src = w.get("metadata", {}).get("source", "<unknown>")
            failures.append(
                f"canonical from {src!r}: " + "; ".join(
                    f"{e.json_path} {e.message}" for e in errors
                )
            )
    assert not failures, "wrapper-schema violations:\n  " + "\n  ".join(failures)


def test_every_canonical_descriptor_passes_validator():
    """**Cornerstone**: every canonical's inner descriptor must
    validate against the current descriptor schema + all semantic
    rules.

    This is the scaled-up "positive examples pass all validators"
    discipline from the registry's example-face pilot. When schemas or
    semantic rules tighten, broken canonicals fail HERE and a
    maintainer must reconcile — no silent drift.
    """
    wrappers = load_all()
    failures: list[str] = []
    for w in wrappers:
        descriptor = w.get("descriptor")
        if not isinstance(descriptor, dict):
            failures.append(
                f"{w.get('metadata', {}).get('source', '<unknown>')}: "
                f"descriptor field missing or not an object"
            )
            continue
        rep = validate(descriptor, schema_version="0.3", surface=None)
        error_severity = [e for e in rep.errors if e.severity == "error"]
        if error_severity:
            src = w.get("metadata", {}).get("source", "<unknown>")
            errs_summary = "; ".join(
                f"[{e.rule}] {e.path}: {e.message[:80]}"
                for e in error_severity[:3]
            )
            if len(error_severity) > 3:
                errs_summary += f"; ... +{len(error_severity) - 3} more"
            failures.append(f"canonical from {src!r}: {errs_summary}")
    assert not failures, (
        "canonical descriptor(s) failing current schema + semantic rules:\n  "
        + "\n  ".join(failures)
    )


def test_filename_encodes_classification():
    """Each canonical's filename should encode its classification
    triple, format: ``<archetype>_<shape>_<modality>.json``. Keeps the
    filesystem index consistent with the wrapper metadata.

    Example: ``A_Sequential_synchronous-oneshot.json`` for an
    archetype-A sequential synchronous-oneshot canonical.

    When the library is empty, this test passes trivially.
    """
    pkg_files = resources.files(_PKG)
    failures: list[str] = []
    for entry in pkg_files.iterdir():
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        if entry.name in {"index.json", "schema.json"}:
            continue
        text = entry.read_text(encoding="utf-8")
        try:
            w = json.loads(text)
        except json.JSONDecodeError:
            continue  # caught by load_loads_without_error
        c = (w.get("metadata") or {}).get("classification") or {}
        archetype = c.get("archetype", "?")
        shape = c.get("shape", "?")
        modality = c.get("modality", "?")
        # Normalise modality (snake_case → kebab-case in filename).
        modality_segment = str(modality).replace("_", "-")
        expected = f"{archetype}_{shape}_{modality_segment}.json"
        if entry.name != expected:
            failures.append(
                f"file {entry.name!r} declares classification "
                f"({archetype}, {shape}, {modality}) but expected "
                f"filename {expected!r}"
            )
    assert not failures, "filename ↔ classification drift:\n  " + "\n  ".join(failures)


def test_selector_empty_library_returns_none():
    """Contract: when no canonicals exist, the selector returns
    ``None`` and the caller falls back to pre-canonical behavior. Once
    canonicals are added, this test will need to be retired — by then
    the selector should ALWAYS return a match.
    """
    if load_all():
        pytest.skip("library is non-empty; this test only applies pre-harvest")
    result = select_canonical({"archetype": "A", "shape": "Sequential", "modality": "synchronous_oneshot"})
    assert result is None


def test_selector_picks_highest_score_when_library_populated():
    """When canonicals exist, the selector returns the best-scoring
    match. Skipped when library is empty.
    """
    wrappers = load_all()
    if not wrappers:
        pytest.skip("library is empty; nothing to score")
    # Score-zero classification (totally unrelated) still returns
    # SOMETHING — selector never None when library is non-empty.
    result = select_canonical({"archetype": "ZZZ", "shape": "ZZZ", "modality": "ZZZ"})
    assert result is not None, "selector must always return a match when library is non-empty"
    # And a perfect-match classification scores higher than zero on
    # the matched canonical.
    first = wrappers[0]
    perfect = first["metadata"]["classification"]
    match = select_canonical(perfect)
    assert match.score > 0, (
        f"perfect-match classification should score > 0; got {match.score}"
    )
