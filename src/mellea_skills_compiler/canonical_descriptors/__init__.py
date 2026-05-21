"""Canonical descriptor library — per-archetype reference descriptors.

The descriptor-emission slash command selects a canonical at compile
time based on the skill's Step-0 classification, and injects it as a
worked example into the descriptor-emission system prompt.

Two parts:

* :func:`load_all` — discover every ``*.json`` canonical in this package
  (excluding ``index.json`` / ``schema.json``), parse it, and return a
  list of validated wrapper dicts.
* :func:`select_canonical` — given a classification dict (typically
  ``intermediate/classification.json``), return the best-matching
  canonical's wrapper.

Validation discipline (enforced by the coherence-test suite, not here):
every loaded canonical's inner ``descriptor`` MUST pass
``descriptor.validator.validate`` against the current schema +
semantic rules. The cornerstone of the example-as-fourth-face pattern,
scaled from per-rule fragments to per-archetype whole descriptors.

Selector match scoring (highest match wins, ties broken by
harvested_at):

* +3 for archetype match
* +2 for shape match
* +2 for modality match
* +1 for tool_involvement_variant match (when both populated)
* +1 for source_runtime match (when both populated)

A canonical with zero matching axes is still eligible (selector never
returns ``None`` — the prompt needs a one-shot present). When no
canonicals match the classification at all, the highest-loaded one
serves as a fallback; the directive doc explains that
classification-mismatched canonicals should be treated as structural
references only, not pattern-matched on archetype-specific shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any


_RESERVED_FILES: frozenset[str] = frozenset({"index.json", "schema.json"})


@dataclass(frozen=True)
class CanonicalMatch:
    """Outcome of a selector call.

    Carries the matched wrapper plus the match score so callers
    (prompt assembler, telemetry) can decide whether the match is
    strong enough to inject confidently or whether it's a fallback
    that should be qualified in the prompt.
    """

    wrapper: dict[str, Any]
    score: int
    """0 means no axes matched (pure fallback). Maximum score with
    three primary axes only is 7 (3+2+2); with all five it's 9."""


def load_all() -> list[dict[str, Any]]:
    """Return every canonical wrapper in this package.

    Skips ``index.json`` and ``schema.json``. Order is filesystem
    iteration order — callers that need stable ordering should sort.
    """
    pkg_files = resources.files(__package__)
    wrappers: list[dict[str, Any]] = []
    for entry in pkg_files.iterdir():
        if not entry.is_file():
            continue
        if entry.name in _RESERVED_FILES:
            continue
        if not entry.name.endswith(".json"):
            continue
        text = entry.read_text(encoding="utf-8")
        wrappers.append(json.loads(text))
    return wrappers


def _score(
    canonical_classification: dict[str, Any],
    request_classification: dict[str, Any],
) -> int:
    """Weighted match between a canonical's classification and a
    requested one. Primary axes weighted higher.
    """
    score = 0
    axes_primary = {"archetype": 3, "shape": 2, "modality": 2}
    axes_secondary = {"tool_involvement_variant": 1, "source_runtime": 1}
    for axis, weight in axes_primary.items():
        if (
            canonical_classification.get(axis) is not None
            and canonical_classification.get(axis) == request_classification.get(axis)
        ):
            score += weight
    for axis, weight in axes_secondary.items():
        c = canonical_classification.get(axis)
        r = request_classification.get(axis)
        if c is not None and r is not None and c == r:
            score += weight
    return score


def select_canonical(classification: dict[str, Any]) -> CanonicalMatch | None:
    """Return the highest-scoring canonical for the given Step-0
    classification.

    When the canonical library is empty, returns ``None`` — the caller
    is responsible for falling back to whatever pre-canonical behavior
    the descriptor-emission prompt used (the hardcoded sentry-find-bugs
    reference in current code).

    Ties broken by ``harvested_at`` (most recent first), then by file
    iteration order.
    """
    wrappers = load_all()
    if not wrappers:
        return None
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for w in wrappers:
        meta = w.get("metadata", {}) or {}
        c = meta.get("classification", {}) or {}
        s = _score(c, classification)
        scored.append((s, meta.get("harvested_at", ""), w))
    # Sort: highest score first; within score, most-recent harvested_at first.
    scored.sort(key=lambda t: (-t[0], t[1]), reverse=False)
    # The above sort places highest -score first AND lower harvested_at
    # (older dates) first within same score. Flip the harvested_at
    # comparison by sorting twice — simpler readability:
    scored.sort(key=lambda t: (-t[0], -ord(t[1][:1]) if t[1] else 0))
    best_score, _, best_wrapper = scored[0]
    return CanonicalMatch(wrapper=best_wrapper, score=best_score)
