"""D2 — Closest-match enrichment for repair prompts.

When a gate (renderer error, schema-gate failure, lint failure)
rejects an LLM emission with a close-but-wrong symbol/path, this
module produces a 'Did you mean X?' hint pointing at the closest
valid alternative drawn from the relevant surface set.

Three public callables make up the contract — see the
``d2-closest-match-enrichment`` entry in the audit-coherence
registry for the full coherence checklist:

  * :func:`closest_module_match` — closest-match algorithm scoped
    to module-path candidates (used for ``import-soundness``
    failures).
  * :func:`suggest_for_failure` — dispatcher that picks the
    appropriate candidate set based on the failure's class
    (kwarg typo, module-path typo, etc.) so the algorithm doesn't
    conflate surface sets.
  * :func:`build_enriched_repair_prompt` — produces the prompt
    fragment that gets appended to the repair-prompt payload sent
    to the LLM. When no suggestion is plausible, the fragment is
    empty — the prompt-builder respects the algorithm's "no match"
    verdict (the ``C-PROMPT-OMITS-WHEN-NO-MATCH`` coherence check).

Discipline: **suggest with high confidence or not at all.** The
threshold + dominance check mirrors C1's symbol-gate fuzzy stage —
a misleading low-confidence hint is worse than no hint, because the
LLM treats the hint as authoritative under the directive's stance.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


# ─── Algorithm thresholds ────────────────────────────────────────────


# Minimum absolute similarity score for any suggestion to be returned.
# Mirrors C1's symbol-gate fuzzy stage (token_set_ratio >= 92).
_MIN_SCORE: float = 92.0

# Minimum margin between the best candidate and the second-best. A close
# top-pair is ambiguous — better to emit no suggestion than to risk
# pointing the LLM at the wrong neighbour.
_MIN_DOMINANCE: float = 5.0


try:
    from rapidfuzz import fuzz, process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_RAPIDFUZZ = False


# ─── Core algorithm ──────────────────────────────────────────────────


def _closest_match(
    needle: str,
    candidates: Iterable[str],
    *,
    min_score: float = _MIN_SCORE,
    min_dominance: float = _MIN_DOMINANCE,
) -> Optional[str]:
    """Return the closest candidate to ``needle``, or ``None`` if no
    candidate meets both the score threshold and the dominance bar.

    The score threshold guarantees a baseline similarity (no
    "well, this one is least-wrong" suggestions). The dominance bar
    guarantees the top match clearly separates from the next-best
    (no ambiguous suggestions when several candidates are equally
    close to the needle).

    Returns ``None`` when rapidfuzz isn't installed — the mechanism
    degrades to "no suggestion" rather than crashing the compile.
    """
    if not _HAS_RAPIDFUZZ:
        return None
    pool = list(candidates)
    if not pool:
        return None

    scored = process.extract(
        needle,
        pool,
        scorer=fuzz.token_set_ratio,
        limit=min(5, len(pool)),
    )
    if not scored:
        return None
    top_name, top_score, _ = scored[0]
    if top_score < min_score:
        return None
    if len(scored) >= 2:
        _, second_score, _ = scored[1]
        if (top_score - second_score) < min_dominance:
            return None
    return top_name


def closest_module_match(
    failing_module: str,
    surface_modules: Iterable[str],
) -> Optional[str]:
    """Suggest the closest valid module path for a typo on a Mellea
    module path. Returns ``None`` when no candidate meets the
    confidence + dominance bar — see :func:`_closest_match`.

    Used for ``import-soundness`` failures where the LLM emitted a
    near-miss like ``mellea.backens.ollama`` instead of
    ``mellea.backends.ollama``.
    """
    return _closest_match(failing_module, surface_modules)


def closest_kwarg_match(
    failing_kwarg: str,
    valid_kwargs: Iterable[str],
) -> Optional[str]:
    """Suggest the closest valid keyword-argument name for a typo on
    a kwarg. Returns ``None`` when no candidate is close enough.

    Used for ``stdlib-arity`` failures with sub-kind ``unknown-kwarg``
    — e.g. the LLM emitted ``bckend=`` for a function accepting
    ``backend=``.
    """
    return _closest_match(failing_kwarg, valid_kwargs)


# ─── Dispatcher ──────────────────────────────────────────────────────


def suggest_for_failure(failure: dict[str, Any]) -> Optional[str]:
    """Dispatch a failure record to the appropriate closest-match
    algorithm based on its class (``kind`` / ``failure_subkind``).

    The dispatch keeps candidate sets scoped to the failure class —
    a kwarg-typo searches kwarg names, a module-path-typo searches
    module paths. Without this scoping, the algorithm could conflate
    surfaces and emit structurally-wrong suggestions (a kwarg
    suggestion for a module-path failure, etc.). The
    ``C-SCOPE-MATCHES-FAILURE-CLASS`` coherence check guards against
    this regression.

    Returns ``None`` when no suggestion is plausible or the failure
    class isn't a closest-match candidate.
    """
    kind = failure.get("kind")
    subkind = failure.get("failure_subkind")

    if kind == "import-soundness":
        failing = failure.get("failing_path", "")
        candidates = failure.get("surface_modules") or ()
        if not failing or not candidates:
            return None
        return closest_module_match(failing, candidates)

    if kind == "stdlib-arity" and subkind == "unknown-kwarg":
        failing = failure.get("failing_token", "")
        candidates = failure.get("valid_kwargs") or ()
        if not failing or not candidates:
            return None
        return closest_kwarg_match(failing, candidates)

    # Other failure classes are not yet wired up to closest-match;
    # returning None matches the "no suggestion" semantic — the
    # prompt-builder will omit the hint line.
    return None


# ─── Repair-prompt enrichment ────────────────────────────────────────


def build_enriched_repair_prompt(
    failure: dict[str, Any],
    surface_modules: Optional[Iterable[str]] = None,
) -> str:
    """Produce the repair-prompt fragment for one failure.

    When a suggestion is available, the fragment includes a clear
    ``Did you mean '<suggestion>'?`` line that the LLM can act on.
    When no suggestion is available, the fragment is the failure's
    descriptive message alone — no fabricated hint
    (``C-PROMPT-OMITS-WHEN-NO-MATCH``).

    ``surface_modules`` is accepted as a convenience for callers
    that already have the module surface loaded — when provided,
    it's merged into ``failure["surface_modules"]`` so the dispatcher
    can pick it up. This lets the integration code at the repair-
    loop site avoid threading the surface through every call site.
    """
    # Merge convenience parameter into the failure record.
    if surface_modules is not None and "surface_modules" not in failure:
        failure = dict(failure)
        failure["surface_modules"] = list(surface_modules)

    suggestion = suggest_for_failure(failure)

    lines: list[str] = []
    base_message = failure.get("message")
    if base_message:
        lines.append(str(base_message))

    if suggestion is not None:
        failing = (
            failure.get("failing_path")
            or failure.get("failing_token")
            or "the emitted symbol"
        )
        lines.append(
            f"Did you mean '{suggestion}'? "
            f"(closest match in the surface to '{failing}')"
        )

    return "\n".join(lines)
