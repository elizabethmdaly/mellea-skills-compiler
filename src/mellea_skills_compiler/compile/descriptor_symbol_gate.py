"""C1 — Pre-render symbol-resolution gate for descriptor IR.

The descriptor renderer requires every ``symbol`` field in the IR to
resolve as an exact key in the introspected Mellea surface (the
``mellea_api_ref.json`` artefact). LLM emissions routinely produce
*partially-qualified* paths: bare class-method shorthand
(``MelleaSession.instruct``), bare names (``RepairTemplateStrategy``),
or module-without-class (``mellea.stdlib.session.chat`` when the
canonical key is ``mellea.stdlib.session.MelleaSession.chat``).

This module walks the descriptor before the renderer is invoked and
attempts to resolve each symbol through a deterministic multi-stage
pipeline:

  1. **Exact** — the candidate is already a valid surface path. Pass
     through unchanged.
  2. **Suffix unification** — the candidate is a unique suffix of
     exactly one surface path. Auto-normalise.
  3. **Module-scoped leaf match** — the candidate's leading segments
     form an existing module key, and exactly one symbol within that
     module ends with the candidate's trailing segment(s). Handles the
     "dropped class prefix" failure pattern (e.g.
     ``mellea.stdlib.session.chat`` →
     ``mellea.stdlib.session.MelleaSession.chat``). Auto-normalise.
  4. **Bare-name resolution** — the candidate is a single segment
     matching exactly one surface symbol's leaf name. Auto-normalise.
  5. **Fuzzy fallback** (optional, requires ``rapidfuzz``) —
     ``token_set_ratio >= 92`` AND the best match dominates the
     second-best by ``>= 5`` points. Auto-normalise.

When none of those produce a unique high-confidence match, the symbol
is flagged unresolvable and surfaced with top-3 candidates so the repair
agent (or human reviewer) has actionable suggestions.

Discipline: **fail loud on ambiguity, fix silently on certainty.**
Every silent rewrite is logged to
``intermediate/symbol_normalisations.jsonl`` so the underlying LLM
emission failure rate is observable as first-class telemetry — not
buried behind invisible auto-fixes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

LOGGER = logging.getLogger(__name__)


# RapidFuzz powers the fuzzy fallback stage (stage 4). If it's not
# installed, stages 1-3 still run; only the fuzzy fallback is skipped.
# This keeps the gate functional in stripped environments without
# requiring an additional hard dependency.
try:
    from rapidfuzz import fuzz, process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_RAPIDFUZZ = False


# Fuzzy-stage thresholds. Both must hold for an auto-normalisation:
#   * absolute score: best match's token_set_ratio >= _FUZZY_MIN_SCORE
#   * dominance: best - second_best >= _FUZZY_MIN_DOMINANCE
# Tuned per the research recommendation; tune empirically once
# telemetry accumulates.
_FUZZY_MIN_SCORE: float = 92.0
_FUZZY_MIN_DOMINANCE: float = 5.0


@dataclass
class SymbolMatch:
    """Outcome of attempting to resolve one symbol against the surface."""

    original: str
    """The symbol value the LLM emitted, verbatim."""

    resolved: Optional[str]
    """The canonical surface path. ``None`` when the gate could not find
    a unique high-confidence match — caller treats this as a failure
    and surfaces the candidates for repair."""

    method: Optional[str]
    """How the resolution was found: ``"exact" | "suffix" | "bare-name"
    | "fuzzy" | None``. ``None`` only when ``resolved is None``."""

    candidates: list[tuple[str, float]] = field(default_factory=list)
    """Top-K candidate paths with similarity scores. Populated when
    fuzzy matching ran (whether or not it produced an
    auto-normalisation); used as suggestion fodder for the repair
    prompt when resolution failed."""

    descriptor_path: str = ""
    """JSON path within the descriptor where the symbol was found.
    Used to localise errors and to write the telemetry entry."""


@dataclass
class SymbolGateResult:
    """Aggregate outcome of running the gate over a whole descriptor."""

    descriptor_after: dict
    """The descriptor with all unambiguous normalisations applied
    in-place. Safe to hand to ``render_descriptor``."""

    normalisations: list[SymbolMatch]
    """Every auto-normalisation that was performed. Each entry is the
    telemetry record for one silent rewrite."""

    unresolvable: list[SymbolMatch]
    """Symbols that did NOT resolve uniquely. Each carries the top-K
    candidate suggestions for repair."""

    @property
    def ok(self) -> bool:
        """True iff every symbol resolved (no failed entries)."""
        return not self.unresolvable


class SymbolResolutionError(Exception):
    """Raised when the descriptor symbol gate cannot resolve one or
    more symbols.

    Carries the structured :class:`SymbolGateResult` so the repair
    pipeline can format a precise error with top-K candidates for each
    failing symbol. Distinct from generic ``Exception`` so the CLI's
    typed-except dispatch can route this to a meaningful exit code.
    """

    def __init__(self, gate_result: SymbolGateResult):
        self.gate_result = gate_result
        n = len(gate_result.unresolvable)
        first = gate_result.unresolvable[0]
        summary = (
            f"Descriptor symbol gate: {n} symbol(s) could not be resolved. "
            f"First: '{first.original}' at {first.descriptor_path}"
        )
        if first.candidates:
            top = ", ".join(
                f"{p} ({s:.0f})" for p, s in first.candidates[:3]
            )
            summary += f"; closest candidates: {top}"
        super().__init__(summary)


def build_surface_index(surface: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute the lookup structures used by :func:`resolve_symbol`.

    The ``mellea_api_ref.json`` surface has the shape::

        {"modules": {
            "<module.path>": {
                "<symbol_key>": {"signature": "..."},
                "ClassName.method": {"signature": "..."},
                ...
            },
            ...
        }}

    The canonical surface path for a symbol is ``<module>.<symbol_key>``.
    From this we build:

      * ``all_paths``  — flat list of every canonical path.
      * ``by_path``    — set form of ``all_paths`` for O(1) exact lookup.
      * ``by_suffix``  — every dotted suffix of every path → list of full
        paths. Lets us look up ``MelleaSession.instruct`` directly.
      * ``by_leaf``    — leaf name (last dotted segment) → list of full
        paths. Used by the bare-name stage.
    """
    all_paths: list[str] = []
    by_suffix: dict[str, list[str]] = {}
    by_leaf: dict[str, list[str]] = {}
    # Module-scoped leaf index: for the module-scoped leaf match stage,
    # map (module_key, leaf_segment) → list of full paths where that
    # module contains a symbol whose key ends with `.<leaf>` OR equals
    # `<leaf>`. Handles "dropped class prefix" patterns.
    by_module_and_leaf: dict[tuple[str, str], list[str]] = {}
    module_keys: set[str] = set()
    # Per-module *head* symbols (single-segment top-level names only).
    # Mirrors the renderer's resolution semantics: a path is valid iff
    # the longest module prefix splits to a head attribute that exists
    # under the module. Methods on classes (``ClassName.method``) are
    # implicit attribute access and are NOT separately validated by
    # the renderer; only the class needs to be in the surface.
    head_symbols_by_module: dict[str, set[str]] = {}
    # Per-path `defined_in` from the surface metadata. Used by stages
    # 2 and 4 to collapse re-export aliases (multiple surface paths for
    # one Python symbol) into a single canonical resolution target —
    # otherwise the uniqueness predicate misclassifies aliases as
    # genuine collisions. None when the surface entry is not a dict or
    # carries no `defined_in` (e.g. introspected runtime surfaces that
    # don't expose it).
    defined_in_by_path: dict[str, Optional[str]] = {}

    modules = surface.get("modules", {}) or {}
    for module_name, module_dict in modules.items():
        if not isinstance(module_dict, dict):
            continue
        module_keys.add(module_name)
        # Two surface shapes are in use:
        #
        #   * Bundled surface (``surface_0.5.0.json``): each module
        #     value is ``{"doc": "...", "symbols": {"<key>": {...}}}``.
        #     The symbol entries are nested under the ``"symbols"`` key.
        #
        #   * Introspected runtime surface
        #     (``intermediate/mellea_api_ref.json``): each module value
        #     is a flat dict whose keys ARE the symbol keys directly.
        #
        # Detect by checking for a nested ``symbols`` sub-dict. The
        # introspected shape has no ``"symbols"`` key at this level
        # (its symbols include names like ``MelleaSession`` and
        # ``MelleaSession.instruct`` instead).
        nested_symbols = module_dict.get("symbols")
        if isinstance(nested_symbols, dict):
            symbols_iter = nested_symbols
        else:
            symbols_iter = module_dict
        head_symbols_by_module.setdefault(module_name, set())
        for symbol_key, entry in symbols_iter.items():
            # Head symbols are single-segment names (a bare class or
            # function). Methods recorded as ``ClassName.method`` add
            # their class to the head set so that the class is
            # recognised as a valid head; the method-attribute
            # validation is the renderer's job, not ours.
            if "." not in symbol_key:
                head_symbols_by_module[module_name].add(symbol_key)
            else:
                head_symbols_by_module[module_name].add(
                    symbol_key.split(".", 1)[0]
                )
            full_path = f"{module_name}.{symbol_key}"
            all_paths.append(full_path)
            # Capture `defined_in` when the surface exposes it. Bundled
            # surfaces (``surface_0.5.0.json``) carry this on every
            # symbol entry; introspected runtime surfaces may not, in
            # which case the path is treated as its own canonical
            # target by the alias-dedupe helper.
            di: Optional[str] = None
            if isinstance(entry, dict):
                di_val = entry.get("defined_in")
                if isinstance(di_val, str) and di_val:
                    di = di_val
            defined_in_by_path[full_path] = di
            # Every non-trivial dotted suffix of the path. For
            # "mellea.stdlib.session.MelleaSession.instruct" we record
            # "instruct", "MelleaSession.instruct",
            # "session.MelleaSession.instruct", etc. The empty / full
            # forms are caught by the exact stage and by_leaf
            # respectively, so we don't double-record them here.
            parts = full_path.split(".")
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                by_suffix.setdefault(suffix, []).append(full_path)
            # Leaf name (bare symbol).
            leaf = parts[-1]
            by_leaf.setdefault(leaf, []).append(full_path)
            # Module-scoped leaf: the symbol_key itself may be
            # "ClassName.method" or a bare name. We record the leaf
            # segment of symbol_key against (module, leaf).
            symbol_leaf = symbol_key.split(".")[-1]
            by_module_and_leaf.setdefault(
                (module_name, symbol_leaf), []
            ).append(full_path)

    return {
        "all_paths": all_paths,
        "by_path": set(all_paths),
        "by_suffix": by_suffix,
        "by_leaf": by_leaf,
        "by_module_and_leaf": by_module_and_leaf,
        "module_keys": module_keys,
        "head_symbols_by_module": head_symbols_by_module,
        "defined_in_by_path": defined_in_by_path,
    }


def _dedupe_alias_paths(
    paths: list[str], defined_in_by_path: dict[str, Optional[str]]
) -> list[str]:
    """Collapse re-export aliases of the same Python symbol.

    Multiple surface paths pointing to one Python symbol share a
    ``defined_in`` value in the surface metadata. The uniqueness
    predicates in stages 2 and 4 must treat such a group as ONE
    candidate, not N — otherwise bare names like ``start_session`` (
    re-exported at ``mellea.start_session`` and defined canonically at
    ``mellea.stdlib.session.start_session``) fail with spurious
    ambiguity.

    Within a same-``defined_in`` group the canonical path is the one
    rooted at the defining module — i.e. it starts with
    ``f"{defined_in}."``. Re-exports under shorter module prefixes are
    dropped; the canonical path is the resolution target the renderer
    will see.

    Paths whose surface entry lacks a ``defined_in`` (e.g. introspected
    runtime surfaces that don't expose the field) are NOT collapsed —
    each is treated as its own group so two distinct symbols without
    metadata remain ambiguous as before.
    """
    by_defined_in: dict[str, list[str]] = {}
    for p in paths:
        di = defined_in_by_path.get(p)
        # Sentinel keys for the no-metadata case keep distinct paths
        # in separate groups (= no collapse).
        key = di if di else f"__no_di__:{p}"
        by_defined_in.setdefault(key, []).append(p)
    deduped: list[str] = []
    for key, group in by_defined_in.items():
        if key.startswith("__no_di__:"):
            deduped.extend(group)
            continue
        canonical = [p for p in group if p.startswith(f"{key}.")]
        if canonical:
            deduped.append(sorted(canonical)[0])
        else:
            # No path in the group is rooted at defined_in — shouldn't
            # happen for a well-formed surface but defensive: keep an
            # arbitrary representative so we don't silently drop the
            # symbol.
            deduped.append(group[0])
    return deduped


def is_valid_for_renderer(candidate: str, index: dict[str, Any]) -> bool:
    """Return True iff ``candidate`` would be accepted by the renderer's
    own ``split_symbol`` + symbols lookup.

    Mirrors ``mellea_skills_compiler.renderer.nodes.resolve_symbol``:
    find the longest dotted prefix that is a module key in the
    surface, then check that the next segment (the *head*) is one of
    that module's top-level symbols. Subsequent attribute-chain
    segments are NOT validated — they are implicit Python attribute
    access (e.g. method calls on a class).

    Used in two places:

      * Stage 1 (exact match) — short-circuits when the LLM emitted
        the canonical form on the first try.
      * Verification after auto-normalisation — every rewrite produced
        by stages 2-5 is re-checked through this function so the
        descriptor we hand to the renderer is guaranteed acceptable.
    """
    if not candidate:
        return False
    parts = candidate.split(".")
    head_symbols = index.get("head_symbols_by_module", {})
    # Walk back from the longest module prefix to the shortest.
    for i in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:i])
        if module in head_symbols:
            head = parts[i]
            return head in head_symbols[module]
    return False


def resolve_symbol(
    candidate: str,
    index: dict[str, Any],
    descriptor_path: str = "",
) -> SymbolMatch:
    """Resolve a single symbol against the surface index.

    Implements the multi-stage matcher described in the module
    docstring. Discipline: any stage may auto-resolve only when the
    surface admits exactly one candidate; otherwise the resolution
    falls through to the next stage. Stage 4 (fuzzy) additionally
    requires the best score to clear ``_FUZZY_MIN_SCORE`` AND dominate
    the runner-up by ``_FUZZY_MIN_DOMINANCE`` points.
    """
    # Stage 1: Renderer-valid as-is. The candidate is already in the
    # form the renderer accepts: longest module prefix + head symbol
    # known to that module. (Methods on classes are implicit
    # attribute access and don't need to be separately in the surface
    # — same semantics as the renderer's ``resolve_symbol``.)
    if is_valid_for_renderer(candidate, index):
        return SymbolMatch(
            original=candidate,
            resolved=candidate,
            method="exact",
            descriptor_path=descriptor_path,
        )

    # Stage 2: Suffix unification. The LLM emitted a substring of a
    # canonical path (e.g., 'MelleaSession.instruct' or
    # 'session.MelleaSession.instruct'); auto-normalise only when the
    # surface contains exactly one path ending with this suffix.
    # Re-export aliases (multiple surface paths for one Python symbol,
    # detected via shared `defined_in`) are collapsed to the canonical
    # path before the uniqueness check.
    suffix_hits = _dedupe_alias_paths(
        index["by_suffix"].get(candidate, []),
        index["defined_in_by_path"],
    )
    if len(suffix_hits) == 1:
        return SymbolMatch(
            original=candidate,
            resolved=suffix_hits[0],
            method="suffix",
            descriptor_path=descriptor_path,
        )

    # Stage 3: Module-scoped leaf match. Handles the "dropped class
    # prefix" failure mode where the LLM emitted
    # ``<module>.<method>`` (e.g. ``mellea.stdlib.session.chat``) when
    # the canonical key is ``<module>.<class>.<method>`` (e.g.
    # ``mellea.stdlib.session.MelleaSession.chat``). We look for an
    # exact module-path prefix and a unique symbol whose leaf segment
    # equals the candidate's trailing segment.
    if "." in candidate:
        head, _, leaf = candidate.rpartition(".")
        if head in index["module_keys"]:
            module_scoped_hits = index["by_module_and_leaf"].get(
                (head, leaf), []
            )
            if len(module_scoped_hits) == 1:
                return SymbolMatch(
                    original=candidate,
                    resolved=module_scoped_hits[0],
                    method="module-scoped-leaf",
                    descriptor_path=descriptor_path,
                )

    # Stage 4: Bare-name resolution. Only fires when the candidate is a
    # single segment (no dots). Auto-normalise only when one path has
    # this leaf name. Multi-segment near-misses are handled by stages
    # 2 and 3 already; pulling them in here would risk silent
    # class-name collisions. Re-export aliases are collapsed (see
    # stage 2) so a bare name that's re-exported under multiple module
    # prefixes still resolves to its canonical defining path.
    if "." not in candidate:
        leaf_hits = _dedupe_alias_paths(
            index["by_leaf"].get(candidate, []),
            index["defined_in_by_path"],
        )
        if len(leaf_hits) == 1:
            return SymbolMatch(
                original=candidate,
                resolved=leaf_hits[0],
                method="bare-name",
                descriptor_path=descriptor_path,
            )

    # Stage 5: Fuzzy fallback. Skipped if rapidfuzz is not installed.
    candidates_with_scores: list[tuple[str, float]] = []
    if _HAS_RAPIDFUZZ and index["all_paths"]:
        top = process.extract(
            candidate,
            index["all_paths"],
            scorer=fuzz.token_set_ratio,
            limit=5,
        )
        # process.extract returns (choice, score, index) tuples.
        candidates_with_scores = [(c, float(s)) for c, s, _ in top]

        if candidates_with_scores:
            best_path, best_score = candidates_with_scores[0]
            second_score = (
                candidates_with_scores[1][1]
                if len(candidates_with_scores) > 1
                else 0.0
            )
            if (
                best_score >= _FUZZY_MIN_SCORE
                and (best_score - second_score) >= _FUZZY_MIN_DOMINANCE
            ):
                return SymbolMatch(
                    original=candidate,
                    resolved=best_path,
                    method="fuzzy",
                    candidates=candidates_with_scores,
                    descriptor_path=descriptor_path,
                )

    # Could not resolve; surface the top candidates (if any) for repair.
    return SymbolMatch(
        original=candidate,
        resolved=None,
        method=None,
        candidates=candidates_with_scores[:3],
        descriptor_path=descriptor_path,
    )


def _walk_symbols(
    node: Any,
    path: str = "",
) -> Iterator[tuple[str, str, Callable[[str], None]]]:
    """Walk a descriptor IR and yield every ``symbol`` field's location.

    Yields ``(descriptor_path, current_value, setter)`` triples. The
    setter mutates the parent dict in-place when called with a new
    value — used by :func:`run_symbol_gate` to apply normalisations to
    the descriptor before it reaches the renderer.

    Visits any field literally named ``symbol`` anywhere in the tree —
    this covers ``state[].symbol``, pipeline-node ``call``/``symbol``
    fields, strategy ``symbol`` args, and any future schema additions
    that adopt the same convention.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            child_path = f"{path}.{k}" if path else k
            if k == "symbol" and isinstance(v, str):
                # Bind `node` and `k` in the default args so each
                # yielded setter mutates the right dict-and-key pair.
                def _set(new_value: str, _d: dict = node, _k: str = k) -> None:
                    _d[_k] = new_value

                yield child_path, v, _set
            else:
                yield from _walk_symbols(v, child_path)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            child_path = f"{path}[{i}]"
            yield from _walk_symbols(item, child_path)


def _looks_like_mellea_reference(symbol: str, index: dict[str, Any]) -> bool:
    """Heuristic: is this ``symbol`` intended to be a Mellea API reference?

    The descriptor's ``symbol`` field is **overloaded** across multiple
    kinds — Mellea API calls, local-package helpers (``loader.X``,
    fully-qualified ``<package>_mellea.X``), Python type markers
    (``builtins.dict``), etc. Only Mellea API references should be
    validated against the surface; everything else is the renderer's
    or the lint suite's job downstream.

    Heuristic:

      * Symbol starts with ``mellea.`` → Mellea reference (validate).
      * Symbol's leading segment is a known top-level Mellea symbol
        in the surface (e.g. bare ``MelleaSession.instruct`` where
        ``MelleaSession`` IS in the surface) → Mellea reference,
        likely a partial qualification we want to catch and rewrite.
      * Anything else → not a Mellea reference; pass through unflagged.

    Catches the failure modes we've observed (locals, builtins,
    fully-qualified package paths) while still flagging genuine
    partial-qualification mistakes against the Mellea API.
    """
    if symbol.startswith("mellea."):
        return True
    first = symbol.split(".", 1)[0]
    head_symbols_by_module = index.get("head_symbols_by_module", {})
    for head_set in head_symbols_by_module.values():
        if first in head_set:
            return True
    return False


def run_symbol_gate(
    descriptor: dict,
    surface: dict,
    intermediate_dir: Optional[Path] = None,
) -> SymbolGateResult:
    """Apply the symbol-resolution gate to a descriptor.

    Mutates ``descriptor`` in-place for any unambiguous
    auto-normalisations and returns a :class:`SymbolGateResult`
    summarising what happened. When ``intermediate_dir`` is provided
    and any normalisations occurred, writes a telemetry record to
    ``<intermediate_dir>/symbol_normalisations.jsonl``.

    Only symbol fields recognised as Mellea API references (see
    :func:`_looks_like_mellea_reference`) are validated. Local
    helpers, Python type markers, and other non-Mellea symbols are
    passed through unflagged — the renderer / lint suite handles
    those downstream.

    The caller is expected to:

      * inspect ``result.ok`` — when ``True``, hand ``descriptor`` to
        the renderer.
      * when ``False``, raise :class:`SymbolResolutionError(result)`
        OR surface the structured failure to the repair pipeline
        (D2 enrichment).
    """
    index = build_surface_index(surface)

    normalisations: list[SymbolMatch] = []
    unresolvable: list[SymbolMatch] = []
    skipped_non_mellea: int = 0

    for desc_path, original, setter in _walk_symbols(descriptor):
        # Shape-based skip. The descriptor's ``symbol`` field is
        # overloaded — only Mellea API references should be validated
        # against the surface. Local-helper references and Python
        # type markers are passed through unflagged.
        if not _looks_like_mellea_reference(original, index):
            skipped_non_mellea += 1
            continue
        match = resolve_symbol(original, index, descriptor_path=desc_path)
        if match.resolved is None:
            unresolvable.append(match)
            continue
        if match.method == "exact":
            # No rewrite needed; the LLM emitted the canonical form.
            continue
        # Stages 2-4: auto-normalisation. Mutate the descriptor and
        # record the rewrite for telemetry.
        setter(match.resolved)
        normalisations.append(match)

    if skipped_non_mellea:
        LOGGER.debug(
            "[symbol-gate] skipped %d non-Mellea symbol(s) "
            "(local helpers / type markers / package-qualified refs)",
            skipped_non_mellea,
        )

    # Persist telemetry for any rewrites. We do NOT write an empty
    # file — absence of the file means "no normalisations needed",
    # which is the desired steady state.
    if normalisations and intermediate_dir is not None:
        telemetry_path = intermediate_dir / "symbol_normalisations.jsonl"
        try:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            with telemetry_path.open("w", encoding="utf-8") as f:
                for entry in normalisations:
                    f.write(
                        json.dumps(
                            {
                                "descriptor_path": entry.descriptor_path,
                                "original": entry.original,
                                "resolved": entry.resolved,
                                "method": entry.method,
                                "candidates": entry.candidates,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            LOGGER.info(
                "[symbol-gate] auto-normalised %d symbol(s); telemetry at %s",
                len(normalisations),
                telemetry_path.name,
            )
        except OSError as exc:  # pragma: no cover - defensive
            # Telemetry-write failure should never block the compile.
            LOGGER.warning(
                "[symbol-gate] could not persist normalisation telemetry: %s",
                exc,
            )

    if unresolvable:
        LOGGER.warning(
            "[symbol-gate] %d symbol(s) could not be resolved; first: '%s'",
            len(unresolvable),
            unresolvable[0].original,
        )

    return SymbolGateResult(
        descriptor_after=descriptor,
        normalisations=normalisations,
        unresolvable=unresolvable,
    )
