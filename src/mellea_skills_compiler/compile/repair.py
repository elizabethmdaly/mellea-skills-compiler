"""Phase 3.B — bounded descriptor-repair loop with legacy escalation.

Wraps Phase 3.A's :func:`compile_via_descriptor` with the repair-then-escalate
policy from
``melleafy-handoff/process/regression-and-extension-policy.md`` §(b) and plan
``…2026-05-16-introspection-driven-descriptor-compiler-plan.md`` §7.5.

Flow::

    1. Drive P3.A's compile_via_descriptor (emit → validate → render → smoke).
    2. If a stage fails, identify the failing descriptor node (via the source
       map for render/smoke; via the validator error path for validation),
       construct a node-focused repair prompt, call the LLM via P3.A's
       emit_descriptor with a special repair prompt (one-shot of "<repaired_node>"
       tag shape), splice the corrected node back into the descriptor by id,
       and re-run validate → render → smoke.
    3. After max_attempts_per_stage at any single stage, classify the failure
       into one of 5 escalation classifications and write
       ``<skill_root>/.melleafy-routing.json`` with ``compiler: legacy``.

This module does NOT invoke the legacy pipeline itself. That is the caller's
responsibility (Stream C / Stream D).

Hard constraints (per the kickoff brief):

* No hardcoded credentials. ``anthropic.Anthropic()`` reads from env as usual.
* No network in tests by default. The :class:`DescriptorPipeline` Protocol
  is the seam P3.A's real implementation slots into; tests substitute a
  deterministic fake.
* Idempotent: re-running on the same input twice with deterministic mocks
  yields the same outcome shape.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


EscalationClassification = Literal[
    "operator-gap",
    "expressivity-gap",
    "repair-loop-cost",
    "infra-failure",
    "surface-gap",
]

Stage = Literal["emission", "validation", "render", "smoke"]


# Canonical 5 composition operators — the curated set from plan §0/§4.5 and the
# regression-and-extension-policy. The repair loop uses this to classify
# operator-gap failures. If/when an RFC promotes a 6th operator, update this
# tuple AND ``descriptor/semantic_rules._OPERATOR_REQUIRED_FIELDS`` in the same
# patch.
CANONICAL_OPERATORS: tuple[str, ...] = (
    "sequential",
    "branch",
    "parallel",
    "map",
    "retry_with_feedback",
)


@dataclass
class RepairAttempt:
    """Record of one repair iteration."""

    stage: Stage  # "validation" | "render" | "smoke"
    node_id: str | None
    failure_message: str
    repair_prompt: str
    repaired_node: dict | None
    succeeded: bool
    error: str | None = None


@dataclass
class RepairOutcome:
    """Result of :func:`compile_with_repair`."""

    final_compile_result: Any  # P3.A's CompileResult — duck-typed.
    attempts: list[RepairAttempt] = field(default_factory=list)
    escalated_to_legacy: bool = False
    escalation_classification: EscalationClassification | None = None
    routing_manifest_written_to: Path | None = None


@dataclass
class RepairConfig:
    """Repair loop budgets and tunables."""

    max_attempts_per_stage: int = 2  # per validation, render, smoke
    repair_max_tokens: int = 16_000
    model: str = "aws/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Pipeline protocol (the seam to P3.A)
# ---------------------------------------------------------------------------


@runtime_checkable
class DescriptorPipeline(Protocol):
    """Integration surface this module requires from P3.A.

    Three methods only:

    - :meth:`compile` — the full happy path: emit → validate → render → smoke.
      Returns whatever shape the underlying compiler produces; this module
      reads attributes via the :func:`_view_result` helper.
    - :meth:`re_run` — re-run validate → render → smoke on a spliced
      descriptor, without re-emitting. Returns the same shape as :meth:`compile`.
    - :meth:`emit_repaired_node` — call the LLM with a focused repair prompt;
      return the parsed corrected node dict (with ``id``) or ``None`` if
      parsing failed.
    """

    def compile(
        self,
        spec_text: str,
        surface: dict,
        skill_root: Path,
        emission_config: Any | None,
        *,
        client: Any | None = None,
    ) -> Any:
        """Emit descriptor → validate → render → smoke."""

    def re_run(
        self,
        descriptor: dict,
        surface: dict,
        skill_root: Path,
        *,
        client: Any | None = None,
    ) -> Any:
        """Validate → render → smoke on an already-emitted (and possibly spliced) descriptor."""

    def emit_repaired_node(
        self,
        *,
        repair_prompt: str,
        repair_config: RepairConfig,
        client: Any | None = None,
    ) -> dict | None:
        """Call the LLM and parse the response into a single repaired node."""


# ---------------------------------------------------------------------------
# CompileResult duck-typed view
# ---------------------------------------------------------------------------


@dataclass
class _ResultView:
    """Normalised view over P3.A's CompileResult (or a test fake)."""

    succeeded: bool
    stage_failed: Stage | None
    descriptor: dict | None
    validation_errors: list[dict]
    render_error: str | None
    render_traceback: str | None
    smoke_error: str | None
    smoke_traceback: str | None
    source_map_entries: list[dict]


def _view_result(result: Any) -> _ResultView:
    """Normalise any duck-typed result into the shape the repair loop reads.

    Tolerates two shapes:

    1. **P3.A's CompileResult** — fields: ``emission``, ``validation``,
       ``failure_stage``, ``failure_reason``, ``source_map``, ``smoke_ok``,
       ``pipeline_py``. Validation errors live on
       ``result.validation.errors`` (list of ValidationError objects).
    2. **A simple test fake** — exposes ``succeeded``, ``stage_failed``,
       ``descriptor``, ``validation_errors`` (list[dict]), ``render_error``,
       ``render_traceback``, ``smoke_error``, ``smoke_traceback``,
       ``source_map`` (list[dict]). Designed for unit tests so they don't
       have to construct a real P3.A CompileResult.
    """
    # Fast path: object already looks like _ResultView (or test fake).
    if hasattr(result, "succeeded") and hasattr(result, "stage_failed"):
        validation_errors_raw = getattr(result, "validation_errors", []) or []
        validation_errors: list[dict] = []
        for ve in validation_errors_raw:
            if isinstance(ve, dict):
                validation_errors.append(ve)
            else:
                validation_errors.append(
                    {
                        "path": getattr(ve, "path", ""),
                        "rule": getattr(ve, "rule", ""),
                        "message": getattr(ve, "message", ""),
                    }
                )
        smap = getattr(result, "source_map", None) or []
        if hasattr(smap, "to_json"):
            smap = smap.to_json()
        elif not isinstance(smap, list):
            smap = []
        return _ResultView(
            succeeded=bool(result.succeeded),
            stage_failed=getattr(result, "stage_failed", None),
            descriptor=getattr(result, "descriptor", None),
            validation_errors=validation_errors,
            render_error=getattr(result, "render_error", None),
            render_traceback=getattr(result, "render_traceback", None),
            smoke_error=getattr(result, "smoke_error", None),
            smoke_traceback=getattr(result, "smoke_traceback", None),
            source_map_entries=smap,
        )

    # P3.A shape: ``failure_stage`` is either None or one of the stage strings.
    failure_stage: Stage | None = getattr(result, "failure_stage", None)
    succeeded = failure_stage is None
    emission = getattr(result, "emission", None)
    descriptor = getattr(emission, "descriptor", None) if emission is not None else None

    validation = getattr(result, "validation", None)
    validation_errors: list[dict] = []
    if validation is not None and getattr(validation, "errors", None):
        for ve in validation.errors:
            validation_errors.append(
                {
                    "path": getattr(ve, "path", ""),
                    "rule": getattr(ve, "rule", ""),
                    "message": getattr(ve, "message", ""),
                }
            )

    smap = getattr(result, "source_map", None)
    smap_entries: list[dict] = []
    if smap is not None and hasattr(smap, "to_json"):
        smap_entries = smap.to_json()
    elif isinstance(smap, list):
        smap_entries = smap

    failure_reason = getattr(result, "failure_reason", None)
    # Bucket the reason text into render or smoke depending on stage.
    render_error = failure_reason if failure_stage == "render" else None
    smoke_error = failure_reason if failure_stage == "smoke" else None
    # P3.A's _smoke_import bundles the traceback into failure_reason already;
    # we treat the same string as both message and traceback for source-map
    # lookup. validation_errors below already cover the validation case.
    render_traceback = render_error
    smoke_traceback = smoke_error

    return _ResultView(
        succeeded=succeeded,
        stage_failed=failure_stage,
        descriptor=descriptor,
        validation_errors=validation_errors,
        render_error=render_error,
        render_traceback=render_traceback,
        smoke_error=smoke_error,
        smoke_traceback=smoke_traceback,
        source_map_entries=smap_entries,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compile_with_repair(
    spec_text: str,
    surface: dict,
    skill_root: Path,
    emission_config: Any | None = None,
    repair_config: RepairConfig | None = None,
    *,
    client: Any | None = None,
    pipeline: DescriptorPipeline | None = None,
) -> RepairOutcome:
    """Wrap descriptor compilation with a bounded repair cascade.

    Args:
        spec_text: skill spec markdown.
        surface: the introspected Mellea surface (parsed JSON dict).
        skill_root: directory where ``.melleafy-routing.json`` is written on
            escalation. Must exist.
        emission_config: passed through to P3.A on the initial compile call.
        repair_config: budgets/tunables for the repair loop.
        client: optional Anthropic client. If ``None``, P3.A constructs its
            own.
        pipeline: optional injection point for the descriptor pipeline.
            Production callers leave this ``None`` and the module loads
            P3.A's default; tests inject a mock.

    Returns:
        A :class:`RepairOutcome`. Inspect ``escalated_to_legacy`` to decide
        whether to invoke the legacy pipeline.
    """
    repair_config = repair_config or RepairConfig()
    if pipeline is None:
        pipeline = _load_default_pipeline()

    # --- Initial happy-path compile -----------------------------------------
    result = pipeline.compile(
        spec_text=spec_text,
        surface=surface,
        skill_root=skill_root,
        emission_config=emission_config,
        client=client,
    )

    outcome = RepairOutcome(final_compile_result=result)
    view = _view_result(result)
    if view.succeeded:
        return outcome

    # Emission itself failed — there is no descriptor to splice. Escalate.
    if view.stage_failed == "emission" or view.descriptor is None:
        return _escalate(
            outcome=outcome,
            result=result,
            skill_root=skill_root,
            classification="expressivity-gap",
            summary="LLM failed to emit a descriptor",
        )

    # --- Repair loop --------------------------------------------------------
    current = result
    current_view = view
    for stage in ("validation", "render", "smoke"):
        if current_view.succeeded:
            break
        if current_view.stage_failed != stage:
            # Stage transitioned forward (e.g. validation→render); next loop
            # iteration will pick it up.
            continue

        # Early classification: operator-gap and surface-gap don't benefit
        # from repair (LLM will likely re-emit the same mistake).
        early = _classify_no_repair(current_view, surface)
        if early is not None:
            classification, summary = early
            if classification == "surface-gap":
                # Per policy §(b): surface-gap is Mellea version drift, not
                # a skill problem — surface to caller without auto-escalation.
                outcome.escalation_classification = "surface-gap"
                outcome.escalated_to_legacy = False
                return outcome
            return _escalate(
                outcome=outcome,
                result=current,
                skill_root=skill_root,
                classification=classification,
                summary=summary,
            )

        current = _repair_stage(
            stage=stage,  # type: ignore[arg-type]
            initial_result=current,
            pipeline=pipeline,
            surface=surface,
            skill_root=skill_root,
            repair_config=repair_config,
            client=client,
            outcome=outcome,
        )
        outcome.final_compile_result = current
        current_view = _view_result(current)

    if current_view.succeeded:
        return outcome

    # All repairs exhausted. Classify and decide.
    classification, summary = _classify_after_exhaustion(
        current_view, outcome.attempts, surface
    )
    if classification in ("infra-failure", "surface-gap"):
        # Surface to caller without writing the routing manifest.
        outcome.escalation_classification = classification
        outcome.escalated_to_legacy = False
        return outcome

    return _escalate(
        outcome=outcome,
        result=current,
        skill_root=skill_root,
        classification=classification,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Per-stage repair
# ---------------------------------------------------------------------------


def _repair_stage(
    *,
    stage: Stage,
    initial_result: Any,
    pipeline: DescriptorPipeline,
    surface: dict,
    skill_root: Path,
    repair_config: RepairConfig,
    client: Any | None,
    outcome: RepairOutcome,
) -> Any:
    """Run up to ``max_attempts_per_stage`` repair attempts at a single stage."""
    current = initial_result
    current_view = _view_result(current)

    for _attempt in range(repair_config.max_attempts_per_stage):
        node_id, failure_message = _identify_failing_node(current_view, stage)
        descriptor = current_view.descriptor
        if descriptor is None or node_id is None:
            outcome.attempts.append(
                RepairAttempt(
                    stage=stage,
                    node_id=None,
                    failure_message=failure_message,
                    repair_prompt="",
                    repaired_node=None,
                    succeeded=False,
                    error="could not identify failing descriptor node",
                )
            )
            return current

        offending = _find_node_by_id(descriptor, node_id)
        prompt = _build_repair_prompt(
            stage=stage,
            failing_node=offending,
            failure_message=failure_message,
            descriptor=descriptor,
            surface=surface,
        )

        try:
            repaired = pipeline.emit_repaired_node(
                repair_prompt=prompt,
                repair_config=repair_config,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 — record any LLM-side error
            outcome.attempts.append(
                RepairAttempt(
                    stage=stage,
                    node_id=node_id,
                    failure_message=failure_message,
                    repair_prompt=prompt,
                    repaired_node=None,
                    succeeded=False,
                    error=f"LLM call failed: {type(exc).__name__}: {exc}",
                )
            )
            return current

        if not isinstance(repaired, dict) or "id" not in repaired:
            outcome.attempts.append(
                RepairAttempt(
                    stage=stage,
                    node_id=node_id,
                    failure_message=failure_message,
                    repair_prompt=prompt,
                    repaired_node=repaired if isinstance(repaired, dict) else None,
                    succeeded=False,
                    error="LLM did not return a node with an 'id' field",
                )
            )
            continue

        try:
            patched = _splice_node(descriptor, repaired)
        except ValueError as exc:
            outcome.attempts.append(
                RepairAttempt(
                    stage=stage,
                    node_id=node_id,
                    failure_message=failure_message,
                    repair_prompt=prompt,
                    repaired_node=repaired,
                    succeeded=False,
                    error=f"splice failed: {exc}",
                )
            )
            continue

        new_result = pipeline.re_run(
            descriptor=patched,
            surface=surface,
            skill_root=skill_root,
            client=client,
        )
        new_view = _view_result(new_result)

        outcome.attempts.append(
            RepairAttempt(
                stage=stage,
                node_id=node_id,
                failure_message=failure_message,
                repair_prompt=prompt,
                repaired_node=repaired,
                succeeded=new_view.succeeded or (new_view.stage_failed != stage),
                error=None,
            )
        )
        current = new_result
        current_view = new_view
        if new_view.succeeded:
            return current
        if new_view.stage_failed != stage:
            # Progress: outer loop will pick up the new stage.
            return current

    return current


# ---------------------------------------------------------------------------
# Failing-node identification
# ---------------------------------------------------------------------------


_TRACEBACK_LINE_RE = re.compile(r'pipeline\.py", line (\d+)')
_PATH_EXPR_RE = re.compile(r"(\w+)|\[(\d+)\]")


def _identify_failing_node(view: _ResultView, stage: Stage) -> tuple[str | None, str]:
    """Return (node_id, human-readable failure message) for the failing stage."""
    if stage == "validation":
        for err in view.validation_errors:
            path = err.get("path", "")
            node_id = _node_id_from_descriptor_path(path, view.descriptor or {})
            if node_id is not None:
                return node_id, err.get("message", "validation error")
        if view.validation_errors:
            return None, view.validation_errors[0].get("message", "validation error")
        return None, "unknown validation failure"

    if stage == "render":
        err = view.render_error or "render error"
        m = re.search(r"at descriptor path:\s*([^)]+?)\)", err)
        if m:
            expr = m.group(1).strip()
            node_id = _node_id_from_path_expression(expr, view.descriptor or {})
            if node_id is not None:
                return node_id, err
        # Fall back: scan validation_errors (P2.D rules may have run in
        # render context). Not always populated.
        return None, err

    if stage == "smoke":
        tb = view.smoke_traceback or view.smoke_error or ""
        m = _TRACEBACK_LINE_RE.search(tb)
        if m and view.source_map_entries:
            line_no = int(m.group(1))
            node_id = _source_map_lookup(view.source_map_entries, line_no)
            if node_id is not None:
                return node_id, view.smoke_error or tb
        return None, view.smoke_error or "smoke check failed"

    return None, "unknown stage"


def _source_map_lookup(entries: list[dict], line_no: int) -> str | None:
    """O(N) lookup over the serialised source map.

    Mirrors :meth:`SourceMap.lookup` — inner-most (smallest-span) entry
    wins for any given line.
    """
    best: dict | None = None
    best_span = float("inf")
    for entry in entries:
        start = entry.get("start_line", 0)
        end = entry.get("end_line", 0)
        if start <= line_no <= end:
            span = end - start
            if span < best_span:
                best = entry
                best_span = span
    if best is None:
        return None
    return best.get("descriptor_node_id")


def _node_id_from_descriptor_path(json_pointer: str, descriptor: dict) -> str | None:
    """Walk a JSON Pointer to the enclosing pipeline node's ``id``."""
    if not json_pointer or not json_pointer.startswith("/"):
        return None
    parts = json_pointer.split("/")[1:]
    cursor: Any = descriptor
    last_node_with_id: str | None = None
    for raw in parts:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cursor, list):
            try:
                cursor = cursor[int(token)]
            except (ValueError, IndexError):
                break
        elif isinstance(cursor, dict):
            if isinstance(cursor.get("id"), str):
                last_node_with_id = cursor["id"]
            cursor = cursor.get(token)
        else:
            break
        if isinstance(cursor, dict) and isinstance(cursor.get("id"), str):
            last_node_with_id = cursor["id"]
    return last_node_with_id


def _node_id_from_path_expression(expr: str, descriptor: dict) -> str | None:
    """Resolve ``pipeline[1].body[0]``-style paths to a node id."""
    cursor: Any = descriptor
    last_node_with_id: str | None = None
    for m in _PATH_EXPR_RE.finditer(expr):
        if m.group(2) is not None:
            try:
                cursor = cursor[int(m.group(2))]
            except (TypeError, IndexError, ValueError):
                break
        else:
            key = m.group(1)
            if isinstance(cursor, dict):
                cursor = cursor.get(key)
            else:
                break
        if isinstance(cursor, dict) and isinstance(cursor.get("id"), str):
            last_node_with_id = cursor["id"]
    return last_node_with_id


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------


def _find_node_by_id(descriptor: dict, node_id: str) -> dict | None:
    """Locate a node anywhere in the descriptor's pipeline by ``id``."""
    return _walk_for_id(descriptor.get("pipeline", []), node_id)


def _walk_for_id(nodes: list[dict], node_id: str) -> dict | None:
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if n.get("id") == node_id:
            return n
        if n.get("kind") == "composition":
            for child_list in _composition_child_lists(n):
                found = _walk_for_id(child_list, node_id)
                if found is not None:
                    return found
    return None


def _composition_child_lists(node: dict) -> list[list[dict]]:
    out: list[list[dict]] = []
    body = node.get("body")
    if isinstance(body, list):
        out.append(body)
    for branch in (node.get("cases") or {}).values():
        if isinstance(branch, list):
            out.append(branch)
    for branch in node.get("branches") or []:
        if isinstance(branch, list):
            out.append(branch)
    return out


def _splice_node(descriptor: dict, repaired: dict) -> dict:
    """Return a deep-copied descriptor with the matching-id node replaced.

    Matches by ``id``; the repaired node must carry an ``id``. Preserves all
    other nodes verbatim. Handles nested composition shapes (``body``,
    ``cases``, ``branches``).
    """
    target_id = repaired.get("id")
    if not isinstance(target_id, str):
        raise ValueError("repaired node missing 'id'; cannot splice")
    new_descriptor = json.loads(json.dumps(descriptor))
    pipeline = new_descriptor.get("pipeline", [])
    replaced = _splice_into_list(pipeline, target_id, repaired)
    if not replaced:
        raise ValueError(
            f"repaired node id {target_id!r} not found in descriptor pipeline"
        )
    return new_descriptor


def _splice_into_list(nodes: list[dict], target_id: str, repaired: dict) -> bool:
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("id") == target_id:
            nodes[i] = json.loads(json.dumps(repaired))
            return True
        if node.get("kind") == "composition":
            for child_list in _composition_child_lists(node):
                if _splice_into_list(child_list, target_id, repaired):
                    return True
    return False


# ---------------------------------------------------------------------------
# Repair prompts
# ---------------------------------------------------------------------------


# Approximate token budget: 1 token ~ 4 chars for English; we cap the raw
# character length so the prompt stays well under 8K tokens.
_REPAIR_PROMPT_CHAR_BUDGET = 32_000


def _build_repair_prompt(
    *,
    stage: Stage,
    failing_node: dict | None,
    failure_message: str,
    descriptor: dict,
    surface: dict,
) -> str:
    """Construct a node-focused repair prompt.

    The prompt is intentionally narrow: only the offending node, only the
    relevant error, only the relevant symbol's introspected signature (for
    render/smoke) or the relevant 2-3 lines of schema doc (for validation).
    Response shape: ``<repaired_node>{...}</repaired_node>``.
    """
    node_json = json.dumps(failing_node or {}, indent=2)
    if len(node_json) > 8_000:
        node_json = node_json[:8_000] + "\n... (truncated)"

    descriptor_version = descriptor.get("descriptor_version", "0.2")
    mellea_version = descriptor.get("mellea_version", "0.5.0")

    header = (
        f"You are repairing a single node in a Mellea descriptor "
        f"(v{descriptor_version}, Mellea {mellea_version}).\n"
        f"Stage: {stage}.\n"
        "Return ONLY the corrected node as JSON inside "
        "<repaired_node>...</repaired_node>. "
        "Do NOT re-emit the full descriptor. Preserve the node 'id' so it "
        "can be spliced back.\n"
    )

    if stage == "validation":
        body = (
            f"## Failure\n{failure_message}\n\n"
            f"## Relevant schema rule\n{_validation_rule_hint(failure_message)}\n\n"
            f"## Offending node\n```json\n{node_json}\n```\n\n"
            "Fix the node to satisfy the rule. Keep the same 'id'.\n"
        )
    elif stage == "render":
        body = (
            f"## Renderer error\n{failure_message}\n\n"
            f"## Symbol signature (from introspected surface)\n"
            f"{_signature_excerpt(failing_node, surface)}\n\n"
            f"## Offending node\n```json\n{node_json}\n```\n\n"
            "Adjust args / refs so the node renders. Keep the same 'id'.\n"
        )
    elif stage == "smoke":
        body = (
            f"## Python traceback\n{failure_message}\n\n"
            f"## Relevant Mellea API snippet\n"
            f"{_smoke_doc_hint(failing_node, surface)}\n\n"
            f"## Offending node\n```json\n{node_json}\n```\n\n"
            "Fix the node so the rendered Python executes. Keep the same 'id'.\n"
        )
    else:
        body = f"## Failure\n{failure_message}\n\n## Node\n```json\n{node_json}\n```\n"

    prompt = header + "\n" + body
    if len(prompt) > _REPAIR_PROMPT_CHAR_BUDGET:
        prompt = prompt[:_REPAIR_PROMPT_CHAR_BUDGET] + "\n... (truncated)"
    return prompt


def _validation_rule_hint(message: str) -> str:
    """Return a 2-3 line hint for the failing validator rule."""
    msg = message.lower()
    if "symbol" in msg and "does not resolve" in msg:
        return (
            "R-SEM-SYMBOL: every 'symbol' must be a dotted path that resolves in "
            "surface.modules. Use only symbols listed in the surface."
        )
    if "ref" in msg and "does not resolve" in msg:
        return (
            "R-SEM-REF: every {ref} must point to a declared state.id, an "
            "inputs[].name, or a prior pipeline node id."
        )
    if "schema_ref" in msg:
        return (
            "R-SEM-SCHEMA-REF: every schema_ref must be '#/schemas/<Name>' where "
            "<Name> is a key declared in the top-level 'schemas' map."
        )
    if "operator" in msg:
        return (
            "R-SEM-OPERATOR: composition operators are limited to "
            f"{CANONICAL_OPERATORS}. Each has required fields — see the schema."
        )
    if "on_parse_failure" in msg:
        return (
            "R-SEM-FMT-DEFAULT: when on_parse_failure.strategy == 'default', the "
            "default's top-level fields must match the format.schema_ref schema."
        )
    return "See descriptor schema v0.2 RFC for the rule definition."


def _signature_excerpt(failing_node: dict | None, surface: dict) -> str:
    if not failing_node:
        return "(no node)"
    symbol = failing_node.get("symbol") or ""
    if not symbol:
        return "(node has no symbol)"
    sig = _resolve_signature(symbol, surface)
    if sig is None:
        return f"{symbol}: not found in surface"
    return f"{symbol}{sig}"


def _resolve_signature(symbol: str, surface: dict) -> str | None:
    modules = surface.get("modules", {})
    parts = symbol.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        if module_path in modules:
            cursor = modules[module_path].get("symbols", {})
            remaining = parts[i:]
            for j, name in enumerate(remaining):
                sym = cursor.get(name) if isinstance(cursor, dict) else None
                if sym is None:
                    return None
                if j == len(remaining) - 1:
                    return sym.get("signature") or "(...)"
                cursor = sym.get("members", {}) if isinstance(sym, dict) else {}
            return None
    return None


def _smoke_doc_hint(failing_node: dict | None, surface: dict) -> str:
    if not failing_node:
        return "(no node)"
    symbol = failing_node.get("symbol") or ""
    if not symbol:
        return "(node has no symbol)"
    modules = surface.get("modules", {})
    parts = symbol.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        if module_path in modules:
            cursor = modules[module_path].get("symbols", {})
            remaining = parts[i:]
            for j, name in enumerate(remaining):
                sym = cursor.get(name) if isinstance(cursor, dict) else None
                if sym is None:
                    return f"{symbol}: not documented in surface"
                if j == len(remaining) - 1:
                    doc = (
                        sym.get("doc")
                        or sym.get("docstring")
                        or sym.get("signature")
                        or ""
                    )
                    return f"{symbol}\n{doc[:500]}"
                cursor = sym.get("members", {}) if isinstance(sym, dict) else {}
    return f"{symbol}: not documented in surface"


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------


def _classify_no_repair(
    view: _ResultView, surface: dict
) -> tuple[EscalationClassification, str] | None:
    """Pre-repair classifications that should skip the repair loop."""
    descriptor = view.descriptor
    if not descriptor:
        return None

    invalid_operator = _find_invalid_operator(descriptor.get("pipeline", []))
    if invalid_operator is not None:
        return "operator-gap", (
            f"descriptor uses operator {invalid_operator!r} not in canonical set "
            f"{CANONICAL_OPERATORS}"
        )

    if view.stage_failed == "validation":
        for err in view.validation_errors:
            if (err.get("rule") or "") == "R-SEM-SYMBOL":
                return "surface-gap", err.get("message", "symbol not in surface")

    if view.stage_failed == "render" and view.render_error:
        err = view.render_error.lower()
        if "not in surface" in err or "unknown symbol" in err:
            return "surface-gap", view.render_error
    return None


def _find_invalid_operator(nodes: list[dict]) -> str | None:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("kind") == "composition":
            op = node.get("operator")
            if op not in CANONICAL_OPERATORS:
                return op if isinstance(op, str) else "<missing>"
            for child_list in _composition_child_lists(node):
                nested = _find_invalid_operator(child_list)
                if nested is not None:
                    return nested
    return None


def _classify_after_exhaustion(
    view: _ResultView, attempts: list[RepairAttempt], surface: dict
) -> tuple[EscalationClassification, str]:
    """Pick a classification once the per-stage budget is exhausted."""
    # Re-check no-repair classifications.
    early = _classify_no_repair(view, surface)
    if early is not None:
        return early

    tb = (view.render_traceback or "") + "\n" + (view.smoke_traceback or "")
    if _looks_like_infra_failure(tb):
        return "infra-failure", (
            "validator/renderer/smoke raised an unexpected exception"
        )

    # ≥3 distinct validation-rule classes hit → descriptor structure is wrong.
    distinct_rules: set[str] = set()
    for err in view.validation_errors:
        rule = err.get("rule") or ""
        if rule:
            distinct_rules.add(rule)
    for attempt in attempts:
        if attempt.stage == "validation":
            for r in (
                "R-SEM-SYMBOL",
                "R-SEM-REF",
                "R-SEM-SCHEMA-REF",
                "R-SEM-OPERATOR",
                "R-SEM-FMT-DEFAULT",
            ):
                if r in attempt.failure_message:
                    distinct_rules.add(r)
    if len(distinct_rules) >= 3:
        return "expressivity-gap", (
            f"≥3 distinct validation rules failed across attempts: "
            f"{sorted(distinct_rules)}"
        )

    return "repair-loop-cost", (
        f"max_attempts_per_stage exhausted at stage={view.stage_failed!r}; "
        f"{len(attempts)} repair attempt(s) made"
    )


_INFRA_EXCEPTION_NAMES = (
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "IsADirectoryError",
    "NotADirectoryError",
    "ImportError",
    "ModuleNotFoundError",
    "MemoryError",
    "RecursionError",
)


def _looks_like_infra_failure(traceback_text: str) -> bool:
    """Heuristic: an OSError/ImportError in code NOT under the rendered package.

    Conservative rule: only classify as infra-failure if an infra-exception
    name appears AND the traceback does not mention ``pipeline.py``. A
    pipeline.py mention means the failure is attributable to a descriptor
    node (and thus repairable in principle).
    """
    if not traceback_text:
        return False
    if "pipeline.py" in traceback_text:
        return False
    return any(name in traceback_text for name in _INFRA_EXCEPTION_NAMES)


# ---------------------------------------------------------------------------
# Escalation: write routing manifest
# ---------------------------------------------------------------------------


def _escalate(
    *,
    outcome: RepairOutcome,
    result: Any,
    skill_root: Path,
    classification: EscalationClassification,
    summary: str,
) -> RepairOutcome:
    """Write the .melleafy-routing.json (unless one already exists with legacy)."""
    manifest_path = skill_root / ".melleafy-routing.json"
    today = datetime.now(timezone.utc).date()
    review = today + timedelta(days=30)
    reason = f"Auto-escalated {today.isoformat()}: {classification} — {summary}".strip()

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("compiler") == "legacy":
            outcome.escalated_to_legacy = True
            outcome.escalation_classification = classification
            outcome.routing_manifest_written_to = manifest_path
            outcome.final_compile_result = result
            return outcome

    manifest = {
        "compiler": "legacy",
        "reason": reason,
        "decided_at": today.isoformat(),
        "review_at": review.isoformat(),
    }
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        LOGGER.error(
            "Failed to write routing manifest at %s: %s", manifest_path, exc
        )

    outcome.escalated_to_legacy = True
    outcome.escalation_classification = classification
    outcome.routing_manifest_written_to = manifest_path
    outcome.final_compile_result = result
    return outcome


# ---------------------------------------------------------------------------
# Default pipeline (the P3.A adapter)
# ---------------------------------------------------------------------------


class _DefaultPipeline:
    """Adapts P3.A's :mod:`descriptor_emission` module to the
    :class:`DescriptorPipeline` Protocol.

    P3.A exposes:

    * ``compile_via_descriptor(spec_text, surface, config=, write_to=, client=)``
      → ``CompileResult``.
    * ``emit_descriptor(spec_text, surface, config=, client=)`` →
      ``EmissionResult``.

    P3.A does NOT expose a ``re_run`` (validate→render→smoke without
    emission). We synthesise one from the renderer + validator + smoke
    pieces P3.A already factored out.

    For ``emit_repaired_node``, we build a one-shot message to the LLM
    asking for ``<repaired_node>...</repaired_node>`` and parse it. We
    deliberately do NOT route this through P3.A's
    ``emit_descriptor`` — that builds the heavy emission prompt with the
    full schema + surface + 1-shot; a repair call is much narrower.
    """

    def compile(self, *, spec_text, surface, skill_root, emission_config, client=None):  # type: ignore[override]
        from mellea_skills_compiler.compile.descriptor_emission import (
            compile_via_descriptor,
        )

        return compile_via_descriptor(
            spec_text=spec_text,
            surface=surface,
            config=emission_config,
            client=client,
        )

    def re_run(self, *, descriptor, surface, skill_root, client=None):  # type: ignore[override]
        return _run_validate_render_smoke(descriptor, surface)

    def emit_repaired_node(self, *, repair_prompt, repair_config, client=None):  # type: ignore[override]
        return _llm_repair_call(
            prompt=repair_prompt,
            repair_config=repair_config,
            client=client,
        )


def _run_validate_render_smoke(descriptor: dict, surface: dict) -> Any:
    """Drive P3.A's validate → render → smoke pipeline without re-emitting.

    Returns a ``CompileResult``-shaped object (uses P3.A's
    ``CompileResult`` so :func:`_view_result` reads it the same way as
    P3.A's ``compile_via_descriptor`` output).
    """
    # Local imports to avoid loading the renderer / SDK at module-import time.
    from mellea_skills_compiler.compile import descriptor_emission as _de
    from mellea_skills_compiler.descriptor import validate
    from mellea_skills_compiler.renderer import RendererError

    # Synthetic emission record so the result shape is unchanged.
    synth_emission = _de.EmissionResult(
        raw_output="",
        reasoning_text=None,
        descriptor=descriptor,
        descriptor_text=json.dumps(descriptor),
        parse_error=None,
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        latency_seconds=0.0,
        model="repair-rerun",
    )

    validation = validate(descriptor, schema_version="0.2", surface=surface)
    if not validation.valid:
        first_err = validation.errors[0] if validation.errors else None
        reason = (
            f"{first_err.rule} at {first_err.path or '<root>'}: {first_err.message}"
            if first_err is not None
            else "validation failed"
        )
        return _de.CompileResult(
            emission=synth_emission,
            validation=validation,
            pipeline_py=None,
            schemas_py=None,
            init_py=None,
            fixtures_py=None,
            melleafy_json=None,
            setup_md=None,
            readme_md=None,
            source_map=None,
            smoke_ok=False,
            failure_stage="validation",
            failure_reason=reason,
        )

    try:
        (
            pipeline_py,
            schemas_py,
            init_py,
            fixtures_py,
            melleafy_json,
            setup_md,
            readme_md,
            source_map,
            _render_result,
        ) = _de._render_full_package(descriptor, surface)
    except RendererError as exc:
        return _de.CompileResult(
            emission=synth_emission,
            validation=validation,
            pipeline_py=None,
            schemas_py=None,
            init_py=None,
            fixtures_py=None,
            melleafy_json=None,
            setup_md=None,
            readme_md=None,
            source_map=None,
            smoke_ok=False,
            failure_stage="render",
            failure_reason=str(exc),
        )

    smoke_ok, smoke_error = _de._smoke_import(
        descriptor, pipeline_py, schemas_py, init_py, fixtures_py
    )
    if not smoke_ok:
        return _de.CompileResult(
            emission=synth_emission,
            validation=validation,
            pipeline_py=pipeline_py,
            schemas_py=schemas_py,
            init_py=init_py,
            fixtures_py=fixtures_py,
            melleafy_json=melleafy_json,
            setup_md=setup_md,
            readme_md=readme_md,
            source_map=source_map,
            smoke_ok=False,
            failure_stage="smoke",
            failure_reason=smoke_error or "smoke import failed",
        )

    return _de.CompileResult(
        emission=synth_emission,
        validation=validation,
        pipeline_py=pipeline_py,
        schemas_py=schemas_py,
        init_py=init_py,
        fixtures_py=fixtures_py,
        melleafy_json=melleafy_json,
        setup_md=setup_md,
        readme_md=readme_md,
        source_map=source_map,
        smoke_ok=True,
        failure_stage=None,
        failure_reason=None,
    )


def _llm_repair_call(
    *,
    prompt: str,
    repair_config: RepairConfig,
    client: Any | None = None,
) -> dict | None:
    """Single LLM call asking for one repaired node.

    Uses the same Anthropic SDK contract as P3.A's :func:`emit_descriptor`
    but with a narrow system + user prompt and ``max_tokens =
    repair_config.repair_max_tokens``. Streams the response (the SDK
    refuses non-streaming above ~16K, and we want headroom).

    Returns the parsed node dict, or ``None`` on any parse failure.
    """
    if client is None:
        try:
            import anthropic  # local import
        except ImportError:
            return None
        client = anthropic.Anthropic()

    system_blocks = [
        {
            "type": "text",
            "text": (
                "You repair a single node of a Mellea descriptor. "
                "Respond with ONE JSON object inside "
                "<repaired_node>...</repaired_node>. No prose around it. "
                "Preserve the 'id' field exactly as given."
            ),
        }
    ]
    try:
        with client.messages.stream(
            model=repair_config.model,
            max_tokens=repair_config.repair_max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = stream.get_final_message()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("repair LLM call failed: %s", exc)
        return None

    text_chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text_chunks.append(getattr(block, "text", "") or "")
    raw = "".join(text_chunks)

    start = raw.find("<repaired_node>")
    end = raw.find("</repaired_node>", start + 1)
    if start == -1 or end == -1:
        return None
    body = raw[start + len("<repaired_node>") : end].strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body[3:]
        if body.rstrip().endswith("```"):
            body = body.rsplit("```", 1)[0]
    try:
        node = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(node, dict) and "id" in node:
        return node
    return None


def _load_default_pipeline() -> DescriptorPipeline:
    """Return the production P3.A adapter.

    Raises a clear error if P3.A's module is missing (e.g. in stripped
    deployments). Tests bypass this entirely by passing ``pipeline=...``
    into :func:`compile_with_repair`.
    """
    try:
        # Import lazily — surfacing a clear error rather than a generic
        # ImportError if P3.A is absent.
        from mellea_skills_compiler.compile import descriptor_emission  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Phase 3.A's descriptor_emission module is not available. "
            "Pass a DescriptorPipeline via `pipeline=` to compile_with_repair "
            "to bypass this loader."
        ) from exc
    return _DefaultPipeline()


__all__ = [
    "CANONICAL_OPERATORS",
    "DescriptorPipeline",
    "EscalationClassification",
    "RepairAttempt",
    "RepairConfig",
    "RepairOutcome",
    "Stage",
    "compile_with_repair",
]
