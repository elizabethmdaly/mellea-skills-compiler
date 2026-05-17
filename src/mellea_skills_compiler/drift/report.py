"""Markdown formatter for ``recompile_drift_report.md`` (Phase 3.5 §14.2).

Aggregates :class:`SkillDriftVerdict` records into a per-skill verdict table
plus a per-class summary, matching the decision-doc templates in
``melleafy-handoff/process/decision-template-*-drift.md``.

The report is intentionally human-skimmable.  CI consumers should also have
the structured-JSON variant (corpus_compare exposes both).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from mellea_skills_compiler.drift.classifier import DriftClass, SkillDriftVerdict


def _class_emoji(drift_class: DriftClass) -> str:
    # Plain-text-only — no emojis per project conventions.  Spell the class.
    return drift_class.value


def _summary_table(verdicts: list[SkillDriftVerdict]) -> str:
    """Markdown table summarising counts per drift class."""
    counts = Counter(v.drift_class for v in verdicts)
    total = len(verdicts)
    rows = ["| Class | Count | % |", "|---|---|---|"]
    for cls in (
        DriftClass.NONE,
        DriftClass.COSMETIC,
        DriftClass.STRUCTURAL,
        DriftClass.SEMANTIC,
    ):
        c = counts.get(cls, 0)
        pct = (c / total * 100) if total else 0.0
        rows.append(f"| {cls.value} | {c} | {pct:.1f}% |")
    return "\n".join(rows)


def _verdict_table(verdicts: list[SkillDriftVerdict]) -> str:
    """Per-skill verdict table."""
    rows = [
        "| Skill | Drift class | Pipeline obs. | Schemas obs. | Descriptor obs. |",
        "|---|---|---|---|---|",
    ]
    for v in sorted(verdicts, key=lambda x: x.skill):
        rows.append(
            "| {skill} | {cls} | {pipe} | {sch} | {desc} |".format(
                skill=v.skill,
                cls=v.drift_class.value,
                pipe=_inline_obs(v.pipeline_observations) or "—",
                sch=_inline_obs(v.schemas_observations) or "—",
                desc=_inline_obs(v.descriptor_observations) or "—",
            )
        )
    return "\n".join(rows)


def _inline_obs(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return ""
    # Markdown table cells can't contain newlines; semicolon-join and replace
    # pipe characters defensively so the table doesn't break.
    joined = "; ".join(item.replace("|", "/") for item in items)
    if len(joined) > 200:
        joined = joined[:197] + "..."
    return joined


def _detailed_section(verdicts: list[SkillDriftVerdict]) -> str:
    """Expanded per-skill block with full observation lists."""
    out: list[str] = []
    for v in sorted(verdicts, key=lambda x: x.skill):
        if v.drift_class == DriftClass.NONE and not v.notes:
            # Don't waste lines on every NONE-class skill; they're expected.
            continue
        out.append(f"### {v.skill} — {v.drift_class.value}")
        out.append("")
        if v.notes:
            out.append("**Notes:**")
            for n in v.notes:
                out.append(f"- {n}")
            out.append("")
        if v.pipeline_observations:
            out.append("**pipeline.py:**")
            for line in v.pipeline_observations:
                out.append(f"- {line}")
            out.append("")
        if v.schemas_observations:
            out.append("**schemas.py:**")
            for line in v.schemas_observations:
                out.append(f"- {line}")
            out.append("")
        if v.descriptor_observations:
            out.append("**melleafy.json:**")
            for line in v.descriptor_observations:
                out.append(f"- {line}")
            out.append("")
    return "\n".join(out).rstrip()


def render_drift_report(
    verdicts: list[SkillDriftVerdict],
    *,
    mellea_version: str | None = None,
    recompile_trigger: str | None = None,
    previous_ref: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Render the aggregate ``recompile_drift_report.md`` markdown."""

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")

    counts = Counter(v.drift_class for v in verdicts)
    total = len(verdicts)
    semantic_pct = (counts[DriftClass.SEMANTIC] / total * 100) if total else 0.0
    hard_stop = semantic_pct > 5.0  # §14.4 hard-stop

    lines: list[str] = []
    lines.append("# Corpus recompile — drift report")
    lines.append("")
    lines.append(f"- **Generated at**: {generated_at}")
    if mellea_version:
        lines.append(f"- **Mellea version**: `{mellea_version}`")
    if recompile_trigger:
        lines.append(f"- **Recompile trigger**: {recompile_trigger}")
    if previous_ref:
        lines.append(f"- **Compared against**: `{previous_ref}`")
    lines.append(f"- **Total skills compared**: {total}")
    lines.append("")
    lines.append("## Drift class summary")
    lines.append("")
    lines.append(_summary_table(verdicts))
    lines.append("")
    if hard_stop:
        lines.append(
            f"> **HARD STOP**: semantic-drift on {semantic_pct:.1f}% of corpus "
            "(>5% threshold per Phase 3.5 plan §14.4).  Halt the bump rollout "
            "and review Mellea release notes before proceeding."
        )
        lines.append("")
    lines.append("## Per-skill verdicts")
    lines.append("")
    lines.append(_verdict_table(verdicts))
    lines.append("")
    detail = _detailed_section(verdicts)
    if detail:
        lines.append("## Details (non-cosmetic / notable skills)")
        lines.append("")
        lines.append(detail)
        lines.append("")
    lines.append("## Decision authority")
    lines.append("")
    lines.append(
        "Per `melleafy-handoff/analyses/2026-05-16-phase-3.5-functional-parity-plan.md` §14.4:"
    )
    lines.append("")
    lines.append("- **Cosmetic** — auto-accepted; no record needed.")
    lines.append(
        "- **Structural** — project lead approves; record per-skill verdicts in "
        "`melleafy-handoff/decisions/<date>-mellea-<version>-structural-drift.md` "
        "(template at `melleafy-handoff/process/decision-template-structural-drift.md`)."
    )
    lines.append(
        "- **Semantic** — team review meeting; record in "
        "`melleafy-handoff/decisions/<date>-mellea-<version>-semantic-drift.md` "
        "(template at `melleafy-handoff/process/decision-template-semantic-drift.md`)."
    )
    lines.append("")
    lines.append("## Lint severity drift")
    lines.append("")
    lines.append(
        "When `step_7_report.json` schemas differ between the previous and "
        "current compile, any per-lint **severity** change is itself a "
        "structural drift class — not cosmetic. A drift that promotes a "
        "lint from `warning` to `error` (or vice versa) changes the gate "
        "behaviour for every consumer and must be recorded in the "
        "structural-drift decision template alongside any Python-level "
        "diffs. The `_LINT_SEVERITY` table in `compile/lints.py` is the "
        "source of truth; reviewers should diff that table when comparing "
        "compiler versions."
    )
    lines.append("")
    return "\n".join(lines) + "\n"
