"""Drift detection package for corpus-recompile diffs.

Phase 3.5 Stream F (§14 of the Phase 3.5 functional parity plan) introduces a
"previous vs current" compilation-diff mode on ``scripts/corpus_compare.py``.
The diff is classified into one of three drift classes — *cosmetic*,
*structural*, *semantic* — per §14.3.  This package owns the classifier and
the report formatter, isolated from corpus_compare itself so they can be
unit-tested standalone.

Public API:

* :class:`DriftClass`            — enum of cosmetic/structural/semantic/none.
* :class:`SkillDriftVerdict`     — per-skill drift record produced by the
                                   classifier.
* :func:`classify_skill_drift`   — the heuristic classifier entrypoint.
* :func:`render_drift_report`    — markdown formatter for the aggregated
                                   ``recompile_drift_report.md``.

See ``melleafy-handoff/analyses/2026-05-16-phase-3.5-functional-parity-plan.md``
§14 for the design rationale + decision-authority table.
"""

from __future__ import annotations

from mellea_skills_compiler.drift.classifier import (
    DriftClass,
    SkillDriftVerdict,
    classify_skill_drift,
)
from mellea_skills_compiler.drift.report import render_drift_report

__all__ = [
    "DriftClass",
    "SkillDriftVerdict",
    "classify_skill_drift",
    "render_drift_report",
]
