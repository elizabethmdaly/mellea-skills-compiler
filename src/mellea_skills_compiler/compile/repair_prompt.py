"""F1 — Repair-prompt builder: assemble per-lint fix prescriptions.

This module is the orchestrator-side glue between
:mod:`mellea_skills_compiler.compile.repair_templates` (the per-lint fix
prescription renderers) and the repair loop documented in
``.claude/commands/mellea-fy-repair.md``.

The flow:

1. The repair loop reads ``intermediate/step_7_report.json`` from a
   compiled package whose Step 7 verdict is ``"fail"``.
2. For each ``error``-severity (or smoke-escalated) lint failure, it
   calls :func:`mellea_skills_compiler.compile.repair_templates.get_repair_template`
   to render a structured fix prescription.
3. The rendered prescriptions are concatenated under a single
   ``"## Per-lint fix prescriptions"`` section, which is appended to the
   Claude repair prompt BEFORE today's generic "fix these lint failures"
   instruction. Claude sees the targeted prescriptions first; the generic
   instruction acts as the fallback for lints without a registered template.

The slash command in ``.claude/commands/mellea-fy-repair.md`` invokes
:func:`build_repair_prompt_lint_section` via a small ``python -c`` snippet
(or the equivalent inline Bash call). Lints without a registered template
contribute nothing to this section — the slash command's generic repair
prompt still covers them.

Wave 1 templates (registered in :mod:`compile.repair_templates`):

  - ``runtime-defaults-bound``
  - ``bundled-asset-path-resolution``
  - ``stdlib-arity``
  - ``instruct-result-parse-before-access``

This module is intentionally read-only — it does not mutate the report,
the package, or the templates themselves. It is fully covered by unit
tests in ``tests/mellea_skills_compiler/compile/test_repair_prompt.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mellea_skills_compiler.compile.lints import LintFailure
from mellea_skills_compiler.compile.repair_templates import get_repair_template


# ─── Helpers ───────────────────────────────────────────────────────────


def _to_lint_failure(d: dict) -> LintFailure:
    """Convert a JSON-serialised failure dict back into a ``LintFailure``.

    The report stores failures as ``dataclasses.asdict(LintFailure)``. We
    reconstruct the dataclass defensively — unknown extra keys (e.g. a
    forward-compatible ``suggestion`` field) are dropped, missing optional
    fields default to ``None``.
    """
    return LintFailure(
        file=d.get("file", "") or "",
        line=d.get("line"),
        column=d.get("column"),
        message=d.get("message", "") or "",
        rule_ref=d.get("rule_ref"),
    )


def _load_surface_for_package(package_dir: Path) -> Optional[dict]:
    """Load ``<package_dir>/intermediate/mellea_api_ref.json`` if present.

    Returns ``None`` when the file is absent, unreadable, or marked
    ``grounding_unavailable: true``. The ``stdlib-arity`` template uses
    this to interpolate the canonical signature; other templates ignore
    the kwarg.
    """
    if package_dir is None:
        return None
    surface_path = Path(package_dir) / "intermediate" / "mellea_api_ref.json"
    if not surface_path.is_file():
        return None
    try:
        data = json.loads(surface_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or data.get("grounding_unavailable", False):
        return None
    return data


def _is_error_severity(lint: dict) -> bool:
    """Severity gate: only ERROR (or smoke-escalated) lints get a prescription.

    Honours both the static ``severity`` field and the per-run
    ``effective_severity`` (set when a WARNING is escalated by smoke).
    INFO and WARNING lints that were NOT escalated are out of scope here —
    they don't trigger automated repair (see F2 in the follow-up doc).

    Either signal is sufficient: a lint declared ERROR is always in scope;
    a WARNING whose ``effective_severity`` was bumped to "error" by smoke
    is also in scope. ``effective_severity`` defaults to ``severity`` in
    1.1 reports, so checking it covers both shapes — and older 1.0 reports
    (no ``effective_severity`` key) fall through to the ``severity`` check.
    """
    sev = lint.get("severity")
    eff_sev = lint.get("effective_severity", sev)
    return eff_sev == "error"


# ─── Public entry point ─────────────────────────────────────────────────


def build_repair_prompt_lint_section(
    report: dict,
    package_dir: Path,
) -> Optional[str]:
    """Build the "Per-lint fix prescriptions" section of the repair prompt.

    Walks ``report["lints"]`` (a parsed ``step_7_report.json``), filters to
    ERROR-severity failures (including smoke-escalated WARNINGs), and calls
    :func:`get_repair_template` for each. Renderings are concatenated under
    a single ``## Per-lint fix prescriptions`` markdown header.

    Parameters
    ----------
    report:
        Parsed ``step_7_report.json`` contents (a dict).
    package_dir:
        Path to the compiled package directory. Used to load
        ``intermediate/mellea_api_ref.json`` for grounding (only needed by
        the ``stdlib-arity`` template; other templates ignore it).

    Returns
    -------
    The rendered markdown section, or ``None`` when no rendered
    prescriptions are produced. ``None`` means "fall back to the generic
    repair prompt" — the caller (the slash command in
    ``mellea-fy-repair.md``) is responsible for that fallback.

    ``None`` covers two distinct cases:

    1. There are no ERROR-severity failures (report has only WARNING or
       INFO failures, or only passed lints).
    2. There are ERROR-severity failures but none of the lints have a
       registered template (the registry is wave-1 today; wave-2 expands
       coverage — see ``melleafy-handoff/analyses/2026-05-17-lint-
       severity-repair-loop-followups.md``).

    Either case yields ``None`` so the caller can apply a uniform fallback.
    """
    if not isinstance(report, dict):
        return None
    lints = report.get("lints")
    if not isinstance(lints, list):
        return None

    surface_json = _load_surface_for_package(Path(package_dir))
    prescriptions: list[str] = []

    for lint in lints:
        if not isinstance(lint, dict):
            continue
        if lint.get("verdict") != "fail":
            continue
        if not _is_error_severity(lint):
            continue
        lint_id = lint.get("lint_id")
        if not isinstance(lint_id, str):
            continue
        failures = lint.get("failures")
        if not isinstance(failures, list):
            continue
        for failure_dict in failures:
            if not isinstance(failure_dict, dict):
                continue
            failure = _to_lint_failure(failure_dict)
            rendered = get_repair_template(
                lint_id,
                failure,
                package_dir=Path(package_dir),
                surface_json=surface_json,
            )
            if rendered:
                prescriptions.append(rendered)

    if not prescriptions:
        return None

    header = (
        "## Per-lint fix prescriptions\n"
        "\n"
        "The following structured fix prescriptions are derived from the "
        "lint failures in `intermediate/step_7_report.json`. Apply each one "
        "in order; they reference exact files, line numbers, and corrective "
        "patterns. Lints without a registered prescription fall through to "
        "the generic repair instruction below.\n"
        "\n"
    )
    return header + "\n\n---\n\n".join(prescriptions)


__all__ = ["build_repair_prompt_lint_section"]
