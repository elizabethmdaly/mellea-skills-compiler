"""Canonical exit codes for the ``mellea-skills`` CLI.

This module is the single source of truth for every exit code the
compiler emits. CI scripts, repair handlers, and slash-command docs all
import from here rather than hard-coding numeric literals.

Codes split into two ranges:

  - 0–4:  generic CLI outcomes (success, general error, bad invocation,
          output conflict, export-conformance failure).
  - 10+:  stage-specific halts (one per stage that can refuse to write
          output). A new stage that wants a distinct halt code claims
          the next unused number ≥ 10 and adds a constant here.

Generated-package exit codes (the codes that a compiled skill's
``main.py`` emits when *the skill itself* runs) are deliberately NOT
defined here — they belong to the generated package's own template.

Wiring status (2026-05-18):
  - ``SUCCESS``, ``GENERAL_ERROR``, ``INVALID_INVOCATION`` are emitted
    by ``cli.py`` today.
  - ``EXPORT_LINT_FAIL`` is emitted by
    ``export.exporter._stage_5_lint`` (the only Stage-5 halt site).
  - ``LINT_FAIL`` is emitted by ``cli.py`` for the ``compile`` and
    ``validate`` commands when ``compile/mellea_skills.py::validate``
    raises ``LintFailureError`` (Step 7 lint suite had at least one
    ERROR-severity failure).
  - ``SMOKE_CHECK_FAIL`` is emitted by ``cli.py`` for the ``compile``,
    ``validate``, and ``run`` commands when
    ``compile/smoke_check.py::SmokeCheckFailure`` propagates from the
    fixture smoke-check.
  - ``DEPENDENCY_STRICT_HALT`` (code 10) and ``OUTPUT_CONFLICT``
    (code 3) are documented contracts but not yet emitted. Migration
    follows the same pattern: introduce a typed exception at the
    raise site, propagate through the wrapper, dispatch in ``cli.py``.

If you add a new code, update:
  1. This module (constant + one-line rationale).
  2. The slash command that documents the failure mode (point at the
     constant by name; never re-state the number).
  3. The corresponding CI integration / test harness that consumes it.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["ExitCode"]


class ExitCode(IntEnum):
    """Numeric exit codes emitted (or intended to be emitted) by the CLI.

    Subclass of :class:`IntEnum` so callers can pass an ``ExitCode``
    directly to ``typer.Exit(code=...)`` and ``sys.exit(...)`` without
    casting, and so equality with bare ``int`` literals still works for
    CI scripts that test against numeric values.
    """

    #: Clean exit; everything the command was asked to do succeeded.
    SUCCESS = 0

    #: Unhandled exception caught at the CLI boundary. The fallback
    #: classification — emitted when no more specific code applies.
    GENERAL_ERROR = 1

    #: Bad CLI invocation (validated before any work starts): missing
    #: required argument, unknown flag value, mutually-exclusive flags
    #: both set, etc. Distinct from ``GENERAL_ERROR`` so CI can tell
    #: "user passed bad args" apart from "the command tried and failed".
    INVALID_INVOCATION = 2

    #: Output path conflict — a non-empty directory existed at
    #: ``--out`` without ``--force``. Reserved for future use by the
    #: export CLI; not currently emitted.
    OUTPUT_CONFLICT = 3

    #: Export Stage 5 (adapter-conformance lint) refused the adapter.
    #: Emitted by ``export.exporter._stage_5_lint``.
    EXPORT_LINT_FAIL = 4

    #: ``--dependencies=strict`` halt at Step 2.5 — at least one
    #: detected dependency would land on ``stub`` or
    #: ``delegate_to_runtime`` and the user opted into strict-no-stubs
    #: mode. Documented in ``mellea-fy-deps.md``; the strict mode CLI
    #: flag is not yet wired, so this code is not yet emitted.
    DEPENDENCY_STRICT_HALT = 10

    #: Step 7 static lint suite has at least one ERROR-severity
    #: failure. Documented in ``mellea-fy-validate.md``; today the CLI
    #: collapses lint failures to ``GENERAL_ERROR`` (see migration note
    #: in the module docstring).
    LINT_FAIL = 11

    #: Step 7b fixture smoke-check raised an exception that the
    #: detection heuristic classified as a real failure (not a
    #: backend-unreachable skip). Documented in
    #: ``mellea-fy-validate.md``; today the CLI collapses smoke-check
    #: failures to ``GENERAL_ERROR``.
    SMOKE_CHECK_FAIL = 12

    #: Descriptor IR schema validation failed — the LLM-emitted
    #: ``intermediate/descriptor_emission.json`` does not conform to the
    #: declared ``descriptor.schema.vX.Y.json``. Emitted by the pre-render
    #: schema gate in ``writer_renderer.render_descriptor_to_python``.
    #: D1 (auto-repair) is the natural consumer — when wired, it feeds the
    #: structured schema errors back to the LLM for a repair attempt.
    DESCRIPTOR_SCHEMA_FAIL = 13
