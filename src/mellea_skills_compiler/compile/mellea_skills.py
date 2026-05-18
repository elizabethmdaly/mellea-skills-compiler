import json
import os
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from anthropic import Anthropic
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from mellea_skills_compiler.compile import CLAUDE_DIR
from mellea_skills_compiler.compile.claude_directives import (
    build_system_prompt,
    derive_package_name,
    mirror_companion_dirs,
    resolve_runtime_defaults,
    write_compile_settings,
    write_runtime_directive,
)
from mellea_skills_compiler.compile.grounding import (
    write_mellea_api_ref,
    write_mellea_doc_index,
)
from mellea_skills_compiler.compile.proxy import ContextMgmtStrippingProxy
from mellea_skills_compiler.enums import (
    ClaudeResponseMessageType,
    ClaudeResponseType,
    InferenceModel,
    SpecFileFormat,
)
from mellea_skills_compiler.toolkit.file_utils import parse_spec_file
from mellea_skills_compiler.toolkit.logging import configure_logger


LOGGER = configure_logger()
console = Console(log_time=True)


def _resolve_writers_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for a directory containing
    `.claude/melleafy/writers/`.

    Used to locate the writers directory that the deterministic writer
    renderer (`compile/writer_renderer.py`) reads. The caller passes in the
    installed `mellea_skills_compiler` package directory (NOT the generated
    skill package directory) so the walk-up succeeds even when the spec was
    compiled out-of-tree.

    Raises:
        FileNotFoundError: if no ancestor directory contains
            `.claude/melleafy/writers/`. This indicates the compiler is
            installed against a source tree that has been stripped of its
            companion `.claude/` directory — the compile cannot proceed.
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / ".claude" / "melleafy" / "writers").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not locate .claude/melleafy/writers/ relative to "
        f"{start}. The compiler must be installed (editable or otherwise) "
        "from a repo that contains .claude/melleafy/writers/."
    )


def _get_spec_md_path(spec_path: Path):
    spec_file_path = None
    if spec_path.is_dir():
        if (spec_path / SpecFileFormat.SKILL_FILE_MD).exists():
            spec_file_path = spec_path / SpecFileFormat.SKILL_FILE_MD
        elif (spec_path / SpecFileFormat.SPEC_FILE_MD).exists():
            spec_file_path = spec_path / SpecFileFormat.SPEC_FILE_MD
    elif spec_path.suffix == ".md":
        spec_file_path = spec_path

    return spec_file_path


def _tolerant_name_extract(spec_path: Path) -> dict:
    """Extract the ``name:`` field from a spec's frontmatter without strict
    YAML parsing.

    Used as a fallback when ``parse_spec_file`` raises (typically because
    the spec's ``description:`` value contains unquoted colons / parens
    that YAML can't disambiguate). The slash command tolerates this and
    so must the wrapper — otherwise package-name derivation falls back to
    the directory name and the wrapper / LLM end up writing to different
    ``<package>_mellea/`` directories.

    Returns a dict shaped like ``parse_spec_file(...).get("frontmatter")``
    but containing only ``name`` (and only if found). Returns an empty
    dict when no frontmatter block exists or no ``name:`` line is in it.
    """
    import re as _re

    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    match = _re.match(r"^---\s*\n(.*?)\n---\s*\n", text, _re.DOTALL)
    if not match:
        return {}
    fm_block = match.group(1)
    # Match the first top-level (un-indented) `name:` line.
    name_match = _re.search(
        r"^name:\s*(?P<value>.+?)\s*$", fm_block, _re.MULTILINE
    )
    if not name_match:
        return {}
    raw = name_match.group("value").strip()
    # Strip surrounding quotes if present.
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        raw = raw[1:-1]
    return {"name": raw} if raw else {}


def _build_claude_argv(
    *,
    spec_path: Path,
    model: str,
    system_prompt: str,
    compile_settings_path: Optional[Path],
    repair_mode: bool,
    use_descriptor: bool = False,
) -> list[str]:
    """Build the ``claude -p`` argv for the /mellea-fy compile session.

    Extracted from :func:`compile` (Phase 3.5.A §3 — refactor invariant test
    in §12.2 of the parity plan): when ``use_descriptor`` is ``False`` the
    resulting argv MUST be byte-identical to the pre-Phase-3.5.A build. The
    only addition for descriptor mode is appending ``--use-descriptor`` to
    the quoted slash-command invocation (which the slash-command orchestrator
    in ``.claude/commands/mellea-fy.md`` parses out of ``$ARGUMENTS``).

    Args:
        spec_path: Path forwarded as the slash-command argument.
        model: Claude model id (already verified by the caller).
        system_prompt: The wrapper's compile-time system prompt.
        compile_settings_path: Optional per-invocation settings file.
        repair_mode: ``True`` switches the slash command from
            ``./mellea-fy`` to ``./mellea-fy-repair``.
        use_descriptor: Phase 3.5.A — when ``True`` appends
            ``--use-descriptor`` to the slash-command argument so Step 5
            routes through descriptor IR emission + render.

    Returns:
        The argv list as it will be passed to :func:`subprocess.Popen`.
    """
    argv: list[str] = [
        "claude",
        "-p",
        "--model",
        f"{model}",
        "--append-system-prompt",
        system_prompt,
        "--allowed-tools",
        "Read,Write,Edit",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
    ]
    if compile_settings_path is not None:
        argv.extend(["--settings", str(compile_settings_path)])

    slash_command = "./mellea-fy-repair" if repair_mode else "./mellea-fy"
    slash_args = str(spec_path)
    if use_descriptor:
        slash_args = f"{slash_args} --use-descriptor"
    argv.append(f"'{slash_command} {slash_args}'")
    return argv


# ---------------------------------------------------------------------------
# F6 — Pipeline resume on early turn-end.
#
# The Claude Code SDK ends its session naturally when the model's turn ends
# without a follow-up tool call. For complex multi-step pipelines this is
# fragile — a stray "Step N complete. Proceeding to Step N+1." narrative
# ends the turn before the first tool call of Step N+1 fires, so the wrapper
# returns from `_spawn_claude` even though the pipeline is half-done. The
# constants and helpers below let the wrapper detect this gap and re-invoke
# Claude with a "resume from Step N+1" directive.
#
# Detection is purely artefact-based — we look for canonical files each step
# is expected to produce. The order matches `.claude/commands/mellea-fy.md`
# §"Step-by-step" and Rule OUT-3's layout under `intermediate/`. Step 7
# (lint) is intentionally NOT included: it is run by the wrapper *after*
# the resume loop returns. The resume loop only ensures Steps 0–6 produce
# their artefacts.
# ---------------------------------------------------------------------------

# Per-step canonical artefact paths, relative to the package directory.
# Several entries fan out to multiple possible filenames so the detector
# tolerates the alternative emission locations the orchestrator uses today:
#
#   step_4_fixtures: descriptor mode emits `fixtures_emission.json`; legacy
#       mode populates `fixtures/` directly. Either is sufficient.
#   step_5_descriptor: descriptor mode writes
#       `intermediate/descriptor_emission.json` (single canonical filename;
#       the slash command does not vary the name by package). Legacy mode
#       has no descriptor; the `pipeline.py` fallback covers that path.
#       Historical glob `*.descriptor.json` is stale — it never matched
#       the actual emission filename, so descriptor-mode F6 detection
#       silently failed and used pipeline.py as the only signal. Because
#       the wrapper renders pipeline.py post-session, this meant Step 5
#       was never detected as "done" inside F6.
#   step_6_artefacts: `melleafy.json` lives at the package root, NOT under
#       `intermediate/`. Detection accepts either location.
_STEP_ARTEFACTS: list[tuple[str, tuple[str, ...]]] = [
    ("step_0_classify", ("intermediate/classification.json",)),
    ("step_1a_inventory", ("intermediate/inventory.json",)),
    (
        "step_2_map",
        ("intermediate/element_mapping.json",),
    ),
    ("step_2_5_deps", ("intermediate/dependency_plan.json",)),
    (
        "step_4_fixtures",
        ("intermediate/fixtures_emission.json", "fixtures/__init__.py"),
    ),
    (
        # Step 5 emits per-element code bodies into `pipeline.py`. In
        # descriptor mode the LLM additionally writes
        # `intermediate/<package>.descriptor.json` which the wrapper renders
        # into pipeline.py. Either is sufficient evidence that Step 5 ran.
        #
        # NOTE: `intermediate/config_emission.json` is NOT a Step 5 signal
        # — the writer-renderer synthesizes it from `runtime_directive.json`
        # when the LLM didn't emit it, so its presence does not prove Step 5
        # completed (the dpia-sentinel empirical failure had it but Steps 1-6
        # were nonetheless not done).
        "step_5_descriptor",
        ("intermediate/descriptor_emission.json", "pipeline.py"),
    ),
    (
        "step_6_artefacts",
        ("melleafy.json", "intermediate/melleafy.json"),
    ),
]


def _detect_completed_steps(package_dir: Path) -> list[str]:
    """Return the step keys whose canonical artefact exists in ``package_dir``.

    Order matches :data:`_STEP_ARTEFACTS` (Step 0 → Step 6). The list is
    a flat sequence of step keys, NOT a max-completed-step pointer, so the
    caller can present the exact set in the resume directive (e.g.
    "Steps 0, 1a, 2 are done; resume at 2.5"). Each step's tuple of
    candidate paths is OR'd: any one of the candidates existing counts
    as "done". Glob patterns (`*`) are matched against the package
    directory.
    """
    done: list[str] = []
    for step_key, candidates in _STEP_ARTEFACTS:
        for candidate in candidates:
            if "*" in candidate:
                # Glob match — split into parent + filename pattern.
                parent_rel, _, pattern = candidate.rpartition("/")
                glob_parent = package_dir / parent_rel if parent_rel else package_dir
                if glob_parent.is_dir() and any(glob_parent.glob(pattern)):
                    done.append(step_key)
                    break
            else:
                if (package_dir / candidate).exists():
                    done.append(step_key)
                    break
    return done


# Human-friendly labels paired with each step key, used to surface the
# "Steps X, Y, Z are complete; resume at W" line in the resume directive.
_STEP_LABELS: dict[str, str] = {
    "step_0_classify": "Step 0 (classification)",
    "step_1a_inventory": "Step 1a (inventory)",
    "step_2_map": "Step 2 (element mapping)",
    "step_2_5_deps": "Step 2.5 (dependency plan)",
    "step_4_fixtures": "Step 4 (fixtures)",
    "step_5_descriptor": "Step 5 (descriptor / config emission)",
    "step_6_artefacts": "Step 6 (mapping report + melleafy.json)",
}


def _build_resume_system_prompt(
    base_prompt: str, completed_steps: list[str]
) -> str:
    """Augment the base system prompt with a "resume from Step N+1" directive.

    Lists the completed steps explicitly and tells Claude to start with the
    first tool call of the next incomplete step — narrative text BEFORE the
    tool invocation is forbidden, because that's exactly what caused the
    original turn to end early.
    """
    completed_labels = [_STEP_LABELS.get(k, k) for k in completed_steps]
    all_step_keys = [k for k, _ in _STEP_ARTEFACTS]
    remaining = [k for k in all_step_keys if k not in completed_steps]
    next_step_label = (
        _STEP_LABELS.get(remaining[0], remaining[0])
        if remaining
        else "Step 7 (lint)"
    )
    completed_block = (
        "\n  - " + "\n  - ".join(completed_labels)
        if completed_labels
        else "\n  (none — start from Step 0)"
    )
    return (
        f"{base_prompt}\n\n"
        "PIPELINE RESUME — a previous Claude session ended early. The "
        "following pipeline steps are ALREADY COMPLETE (see "
        "intermediate/ for their artefacts). DO NOT redo them:"
        f"{completed_block}\n"
        f"Begin IMMEDIATELY at {next_step_label} by invoking the first tool "
        "for that step. Any narrative text BEFORE the tool invocation is "
        "forbidden — start your response with the tool call itself. After "
        "each subsequent step completes, IMMEDIATELY invoke the first tool "
        "of the next step in the same turn; do not end your turn with a "
        '"Proceeding to Step N+1." narrative line.'
    )


def _spawn_claude_with_resume(
    *,
    spec_path: Path,
    model: str,
    base_system_prompt: str,
    compile_settings_path: Optional[Path],
    repair_mode: bool,
    use_descriptor: bool,
    subprocess_env: dict,
    intermediate_dir: Path,
    package_dir: Path,
    timeout: int,
    processing,
    max_resumes: int = 3,
    spawn_fn=None,
) -> None:
    """Run ``claude -p`` and re-invoke if Claude's turn ended before Step 6.

    Wraps :func:`_spawn_claude` with artefact-based resume detection:

    1. Spawns Claude with the legacy argv (built from ``base_system_prompt``).
    2. After it returns, inspects ``package_dir`` to detect which steps
       produced their canonical artefact.
    3. If the highest-completed step is below Step 6 AND ``max_resumes``
       has not been hit, builds a new argv with an augmented system prompt
       ("Steps X, Y are done — resume at Z") and re-invokes.
    4. Loops until Step 6 is complete OR ``max_resumes`` is exhausted OR
       a resume round produces no new artefacts (forward progress check).

    Step 7 (lint) is intentionally NOT covered here — the caller runs lints
    after this helper returns.

    ``spawn_fn`` is injected for tests; defaults to :func:`_spawn_claude`.
    """
    spawn_fn = spawn_fn or _spawn_claude
    resumes_used = 0
    last_completed: list[str] = []

    # First invocation: identical to the legacy single-shot path.
    claude_argv = _build_claude_argv(
        spec_path=spec_path,
        model=model,
        system_prompt=base_system_prompt,
        compile_settings_path=compile_settings_path,
        repair_mode=repair_mode,
        use_descriptor=use_descriptor,
    )
    spawn_fn(
        claude_argv=claude_argv,
        subprocess_env=subprocess_env,
        intermediate_dir=intermediate_dir,
        timeout=timeout,
        processing=processing,
    )

    while resumes_used < max_resumes:
        completed = _detect_completed_steps(package_dir)
        # Resume-end condition: BOTH Step 5 (descriptor / pipeline.py)
        # and Step 6 (melleafy.json) artefacts present. Originally only
        # Step 6 was required, but the nil-contract overnight failure
        # showed Claude can write `melleafy.json` while hallucinating
        # that Step 5 produced `descriptor_emission.json`. The wrapper
        # then trusted the Step 6 marker, returned from F6, ran the
        # descriptor renderer with no emission to render, and crashed
        # at smoke-check on "pipeline.py missing". Requiring Step 5
        # here forces a resume round that re-prompts Claude for the
        # missing descriptor.
        if "step_5_descriptor" in completed and "step_6_artefacts" in completed:
            return
        # Forward-progress check: if a resume round produced no new
        # artefacts compared to the previous round, abort to avoid a
        # wasted final round.
        if resumes_used > 0 and set(completed) == set(last_completed):
            LOGGER.warning(
                "Resume round %d produced no new artefacts (still: %s). "
                "Aborting resume loop.",
                resumes_used,
                completed or "none",
            )
            return
        last_completed = list(completed)
        resumes_used += 1
        LOGGER.warning(
            "Claude ended its turn before Step 6 completed (completed so "
            "far: %s). Resume round %d/%d — re-invoking with a resume "
            "directive.",
            ", ".join(completed) if completed else "none",
            resumes_used,
            max_resumes,
        )
        resume_prompt = _build_resume_system_prompt(base_system_prompt, completed)
        claude_argv = _build_claude_argv(
            spec_path=spec_path,
            model=model,
            system_prompt=resume_prompt,
            compile_settings_path=compile_settings_path,
            repair_mode=repair_mode,
            use_descriptor=use_descriptor,
        )
        spawn_fn(
            claude_argv=claude_argv,
            subprocess_env=subprocess_env,
            intermediate_dir=intermediate_dir,
            timeout=timeout,
            processing=processing,
        )

    # Loop exited via max_resumes — Step 5 and/or Step 6 still not done.
    # Log and let the caller's Step 7 lint run produce the diagnostic
    # step_7_report.json (or the wrapper's own raise) — we don't raise
    # here because the caller has its own teardown path and we want
    # behaviour to look like "single shot didn't finish" not "resume
    # helper crashed".
    final_completed = _detect_completed_steps(package_dir)
    missing_terminal_steps = [
        s for s in ("step_5_descriptor", "step_6_artefacts")
        if s not in final_completed
    ]
    if missing_terminal_steps:
        LOGGER.warning(
            "Resume loop exhausted %d round(s) with terminal step(s) still "
            "incomplete: %s (completed overall: %s). Proceeding to lint "
            "anyway; the lint will produce the diagnostic.",
            max_resumes,
            ", ".join(missing_terminal_steps),
            ", ".join(final_completed) if final_completed else "none",
        )


# ---------------------------------------------------------------------------
# Fix A — Post-session lint-failure retry.
#
# Distinct from F6 (`_spawn_claude_with_resume`), which addresses the
# *incomplete* pipeline case (Claude bailed mid-pipeline; canonical
# artefacts missing). Fix A addresses the *complete-but-broken* case:
# Claude self-reported in-session that lints passed, but the wrapper-side
# `run_lints` disagrees post-session — and the in-session repair loop has
# already exited so there's no remaining mechanism to re-engage Claude.
#
# The retry is wrapper-side: we spawn a second Claude session in repair
# mode (./mellea-fy-repair) with structured F1 fix prescriptions baked to
# disk at `intermediate/repair_prescriptions.md`. The repair slash command
# reads that file plus the underlying `step_7_report.json` and applies the
# targeted fixes.
# ---------------------------------------------------------------------------


def _bake_repair_prescriptions(
    package_dir: Path,
    *,
    lint_report_path: Path,
) -> Optional[Path]:
    """Read ``step_7_report.json``, build F1 prescriptions, persist to disk.

    Thin wrapper around
    :func:`mellea_skills_compiler.compile.repair_prompt.build_repair_prompt_lint_section`
    that loads the JSON report and writes the rendered markdown to
    ``<package_dir>/intermediate/repair_prescriptions.md`` so a downstream
    Claude repair session can ``Read`` it.

    Severity gating lives inside ``build_repair_prompt_lint_section``:
    only ERROR-severity (or smoke-escalated WARNING) failures contribute
    prescriptions. A report with only WARNING / INFO failures yields
    ``None`` and no file is written.

    Returns:
        The path to ``intermediate/repair_prescriptions.md`` when a non-
        empty prescription section was rendered. ``None`` when no
        prescription was produced (either no ERROR-severity failures, or
        none of the failing lint ids have a registered template).
    """
    from mellea_skills_compiler.compile.repair_prompt import (
        build_repair_prompt_lint_section,
    )

    if not lint_report_path.is_file():
        return None
    try:
        report = json.loads(lint_report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    rendered = build_repair_prompt_lint_section(report, Path(package_dir))
    if not rendered:
        return None
    out_path = package_dir / "intermediate" / "repair_prescriptions.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def _build_lint_repair_system_prompt(
    base_prompt: str,
    *,
    failing_lints: list[tuple[str, list[str]]],
    prescriptions_path: Optional[Path],
    lint_report_path: Path,
    descriptor_render_failure_reason: Optional[str] = None,
) -> str:
    """Augment the base system prompt for a post-session lint-repair spawn.

    ``failing_lints`` is a list of ``(lint_id, [file:line, ...])`` tuples
    describing the error-severity failures the wrapper saw post-session.
    The augmented prompt:

    - Explicitly tells Claude this is a wrapper-side retry triggered by
      lint failures, NOT a fresh compile.
    - Points to ``intermediate/repair_prescriptions.md`` (when present)
      and ``intermediate/step_7_report.json``.
    - Names the failing lint ids and files inline so Claude doesn't have
      to discover them.
    - Forbids redoing the full pipeline; the goal is targeted file fixes.

    ``descriptor_render_failure_reason``, when set, indicates the wrapper's
    descriptor-to-Python renderer failed on the previous round's
    ``descriptor_emission.json``. The reason string carries a precise
    error (e.g. ``"no module prefix of 'config.SKILL_DESCRIPTION' resolves
    in surface"``). Surfaced ahead of the lint list so Claude fixes the
    descriptor itself rather than chasing the downstream "pipeline.py
    missing" lint that the failed render produces.
    """
    lines = [f"  - {lid}: {', '.join(locs) if locs else '(no locations)'}"
             for lid, locs in failing_lints]
    failing_block = "\n".join(lines) if lines else "  (none — see report)"
    prescription_hint = (
        f"Structured fix prescriptions have been pre-baked to "
        f"`{prescriptions_path.relative_to(prescriptions_path.parents[1])}` "
        "— READ THAT FILE FIRST and apply each fix to the named file."
        if prescriptions_path is not None
        else "No structured prescription templates matched these lint ids; "
        "consult `intermediate/step_7_report.json` directly and apply the "
        "fix described by each failure's `message` field."
    )
    descriptor_block = ""
    if descriptor_render_failure_reason:
        descriptor_block = (
            "\n\nDESCRIPTOR RENDER FAILED — the wrapper attempted to render "
            "`pipeline.py` + `schemas.py` from "
            "`intermediate/descriptor_emission.json` and the renderer rejected "
            "the descriptor with this error:\n"
            f"  {descriptor_render_failure_reason}\n"
            "Fix the descriptor JSON itself (not pipeline.py — that file is "
            "wrapper-rendered output and will be regenerated). The downstream "
            "lint failures below are consequences of pipeline.py being absent "
            "or stale; address the descriptor error first and they resolve."
        )
    return (
        f"{base_prompt}\n\n"
        "WRAPPER-SIDE LINT REPAIR — Step 7 lint failures were detected "
        "AFTER the previous Claude session ended. This is a fresh session "
        "spawned by the wrapper to apply targeted fixes; the in-session "
        "repair loop did NOT engage because the previous session "
        "self-reported success."
        f"{descriptor_block}\n\n"
        f"Failing lints (ERROR severity):\n{failing_block}\n\n"
        f"{prescription_hint}\n\n"
        f"The full lint report is at "
        f"`{lint_report_path.name}` (under `intermediate/`).\n\n"
        "DO NOT redo the full pipeline. Apply the prescribed fixes to "
        "the named files and exit. Do not run Steps 0–6 again; their "
        "artefacts are already in `intermediate/`."
    )


def _build_smoke_repair_system_prompt(
    base_prompt: str,
    *,
    smoke_failure,
) -> str:
    """Augment the base system prompt for a post-smoke-check repair spawn.

    ``smoke_failure`` is a
    :class:`mellea_skills_compiler.compile.smoke_check.SmokeCheckFailure`
    instance carrying the failure kind (``"infrastructure"`` or
    ``"fixtures"``) and details. The augmented prompt names the failing
    fixtures (or the infrastructure error), points at the smoke report,
    and forbids re-running the full pipeline — the goal is a targeted
    fix to ``pipeline.py`` / ``slots.py`` / ``fixtures/`` and exit.
    """
    if smoke_failure.kind == "infrastructure":
        body = (
            "WRAPPER-SIDE SMOKE-CHECK REPAIR — Step 8 smoke-check could not "
            "start. The pipeline package failed at import or top-level "
            "evaluation BEFORE the first fixture ran.\n\n"
            "Infrastructure error:\n"
            f"  {smoke_failure.infrastructure_error}\n\n"
            "Typical causes:\n"
            "  - `pipeline.py` references a name that isn't imported "
            "(NameError on import)\n"
            "  - `slots.py` or `requirements.py` raises at module top "
            "level\n"
            "  - A `from .config import X` where X was renamed/removed\n\n"
            "Fix the imports / top-level errors. Do NOT run Steps 0–6 "
            "again; their artefacts are already in `intermediate/`."
        )
    else:
        lines: list[str] = []
        for fixture_id, message in smoke_failure.fixture_failures:
            msg = message if message else "(no message)"
            # Trim very long failure messages so the prompt stays focused
            # — the full traceback is in step_7b_report.json.
            if len(msg) > 800:
                msg = msg[:800] + " …[truncated; see report]"
            lines.append(f"  - {fixture_id}: {msg}")
        failing_block = "\n".join(lines) if lines else "  (no fixture details)"
        body = (
            "WRAPPER-SIDE SMOKE-CHECK REPAIR — Step 8 smoke-check ran but "
            "one or more fixtures reported failure. This is a fresh "
            "session spawned by the wrapper to apply targeted fixes; "
            "the in-session repair loop did NOT engage because the "
            "previous session self-reported success at Step 7.\n\n"
            f"Failing fixtures:\n{failing_block}\n\n"
            "The full smoke report (including tracebacks) is at "
            "`intermediate/step_7b_report.json`.\n\n"
            "Typical causes:\n"
            "  - `@generative` stub signature mismatch with the fixture "
            "call site (missing/extra required parameter)\n"
            "  - `requirements.py` checker raising on real data\n"
            "  - Fixture inputs not matching the schema declared in "
            "`schemas.py`\n\n"
            "Fix the relevant files. Do NOT run Steps 0–6 again; their "
            "artefacts are already in `intermediate/`."
        )
    return f"{base_prompt}\n\n{body}"


def _collect_error_failures(report: dict) -> list[tuple[str, list[str]]]:
    """Return the list of ERROR-severity failing lints + their locations.

    Used to populate the augmented system prompt. The shape matches
    :func:`_build_lint_repair_system_prompt`'s ``failing_lints`` arg:
    ``[(lint_id, ["file:line", ...]), ...]``. Mirrors the same severity
    gate used by :func:`build_repair_prompt_lint_section` so the list and
    the prescription file stay in lockstep.
    """
    out: list[tuple[str, list[str]]] = []
    if not isinstance(report, dict):
        return out
    lints = report.get("lints")
    if not isinstance(lints, list):
        return out
    for lint in lints:
        if not isinstance(lint, dict):
            continue
        if lint.get("verdict") != "fail":
            continue
        sev = lint.get("severity")
        eff_sev = lint.get("effective_severity", sev)
        if eff_sev != "error":
            continue
        lint_id = lint.get("lint_id")
        if not isinstance(lint_id, str):
            continue
        locs: list[str] = []
        for failure in lint.get("failures", []) or []:
            if not isinstance(failure, dict):
                continue
            file_ = failure.get("file") or "?"
            line = failure.get("line")
            locs.append(f"{file_}:{line}" if line is not None else file_)
        out.append((lint_id, locs))
    return out


def _lint_repair_retry_loop(
    *,
    spec_path: Path,
    model: str,
    base_system_prompt: str,
    compile_settings_path: Optional[Path],
    use_descriptor: bool,
    subprocess_env: dict,
    intermediate_dir: Path,
    package_dir: Path,
    timeout: int,
    processing,
    max_repair_rounds: int = 2,
    run_post_session_lints: Optional[Callable[[Path], object]] = None,
    run_writer_renderer: Optional[Callable[[Path], None]] = None,
    spawn_fn: Optional[Callable[..., None]] = None,
) -> None:
    """Post-session lint-repair retry. Assumes the initial compile spawn
    has ALREADY happened and the package state on disk reflects that.

    Steps:
      1. Run writer-renderer (synthesizers + config.py + fixtures/).
      2. ``run_lints(package_dir)``. If ``overall_verdict != "fail"`` →
         return (lints clean or only WARNING / INFO failures).
      3. Bake F1 prescriptions → augmented system prompt → spawn repair
         Claude with the ``./mellea-fy-repair`` slash command → re-render
         → re-lint. Loop up to ``max_repair_rounds``.
      4. After the cap, return silently (caller raises the final lint
         failure; this helper does NOT raise, to keep the legacy error
         path intact).

    Decoupled from :func:`_spawn_claude` so the layer composes naturally
    with :func:`_spawn_claude_with_resume`: ``compile()`` runs the
    initial spawn (with or without resume) and then runs THIS helper
    when ``--repair-on-lint-failure`` is set. The two retry layers
    are independent flag checks; either can fire alone, both can fire
    in sequence on the same compile.

    ``run_post_session_lints``: callable ``(package_dir) -> LintRunResult``.
    Defaults to ``run_lints(package_dir, strict=False, smoke_check="never")``.

    ``run_writer_renderer``: callable ``(package_dir) -> None`` that runs
    the deterministic config.py / fixtures synthesizers. Defaults to a
    closure that locates the writers and invokes ``render_writers``.

    ``spawn_fn``: injected for tests; defaults to :func:`_spawn_claude`.
    Only invoked on repair re-spawns — this helper assumes the *initial*
    spawn has already happened.

    The CALLER is responsible for the downstream ``validate(...)`` call
    (smoke-check, etc.) so the unified failure path stays in ``compile``.
    """
    spawn_fn = spawn_fn or _spawn_claude
    lint_report_path = intermediate_dir / "step_7_report.json"

    def _default_lints(pkg: Path):
        from mellea_skills_compiler.compile.lints import run_lints
        return run_lints(pkg, strict=False, smoke_check="never")

    def _default_render(pkg: Path) -> Optional[dict]:
        """Run config.py / fixtures writers, then (in descriptor mode) the
        descriptor-to-Python renderer. Returns the descriptor render result
        dict so the loop can carry its failure reason into repair prescriptions.
        Returns ``None`` in legacy mode (no descriptor emission expected).
        """
        import mellea_skills_compiler
        from mellea_skills_compiler.compile.writer_renderer import (
            default_writer_specs,
            render_descriptor_to_python,
            render_writers,
        )
        _compiler_pkg_dir = Path(mellea_skills_compiler.__file__).resolve().parent
        _writers_repo_root = _resolve_writers_repo_root(_compiler_pkg_dir)
        render_writers(
            pkg,
            default_writer_specs(_writers_repo_root),
            enforce=True,
        )
        # Mirror the descriptor-mode render that ``compile()`` does
        # post-session, so lint-repair loops observe the same on-disk
        # state. No-op when descriptor_emission.json is absent (legacy
        # compiles), so this is safe for both modes.
        if use_descriptor:
            descriptor_render_result = render_descriptor_to_python(
                pkg, skill_root=spec_path.parent if not spec_path.is_dir() else spec_path
            )
            if descriptor_render_result is not None:
                verdict = descriptor_render_result.get("verdict")
                if verdict == "failed":
                    LOGGER.warning(
                        "[writer:descriptor] render failed during lint-repair: %s",
                        descriptor_render_result.get("reason", "unknown"),
                    )
            return descriptor_render_result
        return None

    run_lints_fn = run_post_session_lints or _default_lints
    render_fn = run_writer_renderer or _default_render

    rounds_used = 0
    descriptor_render_failure_reason: Optional[str] = None
    while True:
        # 1. Run writer-renderer (config.py + fixtures/) so lint sees the
        #    authoritative wrapper-rendered files. Best-effort: a render
        #    failure should not crash the repair loop — the lint will
        #    surface the problem.
        descriptor_render_failure_reason = None
        try:
            render_result = render_fn(package_dir)
            if (
                isinstance(render_result, dict)
                and render_result.get("verdict") == "failed"
            ):
                descriptor_render_failure_reason = render_result.get(
                    "reason", "unknown"
                )
        except Exception as render_exc:
            LOGGER.warning(
                "Writer renderer failed during lint-repair round %d: %s",
                rounds_used,
                render_exc,
            )

        # 2. Run lints.
        try:
            lint_result = run_lints_fn(package_dir)
        except Exception as lint_exc:
            LOGGER.warning(
                "Lint runner crashed during lint-repair round %d: %s. "
                "Aborting retry loop; caller will see the original error.",
                rounds_used,
                lint_exc,
            )
            return

        overall = getattr(lint_result, "overall_verdict", None)
        if overall != "fail" and descriptor_render_failure_reason is None:
            # Lints clean (pass) or only WARNING / INFO failures present
            # AND descriptor render succeeded — the gate considers this
            # success. Descriptor render failure forces a repair round
            # even when lints pass (e.g. lint observed stale pipeline.py
            # from a prior round) so the repair prompt always carries the
            # authoritative renderer reason.
            return

        if rounds_used >= max_repair_rounds:
            LOGGER.warning(
                "Lint-repair loop exhausted %d repair round(s); ERROR-severity "
                "lint failures persist. Returning to caller for the final "
                "failure path.",
                max_repair_rounds,
            )
            return

        # 3. Bake prescriptions + build augmented prompt + re-spawn.
        prescriptions_path = _bake_repair_prescriptions(
            package_dir, lint_report_path=lint_report_path
        )
        try:
            report = json.loads(lint_report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            report = {}
        failing_lints = _collect_error_failures(report)
        rounds_used += 1
        LOGGER.warning(
            "Wrapper-side lint check found %d ERROR-severity failure(s) "
            "post-session. Lint-repair round %d/%d — re-invoking Claude in "
            "repair mode (prescriptions: %s).",
            len(failing_lints),
            rounds_used,
            max_repair_rounds,
            "yes" if prescriptions_path else "none-matched",
        )
        repair_system_prompt = _build_lint_repair_system_prompt(
            base_system_prompt,
            failing_lints=failing_lints,
            prescriptions_path=prescriptions_path,
            lint_report_path=lint_report_path,
            descriptor_render_failure_reason=descriptor_render_failure_reason,
        )
        repair_argv = _build_claude_argv(
            spec_path=spec_path,
            model=model,
            system_prompt=repair_system_prompt,
            compile_settings_path=compile_settings_path,
            repair_mode=True,
            use_descriptor=use_descriptor,
        )
        spawn_fn(
            claude_argv=repair_argv,
            subprocess_env=subprocess_env,
            intermediate_dir=intermediate_dir,
            timeout=timeout,
            processing=processing,
        )
        # Loop back: re-render, re-lint, decide.


def _smoke_repair_retry(
    smoke_failure,
    *,
    spec_path: Path,
    model: str,
    base_system_prompt: str,
    compile_settings_path: Optional[Path],
    use_descriptor: bool,
    subprocess_env: dict,
    intermediate_dir: Path,
    timeout: int,
    processing,
    spawn_fn: Optional[Callable[..., None]] = None,
) -> None:
    """Re-invoke Claude with a smoke-aware repair prompt and return.

    Single-shot (no internal loop): caller wraps this in their own
    "spawn → re-validate → spawn again?" composition because the
    re-validate step belongs in ``compile()`` where it can also re-run
    the writer-renderer between rounds. Mirrors the shape of the
    lint-repair re-spawn block in :func:`_lint_repair_retry_loop`.

    ``smoke_failure``: the
    :class:`mellea_skills_compiler.compile.smoke_check.SmokeCheckFailure`
    instance that triggered this retry. Its details are folded into the
    repair prompt by :func:`_build_smoke_repair_system_prompt`.
    """
    spawn_fn = spawn_fn or _spawn_claude
    repair_system_prompt = _build_smoke_repair_system_prompt(
        base_system_prompt,
        smoke_failure=smoke_failure,
    )
    repair_argv = _build_claude_argv(
        spec_path=spec_path,
        model=model,
        system_prompt=repair_system_prompt,
        compile_settings_path=compile_settings_path,
        repair_mode=True,
        use_descriptor=use_descriptor,
    )
    LOGGER.warning(
        "Wrapper-side smoke-check failed (%s); spawning Claude in repair "
        "mode with smoke-aware prompt.",
        smoke_failure.kind,
    )
    spawn_fn(
        claude_argv=repair_argv,
        subprocess_env=subprocess_env,
        intermediate_dir=intermediate_dir,
        timeout=timeout,
        processing=processing,
    )


def _spawn_claude_with_lint_repair(
    *,
    initial_claude_argv: list[str],
    spec_path: Path,
    model: str,
    base_system_prompt: str,
    compile_settings_path: Optional[Path],
    use_descriptor: bool,
    subprocess_env: dict,
    intermediate_dir: Path,
    package_dir: Path,
    timeout: int,
    processing,
    max_repair_rounds: int = 2,
    run_post_session_lints: Optional[Callable[[Path], object]] = None,
    run_writer_renderer: Optional[Callable[[Path], None]] = None,
    spawn_fn: Optional[Callable[..., None]] = None,
) -> None:
    """Single-shot compile + post-session lint-repair retry.

    Thin composition of :func:`_spawn_claude` (initial spawn) and
    :func:`_lint_repair_retry_loop` (post-session retry). Kept as a
    convenience entry point and to preserve backward compatibility for
    existing callers; the two layers can also be invoked separately by
    ``compile()`` when ``--resume-on-early-end`` is in play (the resume
    helper handles the initial spawn, then the retry loop runs
    independently).

    See :func:`_lint_repair_retry_loop` for the retry semantics.
    """
    spawn_fn = spawn_fn or _spawn_claude

    # Layer 1: initial spawn (identical to the legacy single-shot path).
    spawn_fn(
        claude_argv=initial_claude_argv,
        subprocess_env=subprocess_env,
        intermediate_dir=intermediate_dir,
        timeout=timeout,
        processing=processing,
    )

    # Layer 2: post-session lint check + repair retry.
    _lint_repair_retry_loop(
        spec_path=spec_path,
        model=model,
        base_system_prompt=base_system_prompt,
        compile_settings_path=compile_settings_path,
        use_descriptor=use_descriptor,
        subprocess_env=subprocess_env,
        intermediate_dir=intermediate_dir,
        package_dir=package_dir,
        timeout=timeout,
        processing=processing,
        max_repair_rounds=max_repair_rounds,
        run_post_session_lints=run_post_session_lints,
        run_writer_renderer=run_writer_renderer,
        spawn_fn=spawn_fn,
    )


def _spawn_claude(
    *,
    claude_argv: list[str],
    subprocess_env: dict,
    intermediate_dir: Path,
    timeout: int,
    processing,
) -> None:
    """Run the prepared ``claude -p`` subprocess and stream its output.

    Extracted from :func:`compile` (Phase 3.5.A §3). Encapsulates:

    * spawning the subprocess with the prepared argv + env,
    * a background thread that captures stderr lines,
    * the stdout streaming loop that pretty-prints assistant text and
      persists every ``claude -p`` stream-json event to
      ``intermediate/claude_stream.jsonl``,
    * timeout enforcement,
    * non-zero return-code translation to :class:`subprocess.SubprocessError`.

    The caller is responsible for tearing down anything outside this helper
    (proxy server, status spinner stop, etc.) — exceptions propagate.
    """
    process = None
    try:
        process = subprocess.Popen(
            claude_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=subprocess_env,
        )

        stderr_lines: list[str] = []

        def read_stderr() -> None:
            for line in iter(process.stderr.readline, ""):
                if line:
                    stderr_lines.append(line.strip())

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        stream_dump_path = intermediate_dir / "claude_stream.jsonl"
        stream_dump_path.parent.mkdir(parents=True, exist_ok=True)
        stream_dump = stream_dump_path.open("w")

        start_time = time.time()
        processing.start()
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Mellea-fy skill compilation failed due to timeout. "
                        f"Process timed out after {elapsed}s (limit: {timeout}s)"
                    )

                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    processing.stop()
                    break

                if output:
                    stream_dump.write(output)
                    stream_dump.flush()
                    try:
                        response = json.loads(output.strip())
                        if response.get("type", None) == ClaudeResponseType.ASSISTANT:
                            for message_content in response.get("message", {}).get(
                                "content", []
                            ):
                                if (
                                    message_content.get("type", None)
                                    == ClaudeResponseMessageType.TEXT
                                ):
                                    console.print(
                                        f"[cyan]{message_content.get('text', '')}[/]\n"
                                    )
                    except json.decoder.JSONDecodeError as e:
                        console.print("Claude message parsing error: " + str(e))
        finally:
            stream_dump.close()

        stderr_thread.join(timeout=1)
        return_code = process.wait(timeout=1)
        if return_code != 0:
            raise subprocess.SubprocessError(
                f"Mellea-fy skill compilation failed with return code {return_code}. "
                f"Error: {' '.join(stderr_lines)}"
            )
    except Exception:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        raise


def validate(
    package_dir: Path,
    *,
    no_run: bool,
    all_fixtures: bool,
    strict: bool = False,
    smoke_check_mode: str = "never",
) -> None:
    """Shared implementation for the validate command and the compile auto-chain.

    ``strict``: when True, any lint failure (regardless of severity) blocks
    compile. Default False — only ERROR-severity lint failures block. See
    ``.claude/commands/mellea-fy-validate.md`` for the severity table.

    ``smoke_check_mode``: ``"never"`` (default — historic behaviour where
    smoke is handled separately by ``run_smoke_check`` below the gate),
    ``"auto"`` (in-gate smoke when backend available), or ``"always"``
    (in-gate smoke required; failure if backend absent). The historical
    post-lint smoke-check (`--no-run` honoured) still runs after the gate;
    `smoke_check_mode` controls the *in-gate evidence downgrade*, which
    is what determines whether a WARNING-only lint result can be escalated
    by smoke evidence.
    """
    if not package_dir.exists() or not package_dir.is_dir():
        raise Exception("Package directory does not exist: %s", package_dir)

    from mellea_skills_compiler.compile.lints import LintSeverity, run_lints

    lint_result = run_lints(
        package_dir, strict=strict, smoke_check=smoke_check_mode
    )
    if lint_result.failed:
        for lint in lint_result.lints:
            if lint.verdict != "fail":
                continue
            # In strict mode, surface WARNING-severity failures as errors too
            # because they're being treated as blocking.
            if lint.severity == LintSeverity.ERROR or strict:
                LOGGER.error(
                    "[%s][%s] %d failure(s):",
                    lint.lint_id,
                    lint.severity.value,
                    len(lint.failures),
                )
                for failure in lint.failures:
                    location = failure.file
                    if failure.line is not None:
                        location = f"{location}:{failure.line}"
                    LOGGER.error("  %s — %s", location, failure.message)
        raise Exception(
            "Step 7 lints failed. Report at %s/intermediate/step_7_report.json",
            package_dir,
        )

    # Surface non-blocking findings — these are now both legacy "warning"-
    # verdict lints AND new graduated-severity WARNING/INFO failures that
    # didn't block the gate. The operator should still see them.
    for lint in lint_result.lints:
        # Legacy advisory-verdict path (some lints emit verdict="warning"
        # directly rather than verdict="fail" with WARNING severity).
        if lint.verdict == "warning":
            LOGGER.warning(
                "[%s] %d advisory finding(s) (does not block compile):",
                lint.lint_id,
                len(lint.failures),
            )
            for failure in lint.failures:
                location = failure.file
                if failure.line is not None:
                    location = f"{location}:{failure.line}"
                LOGGER.warning("  %s — %s", location, failure.message)
            continue
        # Graduated-severity path: WARNING/INFO failures that didn't block.
        if lint.verdict == "fail" and lint.severity != LintSeverity.ERROR:
            log_fn = (
                LOGGER.warning
                if lint.severity == LintSeverity.WARNING
                else LOGGER.info
            )
            log_fn(
                "[%s][%s] %d finding(s) (does not block compile):",
                lint.lint_id,
                lint.severity.value,
                len(lint.failures),
            )
            for failure in lint.failures:
                location = failure.file
                if failure.line is not None:
                    location = f"{location}:{failure.line}"
                log_fn("  %s — %s", location, failure.message)

    LOGGER.info(
        "Step 7 structural lints passed (%d lints checked; %d warning(s), %d info finding(s)).",
        len(lint_result.lints),
        lint_result.warnings,
        lint_result.info_failures,
    )

    if no_run:
        LOGGER.info("Smoke-check skipped (--no-run).")
        return

    from mellea_skills_compiler.compile.smoke_check import (
        SmokeCheckFailure,
        run_smoke_check,
    )

    try:
        smoke_result = run_smoke_check(package_dir, all_fixtures=all_fixtures)
    except Exception as exc:
        raise SmokeCheckFailure(
            kind="infrastructure",
            infrastructure_error=str(exc),
        )

    if smoke_result.overall_verdict == "failed":
        fixture_failures: list[tuple[str, Optional[str]]] = []
        for fixture in smoke_result.fixtures:
            if fixture.verdict == "failed":
                LOGGER.error(
                    "Fixture '%s' failed: %s",
                    fixture.fixture_id,
                    fixture.failure_message,
                )
                fixture_failures.append(
                    (fixture.fixture_id, fixture.failure_message)
                )
        raise SmokeCheckFailure(
            kind="fixtures",
            fixture_failures=fixture_failures,
            report_path=str(package_dir / "intermediate" / "step_7b_report.json"),
        )

    LOGGER.info(
        "Smoke-check %s — %d fixture(s) executed.",
        smoke_result.overall_verdict,
        len(smoke_result.fixtures),
    )


def compile(
    spec_path: Path,
    model: Optional[str] = None,
    timeout: int = 14400,
    repair_mode: bool = False,
    no_run: bool = False,
    refresh_cache: bool = False,
    skill_backend: Optional[str] = None,
    skill_model: Optional[str] = None,
    use_descriptor: bool = False,
    strict: bool = False,
    smoke_check_mode: str = "never",
    resume_on_early_end: bool = False,
    repair_on_lint_failure: bool = False,
) -> None:
    # clears screen
    subprocess.call("clear")

    # print mellea-fy header
    console.print()
    if repair_mode:
        console.rule(
            f"[bold yellow] Melleafy Repair: Inspect and Resume a Partial or Failed Run[/]"
        )
    else:
        console.rule(
            f"[bold yellow] Melleafy: Decompose an Agent Spec into Mellea Code[/]"
        )
    console.print()

    # For spec file input only: verify that file ends in a .md extension
    if spec_path.suffix and spec_path.suffix != ".md":
        raise ValueError(
            f"Skill specification input can only be a markdown (.md) file or a valid skill directory."
        )
    # For [spec file / spec directory] input, Verify that destination exists
    elif not spec_path.exists():
        raise FileNotFoundError(
            f"The skill specification file or directory cannot be found: {spec_path}"
        )

    # print specs frontmatter if available
    if spec_md_path := _get_spec_md_path(spec_path):
        try:
            specs = parse_spec_file(spec_md_path)
            rprint(
                Panel(
                    json.dumps(
                        specs.get("frontmatter", {"Name", spec_path.name}), indent=2
                    ),
                    title="Specification",
                    subtitle=str(spec_path),
                )
            )
        except Exception:
            console.print(f"Spec Path: " + str(spec_path))
    else:
        console.print(f"Spec Path: " + str(spec_path))

    # Check and verify claude model
    available_models = [model.id for model in Anthropic().models.list()]
    if not available_models:
        raise ValueError(f"No claude models available with your API key.")

    if model:
        if model in available_models:
            # user provided model in available in available models.
            pass
        else:
            raise ValueError(
                f"Invalid Claude model provided - {model}\nAvailable: {available_models}"
            )
    else:
        # User did not provide the Claude model. Therefore, filter the available models by the GraniteClaw default and select the first one.
        models = [
            model
            for model in available_models
            if InferenceModel.CLAUDE_MODEL in model.lower()
        ]
        if not models:
            # Available models does not have the GraniteClaw default. Ask user to choose one.
            raise ValueError(
                f"Please provide claude model via --model option.\nAvailable: {available_models}"
            )
        else:
            # Use the first model to compile given skill
            model = models[0]

    console.print(
        f"\n[green]{'Repairing' if repair_mode else 'Compiling'} using Claude model:[/] {model}\n"
    )

    # Start a local proxy that strips context_management from API requests.
    # The IBM LiteLLM proxy rejects that field; Claude Code sends it automatically.
    # Forward to the real upstream (ANTHROPIC_BASE_URL if set, else api.anthropic.com).
    _real_base = os.environ.get(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
    ).rstrip("/")
    _parsed = urlparse(_real_base)
    proxy_server = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0), ContextMgmtStrippingProxy
    )
    proxy_server.allow_reuse_address = True
    proxy_server.upstream_scheme = _parsed.scheme
    proxy_server.upstream_host = _parsed.netloc
    proxy_server.upstream_path_prefix = _parsed.path
    proxy_port = proxy_server.server_address[1]
    proxy_thread = threading.Thread(target=proxy_server.serve_forever)
    proxy_thread.daemon = True
    proxy_thread.start()

    subprocess_env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{proxy_port}",
    }

    # Rule OUT-6 — mirror companion directories from skill root into the
    # package directory BEFORE invoking mellea-fy. This is deterministic
    # plumbing (not the LLM's job) so the mirror cannot be skipped or
    # mis-applied. The LLM then generates code in a package directory that
    # already contains its bundled scripts/references/assets, reinforcing
    # the Path(__file__).parent path-resolution invariant.
    skill_dir = spec_path if spec_path.is_dir() else spec_path.parent
    _frontmatter: dict | None = None
    # Locate the spec.md/SKILL.md regardless of whether the input is a file
    # or a workspace directory. Parsing its frontmatter is what lets the
    # wrapper's package-name choice match the slash command's choice (which
    # always reads frontmatter). Without this step, dir-input compiles emit
    # files into one `<dir-name>_mellea/` while the LLM writes intermediates
    # into a different `<frontmatter-name>_mellea/`.
    _spec_for_frontmatter = _get_spec_md_path(spec_path)
    if _spec_for_frontmatter is not None:
        try:
            _frontmatter = parse_spec_file(_spec_for_frontmatter).get("frontmatter")
        except Exception:
            # parse_spec_file uses strict yaml.safe_load and fails on specs
            # whose `description:` contains unquoted colons / parens. The
            # slash command tolerates this; the wrapper must too. Fall back
            # to a forgiving regex extraction of just the `name:` field
            # (everything else can stay None — we only need name for the
            # package-directory choice).
            _frontmatter = _tolerant_name_extract(_spec_for_frontmatter)
    package_name = derive_package_name(spec_path, _frontmatter)
    package_dir = skill_dir / package_name
    try:
        mirrored = mirror_companion_dirs(skill_dir, package_dir)
        if mirrored:
            LOGGER.info(
                "Mirrored companion dirs into %s/: %s (Rule OUT-6)",
                package_name,
                ", ".join(mirrored),
            )
    except Exception as mirror_exc:
        LOGGER.warning(
            "Companion-directory mirror failed for %s: %s. mellea-fy will continue.",
            package_dir,
            mirror_exc,
        )

    # Pre-populate the deterministic grounding artifacts (Steps 2.5e and 2.5f
    # of mellea-fy). The slash command runs with --allowed-tools Read,Write,Edit,
    # so it cannot introspect the installed mellea package or fetch
    # docs.mellea.ai itself. We write `mellea_api_ref.json` and
    # `mellea_doc_index.json` here; the slash command's responsibility shrinks
    # to verifying the files exist and consuming them.
    intermediate_dir = package_dir / "intermediate"
    try:
        write_mellea_api_ref(intermediate_dir, refresh=refresh_cache)
        write_mellea_doc_index(intermediate_dir, refresh=refresh_cache)
    except Exception as exc:
        LOGGER.warning(
            "Grounding generation failed: %s. mellea-fy will fall back.", exc
        )

    # Resolve which backend and model the compiled skill will use at runtime,
    # record the choice for the post-compile lint, and bake the values into
    # the system prompt so the LLM puts the correct constants in config.py.
    chosen_backend, chosen_model_id, defaults_source = resolve_runtime_defaults(
        skill_backend, skill_model
    )
    LOGGER.info(
        "Compiled skill will use backend=%r, model=%r (from %s).",
        chosen_backend,
        chosen_model_id,
        defaults_source,
    )
    try:
        write_runtime_directive(
            intermediate_dir, chosen_backend, chosen_model_id, defaults_source
        )
    except Exception as exc:
        LOGGER.warning(
            "Could not record runtime directive (%s). Compile will continue; "
            "the post-compile lint will skip its runtime-defaults check.",
            exc,
        )
    system_prompt = build_system_prompt(
        chosen_backend, chosen_model_id, defaults_source
    )

    # Write the per-invocation Claude Code settings file with deny rules for
    # the paths the wrapper renders authoritatively (currently config.py).
    # Passed to claude via --settings; deny rules are honoured deterministically
    # in -p mode (verified in the synthetic test).
    try:
        compile_settings_path = write_compile_settings(
            intermediate_dir, package_dir, use_descriptor=use_descriptor
        )
    except Exception as exc:
        LOGGER.warning(
            "Could not write per-invocation settings (%s). Falling back to no "
            "deny rules; the wrapper will still overwrite wrapper-rendered paths.",
            exc,
        )
        compile_settings_path = None

    # Start compilation process. Both legacy and descriptor branches use the
    # same /mellea-fy slash command via the same `_spawn_claude` helper; the
    # `use_descriptor` flag is appended to the slash-command argument so the
    # orchestrator routes Step 5 accordingly.
    claude_argv = _build_claude_argv(
        spec_path=spec_path,
        model=model,
        system_prompt=system_prompt,
        compile_settings_path=compile_settings_path,
        repair_mode=repair_mode,
        use_descriptor=use_descriptor,
    )

    processing = console.status(
        "[italic bold yellow]Processing...[/]", spinner_style="status.spinner"
    )

    try:
        # The two retry layers compose independently: resume handles
        # incomplete pipelines (Claude bailed mid-pipeline; canonical
        # artefacts missing), lint-repair handles complete-but-broken
        # pipelines (Claude self-reported success but wrapper-side lints
        # disagree). Either can fire alone; both can fire in sequence on
        # the same compile. Both default OFF — opt in via the CLI flags
        # ``--resume-on-early-end`` and ``--repair-on-lint-failure``.

        # Layer 1: spawn Claude (with optional resume for incomplete
        # pipelines). The resume helper internally spawns the first
        # ``claude -p`` and re-invokes up to ``max_resumes`` times when
        # canonical step artefacts are missing.
        if resume_on_early_end:
            _spawn_claude_with_resume(
                spec_path=spec_path,
                model=model,
                base_system_prompt=system_prompt,
                compile_settings_path=compile_settings_path,
                repair_mode=repair_mode,
                use_descriptor=use_descriptor,
                subprocess_env=subprocess_env,
                intermediate_dir=intermediate_dir,
                package_dir=package_dir,
                timeout=timeout,
                processing=processing,
            )
        else:
            _spawn_claude(
                claude_argv=claude_argv,
                subprocess_env=subprocess_env,
                intermediate_dir=intermediate_dir,
                timeout=timeout,
                processing=processing,
            )

        # Layer 2: post-session lint-failure retry (independent flag
        # check). Runs after Layer 1 has produced its final package
        # state; bakes F1 prescriptions and re-spawns Claude in repair
        # mode up to ``max_repair_rounds`` times when wrapper-side Step 7
        # lints detect ERROR-severity failures.
        if repair_on_lint_failure:
            _lint_repair_retry_loop(
                spec_path=spec_path,
                model=model,
                base_system_prompt=system_prompt,
                compile_settings_path=compile_settings_path,
                use_descriptor=use_descriptor,
                subprocess_env=subprocess_env,
                intermediate_dir=intermediate_dir,
                package_dir=package_dir,
                timeout=timeout,
                processing=processing,
            )

        # copy spec file into the compiled directory (name may differ from frontmatter
        # because melleafy normalises hyphens → underscores per Rule OUT-2)
        skill_dir = spec_path if spec_path.is_dir() else spec_path.parent
        mellea_dirs = [
            d for d in skill_dir.iterdir() if d.is_dir() and d.name.endswith("_mellea")
        ]
        if mellea_dirs:
            # Wrapper-side writer invocation.
            # Reads intermediate/<artifact>_emission.json, runs the deterministic
            # writer in .claude/melleafy/writers/, and writes the canonical
            # artifact (config.py + fixtures/) over whatever the LLM produced.
            #
            # Locate the writers directory by walking up from the *installed
            # compiler package*, NOT from the generated package's directory.
            # The older walk-up-from-package logic only worked when the spec
            # was compiled in-tree; out-of-tree skill specs (the standard
            # eval-harness use case) silently fell off the top of the
            # filesystem, defaulted repo_root to the package dir, and shipped
            # packages missing config.py and fixtures/. The renderer error was
            # then swallowed as a "non-fatal during migration" WARNING. Both
            # failure modes are now hard errors. See
            # `_resolve_writers_repo_root` for the resolution helper.
            import mellea_skills_compiler
            from mellea_skills_compiler.compile.writer_renderer import (
                default_writer_specs,
                render_descriptor_to_python,
                render_writers,
            )

            _compiler_pkg_dir = Path(mellea_skills_compiler.__file__).resolve().parent
            _writers_repo_root = _resolve_writers_repo_root(_compiler_pkg_dir)
            render_writers(
                mellea_dirs[0],
                default_writer_specs(_writers_repo_root),
                enforce=True,  # config.py + fixtures/ are mandatory artifacts
            )

            # Descriptor-mode post-session render: when ``--use-descriptor``
            # is in play, Claude emits ``intermediate/descriptor_emission.json``
            # and the slash command's contract is "the wrapper renders
            # pipeline.py / schemas.py from this". Three skills in the May
            # 17 overnight batch shipped descriptor_emission.json on disk
            # with NO pipeline.py / schemas.py because this render step
            # never ran — the lint then false-positive-passed (skipped on
            # missing pipeline.py) and the smoke check crashed. The
            # invocation is idempotent and a no-op in legacy mode.
            if use_descriptor:
                descriptor_render_result = render_descriptor_to_python(
                    mellea_dirs[0],
                    skill_root=skill_dir,
                )
                if descriptor_render_result is None:
                    # Defensive: render_descriptor_to_python returns a dict
                    # by contract; if a future change relaxes that to None,
                    # treat it as a soft skip.
                    LOGGER.warning(
                        "[writer:descriptor] render returned None — "
                        "treating as skipped; lint will validate disk state."
                    )
                elif descriptor_render_result["verdict"] == "rendered":
                    LOGGER.info(
                        "[writer:descriptor] rendered %d file(s) from "
                        "descriptor_emission.json: %s",
                        len(descriptor_render_result["files_written"]),
                        ", ".join(descriptor_render_result["files_written"]),
                    )
                elif descriptor_render_result["verdict"] == "skipped":
                    LOGGER.warning(
                        "[writer:descriptor] descriptor_emission.json absent "
                        "— Claude must have written pipeline.py / schemas.py "
                        "directly (legacy path inside --use-descriptor mode). "
                        "pipeline-entry-canonical lint will validate the result."
                    )
                elif descriptor_render_result["verdict"] == "failed":
                    # Hard-fail rather than log-and-proceed: smoke-check would
                    # otherwise crash with the misleading "pipeline.py missing"
                    # error, hiding the precise descriptor reason. If
                    # --repair-on-lint-failure was active, the lint-repair
                    # loop has already had its chance with this reason in the
                    # repair prompt; reaching this branch means repair did
                    # not produce a renderable descriptor. Raise with the
                    # renderer's reason so the failure class is unambiguous.
                    raise Exception(
                        "descriptor render failed: "
                        f"{descriptor_render_result.get('reason', 'unknown')}"
                    )

            # validate compiled skill pipeline. When --repair-on-lint-failure
            # is set, a smoke-check failure (either infrastructure or fixture)
            # triggers ONE smoke-aware repair re-spawn before failing — the
            # lint-repair loop covers Step 7, this covers Step 8. Anything
            # other than SmokeCheckFailure propagates unchanged.
            from mellea_skills_compiler.compile.smoke_check import SmokeCheckFailure
            try:
                validate(
                    mellea_dirs[0],
                    no_run=no_run,
                    all_fixtures=False,
                    strict=strict,
                    smoke_check_mode=smoke_check_mode,
                )
            except SmokeCheckFailure as smoke_failure:
                if not repair_on_lint_failure:
                    raise
                _smoke_repair_retry(
                    smoke_failure,
                    spec_path=spec_path,
                    model=model,
                    base_system_prompt=system_prompt,
                    compile_settings_path=compile_settings_path,
                    use_descriptor=use_descriptor,
                    subprocess_env=subprocess_env,
                    intermediate_dir=intermediate_dir,
                    timeout=timeout,
                    processing=processing,
                )
                # Re-render wrapper-side artefacts (Claude may have changed
                # the descriptor or fixture-emission JSON during the smoke
                # repair) and re-validate. A second smoke failure is
                # terminal — we do not loop smoke repair the way lint repair
                # loops; one chance to fix the smoke is the contract.
                render_writers(
                    mellea_dirs[0],
                    default_writer_specs(_writers_repo_root),
                    enforce=True,
                )
                if use_descriptor:
                    second_render = render_descriptor_to_python(
                        mellea_dirs[0], skill_root=skill_dir
                    )
                    if (
                        second_render is not None
                        and second_render.get("verdict") == "failed"
                    ):
                        raise Exception(
                            "descriptor render failed after smoke repair: "
                            f"{second_render.get('reason', 'unknown')}"
                        )
                validate(
                    mellea_dirs[0],
                    no_run=no_run,
                    all_fixtures=False,
                    strict=strict,
                    smoke_check_mode=smoke_check_mode,
                )

            if spec_md_path:
                shutil.copy(spec_md_path, mellea_dirs[0] / SpecFileFormat.SKILL_FILE_MD)
        else:
            raise Exception(
                f"No *_mellea directory found in {skill_dir} after compilation"
            )

    except (TimeoutError, subprocess.SubprocessError):
        processing.stop()
        raise
    except Exception as e:
        processing.stop()
        raise Exception(f"Mellea-fy skill compilation failed: {str(e)}") from e
    finally:
        proxy_server.shutdown()

    console.print(
        f"\nMelleafy {'Repair' if repair_mode else 'Compile'} completed successfully.\n"
    )
