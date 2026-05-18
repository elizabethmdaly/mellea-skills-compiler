import signal
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer

from mellea_skills_compiler.enums import InferenceEngineType
from mellea_skills_compiler.toolkit.logging import configure_logger


app = typer.Typer(no_args_is_help=True)
contract_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and verify the pipeline contract registry "
    "(see `src/mellea_skills_compiler/compile/pipeline_contract.py`).",
)
app.add_typer(contract_app, name="contract")
LOGGER = configure_logger()


def signal_handler(sig, frame):
    print("\nInterrupted. Exiting...")
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@app.callback()
def main() -> None:
    """
    Mellea Skills Compiler CLI - Agent specification certification pipeline.

    Transform AI agent skill specifications into certified, governed pipelines
    with comprehensive risk analysis and compliance reporting.
    """


@app.command(
    help="Melleafy Compile: Decompose an Agent Spec into Mellea Code",
    epilog="Compile Mellea skill specification into a Mellea pipeline using mellea-fy Claude command.",
)
def compile(
    ctx: typer.Context,
    spec_path: Annotated[
        str,
        typer.Argument(
            help="Path to the Mellea skill specification or the folder containing the skill specs.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Claude model for compiling a skill. Default to Sonnet.",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            help=(
                "Claude session timeout in seconds. Default 14400 (4 hours). "
                "Raised from the previous 4500s (75min) because complex skills "
                "with multi-lint repair sequences were being cut mid-fix. "
                "Increase further on demand; the only downside of a high cap "
                "is that a genuinely-hung session takes longer to detect."
            ),
        ),
    ] = 14400,
    repair_mode: Annotated[
        bool,
        typer.Option(
            "-r",
            "--repair-mode",
            help="Identify and correct any errors effectively in Mellea skill compilation.",
        ),
    ] = False,
    no_run: Annotated[
        bool,
        typer.Option(
            "--no-run",
            help="Skip the post-compile fixture smoke-check (default ON).",
        ),
    ] = False,
    refresh_cache: Annotated[
        bool,
        typer.Option(
            "--refresh-cache",
            help="Force refresh of grounding caches (~/.cache/mellea-skills-compiler/) before compile.",
        ),
    ] = False,
    skill_backend: Annotated[
        Optional[str],
        typer.Option(
            "--skill-backend",
            help="Override the LLM backend used by the compiled skill at runtime "
            "(default from mellea_skills_compiler/compile/claude/data/runtime_defaults.json).",
        ),
    ] = None,
    skill_model: Annotated[
        Optional[str],
        typer.Option(
            "--skill-model",
            help="Override the model id used by the compiled skill at runtime "
            "(default from mellea_skills_compiler/compile/claude/data/runtime_defaults.json).",
        ),
    ] = None,
    use_descriptor: Annotated[
        bool,
        typer.Option(
            "--use-descriptor",
            help="Route Step 5 through descriptor IR emission + render instead of "
            "free-form Python emission. Default off until the Phase 5 flip. "
            "When set, --timeout is not directly applicable (streaming + 64K). "
            "See .claude/commands/mellea-fy-generate.md §'Descriptor mode' and "
            "descriptor-schema-v0.x.md for the IR spec.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail compile on any lint failure, regardless of severity. "
            "Use this to restore pre-graduated-severity behaviour. "
            "Default: only ERROR-severity failures block compilation. "
            "See .claude/commands/mellea-fy-validate.md for the per-lint severity table.",
        ),
    ] = False,
    smoke_check: Annotated[
        str,
        typer.Option(
            "--smoke-check",
            help="Evidence-based smoke check after lints. 'auto' (default): "
            "run if backend available; 'always': require backend, fail if "
            "unavailable; 'never': skip entirely (use --strict to compensate). "
            "One of: auto|always|never.",
        ),
    ] = "auto",
    resume_on_early_end: Annotated[
        bool,
        typer.Option(
            "--resume-on-early-end",
            help="Detect when Claude ended its turn before Step 6 completed "
            "(e.g., 'Step 0 complete. Proceeding to Step 1a.' with no "
            "follow-up tool call) and automatically re-invoke Claude to "
            "resume from the next step. Up to 3 resume rounds per skill. "
            "Default: False (legacy behaviour — single shot).",
        ),
    ] = False,
    repair_on_lint_failure: Annotated[
        bool,
        typer.Option(
            "--repair-on-lint-failure",
            help=(
                "When the wrapper detects error-severity Step 7 lint failures "
                "after the Claude session ends, automatically spawn a second "
                "Claude session in repair mode with structured F1 fix "
                "prescriptions. Up to 2 repair rounds per compile. "
                "Default: False (legacy behaviour — compile fails hard on lint)."
            ),
        ),
    ] = False,
):
    """
    Compile Mellea skill specification into a Mellea pipeline using mellea-fy Claude command.

    After mellea-fy returns successfully, automatically chains into validate:
    runs the structural lints (Step 7) and (unless --no-run) executes one fixture
    against the LLM backend. A green compile means compiled + lints passed +
    smoke-check passed (or skipped because backend was unreachable).
    """
    spec_path = Path(spec_path)
    if smoke_check not in {"auto", "always", "never"}:
        LOGGER.error(
            "--smoke-check must be one of auto|always|never, got %r", smoke_check
        )
        raise typer.Exit(code=2)
    try:
        if use_descriptor:
            _run_descriptor_compile(
                spec_path=spec_path,
                model=model,
                timeout=timeout,
                repair_mode=repair_mode,
                no_run=no_run,
                refresh_cache=refresh_cache,
                skill_backend=skill_backend,
                skill_model=skill_model,
                strict=strict,
                smoke_check=smoke_check,
                resume_on_early_end=resume_on_early_end,
                repair_on_lint_failure=repair_on_lint_failure,
            )
            return

        from mellea_skills_compiler.compile import mellea_skills

        mellea_skills.compile(
            spec_path,
            model,
            timeout,
            repair_mode=repair_mode,
            no_run=no_run,
            refresh_cache=refresh_cache,
            skill_backend=skill_backend,
            skill_model=skill_model,
            strict=strict,
            smoke_check_mode=smoke_check,
            resume_on_early_end=resume_on_early_end,
            repair_on_lint_failure=repair_on_lint_failure,
        )
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


def _resolve_descriptor_spec_path(spec_path: Path) -> Path:
    """Resolve a CLI `spec_path` argument to a concrete spec file.

    Thin wrapper around
    :func:`mellea_skills_compiler.compile.spec_path_resolver.resolve_spec_path`
    that returns just the spec file (matching the legacy callsite shape).
    Kept for backward compatibility with any direct callers.

    Widened (Phase 3.5.A §2.1) so a directory with no ``spec.md`` / ``SKILL.md``
    falls back to the first ``.md`` file found in the directory, and a
    direct ``.md`` file path resolves to itself. Precedence inside a
    directory remains ``SKILL.md`` → ``spec.md`` → first ``*.md`` alphabetical.
    """
    from mellea_skills_compiler.compile.spec_path_resolver import (
        SpecPathResolutionError,
        resolve_spec_path,
    )

    try:
        return resolve_spec_path(spec_path).spec_file
    except SpecPathResolutionError as exc:
        # Preserve the legacy contract: callers expect ``ValueError`` /
        # ``FileNotFoundError`` here, not a custom subtype.
        raise ValueError(str(exc)) from exc


def _run_descriptor_compile(
    *,
    spec_path: Path,
    model: Optional[str],
    timeout: int,
    repair_mode: bool,
    no_run: bool,
    refresh_cache: bool,
    skill_backend: Optional[str],
    skill_model: Optional[str],
    strict: bool = False,
    smoke_check: str = "auto",
    resume_on_early_end: bool = False,
    repair_on_lint_failure: bool = False,
) -> None:
    """Descriptor-IR branch of `compile` (Phase 3.5.A wiring).

    Spawns the same ``claude`` subprocess the legacy path spawns, driving the
    full 10-step ``/mellea-fy`` workflow with Step 5 swapped for descriptor IR
    emission + render. The slash-command orchestrator parses
    ``--use-descriptor`` out of ``$ARGUMENTS`` (see ``.claude/commands/mellea-fy.md``)
    and routes Step 5 accordingly.

    Prior to Phase 3.5.A this routed to an in-process
    :func:`mellea_skills_compiler.compile.descriptor_emission.compile_via_descriptor`
    call, which skipped Steps 0–4 and Step 6 entirely. That in-process
    entrypoint is preserved for unit tests, ``scripts/batch_descriptor_test.py``,
    and ``scripts/corpus_compare.py`` (one-shot mode), but end-user
    invocations now go through the full ``/mellea-fy`` workflow so descriptor
    mode is functionally equivalent to legacy.
    """
    from mellea_skills_compiler.compile import mellea_skills
    from mellea_skills_compiler.compile.spec_path_resolver import (
        SpecPathResolutionError,
        resolve_spec_path,
    )

    # Validate the spec-path shape early with the widened resolver so the user
    # gets a clear error when neither a .md file nor a workspace directory is
    # supplied. `mellea_skills.compile` itself accepts the same shapes via the
    # legacy `_get_spec_md_path`; we resolve here purely for early validation.
    try:
        resolve_spec_path(spec_path)
    except SpecPathResolutionError as exc:
        raise ValueError(str(exc)) from exc

    mellea_skills.compile(
        spec_path,
        model,
        timeout,
        repair_mode=repair_mode,
        no_run=no_run,
        refresh_cache=refresh_cache,
        skill_backend=skill_backend,
        skill_model=skill_model,
        use_descriptor=True,
        strict=strict,
        smoke_check_mode=smoke_check,
        resume_on_early_end=resume_on_early_end,
        repair_on_lint_failure=repair_on_lint_failure,
    )


@app.command(
    help="Validate a compiled Mellea skill (Step 7 lints + fixture smoke-check)"
)
def validate(
    ctx: typer.Context,
    pipeline_dir: Annotated[
        str,
        typer.Argument(
            help="Compiled skill pipeline directory (e.g. skills/<name>/<name>_mellea/).",
        ),
    ],
    no_run: Annotated[
        bool,
        typer.Option(
            "--no-run",
            help="Skip the post-lint fixture smoke-check (default ON).",
        ),
    ] = False,
    all_fixtures: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Run all fixtures (default: first only).",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail validation on any lint failure, regardless of severity. "
            "Default: only ERROR-severity failures block. "
            "See .claude/commands/mellea-fy-validate.md for the per-lint severity table.",
        ),
    ] = False,
    smoke_check: Annotated[
        str,
        typer.Option(
            "--smoke-check",
            help="Evidence-based in-gate smoke check. 'auto': run if backend "
            "available; 'always': require backend; 'never' (default): skip "
            "the in-gate smoke (legacy post-lint smoke still runs unless "
            "--no-run is set). One of: auto|always|never.",
        ),
    ] = "never",
):
    """
    Run Step 7 structural lints and (unless --no-run) the fixture smoke-check.

    Exit codes:
      0  — lints passed, smoke-check passed or skipped (backend unreachable)
      11 — at least one lint failed
      12 — smoke-check failed (lint pass, but a fixture raised an exception)
    """
    if smoke_check not in {"auto", "always", "never"}:
        LOGGER.error(
            "--smoke-check must be one of auto|always|never, got %r", smoke_check
        )
        raise typer.Exit(code=2)
    try:
        from mellea_skills_compiler.compile import mellea_skills

        mellea_skills.validate(
            Path(pipeline_dir),
            no_run=no_run,
            all_fixtures=all_fixtures,
            strict=strict,
            smoke_check_mode=smoke_check,
        )
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


@app.command(help="Run Mellea Skill Pipeline")
def run(
    ctx: typer.Context,
    pipeline_dir: Annotated[
        str,
        typer.Argument(
            help="Compiled skill pipeline directory.",
        ),
    ],
    fixture: Annotated[
        str,
        typer.Option(
            "--fixture",
            "-f",
            help="Run pipeline for a specific fixture.",
        ),
    ],
    enforce: Annotated[
        bool,
        typer.Option(
            "--enforce",
            "-e",
            help="Block execution when Guardian detects risks (default: audit-only).",
        ),
    ] = False,
    no_guardian: Annotated[
        bool,
        typer.Option(
            "--no-guardian",
            "-ng",
            help="Skip Guardian checks even if a policy manifest exists.",
        ),
    ] = False,
):
    """
    Run Mellea Skill Pipeline.
    """

    try:
        from mellea_skills_compiler.certification.pipeline import skill_pipeline

        skill_pipeline(
            Path(pipeline_dir), fixture, enforce=enforce, no_guardian=no_guardian
        )
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


@app.command(help="Run Risk Analysis and Policy Generation Pipeline for Mellea skills")
def ingest(
    ctx: typer.Context,
    spec_path: Annotated[
        str,
        typer.Argument(
            help="Mellea Skill spec path",
            rich_help_panel="Arguments",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Preview without making LLM calls", show_default=True
        ),
    ] = False,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model to use for Risk and Action Identification. The `--inference-engine` option must support the model. If set to None, the default model for the inference engine will be used.",
        ),
    ] = None,
    inference_engine: Annotated[
        Literal["ollama"],
        typer.Option(
            "--inference-engine",
            "-i",
            callback=lambda x: x.upper(),
            help="Service to use for LLM inference. Supported: ollama",
        ),
    ] = "ollama",
):
    """
    Run risk analysis on a skill specification.

    Analyzes the skill spec to identify risks and generate a policy document
    without running full certification.
    """
    try:
        from mellea_skills_compiler.certification.ingest import ingest_one

        ingest_one(
            Path(spec_path),
            dry_run,
            model,
            InferenceEngineType[inference_engine],
        )
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


@app.command(help="Run Full Certification Pipeline for Mellea skill")
def certify(
    ctx: typer.Context,
    pipeline_dir: Annotated[
        str,
        typer.Argument(
            help="Compiled skill pipeline directory.",
            rich_help_panel="Arguments",
        ),
    ],
    fixture: Annotated[
        Optional[str],
        typer.Option(
            "--fixture",
            "-f",
            help="Run pipeline for a specific fixture.",
        ),
    ] = None,
    enforce: Annotated[
        bool,
        typer.Option(
            "--enforce",
            "-e",
            help="Run pipeline in enforce mode (block on risk detection)",
        ),
    ] = False,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model to use for Risk and Action Identification. The `--inference-engine` option must support the model. If set to None, the default model for the inference engine will be used.",
        ),
    ] = None,
    guardian_model: Annotated[
        Optional[str],
        typer.Option(
            "--guardian-model",
            "-g",
            help="Model to use for Risk Assessment. The `--inference-engine` option must support the model. If set to None, the default guardian model for the inference engine will be used.",
        ),
    ] = None,
    inference_engine: Annotated[
        Literal["ollama"],
        typer.Option(
            "--inference-engine",
            "-i",
            callback=lambda x: x.upper(),
            help="Service to use for LLM inference. Supported: ollama",
        ),
    ] = "ollama",
):
    """
    Run full certification pipeline on a compiled skill.

    Performs comprehensive certification including risk identification,
    compliance classification, and runtime Guardian checks.
    """
    try:
        from mellea_skills_compiler.certification.pipeline import full_pipeline

        full_pipeline(
            Path(pipeline_dir),
            fixture,
            enforce,
            model,
            guardian_model,
            InferenceEngineType[inference_engine],
        )
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


@app.command(
    help="[EXPERIMENTAL] Export a compiled Mellea skill to a deployment target"
)
def export(
    ctx: typer.Context,
    package_path: Annotated[
        str,
        typer.Argument(
            help="Path to the compiled skill directory (must contain melleafy.json).",
        ),
    ],
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="Deployment target: langgraph | claude-code | mcp",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite output directory if it already exists.",
        ),
    ] = False,
):
    """
    Export a compiled Mellea skill to a deployment target (langgraph, claude-code, or mcp).

    Output is written to <package_name>/<package_name>-<target> inside the skill directory.

    NOTE: This command is experimental. Output structure and CLI interface may change
    in future releases without a deprecation period.
    """
    LOGGER.warning(
        "export is an experimental feature — output structure and interface may change between releases"
    )
    try:
        from mellea_skills_compiler.export.exporter import Invocation, run_export

        inv = Invocation(
            package_path=Path(package_path),
            target=target,
            force=force,
        )
        result = run_export(inv)
        LOGGER.info(
            "Export complete: %d files, %d bytes → %s",
            result.files_written,
            result.bytes_written,
            result.out_path,
        )
    except SystemExit:
        raise
    except Exception as e:
        LOGGER.error(str(e))
        raise typer.Exit(code=1)


@contract_app.command("show")
def contract_show(
    json_out: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the contract + computed graph as JSON (machine-readable).",
        ),
    ] = False,
    order_only: Annotated[
        bool,
        typer.Option(
            "--order",
            help="Print only the topological step order (one step_id per line).",
        ),
    ] = False,
) -> None:
    """Show the pipeline contract — Mermaid + topological order + per-step I/O."""
    from mellea_skills_compiler.compile.pipeline_contract import (
        PIPELINE_CONTRACT,
        render_contract_json,
        render_markdown_report,
        topological_order,
    )

    if order_only:
        for sid in topological_order(PIPELINE_CONTRACT):
            typer.echo(sid)
        return
    if json_out:
        typer.echo(render_contract_json(PIPELINE_CONTRACT))
        return
    typer.echo(render_markdown_report(PIPELINE_CONTRACT))


@contract_app.command("verify")
def contract_verify() -> None:
    """Run static checks against the pipeline contract.

    Exit codes:
      0 — no errors (warnings may still surface)
      1 — at least one error
    """
    from mellea_skills_compiler.compile.pipeline_contract import (
        PIPELINE_CONTRACT,
        verify_contract,
    )

    result = verify_contract(PIPELINE_CONTRACT)
    if result.errors:
        typer.echo("Errors:")
        for line in result.errors:
            typer.echo(f"  [x] {line}")
    if result.warnings:
        typer.echo("Warnings:")
        for line in result.warnings:
            typer.echo(f"  [!] {line}")
    if result.ok and not result.warnings:
        typer.echo("[ok] pipeline contract verified — no errors, no warnings")
    elif result.ok:
        typer.echo(
            f"[ok] pipeline contract verified — no errors "
            f"({len(result.warnings)} warning"
            f"{'s' if len(result.warnings) != 1 else ''})"
        )
    else:
        typer.echo(
            f"[fail] pipeline contract verification failed: "
            f"{len(result.errors)} error"
            f"{'s' if len(result.errors) != 1 else ''}, "
            f"{len(result.warnings)} warning"
            f"{'s' if len(result.warnings) != 1 else ''}"
        )
    if not result.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
