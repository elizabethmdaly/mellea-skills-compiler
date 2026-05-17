import json
import os
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
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

    from mellea_skills_compiler.compile.smoke_check import run_smoke_check

    try:
        smoke_result = run_smoke_check(package_dir, all_fixtures=all_fixtures)
    except Exception as exc:
        raise Exception(
            "Smoke-check infrastructure error (could not even start): %s", exc
        )

    if smoke_result.overall_verdict == "failed":
        for fixture in smoke_result.fixtures:
            if fixture.verdict == "failed":
                LOGGER.error(
                    "Fixture '%s' failed: %s",
                    fixture.fixture_id,
                    fixture.failure_message,
                )
        raise Exception(
            "Smoke-check failed. Report at %s/intermediate/step_7b_report.json",
            package_dir,
        )

    LOGGER.info(
        "Smoke-check %s — %d fixture(s) executed.",
        smoke_result.overall_verdict,
        len(smoke_result.fixtures),
    )


def compile(
    spec_path: Path,
    model: Optional[str] = None,
    timeout: int = 4500,
    repair_mode: bool = False,
    no_run: bool = False,
    refresh_cache: bool = False,
    skill_backend: Optional[str] = None,
    skill_model: Optional[str] = None,
    use_descriptor: bool = False,
    strict: bool = False,
    smoke_check_mode: str = "never",
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
        compile_settings_path = write_compile_settings(intermediate_dir, package_dir)
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
        _spawn_claude(
            claude_argv=claude_argv,
            subprocess_env=subprocess_env,
            intermediate_dir=intermediate_dir,
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
                render_writers,
            )

            _compiler_pkg_dir = Path(mellea_skills_compiler.__file__).resolve().parent
            _writers_repo_root = _resolve_writers_repo_root(_compiler_pkg_dir)
            render_writers(
                mellea_dirs[0],
                default_writer_specs(_writers_repo_root),
                enforce=True,  # config.py + fixtures/ are mandatory artifacts
            )

            # validate compiled skill pipeline
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
