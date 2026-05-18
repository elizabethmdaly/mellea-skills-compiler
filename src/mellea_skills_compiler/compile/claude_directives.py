import json
import shutil
from pathlib import Path
from typing import Optional

from mellea_skills_compiler.toolkit.logging import configure_logger


LOGGER = configure_logger()

# Rule OUT-2 — package name derivation; Rule OUT-6 — companion-directory mirror.
# Kept here so the pre-mellea-fy mirror step can resolve the destination
# without invoking the slash command.
_COMPANION_DIRS = ("scripts", "references", "assets")


def derive_package_name(spec_path: Path, frontmatter: dict | None) -> str:
    """Apply Rule OUT-2 (lowercase, hyphens/spaces → underscores, append `_mellea`).

    Resolution order (matches the slash command's convention):

    1. Frontmatter ``name:`` field, if present. This is the source of truth
       per Rule OUT-2 in ``.claude/commands/mellea-fy.md`` — the slash
       command reads frontmatter, and the wrapper MUST agree so file
       writes land in the same package directory.
    2. For multi-file runtimes (CrewAI, LangGraph, Letta — typically
       directory inputs with no ``spec.md`` / ``SKILL.md``), fall back to
       the directory name.
    3. For bare ``.md`` file inputs without frontmatter, fall back to the
       parent directory name.
    """
    frontmatter_name = (frontmatter or {}).get("name")
    if frontmatter_name:
        raw = frontmatter_name
    elif spec_path.is_dir():
        raw = spec_path.name
    else:
        raw = spec_path.parent.name
    name = str(raw).lower().replace("-", "_").replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_") or "skill"
    return f"{name}_mellea"


def mirror_companion_dirs(skill_dir: Path, package_dir: Path) -> list[str]:
    """Rule OUT-6 — mirror companion directories from skill root into the package.

    Runs deterministically before mellea-fy so the LLM sees the mirrored
    assets in <package_name>/ when it generates code, reinforcing the
    package-relative path-resolution invariant. Returns the list of
    directory names actually mirrored (for logging).
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    mirrored: list[str] = []
    for asset_dir in _COMPANION_DIRS:
        src = skill_dir / asset_dir
        if src.is_dir():
            dst = package_dir / asset_dir
            shutil.copytree(src, dst, dirs_exist_ok=True)
            mirrored.append(asset_dir)
    return mirrored


# Defaults for the LLM backend and model that compiled skills use at runtime.
# Sourced from .claude/data/runtime_defaults.json with optional CLI overrides.
_RUNTIME_DEFAULTS_PATH = Path(".claude/data/runtime_defaults.json")
_RUNTIME_DEFAULTS_FALLBACK = {"backend": "ollama", "model_id": "granite4.1:8b"}


def resolve_runtime_defaults(
    skill_backend_override: Optional[str], skill_model_override: Optional[str]
) -> tuple[str, str, str]:
    """Pick the backend and model_id to bake into the compiled skill.

    Precedence: CLI overrides win, then the defaults file, then a built-in
    fallback if the file is missing or unreadable. Returns
    (backend, model_id, source) where source records where the values came
    from for the compile log.
    """
    file_backend = _RUNTIME_DEFAULTS_FALLBACK["backend"]
    file_model_id = _RUNTIME_DEFAULTS_FALLBACK["model_id"]
    source = "built-in fallback"
    if _RUNTIME_DEFAULTS_PATH.exists():
        try:
            data = json.loads(_RUNTIME_DEFAULTS_PATH.read_text())
            file_backend = data.get("backend", file_backend)
            file_model_id = data.get("model_id", file_model_id)
            source = f"defaults file ({_RUNTIME_DEFAULTS_PATH})"
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning(
                "Could not read %s (%s). Using built-in fallback values: backend=%r, model_id=%r.",
                _RUNTIME_DEFAULTS_PATH,
                exc,
                file_backend,
                file_model_id,
            )
    if skill_backend_override or skill_model_override:
        source = "command-line override"
    return (
        skill_backend_override or file_backend,
        skill_model_override or file_model_id,
        source,
    )


# Paths under <package_name>/ that the wrapper renders authoritatively from
# emission JSON in LEGACY mode (no `--use-descriptor`). The slash command must
# NOT write or edit these — `--settings` deny rules block the Write/Edit tool
# calls, and the wrapper overwrites with the writer's output post-mellea-fy.
# Add new entries here as each writer is migrated to enforce mode. Glob
# patterns are supported (verified end-to-end in the synthetic deny-rule
# test: `Write(forbidden/**)` blocked nested writes).
_LEGACY_WRAPPER_RENDERED_PATHS: tuple[str, ...] = (
    "config.py",
    "fixtures/**",
)

# Additional paths the wrapper renders in DESCRIPTOR mode (Fix A,
# 2026-05-18). Claude emits `intermediate/descriptor_emission.json` and the
# wrapper's `compile/writer_renderer.py::render_descriptor_to_python`
# materialises these files from the descriptor IR. In descriptor mode the
# wrapper is the authoritative source for both `pipeline.py` and `schemas.py`
# — direct Write/Edit by Claude would be overwritten post-session, wasting
# tokens on a clobbered emission. Deny rules make that contract explicit at
# the tool level so the LLM emits the descriptor IR instead.
_DESCRIPTOR_MODE_ADDITIONAL_PATHS: tuple[str, ...] = (
    "pipeline.py",
    "schemas.py",
)

# Backward-compat alias — legacy-mode default. Existing imports (e.g.
# `build_system_prompt` and any external consumers) keep working with the
# legacy path list; the descriptor-mode union is exposed only via
# `_resolve_wrapper_rendered_paths`.
_WRAPPER_RENDERED_PATHS = _LEGACY_WRAPPER_RENDERED_PATHS


def _resolve_wrapper_rendered_paths(use_descriptor: bool) -> tuple[str, ...]:
    """Return the wrapper-rendered path list for the given compile mode.

    Legacy mode (``use_descriptor=False``) returns the always-rendered set
    (`config.py`, `fixtures/**`). Descriptor mode (``use_descriptor=True``)
    additionally includes `pipeline.py` and `schemas.py`, which the wrapper
    renders post-session from `intermediate/descriptor_emission.json` via
    `writer_renderer.render_descriptor_to_python`.
    """
    if use_descriptor:
        return _LEGACY_WRAPPER_RENDERED_PATHS + _DESCRIPTOR_MODE_ADDITIONAL_PATHS
    return _LEGACY_WRAPPER_RENDERED_PATHS


def write_compile_settings(
    intermediate_dir: Path,
    package_dir: Path,
    *,
    use_descriptor: bool = False,
) -> Path:
    """Write a per-invocation Claude Code --settings file with the deny rules.

    Path-scoped Write/Edit denies prevent the LLM from clobbering files the
    wrapper will render after the slash command exits. acceptEdits permission
    mode does NOT override deny — Anthropic's evaluation order is deny → mode
    → allow (verified end-to-end in the synthetic deny-rule test).

    The deny list is mode-aware: in legacy mode it covers `config.py` and
    `fixtures/**`; in descriptor mode (``use_descriptor=True``) it also
    covers `pipeline.py` and `schemas.py` because the wrapper renders them
    deterministically from `intermediate/descriptor_emission.json`. The
    default (``use_descriptor=False``) preserves the pre-2026-05-18 contract
    for callers that haven't been updated.

    Important: the deny rule paths must match what the LLM actually tries to
    write. Claude Code interprets relative paths against the subprocess cwd
    (the user's shell cwd, typically the repo root). So the deny path must be
    the full path-from-cwd to the package, not just the bare package name.
    Earlier versions used `package_name/<rel>`, which mis-scoped the deny to
    `<cwd>/<package_name>/<rel>` and confused the LLM into creating the
    package at the wrong location.
    """
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the path the LLM's Write tool calls will actually use.
    try:
        rel_pkg = package_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # package_dir is outside cwd — Claude Code accepts absolute paths too.
        rel_pkg = package_dir.resolve()
    deny_rules: list[str] = []
    for rel in _resolve_wrapper_rendered_paths(use_descriptor):
        target = f"{rel_pkg.as_posix()}/{rel}"
        deny_rules.append(f"Write({target})")
        deny_rules.append(f"Edit({target})")
    settings = {
        "permissions": {
            "deny": deny_rules,
        },
    }
    path = intermediate_dir / "_compile_settings.json"
    path.write_text(json.dumps(settings, indent=2))
    return path


def write_runtime_directive(
    intermediate_dir: Path, backend: str, model_id: str, source: str
) -> Path:
    """Save the chosen backend and model_id alongside the package.

    The post-compile lint reads this file to confirm the generated config.py
    actually used these values (and didn't drift from what we instructed).
    """
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "1.0",
        "backend": backend,
        "model_id": model_id,
        "source": source,
    }
    path = intermediate_dir / "runtime_directive.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def build_system_prompt(backend: str, model_id: str, source: str) -> str:
    """Assemble the instruction string passed to the mellea-fy slash command.

    Combines the existing autonomous-run directive with the resolved backend
    and model_id values, plus a notice listing the paths the wrapper renders
    authoritatively from JSON (the LLM cannot Write/Edit them — denied via
    --settings). This tells the LLM to emit the IR JSON instead of trying to
    write the rendered source itself, so the first attempt is correct rather
    than denied-then-retried.
    """
    wrapper_rendered_lines = "\n".join(
        f"  - <package_name>/{p}" for p in _WRAPPER_RENDERED_PATHS
    )
    return (
        "Run the complete 10-step pipeline (Steps 0 through 7) autonomously from start to finish. "
        "Do NOT pause between steps, do NOT ask for user confirmation to proceed, and do NOT stop "
        "after any individual step completes. Invoke each sub-command in sequence and continue "
        "immediately to the next step.\n\n"
        "**Autonomous execution — IMMEDIATE tool invocation at step boundaries.** "
        "After completing each step, do NOT end your turn with a narrative line "
        "(\"Step N complete. Proceeding to step N+1.\"). Instead, IMMEDIATELY invoke "
        "the first tool for the next step in the SAME turn. Narrative may follow "
        "the tool call, but the tool call MUST come first at each step boundary. "
        "The session is considered done only after Step 7 (`step_7_report.json`) "
        "is written.\n\n"
        "Empirical observation: sessions that end with \"Proceeding to...\" narrative "
        "without an immediate follow-up tool call cause the pipeline to halt at the "
        "step boundary. To avoid this, treat every step transition as a single "
        "tool invocation, not a sentence.\n\n"
        "**Step 5 emission discipline — canonical signature verification.** For "
        "every Mellea-stdlib call you emit (m.instruct, req, check, "
        "simple_validate, etc.), you MUST:\n\n"
        "1. Locate the canonical signature in "
        "intermediate/mellea_api_ref.json:.modules.<module>.<symbol>.signature\n"
        "2. Match positional argument count, keyword argument names, and "
        "import paths EXACTLY against the canonical.\n"
        "3. If you cannot find the symbol in mellea_api_ref.json, do NOT "
        "invent it — use a documented alternative or surface the limitation.\n"
        "4. After emitting each auxiliary file (requirements.py, slots.py, "
        "tools.py), enumerate every Mellea call site in that file and "
        "verify against the canonical surface before moving on.\n\n"
        "Empirical observation: the most common cause of post-compile failures "
        "is hallucinated `check(arg)` (1-arg) when the canonical signature is "
        "`check(requirement, ctx)` (2-arg). Either look it up exactly or use "
        "`req()` (the 1-arg factory) instead.\n\n"
        "The Step 5b validation gate exists to catch these — but the cheaper "
        "path is to get it right the first time in Step 5.\n\n"
        "The following paths are rendered by the compile pipeline from the JSON you emit "
        "in <package_name>/intermediate/ — DO NOT write or edit them yourself; the Write "
        "and Edit tools are denied for these paths and the wrapper will render them "
        "deterministically after you exit:\n"
        f"{wrapper_rendered_lines}\n"
        "Emit the corresponding *_emission.json files under <package_name>/intermediate/ "
        "conforming to .claude/schemas/*_emission.schema.json instead.\n\n"
        f"Runtime defaults (source: {source}). The values below MUST appear in "
        "config_emission.json so the wrapper renders them into <package_name>/config.py:\n"
        f"  BACKEND = {backend!r}\n"
        f"  MODEL_ID = {model_id!r}\n"
        "Do not invent alternative values, and do not omit either constant. "
        "The post-compile lint will verify config.py against the recorded values "
        "at <package_name>/intermediate/runtime_directive.json."
    )
