"""Descriptor-emission compile step (Phase 3.A).

Productionised port of ``scripts/llm_descriptor_spike.py``. Given a skill
specification and an introspected Mellea surface, this module:

1. Builds the locked-in Phase 1 prompt — schema doc + filtered surface +
   one hand-authored 1-shot — in a system block that is **byte-stable**
   across calls for the same :class:`EmissionConfig`, so the Anthropic
   prompt cache can amortise the cost.
2. Streams the model response (the SDK refuses non-streaming above
   ~16K output tokens; we run at ``max_tokens=64000`` by default per the
   Phase 1 prompt-design findings).
3. Extracts the ``<reasoning>`` and ``<descriptor>`` tag bodies.
4. Hands the parsed descriptor off to :mod:`mellea_skills_compiler.descriptor`
   (P2.D's validator) and :mod:`mellea_skills_compiler.renderer` (P2.A/B/C's
   renderer / artefact emitters).
5. Returns a structured :class:`EmissionResult` / :class:`CompileResult`
   capturing the full attempt — including failure reason — so the
   repair-loop (P3.B), corpus-compare (P3.C), and ``/mellea-fy`` integration
   (P3.D) callers can introspect every stage.

This module owns **the only** LLM call site in the descriptor-emission
compile flow. Any subsequent call (e.g. the repair loop) is its own,
separate concern.

Credentials: the Anthropic SDK reads ``ANTHROPIC_AUTH_TOKEN`` /
``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` from the environment by
default. We do not import any secret-bearing files. Tests inject a mock
client; production callers pass ``client=None`` and rely on the env.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from importlib import metadata as _md
from importlib.resources import files as _resource_files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import anthropic

    from mellea_skills_compiler.descriptor import ValidationReport
    from mellea_skills_compiler.renderer.core import RenderResult
    from mellea_skills_compiler.renderer.source_map import SourceMap


_LOGGER = logging.getLogger(__name__)


# --- Defaults: bundled prompt resources -------------------------------------
#
# These ship with the wheel under ``src/mellea_skills_compiler/.../data/``
# and are resolved via :mod:`importlib.resources` so they're locatable
# regardless of where the package is installed.

_DEFAULT_SCHEMA_DOC: Traversable = (
    _resource_files("mellea_skills_compiler.descriptor") / "data" / "descriptor-schema-v0.md"
)
_DEFAULT_ONE_SHOT_DESCRIPTOR: Traversable = (
    _resource_files("mellea_skills_compiler.compile") / "data" / "sentry-find-bugs.descriptor.json"
)


_FailureStage = Literal["emission", "validation", "render", "smoke"]


_INTERMEDIATE_ARTEFACT_NAMES: tuple[str, ...] = (
    "classification",
    "inventory",
    "element_mapping",
    "element_mapping_amendments",
    "dependency_plan",
    "mellea_api_ref",
    "mellea_doc_index",
    "expected_signature",
)


@dataclass(frozen=True)
class EmissionConfig:
    """Prompt + LLM configuration.

    Defaults are the Phase 1 locked-in values
    (see ``melleafy-handoff/kickoff/spike-outputs/phase1-prompt-design-findings.md``):

    * ``model = "aws/claude-sonnet-4-6"`` (routed via the IBM-internal LiteLLM
      proxy in dev; overridable for other deployments).
    * ``max_tokens = 64000`` — the verified-passing ceiling. No observed run
      ever produced more than ~27K output tokens; the higher ceiling is
      headroom for the long tail.
    * ``relevant_modules`` — the 9-module surface filter that keeps the
      cached system block tractable.
    * ``one_shot_descriptor_path`` / ``schema_doc_path`` — when ``None``,
      we use the bundled defaults shipped with the repo.
    * ``intermediate_artefacts`` — when set (Phase 3.5.A), inline each named
      artefact JSON into the system prompt under a stable
      ``<artefact name="...">`` delimiter so the LLM can ground descriptor
      emission in the full Step 0–2.5 analytical output. **Byte-stability
      invariant**: when ``None`` (the default), the system prompt is
      byte-identical to pre-Phase-3.5.A output, so existing one-shot callers
      (``batch_descriptor_test.py``, ``corpus_compare.py``) are unaffected.
    """

    model: str = "aws/claude-sonnet-4-6"
    max_tokens: int = 64000
    relevant_modules: tuple[str, ...] = (
        "mellea.stdlib.session",
        "mellea.stdlib.requirements.requirement",
        "mellea.stdlib.requirements.md",
        "mellea.backends.openai",
        "mellea.backends.model_options",
        "mellea.backends.model_ids",
        "mellea.core.requirement",
        "mellea.core.sampling",
        "mellea.stdlib.sampling.base",
    )
    one_shot_descriptor_path: Path | None = None
    schema_doc_path: Path | None = None
    descriptor_schema_version: str = "0.2"
    intermediate_artefacts: dict[str, Any] | None = None

    def resolved_one_shot_path(self) -> Path | Traversable:
        return self.one_shot_descriptor_path or _DEFAULT_ONE_SHOT_DESCRIPTOR

    def resolved_schema_doc_path(self) -> Path | Traversable:
        return self.schema_doc_path or _DEFAULT_SCHEMA_DOC


@dataclass
class EmissionResult:
    """Returned by :func:`emit_descriptor`.

    Captures everything observable about a single LLM emission attempt so the
    repair loop, corpus-compare, and the ``/mellea-fy`` integration can
    introspect a failed attempt without re-running it.

    ``descriptor`` is the parsed Python ``dict`` (None on parse failure).
    ``descriptor_text`` is the raw JSON text inside ``<descriptor>...</descriptor>``;
    we keep it even on parse success so callers can diff the bytes if needed.
    ``usage`` carries the four token-bucket counts we report on; cache fields
    default to 0 when the SDK does not surface them.
    """

    raw_output: str
    reasoning_text: str | None
    descriptor: dict[str, Any] | None
    descriptor_text: str | None
    parse_error: str | None
    usage: dict[str, int]
    latency_seconds: float
    model: str


@dataclass
class CompileResult:
    """End-to-end result: emission + validation + render + smoke."""

    emission: EmissionResult
    validation: "ValidationReport | None"
    pipeline_py: str | None
    schemas_py: str | None
    init_py: str | None
    fixtures_py: str | None
    melleafy_json: str | None
    setup_md: str | None
    readme_md: str | None
    source_map: "SourceMap | None"
    smoke_ok: bool
    failure_stage: _FailureStage | None
    failure_reason: str | None


# --- Prompt construction ----------------------------------------------------


def _filter_surface(surface: dict[str, Any], relevant_modules: tuple[str, ...]) -> dict[str, Any]:
    """Restrict the surface JSON to ``relevant_modules`` for prompt-cache stability.

    The full ``surface_<version>.json`` is multi-megabyte; the prompt cache
    only stays warm if the cached block is byte-stable. We always serialise
    the filter with ``sort_keys=True`` (see :func:`_serialise_filtered_surface`)
    so two callers with the same :class:`EmissionConfig` produce identical
    system blocks regardless of dict iteration order.
    """
    modules = surface.get("modules", {}) if isinstance(surface, dict) else {}
    selected = {m: modules[m] for m in relevant_modules if m in modules}
    return {
        "mellea_version": surface.get("mellea_version"),
        "python_version": surface.get("python_version"),
        "modules": selected,
    }


def _serialise_filtered_surface(
    surface: dict[str, Any], relevant_modules: tuple[str, ...]
) -> str:
    """Byte-stable JSON for the surface block of the system prompt."""
    subset = _filter_surface(surface, relevant_modules)
    return json.dumps(subset, indent=2, sort_keys=True)


def _serialise_artefact(name: str, body: Any) -> str:
    """Serialise an intermediate artefact for inclusion in the system prompt.

    Returns a single ``<artefact name="..."> ... </artefact>`` block whose
    payload is JSON (``sort_keys=True``, ``indent=2``) when ``body`` is a
    Python dict / list, and otherwise the raw ``str(body)``. Tag delimiters
    are unlikely to collide with descriptor schema-doc / one-shot content
    (which is fenced markdown / JSON).
    """
    if isinstance(body, (dict, list)):
        payload = json.dumps(body, indent=2, sort_keys=True)
    elif body is None:
        payload = "null"
    else:
        payload = str(body)
    return f'<artefact name="{name}">\n{payload}\n</artefact>'


_EXPECTED_SIGNATURE_HARD_CONSTRAINT_PREAMBLE = (
    "\n\n### HARD CONSTRAINT — expected_signature lock\n\n"
    "The artefact below is the canonical I/O signature derived deterministically "
    "from `element_mapping.json` + `classification.json` + `inventory.json` BEFORE "
    "descriptor emission. **Step 4 fixtures were already generated against it.** "
    "Your descriptor's `inputs` and `outputs` MUST match this expected signature "
    "exactly:\n\n"
    "- Every input name MUST appear with the matching type.\n"
    "- No extra inputs may be introduced; no required inputs may be dropped.\n"
    "- Every output name and type MUST match.\n"
    "- Every schema referenced by `inputs`/`outputs` MUST have these exact field "
    "names and types under `descriptor.schemas`.\n"
    "- The function name is `run_pipeline` (Rule 3-2, never overridable).\n\n"
    "Violating this constraint fires `R-SEM-SIGNATURE-MATCH` and triggers the repair "
    "loop. Match it on the first emission."
)


def _render_intermediate_artefacts_block(
    artefacts: dict[str, Any] | None,
) -> str:
    """Render the optional intermediate-artefacts section of the system prompt.

    Returns ``""`` when no artefacts are supplied — preserving byte-stability
    of the system prompt for one-shot callers. When artefacts are supplied,
    emits a labelled section in the canonical 8-artefact order
    (``_INTERMEDIATE_ARTEFACT_NAMES``); unknown extra keys are appended after
    the known ones, sorted alphabetically, so the block is deterministic
    regardless of caller iteration order.

    P3.5.D: when ``expected_signature`` is present in ``artefacts``, it is
    preceded by an explicit HARD CONSTRAINT preamble emphasising that the
    descriptor's ``inputs``/``outputs`` must match the signature exactly.
    The artefact's ``<artefact name="expected_signature">`` delimiter is
    preserved so callers that scan for the delimiter (and the byte-stability
    test for absent artefacts) continue to work.
    """
    if not artefacts:
        return ""
    ordered: list[str] = []
    known = set(_INTERMEDIATE_ARTEFACT_NAMES)
    for name in _INTERMEDIATE_ARTEFACT_NAMES:
        if name in artefacts:
            piece = _serialise_artefact(name, artefacts[name])
            if name == "expected_signature":
                piece = _EXPECTED_SIGNATURE_HARD_CONSTRAINT_PREAMBLE + "\n\n" + piece
            ordered.append(piece)
    extras = sorted(k for k in artefacts.keys() if k not in known)
    for name in extras:
        ordered.append(_serialise_artefact(name, artefacts[name]))
    if not ordered:
        return ""
    return (
        "\n\n## Intermediate analytical artefacts (Steps 0–2.5 outputs)\n\n"
        "These are the deterministic outputs of the /mellea-fy workflow's Steps 0 through "
        "2.5 — classification, inventory, element mapping (and Step 2.5d amendments), "
        "dependency plan, the introspected Mellea API reference, and the doc-index "
        "grounding. When present, treat them as authoritative: every input/output type, "
        "every dependency disposition, every symbol mapping has already been decided by "
        "the analytical pipeline. Your descriptor must reflect these decisions faithfully "
        "rather than re-derive them.\n\n"
        "If `expected_signature` is present, the `run_pipeline` signature is locked — "
        "match it exactly. If it is absent, derive the signature from `inventory`, "
        "`element_mapping`, and `classification`.\n\n"
        + "\n\n".join(ordered)
    )


def build_system_prompt(config: EmissionConfig, surface: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the cached system-prompt block.

    ONE ``cache_control: ephemeral`` breakpoint on the entire block, exactly
    as Phase 0.3 / Phase 1 verified. Content order: schema doc → filtered
    surface JSON → 1-shot descriptor → (optional) intermediate artefacts.
    The wrapping prose matches the spike so we keep the empirical evidence
    intact.

    **Byte-stability invariant** (Phase 3.5.A): when
    ``config.intermediate_artefacts is None`` (or empty), the system prompt is
    byte-identical to its pre-Phase-3.5.A form. Existing one-shot callers
    (``batch_descriptor_test.py``, ``corpus_compare.py``) pass no artefacts
    and continue to work unchanged.
    """
    schema_md = config.resolved_schema_doc_path().read_text(encoding="utf-8")
    one_shot_path = config.resolved_one_shot_path()
    one_shot = one_shot_path.read_text(encoding="utf-8")
    surface_text = _serialise_filtered_surface(surface, config.relevant_modules)
    artefacts_block = _render_intermediate_artefacts_block(config.intermediate_artefacts)
    text = (
        "You are emitting a *descriptor* for the GraniteClawHub Mellea skills compiler. "
        "A descriptor is a typed, JSON-only call graph that another program will render into a "
        "Mellea pipeline package. Read the schema, the introspected Mellea call vocabulary, and "
        "the worked example below carefully. Your reasoning is allowed to be long; your final "
        "descriptor must be one JSON object and nothing else.\n\n"
        "## Descriptor schema (v0 strawman)\n\n"
        f"{schema_md}\n\n"
        "## Introspected Mellea 0.5 surface — filtered to symbols you are most likely to need\n\n"
        "Use these `symbol` paths and `args` keys verbatim. Do not invent symbols not in this list.\n\n"
        "```json\n"
        f"{surface_text}\n"
        "```\n\n"
        "## One-shot ground-truth example: `sentry-find-bugs.descriptor.json`\n\n"
        "This descriptor was hand-authored against the same spec you will be given. Treat it as a "
        "reference for style, depth, and shape — your output should be a peer of it, not a copy of it.\n\n"
        "```json\n"
        f"{one_shot}\n"
        "```"
        f"{artefacts_block}"
    )
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_user_message(spec_text: str) -> str:
    """Build the per-call user message (spec + reasoning-then-structure instructions)."""
    return (
        "Generate the descriptor for the skill whose spec.md is below.\n\n"
        "## Reasoning step (free-form, plan §4.4)\n\n"
        "Before you emit the descriptor, think step-by-step inside `<reasoning>...</reasoning>` tags:\n"
        "- What input does the skill take? What output does it produce?\n"
        "- What Pydantic schemas do you need to define?\n"
        "- How many phases / pipeline nodes? Which use `map` over a list? Which are sequential?\n"
        "- Which Mellea symbols (from the surface above) will each call node use?\n"
        "- Any v0 schema gaps you have to work around (tool allowlists, model overrides, filter ops, etc.)?\n\n"
        "## Final emission\n\n"
        "After reasoning, emit the descriptor as one JSON object inside `<descriptor>...</descriptor>` tags. "
        "No prose between the tags. No code fences. Just the raw JSON. Validate against the schema before "
        "emitting — every `symbol` must exist in the surface above; every `ref` must point to a declared "
        "state entry or earlier pipeline node id; every `schema_ref` must point to a declared schema.\n\n"
        "## Skill spec\n\n"
        "```markdown\n"
        f"{spec_text}\n"
        "```"
    )


# --- Tag extraction ---------------------------------------------------------


def _extract_tag_body(raw: str, tag: str) -> str | None:
    """Return the body inside the first ``<tag>...</tag>`` pair, or ``None``."""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = raw.find(open_tag)
    if start == -1:
        return None
    end = raw.find(close_tag, start + len(open_tag))
    if end == -1:
        return None
    return raw[start + len(open_tag) : end].strip()


def extract_descriptor_text(raw: str) -> tuple[str | None, str | None]:
    """Pull the JSON descriptor body from the model output.

    Returns ``(body, None)`` on success, ``(None, error_message)`` on
    failure (missing tags). Tolerates an accidental ```` ``` ```` code fence
    inside the tags (the spike has had to defend against this).
    """
    body = _extract_tag_body(raw, "descriptor")
    if body is None:
        return None, "missing <descriptor>...</descriptor> tags"
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body[3:]
        if body.rstrip().endswith("```"):
            body = body.rsplit("```", 1)[0]
    return body.strip(), None


def extract_reasoning(raw: str) -> str | None:
    """Pull ``<reasoning>...</reasoning>`` body if present, else ``None``."""
    return _extract_tag_body(raw, "reasoning")


# --- LLM invocation ---------------------------------------------------------


def _raw_text_from_response(response: Any) -> str:
    """Concatenate the text blocks of a non-streaming or final-message object."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text" and getattr(block, "text", None) is not None:
            chunks.append(block.text)
    return "".join(chunks)


def _usage_to_dict(usage: Any) -> dict[str, int]:
    """Project an Anthropic ``Usage`` object into a plain dict.

    Returns the four token-bucket counts we surface to callers. Missing
    fields default to ``0`` (older SDKs / proxies that don't surface cache
    counters won't 500 us).
    """
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
    }


def _stream_final_message(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    system_blocks: list[dict[str, Any]],
    user_text: str,
) -> Any:
    """Invoke the streaming endpoint and return the final ``Message`` object.

    Streaming is required: the Anthropic SDK refuses non-streaming responses
    above ~16K output tokens (Phase 1 finding). The context-manager form lets
    the SDK clean up the connection if our caller raises.
    """
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        return stream.get_final_message()


def emit_descriptor(
    spec_text: str,
    surface: dict[str, Any],
    config: EmissionConfig | None = None,
    *,
    client: "anthropic.Anthropic | None" = None,
) -> EmissionResult:
    """Run a single descriptor-emission attempt against the LLM.

    Args:
        spec_text: The skill specification as Markdown text (the contents
            of ``spec.md`` / ``SKILL.md``).
        surface: The introspected Mellea surface JSON (as returned by
            :mod:`mellea_skills_compiler.mellea_surface` / loaded from
            ``surface_<version>.json``). Filtered internally to
            ``config.relevant_modules`` for the cached prompt.
        config: Prompt + model configuration. Defaults to the Phase 1
            locked-in values.
        client: Optional Anthropic client. If ``None``, a fresh
            ``anthropic.Anthropic()`` is constructed (reads credentials from
            ``ANTHROPIC_AUTH_TOKEN``/``ANTHROPIC_BASE_URL`` env vars). Tests
            inject a mock here.

    Returns:
        An :class:`EmissionResult` capturing the LLM's raw output, the
        extracted reasoning and descriptor (when present), and the
        latency/usage telemetry.
    """
    cfg = config or EmissionConfig()
    if client is None:
        import anthropic  # local import keeps test-time path free of SDK requirement

        client = anthropic.Anthropic()

    system_blocks = build_system_prompt(cfg, surface)
    user_text = build_user_message(spec_text)

    started_at = time.time()
    response = _stream_final_message(
        client,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system_blocks=system_blocks,
        user_text=user_text,
    )
    latency = time.time() - started_at

    raw_text = _raw_text_from_response(response)
    reasoning = extract_reasoning(raw_text)
    descriptor_text, extract_err = extract_descriptor_text(raw_text)

    descriptor: dict[str, Any] | None = None
    parse_error: str | None = extract_err
    if descriptor_text is not None and extract_err is None:
        try:
            descriptor = json.loads(descriptor_text)
        except json.JSONDecodeError as exc:
            parse_error = f"JSON parse failed: {exc}"
            descriptor = None

    return EmissionResult(
        raw_output=raw_text,
        reasoning_text=reasoning,
        descriptor=descriptor,
        descriptor_text=descriptor_text,
        parse_error=parse_error,
        usage=_usage_to_dict(getattr(response, "usage", None)),
        latency_seconds=round(latency, 3),
        model=cfg.model,
    )


# --- Render + smoke pieces --------------------------------------------------


def _detect_mellea_version() -> str:
    try:
        return _md.version("mellea")
    except Exception:  # noqa: BLE001 - resilient stamping
        return "unknown"


def _render_full_package(
    descriptor: dict[str, Any],
    surface: dict[str, Any],
    *,
    skill_root: Path | None = None,
) -> tuple[
    str,  # pipeline_py
    str,  # schemas_py
    str,  # init_py
    str,  # fixtures_py
    str,  # melleafy_json
    str,  # setup_md
    str,  # readme_md
    "SourceMap",
    "RenderResult",
]:
    """Render every file the ``render-package`` CLI emits, in-memory.

    We replicate that CLI's exact recipe so the descriptor-emission compile
    flow's output layout is byte-identical to ``python -m
    mellea_skills_compiler.renderer render-package`` for the same inputs.

    The trailing :class:`RenderResult` carries the v0.3 file-copy /
    package-data / source-elements artefacts that :func:`_write_package_to`
    consumes when materialising the package directory.
    """
    from mellea_skills_compiler.renderer import render_descriptor
    from mellea_skills_compiler.renderer.artefacts import (
        render_fixtures,
        render_init,
        render_melleafy_json,
        render_readme_md,
        render_setup_md,
    )
    from mellea_skills_compiler.renderer.schemas import render_schemas

    render_result = render_descriptor(descriptor, surface, skill_root=skill_root)
    pipeline_py = render_result.pipeline_py
    schemas_py = render_schemas(descriptor.get("schemas", {}) or {})
    init_py = render_init(descriptor)
    # The fixtures helper renders source relative to a sibling ``fixtures/``
    # directory; we pass a Path whose ``.name`` is "fixtures" to match the
    # CLI's layout.
    fixtures_py = render_fixtures(descriptor, Path("fixtures"))
    melleafy_json = render_melleafy_json(
        descriptor,
        mellea_version=_detect_mellea_version(),
        renderer_version="p2-0.1",
        compiled_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )
    setup_md = render_setup_md(descriptor)
    readme_md = render_readme_md(descriptor)
    return (
        pipeline_py,
        schemas_py,
        init_py,
        fixtures_py,
        melleafy_json,
        setup_md,
        readme_md,
        render_result.source_map,
        render_result,
    )


def _smoke_import(
    descriptor: dict[str, Any],
    pipeline_py: str,
    schemas_py: str,
    init_py: str,
    fixtures_py: str,
) -> tuple[bool, str | None]:
    """Try to ``py_compile`` and import the rendered package in a temp dir.

    Mirrors the spike's smoke check: write the package out, ``importlib``
    it under a unique throwaway module name, assert ``run_pipeline`` exists.
    We do **not** invoke the pipeline — that's a fixture concern.

    The temp dir is removed on success or failure. Returns ``(True, None)``
    on success, ``(False, error_text)`` on failure (including any exception
    traceback from the import).
    """
    skill = descriptor.get("skill") or {}
    raw_name = (skill.get("name") or "compiled_skill").replace("-", "_")
    # Compose a unique package name so re-running tests doesn't trigger
    # importlib cache collisions in long-lived runners.
    pkg_name = f"_msc_descriptor_pkg_{raw_name}_{int(time.time() * 1000)}"

    tmpdir = Path(tempfile.mkdtemp(prefix="msc-descriptor-smoke-"))
    pkg_dir = tmpdir / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = pkg_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    try:
        (pkg_dir / "pipeline.py").write_text(pipeline_py, encoding="utf-8")
        (pkg_dir / "schemas.py").write_text(schemas_py, encoding="utf-8")
        (pkg_dir / "__init__.py").write_text(init_py, encoding="utf-8")
        (pkg_dir / "fixtures.py").write_text(fixtures_py, encoding="utf-8")
        (fixtures_dir / "__init__.py").write_text("", encoding="utf-8")

        # Try compiling first — gives us a clean syntax-error message before
        # we touch importlib.
        import py_compile

        for filename in ("pipeline.py", "schemas.py"):
            try:
                py_compile.compile(str(pkg_dir / filename), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, f"py_compile {filename} failed: {exc}"

        sys.path.insert(0, str(tmpdir))
        # The rendered pipeline reads MODEL_ID / OPENAI_API_KEY at import time
        # (the OpenAIBackend constructor pulls them from os.environ). We stub
        # them with fake values so the smoke check exercises *import-ability*,
        # not LLM-reachability. Restored in the finally block.
        env_overrides = {
            "MODEL_ID": "smoke-fake-model",
            "OPENAI_API_KEY": "sk-smoke-fake",
        }
        prior_env = {k: os.environ.get(k) for k in env_overrides}
        try:
            for k, v in env_overrides.items():
                os.environ[k] = v
            # Drop any stale cache entry for this package name (defensive).
            for mod_name in list(sys.modules):
                if mod_name == pkg_name or mod_name.startswith(f"{pkg_name}."):
                    sys.modules.pop(mod_name, None)
            mod = importlib.import_module(f"{pkg_name}.pipeline")
            if not hasattr(mod, "run_pipeline"):
                return False, "pipeline module has no `run_pipeline` attribute"
            return True, None
        except Exception as exc:  # noqa: BLE001 - we want every failure
            tb = traceback.format_exc()
            return False, f"import failed: {type(exc).__name__}: {exc}\n{tb}"
        finally:
            try:
                sys.path.remove(str(tmpdir))
            except ValueError:
                pass
            for mod_name in list(sys.modules):
                if mod_name == pkg_name or mod_name.startswith(f"{pkg_name}."):
                    sys.modules.pop(mod_name, None)
            for k, prior in prior_env.items():
                if prior is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prior
    finally:
        # Best-effort cleanup of the temp dir.
        try:
            for child in sorted(tmpdir.rglob("*"), reverse=True):
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                except OSError:
                    pass
            tmpdir.rmdir()
        except OSError:
            pass


def _write_package_to(
    out_dir: Path,
    *,
    pipeline_py: str,
    schemas_py: str,
    init_py: str,
    fixtures_py: str,
    melleafy_json: str,
    setup_md: str,
    readme_md: str,
    source_map: "SourceMap",
    render_result: "RenderResult | None" = None,
    skill_root: Path | None = None,
) -> None:
    """Persist the rendered package to ``out_dir`` (created if missing).

    When ``render_result`` carries v0.3 artefacts (``file_copies`` /
    ``package_data_dirs`` / ``deferred_binaries`` / ``source_elements_map``),
    this function also:

    * Executes every :class:`~mellea_skills_compiler.renderer.dependencies.FileCopyOp`
      (resolving ``source`` relative to ``skill_root``) into the package
      directory; INFO-logs each copy.
    * Emits WARNING-level log messages for ``deferred_binaries`` (plan §11.5
      defers them to v0.4 — they are NOT copied).
    * Creates or updates a ``pyproject.toml`` next to the package's
      ``__init__.py`` with a ``[tool.setuptools.package-data]`` block listing
      the new dirs, via
      :func:`~mellea_skills_compiler.renderer.bundled.update_pyproject_package_data`.
    * Writes ``intermediate/source_elements_map.json`` for Step 6's
      ``mapping_report.md`` builder.

    When ``render_result`` is ``None`` or carries empty v0.3 artefacts
    (v0.1/v0.2 descriptors), output is byte-identical to the pre-v0.3 writer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = out_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    (out_dir / "pipeline.py").write_text(pipeline_py, encoding="utf-8")
    (out_dir / "schemas.py").write_text(schemas_py, encoding="utf-8")
    (out_dir / "__init__.py").write_text(init_py, encoding="utf-8")
    (out_dir / "fixtures.py").write_text(fixtures_py, encoding="utf-8")
    (fixtures_dir / "__init__.py").write_text("", encoding="utf-8")
    (out_dir / "melleafy.json").write_text(melleafy_json, encoding="utf-8")
    (out_dir / "SETUP.md").write_text(setup_md, encoding="utf-8")
    (out_dir / "README.md").write_text(readme_md, encoding="utf-8")
    (out_dir / "source_map.json").write_text(
        json.dumps(source_map.to_json(), indent=2) + "\n",
        encoding="utf-8",
    )

    if render_result is None:
        return
    _apply_v03_artefacts(
        out_dir,
        render_result=render_result,
        skill_root=skill_root,
    )


def _apply_v03_artefacts(
    out_dir: Path,
    *,
    render_result: "RenderResult",
    skill_root: Path | None,
) -> None:
    """Apply v0.3 file-copy / package-data / source-elements artefacts.

    Gated: every step is a no-op when the corresponding ``RenderResult`` field
    is empty, so v0.1 / v0.2 descriptors hit this function but produce no
    side effects beyond the legacy writes.

    Rule OUT-6 dedup: legacy ``mirror_companion_dirs`` (in
    ``compile/claude_directives.py``) already copies ``scripts/`` /
    ``references/`` / ``assets/`` from the skill root into the package at
    Step 3a-pre, and the legacy ``pyproject.toml`` template already declares
    those three under ``[tool.setuptools.package-data]``. We filter those
    paths out before executing file copies or merging pyproject entries so
    we don't redo what the legacy path already did.
    """
    from mellea_skills_compiler.compile.claude_directives import _COMPANION_DIRS

    out6_dirs = set(_COMPANION_DIRS)

    file_copies = list(getattr(render_result, "file_copies", []) or [])
    package_data_dirs = set(getattr(render_result, "package_data_dirs", set()) or set())
    deferred_binaries = list(getattr(render_result, "deferred_binaries", []) or [])
    source_elements_map = dict(
        getattr(render_result, "source_elements_map", {}) or {}
    )

    # 1. Execute scheduled file copies for paths NOT covered by Rule OUT-6.
    #    Sources are interpreted relative to skill_root; when skill_root is
    #    unset we have no anchor and the copy raises (matches the renderer's
    #    own "skill_root may be None in headless contexts" contract).
    for op in file_copies:
        dest_rel = getattr(op, "dest", "") or ""
        top = dest_rel.split("/", 1)[0] if dest_rel else ""
        if top in out6_dirs:
            _LOGGER.debug(
                "Skipping v0.3 file-copy for %r — already mirrored by "
                "mirror_companion_dirs at Step 3a-pre (Rule OUT-6)",
                dest_rel,
            )
            continue
        _execute_file_copy_op(op, out_dir=out_dir, skill_root=skill_root)

    # 2. Surface the deferred binaries (renderer already warned at render
    #    time; we re-surface at write time so compile-log readers see them).
    for binary_path in deferred_binaries:
        _LOGGER.warning(
            "Deferred binary resource %r (plan §11.5: bundling deferred to "
            "v0.4); NOT copied into %s",
            binary_path,
            out_dir,
        )

    # 3. Apply pyproject.toml package-data merge ONLY for entries beyond the
    #    Rule OUT-6 trio (which the legacy pyproject template already lists).
    extra_package_data_dirs = {
        d for d in package_data_dirs if d.split("/", 1)[0] not in out6_dirs
    }
    if extra_package_data_dirs:
        _write_pyproject_package_data(out_dir, extra_package_data_dirs)

    # 4. Source-elements map → intermediate artefact for Step 6.
    if source_elements_map:
        intermediate_dir = out_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        (intermediate_dir / "source_elements_map.json").write_text(
            json.dumps(source_elements_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _execute_file_copy_op(
    op: Any,
    *,
    out_dir: Path,
    skill_root: Path | None,
) -> None:
    """Resolve a single :class:`FileCopyOp` and perform the copy.

    ``op`` is duck-typed (the renderer ships
    :class:`mellea_skills_compiler.renderer.dependencies.FileCopyOp`; we
    accept anything with ``source`` / ``dest`` / ``is_directory`` attributes).

    Missing source paths (skill_root unset, or path doesn't exist under
    skill_root) raise :class:`FileNotFoundError` with a clear message. The
    caller decides whether to surface as a hard failure or swallow.
    """
    source_rel = getattr(op, "source", None)
    dest_rel = getattr(op, "dest", None)
    is_directory = bool(getattr(op, "is_directory", False))
    if not source_rel or not dest_rel:
        raise ValueError(
            f"file copy op missing source/dest: source={source_rel!r}, "
            f"dest={dest_rel!r}"
        )
    dest_path = out_dir / dest_rel
    if skill_root is None:
        raise FileNotFoundError(
            f"cannot resolve bundled-resource source {source_rel!r}: "
            "skill_root not provided to the package writer (set "
            "compile_via_descriptor(skill_root=...) or pass to "
            "_write_package_to(skill_root=...))"
        )
    source_path = skill_root / source_rel
    if not source_path.exists():
        raise FileNotFoundError(
            f"bundled resource source {str(source_path)!r} not found "
            f"(skill_root={skill_root}, source={source_rel!r})"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if is_directory or source_path.is_dir():
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        _LOGGER.info(
            "Copied bundled directory %s -> %s", source_path, dest_path
        )
    else:
        shutil.copy(source_path, dest_path)
        _LOGGER.info(
            "Copied bundled file %s -> %s", source_path, dest_path
        )


def _write_pyproject_package_data(
    out_dir: Path, package_data_dirs: set[str]
) -> None:
    """Create or merge ``out_dir/pyproject.toml`` with a package-data block.

    The destination filename is the package directory name (e.g.
    ``my_skill_mellea``); that matches setuptools' convention for declaring
    package-data scoped to one package.

    If a ``pyproject.toml`` already lives at ``out_dir/pyproject.toml`` (a
    user-supplied one), we merge into it preserving existing content via
    :func:`update_pyproject_package_data`. Otherwise we synthesise a minimal
    one with just the package-data block.
    """
    from mellea_skills_compiler.renderer.bundled import (
        update_pyproject_package_data,
    )

    pyproject_path = out_dir / "pyproject.toml"
    package_name = out_dir.name or "package"
    if pyproject_path.is_file():
        existing = pyproject_path.read_text(encoding="utf-8")
    else:
        # Minimal stub — keep it permissive so `update_pyproject_package_data`
        # has something to work against, and so the resulting file is a valid
        # TOML document.
        existing = ""
    merged = update_pyproject_package_data(existing, package_name, package_data_dirs)
    pyproject_path.write_text(merged, encoding="utf-8")
    _LOGGER.info(
        "Wrote pyproject.toml package-data entries for %s: %s",
        package_name,
        sorted(package_data_dirs),
    )


# --- Top-level pipeline -----------------------------------------------------


def compile_via_descriptor(
    spec_text: str,
    surface: dict[str, Any],
    config: EmissionConfig | None = None,
    *,
    write_to: Path | None = None,
    client: "anthropic.Anthropic | None" = None,
    skill_root: Path | None = None,
) -> CompileResult:
    """End-to-end compile: emission → validation → render → smoke.

    The result's ``failure_stage`` field tells the caller where the pipeline
    stopped:

    * ``"emission"`` — parse error (missing tags / malformed JSON) or
      streaming failure.
    * ``"validation"`` — P2.D's validator rejected the descriptor.
    * ``"render"`` — the renderer (or one of P2.C's artefact emitters)
      raised :class:`mellea_skills_compiler.renderer.RendererError`.
    * ``"smoke"`` — render succeeded but the rendered package failed to
      ``py_compile`` / import.
    * ``None`` — full success.

    On full success (``failure_stage is None``), if ``write_to`` is provided
    the package is also persisted there (creating the directory if needed).
    """
    from mellea_skills_compiler.descriptor import validate
    from mellea_skills_compiler.renderer import RendererError

    cfg = config or EmissionConfig()

    emission = emit_descriptor(spec_text, surface, cfg, client=client)
    if emission.descriptor is None:
        return CompileResult(
            emission=emission,
            validation=None,
            pipeline_py=None,
            schemas_py=None,
            init_py=None,
            fixtures_py=None,
            melleafy_json=None,
            setup_md=None,
            readme_md=None,
            source_map=None,
            smoke_ok=False,
            failure_stage="emission",
            failure_reason=emission.parse_error or "no descriptor parsed from LLM output",
        )

    # Validate against the schema version the descriptor declares for itself.
    # Until the bundled 1-shot example migrates to v0.2 the LLM mirrors v0.1; this
    # makes the validator track that without forcing every caller to set the
    # config default in lockstep.
    declared_version = (
        (emission.descriptor.get("descriptor_version") if emission.descriptor else None)
        or cfg.descriptor_schema_version
    )
    # P3.5.D: thread expected_signature (when present in the intermediate
    # artefacts) through to the validator so R-SEM-SIGNATURE-MATCH can fire.
    expected_signature_for_validate: dict[str, Any] | None = None
    if cfg.intermediate_artefacts is not None:
        es = cfg.intermediate_artefacts.get("expected_signature")
        if isinstance(es, dict):
            expected_signature_for_validate = es
    validation = validate(
        emission.descriptor,
        schema_version=declared_version,
        surface=surface,
        expected_signature=expected_signature_for_validate,
    )
    if not validation.valid:
        first_err = validation.errors[0] if validation.errors else None
        reason = (
            f"{first_err.rule} at {first_err.path or '<root>'}: {first_err.message}"
            if first_err is not None
            else "validation failed (no error detail surfaced)"
        )
        return CompileResult(
            emission=emission,
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
            render_result,
        ) = _render_full_package(emission.descriptor, surface, skill_root=skill_root)
    except RendererError as exc:
        return CompileResult(
            emission=emission,
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

    smoke_ok, smoke_error = _smoke_import(
        emission.descriptor, pipeline_py, schemas_py, init_py, fixtures_py
    )
    if not smoke_ok:
        return CompileResult(
            emission=emission,
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
            failure_reason=smoke_error or "smoke import failed (no detail)",
        )

    if write_to is not None:
        _write_package_to(
            write_to,
            pipeline_py=pipeline_py,
            schemas_py=schemas_py,
            init_py=init_py,
            fixtures_py=fixtures_py,
            melleafy_json=melleafy_json,
            setup_md=setup_md,
            readme_md=readme_md,
            source_map=source_map,
            render_result=render_result,
            skill_root=skill_root,
        )

    return CompileResult(
        emission=emission,
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


__all__ = [
    "CompileResult",
    "EmissionConfig",
    "EmissionResult",
    "build_system_prompt",
    "build_user_message",
    "compile_via_descriptor",
    "emit_descriptor",
    "extract_descriptor_text",
    "extract_reasoning",
    "_INTERMEDIATE_ARTEFACT_NAMES",
]
