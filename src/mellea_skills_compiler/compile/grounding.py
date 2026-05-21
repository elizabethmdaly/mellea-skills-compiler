"""Deterministic grounding artifact generation for the mellea-fy pipeline.

This module produces `mellea_api_ref.json` and `mellea_doc_index.json` in the
compile pipeline's intermediate directory before the slash command runs. The
slash command itself runs with `--allowed-tools Read,Write,Edit,Glob` and so
cannot introspect the installed `mellea` package via Python or fetch
`https://docs.mellea.ai/`.
Doing it here means Steps 2.5e and 2.5f of the slash command have real
grounding data to consume rather than silently degrading to static fallbacks.

Both functions are idempotent and cache results under
`~/.cache/mellea-skills-compiler/`. The api_ref cache is keyed by the installed
mellea version; the doc_index cache uses a configurable TTL (default 24h) with
a stale-cache fallback if the network is unreachable.
"""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import pkgutil
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mellea_skills_compiler.toolkit.logging import configure_logger


LOGGER = configure_logger()

CACHE_DIR = Path.home() / ".cache" / "mellea-skills-compiler"

# Modules always introspected, regardless of which skill modality is being
# compiled. These cover the universal surface every generated skill uses
# (session, requirements, sampling, message types, model options, generative
# slots). Modality-specific additions live in `_MODALITY_MODULES` below and
# are appended at compile time based on `classification.json:modality`.
#
# Why CORE_MODULES is the de-facto contract:
# The original design routed per-skill additions through
# `dependency_plan.json:plan[*].target`, expecting the LLM to declare its
# Mellea imports there. In practice that field holds OUTPUT paths
# (`config.py:SKILL_NAME`, `pipeline.py:run_pipeline`) — it was always
# populated with the wrong shape, so dynamic expansion silently returned
# an empty set. Empirically verified across three real compiles. The
# dependency-plan path was removed; modality-driven expansion replaces it.
CORE_MODULES = {
    "mellea.backends.model_options",
    "mellea.stdlib.components.chat",
    "mellea.stdlib.components.docs.document",
    "mellea.stdlib.components.genstub",
    "mellea.stdlib.components.instruction",
    "mellea.stdlib.context",
    "mellea.stdlib.functional",
    "mellea.stdlib.requirements",
    "mellea.stdlib.sampling",
    "mellea.stdlib.session",
}


# Per-modality module additions, keyed by `classification.json:modality`.
# An unknown modality contributes no extra modules (no failure — the base
# CORE_MODULES still applies, and downstream lints surface gaps).
_MODALITY_MODULES: dict[str, set[str]] = {
    "synchronous_oneshot": set(),
    "conversational_session": set(),  # components.chat is already in CORE
    "streaming": {
        "mellea.stdlib.chunking",
        "mellea.stdlib.sampling.budget_forcing",
    },
    "realtime_media": {
        "mellea.stdlib.chunking",
        "mellea.stdlib.sampling.budget_forcing",
    },
    "stateful": {
        "mellea.stdlib.components.mify",
        "mellea.stdlib.components.mobject",
    },
    "review_gated": set(),
    "scheduled": set(),
    "event_triggered": set(),
    "heartbeat": set(),
}


# Tool-involvement-variant module additions. P2 = "skill calls tools" which
# uses the React framework; P3 = LLM-orchestrated tool calls. Skills with
# these variants need React/tool surface even when the modality is otherwise
# synchronous_oneshot.
_TOOL_VARIANT_MODULES: dict[str, set[str]] = {
    "P2": {
        "mellea.stdlib.components.react",
        "mellea.stdlib.frameworks.react",
    },
    "P3": {
        "mellea.stdlib.components.react",
        "mellea.stdlib.frameworks.react",
    },
}

# Static fallback for `forbidden_param_names` if the genslot symbol is not
# importable. Source: mellea-fy-deps.md:134-137 (snapshot 2026-04-28).
_FORBIDDEN_PARAM_NAMES_FALLBACK = [
    "f_args",
    "f_kwargs",
    "m",
    "context",
    "backend",
    "model_options",
    "strategy",
    "precondition_requirements",
    "requirements",
]

# Static fallback for the doc_index when docs.mellea.ai is unreachable and no
# cached copy exists. Source: mellea-fy-deps.md:217-242 (snapshot 2026-04-28).
_DOC_PAGES_FALLBACK = [
    "/getting-started/installation",
    "/tutorials/01-your-first-generative-program",
    "/tutorials/04-making-agents-reliable",
    "/concepts/generative-functions",
    "/concepts/requirements-system",
    "/concepts/instruct-validate-repair",
    "/concepts/mobjects-and-mify",
    "/concepts/context-and-sessions",
    "/how-to/enforce-structured-output",
    "/how-to/write-custom-verifiers",
    "/how-to/use-async-and-streaming",
    "/how-to/use-context-and-sessions",
    "/how-to/configure-model-options",
    "/how-to/use-images-and-vision",
    "/how-to/build-a-rag-pipeline",
    "/guide/backends-and-configuration",
    "/guide/tools-and-agents",
    "/advanced/inference-time-scaling",
    "/integrations/ollama",
    "/integrations/openai",
    "/integrations/bedrock",
    "/integrations/watsonx",
    "/integrations/huggingface",
    "/integrations/vertex-ai",
    "/integrations/langchain",
]


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically via a sibling .tmp file + os.replace.

    Uses os.replace so concurrent compiles in different processes cannot leave
    a half-written file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content)
    os.replace(tmp, path)


def _modality_specific_modules(intermediate_dir: Path) -> set[str]:
    """Return additional mellea modules to introspect based on the skill's
    modality + tool-involvement variant (recorded in classification.json).

    classification.json is written by Step 0 of mellea-fy. Modality drives
    most additions (e.g. streaming → chunking + budget_forcing). Tool
    involvement variants P2/P3 add React/tool surface even when the
    modality is otherwise non-streaming. Unknown values contribute no
    extras; CORE_MODULES still applies.
    """
    classification_path = intermediate_dir / "classification.json"
    if not classification_path.exists():
        return set()
    try:
        classification = json.loads(classification_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()

    extras: set[str] = set()
    modality = classification.get("modality")
    if isinstance(modality, str):
        extras |= _MODALITY_MODULES.get(modality, set())
    variant = classification.get("tool_involvement_variant")
    if isinstance(variant, str):
        extras |= _TOOL_VARIANT_MODULES.get(variant, set())
    return extras


def _introspect_mellea(referenced_modules: set[str]) -> dict[str, dict[str, Any]]:
    """Walk the installed `mellea` package and collect public callable signatures.

    Restricted to the union of CORE_MODULES and `referenced_modules` (from
    dependency_plan.json). Modules that fail to import or symbols that resolve
    to objects without inspectable signatures are skipped silently.

    For class objects defined in the module, additionally enumerate public
    methods and surface them as `ClassName.method` entries. Without this,
    `MelleaSession.chat()` and `MelleaSession.instruct()` — the core
    operations every skill uses — are invisible to the grounding because
    they're class attributes, not module-level callables. The drift we hit:
    LLM emits `m.chat(...) -> str` (Mellea 0.4 mental model) when the actual
    0.5 return type is `Message`. Surfacing the method signatures lets the
    grounding contradict that mental model directly.
    """
    try:
        mellea_pkg = importlib.import_module("mellea")
    except ImportError:
        return {}

    all_mellea = {
        m.name
        for m in pkgutil.walk_packages(path=mellea_pkg.__path__, prefix="mellea.")
    }
    # The ``mellea.backends`` namespace is enumerated WHOLESALE rather
    # than curated via CORE_MODULES. Rationale: the directive
    # (``mellea-fy-generate.md`` Rule 5-2) tells the LLM that
    # ``mellea_api_ref.json:.modules`` is the authoritative source for
    # what's importable — any path not present is invalid. When the
    # backends list drifts from runtime (as of 2026-05-19,
    # ``mellea.backends.ollama`` was importable but missing from the
    # curated set), the directive promises the LLM something the runtime
    # contradicts: real imports get flagged as false positives by
    # ``import-soundness``. Backends are small in count and high in
    # user-facing-ness, so enumerating them is a better discipline than
    # curating. Other namespaces (stdlib.components.*, etc.) remain
    # curated via CORE_MODULES to keep the surface focused.
    backends_namespace = {
        name for name in all_mellea if name.startswith("mellea.backends.")
    }
    to_scan = all_mellea & (
        CORE_MODULES | referenced_modules | backends_namespace
    )

    api_ref: dict[str, dict[str, Any]] = {}
    for module_name in sorted(to_scan):
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        symbols: dict[str, Any] = {}
        for name, obj in inspect.getmembers(mod, callable):
            if name.startswith("_"):
                continue
            try:
                symbols[name] = {"signature": f"{name}{inspect.signature(obj)}"}
            except (ValueError, TypeError):
                pass
            # If this is a class defined in (or re-exported into) the module
            # being introspected, also record its public method signatures
            # as `ClassName.method` entries.
            if inspect.isclass(obj):
                for method_name, method_obj in inspect.getmembers(obj):
                    if method_name.startswith("_"):
                        continue
                    if not (
                        inspect.isfunction(method_obj)
                        or inspect.ismethod(method_obj)
                    ):
                        continue
                    try:
                        sig = inspect.signature(method_obj)
                    except (ValueError, TypeError):
                        continue
                    symbols[f"{name}.{method_name}"] = {
                        "signature": f"{name}.{method_name}{sig}"
                    }
        if symbols:
            api_ref[module_name] = symbols
    return api_ref


def _extract_forbidden_param_names() -> list[str]:
    """Pull the live disallowed-param list from genslot, with static fallback."""
    try:
        from mellea.stdlib.components.genslot import (  # type: ignore
            _disallowed_param_names,
        )

        return list(_disallowed_param_names)
    except (ImportError, AttributeError):
        return list(_FORBIDDEN_PARAM_NAMES_FALLBACK)


def _load_compatibility_entries(mellea_version: str) -> list[dict[str, Any]]:
    """Load `.claude/data/compatibility.yaml` and filter to applicable entries.

    Filtering uses `packaging.specifiers.SpecifierSet` against the installed
    mellea version. An `applies_when` of "*" or missing means "always applies".
    Returns an empty list if the file is missing or unparseable.
    """
    compat_path = Path(".claude/data/compatibility.yaml")
    if not compat_path.exists():
        return []
    try:
        import yaml  # type: ignore

        compat = yaml.safe_load(compat_path.read_text()) or {}
    except Exception:
        return []
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
    except ImportError:
        return list(compat.get("entries", []))

    entries: list[dict[str, Any]] = []
    for entry in compat.get("entries", []):
        applies = entry.get("applies_when", "*")
        if applies == "*":
            entries.append(entry)
            continue
        try:
            if mellea_version in SpecifierSet(applies):
                entries.append(entry)
        except Exception:
            # If the specifier is malformed, include the entry rather than drop it.
            entries.append(entry)
    return entries


def _grounding_unavailable_payload() -> str:
    """JSON payload written when `mellea` is not installed (deps.md:163-175)."""
    return json.dumps(
        {
            "format_version": "1.0",
            "mellea_version": None,
            "grounding_unavailable": True,
            "modules": {},
            "forbidden_param_names": list(_FORBIDDEN_PARAM_NAMES_FALLBACK),
            "compatibility": [],
        },
        indent=2,
    )


def _api_ref_cache_path(version: str, extras: set[str] | None = None) -> Path:
    """Return the cache filename for the given mellea version + extras.

    Cache key includes a short hash of CORE_MODULES *and* the modality-driven
    `extras` so:
      - Changes to CORE_MODULES auto-invalidate stale caches without
        manual `rm -rf ~/.cache/mellea-skills-compiler/`.
      - A P2 (tool-dispatch) compile cannot inadvertently consume a cache
        produced by an earlier P4 (synchronous_oneshot) compile that was
        missing React modules.

    Exposed as a helper so tests can construct the same path without
    duplicating the hashing logic.
    """
    combined = set(CORE_MODULES) | (extras or set())
    digest_input = "|".join(sorted(combined))
    cache_hash = hashlib.sha256(digest_input.encode()).hexdigest()[:8]
    return CACHE_DIR / f"api_ref_{version}_{cache_hash}.json"


def _write_api_ref_sidecars(intermediate_dir: Path, payload: dict[str, Any]) -> None:
    """Write small-field sidecars alongside the monolithic ``mellea_api_ref.json``.

    The monolithic file is ~280 KB and exceeds the LLM-side Read tool's
    256 KB hard limit. Two of its top-level fields (`forbidden_param_names`,
    `compatibility`) are small (~1–2 KB each) and frequently consulted —
    extracting them as separate files lets the LLM consume them in
    full without seeking inside a file too large to read end-to-end.

    Schema for each sidecar::

        {
          "format_version": "1.0",
          "mellea_version": "<version>" | null,
          "grounding_unavailable": <bool>,
          "<field>": <field-payload>
        }

    Backwards-compatible: monolithic file is unchanged; sidecars are
    additive. Coherence test (`test_grounding.py`) verifies the sidecar
    field values match the monolithic file's corresponding top-level
    fields exactly — preventing producer drift.
    """
    meta = {
        "format_version": "1.0",
        "mellea_version": payload.get("mellea_version"),
        "grounding_unavailable": bool(payload.get("grounding_unavailable", False)),
    }
    compatibility_sidecar = {
        **meta,
        "compatibility": payload.get("compatibility", []),
    }
    forbidden_sidecar = {
        **meta,
        "forbidden_param_names": payload.get("forbidden_param_names", []),
    }
    _atomic_write(
        intermediate_dir / "mellea_api_ref.compatibility.json",
        json.dumps(compatibility_sidecar, indent=2),
    )
    _atomic_write(
        intermediate_dir / "mellea_api_ref.forbidden_param_names.json",
        json.dumps(forbidden_sidecar, indent=2),
    )


def write_mellea_api_ref(intermediate_dir: Path, refresh: bool = False) -> Path:
    """Write `mellea_api_ref.json` to `intermediate_dir`.

    Cached by installed mellea version under
    `~/.cache/mellea-skills-compiler/api_ref_<version>.json`. If `mellea` is
    not installed, writes the `grounding_unavailable: true` shape and returns.

    Also writes two small-field sidecars (`mellea_api_ref.compatibility.json`
    and `mellea_api_ref.forbidden_param_names.json`) so LLM consumers can
    read those fields without seeking inside the 280+ KB monolithic file —
    see :func:`_write_api_ref_sidecars`.
    """
    out_path = intermediate_dir / "mellea_api_ref.json"

    try:
        version = importlib.metadata.version("mellea")
    except importlib.metadata.PackageNotFoundError:
        LOGGER.warning(
            "mellea package not installed; writing grounding_unavailable api_ref"
        )
        serialized = _grounding_unavailable_payload()
        _atomic_write(out_path, serialized)
        _write_api_ref_sidecars(intermediate_dir, json.loads(serialized))
        return out_path

    # Cache key includes modality-driven extras so a P2 (tool-dispatch)
    # compile doesn't reuse a P4 (synchronous_oneshot) cache that's missing
    # React/tool modules. Resolve extras first, then look up.
    referenced = _modality_specific_modules(intermediate_dir)
    cache_path = _api_ref_cache_path(version, referenced)

    if cache_path.exists() and not refresh:
        LOGGER.info("Using cached mellea_api_ref for version %s", version)
        serialized = cache_path.read_text()
        _atomic_write(out_path, serialized)
        _write_api_ref_sidecars(intermediate_dir, json.loads(serialized))
        return out_path

    LOGGER.info("Introspecting mellea %s for api_ref", version)
    modules = _introspect_mellea(referenced)
    forbidden = _extract_forbidden_param_names()
    compatibility = _load_compatibility_entries(version)

    payload = {
        "format_version": "1.0",
        "mellea_version": version,
        "grounding_unavailable": False,
        "modules": modules,
        "forbidden_param_names": forbidden,
        "compatibility": compatibility,
    }
    serialized = json.dumps(payload, indent=2)

    _atomic_write(cache_path, serialized)
    _atomic_write(out_path, serialized)
    _write_api_ref_sidecars(intermediate_dir, payload)
    return out_path


def _fetch_doc_pages() -> list[str]:
    """Fetch and parse navigation hrefs from docs.mellea.ai. Raises on failure."""
    with urllib.request.urlopen("https://docs.mellea.ai/", timeout=10) as resp:
        html = resp.read().decode()
    return sorted(set(re.findall(r'href="(/[^"]+)"', html)))


def write_mellea_doc_index(
    intermediate_dir: Path, refresh: bool = False, ttl_hours: int = 24
) -> Path:
    """Write `mellea_doc_index.json` to `intermediate_dir`.

    Cached at `~/.cache/mellea-skills-compiler/doc_index.json` with a
    `ttl_hours` TTL (default 24h). On fetch failure, reuses a stale cache if
    one exists; otherwise writes the static 2026-04-28 fallback list.
    """
    out_path = intermediate_dir / "mellea_doc_index.json"
    cache_path = CACHE_DIR / "doc_index.json"

    # Cache hit within TTL — reuse without touching the network.
    if cache_path.exists() and not refresh:
        try:
            cached = json.loads(cache_path.read_text())
            fetched_at_str = cached.get("fetched_at", "")
            fetched_at = datetime.fromisoformat(fetched_at_str)
            age = datetime.now(timezone.utc) - fetched_at
            if age.total_seconds() < ttl_hours * 3600:
                _atomic_write(out_path, cache_path.read_text())
                return out_path
        except (OSError, ValueError, json.JSONDecodeError):
            # Treat a corrupt cache as a miss; we'll try to refetch.
            pass

    # Need a fresh fetch (cache missing, expired, corrupt, or refresh=True).
    try:
        doc_pages = _fetch_doc_pages()
        LOGGER.info("Fetched %d doc pages from docs.mellea.ai", len(doc_pages))
        payload = {
            "format_version": "1.0",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "https://docs.mellea.ai/",
            "fetch_status": "ok",
            "doc_pages": doc_pages,
        }
        serialized = json.dumps(payload, indent=2)
        _atomic_write(cache_path, serialized)
        _atomic_write(out_path, serialized)
        return out_path
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Fetch failed. Prefer a stale cache (better than nothing) over the
        # hardcoded fallback.
        if cache_path.exists():
            try:
                cached_text = cache_path.read_text()
                cached = json.loads(cached_text)
                LOGGER.warning(
                    "docs.mellea.ai unreachable; using stale cache from %s",
                    cached.get("fetched_at", "<unknown>"),
                )
                _atomic_write(out_path, cached_text)
                return out_path
            except (OSError, json.JSONDecodeError):
                pass

        LOGGER.warning(
            "docs.mellea.ai unreachable and no cache; using static fallback "
            "(2026-04-28 snapshot)"
        )
        payload = {
            "format_version": "1.0",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "https://docs.mellea.ai/",
            "fetch_status": f"failed: {exc}",
            "doc_pages": list(_DOC_PAGES_FALLBACK),
        }
        _atomic_write(out_path, json.dumps(payload, indent=2))
        return out_path
