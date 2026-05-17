"""Tests for ``mellea_skills_compiler.compile.descriptor_emission``.

The test suite is **fully offline** by default — every test that isn't
explicitly marked ``@pytest.mark.live`` uses :class:`_FakeAnthropicClient`
to drive the emission. The fake mimics the SDK's ``messages.stream(...)``
context-manager + ``get_final_message()`` shape and exposes the prompt
arguments to assertions.

The single ``@pytest.mark.live`` test calls the real LiteLLM proxy against
the ``sentry-find-bugs`` spec. It is skipped by default. To run it:

.. code-block:: bash

    source /tmp/anthropic-litellm.env
    .venv-spike/bin/python -m pytest \\
        tests/mellea_skills_compiler/compile/test_descriptor_emission.py \\
        -m live --run-live -q
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# --- Helpers -----------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)
V02_SENTRY_DESCRIPTOR_PATH = (
    REPO_ROOT
    / "tests"
    / "mellea_skills_compiler"
    / "descriptor"
    / "fixtures"
    / "v02"
    / "sentry-find-bugs.descriptor.json"
)
SENTRY_SPEC_PATH = REPO_ROOT / "skills" / "sentry-find-bugs" / "spec.md"


@pytest.fixture(scope="module")
def surface() -> dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sentry_descriptor() -> dict[str, Any]:
    return json.loads(V02_SENTRY_DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def _wrap_descriptor_response(
    descriptor: dict[str, Any], *, reasoning: str = "default reasoning"
) -> str:
    """Format a fake LLM response that matches the production prompt contract."""
    body = json.dumps(descriptor, indent=2)
    return (
        f"<reasoning>{reasoning}</reasoning>\n"
        f"<descriptor>\n{body}\n</descriptor>\n"
    )


class _FakeStream:
    """Stand-in for the Anthropic ``MessageStreamManager``.

    Supports the ``with client.messages.stream(...) as stream:`` /
    ``stream.get_final_message()`` shape used by production code.
    """

    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get_final_message(self) -> Any:
        return self._message


class _FakeMessages:
    def __init__(self, owner: "_FakeAnthropicClient") -> None:
        self._owner = owner

    def stream(self, **kwargs: Any) -> _FakeStream:
        self._owner.last_stream_kwargs = kwargs
        self._owner.stream_call_count += 1
        message = self._owner.next_message
        if callable(message):
            message = message(kwargs)
        return _FakeStream(message)


class _FakeAnthropicClient:
    """Minimal Anthropic client double for unit tests.

    Attributes the tests read:
      * ``last_stream_kwargs`` — the kwargs of the most recent ``stream()`` call.
      * ``stream_call_count`` — number of times ``stream()`` has been invoked.

    Set ``next_message`` to either a fake message object or a callable
    ``(kwargs) -> message``.
    """

    def __init__(self, response_text: str, usage: dict[str, int] | None = None) -> None:
        usage_obj = SimpleNamespace(
            input_tokens=(usage or {}).get("input_tokens", 1500),
            output_tokens=(usage or {}).get("output_tokens", 4200),
            cache_read_input_tokens=(usage or {}).get("cache_read_input_tokens", 1200),
            cache_creation_input_tokens=(usage or {}).get(
                "cache_creation_input_tokens", 0
            ),
        )
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=response_text)],
            usage=usage_obj,
        )
        self.next_message: Any = message
        self.messages = _FakeMessages(self)
        self.last_stream_kwargs: dict[str, Any] | None = None
        self.stream_call_count = 0


# --- EmissionConfig / prompt construction -----------------------------------


def test_emission_config_defaults() -> None:
    """Locked-in Phase 1 values must be the dataclass defaults."""
    from mellea_skills_compiler.compile.descriptor_emission import EmissionConfig

    cfg = EmissionConfig()
    assert cfg.model == "aws/claude-sonnet-4-6"
    assert cfg.max_tokens == 64000
    assert cfg.descriptor_schema_version == "0.2"
    assert cfg.one_shot_descriptor_path is None
    assert cfg.schema_doc_path is None
    assert cfg.relevant_modules == (
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


def test_emission_config_default_paths_exist() -> None:
    """Bundled defaults must resolve to real files in the repo."""
    from mellea_skills_compiler.compile.descriptor_emission import EmissionConfig

    cfg = EmissionConfig()
    assert cfg.resolved_one_shot_path().is_file()
    assert cfg.resolved_schema_doc_path().is_file()


def test_system_prompt_is_byte_stable(surface: dict[str, Any]) -> None:
    """Same config + surface produces byte-identical system blocks."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig()
    a = build_system_prompt(cfg, surface)
    b = build_system_prompt(cfg, surface)
    assert a == b
    assert len(a) == 1
    assert a[0]["type"] == "text"
    assert a[0]["cache_control"] == {"type": "ephemeral"}
    # Sanity: the cache block must contain the three required elements.
    text = a[0]["text"]
    assert "Descriptor schema" in text
    assert "Introspected Mellea 0.5 surface" in text
    assert "One-shot ground-truth example" in text


def test_system_prompt_stable_under_dict_order(surface: dict[str, Any]) -> None:
    """Reordering the surface dict's keys must not change the prompt bytes."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig()
    a = build_system_prompt(cfg, surface)
    shuffled = {
        "python_version": surface.get("python_version"),
        "mellea_version": surface.get("mellea_version"),
        "modules": dict(reversed(list(surface["modules"].items()))),
    }
    b = build_system_prompt(cfg, shuffled)
    assert a[0]["text"] == b[0]["text"]


def test_filtered_surface_serialises_deterministically(surface: dict[str, Any]) -> None:
    """The serialised surface subset uses ``sort_keys=True`` for prompt-cache stability."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        _filter_surface,
        _serialise_filtered_surface,
    )

    cfg = EmissionConfig()
    text_a = _serialise_filtered_surface(surface, cfg.relevant_modules)
    text_b = _serialise_filtered_surface(surface, cfg.relevant_modules)
    assert text_a == text_b

    parsed = json.loads(text_a)
    assert set(parsed["modules"].keys()) == {
        m for m in cfg.relevant_modules if m in surface["modules"]
    }

    # Independent check: every keys-listing in the dumped string must be sorted.
    re_dumped = json.dumps(parsed, indent=2, sort_keys=True)
    assert text_a == re_dumped


def test_filter_surface_drops_unrequested_modules(surface: dict[str, Any]) -> None:
    """Only modules in ``relevant_modules`` survive the filter."""
    from mellea_skills_compiler.compile.descriptor_emission import _filter_surface

    subset = _filter_surface(surface, ("mellea.stdlib.session",))
    assert set(subset["modules"].keys()) == {"mellea.stdlib.session"}


def test_user_message_contains_spec_and_instructions() -> None:
    """User message must carry the spec text and reasoning/descriptor tag instructions."""
    from mellea_skills_compiler.compile.descriptor_emission import build_user_message

    msg = build_user_message("Spec line 1\nSpec line 2")
    assert "Spec line 1" in msg
    assert "<reasoning>" in msg
    assert "<descriptor>" in msg
    assert "spec.md" in msg


# --- Tag extraction ---------------------------------------------------------


def test_extract_descriptor_handles_well_formed_response(
    sentry_descriptor: dict[str, Any],
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        extract_descriptor_text,
        extract_reasoning,
    )

    raw = _wrap_descriptor_response(sentry_descriptor, reasoning="step one")
    body, err = extract_descriptor_text(raw)
    assert err is None
    assert body is not None
    parsed = json.loads(body)
    assert parsed["descriptor_version"] == "0.2"
    assert parsed["skill"]["name"] == "find-bugs"
    assert extract_reasoning(raw) == "step one"


def test_extract_descriptor_handles_missing_tags() -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        extract_descriptor_text,
    )

    body, err = extract_descriptor_text("just some text without tags")
    assert body is None
    assert err is not None and "missing <descriptor>" in err


def test_extract_descriptor_strips_accidental_code_fence(
    sentry_descriptor: dict[str, Any],
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        extract_descriptor_text,
    )

    body_inner = json.dumps(sentry_descriptor, indent=2)
    raw = f"<descriptor>\n```json\n{body_inner}\n```\n</descriptor>"
    body, err = extract_descriptor_text(raw)
    assert err is None
    assert body is not None
    # The cleaned body must JSON-parse — the bug we are defending against is
    # ``json.loads(...)`` choking on the leading/trailing fence lines.
    parsed = json.loads(body)
    assert parsed["descriptor_version"] == "0.2"


def test_extract_reasoning_returns_none_when_absent() -> None:
    from mellea_skills_compiler.compile.descriptor_emission import extract_reasoning

    assert extract_reasoning("<descriptor>{}</descriptor>") is None


# --- emit_descriptor: mocked LLM --------------------------------------------


def test_emit_descriptor_well_formed(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    result = emit_descriptor("dummy spec", surface, client=client)
    assert result.descriptor is not None
    assert result.descriptor["skill"]["name"] == "find-bugs"
    assert result.parse_error is None
    assert result.reasoning_text == "default reasoning"
    assert result.model == "aws/claude-sonnet-4-6"


def test_emit_descriptor_handles_missing_tags(surface: dict[str, Any]) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    client = _FakeAnthropicClient("the model went off the rails")
    result = emit_descriptor("spec", surface, client=client)
    assert result.descriptor is None
    assert result.descriptor_text is None
    assert result.parse_error is not None and "missing <descriptor>" in result.parse_error


def test_emit_descriptor_handles_malformed_json(surface: dict[str, Any]) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    client = _FakeAnthropicClient(
        "<reasoning>r</reasoning><descriptor>{not: valid, json,}</descriptor>"
    )
    result = emit_descriptor("spec", surface, client=client)
    assert result.descriptor is None
    assert result.descriptor_text is not None  # We kept the raw text for debugging.
    assert result.parse_error is not None
    assert "JSON parse failed" in result.parse_error


def test_emit_descriptor_usage_fields_populated(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    client = _FakeAnthropicClient(
        _wrap_descriptor_response(sentry_descriptor),
        usage={
            "input_tokens": 12345,
            "output_tokens": 6789,
            "cache_read_input_tokens": 999,
            "cache_creation_input_tokens": 111,
        },
    )
    result = emit_descriptor("spec", surface, client=client)
    assert result.usage == {
        "input_tokens": 12345,
        "output_tokens": 6789,
        "cache_read_input_tokens": 999,
        "cache_creation_input_tokens": 111,
    }
    assert result.latency_seconds >= 0.0


def test_emit_descriptor_uses_streaming_and_cached_system(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    """Verify the SDK call shape: streaming, single cache-controlled system block."""
    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    emit_descriptor("spec body", surface, client=client)

    kwargs = client.last_stream_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "aws/claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 64000
    assert isinstance(kwargs["system"], list) and len(kwargs["system"]) == 1
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    msgs = kwargs["messages"]
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "spec body" in msgs[0]["content"]


def test_emit_descriptor_honours_overridden_model(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        emit_descriptor,
    )

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    cfg = EmissionConfig(model="custom/test-model", max_tokens=8000)
    result = emit_descriptor("spec", surface, cfg, client=client)
    assert result.model == "custom/test-model"
    assert client.last_stream_kwargs is not None
    assert client.last_stream_kwargs["model"] == "custom/test-model"
    assert client.last_stream_kwargs["max_tokens"] == 8000


# --- compile_via_descriptor: end-to-end mocked ------------------------------


def test_compile_end_to_end_with_canned_descriptor(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], tmp_path: Path
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    out_dir = tmp_path / "rendered"
    result = compile_via_descriptor(
        "dummy spec",
        surface,
        client=client,
        write_to=out_dir,
    )
    assert result.failure_stage is None, result.failure_reason
    assert result.failure_reason is None
    assert result.smoke_ok is True
    assert result.pipeline_py is not None and "def run_pipeline" in result.pipeline_py
    assert result.schemas_py is not None and "class FindingsReport" in result.schemas_py
    assert result.init_py is not None and "run_pipeline" in result.init_py
    assert result.fixtures_py is not None
    assert result.melleafy_json is not None
    assert result.setup_md is not None
    assert result.readme_md is not None
    assert result.source_map is not None
    assert result.validation is not None and result.validation.valid

    # The persisted package matches the in-memory artefacts.
    assert (out_dir / "pipeline.py").is_file()
    assert (out_dir / "schemas.py").is_file()
    assert (out_dir / "__init__.py").is_file()
    assert (out_dir / "fixtures.py").is_file()
    assert (out_dir / "melleafy.json").is_file()
    assert (out_dir / "SETUP.md").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "source_map.json").is_file()
    assert (out_dir / "fixtures" / "__init__.py").is_file()


def test_compile_propagates_emission_failure(surface: dict[str, Any]) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    client = _FakeAnthropicClient("nothing useful here")
    result = compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage == "emission"
    assert result.failure_reason is not None
    assert result.validation is None
    assert result.pipeline_py is None
    assert result.smoke_ok is False


def test_compile_propagates_validation_failure(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    """An unknown symbol must surface as failure_stage='validation'."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    bad = copy.deepcopy(sentry_descriptor)
    # Replace the first call node's symbol with one that resolves nowhere.
    def _spoil(nodes: list[dict[str, Any]]) -> bool:
        for n in nodes:
            if n.get("kind") == "call":
                n["symbol"] = "mellea.stdlib.session.NotARealSymbol"
                return True
            if n.get("kind") == "composition" and _spoil(n.get("body", [])):
                return True
        return False

    assert _spoil(bad["pipeline"]), "test setup: could not find a call node to spoil"

    client = _FakeAnthropicClient(_wrap_descriptor_response(bad))
    result = compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage == "validation"
    assert result.validation is not None
    assert not result.validation.valid
    assert result.pipeline_py is None
    assert result.smoke_ok is False
    assert result.failure_reason is not None


def test_compile_propagates_render_failure(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the renderer to raise RendererError after validation passes."""
    from mellea_skills_compiler.compile import descriptor_emission as mod
    from mellea_skills_compiler.renderer import RendererError

    def _boom(descriptor, surface, **kwargs):  # noqa: ANN001
        raise RendererError("synthetic operator failure")

    # The compile path imports render_descriptor inside _render_full_package;
    # patch the import target rather than the symbol on `mod`.
    import mellea_skills_compiler.renderer as renderer_pkg

    monkeypatch.setattr(renderer_pkg, "render_descriptor", _boom, raising=True)
    # Defensive: also patch on `core` in case any code path imports from there.
    from mellea_skills_compiler.renderer import core as renderer_core

    monkeypatch.setattr(renderer_core, "render_descriptor", _boom, raising=True)

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    result = mod.compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage == "render"
    assert result.failure_reason is not None
    assert "synthetic operator failure" in result.failure_reason
    assert result.pipeline_py is None
    assert result.smoke_ok is False


def test_compile_propagates_smoke_failure(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline that compiles cleanly but raises on import → failure_stage='smoke'."""
    from mellea_skills_compiler.compile import descriptor_emission as mod

    # Patch the internal renderer recipe to swap pipeline.py for a guaranteed-
    # broken-on-import module. We replicate the rest of the call exactly so we
    # only stress the smoke stage.
    original = mod._render_full_package

    def _patched(descriptor, surface, **kwargs):  # noqa: ANN001
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
        ) = original(descriptor, surface, **kwargs)
        broken_pipeline = (
            "from __future__ import annotations\n"
            "raise RuntimeError('synthetic import-time failure')\n"
            "def run_pipeline():\n    return None\n"
        )
        return (
            broken_pipeline,
            schemas_py,
            init_py,
            fixtures_py,
            melleafy_json,
            setup_md,
            readme_md,
            source_map,
            render_result,
        )

    monkeypatch.setattr(mod, "_render_full_package", _patched, raising=True)

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    result = mod.compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage == "smoke"
    assert result.failure_reason is not None
    assert "synthetic import-time failure" in result.failure_reason
    assert result.pipeline_py is not None  # render succeeded; we kept its output
    assert result.smoke_ok is False


def test_compile_does_not_write_on_failure(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], tmp_path: Path
) -> None:
    """write_to must NOT receive partial output on a failure path."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    bad = copy.deepcopy(sentry_descriptor)
    # Make the descriptor invalid by referencing a symbol that won't resolve in
    # the surface. (Previously this test bumped `descriptor_version` to "0.1" to
    # trigger a JSON-Schema mismatch against v0.2 — but the validator now picks
    # the schema version from the descriptor itself, so a v0.1 descriptor validates
    # against v0.1. We need a *semantic* failure instead.)
    bad["state"][0]["symbol"] = "mellea.nonexistent.module.DoesNotExist"
    client = _FakeAnthropicClient(_wrap_descriptor_response(bad))
    out_dir = tmp_path / "should_be_empty"
    result = compile_via_descriptor(
        "spec",
        surface,
        client=client,
        write_to=out_dir,
    )
    assert result.failure_stage == "validation"
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_compile_writes_only_on_success(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], tmp_path: Path
) -> None:
    """write_to is optional; success without write_to leaves no files behind."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    out_dir = tmp_path / "absent"
    result = compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage is None
    assert not out_dir.exists()


def test_compile_records_validation_on_success(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    client = _FakeAnthropicClient(_wrap_descriptor_response(sentry_descriptor))
    result = compile_via_descriptor("spec", surface, client=client)
    assert result.validation is not None
    assert result.validation.valid is True
    # The emission attempt is always preserved.
    assert result.emission.descriptor is not None
    assert result.emission.parse_error is None


def test_compile_returns_emission_even_on_emission_failure(
    surface: dict[str, Any],
) -> None:
    """Failure paths must still surface the emission usage / raw output for telemetry."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        compile_via_descriptor,
    )

    client = _FakeAnthropicClient(
        "no tags at all",
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    )
    result = compile_via_descriptor("spec", surface, client=client)
    assert result.failure_stage == "emission"
    assert result.emission.usage["input_tokens"] == 100
    assert result.emission.raw_output == "no tags at all"


# --- One-shot path resolution + bundled defaults ----------------------------


def test_emit_descriptor_respects_custom_one_shot(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any], tmp_path: Path
) -> None:
    """A user-supplied one-shot path must appear in the prompt text."""
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    custom = tmp_path / "custom.descriptor.json"
    sentinel = {"hello": "world-distinctive-marker-7777"}
    custom.write_text(json.dumps(sentinel), encoding="utf-8")
    cfg = EmissionConfig(one_shot_descriptor_path=custom)
    blocks = build_system_prompt(cfg, surface)
    assert "world-distinctive-marker-7777" in blocks[0]["text"]


def test_emit_descriptor_respects_custom_schema_doc(
    surface: dict[str, Any], tmp_path: Path
) -> None:
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    custom_doc = tmp_path / "custom.md"
    custom_doc.write_text("# Custom schema doc with sentinel-zzz-9090\n", encoding="utf-8")
    cfg = EmissionConfig(schema_doc_path=custom_doc)
    blocks = build_system_prompt(cfg, surface)
    assert "sentinel-zzz-9090" in blocks[0]["text"]


# --- Phase 3.5.A: intermediate-artefact prompt enrichment -------------------


# Captured against the pre-Phase-3.5.A `build_system_prompt` output (using
# `EmissionConfig()` defaults + `surface_0.5.0.json`). Regenerate via:
#
#     .venv-spike/bin/python -c "
#     import json, hashlib
#     from pathlib import Path
#     from mellea_skills_compiler.compile.descriptor_emission import (
#         EmissionConfig, build_system_prompt,
#     )
#     surface = json.loads(Path(
#         'melleafy-handoff/kickoff/spike-outputs/surface_0.5.0.json'
#     ).read_text())
#     blocks = build_system_prompt(EmissionConfig(), surface)
#     print(hashlib.sha256(blocks[0]['text'].encode()).hexdigest())
#     "
#
# Updating this hash requires explicit human sign-off — a byte-stable system
# prompt is the prompt-cache contract (P3.5.A §3 / parity plan §12.2).
SYSTEM_PROMPT_BYTE_STABILITY_SHA256 = (
    "bafbd6a1c337e4a24a95e5b8038681bbd16604df923b9d1df72e7cc1fe0bc217"
)


def test_system_prompt_byte_stable_without_artefacts(surface: dict[str, Any]) -> None:
    """Snapshot guard: when `intermediate_artefacts` is None (the default and
    the one-shot caller contract), `build_system_prompt` MUST produce the
    pre-Phase-3.5.A output byte-for-byte. Existing one-shot callers
    (`batch_descriptor_test.py`, `corpus_compare.py`) rely on this.
    """
    import hashlib

    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig()
    assert cfg.intermediate_artefacts is None
    blocks = build_system_prompt(cfg, surface)
    actual = hashlib.sha256(blocks[0]["text"].encode("utf-8")).hexdigest()
    assert actual == SYSTEM_PROMPT_BYTE_STABILITY_SHA256, (
        "system prompt has drifted from the byte-stable snapshot. "
        "If this drift is intentional, regenerate the SHA per the module-level "
        "comment AND get explicit human sign-off — it breaks prompt-cache "
        "warmth for every one-shot caller."
    )


def test_system_prompt_includes_each_artefact_when_present(
    surface: dict[str, Any],
) -> None:
    """For each of the eight artefact names listed in P3.5.A, when supplied
    via `intermediate_artefacts`, the artefact's content must appear in the
    rendered system prompt under a stable `<artefact name="...">` delimiter.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        _INTERMEDIATE_ARTEFACT_NAMES,
        EmissionConfig,
        build_system_prompt,
    )

    sentinel_value = "marker-distinctive-{name}-9090"
    artefacts: dict[str, Any] = {
        name: {"sentinel": sentinel_value.format(name=name)}
        for name in _INTERMEDIATE_ARTEFACT_NAMES
    }
    cfg = EmissionConfig(intermediate_artefacts=artefacts)
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    for name in _INTERMEDIATE_ARTEFACT_NAMES:
        assert f'<artefact name="{name}">' in text, f"{name} missing delimiter"
        assert f"</artefact>" in text
        assert sentinel_value.format(name=name) in text, (
            f"{name} content not inlined into system prompt"
        )


def test_system_prompt_artefact_order_is_deterministic(
    surface: dict[str, Any],
) -> None:
    """Two builds with the same artefacts (in different dict iteration order)
    must produce byte-identical prompts — the artefact section uses a
    canonical 8-artefact order so the cache stays warm.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        _INTERMEDIATE_ARTEFACT_NAMES,
        EmissionConfig,
        build_system_prompt,
    )

    a_dict = {name: {"k": f"v-{name}"} for name in _INTERMEDIATE_ARTEFACT_NAMES}
    b_dict = {
        name: a_dict[name] for name in reversed(_INTERMEDIATE_ARTEFACT_NAMES)
    }
    a = build_system_prompt(EmissionConfig(intermediate_artefacts=a_dict), surface)
    b = build_system_prompt(EmissionConfig(intermediate_artefacts=b_dict), surface)
    assert a[0]["text"] == b[0]["text"]


def test_system_prompt_artefact_section_absent_when_empty_dict(
    surface: dict[str, Any],
) -> None:
    """An empty `intermediate_artefacts` dict is equivalent to `None` —
    the system prompt must remain byte-stable so callers passing `{}`
    don't accidentally break cache warmth.
    """
    import hashlib

    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig(intermediate_artefacts={})
    blocks = build_system_prompt(cfg, surface)
    actual = hashlib.sha256(blocks[0]["text"].encode("utf-8")).hexdigest()
    assert actual == SYSTEM_PROMPT_BYTE_STABILITY_SHA256


def test_system_prompt_handles_partial_artefacts(surface: dict[str, Any]) -> None:
    """Phase 3.5.A treats every artefact as optional — when only some are
    present (e.g. `expected_signature` absent until P3.5.D lands), the
    section renders the ones supplied without complaining about missing
    keys.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    partial = {
        "classification": {"primary_axis": "AGENT"},
        "inventory": {"elements": []},
        # The other 6 (including expected_signature from P3.5.D) intentionally absent.
    }
    cfg = EmissionConfig(intermediate_artefacts=partial)
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    assert '<artefact name="classification">' in text
    assert '<artefact name="inventory">' in text
    # Other artefacts should NOT appear when absent.
    assert '<artefact name="dependency_plan">' not in text
    assert '<artefact name="expected_signature">' not in text


def test_system_prompt_artefact_section_appears_after_one_shot(
    surface: dict[str, Any],
) -> None:
    """The artefact section must appear after the one-shot example — the
    schema doc + surface + one-shot stay at the top of the cached block so
    descriptor authors see them first, then the per-skill analytical artefacts.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig(
        intermediate_artefacts={"classification": {"primary_axis": "DSL"}}
    )
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    one_shot_idx = text.find("One-shot ground-truth example")
    artefacts_idx = text.find("Intermediate analytical artefacts")
    assert one_shot_idx >= 0
    assert artefacts_idx > one_shot_idx, (
        "intermediate-artefacts section must come AFTER the one-shot example"
    )


# --- Phase 3.5.D: expected_signature HARD CONSTRAINT block ------------------


def test_system_prompt_expected_signature_renders_hard_constraint_block(
    surface: dict[str, Any],
) -> None:
    """Phase 3.5.D: when `expected_signature` is present in
    `intermediate_artefacts`, the system prompt MUST inline it as a HARD
    CONSTRAINT block emphasising that the descriptor's inputs/outputs
    must match exactly. This is the contract that closes the
    signature-drift bug (G6).
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    expected_sig = {
        "format_version": "1.0",
        "function_name": "run_pipeline",
        "inputs": [{"name": "diff", "type": "str"}],
        "outputs": [{"name": "result", "type": "str"}],
        "schemas": [],
    }
    cfg = EmissionConfig(
        intermediate_artefacts={"expected_signature": expected_sig}
    )
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    # Hard-constraint preamble + canonical artefact delimiter + content.
    assert "HARD CONSTRAINT" in text
    assert "expected_signature lock" in text
    assert "MUST match this expected signature" in text
    # Standard delimiter preserved so the test_system_prompt_includes_each_artefact_when_present
    # contract is upheld:
    assert '<artefact name="expected_signature">' in text
    # And the content was inlined:
    assert "run_pipeline" in text


def test_system_prompt_expected_signature_appears_after_other_artefacts(
    surface: dict[str, Any],
) -> None:
    """When BOTH expected_signature and other artefacts are present, the
    HARD CONSTRAINT block appears LAST in the artefact section — the
    descriptor author sees the analytical context first, then the locked
    signature as the closing constraint.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    cfg = EmissionConfig(
        intermediate_artefacts={
            "classification": {"primary_axis": "DSL"},
            "expected_signature": {
                "format_version": "1.0",
                "function_name": "run_pipeline",
                "inputs": [{"name": "diff", "type": "str"}],
                "outputs": [{"name": "result", "type": "str"}],
                "schemas": [],
            },
        }
    )
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    classification_idx = text.find('<artefact name="classification">')
    hard_idx = text.find("HARD CONSTRAINT — expected_signature lock")
    expected_sig_idx = text.find('<artefact name="expected_signature">')
    assert classification_idx >= 0
    assert hard_idx > classification_idx
    assert expected_sig_idx > hard_idx


def test_system_prompt_byte_stable_when_expected_signature_absent(
    surface: dict[str, Any],
) -> None:
    """P3.5.D byte-stability invariant: the HARD CONSTRAINT preamble must
    NOT appear when expected_signature is absent from intermediate_artefacts.
    This is what lets the snapshot-SHA test continue to pass — the legacy
    one-shot callers never see the preamble.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        build_system_prompt,
    )

    # 1) Default (intermediate_artefacts=None): no preamble.
    blocks = build_system_prompt(EmissionConfig(), surface)
    text = blocks[0]["text"]
    assert "HARD CONSTRAINT" not in text

    # 2) Some artefacts but not expected_signature: still no preamble.
    cfg = EmissionConfig(
        intermediate_artefacts={"classification": {"primary_axis": "DSL"}}
    )
    blocks = build_system_prompt(cfg, surface)
    text = blocks[0]["text"]
    assert "HARD CONSTRAINT" not in text


def test_compile_via_descriptor_threads_expected_signature_to_validator(
    surface: dict[str, Any], sentry_descriptor: dict[str, Any]
) -> None:
    """End-to-end: when intermediate_artefacts contains expected_signature,
    compile_via_descriptor passes it to validate() so R-SEM-SIGNATURE-MATCH
    can fire. A mismatched signature should produce a validation failure
    citing R-SEM-SIGNATURE-MATCH.
    """
    from mellea_skills_compiler.compile.descriptor_emission import (
        EmissionConfig,
        compile_via_descriptor,
    )

    # Build a tiny expected_signature whose input name diverges from the
    # sentry one-shot's actual inputs. Whatever the sentry descriptor's
    # first input is named, we expect a different name to force a mismatch.
    canned_response = _wrap_descriptor_response(sentry_descriptor)
    client = _FakeAnthropicClient(canned_response)

    expected_sig = {
        "format_version": "1.0",
        "function_name": "run_pipeline",
        "inputs": [
            {"name": "totally_fictional_input_for_mismatch_test", "type": "str"}
        ],
        "outputs": [{"name": "result", "type": "str"}],
        "schemas": [],
    }
    cfg = EmissionConfig(
        intermediate_artefacts={"expected_signature": expected_sig}
    )
    result = compile_via_descriptor(
        spec_text="# fake spec", surface=surface, config=cfg, client=client
    )
    # The validation step should fail with R-SEM-SIGNATURE-MATCH.
    assert result.failure_stage == "validation"
    assert result.validation is not None
    sig_errors = [
        e for e in result.validation.errors if e.rule == "R-SEM-SIGNATURE-MATCH"
    ]
    assert sig_errors, (
        f"expected at least one R-SEM-SIGNATURE-MATCH error; got rules: "
        f"{[e.rule for e in result.validation.errors]}"
    )


# --- Phase 3.5.A: CLI subprocess plumbing -----------------------------------


def test_cli_use_descriptor_forwards_flag_to_shared_compile(monkeypatch, tmp_path):
    """Integration shape (subprocess plumbing): `mellea-skills compile
    --use-descriptor <path>` must route through the shared
    `mellea_skills.compile` with `use_descriptor=True`, so the slash-command
    orchestrator receives `--use-descriptor` in `$ARGUMENTS`.

    The actual `claude` subprocess is NEVER spawned in this test — we mock
    `mellea_skills.compile` and assert its kwargs. This is the smallest
    test that proves the CLI's `--use-descriptor` flag flows through to
    the slash-command argv builder.
    """
    from unittest.mock import MagicMock
    from typer.testing import CliRunner

    from mellea_skills_compiler.cli import app
    from mellea_skills_compiler.compile import mellea_skills as _ms

    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "spec.md").write_text("# test spec")

    compile_mock = MagicMock(return_value=None)
    monkeypatch.setattr(_ms, "compile", compile_mock)

    runner = CliRunner()
    result = runner.invoke(
        app, ["compile", str(skill), "--use-descriptor", "--no-run"]
    )
    assert result.exit_code == 0, result.stdout
    compile_mock.assert_called_once()
    assert compile_mock.call_args.kwargs.get("use_descriptor") is True


def test_build_claude_argv_byte_identical_for_legacy(tmp_path) -> None:
    """Refactor invariant (parity plan §12.2): the argv constructed for
    legacy mode (`use_descriptor=False`) must be byte-identical to the
    pre-refactor build. This guards the `_spawn_claude` extraction.
    """
    from mellea_skills_compiler.compile.mellea_skills import _build_claude_argv

    spec_path = tmp_path / "spec.md"
    legacy_argv = _build_claude_argv(
        spec_path=spec_path,
        model="aws/claude-sonnet-4-6",
        system_prompt="<sys-prompt>",
        compile_settings_path=tmp_path / "settings.json",
        repair_mode=False,
        use_descriptor=False,
    )
    # Reconstruct the pre-refactor argv shape manually.
    expected = [
        "claude",
        "-p",
        "--model",
        "aws/claude-sonnet-4-6",
        "--append-system-prompt",
        "<sys-prompt>",
        "--allowed-tools",
        "Read,Write,Edit",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--settings",
        str(tmp_path / "settings.json"),
        f"'./mellea-fy {spec_path}'",
    ]
    assert legacy_argv == expected


def test_build_claude_argv_appends_use_descriptor(tmp_path) -> None:
    """Descriptor mode adds `--use-descriptor` to the slash-command argument
    so the orchestrator at `.claude/commands/mellea-fy.md` can parse it from
    `$ARGUMENTS` and forward it to `/mellea-fy-generate`.
    """
    from mellea_skills_compiler.compile.mellea_skills import _build_claude_argv

    spec_path = tmp_path / "spec.md"
    argv = _build_claude_argv(
        spec_path=spec_path,
        model="aws/claude-sonnet-4-6",
        system_prompt="<sys-prompt>",
        compile_settings_path=None,
        repair_mode=False,
        use_descriptor=True,
    )
    last = argv[-1]
    assert "--use-descriptor" in last
    assert "./mellea-fy" in last
    assert str(spec_path) in last


def test_build_claude_argv_repair_mode_uses_repair_slash_command(tmp_path) -> None:
    """`--repair-mode` switches the slash command to `./mellea-fy-repair`."""
    from mellea_skills_compiler.compile.mellea_skills import _build_claude_argv

    spec_path = tmp_path / "spec.md"
    argv = _build_claude_argv(
        spec_path=spec_path,
        model="aws/claude-sonnet-4-6",
        system_prompt="<sys-prompt>",
        compile_settings_path=None,
        repair_mode=True,
        use_descriptor=False,
    )
    assert "./mellea-fy-repair" in argv[-1]


# --- Live: real API call (skipped by default) -------------------------------


@pytest.mark.live
def test_live_emit_descriptor_against_sentry_find_bugs(surface: dict[str, Any]) -> None:
    """Real LLM call against the sentry-find-bugs spec.

    Skipped by default. Enable with::

        source /tmp/anthropic-litellm.env
        pytest tests/mellea_skills_compiler/compile/test_descriptor_emission.py \\
            -m live --run-live -q

    Asserts only the smoke-level guarantees (parse + descriptor present) so
    we don't make this brittle to LLM run-to-run variability.
    """
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        pytest.skip(
            "live test requires ANTHROPIC_AUTH_TOKEN (LiteLLM proxy) or ANTHROPIC_API_KEY"
        )
    if not SENTRY_SPEC_PATH.is_file():
        pytest.skip(f"sentry-find-bugs spec missing at {SENTRY_SPEC_PATH}")

    from mellea_skills_compiler.compile.descriptor_emission import emit_descriptor

    spec_text = SENTRY_SPEC_PATH.read_text(encoding="utf-8")
    result = emit_descriptor(spec_text, surface)
    assert result.parse_error is None, result.parse_error
    assert result.descriptor is not None
    assert result.descriptor.get("skill", {}).get("name")
    # Reasoning should be present given the locked-in prompt.
    assert result.reasoning_text is not None and len(result.reasoning_text) > 0
    # Usage telemetry must be populated.
    assert result.usage["input_tokens"] > 0
    assert result.usage["output_tokens"] > 0
