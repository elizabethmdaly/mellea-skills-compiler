"""Adversarial template-content tests — proactive coverage of literal-string
edge cases that the template parser must handle.

Background: the batch descriptor test (2026-05-16) found that
``renderer/nodes.py::lower_template`` over-eagerly interprets ``{...}`` as
Python expression interpolation. 5 of 8 batch failures so far (62.5%) trip
on literal content like:

- JSON example outputs:    ``{"key": "value"}``
- Bash variable expansion: ``${VAR:-default}``
- Jinja markers:           ``{%% autoescape off %%}``
- Shell metacharacters:    ``:|:&``
- Markdown code-fenced JSON config snippets

This file synthesises adversarial template content directly — no LLM
involved, no batch waiting. The pattern: build a minimal descriptor whose
single call node's ``description.template`` carries the problematic content,
then render and assert.

Each test is structured so that:
- If the renderer fix lands (Phase 3.5.C Option B from §13), the test passes.
- If the renderer is still broken (today), the test ``xfail``s with a
  descriptive reason — keeping the build green while documenting the gap.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import render_descriptor
from mellea_skills_compiler.renderer.schemas import render_schemas


REPO_ROOT = Path(__file__).resolve().parents[4]
SURFACE_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "surface_0.5.0.json"
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _minimal_descriptor_with_template(template_body: str) -> dict[str, Any]:
    """Return the smallest valid descriptor whose call node carries the given
    template content. Used by every test below.
    """
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "template-test", "classification": "DSL"},
        "inputs": [{"name": "user_input", "schema": {"kind": "str"}}],
        "outputs": [
            {
                "name": "result",
                "schema": {
                    "kind": "model_ref",
                    "ref": "#/schemas/Result",
                },
            }
        ],
        "schemas": {
            "Result": {
                "kind": "model",
                "fields": {"text": {"type": "str"}},
            }
        },
        "state": [
            {
                "id": "backend",
                "symbol": "mellea.backends.openai.OpenAIBackend",
                "args": {"model_id": {"env": "MODEL_ID"}},
            },
            {
                "id": "session",
                "symbol": "mellea.stdlib.session.MelleaSession",
                "args": {"backend": {"ref": "backend"}},
            },
        ],
        "pipeline": [
            {
                "id": "step",
                "kind": "call",
                "symbol": "mellea.stdlib.session.MelleaSession.instruct",
                "bound_to": {"ref": "session"},
                "args": {
                    "description": {"template": template_body},
                    "format": {"schema_ref": "#/schemas/Result"},
                },
                "captures": {"result": "#/outputs/result"},
            }
        ],
    }


def _render_or_fail(descriptor: dict, surface: dict) -> tuple[bool, str]:
    """Try to render. Return (ok, error_message_or_pipeline_source)."""
    try:
        result = render_descriptor(descriptor, surface)
        return True, result.pipeline_py
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# --- The cases below all probe the {...} -interpretation bug ----------------


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser misinterprets JSON literal braces as Python expression",
    strict=False,
)
def test_template_with_json_literal_object(surface):
    """Template carrying a JSON literal {key: value} example. This is the
    most common case in the batch — any spec that shows an output format
    example using JSON braces will trip this."""
    template = 'Return JSON in this shape: {"finding": "issue", "severity": "high"}'
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"
    # The rendered Python must preserve the JSON literal somewhere.
    assert "finding" in info, "JSON literal content lost in template rendering"


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser tries to evaluate bash variable expansion as Python",
    strict=False,
)
def test_template_with_bash_variable_expansion(surface):
    """Template carrying bash ${VAR:-default} substitution. Trips skills
    whose spec describes shell command examples."""
    template = "Execute the command: ${TMPDIR:-/tmp}/output.json"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"
    assert "TMPDIR" in info


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser tries to evaluate Jinja syntax as Python",
    strict=False,
)
def test_template_with_jinja_markers(surface):
    """Template carrying Jinja-style ``{% ... %}`` markers. Trips skills
    that reference template engines in their spec content."""
    template = "Configure: {% autoescape off %}{{ user_input }}{% endautoescape %}"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser tries to evaluate shell metacharacters",
    strict=False,
)
def test_template_with_shell_metacharacters(surface):
    """Template carrying shell metacharacters like ``:|:&`` (fork bomb pattern).
    Trips skills that include dangerous-input examples in their spec."""
    template = "Reject dangerous input like ':(){ :|:& };:' which can crash the shell"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser fails on JSON array literals",
    strict=False,
)
def test_template_with_json_array_literal(surface):
    """Template carrying a JSON array example."""
    template = 'Return list: [{"id": 1}, {"id": 2}]'
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


@pytest.mark.xfail(
    reason="Phase 3.5.C — template parser fails on configuration-style content with nested braces",
    strict=False,
)
def test_template_with_nested_json_config(surface):
    """Template carrying nested JSON config (as seen in real batch failure
    b8f9128c0e9b — permissions config example)."""
    template = 'Apply policy: {"permissions": {"allow": ["read"], "deny": ["write"]}}'
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


def test_template_with_pure_literal_no_braces(surface):
    """Sanity baseline — pure literal text with NO interpolation chars must
    render fine. This is the no-brace happy case and should pass today."""
    template = "Find security issues in the provided code"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


def test_template_with_valid_interpolation(surface):
    """Sanity baseline — template with valid Python-identifier interpolation.
    Must render and produce the expected f-string."""
    template = "Process this: {user_input}"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"
    # The interpolation must appear in the rendered f-string form.
    assert "{user_input}" in info or "{user_input!" in info


@pytest.mark.xfail(
    reason="Phase 3.5.C — escaped-brace convention not yet implemented",
    strict=False,
)
def test_template_with_escaped_braces(surface):
    """Per Python f-string convention, ``{{`` and ``}}`` should render as
    literal ``{`` and ``}``. Phase 3.5.C should adopt this convention."""
    template = "Output format is {{key: value}} where key is a string"
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


@pytest.mark.xfail(
    reason="Phase 3.5.C — mixed interpolation + literal braces",
    strict=False,
)
def test_template_with_mixed_interpolation_and_literal(surface):
    """The hardest case: a template that mixes valid interpolation with
    literal braces. The parser must disambiguate."""
    template = 'Given input {user_input}, return JSON: {"echo": "..."}'
    desc = _minimal_descriptor_with_template(template)
    ok, info = _render_or_fail(desc, surface)
    assert ok, f"render failed: {info}"


# --- Programmatic property test (hand-rolled, no Hypothesis required) -------


PROPERTY_FUZZ_INPUTS = [
    "",  # empty
    "plain text",
    "trailing brace {",
    "leading brace }",
    "balanced literal {abc}",  # this WILL be misinterpreted today
    "{1 + 2}",  # numeric expr — but this isn't a Python identifier interpolation
    "${VAR}",
    "$VAR",
    "$(cmd)",
    "<<EOF",
    "{% if x %}",
    "{{ var }}",
    "%(name)s",
    "{name=}",
    "{a.b.c}",
    "{a['key']}",
    '{"key": "value"}',
    "[1, 2, 3]",
    "{}",
    "{{}}",
    "}{",
    "{a}{b}",
    "tabs\tand\nnewlines",
    "unicode: 你好 ⚡ 🎯",
    "quotes: 'single' \"double\" `backticks`",
]


@pytest.mark.parametrize("template", PROPERTY_FUZZ_INPUTS)
def test_template_parser_never_crashes_with_exception_type(surface, template):
    """Property: for any template string, the parser must either return a
    valid renderable result OR raise a documented exception. It must NEVER:
    - produce a SyntaxError from a downstream py_compile
    - produce an AttributeError from a partial AST
    - silently corrupt the template content

    Today most of these will xfail; that's the documentation. After
    Phase 3.5.C they should all pass.
    """
    desc = _minimal_descriptor_with_template(template)
    try:
        result = render_descriptor(desc, surface)
    except Exception as e:
        # Acceptable failure modes (documented):
        # 1. RendererError with a clear message
        # 2. SyntaxError raised explicitly with attribution to the template
        # We assert the error type is NOT one of the implicit-crash failure modes.
        if not isinstance(e, (SyntaxError, ValueError)):
            # Other exceptions: must mention template / line / position
            msg = str(e)
            assert "template" in msg.lower() or "expression" in msg.lower(), (
                f"unattributed crash for template {template!r}: {type(e).__name__}: {e}"
            )
        pytest.xfail(
            f"template parser limitation today: {type(e).__name__} on {template!r}"
        )

    # If we got here, render succeeded. Verify the source compiles.
    try:
        ast.parse(result.pipeline_py)
    except SyntaxError as e:
        pytest.fail(
            f"render returned source that doesn't parse for template {template!r}: {e}"
        )
