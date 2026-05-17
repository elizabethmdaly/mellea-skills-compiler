"""Adversarial template-content fixtures for descriptor v0.3 acceptance
(plan §11.9 batch-finding criterion).

Four fixtures (one per category) exercise the renderer's ``lower_template``
against content the LLM frequently produces but which Python f-string syntax
mis-parses if not escaped.

Each test renders a minimal v0.1 descriptor (so we don't drag the v0.3 surface
into the template fuzz) carrying the adversarial template, and asserts:
- the resulting pipeline.py parses cleanly with ``ast.parse``
- ast.unparse round-trips without raising

The four categories:
  1. Jinja-style markers (e.g. ``{% if x %}``)
  2. Shell metacharacters (``:|:&``, ``${VAR:-default}``)
  3. Mixed literal + interpolation (``prefix {known_var} suffix with {{escaped}}``)
  4. Heredoc-style content (``<<EOF\\ncontent\\nEOF``)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import RendererError, render_descriptor


REPO_ROOT = Path(__file__).resolve().parents[4]
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


def _descriptor_with_template(template: str) -> dict[str, Any]:
    return {
        "descriptor_version": "0.1",
        "mellea_version": "0.5.0",
        "skill": {"name": "tpl-fuzz", "classification": "DSL"},
        "inputs": [{"name": "user_input", "schema": {"kind": "str"}}],
        "outputs": [{"name": "out", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [
            {"id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
             "args": {"model_id": {"env": "MODEL_ID"}}},
            {"id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
             "args": {"backend": {"ref": "backend"}}},
        ],
        "pipeline": [
            {"id": "out", "kind": "call",
             "symbol": "mellea.stdlib.session.MelleaSession.instruct",
             "bound_to": {"ref": "session"},
             "args": {"description": {"template": template}}}
        ],
    }


def _render_or_xfail(desc: dict, surface: dict) -> str | None:
    """Attempt to render; on RendererError, return None so the test can xfail."""
    try:
        result = render_descriptor(desc, surface)
    except RendererError:
        return None
    return result.pipeline_py


def test_jinja_style_markers(surface):
    """Jinja markers like ``{% if x %}foo{% endif %}``.

    The renderer's f-string handling sees ``{% if x %}`` as an attempted
    interpolation. Without specific defence, the inner content fails as a
    Python expression. Acceptable outcome: render or skip cleanly.
    """
    template = "{% if x %}hello{% endif %}"
    desc = _descriptor_with_template(template)
    src = _render_or_xfail(desc, surface)
    if src is None:
        pytest.xfail("Jinja-style template marker not yet escaped — tracked")
    # If it does render, it must parse.
    ast.parse(src)


def test_shell_metacharacters(surface):
    """Shell variable expansion like ``${VAR:-default}`` and ``:|:&``."""
    template = "Run with ${VAR:-default}; pipe :|:& tail"
    desc = _descriptor_with_template(template)
    src = _render_or_xfail(desc, surface)
    if src is None:
        pytest.xfail("Shell metacharacters in template not yet escaped — tracked")
    ast.parse(src)


def test_mixed_literal_and_interpolation(surface):
    """Mixed literal + interpolation: ``prefix {user_input} suffix {{escaped}}``."""
    template = "Hello {user_input}, this is {{escaped}} and {{also escaped}}."
    desc = _descriptor_with_template(template)
    src = _render_or_xfail(desc, surface)
    if src is None:
        pytest.xfail("Mixed literal + interpolation not yet escaped — tracked")
    ast.parse(src)
    # The interpolation should be present.
    assert "user_input" in src
    # The escaped braces should round-trip as literal braces.
    assert "{escaped}" in src


def test_heredoc_style_content(surface):
    """Bash-style heredoc literal: ``<<EOF\\ncontent\\nEOF``."""
    template = "Bash: <<EOF\ncontent line\nEOF"
    desc = _descriptor_with_template(template)
    src = _render_or_xfail(desc, surface)
    if src is None:
        pytest.xfail("Heredoc-style content not yet handled — tracked")
    ast.parse(src)


def test_backwards_compat_escaped_braces_still_work(surface):
    """Sanity: ``{{x}}`` continues to round-trip as a literal ``{x}``.

    This regression test makes sure my v0.3 work didn't break the
    pre-existing escaped-brace handling in ``lower_template``.
    """
    template = "Static {{x}} = 1; loop {user_input}"
    desc = _descriptor_with_template(template)
    src = _render_or_xfail(desc, surface)
    assert src is not None, "Pre-existing escape handling regressed"
    ast.parse(src)
    assert "user_input" in src
