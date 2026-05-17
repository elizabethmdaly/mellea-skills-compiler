"""Top-level test config for the `mellea_skills_compiler` test tree.

Registers shared markers. Currently:

- ``live``: tests that exercise the real `mellea-skills compile` end-to-end
  against a skill in `skills/`. Requires `.venv-spike` Python, mellea +
  anthropic installed, and LiteLLM credentials available at
  `/tmp/anthropic-litellm.env`. Skipped unless ``--run-live`` is passed.

Mirrors the ``network`` marker pattern in
`tests/mellea_skills_compiler/cache/conftest.py`.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: hits real LLM endpoints / runs the full compile end-to-end; "
        "skipped unless --run-live is given.",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    # Guard against duplicate registration when a child conftest also adds
    # the option (pytest will raise ValueError otherwise).
    try:
        parser.addoption(
            "--run-live",
            action="store_true",
            default=False,
            help="Run @pytest.mark.live tests (real LLM calls; ~minutes per test).",
        )
    except ValueError:
        pass


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live to hit real LLM endpoints")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
