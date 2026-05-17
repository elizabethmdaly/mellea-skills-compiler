"""Local pytest config for the compile-step test suite.

Registers the ``live`` marker (real LLM call against the LiteLLM proxy) and
skips it by default. Run live tests explicitly with::

    .venv-spike/bin/python -m pytest tests/mellea_skills_compiler/compile/ -m live --run-live
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: hits the real Anthropic API (via the LiteLLM proxy); skipped unless --run-live is given.",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    # Use --run-live; harmless if another conftest already registered the option.
    try:
        parser.addoption(
            "--run-live",
            action="store_true",
            default=False,
            help="Run @pytest.mark.live tests (real Anthropic API calls).",
        )
    except ValueError:
        # Already registered (e.g. by a parent conftest). Ignore.
        pass


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live", default=False):
        return
    skip_live = pytest.mark.skip(
        reason="live LLM test; pass --run-live and set ANTHROPIC_AUTH_TOKEN to enable"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
