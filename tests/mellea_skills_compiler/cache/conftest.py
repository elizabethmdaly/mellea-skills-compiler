"""Local pytest config for cache tests.

Registers the ``network`` marker and skips network-marked tests by default.
Run them explicitly with::

    pytest tests/mellea_skills_compiler/cache/ -m network --run-network
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: hits the real docs.mellea.ai host; skipped unless --run-network is given.",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run @pytest.mark.network tests (real HTTP to docs.mellea.ai).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network to hit docs.mellea.ai")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
