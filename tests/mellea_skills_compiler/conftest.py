"""Top-level test config for the `mellea_skills_compiler` test tree.

Registers shared markers. Currently:

- ``live``: tests that exercise the real `mellea-skills compile` end-to-end
  against a skill in `skills/`. Requires `.venv-spike` Python, mellea +
  anthropic installed, and LiteLLM credentials available at
  `/tmp/anthropic-litellm.env`. Skipped unless ``--run-live`` is passed.

Mirrors the ``network`` marker pattern in
`tests/mellea_skills_compiler/cache/conftest.py`.

Also exposes shared path helpers for the in-repo data fixtures:

- :data:`SURFACE_PATH` — bundled introspected Mellea surface JSON.
- :data:`SCHEMA_DOC_PATH` — bundled descriptor schema markdown.
- :data:`DESCRIPTOR_FIXTURE_DIR` — hand-authored Phase 0.2 descriptors
  used as renderer / validator inputs (under ``tests/fixtures/descriptors/``).
"""

from __future__ import annotations

from importlib.resources import files as _resource_files
from pathlib import Path

import pytest


# --- Shared fixture paths ---------------------------------------------------
#
# Runtime defaults ship with the package under ``src/...data/``; we resolve
# them through :mod:`importlib.resources` so tests don't depend on the
# (gitignored) developer ``melleafy-handoff/`` tree.

SURFACE_PATH: Path = Path(
    str(_resource_files("mellea_skills_compiler.mellea_surface") / "data" / "surface_0.5.0.json")
)
SCHEMA_DOC_PATH: Path = Path(
    str(_resource_files("mellea_skills_compiler.descriptor") / "data" / "descriptor-schema-v0.md")
)

# Descriptor fixtures live under ``tests/fixtures/descriptors/`` so they don't
# ship in wheels. ``parents[2]`` = the repo's ``tests/`` directory.
_TESTS_DIR = Path(__file__).resolve().parents[1]
DESCRIPTOR_FIXTURE_DIR: Path = _TESTS_DIR / "fixtures" / "descriptors"


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
