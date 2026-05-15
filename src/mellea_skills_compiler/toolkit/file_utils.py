import importlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from rich.console import Console

from mellea_skills_compiler.toolkit.logging import configure_logger


console = Console()
LOGGER = configure_logger()


# ── SKILL.md parser ────────────────────────────────────────────────
def parse_spec_file(spec_path: Path) -> dict:
    """Parse a SKILL.md file into frontmatter dict + markdown body."""
    text = spec_path.read_text()

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        return {"frontmatter": {}, "body": text, "path": str(spec_path)}

    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()

    # Normalize allowed-tools: string → list
    tools_raw = frontmatter.get("allowed-tools", [])
    if isinstance(tools_raw, str):
        if "," in tools_raw:
            frontmatter["allowed-tools"] = [
                t.strip() for t in tools_raw.split(",") if t.strip()
            ]
        else:
            frontmatter["allowed-tools"] = tools_raw.split()
    elif not isinstance(tools_raw, list):
        frontmatter["allowed-tools"] = []

    # OpenClaw convention: tools in metadata.openclaw.requires.bins/anyBins
    openclaw_req = (
        frontmatter.get("metadata", {}).get("openclaw", {}).get("requires", {})
    )
    extra_tools = openclaw_req.get("bins", []) + openclaw_req.get("anyBins", [])
    existing = set(frontmatter.get("allowed-tools", []))
    for t in extra_tools:
        if t not in existing:
            frontmatter.setdefault("allowed-tools", []).append(t)
            existing.add(t)

    return {"frontmatter": frontmatter, "body": body, "path": str(spec_path)}


_ENTRY_SIGNATURE_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _entry_name_from_melleafy(pipeline_dir: Path) -> Optional[str]:
    """Extract the entry function name from `<pkg>/melleafy.json:entry_signature`.

    Returns the leading identifier of `entry_signature` (e.g. `"run_pipeline"`
    from `"run_pipeline(session_id: str, ...) -> AssessmentReport"`), or None
    if the manifest is absent, unreadable, missing the field, or malformed.
    Step 5 writes this field; the manifest is the authoritative source of
    truth for which `run_*` function is the entry point.
    """
    manifest = pipeline_dir / "melleafy.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sig = data.get("entry_signature")
    if not isinstance(sig, str):
        return None
    match = _ENTRY_SIGNATURE_NAME_RE.match(sig)
    return match.group(1) if match else None


# Dynamically import the pipeline from the skill directory
def load_skill_pipeline(pipeline_dir: Path):

    # Add parent directory to sys.path to set PYTHONPAYH
    sys.path.insert(0, str(pipeline_dir.parent))

    try:
        skill_pipeline = importlib.import_module(f"{pipeline_dir.name}.pipeline")
    except ModuleNotFoundError as e:
        raise Exception(
            f"Error: The `pipeline.py` module is missing from the pipeline directory - {pipeline_dir}"
        )
    finally:
        # Remove parent directory from sys.path
        sys.path.pop(0)

    module_name = skill_pipeline.__name__

    # Tier 1 — Authoritative: melleafy.json:entry_signature names the entry.
    # The deterministic `dir()` scan below sorts alphabetically and can pick
    # the wrong public `run_*` helper when pipeline.py defines several (e.g.
    # `run_phase_2_gap_analysis` sorts before `run_pipeline`). The manifest
    # field, written by Step 5 from the typed pipeline signature, removes
    # that ambiguity.
    manifest_name = _entry_name_from_melleafy(pipeline_dir)
    if manifest_name is not None:
        fn = getattr(skill_pipeline, manifest_name, None)
        if callable(fn) and getattr(fn, "__module__", None) == module_name:
            return fn

    # Tier 2 — Convention: every skill SHOULD have `run_pipeline` as the
    # canonical entry name (per mellea-fy-generate.md). Try that next so
    # older packages without melleafy.json still resolve correctly.
    fn = getattr(skill_pipeline, "run_pipeline", None)
    if callable(fn) and getattr(fn, "__module__", None) == module_name:
        return fn

    # Tier 3 — Legacy fallback: alphabetical `run_*` scan, preferring
    # locally-defined functions over imported ones. Preserved for
    # backward compat with packages that don't follow either convention.
    run_fn = None
    imported_run_fn = None
    for attr_name in dir(skill_pipeline):
        if attr_name.startswith("run_") and callable(
            getattr(skill_pipeline, attr_name)
        ):
            fn = getattr(skill_pipeline, attr_name)
            if getattr(fn, "__module__", None) == module_name:
                run_fn = fn
                break
            elif imported_run_fn is None:
                imported_run_fn = fn

    if run_fn is None:
        run_fn = imported_run_fn

    if run_fn is None:
        raise Exception("No run_* function found in %s", pipeline_dir)

    return run_fn


# ── Load Fixtures ────────────────────────────────────────────────
def load_fixtures(pipeline_dir: Path) -> List[Dict]:
    """Load fixtures from the skill's fixtures/ directory.

    Supports two conventions:
    - ALL_FIXTURES: list of factory functions returning (inputs, id, description)
    - FIXTURES: list of dicts with 'id' and 'context' keys
    """

    """Find fixtures/ directly under skill_dir, or inside a *_mellea/ subdirectory."""
    fixtures_dir = pipeline_dir / "fixtures"
    if not fixtures_dir.is_dir():
        for child in pipeline_dir.iterdir():
            if child.is_dir() and child.name.endswith("_mellea"):
                fixtures_dir = child / "fixtures"
                if fixtures_dir.is_dir():
                    break

    if not fixtures_dir.is_dir():  # or not (fixtures_dir / "__init__.py").exists()
        raise Exception(
            f"No fixtures directory found in {pipeline_dir}. Please run `mellea-skills compile` to compile the skill first."
        )

    fixtures = None
    sys.path.insert(0, str(fixtures_dir.parent))
    try:
        mod = importlib.import_module(fixtures_dir.name)
        # Convention 1: FIXTURES list of dicts (e.g., test_fixtures.py)
        if hasattr(mod, "FIXTURES"):
            fixtures = mod.FIXTURES

        # Convention 2: ALL_FIXTURES list of factory functions (mellea-fy generated)
        if hasattr(mod, "ALL_FIXTURES"):
            fixtures = []
            for factory in mod.ALL_FIXTURES:
                inputs, fixture_id, description = factory()
                fixtures.append(
                    {
                        "id": fixture_id,
                        "context": inputs,
                        "description": description,
                    }
                )
    except ImportError:
        raise Exception(
            f"The `__init__.py` module is missing from the fixture directory - {fixtures_dir}"
        )
    finally:
        sys.path.pop(0)

    if not fixtures:
        raise Exception(f"No valid FIXTURES found in {fixtures_dir}")
    else:
        return fixtures
