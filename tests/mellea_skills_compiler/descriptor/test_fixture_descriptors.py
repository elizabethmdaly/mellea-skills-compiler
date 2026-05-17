"""End-to-end tests against the three Phase 0.2 hand-authored descriptors.

Two things this suite checks:

1. The three descriptors, **unmodified**, validate against the v0.1 schema
   (acceptance criterion 1, full extent).
2. Mechanical RFC-documented migrations (``_notes -> notes`` from §3.11;
   classification preserved; no other shape changes for the simple case)
   produce descriptors that validate against v0.2.

The migrated copies are written to
``tests/.../descriptor/fixtures/v02/`` so a human can diff them and so
downstream renderer tests have a known-good v0.2 corpus.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mellea_skills_compiler.descriptor import validate


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "descriptors"
)
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "v02"

DESCRIPTOR_FILES = (
    "sentry-find-bugs.descriptor.json",
    "security-review.descriptor.json",
    "security-engineer.descriptor.json",
)


@pytest.fixture(scope="module")
def surface() -> dict:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


# --- v0.1 unmodified -------------------------------------------------------- #


@pytest.mark.parametrize("descriptor_name", DESCRIPTOR_FILES)
def test_v01_descriptors_validate_unchanged(descriptor_name, surface):
    """Phase 0.2 descriptors must validate against v0.1 as-authored."""
    path = SOURCE_DIR / descriptor_name
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    report = validate(descriptor, schema_version="0.1", surface=surface)
    error_only = [e for e in report.errors if e.severity == "error"]
    assert report.valid, f"{descriptor_name}: {error_only}"


def test_v01_source_descriptors_are_unmodified():
    """Acceptance: the source-of-truth descriptors must not be modified."""
    # Sanity check the file actually still has '_notes' in the two we expect.
    sr = json.loads(
        (SOURCE_DIR / "security-review.descriptor.json").read_text(encoding="utf-8")
    )
    assert "_notes" in sr, (
        "security-review.descriptor.json was modified — this validator "
        "must NEVER mutate the source-of-truth Phase 0.2 descriptors."
    )
    se = json.loads(
        (SOURCE_DIR / "security-engineer.descriptor.json").read_text(encoding="utf-8")
    )
    assert "_notes" in se


# --- v0.2 migration --------------------------------------------------------- #


def _migrate_v01_to_v02(descriptor: dict, name: str) -> dict:
    """RFC-documented mechanical migration.

    Applies only the changes described in the RFC's "migration impact"
    subsections (§3.2, §3.3, §3.6, §3.7, §3.11). Other changes (`filter`
    adoption, `on_parse_failure` adoption, etc.) are quality improvements,
    not migrations, and the RFC says they don't need to ship in the
    migrated copy.
    """
    out = copy.deepcopy(descriptor)
    out["descriptor_version"] = "0.2"

    # §3.11 _notes -> notes.
    if "_notes" in out:
        out["notes"] = out.pop("_notes")

    # §3.6 capabilities — applies to security-review and security-engineer.
    if name == "security-review.descriptor.json":
        out["capabilities"] = {
            "tools": {"allowed": ["Read", "Grep", "Glob", "Bash", "Task"], "denied": []}
        }
    elif name == "security-engineer.descriptor.json":
        out["capabilities"] = {
            "tools": {"allowed": ["Read", "Write", "Edit", "Bash"], "denied": []}
        }

    # §3.3 — state.scope default is "function"; no edits needed (default
    # applies). §3.7 — `value` already legal as an arg kind; sentry/security-
    # review keep env-driven model_id, no edits required.

    return out


@pytest.mark.parametrize("descriptor_name", DESCRIPTOR_FILES)
def test_migrated_descriptors_validate_against_v02(
    descriptor_name, surface, fixtures_dir
):
    """Acceptance criterion 4: migrated copies validate against v0.2."""
    src = json.loads(
        (SOURCE_DIR / descriptor_name).read_text(encoding="utf-8")
    )
    migrated = _migrate_v01_to_v02(src, descriptor_name)

    # Persist the migrated copy for downstream consumers + human review.
    target = fixtures_dir / descriptor_name
    target.write_text(json.dumps(migrated, indent=2), encoding="utf-8")

    report = validate(migrated, schema_version="0.2", surface=surface)
    error_only = [e for e in report.errors if e.severity == "error"]
    assert report.valid, f"{descriptor_name}: {error_only}"


def test_security_review_migration_drops_underscore_notes(surface):
    src = json.loads(
        (SOURCE_DIR / "security-review.descriptor.json").read_text(encoding="utf-8")
    )
    migrated = _migrate_v01_to_v02(src, "security-review.descriptor.json")
    assert "_notes" not in migrated
    assert "notes" in migrated
    assert isinstance(migrated["notes"], list)


def test_security_engineer_migration_adds_capabilities(surface):
    src = json.loads(
        (SOURCE_DIR / "security-engineer.descriptor.json").read_text(encoding="utf-8")
    )
    migrated = _migrate_v01_to_v02(src, "security-engineer.descriptor.json")
    assert "capabilities" in migrated
    assert "Read" in migrated["capabilities"]["tools"]["allowed"]


def test_sentry_migration_is_minimal(surface):
    """sentry-find-bugs is the simplest migration: only the version bump."""
    src = json.loads(
        (SOURCE_DIR / "sentry-find-bugs.descriptor.json").read_text(encoding="utf-8")
    )
    migrated = _migrate_v01_to_v02(src, "sentry-find-bugs.descriptor.json")
    assert migrated["descriptor_version"] == "0.2"
    # No _notes in the original, so no notes in the migrated.
    assert "notes" not in migrated
    assert "capabilities" not in migrated


def test_migrated_v02_descriptor_rejects_under_v01(surface):
    """A descriptor with descriptor_version='0.2' must NOT validate against the
    v0.1 schema."""
    src = json.loads(
        (SOURCE_DIR / "sentry-find-bugs.descriptor.json").read_text(encoding="utf-8")
    )
    migrated = _migrate_v01_to_v02(src, "sentry-find-bugs.descriptor.json")
    report = validate(migrated, schema_version="0.1", surface=surface)
    assert not report.valid
