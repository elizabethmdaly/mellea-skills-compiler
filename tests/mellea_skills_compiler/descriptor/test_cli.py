"""Smoke tests for the descriptor CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mellea_skills_compiler.descriptor.cli import main


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "descriptors"
)
SURFACE_PATH = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
)


def test_cli_validates_known_good_descriptor(capsys):
    rc = main(
        [
            "validate",
            str(SOURCE_DIR / "sentry-find-bugs.descriptor.json"),
            "--schema-version",
            "0.1",
            "--surface",
            str(SURFACE_PATH),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_cli_reports_missing_file(capsys):
    rc = main(["validate", "/no/such/path.json", "--schema-version", "0.2"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_reports_validation_errors(tmp_path, capsys):
    bad = tmp_path / "bad.descriptor.json"
    bad.write_text(json.dumps({"descriptor_version": "0.2"}), encoding="utf-8")
    rc = main(["validate", str(bad), "--schema-version", "0.2"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "valid=False" in captured.err


def test_cli_rejects_unknown_schema_version(capsys):
    """argparse should reject schema-version outside choices."""
    with pytest.raises(SystemExit):
        main(
            [
                "validate",
                str(SOURCE_DIR / "sentry-find-bugs.descriptor.json"),
                "--schema-version",
                "9.9",
            ]
        )


def test_cli_works_without_surface(tmp_path, capsys):
    """Without --surface, symbol resolution is skipped but everything else runs."""
    descriptor = {
        "descriptor_version": "0.2",
        "mellea_version": "0.5.0",
        "skill": {"name": "demo", "classification": "DSL"},
        "inputs": [{"name": "doc", "schema": {"kind": "str"}}],
        "outputs": [{"name": "result", "schema": {"kind": "str"}}],
        "schemas": {},
        "state": [],
        "pipeline": [],
    }
    descriptor_path = tmp_path / "demo.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    rc = main(["validate", str(descriptor_path), "--schema-version", "0.2"])
    assert rc == 0
