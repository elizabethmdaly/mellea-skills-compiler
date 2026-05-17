"""Lightweight CLI smoke tests — verify argparse wiring and exit codes.

The heavy "actually walk Mellea" path is covered by ``test_integration.py``;
these tests use a tiny on-disk fixture package via ``--root``.
"""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from mellea_skills_compiler.mellea_surface.cli import main


@pytest.fixture
def ondisk_package(tmp_path: Path, monkeypatch):
    pkg_name = "msccli_fixture"
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        textwrap.dedent(
            '''\
            """Tiny fixture."""

            def f(x: int) -> int:
                """Echo."""
                return x

            __all__ = ["f"]
            '''
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield pkg_name
    for mod_name in list(sys.modules):
        if mod_name == pkg_name or mod_name.startswith(f"{pkg_name}."):
            del sys.modules[mod_name]


def test_build_emits_surface_json(tmp_path: Path, ondisk_package: str):
    out = tmp_path / "surface.json"
    rc = main(["build", "--out", str(out), "--root", ondisk_package])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "modules" in data
    assert ondisk_package in data["modules"]
    assert "surface_fingerprint" in data


def test_build_is_byte_stable(tmp_path: Path, ondisk_package: str):
    out1 = tmp_path / "s1.json"
    out2 = tmp_path / "s2.json"
    main(["build", "--out", str(out1), "--root", ondisk_package])
    main(["build", "--out", str(out2), "--root", ondisk_package])
    assert out1.read_bytes() == out2.read_bytes()


def test_coverage_with_local_inputs(tmp_path: Path, ondisk_package: str):
    surface_path = tmp_path / "surface.json"
    main(["build", "--out", str(surface_path), "--root", ondisk_package])

    # Hand-roll a tiny llms.txt that points at our fixture module so coverage
    # passes the 95% gate.
    llms_txt = tmp_path / "llms.txt"
    llms_txt.write_text(
        f"- [{ondisk_package}](https://docs.mellea.ai/api/{ondisk_package}.md): Tiny.\n"
    )

    report = tmp_path / "coverage.md"
    rc = main(
        [
            "coverage",
            "--surface",
            str(surface_path),
            "--llms-txt",
            str(llms_txt),
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    assert report.exists()
    text = report.read_text()
    assert "Overall coverage: 100.0%" in text
    assert "Gate: PASS" in text


def test_coverage_fail_returns_nonzero(tmp_path: Path, ondisk_package: str):
    surface_path = tmp_path / "surface.json"
    main(["build", "--out", str(surface_path), "--root", ondisk_package])

    # llms.txt lists a module that isn't in the surface => 0% coverage.
    llms_txt = tmp_path / "llms.txt"
    llms_txt.write_text(
        "- [nonexistent.module](https://docs.mellea.ai/api/nope.md): Gone.\n"
    )

    report = tmp_path / "coverage.md"
    rc = main(
        [
            "coverage",
            "--surface",
            str(surface_path),
            "--llms-txt",
            str(llms_txt),
            "--report",
            str(report),
        ]
    )
    assert rc == 1
