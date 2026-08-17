"""Smoke check must fail on mellea DeprecationWarnings at runtime.

Compat report §3.4: this is what would have caught the ``genslot`` shim
usage before it silently reverted to fallback data. The mechanism has to
fire ONLY on mellea-originated warnings — app deps that emit DeprecationWarnings
for unrelated reasons must not be conflated.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

from mellea_skills_compiler.compile.smoke_check import (
    _mellea_deprecation_warnings,
    _run_one_fixture,
)
from mellea_skills_compiler.models import Fixture


def _warning_record(message, filename="/pkgs/mellea/thing.py", lineno=42, category=DeprecationWarning):
    return SimpleNamespace(
        message=message,
        category=category,
        filename=filename,
        lineno=lineno,
    )


class TestMelleaDeprecationFilter:
    def test_reports_deprecation_from_mellea_path(self):
        w = _warning_record("genslot is deprecated, use genstub")
        assert len(_mellea_deprecation_warnings([w])) == 1

    def test_ignores_deprecation_from_unrelated_lib(self):
        w = _warning_record("something", filename="/pkgs/other_lib/x.py")
        assert _mellea_deprecation_warnings([w]) == []

    def test_matches_by_message_text_when_path_ambiguous(self):
        w = _warning_record(
            "mellea.stdlib.components.genslot is deprecated",
            filename="/some/wrapper.py",
        )
        assert len(_mellea_deprecation_warnings([w])) == 1


class TestRunOneFixtureFailsOnMelleaDeprecation:
    def test_fixture_that_triggers_mellea_deprecation_fails_smoke(self):
        fixture = Fixture(id="fx1", context={}, description="")

        def pipeline_fn(**kwargs):
            warnings.warn(
                "mellea.stdlib.components.genslot is deprecated",
                DeprecationWarning,
                stacklevel=1,
            )

        result = _run_one_fixture(pipeline_fn, fixture)
        assert result.verdict == "failed"
        assert "DeprecationWarning" in (result.failure_message or "")

    def test_fixture_without_deprecation_passes(self):
        fixture = Fixture(id="fx1", context={}, description="")

        def pipeline_fn(**kwargs):
            return None

        result = _run_one_fixture(pipeline_fn, fixture)
        assert result.verdict == "passed"
