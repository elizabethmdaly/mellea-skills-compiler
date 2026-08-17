"""Audit-trail plugin must correlate Guardian verdicts by generation_id.

The pre-0.7 code sliced the tail of ``guardian.all_verdicts`` positionally,
which is race-prone under mellea 0.7 parallel sampling (PR #1175). The
fix indexes verdicts by generation_id inside the Guardian plugin and
looks them up by id in the audit plugin.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mellea_skills_compiler.enums import GovernanceTaxonomy, GuardianScore, HookStage
from mellea_skills_compiler.models import GuardianVerdict, NexusRisk
from mellea_skills_compiler.plugins.audit import AuditTrailPlugin
from mellea_skills_compiler.plugins.guardian import GuardianAuditPlugin


@pytest.fixture
def audit_setup(tmp_path: Path):
    risks = [
        NexusRisk(
            name="harm",
            description="",
            guardian_prompt="harm",
            source="test",
            is_native=True,
            taxonomy=GovernanceTaxonomy.IBM_GRANITE_GUARDIAN,
        )
    ]
    gp = GuardianAuditPlugin(risks=risks)
    ap = AuditTrailPlugin(log_path=tmp_path / "audit.jsonl", guardian_plugin=gp)
    return gp, ap


def test_lookup_uses_id_map_when_generation_id_present(audit_setup):
    guardian_plugin, audit_plugin = audit_setup
    v_target = GuardianVerdict(risk="harm", label=GuardianScore.YES, raw_output="", hook_stage=HookStage.POST)
    v_other = GuardianVerdict(risk="harm", label=GuardianScore.NO, raw_output="", hook_stage=HookStage.POST)
    guardian_plugin._record_verdicts([v_target], generation_id="gen-target")
    guardian_plugin._record_verdicts([v_other], generation_id="gen-other")

    hit = audit_plugin._lookup_verdicts_by_generation_id("gen-target")
    assert hit == [v_target]


def test_lookup_falls_back_to_positional_when_id_missing(audit_setup):
    guardian_plugin, audit_plugin = audit_setup
    v = GuardianVerdict(risk="harm", label=GuardianScore.NO, raw_output="", hook_stage=HookStage.POST)
    guardian_plugin._record_verdicts([v], generation_id=None)
    result = audit_plugin._lookup_verdicts_by_generation_id(None)
    assert result == [v]
