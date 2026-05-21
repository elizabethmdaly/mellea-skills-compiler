"""Retirement-guard test for the (retired) R-SEM-MODALITY-APPROVAL rule.

The rule was retired 2026-05-20 because the schema field it enforced
(`skill.classification.requires_approval_gates`) was itself the
problem — a redundant claim duplicating a structural fact (whether the
pipeline contains a `human_approval` operator). The audit registry
keeps this entry as a regression-guard so the field can't quietly
reappear.

See the registry's `r-sem-modality-approval` entry's `motivation` for
the full incident history that led to retirement.
"""
from __future__ import annotations

import json
from importlib.resources import files

import pytest


def test_field_stays_retired():
    """C-FIELD-STAYS-RETIRED: `requires_approval_gates` MUST NOT exist
    anywhere in the descriptor schema.

    If this fails, the field has been re-added. Before silencing the
    test, the registry maintainer MUST update the retirement motivation
    in the registry entry with a justification, AND add the missing
    directive-coverage check that was the original failure mode.
    """
    schema_text = (
        files("mellea_skills_compiler.descriptor.schemas")
        .joinpath("descriptor.schema.v0.3.json")
        .read_text(encoding="utf-8")
    )
    schema = json.loads(schema_text)

    occurrences: list[str] = []

    def _walk(obj, path: str = "$"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "requires_approval_gates":
                    occurrences.append(f"{path}.{k}")
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(schema)

    if occurrences:
        pytest.fail(
            "C-FIELD-STAYS-RETIRED failed: the retired field "
            "`requires_approval_gates` reappeared in "
            "descriptor.schema.v0.3.json at:\n  - "
            + "\n  - ".join(occurrences)
            + "\n\nBefore re-adding the field, update the audit-registry "
            "entry `r-sem-modality-approval`'s `motivation` with the "
            "justification, and add a coherence check that requires a "
            "slash-command directive explaining the rule's implication to "
            "the LLM. The original rule was retired because the LLM was "
            "graded on the field without any documented stance about it — "
            "do not recreate that drift."
        )
