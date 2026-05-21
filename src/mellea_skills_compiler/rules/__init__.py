"""Audit-coherence registry — single source of truth mapping each
auditable artifact in the compiler to its three faces: directive (what
the LLM is told), implementation (what runtime actually requires), and
validation (what enforces or tests for correctness).

Drift between the three faces is the root cause of multiple failure
classes observed in real compiles:

  * **Impl ↔ validation drift**: the runtime accepts a call shape but
    the lint flags it as wrong (false positive). Example:
    ``MelleaSession(backend=...)`` — valid Python, but stdlib-arity
    treated POSITIONAL_OR_KEYWORD params as positional-only.
  * **Directive ↔ validation drift**: the directive doc describes one
    call pattern, the validation enforces a different one.
  * **Directive ↔ impl drift**: the directive points at an artifact
    (e.g. ``mellea_api_ref.json``) that is incomplete relative to what
    the runtime actually provides.

Scope (v0.2): beyond contract-enforcing rules, the registry now also
covers mechanisms that operate on top of rules' failures — e.g.
repair-prompt enrichments like D2 (closest-match suggestions).
Mechanism entries have the same three faces but the validation face
is correctness tests rather than a gate.

Public surface:

  * :func:`load_registry` — load and schema-validate the on-disk
    registry; returns the parsed dict.
  * :func:`get_rule` — load the registry and return one entry by id.
    Name retained for v0.1 compatibility; entries may now be either
    rules or mechanisms.

The registry itself is at ``registry.json`` (alongside this file); its
schema is at ``registry.schema.json``.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Optional


_REGISTRY_RESOURCE = "registry.json"
_SCHEMA_RESOURCE = "registry.schema.json"


def load_registry() -> dict[str, Any]:
    """Load the on-disk registry and validate it against its schema.

    Raises:
        ValueError: if the registry doesn't conform to its schema.
        FileNotFoundError: if either file is missing from the package
            resources (indicates a packaging bug).
    """
    resource_root = files("mellea_skills_compiler.rules")
    registry_text = (resource_root / _REGISTRY_RESOURCE).read_text(
        encoding="utf-8"
    )
    schema_text = (resource_root / _SCHEMA_RESOURCE).read_text(
        encoding="utf-8"
    )
    registry = json.loads(registry_text)
    schema = json.loads(schema_text)
    # Local import — jsonschema is a hard dep but keeping the import
    # local mirrors the pattern used elsewhere in the compiler.
    import jsonschema

    jsonschema.validate(registry, schema)
    return registry


def get_rule(rule_id: str) -> dict[str, Any]:
    """Return the registry entry for ``rule_id``.

    Raises:
        KeyError: if no rule with that id is in the registry.
    """
    registry = load_registry()
    rules = registry.get("rules", {})
    if rule_id not in rules:
        raise KeyError(
            f"no rule '{rule_id}' in the registry; "
            f"available: {sorted(rules)}"
        )
    return rules[rule_id]


def get_coherence_check(rule_id: str, check_id: str) -> dict[str, Any]:
    """Return the coherence-check entry for ``check_id`` under ``rule_id``.

    Raises:
        KeyError: if no rule or no check with that id is in the registry.
    """
    rule = get_rule(rule_id)
    for check in rule.get("coherence_checks", []):
        if check.get("id") == check_id:
            return check
    raise KeyError(
        f"no coherence_check '{check_id}' under rule '{rule_id}'; "
        f"available: {[c.get('id') for c in rule.get('coherence_checks', [])]}"
    )


def check_xfail_deadline(rule_id: str, check_id: str) -> Optional[str]:
    """For a coherence check carrying an ``xfail_until`` deadline, decide
    whether the calling test should xfail or assert today.

    Returns:
        * A *reason string* if the test should call ``pytest.xfail(reason)``
          right now (deadline is in the future).
        * ``None`` if the test should proceed and assert (no deadline OR
          deadline has passed — past-deadline xfails must be resolved by
          fixing the underlying drift, not by extending the deadline
          without justification).

    Discipline (from the deep-research finding on xfail ossification):
    coherence checks with ``xfail_until`` are *time-bounded* — past the
    deadline, the test fails loudly with a deadline-expired message,
    forcing a deliberate decision (fix the drift, extend the deadline,
    or delete the check).
    """
    check = get_coherence_check(rule_id, check_id)
    deadline_str = check.get("xfail_until")
    if not deadline_str:
        return None
    from datetime import date

    deadline = date.fromisoformat(deadline_str)
    today = date.today()
    if today < deadline:
        days_left = (deadline - today).days
        return (
            f"{rule_id}/{check_id} is xfail-until {deadline_str} "
            f"({days_left} days left). Coherence check encodes a "
            f"contract the implementation hasn't met yet; convert this "
            f"xfail into a passing assertion before the deadline, or "
            f"extend the deadline with explicit rationale in the "
            f"registry entry's `motivation` field."
        )
    # Past the deadline → return None so the test asserts. The assertion
    # failure carries its own context; the registry maintainer must
    # then decide: fix the drift, extend the deadline (rare; needs
    # justification), or remove the coherence check.
    return None
