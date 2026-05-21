"""Meta-validation: the audit-coherence registry validates against its own schema.

Also pins basic structural invariants we want EVERY audit entry to
honour regardless of category:

  * ``coherence_checks`` is non-empty (the whole point of an entry is
    that it has auditable checks).
  * Every ``test`` pointer resolves to an existing test function (so we
    can't add a check that references a non-existent test).
  * When non-null, the ``location`` pointer in the ``validation`` block
    resolves to an existing callable. Nullable since v0.2 for
    repair-enrichment entries (mechanisms distributed across multiple
    call sites).

These are PROCESS guarantees: they don't audit a single entry's three
faces, they guarantee the registry mechanism itself is honest.
Per-entry three-face audits live in sibling test files (e.g.
``test_rule_coherence_stdlib_arity.py``,
``test_rule_coherence_d2_closest_match.py``).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from mellea_skills_compiler.rules import load_registry


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_registry_validates_against_its_schema():
    """``load_registry`` will raise if the registry doesn't conform — this
    test just exercises the load path and pins it as a contract.
    """
    registry = load_registry()
    assert "rules" in registry
    assert registry["rules"], "registry must contain at least one rule entry"


def test_every_rule_has_at_least_one_coherence_check():
    """A rule entry without coherence checks defeats the registry's
    purpose. The schema already enforces ``minItems: 1``, but pinning it
    here too means a future schema-loosening accident still fails the
    test suite.
    """
    registry = load_registry()
    for rule_id, rule in registry["rules"].items():
        assert rule["coherence_checks"], (
            f"rule {rule_id!r} has no coherence checks — registry entries "
            f"must have at least one audit-able check, otherwise they are "
            f"just documentation."
        )


def _resolve_pointer(pointer: str) -> object:
    """Resolve a ``path::callable`` pointer to the actual callable.

    Accepts pointers of the shape ``src/.../<module>.py::<callable>``
    or ``tests/.../<module>.py::<callable>``. Repo-relative.
    """
    if "::" not in pointer:
        raise ValueError(
            f"pointer {pointer!r} missing '::' separator between path and callable"
        )
    path_str, attr_name = pointer.split("::", 1)
    file_path = _REPO_ROOT / path_str
    if not file_path.is_file():
        raise FileNotFoundError(
            f"pointer {pointer!r} → file {file_path} does not exist"
        )
    # Convert repo-relative file path → dotted module path.
    # ``src/`` is stripped because the package layout is
    # ``src/mellea_skills_compiler/...`` → ``mellea_skills_compiler...``.
    # ``tests/`` is NOT stripped because the tests are imported as the
    # ``tests.mellea_skills_compiler...`` package (tests/__init__.py
    # exists, anchoring the package root at the repo root).
    parts = file_path.with_suffix("").relative_to(_REPO_ROOT).parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    module_name = ".".join(parts)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise AttributeError(
            f"pointer {pointer!r} → module {module_name!r} has no attribute "
            f"{attr_name!r}"
        )
    return getattr(module, attr_name)


def test_every_coherence_check_test_pointer_resolves():
    """Every ``coherence_checks[].test`` pointer must resolve to a real
    pytest function. A pointer that doesn't resolve is a registry-rot
    symptom — the test was deleted or renamed but the registry still
    advertises it.
    """
    registry = load_registry()
    unresolved: list[str] = []
    for rule_id, rule in registry["rules"].items():
        for check in rule["coherence_checks"]:
            pointer = check["test"]
            try:
                fn = _resolve_pointer(pointer)
            except (ValueError, FileNotFoundError, AttributeError) as exc:
                unresolved.append(f"{rule_id}/{check['id']}: {exc}")
                continue
            if not callable(fn):
                unresolved.append(
                    f"{rule_id}/{check['id']}: {pointer!r} resolves to "
                    f"a non-callable ({type(fn).__name__})"
                )
    if unresolved:
        pytest.fail(
            "Some coherence-check test pointers do not resolve:\n  - "
            + "\n  - ".join(unresolved)
        )


def test_every_validation_location_resolves():
    """When an entry's ``validation.location`` is non-null, it must
    resolve to a real callable. Catches: registry says lint X lives
    at path P, but P doesn't have an X (renamed, deleted, never
    existed).

    Nullable since v0.2: repair-enrichment entries may not have a
    single canonical callable (the mechanism is distributed across
    multiple call sites). For those, ``location`` may be null and
    the per-entry coherence tests cover the surface.
    """
    registry = load_registry()
    unresolved: list[str] = []
    for rule_id, rule in registry["rules"].items():
        pointer = rule["validation"].get("location")
        if pointer is None:
            continue  # nullable: see docstring above
        try:
            fn = _resolve_pointer(pointer)
        except (ValueError, FileNotFoundError, AttributeError) as exc:
            unresolved.append(f"{rule_id}: {exc}")
            continue
        if not callable(fn):
            unresolved.append(
                f"{rule_id}: validation.location {pointer!r} resolves to "
                f"a non-callable ({type(fn).__name__})"
            )
    if unresolved:
        pytest.fail(
            "Some validation.location pointers do not resolve:\n  - "
            + "\n  - ".join(unresolved)
        )


def test_every_entry_has_substantive_motivation():
    """v0.3 schema property — every audit entry must have a `motivation`
    field of meaningful length. The schema enforces minLength=60; this
    test pins the universal property at the registry level so a future
    schema relaxation can't silently allow short / empty motivations.

    The motivation field is the 4th non-enforced face (per DO-178C ADR
    discipline): coherence checks don't bind it, but its presence is
    mandatory so we never have a registered rule with no rationale.
    """
    registry = load_registry()
    missing_or_short: list[tuple[str, int]] = []
    for rule_id, rule in registry["rules"].items():
        motivation = rule.get("motivation", "") or ""
        if len(motivation) < 60:
            missing_or_short.append((rule_id, len(motivation)))
    if missing_or_short:
        formatted = "\n".join(
            f"  - {rid}: motivation length {ln} chars (need >= 60)"
            for rid, ln in missing_or_short
        )
        pytest.fail(
            "Some entries have missing or short motivation:\n"
            + formatted
            + "\nMotivation is the 4th non-enforced face: record the "
            "incident, design choice, or correctness invariant that "
            "motivated the rule. Future maintainers consult it to "
            "decide whether to weaken or strengthen the rule."
        )


def test_every_directive_has_non_empty_stance():
    """Property — every audit entry's `directive.stance` is non-empty
    and substantive (>50 chars). Without this, a registered rule could
    have a completely empty directive face — meaning the LLM is being
    graded by the validator with no instruction from us about what we
    expect. That's *criteria drift* (Shankar et al. 2024, EvalGen) at
    the registration boundary, which the registry was built to catch.
    """
    registry = load_registry()
    weak: list[tuple[str, int]] = []
    for rule_id, rule in registry["rules"].items():
        stance = rule.get("directive", {}).get("stance", "") or ""
        if len(stance) < 50:
            weak.append((rule_id, len(stance)))
    if weak:
        formatted = "\n".join(
            f"  - {rid}: directive.stance length {ln} chars (need >= 50)"
            for rid, ln in weak
        )
        pytest.fail(
            "Some entries have empty / weak directive.stance — the LLM "
            "is being graded without explicit instruction. Drift class: "
            "directive ↔ validation (criteria drift).\n"
            + formatted
        )


def test_every_xfail_deadline_is_iso_date_in_the_future_or_past():
    """Property — every `xfail_until` deadline parses as an ISO date
    and is either in the past (deadline expired; the test must now
    fail loudly) or in the future. Catches typo'd deadlines that
    could silently never fire (e.g., a deadline 200 years out).
    """
    from datetime import date

    registry = load_registry()
    far_future_cutoff = date(date.today().year + 2, 1, 1)
    issues: list[str] = []
    for rule_id, rule in registry["rules"].items():
        for check in rule.get("coherence_checks", []):
            deadline_str = check.get("xfail_until")
            if not deadline_str:
                continue
            try:
                deadline = date.fromisoformat(deadline_str)
            except ValueError as exc:
                issues.append(
                    f"  - {rule_id}/{check['id']}: xfail_until "
                    f"{deadline_str!r} is not ISO-8601 ({exc})"
                )
                continue
            if deadline > far_future_cutoff:
                issues.append(
                    f"  - {rule_id}/{check['id']}: xfail_until "
                    f"{deadline_str} is more than 2 years out — "
                    f"effectively a permanent xfail. Tighten to a real "
                    f"deadline you'll act on."
                )
    if issues:
        pytest.fail(
            "Some xfail_until deadlines are typo'd or effectively-"
            "infinite:\n" + "\n".join(issues)
        )


def test_directive_doc_paths_exist():
    """When a rule's ``directive.doc`` is non-null, the file must exist
    on disk. Catches: registry points at a slash command that was
    renamed or deleted.
    """
    registry = load_registry()
    missing: list[str] = []
    for rule_id, rule in registry["rules"].items():
        doc_path = rule["directive"].get("doc")
        if doc_path is None:
            continue
        full_path = _REPO_ROOT / doc_path
        if not full_path.is_file():
            missing.append(f"{rule_id}: {doc_path} (resolved: {full_path})")
    if missing:
        pytest.fail(
            "Some directive.doc paths do not exist on disk:\n  - "
            + "\n  - ".join(missing)
        )
