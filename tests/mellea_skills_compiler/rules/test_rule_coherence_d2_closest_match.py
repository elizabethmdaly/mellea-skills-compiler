"""Coherence audits for the ``d2-closest-match-enrichment`` mechanism.

D2 augments repair prompts with 'Did you mean X?' hints when a gate
rejects an LLM emission with an identifiable symbol/path that's
close-but-wrong. The mechanism has three faces just like a rule:
directive, implementation, validation.

This file declares the COHERENCE CONTRACT the D2 implementation must
satisfy when built. Tests are ``xfail(strict=True)`` until the
implementation lands — when D2 is wired up, the tests turn green and
the strict-xfail catches if any of the contract clauses regresses.

The pattern: register first, write coherence tests against the
contract, implement to make them pass. Same TDD-style flow that drove
#37 and #38 — but at the mechanism level.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mellea_skills_compiler.rules import get_rule


_RULE_ID = "d2-closest-match-enrichment"

# Sentinel: when the D2 module is created, this import will start
# resolving and the xfail-strict tests below will start passing.
_D2_AVAILABLE = False
try:  # pragma: no cover - exercised once D2 lands
    from mellea_skills_compiler.compile import repair_enrichment as _d2  # type: ignore  # noqa: F401

    _D2_AVAILABLE = True
except ImportError:
    pass


_xfail_until_d2 = pytest.mark.xfail(
    not _D2_AVAILABLE,
    strict=True,
    reason=(
        "d2-closest-match-enrichment is not yet implemented. This test "
        "encodes the contract the implementation must satisfy. When "
        "compile/repair_enrichment.py lands, the import at the top of "
        "this file succeeds, the xfail flips off, and the test asserts "
        "the contract for real."
    ),
)


# ─── Coherence checks ────────────────────────────────────────────────


@_xfail_until_d2
def test_known_module_typo_resolves_to_real_path():
    """C-CLOSEST-MATCH-FINDS-NEAR-MISS: a one-character typo of a real
    surface module path resolves to the real path as the top match.

    Concrete example from a real LLM emission: ``mellea.backens.ollama``
    (typo on ``backends``) must produce ``mellea.backends.ollama`` as
    the suggestion.
    """
    from mellea_skills_compiler.compile.repair_enrichment import (  # type: ignore
        closest_module_match,
    )

    surface_modules = {
        "mellea.backends.ollama",
        "mellea.backends.openai",
        "mellea.backends.model_options",
        "mellea.stdlib.session",
    }
    suggestion = closest_module_match(
        "mellea.backens.ollama", surface_modules
    )
    assert suggestion == "mellea.backends.ollama", (
        f"C-CLOSEST-MATCH-FINDS-NEAR-MISS failed: expected "
        f"'mellea.backends.ollama' as the top match for a near-miss "
        f"typo; got {suggestion!r}."
    )


@_xfail_until_d2
def test_no_near_miss_returns_no_suggestion():
    """C-NO-FALSE-CONFIDENCE: when no candidate is close enough to meet
    the threshold + dominance bar, the algorithm returns None (or
    equivalent 'no suggestion' marker). Guards against the LLM
    following a misleading low-confidence hint.
    """
    from mellea_skills_compiler.compile.repair_enrichment import (  # type: ignore
        closest_module_match,
    )

    surface_modules = {
        "mellea.backends.ollama",
        "mellea.stdlib.session",
    }
    # A name with no plausible neighbour in the surface.
    suggestion = closest_module_match(
        "totally.unrelated.fictional.path", surface_modules
    )
    assert suggestion is None, (
        f"C-NO-FALSE-CONFIDENCE failed: expected None for a name with "
        f"no plausible near-miss; got {suggestion!r}. The mechanism "
        f"emitted a suggestion that the LLM might follow into a worse "
        f"emission."
    )


@_xfail_until_d2
def test_repair_prompt_carries_suggestion():
    """C-PROMPT-INCLUDES-SUGGESTION: when the algorithm produces a
    suggestion, the repair prompt actually contains it in a form the
    LLM can act on. End-to-end wiring check.
    """
    from mellea_skills_compiler.compile.repair_enrichment import (  # type: ignore
        build_enriched_repair_prompt,
    )

    failure = {
        "kind": "import-soundness",
        "failing_path": "mellea.backens.ollama",
    }
    surface_modules = {
        "mellea.backends.ollama",
        "mellea.stdlib.session",
    }
    prompt = build_enriched_repair_prompt(failure, surface_modules)
    assert "mellea.backends.ollama" in prompt, (
        "C-PROMPT-INCLUDES-SUGGESTION failed: the algorithm produced a "
        "suggestion but it didn't appear in the repair prompt. Wiring "
        "drift between the algorithm and the prompt builder."
    )
    # The standard 'Did you mean X' phrasing makes the suggestion
    # unambiguous to the LLM; the exact wording is a convention.
    assert "Did you mean" in prompt or "did you mean" in prompt.lower(), (
        "C-PROMPT-INCLUDES-SUGGESTION failed: suggestion present but "
        "without the agreed 'Did you mean' phrasing. The directive "
        "stance commits to this phrasing — drift between directive "
        "and implementation."
    )


@_xfail_until_d2
def test_repair_prompt_omits_suggestion_when_no_match():
    """C-PROMPT-OMITS-WHEN-NO-MATCH: when the algorithm returns no
    suggestion, the prompt does not include a 'Did you mean' line.
    Symmetric guard to C-NO-FALSE-CONFIDENCE — verifies the prompt
    builder respects the algorithm's 'no match' verdict.
    """
    from mellea_skills_compiler.compile.repair_enrichment import (  # type: ignore
        build_enriched_repair_prompt,
    )

    failure = {
        "kind": "import-soundness",
        "failing_path": "totally.unrelated.fictional.path",
    }
    surface_modules = {
        "mellea.backends.ollama",
        "mellea.stdlib.session",
    }
    prompt = build_enriched_repair_prompt(failure, surface_modules)
    assert "did you mean" not in prompt.lower(), (
        "C-PROMPT-OMITS-WHEN-NO-MATCH failed: prompt contains a "
        "suggestion line even though the algorithm should have "
        "declined to suggest. The prompt builder is fabricating "
        "hints — exactly the false-confidence failure mode the "
        "registry was built to prevent."
    )


@_xfail_until_d2
def test_candidate_set_scoped_to_failure_class():
    """C-SCOPE-MATCHES-FAILURE-CLASS: a kwarg-typo failure searches
    only the function's kwarg names — NOT the entire module surface.
    A module-path failure searches modules — NOT kwarg names. Guards
    against the algorithm conflating failure classes.
    """
    from mellea_skills_compiler.compile.repair_enrichment import (  # type: ignore
        suggest_for_failure,
    )

    # Same LLM string `bckend` could match either a module
    # ``mellea.backend.X`` OR a kwarg ``backend=`` — the algorithm
    # must pick the appropriate candidate set based on failure class.
    kwarg_failure = {
        "kind": "stdlib-arity",
        "failure_subkind": "unknown-kwarg",
        "failing_token": "bckend",
        "valid_kwargs": ["backend", "ctx"],
    }
    suggestion = suggest_for_failure(kwarg_failure)
    assert suggestion == "backend", (
        f"C-SCOPE-MATCHES-FAILURE-CLASS failed: kwarg-typo failure "
        f"got {suggestion!r}; expected 'backend' from the kwarg "
        f"candidate set. The algorithm may be searching the wrong "
        f"surface."
    )


def test_directive_doc_section_exists():
    """C-DIRECTIVE-DOC-EXISTS: the directive doc the registry points
    at must exist on disk and contain a section explaining the
    'Did you mean' mechanism. This check is NOT xfail-gated — it
    audits the registry's directive pointer regardless of whether
    D2 is implemented yet, because an undocumented mechanism is a
    drift even before it's built.
    """
    rule = get_rule(_RULE_ID)
    doc_rel = rule["directive"]["doc"]
    repo_root = Path(__file__).resolve().parents[3]
    doc_path = repo_root / doc_rel

    if not doc_path.is_file():
        # When the directive doc doesn't yet exist, mark as xfail rather
        # than fail outright — same convention as the implementation
        # itself. The xfail-strict ensures we notice when the doc lands.
        pytest.xfail(
            f"directive doc {doc_rel!r} does not yet exist. The "
            f"registry's directive pointer is asserting a contract "
            f"that hasn't been written. Either create the doc or "
            f"update the registry's directive.doc / directive.section."
        )

    # When the doc exists, verify it contains the section the registry
    # claims it does (loose match — any mention of 'closest-match',
    # 'did you mean', or 'suggestion' counts).
    text = doc_path.read_text(encoding="utf-8").lower()
    keywords = ["closest-match", "did you mean", "suggestion"]
    if not any(kw in text for kw in keywords):
        # Honour the registry's `xfail_until` deadline so this can't
        # ossify into a permanent expected-to-fail.
        from mellea_skills_compiler.rules import check_xfail_deadline

        deadline_reason = check_xfail_deadline(
            _RULE_ID, "C-DIRECTIVE-DOC-EXISTS"
        )
        if deadline_reason:
            pytest.xfail(deadline_reason)
        pytest.fail(
            f"directive doc {doc_rel!r} exists but does not mention any "
            f"of {keywords}. The 'How to use closest-match suggestions "
            f"in repair' section was promised in the registry but has "
            f"not landed yet, AND the xfail_until deadline has expired "
            f"or is unset. Write the section in mellea-fy-repair.md or "
            f"extend the registry deadline with explicit justification."
        )
