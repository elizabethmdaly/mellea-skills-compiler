"""CI doc-consistency: lint count in the architecture reference must match
the canonical Python source.

The architecture reference at
``melleafy-handoff/analyses/2026-05-18-compiler-architecture-reference.md``
makes several quantitative claims about the lint suite (e.g. "25 lints",
"19 ERROR + 5 WARNING + 1 INFO"). The canonical source for those counts
is ``compile/lints.py::_LINT_SEVERITY``.

Historically these have drifted ("14" / "Seventeen" / "17" / "23" / "25"
all appeared in different docs at different times — see §7-Lint-count in
the architecture doc). The audit fix in 2026-05-18 promised a CI test
to close the drift class permanently; this is that test.

When this test fails, do NOT just edit the doc. Verify that the new
canonical count matches what you intended (`tests/.../test_lint_severity.py`
asserts the actual class breakdown); only then bump the architecture doc
to match.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import (
    LintSeverity,
    _LINT_SEVERITY,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARCH_DOC = (
    _REPO_ROOT
    / "melleafy-handoff"
    / "analyses"
    / "2026-05-18-compiler-architecture-reference.md"
)


def _arch_doc_text() -> str:
    """Read the architecture reference, skipping the test if absent.

    The doc lives under ``melleafy-handoff/`` which is part of the
    repository but not part of the installed wheel — when this test runs
    against an installed copy (e.g. ``pip install`` + ``pytest``
    elsewhere) the file may not be present. Skip rather than fail in
    that case.
    """
    if not _ARCH_DOC.is_file():
        pytest.skip(
            f"Architecture reference doc not present at {_ARCH_DOC}; "
            "skipping doc-consistency check."
        )
    return _ARCH_DOC.read_text(encoding="utf-8")


def _canonical_counts() -> tuple[int, int, int, int]:
    """Return (total, error, warning, info) from ``_LINT_SEVERITY``."""
    by_sev: dict[LintSeverity, int] = {
        LintSeverity.ERROR: 0,
        LintSeverity.WARNING: 0,
        LintSeverity.INFO: 0,
    }
    for sev in _LINT_SEVERITY.values():
        by_sev[sev] += 1
    return (
        len(_LINT_SEVERITY),
        by_sev[LintSeverity.ERROR],
        by_sev[LintSeverity.WARNING],
        by_sev[LintSeverity.INFO],
    )


def test_architecture_doc_lint_count_matches_canonical_source():
    """Every "N lints" claim in the architecture doc must match ``len(_LINT_SEVERITY)``.

    Scans for the pattern ``<NUMBER> lints`` (case-insensitive) and
    asserts each captured number equals the canonical total. Catches
    drift the moment a doc edit forgets to update one of the several
    places the count appears (§3 module matrix, §6.1 severity model,
    §8 cross-reference table, end-of-doc footer, the Mermaid diagram
    label).
    """
    text = _arch_doc_text()
    canonical_total, _, _, _ = _canonical_counts()

    # Match "<N> lints" (plural) or "<N> lint functions". Both are the
    # *count claim* phrases. The plural-only constraint already
    # excludes singular "lint" references to specific lints. We then
    # post-filter to exclude structural prefixes where the digit is a
    # step or tier number, not a count — e.g. "Step 7 lints",
    # "Step-7 lint", "Tier 1 lint".
    pattern = re.compile(
        r"(\d+)\s+(?:lints\b|lint\s+functions\b)", re.IGNORECASE
    )

    # Structural prefixes that mean "the digit here names a phase /
    # step / tier", not "the digit here is a count". Add new patterns
    # to this tuple if a future doc edit introduces another structural
    # numbering convention that abuts "lints".
    _STRUCTURAL_PREFIXES = ("Step ", "Step-", "Tier ", "Tier-")

    claims: list[tuple[int, int]] = []
    for m in pattern.finditer(text):
        digit_start = m.start()
        # The longest structural prefix is 5 chars; check the 5 chars
        # immediately before the digit.
        prefix_window = text[max(0, digit_start - 5):digit_start]
        if any(prefix_window.endswith(p) for p in _STRUCTURAL_PREFIXES):
            continue
        claims.append((int(m.group(1)), digit_start))

    assert claims, (
        "Found NO 'N lints' or 'N lint functions' count claims in the "
        "architecture doc after filtering structural prefixes. The "
        "pattern may have been removed or the regex / filter needs "
        f"widening. Inspect {_ARCH_DOC} and update this test."
    )

    mismatches: list[tuple[int, int]] = [
        (claimed, offset)
        for claimed, offset in claims
        if claimed != canonical_total
    ]
    if mismatches:
        formatted = "\n".join(
            f"  - at offset {offset}: claims {claimed} lints "
            f"(canonical is {canonical_total})"
            for claimed, offset in mismatches
        )
        pytest.fail(
            f"Architecture doc has stale lint-count claims that disagree "
            f"with `_LINT_SEVERITY` (canonical: {canonical_total}):\n"
            f"{formatted}\n\n"
            f"Either update the doc to {canonical_total} or update "
            f"`_LINT_SEVERITY` if the canonical count should change."
        )


def test_architecture_doc_severity_split_matches_canonical_source():
    """Every "N ERROR + M WARNING + K INFO" claim must match the canonical histogram.

    The architecture doc states the breakdown in multiple places (§6.1
    prose, §3 module-responsibility-matrix row for ``compile/lints.py``).
    All occurrences must agree with the actual content of ``_LINT_SEVERITY``.
    """
    text = _arch_doc_text()
    _, canonical_error, canonical_warning, canonical_info = _canonical_counts()

    # Match "<E> ERROR + <W> WARNING + <I> INFO" (case-insensitive,
    # tolerant of spaces around "+"). Allows the integer prefix to be
    # exactly that — three integers separated by literal "+ ERROR / + WARNING / + INFO".
    pattern = re.compile(
        r"(\d+)\s+ERROR\s*\+\s*(\d+)\s+WARNING\s*\+\s*(\d+)\s+INFO",
        re.IGNORECASE,
    )
    claims = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.start())
        for m in pattern.finditer(text)
    ]
    assert claims, (
        "Found NO 'E ERROR + W WARNING + I INFO' claims in the "
        "architecture doc. Either the breakdown was removed or the regex "
        "needs widening."
    )

    expected = (canonical_error, canonical_warning, canonical_info)
    mismatches: list[tuple[tuple[int, int, int], int]] = [
        ((e, w, i), offset)
        for e, w, i, offset in claims
        if (e, w, i) != expected
    ]
    if mismatches:
        formatted = "\n".join(
            f"  - at offset {offset}: claims {e} ERROR + {w} WARNING + {i} INFO "
            f"(canonical is {canonical_error} ERROR + {canonical_warning} "
            f"WARNING + {canonical_info} INFO)"
            for (e, w, i), offset in mismatches
        )
        pytest.fail(
            f"Architecture doc has stale severity-split claims that "
            f"disagree with `_LINT_SEVERITY`:\n{formatted}"
        )
