# Melleafy Step 4: Fixture Generation

**Version**: 4.2.0 | **Prereq**: Step 3 complete (skeleton emitted with finalised `run_pipeline` signature) AND `intermediate/expected_signature.json` present (Step 2 emits it; P3.5.D) | **Produces**: `<package_name>/fixtures/`

> **Output path rule** (Rule OUT-4): `fixtures/` is written **inside `<package_name>/`** — NOT at the skill root. It is test-only and intentionally excluded from the installed package by `pyproject.toml`'s `[tool.setuptools.packages.find]`. Run fixtures via `python -m pytest <package_name>/fixtures/` from the skill root.

Step 4 generates 5–8 test fixtures covering ≥3 C-categories. Fixtures are the primary means by which the generated package can be validated at runtime (Step 7's lints are static-only).

> **Rule 4-1 — Batched fixture generation as JSON**: Generate all fixture specifications in a single LLM invocation. The invocation receives the `run_pipeline` signature, the element mapping summary, and the C-category coverage requirement, and returns **one JSON object conforming to `.claude/schemas/fixtures_emission.schema.json`** — not Python source. The deterministic writer at `.claude/melleafy/writers/fixtures_writer.py` renders the per-fixture `.py` files plus `fixtures/__init__.py` from that JSON. **The model never writes Python source for fixtures directly.** This makes shape drift (pytest-style tests, `INPUT`-only modules, hand-rolled `__init__.py` exports) structurally unreachable — every legal JSON instance produces a contract-correct fixture package.

> **Fallback when the model does not emit `fixtures_emission.json`**: if the slash command exits with the per-fixture `.py` files on disk but without writing `intermediate/fixtures_emission.json`, the wrapper reverse-engineers the IR from the surviving `fixtures/*.py` files BEFORE the writer's wipe-then-render runs. Each fixture file declares `make_<fid>() -> tuple[dict, str, str]`; the wrapper AST-parses the return value (or falls back to importing the module if any element is non-literal), reassembles a schema-compliant `fixtures_emission.json`, and writes it to `intermediate/`. The writer then renders the same fixtures from the synthesised IR, keeping the package's on-disk state and its IR self-consistent. The model is still expected to emit the IR directly — this fallback is the last line of defence, mirroring the `config_emission.json` fallback documented in `mellea-fy-generate.md` (Track 1 #4). The synthesizer lives at `mellea_skills_compiler.compile.writer_renderer.synthesize_fixtures_emission_from_existing`.

---

## CRITICAL: Input parameter matching

**The keys in every fixture's `inputs` object MUST be a subset of the parameter names of the `run_pipeline` function in `pipeline.py`.** This is not optional. The smoke-check invokes `run_pipeline(**inputs)`; any extra key raises `TypeError: got an unexpected keyword argument '<key>'` at fixture-run time. Empirically observed regression: a `nis2-navigator` fixture passed `{'session_id': ..., 'regulatory_updates': ...}` to a pipeline that compiled without the matching parameter set, and the smoke-check crashed. Before emitting JSON, verify the exact parameter names from the generated pipeline function. The schema cannot enforce this — it is the model's responsibility to match. The `fixture-signature-bound` lint (Step 7) enforces this mechanically: any `inputs` key not in the entry parameter set is a hard failure. Optional/defaulted parameters MAY be omitted (subset semantics); `**kwargs` in the signature exempts a fixture from the check entirely.

### Source of truth: `intermediate/expected_signature.json` (P3.5.D)

When `intermediate/expected_signature.json` is present (it is, after P3.5.D — Step 2 always emits it), it is the **canonical source of truth** for the `run_pipeline` signature. Read it before generating any fixture:

- Every fixture's `inputs` keys MUST be a subset of `expected_signature.inputs[].name`.
- Fixture values MUST be type-compatible with `expected_signature.inputs[].type` (e.g. an input typed `list[str]` requires a JSON list of strings).
- Optional inputs (`expected_signature.inputs[].optional == true`) MAY be omitted from any fixture.

If `expected_signature.json` is absent (legacy / pre-P3.5.D path), fall back to today's behaviour: read the `run_pipeline` signature from the emitted skeleton in `<package_name>/pipeline.py`. The `fixture-signature-bound` lint (Step 7) catches drift either way.

---

## Fixture structure

The model emits one JSON object; the writer renders the entire `fixtures/` subpackage.

_JSON the model emits (conforms to `fixtures_emission.schema.json`):_

```json
{
  "fixtures": [
    {
      "id": "positive_case",
      "description": "Critical priority ticket — should escalate to L2 immediately",
      "inputs": {
        "ticket_text": "URGENT: Production database is returning 500 errors for all queries.",
        "priority": "critical"
      }
    },
    {
      "id": "clean_case",
      "description": "Routine request — should route to standard support queue",
      "inputs": {
        "ticket_text": "Can you update my email address in the portal?",
        "priority": "normal"
      }
    }
  ],
  "coverage_doc": "Fixture coverage:\n  C1 Identity: all fixtures\n  C2 Operating rules: positive_case\n  C6 Tools: positive_case"
}
```

_Python source the writer renders from that JSON:_

```python
# fixtures/__init__.py
"""
Fixture coverage:
  C1 Identity: all fixtures
  C2 Operating rules: positive_case
  C6 Tools: positive_case
"""
from typing import Callable

from .positive_case import make_positive_case
from .clean_case import make_clean_case

ALL_FIXTURES: list[Callable] = [
    make_positive_case,
    make_clean_case,
]
```

```python
# fixtures/positive_case.py
"""Auto-generated by melleafy from fixtures_emission JSON. Fixture id="positive_case"."""


def make_positive_case():
    inputs = {
        'ticket_text': 'URGENT: Production database is returning 500 errors for all queries.',
        'priority': 'critical',
    }
    return inputs, 'positive_case', 'Critical priority ticket — should escalate to L2 immediately'
```

Each rendered factory is a zero-arg callable returning `(inputs, fixture_id, description)` — the contract enforced by `mellea_skills_compiler.toolkit.file_utils.load_fixtures`. The writer guarantees this shape; the model is responsible only for the JSON content.

### Anti-patterns the writer prevents (do not emit)

The writer architecture makes these shapes unreachable, but they have appeared in past LLM-generated outputs and serve as a sanity check on what _not_ to think you should emit:

- **Pytest-style test functions** — `def test_<name>() -> None: assert ...` is a pytest test, not a melleafy fixture. Melleafy fixtures are factories returning `(inputs, fixture_id, description)`.
- **Bare `INPUT = {...}` modules** with `__init__.py` re-exporting them as `*_INPUT` aliases — there is no `ALL_FIXTURES` list and no factory functions, so `load_fixtures` rejects the package.
- **Hand-rolled `FIXTURES = [{"id": ..., "context": ...}]` dicts** — the alternate convention some early skills used; the writer always produces `ALL_FIXTURES` of factories.

---

## Required fixture categories

Generate at least one fixture per category. Not all categories apply to every skill — use judgment:

| Category                   | Purpose                                                        | When to include                                 |
| -------------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| **Positive case**          | Clear input triggering the skill's primary behaviour           | Always                                          |
| **Clean/negative case**    | Input where the skill finds nothing or produces minimal output | Always                                          |
| **Edge case (structural)** | Empty input, very short, very long, missing optional fields    | Always                                          |
| **Edge case (domain)**     | Input at the boundary of the skill's scope                     | When the skill has meaningful domain boundaries |
| **Mixed case**             | Combination of positive and negative signals                   | For analysis/diagnosis archetypes               |
| **Out-of-scope case**      | Input the skill should explicitly decline                      | When the spec defines "When NOT to Use"         |

---

## C-category coverage requirement

Fixtures must collectively exercise ≥3 dependency categories (C1–C9). This is R16's fixture coverage threshold. Document which categories each fixture exercises in the `description` field.

Example — a ticket-triage skill with C1 (persona), C2 (operating rules), and C6 (tool calls):

- Positive case exercises C1 (persona applied), C2 (escalation rule fires), C6 (Slack notification called)
- Clean case exercises C1 and C2 (no escalation, no tool call)
- Edge case (empty) exercises C2 (out-of-scope handling rule)

Record in `fixtures/__init__.py` as a docstring:

```python
"""
Fixture coverage:
  C1 Identity: all fixtures
  C2 Operating rules: positive_case, edge_empty
  C6 Tools: positive_case, mixed_case
"""
```

---

## Quality rules

- Inputs must have realistic, non-trivial data — not placeholder text like `"some input here"`
- The `description` field states the expected behaviour, not just what the input is
- Mock data for C6 tools goes in `fixtures/mock_tools.py` (for stubs with disposition `mock`), not in fixture inputs
- Do NOT use real credentials, personal data, or production system identifiers in fixtures
- Fixtures should run without network access — if the pipeline makes tool calls, use mock tool implementations or `load_from_disk` with bundled fixture data

---

## `fixtures/mock_tools.py` (when any C6 has disposition `mock`)

For each C6 tool with `disposition: "mock"`, generate a mock implementation using representative fixture data:

```python
# fixtures/mock_tools.py
def mock_doi_lookup(doi: str) -> dict:
    """Mock implementation for testing — returns representative fixture data."""
    return {
        "title": "Example Paper Title",
        "authors": ["Smith, J.", "Jones, A."],
        "year": 2024,
        "doi": doi,
    }
```

---

## Cross-checks before Step 4 declares done

The schema (`fixtures_emission.schema.json`) and writer (`fixtures_writer.py`) together enforce most of the contract structurally. The model is responsible only for content correctness:

- Fixture count: 5 ≤ N ≤ 8 (enforced by `minItems`/`maxItems` in the schema)
- At least 3 distinct C-categories exercised across all fixtures (model self-check; documented in `coverage_doc`)
- Every fixture's `inputs` keys match the generated `run_pipeline` signature exactly (model self-check; verify against `pipeline.py` before emitting JSON)
- Every fixture has a non-empty, realistic `inputs` value (not placeholder text)
- `coverage_doc` is populated with C-category coverage notes

The writer guarantees, regardless of model output:

- `fixtures/__init__.py` exports `ALL_FIXTURES: list[Callable]`
- Each `fixtures/<id>.py` defines a `make_<id>()` factory returning `(inputs, fixture_id, description)`
- All `id` values are unique snake_case identifiers
