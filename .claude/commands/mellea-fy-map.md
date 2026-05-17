# Melleafy Step 2: Element-to-Primitive Mapping

**Version**: 4.2.0 | **Prereq**: `inventory.json`, `classification.json` | **Produces**: `element_mapping.json`, `expected_signature.json`

> **Schema**: Output `intermediate/element_mapping.json` MUST conform to `.claude/schemas/element_mapping.schema.json`. Output `intermediate/expected_signature.json` MUST conform to `src/mellea_skills_compiler/descriptor/schemas/expected_signature.schema.json` (P3.5.D — fixture/signature alignment).

Step 2 reads `inventory.json` and produces `element_mapping.json` — the routing decision for every element: which file in the generated package, which symbol, which Mellea primitive. It also emits `expected_signature.json` — the canonical `run_pipeline` I/O signature locked from this point forward (Step 4 fixtures, Step 5 descriptor emission, and the `R-SEM-SIGNATURE-MATCH` descriptor validator rule all read this artefact).

**Important**: Step 2 does NOT commit dispositions for tool-dependent elements. Every `TOOL_TEMPLATE` mapping entry is provisional (`final_target_file: "pending_step_2.5"`). Step 2.5 decides `real_impl` vs `stub` vs `mock` and amends.

---

## Tag-to-primitive table

| Tag               | Primary primitive                                             | Target file                 | Notes                                                                                                   |
| ----------------- | ------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `EXTRACT`         | `@generative` slot                                            | `slots.py`                  | Two-step pattern when schema complexity warrants (§below)                                               |
| `CLASSIFY`        | `@generative` slot                                            | `slots.py`                  | Return type: `-> Literal[...]` — Ollama supports constrained decoding                                   |
| `GENERATE`        | `m.instruct(format=Schema)`                                   | inline in `pipeline.py`     | `format=` always a concrete Pydantic model, never `dict`                                                |
| `VALIDATE_OUTPUT` | `Requirement`                                                 | `requirements.py`           | Uses `validation_fn=simple_validate(...)` for structural checks; bare `description` for semantic checks |
| `VALIDATE_DOMAIN` | `m.instruct(format=DomainSchema)`                             | inline in `pipeline.py`     | Checks external artifacts; produces structured verdict, not pass/fail boolean                           |
| `TRANSFORM`       | `m.transform()` or `m.instruct(format=Schema)`                | inline in `pipeline.py`     | `m.transform()` when types are known; `m.instruct` when transformation needs prompted reasoning         |
| `QUERY`           | `m.query()`                                                   | inline in `pipeline.py`     | Read-only question against data already in scope                                                        |
| `DECIDE`          | `m.instruct(format=DecisionSchema)`                           | inline in `pipeline.py`     | Gates remediation loops (see Remediate below)                                                           |
| `ORCHESTRATE`     | Plain Python control flow                                     | `pipeline.py`               | Not a Mellea primitive — describes flow (sequential phases, branches, loops)                            |
| `CONVERSE`        | `m.chat()`, pipeline parameter, or `NotImplementedError` stub | varies                      | Three realisations — see below                                                                          |
| `REMEDIATE`       | Bounded `while` loop with `m.instruct(format=PatchSchema)`    | `pipeline.py`               | Three mapping entries: modification + evaluation + loop wrapper                                         |
| `SCHEMA`          | Pydantic `BaseModel` class                                    | `schemas.py`                | One class per schema; no nested submodels buried in function defs                                       |
| `CONFIG`          | `Final[T]` constant                                           | `config.py`                 | Under `# === C<N> ... ===` section header                                                               |
| `TOOL_TEMPLATE`   | Python function                                               | `tools.py` (provisional)    | Amended by Step 2.5d based on disposition                                                               |
| `DETERMINISTIC`   | Plain Python function                                         | `pipeline.py` or `tools.py` | `tools.py` when shared across branches or >15 lines                                                     |
| `TOOL_INPUT`      | Pipeline parameter or `loader.py` call                        | `main.py` or `loader.py`    | Data a tool produces that feeds the pipeline                                                            |
| `NO_DECOMPOSE`    | No primitive                                                  | —                           | Recorded in `element_mapping.json` with `primitive: "none"` for invariant completeness                  |

---

## Alternative rules

### EXTRACT — one-step vs two-step pattern

Default: one `@generative` slot returning the target schema.

**Two-step pattern applies when** any of:

- Target schema has cross-reference fields (a field that references another field's value in the same document)
- Target schema has more than 3 levels of nesting
- Target schema has optional fields whose presence depends on earlier fields' values
- Target schema has more than 4 fields OR contains `Literal` constraints OR has nested `BaseModel` objects OR lists of complex objects

When two-step applies, produce **two** mapping entries sharing the same `element_id` (suffixed `-step1`, `-step2`):

1. `@generative` slot returning a simplified flat structure (`slots.py:extract_X_raw`)
2. `m.instruct(format=FullSchema, strategy=RepairTemplateStrategy(loop_budget=3))` inline in `pipeline.py`

The reason: `@generative` has no retry/repair mechanism — malformed JSON silently returns empty. `m.instruct(format=...)` with `RepairTemplateStrategy` retries and repairs.

### VALIDATE_OUTPUT — executable vs LLM-judged

Default: `Requirement` with executable `validation_fn` (structural check in plain Python).

LLM-judged (bare `Requirement(description=...)`) only when `content_full` contains words like "accurate," "appropriate," "reasonable," "matches the spirit of" — markers of semantic judgement that can't be expressed in Python.

Record the choice as `validation_kind: "executable" | "llm_judged"` in the mapping entry.

### CONVERSE — three realisations

1. **`m.chat()` — LLM self-talk**: when the source describes multi-turn reasoning within the pipeline ("consider counterarguments then respond"). Emitted inline in `pipeline.py`.
2. **Pipeline parameter with default**: when the source says "ask the user for X." X becomes a parameter on `run_pipeline` with a default, exposed as a CLI flag on `main.py`.
3. **`NotImplementedError` stub**: when the source describes genuine interactive back-and-forth that can't be reshaped into either above — e.g., "iterate with the user until they approve the output." SETUP.md §7 explains the host-adapter requirement.

Decision rule: pick (2) when `content_full` contains "ask the user" or "user provides"; (1) when phrasing is about the agent's own reasoning ("consider," "reflect"); (3) when neither fits. If `classification.json:modality == "conversational_session"`, prefer (1).

### REMEDIATE — loop structure

Three mapping entries for one source element:

1. **Modification step** — `m.instruct(format=PatchSchema)` producing a fix
2. **Evaluation step** — `m.instruct(format=VerdictSchema)` checking whether the fix worked
3. **Loop wrapper** — plain Python `while i < MAX_REMEDIATION_ITERATIONS` tying them together

All three route to `pipeline.py`. `MAX_REMEDIATION_ITERATIONS` is always a `config.py` constant with default 3.

### TOOL_TEMPLATE — provisional file routing

Step 2 always routes `TOOL_TEMPLATE` to `tools.py` initially. Step 2.5d amends based on disposition:

- `real_impl` → stays in `tools.py`
- `stub` or `delegate_to_runtime` → moved to `constrained_slots.py`
- `mock` → moved to `fixtures/mock_tools.py`

Record `final_target_file: "pending_step_2.5"` in the mapping entry until Step 2.5d runs.

---

## Dialect-specific overrides

The dialect mapping table in `docs/dialects/<runtime>.md` takes precedence over the general table above.

Precedence (highest first):

1. Dialect doc's mapping table (source-signal-specific rows)
2. Alternative rules above (tag-specific cases)
3. General tag-to-primitive table (the default)

Record every dialect override with `dialect_override_applied: "<runtime>:<row>"` in the mapping entry.

---

## When LLM judgement is invoked

Step 2 is mechanical wherever possible. LLM invocation is bounded to:

- `VALIDATE_OUTPUT` semantic-vs-executable classification when phrase-match heuristic is inconclusive
- `CONVERSE` realisation selection when element phrasing doesn't match the three rules
- `DETERMINISTIC` placement when length is borderline and call graph is unclear
- `EXTRACT` two-step eligibility in rare cases where schema analysis is ambiguous

Each invocation is scoped to a single element. Output goes into `intermediate/element_mapping_judgment_calls.json`.

---

## Output: `element_mapping.json`

```json
{
  "mapping_id": "map_001",
  "element_id": "elem_042",
  "target_file": "pipeline.py",
  "target_symbol": "run_pipeline",
  "primitive": "m.instruct",
  "primitive_details": {
    "format_schema": "TriageVerdict",
    "grounding_context_keys": ["ticket_text", "operating_rules"]
  },
  "final_target_file": "pipeline.py",
  "step_2_confidence": 0.9,
  "step_2_rationale": "DECIDE tag with clear enum output → m.instruct with format=DecisionSchema",
  "llm_judgement_required": false,
  "dialect_override_applied": null,
  "validation_kind": null
}
```

**Cross-checks before Step 2 declares done**:

- Count of mapping entries equals count of inventory entries (plus expansions for two-step and remediation)
- Every `NO_DECOMPOSE` element has a mapping entry with `primitive: "none"`
- No mapping entry has empty `target_file` or `target_symbol` (except `NO_DECOMPOSE`)
- Every `target_file` named is in the shape doc's always-emitted list or a conditional file whose trigger is predicted to fire
- Every `dialect_override_applied` non-null value references a real row in the detected runtime's dialect doc

Failure at any check is a generation-halt error. `.melleafy-partial/` retains the intermediate artifacts for debugging.

---

## Output: `expected_signature.json` (P3.5.D — fixture/signature alignment)

After emitting `element_mapping.json`, derive and emit `intermediate/expected_signature.json` — the canonical `run_pipeline` I/O signature. This artefact:

- **Locks** the signature for Step 4 fixture generation (every fixture's `inputs` dict has keys matching `expected_signature.inputs[].name`).
- **Constrains** Step 5 descriptor emission (the prompt inlines `expected_signature` as a HARD CONSTRAINT block — see `mellea-fy-generate.md` §"Descriptor mode").
- **Enforces** signature parity at validation via `R-SEM-SIGNATURE-MATCH` (descriptor's `inputs`/`outputs`/`schemas` must match exactly).

### Derivation rules (deterministic)

Given the same inputs (`element_mapping.json`, `classification.json`, `inventory.json`, the spec text), produce the same `expected_signature.json`. The derivation:

1. **`function_name`** is always `"run_pipeline"` (per Rule 3-2 in `mellea-fy-generate.md`).

2. **`inputs[]`** — read in declaration order:
   - From `classification.json:modality`, apply the modality-specific entry-point shape (`mellea-fy-generate.md` Rule R21):
     - `synchronous_oneshot` / `streaming` / `review_gated` / `realtime_media` → user-provided params.
     - `conversational_session` → first input is always `{name: "session_id", type: "str"}`, then user-provided params.
     - `event_triggered` → single input `{name: "event", type: "dict"}`.
     - `scheduled` → empty inputs list.
     - `heartbeat` → single input `{name: "state", type: "dict"}`.
   - For each `TOOL_INPUT` mapping entry routed to a pipeline parameter (not `loader.py`), append `{name: <target_symbol>, type: <python_type_from_inventory>}`.
   - For each `CONVERSE` mapping entry with realisation (2) (parameter-with-default), append the parameter.
   - If the spec is untyped or ambiguous (Rule 3-1), default `type` to `"str"`.

3. **`outputs[]`** — one entry per top-level pipeline return value:
   - For modalities returning a structured payload (`synchronous_oneshot`, `review_gated`), one entry whose `type` is the schema name from the final `m.instruct(format=Schema)` mapping entry.
   - For `streaming`, one entry with `type: "Iterator[str]"`.
   - For `scheduled` / `event_triggered`, an empty list (returns `None`).
   - For `heartbeat`, one entry with `type: "dict"`.
   - The first entry's name is conventional: `result` when the output is a primitive, otherwise snake_case of the schema name (e.g. `findings_report` for `FindingsReport`).

4. **`schemas[]`** — every schema referenced by `inputs[].type` or `outputs[].type` (and transitively via field refs), produced from `SCHEMA`-tagged inventory entries:
   - `name` is the PascalCase schema name from the inventory.
   - `kind` is `"model"` for Pydantic models, `"enum"` for `Literal[...]` enums.
   - `fields[]` mirrors the schema's declared fields, in declaration order.
   - `optional` is `true` iff the schema field is annotated `Optional[T]` / `T | None` in the source.

5. **`source_element_refs`** — record the `element_id` values that contributed to the signature derivation (provenance for repair-loop debugging).

6. **`modality`** — copy `classification.json:modality` verbatim.

### Worked example

For a sentry-style code-review skill whose `classification.modality = "synchronous_oneshot"`, with one `TOOL_INPUT` element `elem_004` mapped to a `diff: str` parameter and one final `m.instruct(format=FindingsReport)` call:

```json
{
  "format_version": "1.0",
  "function_name": "run_pipeline",
  "inputs": [
    {"name": "diff", "type": "str"}
  ],
  "outputs": [
    {
      "name": "findings_report",
      "type": "FindingsReport",
      "schema_ref": "#/schemas/FindingsReport"
    }
  ],
  "schemas": [
    {
      "name": "FindingsReport",
      "kind": "model",
      "fields": [
        {"name": "severity", "type": "str"},
        {"name": "issues", "type": "list[str]"}
      ]
    }
  ],
  "source_element_refs": ["elem_004", "elem_017"],
  "modality": "synchronous_oneshot"
}
```

### Cross-checks before Step 2 declares done (signature side)

- Every `inputs[].name` and `outputs[].name` is a valid Python identifier.
- Every schema referenced by an input/output `type` is declared under `schemas[]`.
- The derivation is repeatable: a re-run on the same `element_mapping.json` + `classification.json` + `inventory.json` produces a byte-identical `expected_signature.json`.
