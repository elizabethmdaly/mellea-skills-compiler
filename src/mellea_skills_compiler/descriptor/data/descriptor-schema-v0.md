# Descriptor Schema v0 — Strawman

**Purpose**: Anchor for Phase 0.2. The compiler engineer hand-authors descriptors against this schema; gaps and friction points surface as Phase 0 evidence, not as final design.

**Status**: STRAWMAN. Expect to change during Phase 0.2 as real skills push on it. Track changes in this doc; do not silently extend.

**Authority**: This schema is **not** authoritative for Phase 1. Phase 1's `descriptor.schema.json` is the production version, informed by what Phase 0.2 learned.

## Design constraints (recap from plan §4)

1. Mellea coverage is **comprehensive by introspection** — every public Mellea symbol is callable. No catalog gate.
2. Composition operators are **5, curated** — `sequential`, `branch`, `parallel`, `map`, `retry_with_feedback`.
3. The LLM emits **JSON** (canonical form). YAML is for human review only.
4. **Reasoning happens before structure** (Tam et al. mitigation, plan §4.4). The descriptor is emitted after a free-form reasoning step, not during it.
5. No Python in the descriptor. Period. All Python emission is the renderer's job.

## Top-level shape

```json
{
  "descriptor_version": "0.1",
  "mellea_version": "0.5.0",
  "skill": {
    "name": "string",
    "classification": "string (DSL | COMP | DOM | MIXED)"
  },
  "inputs": [ /* see Inputs */ ],
  "outputs": [ /* see Outputs */ ],
  "schemas": { /* see Schemas */ },
  "state": [ /* see State */ ],
  "pipeline": [ /* see Pipeline */ ]
}
```

## Inputs and Outputs

Both are arrays of typed slots.

```json
{
  "name": "document",
  "schema": {
    "kind": "pydantic_model",
    "ref": "#/schemas/Document"
  },
  "description": "free-form description for prompts"
}
```

`schema` resolves either inline or via `ref` to `#/schemas/<name>`.

## Schemas

Inline Pydantic-equivalent schema definitions. Used for inputs, outputs, and `format=` arguments.

```json
{
  "schemas": {
    "Document": {
      "kind": "model",
      "fields": {
        "text": { "type": "str" },
        "metadata": { "type": "dict[str, str]", "optional": true }
      }
    },
    "Severity": {
      "kind": "enum",
      "members": ["critical", "high", "medium", "low"]
    },
    "Finding": {
      "kind": "model",
      "fields": {
        "file_line": { "type": "str" },
        "severity": { "type": "enum_ref", "ref": "#/schemas/Severity" },
        "problem": { "type": "str" },
        "evidence": { "type": "str" }
      }
    }
  }
}
```

**Supported field types (v0):** `str`, `int`, `float`, `bool`, `dict[K, V]`, `list[T]`, `enum_ref`, `model_ref`, `optional`. Mellea-specific types (e.g. `Requirement`) are referenced through the call-vocabulary, not inlined as schema fields.

## State

Module-level Mellea setup. Each entry is a typed call to a Mellea symbol; the renderer emits the call as a module-level assignment.

```json
{
  "state": [
    {
      "id": "default_backend",
      "symbol": "mellea.backends.openai.OpenAIBackend",
      "args": {
        "model_id": { "env": "MODEL_ID" }
      }
    },
    {
      "id": "session",
      "symbol": "mellea.stdlib.session.MelleaSession",
      "args": {
        "backend": { "ref": "default_backend" }
      }
    }
  ]
}
```

`id` is local to the descriptor. The renderer turns it into a Python variable name (e.g. `default_backend`, `session`).

## Pipeline — node shapes

A pipeline is an ordered array of nodes. Each node is either a **call node** or a **composition node**.

### Call node

```json
{
  "id": "classify_step",
  "kind": "call",
  "symbol": "mellea.stdlib.session.MelleaSession.instruct",
  "bound_to": { "ref": "session" },
  "args": {
    "description": {
      "template": "Classify this document: {document.text}"
    },
    "format": { "schema_ref": "#/schemas/Finding" },
    "requirements": [
      {
        "symbol": "mellea.stdlib.requirements.Requirement",
        "args": {
          "description": "Severity must reflect documented evidence."
        }
      }
    ]
  },
  "captures": { "result": "#/outputs/result" }
}
```

**Argument value kinds:**
- `{ "value": <literal> }` — JSON literal passed through (int, str, bool, list of literals).
- `{ "template": "string with {var} interpolation" }` — f-string-style; vars resolve against `inputs`, prior node `id`s, and `state`.
- `{ "ref": "<id>" }` — reference to a state entry or a prior pipeline node.
- `{ "schema_ref": "#/schemas/<name>" }` — Pydantic schema by ref.
- `{ "env": "<NAME>" }` — environment variable (renderer emits `os.environ["NAME"]`).
- `{ "symbol": "...", "args": {...} }` — nested call construction (e.g. for building a `Requirement` inline).
- Lists of any of the above.

`bound_to` is required for instance methods (introspection determines this), omitted for free functions and classmethods.

`captures` declares which output slot a node fills. Optional — most nodes capture nothing.

### Composition node

```json
{
  "id": "triage_branch",
  "kind": "composition",
  "operator": "branch",
  "on": { "ref": "classify_step", "select": "severity" },
  "cases": {
    "critical": [
      { "id": "escalate", "kind": "call", "symbol": "...", "args": {} }
    ],
    "high": [
      { "id": "notify", "kind": "call", "symbol": "...", "args": {} }
    ],
    "_default": [
      { "id": "log_only", "kind": "call", "symbol": "...", "args": {} }
    ]
  }
}
```

The 5 composition operators:

| `operator` | Required fields | Shape |
|---|---|---|
| `sequential` | (implicit when nodes are array siblings) | Array of nodes; outputs flow by reference |
| `branch` | `on`, `cases` | `cases` is a map from value → array of nodes; `_default` is the else branch |
| `parallel` | `branches` | Array of arrays of nodes; results combined by `collect` (default: dict by branch id) |
| `map` | `over`, `body`, `collect` | Iterates a node body over a list reference; `collect: list \| dict_by_key` |
| `retry_with_feedback` | `body`, `max_attempts`, `failure_check` | Runs `body`; if `failure_check` ref evaluates falsy, retries with the failure as feedback |

**Composition operators are extensible only by RFC (plan §4.5).** Don't invent new operators in Phase 0.2 — flag the need and bring it to the daily sync.

## Worked example — `sentry-find-bugs` (sketch only, not full)

```json
{
  "descriptor_version": "0.1",
  "mellea_version": "0.5.0",
  "skill": { "name": "find-bugs", "classification": "DSL" },

  "inputs": [
    { "name": "diff", "schema": { "kind": "pydantic_model", "ref": "#/schemas/BranchDiff" } }
  ],
  "outputs": [
    { "name": "findings", "schema": { "kind": "list", "of": { "ref": "#/schemas/Finding" } } }
  ],

  "schemas": {
    "BranchDiff": {
      "kind": "model",
      "fields": { "files": { "type": "list[str]" }, "patch": { "type": "str" } }
    },
    "Severity": { "kind": "enum", "members": ["critical", "high", "medium", "low"] },
    "Finding": {
      "kind": "model",
      "fields": {
        "file_line": { "type": "str" },
        "severity": { "type": "enum_ref", "ref": "#/schemas/Severity" },
        "problem": { "type": "str" },
        "evidence": { "type": "str" },
        "fix": { "type": "str" }
      }
    }
  },

  "state": [
    { "id": "backend", "symbol": "mellea.backends.openai.OpenAIBackend",
      "args": { "model_id": { "env": "MODEL_ID" } } },
    { "id": "session", "symbol": "mellea.stdlib.session.MelleaSession",
      "args": { "backend": { "ref": "backend" } } }
  ],

  "pipeline": [
    {
      "id": "map_attack_surface",
      "kind": "call",
      "symbol": "mellea.stdlib.session.MelleaSession.instruct",
      "bound_to": { "ref": "session" },
      "args": {
        "description": { "template": "For this diff, identify attack-surface elements... {diff.patch}" },
        "format": { "schema_ref": "#/schemas/AttackSurface" }
      }
    },
    {
      "id": "find_findings",
      "kind": "call",
      "symbol": "mellea.stdlib.session.MelleaSession.instruct",
      "bound_to": { "ref": "session" },
      "args": {
        "description": { "template": "Given attack surface {map_attack_surface} and diff {diff.patch}, list HIGH-confidence findings only..." },
        "format": { "schema_ref": "#/schemas/Finding" }
      },
      "captures": { "result": "#/outputs/findings" }
    }
  ]
}
```

This is a sketch; the real Phase 0.2 artifact will be more complete. The point is to show what shape the LLM is producing.

## Validation rules (v0)

A descriptor is valid iff:

1. JSON Schema-valid against the document shape above.
2. `descriptor_version` is supported (v0: `"0.1"`).
3. `mellea_version` matches the introspected surface available (hard-pin in v0; soft compat is post-Phase-1 work).
4. Every `symbol` resolves in the introspected `surface.json`.
5. Every required parameter of every called symbol is satisfied; no unknown parameters.
6. Every `ref` resolves to a declared `state.id` or prior pipeline node `id`.
7. Every `schema_ref` resolves to a declared schema in `#/schemas/`.
8. Composition operator's required fields present per the table above.
9. No Python source code in any field. Templates are interpolation strings, not code.

## Known v0 gaps (deliberate)

These will surface in Phase 0.2; we expect to address some in Phase 1:

- **Async/streaming**: no descriptor-level expression yet. Plan §12 Q6 is open.
- **Per-skill model overrides** (cf. `security-engineer`'s `model: opus`): currently shoehorned into the `state` block via env var. Cleaner shape TBD.
- **Tool allowlists** (cf. `security-review`'s `allowed-tools`): no descriptor expression yet. Likely a skeleton concern.
- **Multi-backend**: plan §12 Q3.
- **User-authored Requirement classes**: plan §12 Q4.

If 0.2 needs any of these, flag in the daily sync. Don't extend the schema silently.

## Change log

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-05-16 | Initial strawman for Phase 0 kickoff. |
