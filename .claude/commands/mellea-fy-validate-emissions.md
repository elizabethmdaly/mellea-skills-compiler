# Melleafy Step 5b: In-session Validation of Step 5 Emissions

**Version**: 1.0 (2026-05-17) | **Prereq**: Step 5 complete | **Produces**: `intermediate/step_5b_report.json`

Step 5b is an **in-session, opportunistic structural-lint gate** that runs immediately after Step 5 has emitted the per-element Python code bodies and BEFORE Step 6 begins. Its job is to catch the small set of structural defects that empirically slip through Step 5 and only surface at Step 7 (by which point the session is already past the point of cheap repair).

The lint RULES applied here are a subset of Step 7's — Step 5b does NOT re-implement them, it teaches Claude (you) to apply them mentally while still in the headspace of the just-emitted files. This is feasible because:

- Each rule is mechanical and locally checkable (signature lookup + arity + kwarg match).
- The canonical signatures are already on disk at `intermediate/mellea_api_ref.json`.
- The session has the `Read` and `Edit` tools — sufficient to enumerate call sites and patch them in place.

The wrapper-side Step 7 lint suite (`/mellea-fy-validate`) remains the post-session safety net. Step 5b is purely a faster-feedback layer; it does NOT replace Step 7.

---

## Purpose

Catch structural defects in the Step 5 emissions **immediately**, inside the same Claude session, while the just-emitted files are fresh. This is the realistic prevention layer for the empirically dominant failure mode: hallucinated Mellea-stdlib calls (wrong arity, wrong import path, missing required argument).

Empirical motivation (2026-05-17 evidence): 5 separate skills produced the same `check(req)` arity defect in `requirements.py` despite the canonical `check(requirement, output)` signature being present in `intermediate/mellea_api_ref.json`. Step 7 caught all 5 — but by then the session had already exited and the in-session repair loop never fired. Step 5b closes that gap.

---

## Scope

Step 5b applies to files emitted by **Step 5 only**:

- `pipeline.py`
- `requirements.py`
- `slots.py`
- `tools.py`
- `constrained_slots.py`
- `mobjects.py`
- `loader.py`

Step 5b does NOT apply to:

- Step 6 outputs (`melleafy.json`, `mapping_report.md`, `README.md`, `SETUP.md`, `SKILL.md`).
- `config.py` (rendered deterministically by the wrapper from `config_emission.json`).
- `schemas.py` (Step 3 skeleton — body lives there but the lints below target Mellea-stdlib call sites, not Pydantic BaseModel definitions).
- Files under `fixtures/` (Step 4 — has its own contract).

---

## The 9-lint subset (rules to apply mentally)

Each rule is the SAME rule that `src/mellea_skills_compiler/compile/lints.py` enforces at Step 7. The difference is the gate: Step 5b is in-session and opportunistic; Step 7 is wrapper-side and authoritative.

1. **`stdlib-arity`** — every call to a known `mellea.stdlib.*` function (`m.instruct`, `req`, `check`, `simple_validate`, …) MUST match the canonical signature: positional argument count within `[min_pos, max_pos]` and every keyword name in the canonical `valid_kwargs`. The canonical lookup table:
   - Static-table priority (these are stable across Mellea versions):

     | Function          | Required positional         | Optional keyword |
     | ----------------- | --------------------------- | ---------------- |
     | `simple_validate` | 1 (`fn`)                    | none             |
     | `req`             | 1 (`description`)           | `validation_fn`  |
     | `check`           | 2 (`requirement`, `output`) | none             |

   - For anything else: look it up at `intermediate/mellea_api_ref.json:.modules.<module>.<symbol>.signature`.

   **The dominant Step 5 failure mode is `check(arg)` (1-arg) where the canonical is `check(requirement, output)` (2-arg). When in doubt, prefer `req(description)` (1-arg factory) over `check(...)`.**

2. **`import-soundness`** — every `from mellea.X import Y` (and `import mellea.X`) statement MUST have `mellea.X` as a key in `intermediate/mellea_api_ref.json:.modules`. Common defects: shortened paths (`mellea.model_options` when symbol lives at `mellea.backends.model_options`), or invented paths (`mellea.backends.tools` — does not exist).

3. **`instruct-has-description`** — every `m.instruct(...)` call MUST include a `description` (positional first argument OR `description=` keyword). Missing `description` → `TypeError` at runtime.

4. **`instruct-result-parse-before-access`** — accessing a field on a thunk returned by `m.instruct(format=Model)` MUST be preceded by `Model.model_validate_json(thunk.value)` (or one of the documented helpers: `_parse_instruct_result`, `_safe_parse_with_fallback`). Bare `.field_name` or `.model_dump()` on the thunk fails at runtime.

5. **`format-annotation`** — the `format=` kwarg in `m.instruct(...)` MUST be a `BaseModel` subclass (a class reference, not a string, not an instance), or be omitted entirely.

6. **`validator-soundness`** — in `requirements.py`, every `validation_fn=` either uses `simple_validate(fn)` or a function with signature `(ctx, result) -> ...`. No vacuous lambdas (a lambda that always returns `True` regardless of input is a defect).

7. **`generative-forbidden-params`** — any `@generative`-decorated function in `slots.py`, `constrained_slots.py`, or `pipeline.py` MUST NOT use a reserved Mellea parameter name. The forbidden set is sourced from `intermediate/mellea_api_ref.json:.forbidden_param_names`, with this static fallback when grounding is unavailable: `m`, `context`, `backend`, `model_options`, `strategy`, `precondition_requirements`, `requirements`, `f_args`, `f_kwargs`. Common defect: a parameter named `context` — must be renamed (`surrounding_context`, `finding_context`, etc.).

8. **`generative-call-passes-session`** — every call site of a `@generative`-decorated function MUST pass the session: either (a) at least one positional arg (the `m` session), (b) `m=...` kwarg, or (c) both `context=...` AND `backend=...`. The common defect is `with start_session(BACKEND, MODEL_ID):` (no `as m`) followed by `slot(text=...)` (no positional `m`).

9. **`variable-safety`** — no use of an undefined name. In particular: variables referenced inside `except` / `finally` MUST be initialised before the enclosing `try`. No shadowing of Python builtins in function argument names.

---

## Step-by-step workflow

Run these steps autonomously. Do NOT prompt for confirmation between steps.

### 1. Enumerate emitted files

List which of the 7 in-scope files Step 5 actually emitted (some are conditional — a skill with no slots will have no `slots.py`). Use the `Read` tool to confirm presence by attempting to read each path under `<package_name>/`. Record the resulting set in `files_checked` of the report.

### 2. Load the canonical surface

Read `intermediate/mellea_api_ref.json`. Hold the `.modules` map and `.forbidden_param_names` list (if present) in working memory for the rest of Step 5b. If the file is absent OR `grounding_unavailable: true`, fall back to the static table from Rule 1 and the static forbidden-param list from Rule 7; the remaining rules become best-effort.

### 3. For each in-scope file, enumerate call sites

Read the file. Mentally identify every Mellea-stdlib call site (any `Call` whose callee is `m.instruct`, `req`, `check`, `simple_validate`, or any other `mellea.stdlib.*` function referenced in the file). For each call site, apply rules 1-5 as applicable. Also identify:

- Every `from mellea...` / `import mellea...` import (rule 2).
- Every `@generative`-decorated function definition (rule 7).
- Every call to a `@generative`-decorated function (rule 8).
- Every variable reference inside `except` / `finally` (rule 9).

### 4. Fix in place

When a defect is found:

- For arity/kwarg defects (rule 1): rewrite the call to match the canonical signature. If you cannot resolve the canonical and the call is a `check(...)`, prefer rewriting to `req(...)` (1-arg factory) when the semantic intent is "express a requirement" rather than "check an output".
- For import defects (rule 2): rewrite the import path to one that resolves in `mellea_api_ref.json:.modules`. If no such path exists for the imported symbol, mark the issue as `remaining_issues` (do not invent a path).
- For missing-`description` (rule 3): add a descriptive first positional argument summarising the intent of the instruct call (use surrounding code as context).
- For unparsed-thunk-access (rule 4): wrap with `Model.model_validate_json(thunk.value)` and reassign before the field access.
- For `format=` defects (rule 5): replace string / instance with the BaseModel class reference, or remove the kwarg.
- For validator defects (rule 6): rewrite the `validation_fn=` argument to use `simple_validate(...)` or a `(ctx, result) -> bool` function.
- For forbidden-param names (rule 7): rename the parameter (suggested replacements above).
- For missing-session-at-call (rule 8): edit the `start_session(...)` context manager to bind `as m`, and pass `m` positionally at the call site.
- For variable-safety defects (rule 9): initialise the variable before the `try` block, or rename the shadowed builtin.

Use the `Edit` tool for each fix. Record one entry in `fixes_applied` per edit: `{file, line, issue, fix_summary}`.

### 5. Re-enumerate

After all edits, re-read each modified file and re-apply the rule that triggered the edit. If the defect persists, record it in `remaining_issues` and proceed; do NOT loop indefinitely. The maximum repair effort is one pass per call site.

### 6. Emit the report

Write `intermediate/step_5b_report.json` using `Write`. The structure (governed by `src/mellea_skills_compiler/descriptor/schemas/step_5b_report.schema.json`):

```json
{
  "format_version": "1.0",
  "verdict": "pass",
  "files_checked": ["pipeline.py", "requirements.py", "slots.py"],
  "call_sites_enumerated": 17,
  "fixes_applied": [],
  "remaining_issues": []
}
```

Determine `verdict`:

- `pass` — zero defects detected across all files. `fixes_applied` and `remaining_issues` both empty.
- `fixed` — at least one defect was found AND every defect was repaired in place. `fixes_applied` non-empty, `remaining_issues` empty.
- `fail` — at least one defect could not be repaired in-session. `remaining_issues` non-empty.

Step 5b is complete only after the report is written.

---

## Exit criteria

Step 5b declares completion when `intermediate/step_5b_report.json` exists and conforms to the schema. After the file is written, IMMEDIATELY proceed to Step 6 (per the orchestrator's autonomous-execution directive — no narrative pause between steps).

Step 5b is **opportunistic**. A `verdict: "fail"` does NOT halt the pipeline. The wrapper-side Step 7 lint will catch any residue and engage the post-session repair loop (Fix A, `--repair-on-lint-failure`) if enabled. The deeper follow-up (F9 — extending the descriptor IR to cover the auxiliary files so these defects can be caught at descriptor-validate time before any rendering happens) is tracked separately in `melleafy-handoff/analyses/2026-05-17-lint-severity-repair-loop-followups.md`.

---

## Failure semantics

| Step 5b verdict | Step 5b behaviour | Step 7 still runs? |
|---|---|---|
| `pass` | Proceed to Step 6 immediately. | Yes — authoritative gate. |
| `fixed` | Proceed to Step 6 immediately. Fixes are persisted on disk. | Yes — confirms the in-session edits cleared the defects. |
| `fail` | Proceed to Step 6 anyway. Report records the unfixable issues so the post-session repair loop (if enabled) can act on the same evidence. | Yes — Step 7 IS the authoritative gate. |

Step 5b never raises, never halts, never short-circuits. It only produces structured evidence + best-effort in-session edits.

---

## What Step 5b does NOT check

- Cross-artefact consistency (e.g. `melleafy.json` field agreement) — owned by Step 6 + Step 7's `melleafy-json-consistency` lint.
- Schema priming (KB5, `session-boundary` lint) — needs cross-cutting view of all `m.instruct(format=...)` types within a `start_session()` block. Out of scope for the in-session opportunistic gate; lives in Step 7.
- Configuration drift (`runtime-defaults-bound`) — `config.py` is renderer-emitted, not LLM-emitted.
- Fixture signature binding — owned by Step 7's `fixture-signature-bound` lint.
- Asset path resolution — owned by Step 7's `bundled-asset-path-resolution` lint.
- Anything `category-specific` (secret scanning, MCP tool naming, …) — owned by Step 7 Tier 3.

These checks are EITHER cross-artefact (Step 5b can't see all the inputs at once) OR target wrapper-rendered files (Step 5b doesn't own those). Step 7 stays the single source of truth for the full lint suite.
