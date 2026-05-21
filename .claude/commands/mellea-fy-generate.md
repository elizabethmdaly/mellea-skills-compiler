# Melleafy Steps 3 + 5: Skeleton Emission and Body Generation

**Version**: 4.3.0 (2026-04-28) | **Prereq**: `dependency_plan.json` (Step 2.5 complete) | **Produces**: Populated Python package

> **Output path rule** (Rule OUT-3): All `.py` files (`pipeline.py`, `config.py`, `schemas.py`, `main.py`, etc.) are written inside `<package_name>/`. `pyproject.toml` is the only file written at the skill root (NOT inside `<package_name>/`). See `mellea-fy.md` §Output directory layout for the full tree.

Step 3 emits skeleton files (imports, signatures, docstring placeholders) from the element mapping and dependency plan. Step 5 invokes the LLM to fill in every code body. Read `/mellea-fy-behaviours` before generating any code — the Known Behaviours mitigations must be baked into every generated file.

---

## Step 3: File set and skeleton emission

### File set determined by mapping + dependency plan

| File                   | When generated                                                                           |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `pipeline.py`          | Always                                                                                   |
| `schemas.py`           | Always (at minimum contains `Final[str]` placeholder if no schemas found)                |
| `config.py`            | Always (persona text, model ID, loop budgets from `dependency_plan.json:bundle` entries) |
| `requirements.py`      | When any `VALIDATE_OUTPUT` elements exist                                                |
| `slots.py`             | When any `EXTRACT` or `CLASSIFY` elements map to `@generative`                           |
| `tools.py`             | When any C6 element has disposition `real_impl`                                          |
| `constrained_slots.py` | When any C6 element has disposition `stub` or `delegate_to_runtime`                      |
| `mobjects.py`          | When any `TRANSFORM` or `QUERY` elements exist                                           |
| `loader.py`            | When any C3 element has disposition `load_from_disk`                                     |
| `main.py`              | Always (CLI entry point)                                                                 |
| `pyproject.toml`       | Always                                                                                   |
| `fixtures/`            | Always (5–8 fixtures, generated in Step 4)                                               |
| `SETUP.md`             | When any C4, C5, C9, non-bundled C6, C7, non-default C8, or host-needing modality        |
| `README.md`            | Always                                                                                   |
| `melleafy.json`        | Always (skeleton in Step 3; finalised in Step 6)                                         |
| `dependencies.yaml`    | When any C6/C7/C8 entry is non-bundle                                                    |

### Modality-specific entry-point shape (R21)

The `run_pipeline` function signature and `main.py` shape vary by modality from `classification.json`:

| Modality                 | Entry point shape                                                           |
| ------------------------ | --------------------------------------------------------------------------- |
| `synchronous_oneshot`    | `run_pipeline(*params) -> OutputSchema` — simple function call              |
| `streaming`              | `run_pipeline(*params) -> Iterator[str]` — generator yielding tokens        |
| `conversational_session` | `run_pipeline(session_id: str, *params) -> OutputSchema` — session-keyed    |
| `review_gated`           | `run_pipeline(*params) -> ReviewRequest` — returns for human approval       |
| `scheduled`              | `run_pipeline() -> None` — no user-provided params; data fetched internally |
| `event_triggered`        | `run_pipeline(event: dict) -> None` — event payload as input                |
| `heartbeat`              | `run_pipeline(state: dict) -> dict` — stateful loop, returns updated state  |
| `realtime_media`         | `run_pipeline(stream: Iterator) -> Iterator` — streaming I/O                |

**Rule 3-1 — `run_pipeline` parameter type annotations**: Every parameter in the generated `run_pipeline` function signature MUST have an explicit Python type annotation. If the source spec declares a type for a parameter (e.g. from a typed function signature, a schema field, or an explicit type note in the spec text), use that type. If the source spec is untyped or ambiguous, default to `str`. Do not emit bare parameter names (e.g. `company_domain`) — emit `company_domain: str` instead. This applies to both required and optional (defaulted) parameters.

**Rule 3-2 — `run_pipeline` is the canonical entry point name**: The top-level entry function in `pipeline.py` MUST be named exactly `run_pipeline` — not `run_phase_1`, not `run_assessment`, not any other `run_*` variant. The smoke-check loader at `toolkit/file_utils.py:load_skill_pipeline` uses `melleafy.json:entry_signature` as the authoritative source of truth for which function to invoke, with `run_pipeline` as the fallback when the manifest is absent. Empirically observed regression: a package defining `run_phase_2_gap_analysis`, `run_phase_3_roadmap`, and `run_pipeline` as public top-level functions caused the pre-fix loader to pick `run_phase_2_gap_analysis` (alphabetically first under `dir()`) and crash at fixture smoke-check with a TypeError. Public helper functions named `run_<phase>` are PERMITTED alongside `run_pipeline` — the loader's manifest-driven discovery handles the disambiguation — but `run_pipeline` MUST be present. Step 5 records the canonical signature in `melleafy.json:entry_signature`. The `pipeline-entry-canonical` lint enforces this at Step 7.

**Rule 3-3 — `run_pipeline` is decorated with `@validate_call` for dict-coercion at the entry-point boundary**: The renderer emits `@validate_call(config={"arbitrary_types_allowed": True})` (from `pydantic`) immediately before the `def run_pipeline(...)` signature, and adds `from pydantic import validate_call` to the rendered imports. This makes the entry point's type annotations *enforced and coercive at call time* — callers may pass plain dicts where Pydantic models are typed (the natural shape for JSON-emitted fixtures and external orchestrators), and the dict is coerced to the declared model before the function body runs. `arbitrary_types_allowed=True` lets the decorator accept parameters typed as things Pydantic doesn't know natively (e.g., a session handle, a `Callable[...]` for delegated tools). This closes a class of empirically observed bugs of the form `'dict' object has no attribute 'model_copy'`, where a dict-typed argument reaches downstream code that assumes a Pydantic instance. **Internal helper functions** (those emitted by composition operators like `parallel` / `agent_loop` / `human_approval`, or `_<name>`-prefixed module-level helpers) are NOT decorated — their callers are inside the pipeline where types are already correct, and per-call validation adds non-trivial overhead. The decorator is emitted for **all** descriptor versions (v0.1, v0.2, v0.3+); the contract is the same. Applies equally to `async def run_pipeline(...)` when `skill.async == true`. The renderer guarantees emission idempotency — re-render produces byte-identical output (no duplicate decorator, no duplicate import).

### Step 3a-pre: Bundled assets are already mirrored (Rule OUT-6)

Companion directories from the skill root (`scripts/`, `references/`, `assets/`) are mirrored into `<package_name>/` **deterministically by the compile pipeline**, _before_ mellea-fy runs. The model does not perform the copy — it is plumbing handled by `mellea_skills_compiler.compile.mellea_skills._mirror_companion_dirs`. By the time Step 3 begins, any companion directory that existed at the skill root is already present at `<package_name>/<dir>/` and can be referenced directly.

The model's responsibility is **path-resolution discipline**: any code emitted in Step 5 that loads or invokes a bundled asset MUST resolve its path package-relatively via `Path(__file__).parent / "<dir>/<file>"`, **not** via a user-supplied `repo_root` argument or the process working directory. This invariant is what makes the generated package self-contained — a `pip install`-ed package, or one invoked from any cwd, finds its bundled assets via Python's own module-location machinery.

Step 7's `bundled-asset-path-resolution` lint catches violations at validation time. The `pyproject.toml` template (below) declares these directories under `[tool.setuptools.package-data]` so the mirrored copies ship with the installed wheel.

### Skeleton contents per file

**config.py**: C1 and C2 bundle entries become `Final[str]` constants under `# === C1: Identity ===` and `# === C2: Operating Rules ===` section headers. C8 bundle entries (model ID, backend) also here. Every constant gets a `# PROVENANCE: <source_file>:<source_lines>` comment. **In Step 5, the model emits JSON conforming to `.claude/schemas/config_emission.schema.json` — not Python source. The writer at `.claude/melleafy/writers/config_writer.py` renders the file.**

> **C8 backend rule**: `BACKEND` and `MODEL_ID` values are injected via the system prompt by the compile pipeline (sourced from `.claude/data/runtime_defaults.json`, with optional `--backend` / `--model-id` CLI overrides). Emit them in `config.py` exactly as instructed in the system prompt; do not invent alternatives. The Step 7 `runtime-defaults-bound` lint enforces this — divergence from the injected values is a hard failure.

> **Fallback when the model does not emit `config_emission.json`**: if the slash command exits without writing `intermediate/config_emission.json` at all, the wrapper synthesises one deterministically from `intermediate/runtime_directive.json` (which the wrapper writes before the model session starts). The synthesised emission contains only the C8 `BACKEND` and `MODEL_ID` constants — enough for the deterministic writer to render a schema-compliant `config.py` and for the `runtime-defaults-bound` lint to pass. This means `runtime-defaults-bound` never fails purely because the model omitted the IR; it only fails on genuine drift between the emitted values and the injected runtime defaults. The model should still emit `config_emission.json` with the full constant set (PREFIX_TEXT, LOOP_BUDGET, etc.) — the fallback is the last line of defence, not the intended path.

_JSON the model emits:_

```json
{
  "constants": [
    {
      "name": "PREFIX_TEXT",
      "value": "<persona text from SOUL.md>",
      "type": "str",
      "category": "C1",
      "provenance": { "source_file": "SOUL.md", "source_lines": "1-45" }
    },
    {
      "name": "BACKEND",
      "value": "ollama",
      "type": "str",
      "category": "C8"
    },
    {
      "name": "MODEL_ID",
      "value": "granite4.1:8b",
      "type": "str",
      "category": "C8"
    },
    {
      "name": "LOOP_BUDGET",
      "value": 3,
      "type": "int"
    }
  ]
}
```

_Python source the writer renders from that JSON:_

```python
from typing import Final

# === C1: Identity & Behavioral Context ===
PREFIX_TEXT: Final[str] = """You are an AI assistant.\nYou help users with research tasks."""
# PROVENANCE: SOUL.md:1-45

# === C8: Runtime Environment ===
BACKEND: Final[str] = 'ollama'
MODEL_ID: Final[str] = 'granite4.1:8b'

LOOP_BUDGET: Final[int] = 3
```

> **Scalar-only constraint (Amendment K).** `config_emission.json` constants are scalar-only — each entry's `type` MUST be one of `"str" | "int" | "float" | "bool"`. Dict and list literals do NOT belong in `config_emission.json`; emit them in `pipeline.py` instead (either reconstructed from scalar constants, or inlined as a Python literal). The deterministic writer hard-fails any non-scalar entry and the schema-violation cannot be auto-repaired. **See `mellea-fy-behaviours.md` § ACCEPT-SET-1 for the full rule, the two alternative patterns, and the worked example.**

**schemas.py**: One Pydantic `BaseModel` per `SCHEMA` element. Field descriptions pulled from the spec's output format description. For two-step pattern: include both the simplified raw schema and the full schema.

**requirements.py**: One `Requirement` object per `VALIDATE_OUTPUT` element. Group by spec section. Structural checks use `simple_validate()`; semantic checks use bare `Requirement(description=...)`. Include `check_only=True` for negative constraints.

**slots.py**: One `@generative` function per `EXTRACT`/`CLASSIFY` element. Return types:

- Classifications: `-> Literal[...]` — Ollama supports constrained decoding, so always use the typed return.
- Simple list extractions: `-> str` with "Set `result` to a comma-separated string of..." docstring — never `-> list[str]`. Split in `pipeline.py`: `[p.strip() for p in raw.split(",") if p.strip()] if raw.strip() else []`
- Structured extractions: Pydantic `BaseModel` (model fields provide the JSON structure; no bare-output risk)

Docstrings on all `@generative` slots MUST reference `result` explicitly (e.g. "Set `result` to..."). Never use "Reply with exactly one word", "Output only", "Return only". Simple schemas (≤4 fields, no `Literal` constraints, no nested models) go here directly. Complex schemas: slot extracts simplified version, Step 5 generates the `m.instruct` enrichment step inline in `pipeline.py`.

**`@generative` definition convention**: The decorator forbids `m` as a parameter name — passing it in the definition raises `ValueError` at import time. Function body must be `...`. `m` is passed as the first positional argument only at call time in `pipeline.py`.

```python
# CORRECT — no m in definition, body is ...
@generative
def extract_sentiment(text: str) -> str:
    """Set `result` to one of: positive, negative, neutral."""
    ...

# WRONG — raises ValueError: cannot create a generative slot with disallowed parameter names: ['m']
@generative
def extract_sentiment(m, text: str) -> str:
    ...
```

Calling convention in `pipeline.py` (unchanged — `m` passed as first positional arg):

```python
with start_session(BACKEND, MODEL_ID) as m:
    sentiment = extract_sentiment(m, text=user_input)
```

**tools.py** (when any C6 has disposition `real_impl`):

- Domain/command allowlist — hard-coded `ALLOWED_DOMAINS` or `ALLOWED_COMMANDS`
- Error handling with timeouts and HTTP error codes
- Auth tokens read from environment variables
- `build_api_params()` or equivalent mapping spec-level names to API parameter names
- **Bundled-script invocation (Rule OUT-6)**: when the implementation invokes a script mirrored from the skill root (e.g. `scripts/bash/check-prerequisites.sh`), resolve the script path package-relatively. Use `Path(__file__).parent / "scripts/<...>"` — never `Path(repo_root) / "scripts/<...>"` and never rely on the process working directory. Example: `script_path = Path(__file__).parent / "scripts" / "bash" / "check-prerequisites.sh"`. The mirror is established by Step 3a-pre, so the path is guaranteed to resolve at runtime regardless of where the package is invoked from.
- Example structure:

```python
ALLOWED_DOMAINS = ["api.example.com"]
HTTP_TIMEOUT = 10

def http_get(url: str) -> str:
    """Fetch content from a URL. Validates against domain allowlist."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain '{parsed.hostname}' not in allowlist")
    # ... execute with timeout and error handling
```

**Multi-mode subprocess wrappers**: if a shell script exposes multiple modes (e.g. `--check-prompt`, `--check-url`, `--check-command`), inspect how each mode receives its input before generating the wrapper:

- Modes that read from stdin (e.g. `input=$(cat)` in the script body) → call `subprocess.run(..., input=target, capture_output=True, text=True)`. Do NOT append `target` to the command list.
- Modes that take a positional argument → append `target` to the command list as before.

Apply this per-mode distinction at the call site in `pipeline.py` too: every branch calling the wrapper must forward the input correctly for that mode. A branch that omits `target=` for a stdin-consuming mode is a silent false-negative bug.

**constrained_slots.py** (when any C6 has disposition `stub` or `delegate_to_runtime`):

- Implements `ConstrainedGenerativeSlot`, `constrained` decorator, `filter_actions` locally
- Implements `ReactTool` / `ReactToolbox` locally
- Provides stub tool functions raising `NotImplementedError` with implementation instructions
- Wraps dependent slots from `slots.py` with `constrained()` — does NOT duplicate them

**pipeline.py** (standard structure):

- One function per `ORCHESTRATE` workflow
- `with start_session(BACKEND, MODEL_ID) as m:` context manager
- Calls slots, requirements, and mobjects from other files
- `DECIDE` logic: Python `if/elif/else` wrapping Mellea calls
- MUST convert all non-string `grounding_context` values to `str()`
- MUST use `format=PydanticModel` for every `m.instruct()` that produces structured output
- MUST include a description argument on every `m.instruct(...)` call — either as the first positional argument or as a `description=` keyword. Mellea 0.5+ requires this; calls without it crash at runtime with `TypeError: instruct() missing 1 required positional argument: 'description'`. For format-only schema-extract calls where the natural-language instruction is implicit in the schema, emit a generated fallback description, e.g.:

  ```python
  m.instruct(
      f"Produce a {ModelName} per the schema, grounded in the provided context.",
      format=ModelName,
      ...
  )
  ```
- MUST parse the thunk after every `m.instruct(format=Model)` before accessing any field or calling any Pydantic method. `m.instruct()` returns a `ComputedModelOutputThunk` — NOT a Pydantic model. Direct field access (`thunk.field_name`) or `.model_dump()` raises `AttributeError`. Always include `_parse_instruct_result` and `_safe_parse_with_fallback` helpers in `pipeline.py` and call them immediately after every `m.instruct(format=Model)` call:

  ```python
  def _parse_instruct_result(thunk, model_class):
      return model_class.model_validate_json(thunk.value)

  def _safe_parse_with_fallback(thunk, model_class, **fallback_kwargs):
      try:
          return model_class.model_validate_json(thunk.value)
      except Exception:
          return model_class(**fallback_kwargs)
  ```

- MUST use `model_options={ModelOption.SYSTEM_PROMPT: PREFIX_TEXT}` to establish persona on `m.instruct()` calls (KB7 — `prefix=` is an output prefix, not a system prompt)
- SHOULD use `RepairTemplateStrategy(loop_budget=LOOP_BUDGET)` when requirements include structural `validation_fn`

**Canonical Mellea import paths** — use these exact paths; do not guess or infer alternatives:

```python
from mellea import start_session, generative
from mellea.stdlib.sampling import RepairTemplateStrategy
from mellea.stdlib.requirements import req, check, simple_validate
```

> **Rule 5-2 — Import path grounding**. **Fallback only — primary enforcement is via invariant 3 above.** Before writing any `from mellea.X import Y` statement, verify that the module path `mellea.X` exists in `intermediate/mellea_api_ref.json:.modules`. Any path not present there is invalid and must not be generated.
>
> Common error pattern: generating shortened paths that do not exist (e.g. `mellea.model_options`) when the symbol lives deeper in the hierarchy (e.g. `mellea.backends.model_options`). The `.modules` key is the ground truth — consult it, not training knowledge, for import paths.
>
> The KB entries in `/mellea-fy-behaviours` already show the canonical import for each KB-relevant symbol. For symbols not covered by a KB entry, derive the import path from `mellea_api_ref.json:.modules`.

> **Rule 5-4 — Stdlib function signature grounding**. **Fallback only — primary enforcement is via invariant 3 above.** Before emitting any call to a `mellea.stdlib.*` function, verify the function's argument count and keyword parameter names against the known-signature list below. Do not infer signatures by analogy to similar functions in other libraries (e.g. do not assume `(fn, error_message)` forms that exist in `pytest` or `pydantic` but not in Mellea).
>
> **Known signatures** (static fallback when `mellea_api_ref.json` is absent or `grounding_unavailable: true`):
>
> | Function          | Module                       | Signature                                                                             |
> | ----------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
> | `simple_validate` | `mellea.stdlib.requirements` | `simple_validate(fn)` — **1 positional argument only**                                |
> | `req`             | `mellea.stdlib.requirements` | `req(description, *, validation_fn=None)` — 1 required positional, 1 optional keyword |
> | `check`           | `mellea.stdlib.requirements` | `check(requirement, output)` — 2 positional arguments                                 |
>
> **Common error pattern**: `simple_validate(_check_fn, "error message")` — the two-argument form is invalid. `simple_validate` wraps the validator function; the error message, if needed, is handled inside the validator function itself. The correct call is `simple_validate(_check_fn)`.
>
> For any `mellea.stdlib.*` function not in the table above, derive its signature from `intermediate/mellea_api_ref.json:.modules.<module>.<symbol>.signature` before emitting the call.

**Pipeline structure by tool involvement**:

_P0 — No tools_: pure `pipeline.py` calling `slots.py` and `requirements.py`.

_P4 — Tools provide input_: `main.py` gathers pre-pipeline data, passes as parameters to `run_pipeline()`.

_P2 — Pipeline calls tools (deterministic)_:

1. LLM classifies intent: `intent_thunk = m.instruct(format=IntentSchema, ...)` — result is a `ComputedModelOutputThunk`, NOT a Pydantic object; MUST parse immediately: `intent = _safe_parse_with_fallback(intent_thunk, IntentSchema, query_type="out_of_scope", ...)`
2. Scope check: `if intent.query_type == "out_of_scope": return` (deterministic, no tool call — `intent` here is the parsed Pydantic object, not the thunk)
3. Deterministic construction + tool execution: `TEMPLATES[intent.query_type].format(...)` → `tool_fn(url_or_params)`
4. LLM formats response (optional): `response_thunk = m.instruct(format=ResponseSchema, ...)` → `response = _safe_parse_with_fallback(response_thunk, ResponseSchema, ...)` with raw tool output as grounding
   MUST use two separate `start_session()` calls for steps 1 and 4 (schema priming).

_P3 — Pipeline calls tools (LLM-directed)_:

- Uses `m.react()` with a toolbox, or `m.instruct()` with `ModelOption.TOOLS`
- Mellea's `TOOL_PRE/POST_INVOKE` hooks fire automatically for governance

**pyproject.toml** (always):

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "<package-name>"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mellea[hooks]>=0.4.2",
    "pydantic>=2.0",
]

[project.scripts]
<package-name> = "<package_module>.main:main"

# Rule OUT-6 — declare mirrored companion directories as package data so
# bundled scripts/references/assets ship with the installed wheel. Include
# only the directories that exist after Step 3a-pre's mirror.
[tool.setuptools.package-data]
"<package_module>" = ["scripts/**/*", "references/**/*", "assets/**/*"]
```

Note: the `openai-agents` package is NOT added to dependencies for Agents SDK source specs — the generated package uses Mellea, not the Agents SDK.

---

## Step 5: Code body generation

Step 5 fills every skeleton placeholder with real code. For `config.py`, the model emits JSON and the writer renders Python source (invariant 1).

### Step 5 invariants

Read once; apply throughout all file generation.

**0. Schemas are authoritative — read them BEFORE emitting.** Whenever this slash command (or any sibling step) names a schema file — `descriptor.schema.v0.3.json`, `config_emission.schema.json`, `fixtures_emission.schema.json`, `descriptor_emission` schema, etc. — open and READ that schema in full before drafting any JSON emission against it. The schema is the ground truth, not your prior knowledge of JSON IR conventions. Concretely:

- Every field name in your emission MUST appear in the schema's `properties` for the relevant object (or be allowed by an `additionalProperties` value that isn't `false`).
- Every name listed in `required` MUST be present in your emission for that object.
- `additionalProperties: false` means LITERALLY no extra fields — regardless of how plausible an extra field looks from other JSON IRs you've seen.
- Fields ARE case-sensitive. `callee` is not `symbol`. `fixture_id` is not `id`. `returns` is not a CallNode field at all. The schema decides what's legal; your training memory does not.

**After drafting the emission, re-open the schema and self-check**: walk every top-level field and every repeated sub-shape (each pipeline node, each fixture entry, each schema entry, etc.) against the corresponding schema definition. Verify the field name, presence of all `required` siblings, absence of any field not in `properties`. Invented fields (most-common offenders: `callee` instead of `symbol`, `returns` on CallNode, `fixture_id`/`name`/`expected_output_checks` on fixtures-emission entries) are the dominant cause of descriptor / fixtures schema-gate rejection observed in real compiles 2026-05-19 / 2026-05-20.

If the schema you need isn't reachable, halt and report rather than guess from convention.

**1. `config.py` output is JSON, not Python source.** Emit a JSON object conforming to `.claude/schemas/config_emission.schema.json` (per invariant 0 above — read the schema before emitting). The deterministic writer at `.claude/melleafy/writers/config_writer.py` renders the file — do not write Python source for `config.py` directly.

**2. All other files output Python source.** Generate one file per LLM invocation (Rule 5-3). Wait for each file's body before starting the next.

**3. Before generating any file, consult `intermediate/mellea_api_ref.json`:**

- `.modules` — valid `mellea.*` paths for imports
- `.modules.<module>.<symbol>.signature` — exact signature for any `mellea.stdlib.*` symbol (nested under `.modules`)
- `.forbidden_param_names` — disallowed `@generative` parameter names
- `.compatibility` — Mellea-version-gated workarounds to inject

If `grounding_unavailable: true`, fall back to the KB patterns in `/mellea-fy-behaviours` and the static signature tables in Rules 5-2 and 5-4.

**4. Use canonical fixture pairs from `<package_name>/fixtures/` as concrete examples** (already produced by Step 4) for the file being generated. Use these as the reference for correct Mellea usage, not training memory.

**5. Behavioral guidance is in `/mellea-fy-behaviours`.** Read it once before generating any file body.

**6. Step 7 lint failures, not these instructions, are the source of truth for correctness.** Generate per spec; let the repair loop correct lint failures rather than anticipating every possible check.

> **Rule 5-3 — File-level batching**: Generate all code bodies for a given output file in a single LLM invocation. Do not make one invocation per element. For each file in the skeleton (e.g. `pipeline.py`, `config.py`, `slots.py`, `tools.py`), issue one invocation that generates the complete file contents, guided by all relevant element mapping entries for that file. KB5 schema priming concerns do not apply to melleafy compilation calls (KB5 governs Mellea pipeline sessions inside compiled skills, not the compilation process itself).

Each invocation uses a prompt template with all element-specific mapping entries for that file as variable substitution.

### KB defenses baked into every invocation

Before generating any body, include in the context:

- The specific Known Behaviours relevant to the primitive being generated
- The element's source text and mapping rationale
- The dependency plan entries affecting this element

### Per-file body generation order

Generate bodies in this order (dependency order):

1. `schemas.py` — Pydantic models first (all other files reference them)
2. `config.py` — emit JSON conforming to `.claude/schemas/config_emission.schema.json`; the writer at `.claude/melleafy/writers/config_writer.py` renders the Python source (slots.py references `LOOP_BUDGET`, `PREFIX_TEXT`, etc.)
3. `requirements.py` — requirement functions (pipeline.py references them)
4. `slots.py` — `@generative` slot bodies
5. `tools.py` / `constrained_slots.py` — tool implementations
6. `mobjects.py` — mified object definitions
7. `loader.py` — file loader functions
8. `pipeline.py` — the orchestrating pipeline (references all above)
9. `main.py` — CLI entry point

### Remediation loop bodies (when REMEDIATE elements exist)

```python
# In pipeline.py — bounded remediation loop
MAX_REMEDIATION_ITERATIONS: Final[int] = 3  # in config.py

patched_code = original_code
remediation_count = 0
verdict = initial_verdict

while not verdict.passed and remediation_count < MAX_REMEDIATION_ITERATIONS:
    # Modification step: generate a fix
    with start_session(BACKEND, MODEL_ID) as m_fix:
        fix = m_fix.instruct(
            "Generate a minimal patch to address the identified issue.",
            grounding_context={
                "current_code": patched_code,
                "verdict": str(verdict.model_dump()),
            },
            format=CodeFix,
            strategy=RepairTemplateStrategy(loop_budget=LOOP_BUDGET),
        )
        fix_obj = _parse_instruct_result(fix, CodeFix)

    patched_code = fix_obj.patched_code
    remediation_count += 1

    # Re-evaluation step
    with start_session(BACKEND, MODEL_ID) as m_eval:
        verdict = _parse_instruct_result(
            m_eval.instruct("Re-evaluate...", grounding_context={"code": patched_code}, format=Verdict),
            Verdict
        )
```

### Schema field access rule

After writing `pipeline.py` and `tools.py`, cross-reference every `model.field_name` access against the model's field definitions in `schemas.py`. Accessing a field that doesn't exist raises `AttributeError` at runtime. Correct field access patterns are shown in the fixture examples injected via the grounding context — use those as the reference for how generated code should access schema fields.

### Sequential extraction rule

When 3+ independent slots read the same input, call them sequentially on a single session — do NOT use `asyncio.gather()` with shared sessions (concurrent session sharing is unsafe in Mellea):

```python
with start_session(BACKEND, MODEL_ID) as m:
    # Same BaseModel return type — safe in one session
    primary_findings = extract_primary_findings(m, code_text=code)
    secondary_findings = extract_secondary_findings(m, code_text=code)
    config_issues = extract_config_issues(m, code_text=code)
```

### Two-step pattern in pipeline.py bodies

When Step 2 mapped an element to the two-step pattern:

```python
# Step 1: @generative extracts simplified data (already in slots.py)
raw_paths = extract_raw_attack_paths(m, code_text=code, threat_summary=summary)

# Step 2: m.instruct() structures into full schema with repair strategy
if raw_paths:
    paths_thunk = m.instruct(
        "Enrich these attack paths with risk ratings, impact, and likelihood assessments.",
        model_options={ModelOption.SYSTEM_PROMPT: PREFIX_TEXT},
        grounding_context={
            "raw_attack_paths": str([p.model_dump() for p in raw_paths]),
            "code_text": code,
        },
        format=AttackPathList,
        strategy=RepairTemplateStrategy(loop_budget=LOOP_BUDGET),
    )
    attack_paths = _safe_parse_with_fallback(paths_thunk, AttackPathList, paths=[]).paths
```

### CONVERSE realisation bodies

For realisation (2) — pipeline parameter:

```python
def run_pipeline(
    user_query: str,                    # from CONVERSE element
    reference_context: str = "",        # from TOOL_INPUT or loader
) -> OutputSchema:
    ...
```

For realisation (3) — stub:

```python
def _get_user_approval(draft: str) -> str:
    """Interactive approval step — requires host adapter. See SETUP.md §7."""
    raise NotImplementedError(
        "This step requires a host adapter for interactive user input. "
        "Implement this function or provide the approved draft as a parameter."
    )
```

### `melleafy.json` skeleton (Step 3, finalised in Step 6)

```json
{
  "format_version": "1.0",
  "manifest_version": "1.1.0",
  "package_name": "<package_name>",
  "generated_at": "<ISO timestamp>",
  "melleafy_version": "4.0.0",
  "source_runtime": "<from classification.json>",
  "modality": "<from classification.json>",
  "archetype": "<from classification.json>",
  "categories_resolved": "<populated in Step 6>",
  "entry_signature": "<populated in Step 6>",
  "pipeline_parameters": "<populated in Step 6>",
  "declared_env_vars": []
}
```

---

## Cross-checks before Step 5 declares done

- `intermediate/mellea_api_ref.json` was consulted before code body generation (or `grounding_unavailable: true` was noted and KB fallback used)
- Fixture pair examples from `<package_name>/fixtures/` (Step 4) were used as grounding context for each generated file (invariant 4)

---

## Descriptor mode (`--use-descriptor`)

When `mellea-skills compile` was invoked with the `--use-descriptor` flag (propagated from `mellea-fy.md`'s argument parser to this sub-command), Step 5 takes a different code path. Steps 0–4 (classify, inventory, map, deps, fixtures) and Step 6 (artefacts) run identically to the default path. Step 7 (lints) also runs identically — but its role shifts from "catch LLM Python mistakes" to "catch renderer-emitted Python regressions" per plan §10.5.

### Wrapper-rendered files in descriptor mode

In descriptor mode the wrapper's post-session writer flow renders TWO ADDITIONAL files from `intermediate/descriptor_emission.json` (alongside the existing `config.py` and `fixtures/` it always renders):

| File         | Rendered from                              | Wrapper hook                                                                     | LLM Write/Edit                                                                                              |
|--------------|--------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `config.py`  | `intermediate/config_emission.json`        | `compile/writer_renderer.py::render_writers` (always)                            | DENIED (both modes)                                                                                         |
| `fixtures/`  | `intermediate/fixtures_emission.json`      | `compile/writer_renderer.py::render_writers` (always)                            | DENIED (both modes)                                                                                         |
| `pipeline.py`| `intermediate/descriptor_emission.json`    | `compile/writer_renderer.py::render_descriptor_to_python` (descriptor mode only) | DENIED in descriptor mode (`_DESCRIPTOR_MODE_ADDITIONAL_PATHS`); allowed in legacy mode |
| `schemas.py` | `intermediate/descriptor_emission.json`    | `compile/writer_renderer.py::render_descriptor_to_python` (descriptor mode only) | DENIED in descriptor mode (`_DESCRIPTOR_MODE_ADDITIONAL_PATHS`); allowed in legacy mode |

In legacy mode (no `--use-descriptor`) Claude writes `pipeline.py` and `schemas.py` directly as part of free-form Python emission, and `render_descriptor_to_python` is not invoked. In descriptor mode Claude emits `descriptor_emission.json` only — the wrapper renders the Python from the descriptor IR via `mellea_skills_compiler.renderer.render_descriptor` + `render_schemas`. The wrapper is the source of truth: the `_compile_settings.json` deny list extends to `pipeline.py` and `schemas.py` in descriptor mode, so the LLM's Write/Edit tool calls on those paths are blocked at the tool layer. The post-session `pipeline-entry-canonical` lint hard-fails when `pipeline.py` is absent — so a descriptor that the renderer rejects surfaces as a lint failure, not a silent miss.

**Canonical Step-5 algorithm in descriptor mode** (Phase 3.5.A): the descriptor-emission prompt consumes ALL EIGHT intermediate artefacts produced by Steps 0 through 2.5, alongside the schema doc, filtered surface, and one-shot example. The descriptor must reflect those analytical decisions faithfully rather than re-derive them.

### Role boundary: what Claude does vs. what the wrapper does

This boundary is the single most-misread part of descriptor mode — read it before reading the algorithm steps below.

- **Claude's role (in-session)**: emit `intermediate/descriptor_emission.json` to disk via the `Write` tool. That file must conform to the schema at `src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json` (covering `inputs`, `outputs`, `schemas`, `state`, `pipeline` list of typed nodes, and v0.3 additions like `dependencies` and `bundled_resources`). Claude's job ENDS at writing that JSON file. Claude does NOT invoke any Python function — the only tools available in this slash command are `Read`, `Write`, `Edit`.
- **Claude does NOT write `pipeline.py` or `schemas.py`** in descriptor mode. Those paths are in the wrapper's `_WRAPPER_RENDERED_PATHS` deny-list when `--use-descriptor` is active, so a `Write` against them would be refused; even if it were allowed, the wrapper would overwrite the file post-session.
- **The wrapper's role (post-session, automatic — for situational awareness only, you do not invoke it)**: after the Claude session exits, `mellea_skills_compiler.compile.mellea_skills.compile()` calls `compile/writer_renderer.py::render_descriptor_to_python(package_dir, ...)`. That wrapper hook reads `intermediate/descriptor_emission.json` from disk and dispatches to the deterministic descriptor renderer at `renderer/core.py::render_descriptor` (+ `render_schemas`) to produce `pipeline.py` + `schemas.py`, which it writes to `<package_dir>/`. The wrapper then continues with the rest of the post-session flow (`config.py` + `fixtures/` writers, Step 6 finalisation, Step 7 lints, optional repair retry).
- **Failure surface**: if `descriptor_emission.json` is missing or malformed, `render_descriptor_to_python` cannot produce `pipeline.py`, and the Step 7 `pipeline-entry-canonical` lint hard-fails. When `--repair-on-lint-failure` is set, the wrapper schedules a repair session that gets a fresh chance to emit a correct descriptor.

The in-Python entry points named below (`compile_via_descriptor`, `EmissionConfig`) describe the **wrapper-internal** semantics of the descriptor flow for context only. They are NOT something Claude can or should invoke from inside this slash command — Claude's only deliverable in descriptor mode is the descriptor JSON on disk.

### Algorithm (Claude-side, in-session)

**0. READ the descriptor schema FIRST**, before reading intermediates or drafting anything. The canonical schema is at `src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json` — open it via the `Read` tool and read it in full. The schema is authoritative for field names, `required` lists, and `additionalProperties: false` constraints. Specifically internalise:

- **`CallNode` shape**: the symbol field is named `symbol` (NOT `callee`); there is no `returns` field on CallNode (use `id` + `bound_to` per the schema); the `args` value type is `ArgValue` (a closed `oneOf` over six exact shapes — `value`, `template`, `ref`, `schema_ref`, `env`, `symbol`, plus list — invented shapes like `{"<name>": "#/outputs/<name>"}` will be rejected).
- **`ref` shape** *(examples mirrored from `src/mellea_skills_compiler/rules/registry.json` → `r-sem-ref.examples`)*: value-binding refs (`bound_to`, `over`, `on`, ArgValue's `{ref: ...}` variant) take a PLAIN identifier string that resolves against a declared `state[].id` / `inputs[].name` / prior-node id — `{"ref": "session"}` (✅). Two common rejections: `{"ref": "#/state/session"}` (❌, JSON Pointer syntax — rejected by the schema's `Ref.ref` pattern `^[a-z_][a-z0-9_]*$`) and `{"ref": "completely_undefined_target"}` (❌, valid pattern but the identifier isn't in scope — rejected by the semantic-rule layer). JSON Pointer is reserved for the **distinct** `SchemaRef` shape (`{"ref": "#/schemas/Foo"}`) and the output-binding form (`#/outputs/<name>`); the two ref kinds are separate `$defs` entries and must not be conflated.
- **`ref` scope**: a `ref` resolves only against the descriptor's own `state[].id` / `inputs[].name` / prior-node `id` — NOT config constants (e.g. `LOOP_BUDGET` from config.py), NOT names from sibling sub-compositions invisible to your scope.
- **Composition scoping (`_walk_pipeline` rule)**: ids declared INSIDE a composition body (`sequential`, `branch`, `map`, etc.) are visible only WITHIN that composition's scope and to its descendants — **NOT to sibling compositions at the same nesting level**. If your skill has phases that share data across boundaries (e.g. intake → draft → verify, where `notice_type` is collected in intake and referenced in draft and verify), you have three ways to express the dataflow legally:
  - **Flatten the pipeline**: put the call nodes directly at the top level of `pipeline[]` rather than wrapping them in phase compositions. Every prior node's `id` is then visible to every later node via the `visible_ids` set. Loses the visual phase grouping in the descriptor structure but preserves the dataflow correctly. Simplest fix.
  - **Promote shared values to `state[]`**: if a value persists across the whole pipeline (a session handle, a configuration object), declare it as a top-level `state[].id` with the symbol that produces it. State entries are visible everywhere.
  - **Thread through top-level node ids**: a composition's own `id` is visible to its siblings, but its internal body's ids are not. To share a value collected mid-pipeline, declare the call node that produces it at the TOP level of `pipeline[]` (with an `id` like `notice_type_extract`), then ref that id from later phases. Wrapping the extraction inside a `phase_2_intake` composition would hide the id from `phase_3_draft`.

  Symptom of getting this wrong: `R-SEM-REF: ref 'X' does not resolve to a declared state.id, input name, or prior node id` firing repeatedly across nested-composition paths like `/pipeline/0/body/2/body/0/...`. If you see that pattern, the fix is one of the three above — usually "flatten the pipeline" is the cheapest.
- **`bound_to`**: present only when explicitly binding a method receiver. The renderer ignores it for variable assignment — assignment uses the node's own `id`. Putting `bound_to: {ref: "<name>"}` on every call is an emission anti-pattern and the validator will reject the `<name>` as out-of-scope.
- **Required fields per node kind**: each `kind` (call / composition / etc.) has its own `required` list. Read each variant.

If the schema isn't reachable via Read, halt and report rather than guessing from convention or prior JSON IR exposure.

1. Read every intermediate artefact present under `<package_name>/intermediate/`. The canonical 8-artefact set is:
   - `classification.json` — 5-axis archetype (Step 0)
   - `inventory.json` — element tags, C1–C9 categories, source-line refs (Step 1b)
   - `element_mapping.json` — tag → Mellea symbol mapping (Step 2)
   - `element_mapping_amendments.json` — Step 2.5d overrides (often supersedes the initial mapping)
   - `dependency_plan.json` — 8-disposition dependency plan (Step 2.5c)
   - `mellea_api_ref.json` — introspected Mellea surface (Step 2.5e). **Verification surface, NOT a primary read.** Treat it as a dictionary you consult on demand for specific symbol signatures and module paths — do NOT read end-to-end (it is ~280KB). Use the canonical example in Step 1b as your composition reference; consult this surface only when you need to look up a specific symbol's signature or verify a module path the canonical doesn't already show.
   - `mellea_doc_index.json` — per-symbol doc-page references (Step 2.5f)
   - `expected_signature.json` (P3.5.D — Step 2 always emits this artefact; the system prompt inlines it as a HARD CONSTRAINT block. The `R-SEM-SIGNATURE-MATCH` validator rule fires on any divergence between the descriptor's `inputs`/`outputs`/`schemas` and the locked signature. Absent only on legacy / pre-P3.5.D intermediate artefact sets — in that case the rule is non-firing) — locked I/O signature constraint
**1b. Select and read the canonical-descriptor example for your classification triple.** Using the `classification.json` you just read, locate a worked example matching the skill's archetype × shape × modality:

   - **Filename rule**: `<archetype>_<shape>_<modality>.json`, with the modality value's underscores converted to kebab-case (`synchronous_oneshot` → `synchronous-oneshot`). Example: `{archetype: "A", shape: "Sequential", modality: "synchronous_oneshot"}` → `A_Sequential_synchronous-oneshot.json`.
   - **Read the file** from `src/mellea_skills_compiler/canonical_descriptors/<filename>` via the `Read` tool. The file is a wrapper carrying `metadata.classification`, `metadata.notes` (read these — they describe what the canonical demonstrates and what it deliberately omits), and `descriptor` (the worked example).
   - **Degradation chain if missing**: drop modality first (try `<archetype>_<shape>.json`), then drop shape. If nothing resolves, no canonical exists for your triple — proceed without one and surface that gap in a note.
   - **Use the canonical as a STRUCTURAL reference, not a content source.** Pattern-match on HOW it expresses `state[]`, `inputs`, the pipeline-composition shape, how `dependencies[]` flow into the pipeline, how `schemas` are declared and referenced. Do NOT copy the canonical's skill-specific symbol names, descriptions, or args — those belong to the source skill, not yours. The canonical teaches *idiom*, not *content*.

> **Separation of concerns: composition reference vs verification surface.** Two distinct grounding artifacts serve two distinct cognitive roles. Use each for its intended purpose:
>
> - **The canonical descriptor** (read in Step 1b) is your *composition reference*. It shows how a descriptor for skills of this classification is wired together — where state goes, how the pipeline composes, how dependencies flow into inputs. Pattern-match against it for structure.
> - **`intermediate/mellea_api_ref.json`** is your *verification surface*. It's the dictionary of what's callable in this Mellea version, with signatures. Look up specific symbols on demand. **Do NOT read it end-to-end** — the canonical example already shows you which Mellea symbols are typical for skills of this triple. The descriptor symbol gate provides automated post-emission verification, so emit confidently against the canonical's idiom and consult the surface only for targeted lookups.
>
> In short: canonical teaches composition; surface verifies vocabulary; symbol gate enforces correctness.

2. **Consult `intermediate/mellea_api_ref.json` for targeted symbol lookups** — when the canonical example doesn't already show the symbol you want to use, or when you need a precise signature for a stdlib helper. Use the `Read` tool with line offsets to fetch only the relevant module's entries; do NOT read the file end-to-end. The descriptor symbol gate will post-emission-verify everything; you do not need to scan exhaustively to avoid being wrong.
3. Synthesise the descriptor IR. Each of the 8 artefacts feeds specific descriptor fields:
   - `classification.json` → `metadata.modality`, `metadata.archetype`
   - `inventory.json` + `element_mapping.json` + `element_mapping_amendments.json` → `pipeline` nodes (typed by mapped Mellea symbol), `schemas` entries
   - `dependency_plan.json` → `dependencies`, `bundled_resources` (v0.3); also disposition-tags on pipeline nodes
   - `mellea_api_ref.json` → import paths and signatures used by node `call` fields
   - `expected_signature.json` → `inputs`, `outputs`, and top-level `schemas` (HARD CONSTRAINT — divergence fails `R-SEM-SIGNATURE-MATCH` post-session)
4. Write the assembled JSON to `intermediate/descriptor_emission.json` using the `Write` tool. The file MUST be valid JSON matching `descriptor.schema.v0.3.json`. This is your terminal action for Step 5 in descriptor mode — do not attempt to write `pipeline.py` or `schemas.py`.

### Descriptor accept-set (type expressions, signatures, symbols)

The deterministic descriptor renderer at `mellea_skills_compiler.renderer.render_descriptor` accepts a narrow set of forms. Three rules cover the common rejection cases:

- **Type expressions** must use PEP 585 lowercase forms (`list[str]`, `dict[str, int]`, `str | None`). Do not import from `typing`; capitalised aliases (`List`, `Dict`, `Optional`, `Union`) are rejected.
- **Dependency signatures** are parenthesised only (`(query: str) -> str`) — no leading function name. The `id` field carries the name.
- **Symbol references** in `state[]`, pipeline node `call` fields, and strategy args MUST use the defining-module path as recorded in `intermediate/mellea_api_ref.json`. The Python user-facing re-export form is NOT accepted. See ACCEPT-SET-2 in `mellea-fy-behaviours.md` for the lookup algorithm and rejection patterns; both bare names and `from mellea import X`-style forms are rejected.

Violations produce a `RendererError` from `compile/writer_renderer.py::render_descriptor_to_python`, surfaced both as a `[writer:descriptor] render failed` log line and as a `pipeline-entry-canonical` lint failure (because `pipeline.py` is absent).

**See `mellea-fy-behaviours.md` § ACCEPT-SET-2 for the full rule tables (❌/✅ pairs), where each rule applies inside the descriptor schema, and what the wrapper does on failure.**

### Symbol gate: pre-render normalisation

After the descriptor JSON is written and before the deterministic renderer runs, the wrapper invokes a pre-render gate at `src/mellea_skills_compiler/compile/descriptor_symbol_gate.py::run_symbol_gate`. The gate walks every `symbol` field anywhere in the descriptor IR (`state[].symbol`, pipeline `call` symbols, dependency `symbol`s) and resolves Mellea-API references against the introspected surface (`intermediate/mellea_api_ref.json`) via a deterministic 5-stage matcher:

1. **Exact** — the symbol is already a valid surface path (longest module prefix splits to a known head symbol). Pass through unchanged with `method="exact"`.
2. **Suffix unification** — the symbol is a unique dotted suffix of exactly one canonical surface path. E.g. `MelleaSession.instruct` → `mellea.stdlib.session.MelleaSession.instruct`. Silently auto-normalised with `method="suffix"`.
3. **Module-scoped leaf match** — the symbol's leading segments form a real module key AND exactly one symbol within that module has the trailing segment as its leaf. Handles the "dropped class prefix" pattern: `mellea.stdlib.session.chat` → `mellea.stdlib.session.MelleaSession.chat`. Silently auto-normalised with `method="module-scoped-leaf"`.
4. **Bare-name resolution** — a single-segment symbol whose leaf is unique across the surface (after re-export collapse). E.g. bare `MelleaSession` → `mellea.stdlib.session.MelleaSession`. Silently auto-normalised with `method="bare-name"`.
5. **Fuzzy fallback** (optional, requires `rapidfuzz`) — token-set ratio. Requires best score `>= 92` AND best-vs-second dominance `>= 5`. If both hold, silently auto-normalised with `method="fuzzy"`; else fall through to failure.

**Re-export aliases collapse to canonical**: when multiple surface paths point to the same Python symbol (identifiable via shared `defined_in`), the gate treats them as one canonical target — they are NOT a collision. Example: bare `start_session` resolves to `mellea.stdlib.session.start_session` even though the surface also lists it at `mellea.start_session` as a top-level re-export.

**Non-Mellea symbols pass through unflagged**: symbols whose leading segment isn't a Mellea module head — `loader.SkillLoader`, `builtins.dict`, fully-qualified `<package>_mellea.X` — are skipped without validation. The renderer or downstream lints handle those.

**Discipline: fail loud on ambiguity, fix silently on certainty.** Every silent rewrite is appended to `intermediate/symbol_normalisations.jsonl` as one JSON object per line — telemetry, not state — so the LLM emission accuracy stays observable as first-class signal rather than hidden behind auto-fixes.

**What you should emit**: prefer the canonical surface path on the first try (Step 1 is the cheapest pass-through). When the canonical form is awkward, partial qualifications that uniquely identify the target (Stages 2-4) are auto-normalised. When the gate rejects with `[writer:descriptor] symbol gate rejected N symbol(s); first: '<symbol>' at <path>; closest candidates: <top-3>`, the failure message lists the top-3 candidate canonical paths — pick from those rather than guessing a different shape. If `closest candidates: (no candidates)` appears, the symbol isn't in the surface at all; check `intermediate/mellea_api_ref.json` for the right name.

### What the wrapper does post-session (for situational awareness)

After the Claude session exits, the wrapper automatically (you do not invoke any of this):

a. Reads `intermediate/descriptor_emission.json` from disk via `compile/writer_renderer.py::render_descriptor_to_python(package_dir, ...)`.
b. Validates the descriptor against `descriptor.schema.v0.3.json` via `descriptor/validator.py::validate()` (jsonschema + `R-SEM-*` semantic rules). When `expected_signature.json` is present, `R-SEM-SIGNATURE-MATCH` enforces the locked I/O signature.
c. Dispatches to the deterministic Phase 2 renderer at `renderer/core.py::render_descriptor` (plus `render_schemas`) to produce `pipeline.py` + `schemas.py`, written to `<package_dir>/`. (Note: when descriptor emission is invoked via the standalone `compile/descriptor_emission.py::compile_via_descriptor` entry — a separate Python-harness path, not used by this slash command — additional artefacts like `__init__.py`, `fixtures.py`, `melleafy.json`, `SETUP.md`, `README.md`, `source_map.json` are also rendered. The mainline `compile()` + `render_descriptor_to_python` path renders only `pipeline.py` + `schemas.py`; the other artefacts come from their own writers / Step 6.)
d. Continues with `config.py` + `fixtures/` writers (always-on), Step 6 artefact finalisation, Step 7 lints, and the optional `--repair-on-lint-failure` retry loop.
e. If the descriptor was missing or malformed, `pipeline.py` is not produced and the Step 7 `pipeline-entry-canonical` lint hard-fails — surfacing the problem as a lint failure rather than a silent miss.

Step 6 (`mellea-fy-artifacts`) and Step 7 (`mellea-fy-validate`) run as usual against the wrapper-rendered files.

**Why the 8-artefact prompt matters**: each artefact removes a decision the LLM would otherwise re-derive (and sometimes get wrong). `dependency_plan.json` carries the 8 disposition kinds; the LLM does NOT pick dispositions — it transcribes them. `element_mapping.json` + `element_mapping_amendments.json` carry the tag → symbol decisions. `classification.json` constrains the modality + interaction style. Skipping any of these means asking the LLM to learn what the analytical pipeline has already decided.

**Flag mapping from `mellea-skills compile` to descriptor-mode** (these flags are interpreted by the wrapper, not by Claude — listed here for context):

| CLI flag | Descriptor-mode behaviour |
|---|---|
| `--use-descriptor` | Required to enter this path. Causes the wrapper to invoke `render_descriptor_to_python` post-session and adds `pipeline.py` / `schemas.py` to the slash-command deny-list. |
| `--model` / `-m` | Overrides the model used by the wrapper to spawn the Claude session (informational from a slash-command POV — affects which model is generating the descriptor JSON). |
| `--timeout` | Not directly applicable — streaming + 64K tokens has its own per-stream behaviour. |
| `--repair-mode` / `-r` | Routes the wrapper through `compile/repair.py::compile_with_repair` (bounded retry, possible legacy escalation) instead of the single-shot descriptor path. Informational reference: the underlying wrapper-internal entry point name is `compile_via_descriptor`; Claude does not invoke either function. |
| `--no-run` | Skips the post-session smoke check. |
| `--refresh-cache` | Forces a P1.C cache refresh before emission. |
| `--skill-backend` / `--skill-model` | Apply to the rendered package's runtime defaults (no compile-time effect). |

When `--use-descriptor` is NOT set, this step follows the existing free-form Python emission flow below (Claude writes `pipeline.py` and `schemas.py` directly).

---

## Step 5 repair mode

Invoked by the top-level repair loop (see `mellea-fy.md`) after a Step 7 Tier 1 or structural Tier 2 failure. Distinct from a normal Step 5 invocation in three ways:

**Scope**: read `intermediate/step_7_report.json`. Generate only the files listed under failing lint entries. Pass all files with no failures through unchanged.

**Additional context per failing file**: inject the exact lint failure entries as a structured block before the generation context:

```
LINT FAILURES IN <filename> (repair round <N>):
  [<lint_id>] line <L> col <C>: <message>
```

**Cap**: repair mode may be invoked at most twice (`repair_round ∈ {1, 2}`). If Step 7 still fails after round 2, do not generate further — return control to the top-level orchestrator, which halts and preserves `.melleafy-partial/`.
