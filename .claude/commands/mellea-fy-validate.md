# Melleafy Step 7: Static Validation

**Version**: 4.3.1 (2026-04-28) | **Prereq**: Steps 3–6 complete | **Produces**: `intermediate/step_7_report.json`

Step 7 is the workflow's final gate. Twenty-five lints in three tiers. No LLM invocations. No mutation of the generated package. Outcome is binary overall — pass or halt.

Run as: `melleafy lint <package_path>` (standalone), or automatically at the end of `melleafy run`.

---

## Three-tier architecture

### Tier 1 — Parseability (halt immediately on failure)

**`parseable`**: every `.py` file in the generated package passes `ast.parse()` without error. Then, the package entry module (`<package_name>.pipeline`) must import cleanly via `importlib.import_module()` in a subprocess. This catches wrong external import paths (e.g. `mellea.stdlib.strategies` vs the real `mellea.stdlib.sampling`) that `ast.parse()` cannot detect.

Implementation: run `python -c "import <package_name>.pipeline"` as a subprocess from the package's parent directory. A `ModuleNotFoundError` or `ImportError` is a lint failure, not a missing-dependency advisory.

If this lint fails, Step 7 halts. Tier 2 and Tier 3 don't run. The failure report contains only syntax errors and import errors (nothing else is meaningful before parsing).

### Tier 2 — Structural lints (collect all, halt before Tier 3)

Run in parallel. Each is independent. All tier-2 lints run to completion even if one fails; results are collected then the tier verdict is determined.

**`cross-reference`**: every `element_mapping.json` target symbol exists in the generated package; every external call in `pipeline.py` / `tools.py` has a corresponding `dependency_plan.json` entry. Sub-checks:

- Sub-check A: every `target_symbol` in `element_mapping.json` resolves to a real function/class in the target file
- Sub-check B (intra-package): every relative import and every `from .<module>` import in the generated files resolves to a file within the package. Note: external library imports (e.g. `from mellea.stdlib.sampling import ...`) are validated by the `parseable` importable check, not here.
- Sub-check C: every C6 tool called in `pipeline.py` appears in `tools.py` or `constrained_slots.py`
- Sub-check D: no dead `@generative` slots (defined but never called)
- Sub-check E: no dead requirements lists (defined in `requirements.py` but never attached)
- Sub-check F _(Rule 3-1)_: every parameter in `pipeline.py:run_pipeline` has an explicit Python type annotation. Detection: parse `pipeline.py` with `ast`; for each `arg` in `run_pipeline`'s `arguments`, assert `arg.annotation is not None`. Hard failure — bare parameter names produce untyped CLI interfaces and break downstream validation.

**`validator-soundness`**: scoped to `requirements.py` only. Two sub-checks:

- Sub-check A (KB3): every `validation_fn=` uses `simple_validate()` or a function with signature `(ctx, result) -> ...`
- Sub-check B (KB4): no vacuous lambda body (lambda that always returns `True` regardless of input)

**`session-boundary`** (KB5): each `start_session()` block uses at most one distinct `BaseModel` format type across all `m.instruct(format=...)` calls within it. Note: `@generative` slots each create their own internal `<FunctionName>Response` model; multiple `@generative` slots with different return types in the same session are subject to the same schema-priming risk as `m.instruct(format=...)` with multiple models.

**`variable-safety`**: two sub-checks:

- Sub-check A: no uninitialised names in `except` / `finally` blocks. Detection: any name referenced in an `except` or `finally` block that has no assignment before the enclosing `try` statement is a failure. The correct pattern is to initialise the variable before the `try` block (e.g. `payload = None` before `try: payload = build_payload(...)`).
- Sub-check B: no shadowing of Python builtins in function argument names

**`import-side-effects`** (R19 property 4): no module-level calls at import time outside the allowlist (`logging.getLogger()`, `Final` assignment, `os.environ.get()` for config). No `load_dotenv()` at module level. No network calls at import.

**`import-soundness`**: for every `from mellea.X import Y` or `import mellea.X` statement in the generated package, verify that `mellea.X` appears as a key in `intermediate/mellea_api_ref.json:.modules`. Any import whose module path does not appear there is a hard failure. Detection: parse all `.py` files in `<package_name>/` with `ast`; collect all `ImportFrom` nodes where `module` starts with `"mellea"`; load `mellea_api_ref.json` and check each path against `.modules` keys. If `grounding_unavailable: true`, this lint emits a warning ("module index unavailable — import-soundness check skipped") rather than failing. Common error: shortened paths (e.g. `mellea.model_options`) when the symbol lives deeper in the hierarchy (e.g. `mellea.backends.model_options`). Scope: `mellea.*` imports only — third-party imports (`pydantic`, `anthropic`, etc.) are out of scope.

**`stdlib-arity`**: for each call to a known `mellea.stdlib.*` function in the generated package, verify the argument count and keyword parameter names match the declared signature. A call with the wrong argument count or an unrecognised keyword argument is a hard failure. Detection: parse all `.py` files in `<package_name>/` with `ast`; collect `Call` nodes whose `func` is a `Name` or `Attribute` matching a known stdlib function; check positional arg count and keyword names.

**Signature source**:

The static table below is the primary enforcement mechanism. For functions in this table, the static definition always applies — `mellea_api_ref.json` is not consulted, since these signatures are stable across versions.

| Function          | Required positional         | Optional keyword |
| ----------------- | --------------------------- | ---------------- |
| `simple_validate` | 1 (`fn`)                    | none             |
| `req`             | 1 (`description`)           | `validation_fn`  |
| `check`           | 2 (`requirement`, `output`) | none             |

For `mellea.stdlib.*` calls to functions **not** in the table above: if `intermediate/mellea_api_ref.json` is present and `grounding_unavailable: false`, look up the signature at `.modules["<module>"]["<symbol>"]["signature"]` and apply the same positional/keyword check. If `mellea_api_ref.json` is absent or `grounding_unavailable: true`, emit a warning ("unknown stdlib function — verify signature manually") rather than a hard failure.

**`grounding-context-types`**: every `grounding_context=` dict literal in the generated package has only `str` values. Detection: parse all `.py` files with `ast`; find `Call` nodes where a keyword `grounding_context` has a `Dict` value; for each dict value, assert it is a `Constant` (string literal), a `Call` to `str()`, or a `JoinedStr` (f-string). Any value that is a bare `Name`, `Attribute`, or other expression is a **warning** (not hard failure). Correct pattern: `grounding_context={"key": str(some_object)}`. Scope: generated `.py` files only.

**`format-annotation`**: every `m.instruct(...)` call whose result is passed to `model_validate_json()` or assigned to a variable then used in a Pydantic parse must have a `format=` keyword argument. Detection: parse `pipeline.py` with `ast`; find `Call` nodes that are `m.instruct`; trace the result name; if it appears as the argument to `.model_validate_json(...)` and has no `format=` keyword, hard failure. This catches calls that produce untyped JSON strings when structured output was intended.

**`instruct-has-description`** (KB12, Mellea 0.5+ invariant): every `m.instruct(...)` call in `pipeline.py`, `slots.py`, and `constrained_slots.py` must supply a `description` argument — either as the first positional argument or as a `description=` keyword. Mellea 0.5's `MelleaSession.instruct(description, *, ...)` makes `description` positional-required; calls without it crash at runtime with `TypeError: instruct() missing 1 required positional argument: 'description'`. Detection: parse the three files with `ast`; for every `Call` whose func is `Attribute(value=Name('m'), attr='instruct')`, assert `args` is non-empty OR `keywords` contains `description=`. Hard failure naming filename + line of the offending call. Scope: generated `.py` files only; only `m.instruct` is in scope (other `*.instruct` callsites are ignored).

**`pipeline-entry-canonical`** (canonical entry-point contract): `<package_name>/pipeline.py` MUST define a top-level `run_pipeline` function, and `<package_name>/melleafy.json:entry_signature` (when present) MUST start with `run_pipeline(`. Empirically observed regression: a package with `run_phase_2_gap_analysis`, `run_phase_3_roadmap`, and `run_pipeline` as public top-level functions caused the smoke-check loader to pick `run_phase_2_gap_analysis` (alphabetically first under `dir()`) and crash with `TypeError` when invoked with entry-point kwargs. The loader at `toolkit/file_utils.py:load_skill_pipeline` was patched to use `melleafy.json:entry_signature` as authoritative; this lint backstops the contract so misalignment is caught at compile time. Detection: AST-parse `pipeline.py`; collect names of all top-level `FunctionDef`/`AsyncFunctionDef` starting with `run_`; (a) hard-fail if `run_pipeline` is not among them, naming the helpers that were found instead; (b) parse `melleafy.json:entry_signature` with regex `^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(`; if the captured name is not `run_pipeline`, hard-fail. Public helper functions named `run_*` are PERMITTED alongside `run_pipeline` — the loader uses the manifest, so naming collisions don't matter. **Hard-fails when `pipeline.py` is absent** — descriptor-mode renderer rejection or any other pre-Step-7 omission of `pipeline.py` surfaces as a lint failure rather than a silent miss. Previously skipped in that case, which masked three real overnight-batch failures.

**`fixture-signature-bound`** (R16 mechanical enforcement): every `make_<id>()` factory in `<package_name>/fixtures/*.py` whose body contains an `inputs = {…}` literal MUST have only string-literal keys, and each key MUST be a parameter name of `run_pipeline` in `pipeline.py` (unless the signature includes `**kwargs`). The smoke-check invokes `run_pipeline(**inputs)`; any extra key raises `TypeError: got an unexpected keyword argument '<key>'` at fixture-run time. The prose contract already lives in `mellea-fy-fixtures.md` ("The keys in every fixture's `inputs` object MUST be identical to the parameter names of the `run_pipeline` function") — this lint enforces it mechanically. Detection: AST-parse `pipeline.py`, locate `def run_pipeline(...)`, collect parameter names (positional, kw-only, and the presence of `**kwargs`); for each `fixtures/*.py` file (excluding `__init__.py`), AST-parse and find every `FunctionDef` whose name starts with `make_`; walk its body for an `Assign` to `inputs` with a `Dict` value; report any key not in the entry parameter set. Skip-fixture (silent, not a failure) when `inputs` is constructed dynamically (non-`Dict` RHS) or uses non-string-literal keys — the lint cannot statically verify these and defers to runtime. Skipped (verdict=`skipped`) when `pipeline.py` is absent, when `run_pipeline` is absent from `pipeline.py` (owned by `pipeline-entry-canonical`), when the signature includes `**kwargs`, or when `fixtures/` is absent (owned by `fixtures-loader-contract`).

**`generative-forbidden-params`** (KB6, definition-side): every function decorated with `@generative` in `slots.py`, `constrained_slots.py`, or `pipeline.py` MUST NOT declare any of Mellea's reserved parameter names. The reserved set is sourced from `intermediate/mellea_api_ref.json:forbidden_param_names` (live introspection of `mellea.stdlib.components.genstub._disallowed_param_names`), with a static fallback (`m`, `context`, `backend`, `model_options`, `strategy`, `precondition_requirements`, `requirements`, `f_args`, `f_kwargs`) when grounding is unavailable. At module import the `@generative` decorator raises `ValueError: cannot create a generative stub with disallowed parameter names: [...]` when any reserved name appears, so this lint catches the violation at compile time instead of fixture-import time. The correct pattern is `@generative def slot(text: str) -> str: ...` paired with a call-site `with start_session(...) as m: slot(m, text=...)`. Detection: AST-parse the three files; find every `FunctionDef` / `AsyncFunctionDef` whose decorator list contains `@generative` (Name or Attribute); check every positional and keyword-only parameter name against the forbidden set. Hard failure naming file:line, function name, and all offending parameter names.

**`generative-call-passes-session`** (KB6, call-site half): every call to a `@generative`-decorated function (collected from `slots.py` + `constrained_slots.py`) MUST pass either (a) at least one positional argument (the session `m`), (b) a `m=...` keyword argument, OR (c) both `context=...` and `backend=...` keyword arguments. Mellea's `GenerativeStub.__call__` raises `TypeError: generative stub requires either a MelleaSession (m=...) or both a Context and Backend (context=..., backend=...) to be provided as the first argument(s)` when none of those is satisfied. The most common cause is `with start_session(BACKEND, MODEL_ID):` (no `as m`) followed by `slot(text=...)` (no positional m) — the fix is to bind the session with `as m` and pass it positionally at the call. Detection: collect every `@generative` function name from `slots.py` + `constrained_slots.py`; AST-walk the three pipeline files for `Call` nodes whose `func` resolves to one of those names; assert at least one of the three conditions above holds. Hard failure naming file:line + the offending call.

**`known-behaviours`**: mechanical checks for KB1, KB2, KB3, KB4, KB6, KB7, KB11:

- KB1 (3a): no `m.instruct(format=...)` result accessed as a Pydantic object without a prior parse call. Detection: parse `pipeline.py` with `ast`; identify variables assigned from `m.instruct(...)` calls with a `format=` keyword; flag any attribute access (`.field_name`) or method call (`.model_dump()`, `.model_dump_json()`, `.parsed_repr`) on those variables that does not appear as the argument to `_parse_instruct_result(`, `_safe_parse_with_fallback(`, or `.model_validate_json(`. Hard failure.
- KB2 (3b): complex schemas (BaseModel with >4 fields or any field annotated as `list[...]`) used in `m.instruct(format=...)` must either use `RepairTemplateStrategy` in that call or have the result parsed with `_safe_parse_with_fallback`. Detection: parse `pipeline.py` and `schemas.py` with `ast`; for each `m.instruct(format=Model)` call, look up the model's field count and list annotations in `schemas.py`; if the model qualifies as complex, assert the call has `strategy=` keyword or the result variable is passed to `_safe_parse_with_fallback`. Hard failure.
- KB3 (3c): validator signatures (also checked by `validator-soundness`)
- KB4 (3d): no vacuous validators (also checked by `validator-soundness`)
- KB6 (3f): no `@generative` function parameter uses a name from `intermediate/mellea_api_ref.json:.forbidden_param_names`. Detection: load `forbidden_param_names` from `mellea_api_ref.json`; parse `slots.py` and `constrained_slots.py` with `ast`; for each function decorated with `@generative`, assert no parameter name appears in that list. If `grounding_unavailable: true`, fall back to the static list: `m`, `context`, `backend`, `model_options`, `strategy`, `precondition_requirements`, `requirements`, `f_args`, `f_kwargs`. Hard failure.
- KB7 (3g): no `prefix=<config_constant>` used as a persona/system-prompt mechanism — use `model_options={ModelOption.SYSTEM_PROMPT: <constant>}` instead. Detection: `prefix=` argument whose value is a name from `config.py` (e.g. `PREFIX_TEXT`). Note: `prefix=` for structured output generation (e.g. `prefix='{"result":"`)) is permitted. Import-path validation for `ModelOption` is now handled by the `import-soundness` lint.
- KB11 (3m): every `Optional` field in a P2 `m.instruct` `BaseModel` that corresponds to a user-supplied tool parameter has extraction guidance in its `Field(description=...)`. Detection: parse `schemas.py` with `ast`; find `BaseModel` subclasses whose names end with `Schema` or `Intent` or are referenced as `format=` in a `m.instruct` call; for each `Optional`-annotated field, assert `Field(description=...)` is present and the description string contains at least one of: "extract", "do not ask", "if the" (case-insensitive). Hard failure.

**`doc-citation`**: every `**Verified:**` or `**Ref:**` annotation in `mellea-fy-behaviours.md` that references a `docs.mellea.ai` path must appear in `intermediate/mellea_doc_index.json:.doc_pages`. Detection: read `mellea-fy-behaviours.md`; find all occurrences of `**Verified:**` and `**Ref:**` followed by a URL containing `docs.mellea.ai`; extract the path component; check each path against `doc_pages`. If `doc_pages` is empty (fetch failed at Step 2.5f), emit warning ("doc index unavailable — citation check skipped") rather than failing. Hard failure if `doc_pages` is populated and a cited path is absent.

**`bundled-asset-path-resolution`** (Rule OUT-6, Rule 2.5-2): every reference in the generated package to a path under `scripts/`, `references/`, or `assets/` must be resolved package-relatively via `Path(__file__).parent / ...`. Any code that joins a function-argument path (typically `repo_root`) — or any expression other than `Path(__file__).parent` — with one of those subdirectory names is a hard failure. Detection: parse all `.py` files in `<package_name>/` with `ast`; find `BinOp(left=…, op=Div)` chains and `Call(func=Path)` expressions whose right-hand side begins with a string literal `"scripts/…"`, `"references/…"`, or `"assets/…"` (or the bare component `"scripts"`, `"references"`, `"assets"` followed by another `/` join); for each, resolve the leftmost expression of the join. If it is anything other than `Call(func=Attribute(value=Name("__file__"))…)` rooted at `Path(__file__).parent`, fail with the precise message: _"Bundled asset path '<…>' is resolved via '<expr>'. Bundled assets at `<package_name>/<dir>/` MUST be resolved via `Path(__file__).parent / "<dir>/<file>"` (Rule OUT-6 in `mellea-fy.md`, Rule 2.5-2 in `mellea-fy-deps.md`). Common error: `Path(repo_root) / 'scripts/...'` — must be `Path(__file__).parent / 'scripts' / ...`."_ Scope: generated `.py` files only; the path components are matched against the literal directory names declared in Rule OUT-6.

**`pyproject-package-data-bound`** (Rule OUT-6, second half): every companion directory physically mirrored into `<package_name>/` (i.e. `scripts/`, `references/`, or `assets/` that exists on disk after Step 3) MUST appear in `[tool.setuptools.package-data]` in `pyproject.toml` so a `pip install` includes its contents in the built wheel. Detection: parse `pyproject.toml` with stdlib `tomllib`; resolve the file at `<package_dir>/pyproject.toml` first, falling back to `<package_dir>.parent/pyproject.toml` (skill-root location); for every companion directory present at `<package_dir>/<dir>/`, look up `[tool.setuptools.package-data]` under the exact `<package_name>` key OR the wildcard `"*"` key, and require at least one declared glob covering the directory (acceptable patterns: `<dir>/**/*`, `<dir>/*`, explicit filenames anchored at `<dir>/`). Hard failure on missing coverage with a message naming the offending companion dir; skipped (`verdict=skipped`) when no `pyproject.toml` exists at either location.

**`fixtures-loader-contract`** (R16, Rule 4-1): `<package_name>/fixtures/__init__.py` must export a module-level `ALL_FIXTURES` (or `FIXTURES`) list. Detection: AST-parse `fixtures/__init__.py`; require at least one module-level `Assign` (or `AnnAssign`) whose target name is `ALL_FIXTURES` or `FIXTURES`. Hard failure with a message naming both expected attribute names and pointing at `mellea-fy-fixtures.md` for the contract. This is a defence in depth — under the `fixtures_writer.py` architecture (Step 4), violations should be unreachable; the lint exists to catch hand-edited `fixtures/` directories or any future code path that bypasses the writer.

**`runtime-defaults-bound`** (C8 invariant): `<package_name>/config.py` `BACKEND` and `MODEL_ID` values must match the directive recorded at `<package_name>/intermediate/runtime_directive.json`. The compile pipeline writes the directive pre-mellea-fy from `.claude/data/runtime_defaults.json` (or CLI overrides) and injects the same values into the LLM's system prompt; this lint verifies the LLM honoured the instruction. Detection: AST-parse `config.py`; find module-level `Assign` or `AnnAssign` to `BACKEND` and `MODEL_ID` with `Constant` values; compare against the directive. Hard failure on mismatch with a message naming actual vs expected and pointing at `.claude/data/runtime_defaults.json` for the fix. Skipped when the directive file is absent (e.g. package compiled with an older pipeline that did not write the directive).

### Tier 3 — Cross-artifact lints (run only when Tier 2 passes)

**`category-specific`**: conditional per C-category detected in `dependency_plan.json`:

- C1-A: scan `config.py:PREFIX_TEXT` for high-entropy strings (>4.5 bits/char, >20 chars) — likely secrets leaked into persona text
- C1-B: scan `config.py` constants for high-entropy strings
- C6: every MCP tool name in `tools.py` uses the qualified `mcp__server__tool` format
- C7: scan all generated `.py` files for hardcoded credential patterns (private key headers, AWS access key patterns, connection string patterns)

**`melleafy-json-consistency`**: 7 sub-checks verifying `melleafy.json` matches the other artifacts:

- Sub-check A: `melleafy.json` contains the fields required by the export command (the authoritative consumer). Hard-required fields (`manifest_version`, `entry_signature`, `package_name`) — FAIL if absent or if `manifest_version` < 1.1.0. Completeness fields (`source_runtime`, `modality`, `categories_resolved`, `declared_env_vars`, `pipeline_parameters`) — WARN if absent. Extra fields are permitted; no schema file is consulted.
- Sub-check B: `source_runtime` matches `classification.json:source_runtime`
- Sub-check C: `modality` matches `classification.json:modality`
- Sub-check D: `categories_resolved` counts match `dependency_plan.json` category counts
- Sub-check E: `declared_env_vars` set matches env-var references found in generated `.py` files
- Sub-check F: `entry_signature` matches the AST-derived signature of `pipeline.py:run_pipeline`
- Sub-check G: `pipeline_parameters` list matches `run_pipeline`'s parameter list

---

## Execution rules

**Within a tier**: collect all lint failures; don't halt on the first one.

**Between tiers**: Tier-1 and structural Tier-2 failures trigger the top-level repair loop (see `mellea-fy.md`) before halting. Lints in `_LINT_HALTS_IMMEDIATELY` bypass repair and halt the gate immediately. Tier 3 runs only when Tier 2 is entirely clean.

---

## Severity model

Each lint declares one of three severities. The gate uses the severity to decide whether a `verdict=fail` blocks the compile or is surfaced as a finding only.

| Severity  | Gate behaviour                                                                                                       |
| --------- | -------------------------------------------------------------------------------------------------------------------- |
| `error`   | Blocks compile (`overall_verdict = fail`). Triggers the repair loop unless the lint is in `_LINT_HALTS_IMMEDIATELY`. |
| `warning` | Surfaces in the report and stdout; does NOT block. `--strict` promotes warnings to blocking.                          |
| `info`    | Telemetry only. Never blocks, not even under `--strict`.                                                              |

Per-lint severity, tier, and halt-immediately membership are data, not prose — they live in `src/mellea_skills_compiler/compile/lints.py` (`_LINT_SEVERITY`, `_LINT_TIER`, `_LINT_HALTS_IMMEDIATELY`). Each entry carries an inline rationale comment; `test_each_lint_has_declared_severity` and `test_each_lint_has_declared_tier` guard against drift.

---

## Failure report: `intermediate/step_7_report.json`

The report's shape is a contract, not an example. The authoritative definition is the JSON Schema at `src/mellea_skills_compiler/compile/schemas/step_7_report.schema.json` (Draft-07). The writer (`compile/lints.py::run_lints`) validates every emitted report against the schema and logs a drift warning if they disagree.

Minimal valid instance:

```json
{
  "format_version": "1.1",
  "checked_at": "2026-05-18T12:00:00Z",
  "package_path": "ticket_triage_mellea/",
  "overall_verdict": "pass",
  "blocking_failures": 0,
  "warnings": 0,
  "info_failures": 0,
  "strict": false,
  "lints": []
}
```

For the full field definitions (including per-lint `effective_severity`, the optional `smoke_check` sub-report, `warnings_escalated_by_smoke`, and the `LintFailure` shape), open the schema file directly or run `jsonschema -i intermediate/step_7_report.json src/mellea_skills_compiler/compile/schemas/step_7_report.schema.json` to validate a real report.

---

## Stdout on failure

There is no custom formatter for failure output — the actual stdout is whatever `compile/lints.py::run_lints` and the surrounding CLI layer print today (`rich`-formatted logger output plus the JSON report path). The structured contract for downstream tooling is `intermediate/step_7_report.json` (see schema above), not the human-readable stdout. If you need to grep CI logs, key on `overall_verdict: "fail"` in the JSON report rather than on stdout strings; the stdout wording is not API-stable.

---

## What lints do NOT check

The lint suite is deliberately scoped to mechanical, structural properties. Out of scope by design:

- **Semantic correctness** — lints check structure, not "does this produce the right answer?". Behavioural correctness is the job of fixtures plus the smoke-check (below).
- **Style and formatting** — PEP 8, line length, import order, and similar style concerns. Run your formatter of choice as a separate step.
- **Dependency resolution** — whether `pip install -e .` succeeds is verified separately; see R15 in `mellea-fy.md`.

---

## Post-lint fixture smoke-check (`--run` mode, default ON)

After all three static tiers pass, `melleafy validate` executes a single fixture case from the generated package's `fixtures/` directory by default — the first entry of `ALL_FIXTURES`. The static lint suite alone cannot catch runtime errors (Mellea-output schema mismatches, prompt issues, `_safe_parse_with_fallback` returning wrong values), so default-on closes the loop between "compiled" and "actually executes." Pass `--no-run` to skip the smoke check (e.g. for fast static-only iteration). Pass `--run --all` to execute every fixture (1-2 minute per fixture, allow ~5 minutes per fixture for budget).

The smoke check produces one of three verdicts:

- **`passed`** — fixture executed to completion without exception. Exit code `ExitCode.SUCCESS`. `step_7b_report.json` records the fixture id, duration, and output schema type.
- **`failed`** — fixture raised an exception or violated an assertion the runner can detect. Exit code `ExitCode.SMOKE_CHECK_FAIL` (distinct from `ExitCode.LINT_FAIL` for static lint failure). `step_7b_report.json` records the traceback and fixture context. Does **not** trigger the repair loop — a fixture failure requires human review, not automated re-generation.
- **`skipped`** — LLM backend unreachable (e.g. Ollama not running, API endpoint timing out, missing API key). Exit code `ExitCode.SUCCESS` with a stderr warning: _"Fixture smoke-check skipped — LLM backend unreachable: <reason>. Re-run `mellea-skills validate <pkg> --run` once the backend is up to verify runtime behaviour."_ `step_7b_report.json` records the verdict as `skipped` with the underlying error. This keeps CI green for environments without an LLM while still nudging local users.

Exit codes are defined in `src/mellea_skills_compiler/exit_codes.py` (`ExitCode` IntEnum); CI scripts should import that module rather than hardcoding integer literals.

Detection of "backend unreachable" vs "fixture genuinely failed":

- `ConnectionError`, `TimeoutError`, `requests.exceptions.ConnectionError`, or any `httpx.ConnectError` thrown during `start_session()` → **skipped**
- Authentication errors (401/403 from a remote API) → **skipped** with a more specific message: _"backend unreachable: authentication failed (check API key or env vars)"_
- Any other exception (TypeError, ValueError, AssertionError, schema validation errors, `mellea` exceptions) → **failed**

The `--run` mode is invoked automatically at the end of the `compile` command — a green compile output now means _the package compiled, passed all 16 static lints, and successfully executed at least one fixture_. The `compile` command exits 0 on a `skipped` verdict (matching the local-CI convention) so users without an LLM backend can still get a passing compile, but the skip warning is printed loudly.
