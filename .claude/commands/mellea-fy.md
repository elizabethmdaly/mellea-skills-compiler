# Melleafy: Decompose an Agent Spec into Mellea Code

**Spec version**: 4.3.2 (2026-04-28) — 10-step workflow with source-runtime detection, dependency audit, API reference grounding, and 25 formal lints with repair loop.

You are a Mellea decomposition specialist. Given a path to an agent `.md` file, produce an executable Python package using the Mellea generative programming library. This orchestrator file describes the overall workflow; step-specific guidance lives in the sub-commands listed below.

**Your input**: `$ARGUMENTS` — path to an agent `.md` file (or workspace directory for multi-file source runtimes), optionally followed by compilation flags (e.g. `--use-descriptor`). The first positional token is the spec path; remaining tokens starting with `--` are flags. See §Compilation flags below for the full list.
**Your output**: A generated Python package plus intermediate artifacts and a mapping report.

**Argument parsing**: split `$ARGUMENTS` on whitespace. The first non-flag token is the spec path. Any token of the form `--<flag>` (or `--<flag>=<value>`) is a compilation flag that MUST be carried through to the relevant sub-command. In particular, `--use-descriptor` MUST be propagated to `/mellea-fy-generate` so Step 5 takes the descriptor-mode code path; without this propagation the flag is silently dropped and the default legacy Step 5 runs.

---

## The 10-step workflow

Run these steps in order. Each step has a dedicated sub-command with the full specification.

```
[source spec on disk]
    │
    ▼
 Step 0: Classify the spec along five axes
    │   → classification.json
    │   Sub-command: /mellea-fy-classify
    ▼
 Steps 1a + 1b: Inventory files → tag elements + assign C1-C9 categories
    │   → inventory.json
    │   Sub-command: /mellea-fy-inventory
    ▼
 Step 2: Map elements to Mellea primitives
    │   → element_mapping.json (TOOL_TEMPLATE entries provisional)
    │   Sub-command: /mellea-fy-map
    ▼
 Step 2.5: Dependency audit + elicitation → commit dispositions + API reference
    │   → dependency_plan.json, element_mapping_amendments.json, mellea_api_ref.json
    │   Sub-command: /mellea-fy-deps   ← NEW in v4.0 — do not skip
    ▼
 Step 3: Emit skeleton files
    │   → empty Python files with structure (run_pipeline signature locked here)
    │
 Step 4: Generate fixtures
    │   → fixtures/ subpackage (5-8 fixtures, ≥3 C-categories)
    │   Sub-command: /mellea-fy-fixtures
    │   (uses Step 3 skeleton's run_pipeline signature as grounding source)
    ▼
 Step 5: Generate per-element code bodies
    │   → populated Python files (fixtures/ available as grounding context)
    │   Sub-command: /mellea-fy-generate  (covers Steps 3 + 5)
    ▼
 Step 5b: In-session validation of Step 5 emissions (opportunistic gate)
    │   → step_5b_report.json
    │   Sub-command: /mellea-fy-validate-emissions
    │   (applies a 9-lint subset against the just-emitted Python files; best-effort
    │    in-place repair; never halts the pipeline. Step 7 remains authoritative.)
    ▼
 Step 6: Emit supporting artifacts
    │   → mapping_report.md, melleafy.json, SETUP.md, README.md
    │   → SKILL.md (non-.md sources only — CLI compatibility shim, WIP)
    │   Sub-command: /mellea-fy-artifacts
    ▼
 Step 7: Static validation (25 formal lints)
    │   → step_7_report.json
    │   Sub-command: /mellea-fy-validate
    │
    ├── [PASS] ──────────────────────────────────────────────────────────────►
    │                                                                          ▼
    └── [FAIL — Tier 1 or structural Tier 2, repair_round < 2]        [generated package on disk]
              │
              ▼
         Re-invoke /mellea-fy-generate (repair mode, failing files only)
              │   → re-run Step 7, increment repair_round
              │
              └── [FAIL — repair_round = 2, OR session-boundary / category-specific]
                       → halt, preserve .melleafy-partial/
```

## Sub-command reference

| Sub-command             | Covers                                                                         | Key outputs                                      |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| `/mellea-fy-classify`   | Step 0: 5-axis classification                                                  | `classification.json`                            |
| `/mellea-fy-inventory`  | Steps 1a+1b: file scan + element tagging                                       | `inventory.json`                                 |
| `/mellea-fy-map`        | Step 2: tag → Mellea primitive routing                                         | `element_mapping.json`                           |
| `/mellea-fy-deps`       | Step 2.5: dependency audit + disposition commit                                | `dependency_plan.json`                           |
| `/mellea-fy-fixtures`   | Step 4: fixture generation (after skeleton, before bodies)                     | `fixtures/` subpackage                           |
| `/mellea-fy-generate`   | Steps 3+5: skeleton emit + body generation                                     | All Python files                                 |
| `/mellea-fy-validate-emissions` | Step 5b: in-session structural-lint subset on Step 5 outputs            | `step_5b_report.json`                            |
| `/mellea-fy-artifacts`  | Step 6: mapping report + melleafy.json + SKILL.md (if absent, non-.md sources) | `mapping_report.md`, `melleafy.json`, `SKILL.md` |
| `/mellea-fy-validate`   | Step 7: 25 formal lints                                                        | `step_7_report.json`                             |
| `/mellea-fy-behaviours` | Reference: KB3–KB9, KB11 workarounds                                           | (reference only — read before Step 4)            |

## Intermediate artifacts

The **canonical cross-stage I/O declaration** for the pipeline lives at
`src/mellea_skills_compiler/compile/pipeline_contract.py:PIPELINE_CONTRACT`.
Each step's inputs, outputs, and governing schemas are declared there
as a single source of truth; the per-step slash-command docs describe
the same artefacts in narrative form. Inspect or audit the contract via
`mellea-skills contract show` (Mermaid + topological order +
per-step I/O table) or `mellea-skills contract verify` (static checks
against producer/consumer alignment and schema-path resolvability).

All intermediate artifacts persist in `intermediate/` inside the output directory. A failed run leaves whatever was produced under `.melleafy-partial/` for debugging. The full artifact trail is:

```
intermediate/
  classification.json
  inventory.json
  element_mapping.json
  element_mapping_amendments.json   ← from Step 2.5d
  dependency_plan.json
  mellea_api_ref.json               ← from Step 2.5e (~280 KB; verification surface only — do NOT read end-to-end)
  mellea_api_ref.compatibility.json          ← sidecar (~1 KB; targeted-read: just the `compatibility` field)
  mellea_api_ref.forbidden_param_names.json  ← sidecar (~1 KB; targeted-read: just the `forbidden_param_names` field)
  element_mapping_judgment_calls.json
  coverage_report.json
  step_1b_trace.json
  step_5b_report.json               ← from Step 5b (in-session lint subset)
  step_7_report.json
```

## Key design principles

**Autonomous execution — no confirmation pauses.** Run all 10 steps from start to finish without stopping to ask the user whether to proceed. Do not output phrases like "Ready to proceed?", "Shall I continue?", or "Proceed to Step N?" between steps. Each step completes and the next begins immediately. The only permitted halts are: (a) Step 2.5 `ask` mode disposition elicitation, (b) a `strict` mode disposition conflict, or (c) a repair-loop exhaustion at Step 7. In all other cases, proceed. The full step sequence is: Step 0 → Step 1 → Step 2 → Step 2.5 → Step 3 → Step 4 → Step 5 → **Step 5b** → Step 6 → Step 7. Step 5b is opportunistic and never halts the pipeline regardless of its verdict — it produces structured evidence and best-effort in-place repairs, then transitions immediately to Step 6.

**Autonomous execution — IMMEDIATE tool invocation at step boundaries.** After completing each step, do NOT end your turn with a narrative line ("Step N complete. Proceeding to step N+1."). Instead, IMMEDIATELY invoke the first tool for the next step in the SAME turn. Narrative may follow the tool call, but the tool call MUST come first at each step boundary. Empirical observation: sessions that end with "Proceeding to..." narrative without an immediate follow-up tool call cause the Claude Code SDK to end the session and the wrapper to fail downstream lints. To avoid this failure mode, treat every step transition as a single tool invocation, not a sentence. The wrapper also supports an opt-in `--resume-on-early-end` flag (see `mellea_skills_compiler/cli.py::compile`) that detects this gap via canonical step artefacts and re-invokes Claude with a resume directive — up to 3 resume rounds per skill. The directive is appended automatically; you do not need to do anything special when resumed.

**Wrapper-side lint repair (Fix A — `--repair-on-lint-failure`, opt-in).** Separately from the in-session repair loop, the wrapper supports an opt-in flag (`--repair-on-lint-failure`, see `mellea_skills_compiler/cli.py::compile`) that addresses a different failure mode: Claude self-reports lints PASS in-session, the wrapper's post-session `run_lints` disagrees and detects ERROR-severity failures, but the session has already exited so the in-session repair loop cannot engage. When the flag is on, the wrapper spawns a SECOND Claude session with `./mellea-fy-repair`, pre-baking F1 fix prescriptions to `intermediate/repair_prescriptions.md` for that session to consume. Up to 2 repair rounds per compile. The two opt-in flags (`--resume-on-early-end` and `--repair-on-lint-failure`) address distinct failure modes — incomplete pipeline vs complete-but-broken — and compose naturally as independent retry layers: resume runs first if its flag is set (Layer 1: initial spawn, possibly re-invoked to fill in missing canonical step artefacts), then lint-repair runs after (Layer 2: independent flag check; post-session writer-renderer → lints → repair re-spawn if needed). Either can fire alone; both can fire in sequence on the same compile. Default is OFF for both; opt in per batch.

**Deterministic workflow with scoped LLM invocations** — melleafy is not an LLM agent. LLM invocations occur at specific, scoped steps: Step 1b (element tagging), Step 2 (narrow judgement calls), Step 4 (fixture generation), Step 5 (body generation), Step 6 (narrative prose). Steps 0, 1a, 2.5, 3, and 7 are entirely deterministic.

### Schemas to READ before emitting (universal)

Every intermediate JSON artefact you (the LLM) emit during this 10-step compile MUST conform to a canonical JSON Schema. Before drafting ANY of these emissions, READ the named schema in full via the `Read` tool. The schema is the ground truth — field names, `required` lists, and `additionalProperties: false` closures are non-negotiable.

Do NOT infer field names from training memory. Do NOT carry over field names from JSON-IR formats you've seen elsewhere (OpenAPI, GraphQL, Protobuf-as-JSON, etc.). The schema decides; your priors do not.

| Step | Emit this artefact | READ this schema FIRST |
|---|---|---|
| 0   | `intermediate/classification.json`         | `.claude/schemas/classification.schema.json` |
| 1   | `intermediate/inventory.json`              | `.claude/schemas/inventory.schema.json` |
| 2   | `intermediate/element_mapping.json`        | `.claude/schemas/element_mapping.schema.json` |
| 2   | `intermediate/expected_signature.json`     | `src/mellea_skills_compiler/descriptor/schemas/expected_signature.schema.json` |
| 2.5 | `intermediate/dependency_plan.json`        | `.claude/schemas/dependency_plan.schema.json` |
| 4   | `intermediate/fixtures_emission.json`      | `.claude/schemas/fixtures_emission.schema.json` |
| 5   | `intermediate/descriptor_emission.json` (descriptor mode) | `src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json` |
| 5   | `intermediate/config_emission.json`        | `.claude/schemas/config_emission.schema.json` |
| 6   | `<package>/melleafy.json`                  | `.claude/schemas/melleafy.schema.json` |

The canonical mapping (including consumption-side artefacts the wrapper emits) is at `.claude/data/artefact-schemas.json` — a machine-readable single source of truth that this table reflects.

**After drafting each emission, re-open the schema and self-check.** Walk every top-level field and every repeated sub-shape against the corresponding schema definition. Verify field names, presence of `required` siblings, absence of fields not in `properties`. The schema gate at the writer-renderer post-session catches violations with precise JSON-path errors; emissions that bypass this self-check at draft time consistently produce schema-gate failures that halt the compile.

**Empirically observed failure modes** that this directive exists to prevent (2026-05-19 to 2026-05-20):

- `fixtures_emission.json` emitted with `fixture_id`/`name`/`category_tags`/`expected_output_hints`/`format_version` — none of which are in the schema's closed property set. The canonical fixture entry uses `id` and the closed property set declared in the schema.
- `descriptor_emission.json` `/pipeline` emitted as `{"id":..., "kind":"sequence", "steps":[...]}` object — the schema requires a flat array. The `{id, kind, steps}` shape is the inner-`sequence`-node shape, not the top-level pipeline.
- `descriptor_emission.json` `bound_to` emitted as bare string — the schema requires `{ref: "<name>"}` (a Ref object).
- `descriptor_emission.json` `notes` emitted as scalar string — the schema requires array of strings.
- `descriptor_emission.json` `skill.classification.primary_axis` set to a modality value (`synchronous_oneshot`) — that value belongs in `output_modality`/`input_modality`; `primary_axis` accepts axis enum values (`DOM`/`AGENT`/`DSL`/etc.).
- `descriptor_emission.json` `CallNode` emitted with invented field `callee` (schema uses `symbol`) or invented field `returns` (no such CallNode field — return values are named via the node's own `id`).

If you find yourself emitting any of those shapes, you are emitting from training memory rather than from the schema. Stop and re-read the schema.

**Source fidelity** — every significant line of the source spec becomes an inventory element (≥95% coverage). Nothing is silently skipped.

**Dispositions are explicit** — Step 2.5 produces a `dependency_plan.json` where every external dependency has an explicit disposition (`bundle`, `real_impl`, `stub`, `mock`, `delegate_to_runtime`, `external_input`, `load_from_disk`, or `remove`). In `auto` mode, defaults are applied silently; in `ask` mode, the user approves each; in `strict` mode, any stub-requiring disposition halts before writing files.

**One BaseModel per session** — schema priming (KB 5) is the most impactful Known Behaviour. All generated code must respect the one-schema-per-session rule. See `/mellea-fy-behaviours` for the full KB list.

**Lint severity drives gate behaviour** — Step 7's lints run unconditionally; their classified severity (`error` / `warning` / `info`) decides whether a failure blocks compilation. Only `error`-severity failures block compile and trigger the bounded repair loop: `/mellea-fy-generate` is re-invoked (failing files only, with exact lint messages as context) for up to 2 rounds before halting. `warning` and `info` failures surface in the report but DO NOT trigger repair. `session-boundary` and `category-specific` errors always halt immediately — no repair is attempted. Pass `--strict` to restore the legacy "every failure blocks" behaviour (warnings promote to blocking; info stays telemetry-only). See `mellea-fy-validate.md` for the full severity table and lint details.

## Output directory layout

**Rule OUT-1 — Co-location model.** Output is written into the same directory as the source spec (or the workspace directory for multi-file runtimes). The directory containing the spec IS the skill root. The compiled package is created as a subdirectory of the skill root.

- Input: `<skill-root>/spec.md` (e.g. `path/to/weather/spec.md`)
- Skill root: `<skill-root>/` (e.g. `path/to/weather/`)
- Compiled package: `<skill-root>/<package_name>/` (e.g. `path/to/weather/weather_mellea/`)

**Rule OUT-2 — Package name derivation.** `<package_name>` is a valid Python identifier derived from the skill's frontmatter `name:` field (or skill directory name for multi-file runtimes — CrewAI, LangGraph, Letta, etc.):

1. Take the `name:` value (or directory name)
2. Lowercase
3. Replace hyphens and spaces with underscores
4. Append `_mellea` suffix
5. Strip any leading/trailing underscores; collapse double underscores

Examples: `weather` → `weather_mellea` | `security-review` → `security_review_mellea` | `research-lead` → `research_lead_mellea`

**Rule OUT-3 — Package directory contains all compiled output.** With one exception — `pyproject.toml` (Step 3) — every file generated by melleafy is written inside `<package_name>/`. The skill root contains the source spec, `pyproject.toml`, and any source files preserved for non-.md runtimes:

```
<skill-root>/                           ← wherever the source spec lives
│
├── spec.md / SKILL.md                  ← source spec (untouched by melleafy)
├── pyproject.toml                      ← Step 3 — melleafy-generated file at skill root
│
│   ── Source files for non-.md runtimes (preserved at skill root) ──
├── agents.yaml / crew.py / ...
│
│   ── Companion directories (preserved at skill root; mirrored into <package_name>/ — Rule OUT-6) ──
├── scripts/                            ← optional; mirrored at Step 3
├── references/                         ← optional; mirrored at Step 3
├── assets/                             ← optional; mirrored at Step 3
│
└── <package_name>/                     ← e.g. weather_mellea/ — all other output
    │
    │   ── Python package files ──
    │   (in `--use-descriptor` mode, `pipeline.py` + `schemas.py` are rendered
    │   by the wrapper from `intermediate/descriptor_emission.json` via
    │   `compile/writer_renderer.py::render_descriptor_to_python`, alongside
    │   `config.py` (from `config_emission.json`) and `fixtures/` (from
    │   `fixtures_emission.json`). The wrapper-rendered set is denied to
    │   Claude's Write/Edit tools via `_compile_settings.json` so the LLM
    │   emits the corresponding `intermediate/*_emission.json` IR instead
    │   (audit §7-D2 closed 2026-05-18). In legacy mode Claude writes
    │   `pipeline.py` and `schemas.py` directly.)
    ├── __init__.py
    ├── __main__.py
    ├── pipeline.py
    ├── config.py
    ├── schemas.py
    ├── main.py
    ├── requirements.py                 ← conditional
    ├── slots.py                        ← conditional
    ├── tools.py                        ← conditional
    ├── constrained_slots.py            ← conditional
    ├── mobjects.py                     ← conditional
    └── loader.py                       ← conditional
    │
    │   ── Documentation & manifests ──
    ├── melleafy.json                   ← Step 6
    ├── mapping_report.md               ← Step 6
    ├── README.md                       ← Step 6
    ├── SETUP.md                        ← Step 6, conditional
    ├── SKILL.md                        ← Step 6, non-.md sources only (generated if absent)
    ├── dependencies.yaml               ← Step 2.5, conditional
    │
    │   ── Bundled runtime assets (Rule OUT-6 — mirrored from skill root at Step 3) ──
    ├── scripts/                        ← if <skill-root>/scripts/ exists
    ├── references/                     ← if <skill-root>/references/ exists
    ├── assets/                         ← if <skill-root>/assets/ exists
    │
    │   ── Test fixtures ──
    ├── fixtures/                       ← Step 4
    │   ├── __init__.py
    │   └── <case>.py ...
    │
    │   ── Intermediate artifacts ──
    └── intermediate/
        ├── classification.json         ← Step 0
        ├── inventory.json              ← Step 1b
        ├── element_mapping.json        ← Step 2
        ├── element_mapping_amendments.json ← Step 2.5d
        ├── dependency_plan.json        ← Step 2.5c
        ├── mellea_api_ref.json         ← Step 2.5e
        ├── element_mapping_judgment_calls.json
        ├── coverage_report.json
        ├── step_1b_trace.json
        ├── step_5b_report.json         ← Step 5b
        └── step_7_report.json          ← Step 7
```

**Rule OUT-4 — `fixtures/` is inside `<package_name>/`.** `fixtures/` is written inside `<package_name>/`, not at skill root. The `pyproject.toml` `[tool.setuptools.packages.find]` includes only `<package_name>*` — `fixtures/` is excluded from the installed package but physically inside the package directory for CLI discoverability. Run fixtures via `python -m pytest <package_name>/fixtures/` from the skill root.

**Rule OUT-5 — `.melleafy-partial/` on failure.** When a run fails (Step 7 lint failure or earlier halt), in-progress artifacts are preserved at `<skill-root>/.melleafy-partial/` — a sibling of `<package_name>/` within the skill root. Inspect this directory to debug the failure; it is safe to delete once the issue is resolved. Re-running after fixing will overwrite it.

**Rule OUT-6 — Companion-directory mirror.** Companion directories at the skill root (`scripts/`, `references/`, `assets/`) are mirrored into `<package_name>/` at Step 3 (skeleton emission), _before_ any code body generation. The skill-root copy is the source of truth (untouched by melleafy on subsequent runs); the package copy is treated as compiled output (regenerated each run). The mirror makes the package self-contained: any code inside `<package_name>/` that needs to invoke a bundled script or load a bundled reference MUST resolve the path package-relatively via `Path(__file__).parent / "<dir>/<file>"` — never via a user-supplied `repo_root` argument or the process working directory. Companion directories that are absent at the skill root are not created in the package. The pyproject.toml `[tool.setuptools.package-data]` section (Step 3) declares these directories so they are included in the installed wheel.

---

## Generation modes

Pass `--dependencies=<mode>` to control disposition elicitation:

| Mode            | Behavior                                                             |
| --------------- | -------------------------------------------------------------------- |
| `auto`          | Apply category default dispositions; print recap if any stubs result |
| `ask`           | Interactive terminal UI — approve/override each dependency           |
| `config:<path>` | Read dispositions from a JSON config file                            |
| `strict`        | Halt before writing files if any disposition would produce a stub    |

Default: `auto`.

## Compilation flags

Pass these flags on the `mellea-skills compile` CLI to control the compile path. The slash-command workflow reads them from `$ARGUMENTS` and routes Step 5 accordingly.

| Flag | Behaviour |
|---|---|
| `--use-descriptor` | Route Step 5 through descriptor IR emission + render instead of free-form Python emission. Default: off. Step 7 (the 25 lints) continues to run on the rendered output as a renderer safety net per plan §10.5. See `mellea-fy-generate.md` §"Descriptor mode (`--use-descriptor`)" for details. |
| `--repair-mode` (`-r`) | Bounded repair loop on validation/render/smoke failure. With `--use-descriptor`, repairs splice corrected nodes into the descriptor; with the default legacy path, repairs target failing Python files. See `mellea-fy-validate.md` for repair semantics. |
| `--no-run` | Skip the post-compile fixture smoke check. |
| `--refresh-cache` | Force-refresh the grounding/docs cache (`~/.cache/mellea-skills-compiler/`) before compile. |
| `--model` / `-m` | Override the model used for compilation. Defaults from `runtime_defaults.json`. |
| `--skill-backend` | Override the runtime LLM backend the compiled skill uses. Does NOT affect compilation. |
| `--skill-model` | Override the runtime model the compiled skill uses. Does NOT affect compilation. |

**Descriptor mode is a Step-5-only swap.** Steps 0–4 (classify, inventory, map, deps, fixtures) and Step 6 (artefacts) run identically in both modes. Step 7 (the 25 lints) also runs identically — its *role* changes (from "catch LLM Python mistakes" to "catch renderer-emitted Python regressions"), but the lint code does not. This is the explicit plan §10.5 commitment.

Until Phase 5 flips the default, descriptor mode is opt-in via `--use-descriptor`. The legacy free-form Python path remains the default for backward compatibility.
