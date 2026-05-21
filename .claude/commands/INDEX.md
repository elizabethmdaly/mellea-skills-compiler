# Slash-command reference index

Single registry for every rule id, requirement number, Known Behaviour, dependency category, and phase label that the `/mellea-fy*` slash commands cross-reference. Read this when you encounter a bare `Rule 5-2` or `R19` or `KB6` and want to know what it means without grepping the whole doc set.

Seven distinct numbering schemes are in use; each names a real distinction (different namespace, different owning doc, different lifecycle), so they are not merged. When adding a new entry, pick the scheme that matches the kind of thing you're naming and put the entry both in its owning slash command and in this index.

| Prefix          | What it indexes                                                | Owning doc                                                                                |
| --------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `Rule X-Y`      | A rule that applies at Step `X` of the compile pipeline.       | The step's own slash command (e.g. `mellea-fy-generate.md` for Rule 3-*, 5-*).            |
| `Rule OUT-N`    | A package-output / file-layout convention.                     | `mellea-fy.md` (declared in §Output directory layout).                                    |
| `R-N`           | A numbered requirement from the original spec.                 | `mellea-fy.md` and per-step commands. Likely a closed namespace — see §R-N section below. |
| `R-SEM-NAME`    | A semantic validator rule for descriptor IR (Step 5 emission). | `mellea-fy-generate.md` (descriptor mode section).                                        |
| `KB-N`          | A Mellea-library known-behaviour workaround.                   | `mellea-fy-behaviours.md`.                                                                |
| `C-N`           | A dependency category (C1–C9 taxonomy).                        | `mellea-fy-deps.md`.                                                                      |
| `P-N` / `P-N.X` | Pipeline-dispatch pattern or project phase. Overloaded.        | See §P-N section below — the prefix has two distinct meanings.                            |

When you add or retire an entry, also update `CHANGELOG.md` with a dated line under the affected scheme.

---

## Rule X-Y — step-keyed pipeline rules

Format: `Rule <step-number>-<sequence>`. The step number matches the compile-pipeline step the rule applies to; the sequence is local within that step.

| Id           | One-line meaning                                                                                              | Owner                    | Status |
| ------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------ | ------ |
| `Rule 2.5-2` | Bundled-asset path resolution: `real_impl`/`load_from_disk` code resolves paths via `Path(__file__).parent`.  | `mellea-fy-deps.md`      | live   |
| `Rule 3-1`   | Every `run_pipeline` parameter has an explicit Python type annotation (default `str` if untyped in spec).     | `mellea-fy-generate.md`  | live   |
| `Rule 3-2`   | `run_pipeline` is the canonical entry-point name in `pipeline.py`; helpers may exist alongside it.            | `mellea-fy-generate.md`  | live   |
| `Rule 3-3`   | `run_pipeline` is decorated with `@validate_call(config={"arbitrary_types_allowed": True})`.                  | `mellea-fy-generate.md`  | live   |
| `Rule 4-1`   | Batched fixture generation as a single JSON emission conforming to `fixtures_emission.schema.json`.           | `mellea-fy-fixtures.md`  | live   |
| `Rule 5-2`   | Import path grounding: verify `mellea.X` exists in `mellea_api_ref.json:.modules` before emitting an import.  | `mellea-fy-generate.md`  | live (fallback) |
| `Rule 5-3`   | One file per LLM invocation in Step 5 body generation.                                                        | `mellea-fy-generate.md`  | live   |
| `Rule 5-4`   | Stdlib function signature grounding: verify argument count/keywords against `mellea_api_ref.json`.            | `mellea-fy-generate.md`  | live (fallback) |
| `Rule 6-1`   | `melleafy.json:categories_resolved` is a JSON object keyed by category code, never an array.                  | `mellea-fy-artifacts.md` | live   |
| `Rule 6-2`   | C1 entries in `categories_resolved` include a `description` field sourced from `inventory.json`.              | `mellea-fy-artifacts.md` | live   |

---

## Rule OUT-N — package-output conventions

Constrain how the compiled package is written to disk: where files go, what survives a failure, how `pip install` interacts with the layout.

| Id          | One-line meaning                                                                                              | Owner                      | Status |
| ----------- | ------------------------------------------------------------------------------------------------------------- | -------------------------- | ------ |
| `Rule OUT-1` | Co-location: output is written into the same directory as the source spec; that directory IS the skill root. | `mellea-fy.md`             | live   |
| `Rule OUT-2` | Package directory naming: derive from the source spec's `name:` frontmatter; the `*_mellea/` subdir.          | `mellea-fy.md`             | live   |
| `Rule OUT-3` | Step 6 artifacts (`mapping_report.md`, `melleafy.json`, `SETUP.md`, `README.md`, `SKILL.md`) live inside the package. | `mellea-fy-artifacts.md` | live   |
| `Rule OUT-4` | `fixtures/` lives inside the package and is excluded from the installed wheel.                                | `mellea-fy-fixtures.md`    | live   |
| `Rule OUT-5` | On failure, in-progress artifacts are preserved at `<skill-root>/.melleafy-partial/` (sibling of the package). | `mellea-fy.md`            | live   |
| `Rule OUT-6` | Bundled companion dirs (`scripts/`, `references/`, `assets/`) are mirrored into the package by Step 3a-pre.   | `mellea-fy-deps.md`        | live   |

---

## R-N — numbered spec requirements

Originally from the project's main spec doc. Sparse numbering (R1, R2, R10, R14–R16, R19–R21 referenced; R3–R9, R11–R13, R17–R18 absent) suggests the gaps are retired requirements that have either been folded into other rules or dropped entirely. **Treat R-N as a closed namespace** — when adding a new compile-time rule, prefer `Rule X-Y` (step-keyed) or `Rule OUT-N` (output-keyed) over allocating a new `R-N`.

| Id   | One-line meaning                                                                                  | First referenced from         | Status |
| ---- | ------------------------------------------------------------------------------------------------- | ----------------------------- | ------ |
| `R1` | Hybrid source-runtime detection rule (multi-runtime classification).                              | `mellea-fy-classify.md`       | live   |
| `R2` | Generated `SKILL.md` is suppressed on re-run for safety.                                          | `mellea-fy-artifacts.md`      | live   |
| `R10` | "Detected but not handled (deferred)" mapping bucket for features melleafy v1 does not support.  | `mellea-fy-deps.md`           | live   |
| `R14` | Auto-mode recap section in `mapping_report.md`.                                                  | `mellea-fy-artifacts.md`      | live   |
| `R15` | `pip install -e .` is verified as a separate, user-run check (out of scope for static lints).    | `mellea-fy.md`                | live   |
| `R16` | Fixture coverage threshold: collectively exercise ≥3 dependency categories.                      | `mellea-fy-fixtures.md`       | live   |
| `R19` | Module-level import-time purity (no `load_dotenv`, no network, no I/O). Enforced by `import-side-effects`. | `mellea-fy-validate.md`       | live   |
| `R20` | `melleafy.json` finalisation contract (Step 6 manifest emission).                                | `mellea-fy-artifacts.md`      | live   |
| `R21` | Modality-specific entry-point shape — `run_pipeline` signature varies by detected modality.      | `mellea-fy-generate.md`       | live   |

A future audit pass (deferred from this index work) can confirm whether R3–R9, R11–R13, R17–R18 are properly retired or are still live in docs we haven't pulled into this view.

---

## R-SEM-NAME — descriptor semantic validator rules

Rules enforced by the descriptor IR validator at `descriptor/validator.py::validate()`. These fire during the wrapper's post-session descriptor-render path (descriptor mode only).

| Id                       | One-line meaning                                                                                                          | Owner                   | Status |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ------ |
| `R-SEM-SIGNATURE-MATCH`  | Descriptor's `inputs`/`outputs`/`schemas` must match the locked I/O signature in `intermediate/expected_signature.json`. | `mellea-fy-generate.md` | live   |

Schema home: `src/mellea_skills_compiler/descriptor/schemas/descriptor.schema.v0.3.json`.

---

## KB-N — Mellea-library Known Behaviours

Workarounds for documented or empirically-observed quirks in the Mellea library. Authoritative home: `mellea-fy-behaviours.md` (each KB has its own `## KB<N>:` section, plus a `**Fixture**:` pointer where a regression fixture exists). KB introduction/retirement dates live in `CHANGELOG.md`, not here.

| Id     | One-line meaning                                                                                              | Section header in behaviours.md      | Status |
| ------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------ |
| `KB1`  | `m.instruct(format=...)` returns a `ComputedModelOutputThunk`, not a parsed Pydantic model.                  | `## KB1:`                            | live   |
| `KB2`  | JSON truncation on complex outputs; use `_safe_parse_with_fallback` or `RepairTemplateStrategy`.             | `## KB2:`                            | live   |
| `KB3`  | `validation_fn` receives `Context`, not `str` — wrap lambdas with `simple_validate(...)`.                    | `## KB3:`                            | live   |
| `KB4`  | Validators on `m.instruct(format=...)` receive raw JSON strings; four anti-patterns documented.              | `## KB4:`                            | live   |
| `KB5`  | Schema priming: one `BaseModel` `format=` type per `start_session()` block. Enforced by `session-boundary`.  | `## KB5:`                            | live   |
| `KB6`  | Reserved `@generative` parameter names: `m` and `context` cause `ValueError` at decoration time.             | `## KB6:`                            | live   |
| `KB7`  | Persona text via `ModelOption.SYSTEM_PROMPT`; `prefix=` is an output prefix, not a system prompt.            | `## KB7:`                            | live   |
| `KB8`  | WatsonX deprecation pointer (one-line guidance only; no full section).                                       | — (one-line pointer only)            | retired-as-section, surviving as guidance |
| `KB9`  | `return_sampling_results` for debugging (advisory; no lint).                                                  | `## KB9:`                            | live   |
| `KB10` | (Reserved id — not currently used in the doc set.)                                                            | —                                    | retired |
| `KB11` | `Optional` fields in P2 extraction schemas need explicit extraction guidance ("extract", "do not ask", etc.). | `## KB11:`                           | live   |
| `KB12` | `instruct-has-description` invariant — every `m.instruct(...)` must supply `description=`. (Mellea 0.5+.)    | inline in `mellea-fy-validate.md`    | live (lint-only, no behaviours-doc section) |

---

## C-N — dependency categories

The dependency-plan taxonomy: every detected dependency in `dependency_plan.json` is tagged with one of C1–C9. Defaults table lives in `mellea-fy-deps.md` §"Category defaults".

| Id   | Category                                          | Default disposition                       |
| ---- | ------------------------------------------------- | ----------------------------------------- |
| `C1` | Identity (persona text).                          | `bundle`                                  |
| `C2` | Operating rules (behavioural constraints).        | `bundle`                                  |
| `C3` | User / environment facts.                         | `bundle` (stable) / `external_input` (overridable) |
| `C4` | Short-term / session state.                       | `delegate_to_runtime`                     |
| `C5` | Long-term memory.                                 | `delegate_to_runtime`                     |
| `C6` | Tools and capability declarations.                | `real_impl` (concrete) / `stub` (abstract) |
| `C7` | Credentials and secrets.                          | `external_input`                          |
| `C8` | Runtime environment (model id, backend).          | `bundle`                                  |
| `C9` | Scheduling and triggers (cron, webhooks, events). | `delegate_to_runtime`                     |

---

## P-N — overloaded prefix (pipeline-dispatch patterns AND project phases)

The `P-N` prefix carries two unrelated meanings, distinguished by the form of the suffix. **When adding a new entry, prefer one of the existing forms** rather than introducing a third use of this prefix.

### `P0`, `P2`, `P3`, `P4` — pipeline-dispatch patterns

Patterns for how the generated pipeline calls (or doesn't call) tools. Lives in `mellea-fy-deps.md` §"Pipeline-dispatch patterns".

| Id   | Pattern name                              | Generated files                                |
| ---- | ----------------------------------------- | ---------------------------------------------- |
| `P0` | No tools (pure reasoning).                | No `tools.py`, no `dependencies.yaml`.         |
| `P2` | Pipeline calls tools (deterministic).     | `tools.py` with allowlist + `constrained_slots.py`. |
| `P3` | Pipeline calls tools (LLM-directed).      | `tools.py` with `m.react()`.                   |
| `P4` | Tools provide input (tools run first).    | `dependencies.yaml`, optionally `loader.py`.   |

(P1 is intentionally not in this enum — see below.)

### `P1.X`, `P3.5.X` — project / pipeline phases

Phase identifiers for internal development tracking. The suffix letter (`A`–`Z`) sequences sub-phases within a numbered phase.

| Id        | One-line meaning                                                                                                | Reference                  |
| --------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `P1.C`    | Cache refresh phase for the grounding artifacts (`mellea_api_ref.json`, `mellea_doc_index.json`).               | `mellea-fy-generate.md` (referenced from `--refresh-cache` flag) |
| `P3.5.A`  | Descriptor-emission canonical algorithm — the 8-artefact-prompt pattern in Step 5 descriptor mode.              | `mellea-fy-generate.md` §descriptor mode |
| `P3.5.D`  | `expected_signature.json` emission — Step 2 always produces it; `R-SEM-SIGNATURE-MATCH` enforces conformance.    | `mellea-fy-fixtures.md`, `mellea-fy-generate.md` |

---

## How to add a new entry

1. Decide which scheme matches what you're naming. If it could plausibly belong to two schemes, use the table at the top of this file to pick the one whose owning doc covers your rule's audience.
2. Add the entry to its owning slash command with the canonical definition.
3. Add a row in the appropriate table here with a one-line summary and `status: live`.
4. Add a dated entry to `CHANGELOG.md` under the introduction date, naming the new id.

If you retire an entry, change its `status` here to `retired`, leave the row (do not delete it — bare references in older code might still appear), and note the retirement in `CHANGELOG.md`.

## Cross-references this index does NOT cover

This file indexes prefixes that appear repeatedly across the slash-command pack. It deliberately does NOT enumerate:

- **Lint ids** (e.g. `parseable`, `session-boundary`, `pipeline-entry-canonical`) — those live in `_LINT_SEVERITY` / `_LINT_TIER` in `src/mellea_skills_compiler/compile/lints.py` (canonical) and are summarised in `mellea-fy-validate.md`.
- **Schema file names** (e.g. `fixtures_emission.schema.json`) — discoverable from the file system (`.claude/schemas/*.schema.json` and `src/mellea_skills_compiler/.../*.schema.json`).
- **Intermediate-artifact file names** (e.g. `dependency_plan.json`, `mellea_api_ref.json`) — discoverable from the file system (`<package>/intermediate/`).
- **Exit codes** — live in `src/mellea_skills_compiler/exit_codes.py` (`ExitCode` IntEnum).
- **Writer / renderer accept-sets** (`ACCEPT-SET-N`) — live in `mellea-fy-behaviours.md` §"Writer & renderer accept-sets".

Each of those has its own canonical home and is well-indexed there; duplicating them here would just create the drift problem this index was created to solve.
