# Slash-command CHANGELOG

History of changes to the `/mellea-fy*` slash command pack. Per-entry date annotations (`**Added**:`, `**Last-validated**:`, "Bug N fix") that used to live inline with individual KBs and lints belong here — the slash commands themselves keep only their top-level `**Version**: X.Y.Z (date)` header and link back here for change history.

This file is the single source of truth for "when did this rule land, when was it last re-validated against the library, what bug did this fix". Update it when you bump a version header, add a KB / lint / accept-set, or close out a bug fix that altered a rule.

Entries are reverse-chronological (newest first). Each entry is a single date heading; multiple changes on the same date go under one heading.

---

## 2026-05-18

### Slash-command refactor pass (B5 → B8)

Documentation-hygiene work consolidating duplicated rules into single sources of truth.

- **B5** — Extracted writer/renderer accept-sets into a new "Writer & renderer accept-sets" section in `mellea-fy-behaviours.md`. Replaced the inline scalar-only-constraint block (Amendment K) in `mellea-fy-generate.md` and the descriptor accept-set block (PEP 585 types / parenthesised signatures / module-qualified symbols) with one-paragraph summaries that point at the canonical section. Same pointer added to `mellea-fy-repair.md` and the wrapper-built repair prompt.
- **B6** — Made `_LINT_SEVERITY` (and the new `_LINT_TIER` + `_LINT_HALTS_IMMEDIATELY`) in `src/mellea_skills_compiler/compile/lints.py` the canonical source for per-lint gate classification. Deleted the duplicate severity table and tier enumeration from `mellea-fy-validate.md`. Added `test_each_lint_has_declared_tier` + `test_tier_values_are_valid` + `test_tier_counts_match_documented_classification` + `test_halts_immediately_references_known_lints` regression tests.
- **B7 (Phase 1 doc-only trims)** — Trimmed `mellea-fy-validate.md`'s severity-model preamble, execution-rules section, and "what lints don't check" section. Removed the duplicate "Style or formatting" bullet. Deleted the `melleafy lint` subcommand section (CLI help belongs in `--help`).
- **B8 (Phase 2 contract extractions)** —
  - Created `src/mellea_skills_compiler/exit_codes.py` with the `ExitCode` IntEnum (SUCCESS, GENERAL_ERROR, INVALID_INVOCATION, OUTPUT_CONFLICT, EXPORT_LINT_FAIL, DEPENDENCY_STRICT_HALT, LINT_FAIL, SMOKE_CHECK_FAIL). CLI sites in `cli.py` and `export/exporter.py` updated to import from there; numeric literals replaced.
  - Created `src/mellea_skills_compiler/compile/schemas/step_7_report.schema.json` (Draft-07). `run_lints` validates emitted reports against the schema and logs a drift warning on mismatch. The inline JSON example in `mellea-fy-validate.md` shrank to a minimal valid instance + pointer.
  - Created this CHANGELOG.md. Migrated date annotations out of per-KB / per-lint / per-accept-set descriptions.

### Writer/renderer accept-sets

- **ACCEPT-SET-1** (`config_emission` scalar-only, Amendment K) — added to `mellea-fy-behaviours.md`. Source amendment: `melleafy-handoff/amendments/2026-04-27-K-string-literal-safety-in-config.md`.
- **ACCEPT-SET-2** (descriptor accept-set: PEP 585 types, parenthesised signatures, module-qualified symbols) — added to `mellea-fy-behaviours.md`.

---

## 2026-05-17

### Bug 1 fix — descriptor-mode `pipeline.py` absence

- **`pipeline-entry-canonical` lint hard-fails when `pipeline.py` is absent.** Previously it skipped, which masked three real overnight-batch failures where the descriptor renderer rejected the IR and pipeline.py was never produced. Now any pre-Step-7 omission of `pipeline.py` surfaces as a lint failure rather than a silent miss.
- **Repair-mode invocation contexts split (Fix A).** `mellea-fy-repair.md` now documents two distinct invocation contexts: in-session repair (legacy) and wrapper-side repair via `--repair-on-lint-failure`. The wrapper pre-bakes structured fix prescriptions to `intermediate/repair_prescriptions.md` before spawning the repair session.
- **Audit §7-D2 closed.** Descriptor-mode `_compile_settings.json` deny-list now extends to `pipeline.py` and `schemas.py` so LLM Write/Edit calls on those paths are blocked at the tool layer.
- **`mellea-fy-validate-emissions.md` (Step 5b) added.** New opportunistic in-session check after Step 5 emissions; catches the `check(req)` arity defect class (5 separate skills hit the same defect in one batch, all caught by Step 7 but after the session had ended).

---

## 2026-04-29

### Known Behaviours (KBs) — entry creation / restoration

- **KB1** (`m.instruct()` returns `ComputedModelOutputThunk`, not Pydantic model) — restored from v0c116fa; scope broadened from `.parsed_repr` to all field access.
- **KB2** (JSON truncation on complex outputs) — restored from v0c116fa.
- **`mellea-fy-repair.md` v1.0.0** initial release.

---

## 2026-04-28

### `mellea-fy-validate.md` v4.3.1 / `mellea-fy.md` 4.3.2 / `mellea-fy-generate.md` v4.3.0

- **KB11** (Optional fields in P2 extraction schemas need explicit extraction guidance) — added. Lint: `known-behaviours` sub-check 3m. Fixture: `tests/promptfoo/kb_11.yaml`.
- **Doc-citation lint** doc-pages fallback list snapshot taken from `docs.mellea.ai` navigation as of this date; also imported by `compile/grounding.py` as its hardcoded fallback.

### Last-validated bulk-tag

The following KBs were re-validated against mellea 0.4.x on this date with their per-KB promptfoo fixtures: KB3, KB4, KB5, KB6, KB7, KB8 (one-line pointer), KB9 (one-line pointer), KB11. Each KB's `**Fixture**: tests/promptfoo/kb_NN.yaml` pointer (kept in the behaviours doc) is the location to re-run the validation.

---

## 2026-04-27

### KB6 — initial entry

- **KB6** (Reserved parameter names in `@generative` slots: `m`, `context`) — added. Mixed citation tier: `m` source-verified via `mellea.stdlib.components.genslot.py`; `context` empirically observed.

---

## How to use this file

When adding a new KB, lint, or accept-set entry to one of the slash commands:

1. Put the rule (the maintained content) in the slash command.
2. Add a `**Fixture**:` pointer in the entry if there's a regression fixture for it — this stays in the slash command since it's "where to test", not history.
3. Add an entry to this CHANGELOG with the date heading, the rule's id (KB-N, lint id, ACCEPT-SET-N), and one sentence describing what landed and why. Avoid `**Added**:` / `**Last-validated**:` annotations inside the slash-command entry itself.
4. When you re-validate an existing entry against a new library version, add a new dated section here (`### Re-validated against mellea X.Y.Z`) listing which entries you ran — don't update an inline `**Last-validated**:` field.

The slash-command `**Version**: X.Y.Z (date)` header stays as the at-a-glance "what version of this doc am I reading"; the history sits here.
