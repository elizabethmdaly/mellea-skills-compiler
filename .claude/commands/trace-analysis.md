# Trace Analysis

**Version**: 0.1 (2026-05-21) — initial extensible scaffold. Add new check categories as we learn what to watch for.

**Input**: a path to `intermediate/claude_stream.jsonl` from a Mellea-fy compile (or a skill-root directory containing `<package>/intermediate/`).

**Output**: a structured markdown report covering trace integrity, grounding usage, schema/validation behavior, confusion signals, failure patterns, and known-pattern matches. Ends with a "Top observations" summary highlighting what the user should pay attention to.

---

## Instructions

1. **Resolve the trace path**. If the user supplied a directory, locate `<dir>/<package>/intermediate/claude_stream.jsonl`. If they supplied a file path, use it directly. If neither resolves, halt with the resolution attempt.
2. **Inspect file size first** with `ls` or `stat`. If >256 KB, plan to scan in chunks using `Read` with offsets — do NOT attempt a single end-to-end read.
3. **Run every check in §1–§8 below**. For each, report what was found with concrete evidence (line numbers, counts, quoted snippets where useful). When a signal is *absent*, explicitly state that — silence is itself information.
4. **Cross-reference adjacent artifacts** when relevant: `descriptor_emission.json`, `classification.json`, `step_7_report.json`, `symbol_normalisations.jsonl`. The trace says what the LLM *did*; these say what the *outcome* was. Both matter.
5. **Produce the report in the format defined in §9** at the end.

---

## §1. Trace integrity

What to check:

- **File size + estimated event count**. Quick gauge of session length.
- **`_meta_event": "session_boundary"` markers**. Count. Zero = single session; ≥1 = repair-loop fired and traces preserved across rounds (the fix landed 2026-05-21). Report each session-boundary's timestamp.
- **Session UUIDs**. Note all distinct `session_id` values that appear — useful to correlate across log layers.
- **Tool denial errors**. Count occurrences of `is_error: true` tool results. High counts suggest the LLM is hitting permission walls (`_compile_settings.json` deny list).

## §2. Phase 2 canonical-injection signals

What to check:

- Did the LLM **read a canonical file** from `src/mellea_skills_compiler/canonical_descriptors/`? Search for `canonical_descriptors` in `Read` tool input paths.
- If so, **which canonical** (`<archetype>_<shape>_<modality>.json`)? Did it match the skill's `classification.json` triple?
- If not, **why not**? Look for: the LLM constructing the filename in thinking, attempting a Read that returned "file not found", or skipping the lookup entirely.
- Did the LLM **consult the canonical's `metadata.notes`** field after reading? The notes describe what the canonical demonstrates vs. what it deliberately omits — silent consumption means the LLM may pattern-match too aggressively.

Expected behavior (post-Phase-2 wiring 2026-05-21): the LLM reads `intermediate/classification.json`, constructs the canonical filename, reads it from `canonical_descriptors/`, and treats it as a *structural reference* for descriptor composition.

## §3. Surface usage patterns

What to check:

- **Read attempts on `mellea_api_ref.json`** — count, and check for offsets. End-to-end reads of the 280 KB monolithic file are wasteful and may hit the 256 KB Read-tool limit.
- **256 KB Read-limit errors** — search for `"exceeds maximum allowed size (256KB)"`. This is the smoking gun for surface-as-primary-read misuse.
- **Sidecar usage** — did the LLM read `mellea_api_ref.compatibility.json` or `mellea_api_ref.forbidden_param_names.json` (~1 KB targeted files)? These landed 2026-05-21 and should be the LLM's lookup path for those small fields.
- **"Let me search for" / "search toward the end" patterns** — narration that indicates the LLM is treating a large file as a discoverability surface instead of a verification surface.
- **Re-reads of the same file** — count unique paths the LLM re-read. Some re-reading is normal; excessive re-reads suggest confusion or working-memory loss.

Expected behavior: targeted reads with line offsets for specific symbol lookups. Sidecars used for `compatibility` / `forbidden_param_names`. No 256 KB-limit errors.

## §4. Schema and validation behavior — known wrong shapes

What to check:

- **JSON Pointer refs** in value-binding positions — search for `#/state/`, `#/inputs/`, `#/outputs/` inside `"ref":` fields. Should be zero post-2026-05-20 (Ref.ref pattern catches these structurally).
- **Self-referential `bound_to`** — search for patterns where `bound_to.ref` matches the same node's `id`. Repair-recoverable but wasteful.
- **Local-helper symbols in `pipeline.call.symbol`** — search for `loader.`, `slots.`, `helpers.`, `tools.` as the head segment of a call symbol. Class 2 architectural gap (no fix yet); confirms the failure pattern is recurring.
- **Function-name prefix in signatures** — search for signature strings starting with a word + `(` rather than just `(`. Fails parseability; directive says don't do it but the LLM still does it on first emission.
- **Invented CallNode fields** — search for `"callee"` (should be `"symbol"`) or `"returns"` (no such field; the node's own `id` carries the result name).
- **Non-scalar values in `config_emission.json`** — search for dict / list values inside config entries (Amendment K closed these).
- **`bytes` type in `SchemaField.type`** — should now pass (added to vocab 2026-05-20). If rejected, regression.

## §5. Confusion / uncertainty markers

What to check (these are narrative patterns — paraphrase, don't require exact strings):

- Phrases like *"let me check"*, *"I need to verify"*, *"I'm not sure if"*, *"let me look"*, *"let me re-read"* — count occurrences per session.
- Repeated tool invocations on the same file (especially `Read` on the same path with similar/identical offsets) — suggests the LLM didn't extract what it needed the first time.
- The LLM **abandoning a partial attempt and restarting** — search for "let me try a different approach" / "actually" pivots after partial work.
- The LLM **mentioning conflicting information** between two sources — e.g., "the spec says X but the API ref says Y."

Expected baseline: some uncertainty is normal. Track magnitude over time — if a future trace has 3× the uncertainty markers vs an earlier trace on a similar skill, something regressed.

## §6. Failure / repair signals

What to check (cross-reference with the wrapper-side log if present):

- **Validator errors** appearing in tool results — search for `R-SEM-` rule tags, `descriptor_emission.json violates`, `symbol gate rejected`, `RendererError`.
- **Repair-round count** — number of `_meta_event` session_boundary markers + 1 = total sessions = (1 initial + N repair rounds).
- **Error class transitions across rounds** — did each repair round close *all* error classes, or did it touch some and leave others (the "triage failure" pattern)?
- **Repair targeting derived files** — did the repair edit `pipeline.py` / `schemas.py` in descriptor mode when it should have edited `descriptor_emission.json`? Task #16 tracks this; if present, the bug is still active.

## §7. Behavior conflicting with current expected guidance

What to check (these are anti-patterns the directive is supposed to prevent):

- LLM reading **mellea_api_ref.json end-to-end** despite the new "verification surface, not primary read" framing.
- LLM **skipping Step 1b** (canonical-descriptor read) despite the directive.
- LLM **emitting JSON Pointer refs** despite the schema pattern + directive + canonical example all saying not to.
- LLM **putting local helpers in pipeline.call.symbol** despite the directive saying these belong in dependencies/loader.py.
- LLM **including function name in signature** despite both the directive and the canonicals showing the bare `(args) -> Return` form.

Each of these is signal that the current grounding strategy isn't fully landing. Quantify, don't just note presence.

## §8. Token / cost telemetry (best-effort)

What to check:

- Sum `usage.input_tokens`, `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens`, `usage.output_tokens` across all `type: assistant` events.
- Compute cache hit ratio (cache_read / (cache_read + cache_creation)).
- Identify the largest single message by `cache_creation_input_tokens` — that's probably the system-prompt + intermediate-artifact bundle.

Useful for spotting regressions where a prompt rewrite balloons cache misses.

---

## §9. Report format

End the analysis with this structure (markdown):

```markdown
# Trace Analysis — <skill-name>

**Trace file**: <path>
**Size**: <bytes>, **Sessions**: <N>, **Repair rounds**: <N-1>
**Classification**: <archetype> / <shape> / <modality> / <P-variant>
**Final outcome**: <compile-succeeded | compile-failed | mid-flight>

## Trace integrity
- ...

## Grounding usage
**Canonical**: <which file, or "not consulted">
**Surface**: <reads count, sidecar usage, any 256KB errors>

## Schema / validation behavior
| Anti-pattern | Count | Notes |
| --- | --- | --- |
| JSON Pointer refs | N | ... |
| Self-referential bound_to | N | ... |
| Local-helper symbols in pipeline.call | N | ... |
| Function-name prefix in signature | N | ... |
| Invented CallNode fields | N | ... |

## Confusion signals
- Uncertainty markers: <count>
- Re-reads on same file: <count>
- Abandoned attempts: <count>

## Failure / repair
- Error classes hit: <list>
- Repair targeting derived files (task #16 signal): <yes / no>

## Behavior conflicting with expected guidance
- ...

## Token telemetry
- Input: <N>, Cache-creation: <N>, Cache-read: <N>, Output: <N>
- Cache hit ratio: <%>

## Top 3 observations
1. ...
2. ...
3. ...
```

---

## §10. Extension notes (grow this over time)

Things to add when we encounter them:

- **New failure modes**: when a new error class shows up in production, add a check for it in the relevant §3 / §4 / §6 section. Each new check should look for both the *behavior* and the *outcome* (LLM did X → which produced Y).
- **New grounding artifacts**: when we add a new file the LLM should consult (e.g., if docs-on-demand lands per `2026-05-21-docs-on-demand-consultation-plan.md`), add a §2-style "did the LLM use it" check.
- **New anti-patterns**: when we observe the LLM doing something the directive says not to, add a §7 check + reference the directive section that prohibits it.

When adding, keep the structure: *what to look for → what it signals → expected behavior*. Avoid pure narrative ("look for confusion") — be specific enough that two analysts running this would produce the same report.
