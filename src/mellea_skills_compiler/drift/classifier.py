"""Three-class drift classifier for the corpus-recompile workflow.

Implements the heuristic described in Phase 3.5 plan §14.3:

| Class       | Looks like                                                  |
|-------------|-------------------------------------------------------------|
| cosmetic    | whitespace / import-order / transient var renames; identical
|             | AST after normalisation; canonical descriptor hash may differ
|             | but pipeline.py AST is structurally identical.              |
| structural  | different operator picks, different schema field counts,
|             | different ``inputs``/``outputs`` arity, different number of
|             | pipeline nodes.                                              |
| semantic    | prompt-template text changed materially, symbol switched
|             | (e.g. ``instruct`` → ``aact``), output-schema field renames,
|             | reworded instructions.                                       |

The classifier is intentionally heuristic — flag generously, let humans
triage.  §14.3 explicitly says: *"It need not be perfect — flag generously,
let humans triage."*

Inputs are two compiled-package directories (the *previous* and *current*
``<skill>_mellea/`` trees).  Outputs are a :class:`SkillDriftVerdict` with a
top-level :class:`DriftClass` plus per-artefact diff observations.

Heuristic rules encoded (highest class wins):

1. **none** — every artefact byte-identical (or pipeline.py AST identical
   AND schemas/melleafy.json byte-identical).
2. **cosmetic** — pipeline.py AST identical after *normalisation*
   (transient identifier names dropped, docstrings + literals removed
   from comparison key), schemas.py field signatures unchanged,
   melleafy.json hash may differ but symbol+operator+template content
   unchanged.
3. **structural** — pipeline.py operator-symbol bag or call-count delta;
   schemas.py Pydantic class set or field-count delta; ``inputs``/
   ``outputs`` arity delta in descriptor.
4. **semantic** — prompt-template fuzzy-hash delta beyond
   ``SEMANTIC_FUZZY_THRESHOLD``; symbol pick switched (e.g. ``instruct``
   ↔ ``aact``); schema field renames (same count but different name set).

The classifier does NOT need the rendered code to import or run; it operates
purely on text/AST.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


# --- Tunables -------------------------------------------------------------

# Fuzzy threshold for semantic prompt-template drift.  Computed via a
# whitespace-normalised Jaccard token similarity; <= threshold → semantic
# drift.  Tuned generously per §14.3 — over-flagging is the explicit goal.
SEMANTIC_FUZZY_THRESHOLD: float = 0.85

# Transient variable-name patterns the AST normaliser collapses to a sentinel.
# Matches generic numbered identifiers the LLM/wrapper emits during
# scaffolding (e.g. ``step_1``, ``result_3``, ``out_12``) where the prefix
# carries no semantic weight.
_TRANSIENT_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^step_\d+$"),
    re.compile(r"^result_\d+$"),
    re.compile(r"^out_\d+$"),
    re.compile(r"^tmp_\d+$"),
    re.compile(r"^var_\d+$"),
    re.compile(r"^_\d+$"),
)

_TRANSIENT_NAME_SENTINEL = "__TRANSIENT__"

# Mellea call-symbol pairs the classifier treats as a "semantic switch"
# (one of the §14.3 examples: ``instruct → aact``).
_SEMANTIC_SYMBOL_SET = {"instruct", "aact", "agent_loop", "chat", "transform"}


# --- Public types ---------------------------------------------------------


class DriftClass(str, Enum):
    """Three drift classes from §14.3, plus a ``none`` sentinel for unchanged."""

    NONE = "none"
    COSMETIC = "cosmetic"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass
class SkillDriftVerdict:
    """Per-skill drift record produced by :func:`classify_skill_drift`."""

    skill: str
    drift_class: DriftClass
    pipeline_observations: list[str] = field(default_factory=list)
    schemas_observations: list[str] = field(default_factory=list)
    descriptor_observations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "drift_class": self.drift_class.value,
            "pipeline_observations": list(self.pipeline_observations),
            "schemas_observations": list(self.schemas_observations),
            "descriptor_observations": list(self.descriptor_observations),
            "notes": list(self.notes),
        }


# --- Internal stats records ----------------------------------------------


@dataclass
class _PipelineFingerprint:
    """Minimal AST-derived signature of a ``pipeline.py``."""

    found: bool
    normalised_ast_hash: str = ""
    public_signatures: list[str] = field(default_factory=list)
    call_symbol_bag: dict[str, int] = field(default_factory=dict)
    prompt_templates: list[str] = field(default_factory=list)
    for_loops: int = 0
    if_chains: int = 0
    while_loops: int = 0


@dataclass
class _SchemasFingerprint:
    """Pydantic class set + field signatures extracted from ``schemas.py``."""

    found: bool
    classes: dict[str, list[str]] = field(default_factory=dict)
    """Map of class name → sorted list of ``"field_name: annotation"`` strings."""


@dataclass
class _DescriptorFingerprint:
    """Salient fields of ``melleafy.json`` for drift detection."""

    found: bool
    canonical_hash: str = ""
    inputs_arity: int = 0
    outputs_arity: int = 0
    operator_bag: dict[str, int] = field(default_factory=dict)
    symbol_bag: dict[str, int] = field(default_factory=dict)


# --- Pipeline.py extraction ----------------------------------------------


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _normalise_transient_name(name: str) -> str:
    for pat in _TRANSIENT_NAME_PATTERNS:
        if pat.match(name):
            return _TRANSIENT_NAME_SENTINEL
    return name


class _Normaliser(ast.NodeTransformer):
    """AST transform that erases drift sources we treat as cosmetic.

    * Drops module/function/class docstrings.
    * Rewrites transient identifiers (``step_1``, ``result_3``, …) to a sentinel.
    * Erases the value of every string/numeric literal (only the *kind* is kept).
    """

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node = self._strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = _normalise_transient_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = _normalise_transient_name(node.arg)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # Erase literal values; only the type is kept so AST shape stays
        # comparable.  This means changes to prompt-template text DO NOT
        # show up in the normalised AST hash — those changes are caught
        # separately via the prompt_templates list on _PipelineFingerprint.
        node.value = type(node.value).__name__ + "::LITERAL"
        return node

    @staticmethod
    def _strip_docstring(node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if not body:
            return node
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = body[1:]  # type: ignore[attr-defined]
        return node


def _normalised_ast_hash(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Don't let a malformed pipeline crash the diff; hash the raw
        # source so the verdict still differentiates broken-vs-broken from
        # broken-vs-fixed.
        return "SYNTAX_ERROR::" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    normalised = _Normaliser().visit(tree)
    ast.fix_missing_locations(normalised)
    dump = ast.dump(normalised, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _collect_call_symbol(node: ast.Call) -> str | None:
    """Best-effort: identify the ``.attr`` name of a method call.

    Used to build a "call-symbol bag" so the classifier can spot e.g.
    ``m.instruct`` being swapped for ``m.aact`` even if other operator
    machinery around it stays the same.
    """
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _collect_prompt_templates(tree: ast.AST) -> list[str]:
    """Pull every string-literal arg passed positionally to a ``.<attr>(...)`` call.

    This catches ``m.instruct("find security issues")`` and similar — exactly
    the surface §14.3 names as the semantic-drift trigger.  Multi-line
    triple-quoted f-strings without interpolations also fall into ``ast.Constant``
    with ``str`` value, so they're captured too.
    """
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    text = arg.value.strip()
                    if len(text) >= 20:  # heuristic: real templates, not "ok" / etc.
                        out.append(text)
                elif isinstance(arg, ast.JoinedStr):
                    # f-string: capture the literal-only fragments
                    pieces = [
                        v.value
                        for v in arg.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    ]
                    text = "".join(pieces).strip()
                    if len(text) >= 20:
                        out.append(text)
    return out


def _fingerprint_pipeline(pipeline_py: Path) -> _PipelineFingerprint:
    source = _read_text(pipeline_py)
    if source is None:
        return _PipelineFingerprint(found=False)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _PipelineFingerprint(
            found=True,
            normalised_ast_hash=_normalised_ast_hash(source),
        )

    fp = _PipelineFingerprint(
        found=True,
        normalised_ast_hash=_normalised_ast_hash(source),
        prompt_templates=_collect_prompt_templates(tree),
    )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            arg_names = [a.arg for a in node.args.args]
            fp.public_signatures.append(f"{node.name}({', '.join(arg_names)})")

    bag: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            sym = _collect_call_symbol(node)
            if sym is not None:
                bag[sym] = bag.get(sym, 0) + 1
        elif isinstance(node, ast.For):
            fp.for_loops += 1
        elif isinstance(node, ast.If):
            fp.if_chains += 1
        elif isinstance(node, ast.While):
            fp.while_loops += 1
    fp.call_symbol_bag = bag
    return fp


# --- Schemas.py extraction ------------------------------------------------


def _fingerprint_schemas(schemas_py: Path) -> _SchemasFingerprint:
    source = _read_text(schemas_py)
    if source is None:
        return _SchemasFingerprint(found=False)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _SchemasFingerprint(found=True)

    fp = _SchemasFingerprint(found=True)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields: list[str] = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                ann_src = ast.unparse(item.annotation) if item.annotation else "Any"
                fields.append(f"{item.target.id}: {ann_src}")
            elif isinstance(item, ast.Assign):
                for tgt in item.targets:
                    if isinstance(tgt, ast.Name):
                        fields.append(f"{tgt.id}: <unannotated>")
        fp.classes[node.name] = sorted(fields)
    return fp


# --- Descriptor (melleafy.json) extraction -------------------------------


def _canonical_json_hash(obj: object) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_pipeline_for_bags(
    pipeline_node: object,
    operators: dict[str, int],
    symbols: dict[str, int],
) -> None:
    """Recursively count ``operator`` and ``symbol`` occurrences in a
    descriptor's pipeline tree.  Tolerates v0.1/v0.2/v0.3 shapes by walking
    plain dicts/lists without committing to a schema version.
    """
    if isinstance(pipeline_node, dict):
        op = pipeline_node.get("operator")
        if isinstance(op, str):
            operators[op] = operators.get(op, 0) + 1
        sym = pipeline_node.get("symbol")
        if isinstance(sym, str):
            symbols[sym] = symbols.get(sym, 0) + 1
        for v in pipeline_node.values():
            _walk_pipeline_for_bags(v, operators, symbols)
    elif isinstance(pipeline_node, list):
        for item in pipeline_node:
            _walk_pipeline_for_bags(item, operators, symbols)


def _fingerprint_descriptor(melleafy_json: Path) -> _DescriptorFingerprint:
    source = _read_text(melleafy_json)
    if source is None:
        return _DescriptorFingerprint(found=False)
    try:
        data = json.loads(source)
    except json.JSONDecodeError:
        return _DescriptorFingerprint(
            found=True,
            canonical_hash="MALFORMED::"
            + hashlib.sha256(source.encode("utf-8")).hexdigest(),
        )

    operators: dict[str, int] = {}
    symbols: dict[str, int] = {}
    if isinstance(data, dict):
        _walk_pipeline_for_bags(data.get("pipeline"), operators, symbols)

    inputs = data.get("inputs") if isinstance(data, dict) else None
    outputs = data.get("outputs") if isinstance(data, dict) else None
    return _DescriptorFingerprint(
        found=True,
        canonical_hash=_canonical_json_hash(data),
        inputs_arity=len(inputs) if isinstance(inputs, list) else 0,
        outputs_arity=len(outputs) if isinstance(outputs, list) else 0,
        operator_bag=operators,
        symbol_bag=symbols,
    )


# --- Comparison heuristics -----------------------------------------------


def _tokenise(text: str) -> set[str]:
    # Coarse whitespace tokenisation with lowercasing; good enough for a
    # fuzzy "did the template change a lot" signal.
    return {t for t in re.split(r"\W+", text.lower()) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _semantic_template_drift(
    prev: Iterable[str], curr: Iterable[str]
) -> tuple[bool, float, str]:
    """Return ``(drifted, score, summary)`` for prompt-template fuzzy drift.

    Aggregates all prompt-template tokens per side; computes Jaccard.  A
    score at or below :data:`SEMANTIC_FUZZY_THRESHOLD` means semantic drift.
    Tuned generously: any non-trivial rewrite tends to drop tokens below
    0.85.  The intent is to over-flag.
    """
    prev_tokens: set[str] = set()
    curr_tokens: set[str] = set()
    for t in prev:
        prev_tokens |= _tokenise(t)
    for t in curr:
        curr_tokens |= _tokenise(t)

    if not prev_tokens and not curr_tokens:
        return False, 1.0, "no prompt templates on either side"
    if not prev_tokens or not curr_tokens:
        return (
            True,
            0.0,
            "prompt templates present on only one side "
            f"(prev={len(prev_tokens)} tokens, curr={len(curr_tokens)} tokens)",
        )

    score = _jaccard(prev_tokens, curr_tokens)
    drifted = score <= SEMANTIC_FUZZY_THRESHOLD
    summary = (
        f"prompt-template fuzzy similarity = {score:.3f} "
        f"(threshold {SEMANTIC_FUZZY_THRESHOLD})"
    )
    return drifted, score, summary


def _semantic_symbol_drift(
    prev_bag: dict[str, int], curr_bag: dict[str, int]
) -> tuple[bool, str]:
    """Detect ``instruct → aact``-class swaps in the call-symbol bag."""
    prev_keys = set(prev_bag) & _SEMANTIC_SYMBOL_SET
    curr_keys = set(curr_bag) & _SEMANTIC_SYMBOL_SET
    added = curr_keys - prev_keys
    removed = prev_keys - curr_keys
    if added or removed:
        return True, (
            f"semantic-symbol shift: removed={sorted(removed) or '[]'} "
            f"added={sorted(added) or '[]'}"
        )
    return False, ""


def _structural_pipeline_drift(
    prev: _PipelineFingerprint, curr: _PipelineFingerprint
) -> list[str]:
    out: list[str] = []
    if prev.public_signatures != curr.public_signatures:
        out.append(
            f"public signatures differ: prev={prev.public_signatures or ['<none>']} "
            f"curr={curr.public_signatures or ['<none>']}"
        )
    # Operator-call-count delta on *non-semantic* symbols, i.e. anything
    # that ISN'T a Mellea verb.  (Verb swaps are handled separately as
    # semantic.)
    prev_struct = {
        k: v for k, v in prev.call_symbol_bag.items() if k not in _SEMANTIC_SYMBOL_SET
    }
    curr_struct = {
        k: v for k, v in curr.call_symbol_bag.items() if k not in _SEMANTIC_SYMBOL_SET
    }
    if prev_struct != curr_struct:
        out.append(
            f"non-verb call-symbol bag differs: prev={prev_struct} curr={curr_struct}"
        )
    # Verb call-count change WITHOUT an added/removed verb is structural
    # (e.g. two instruct calls collapsed into one).  Verb swap → semantic.
    prev_verbs = {
        k: v for k, v in prev.call_symbol_bag.items() if k in _SEMANTIC_SYMBOL_SET
    }
    curr_verbs = {
        k: v for k, v in curr.call_symbol_bag.items() if k in _SEMANTIC_SYMBOL_SET
    }
    if set(prev_verbs) == set(curr_verbs) and prev_verbs != curr_verbs:
        out.append(
            f"verb-call counts differ (same verbs, different counts): "
            f"prev={prev_verbs} curr={curr_verbs}"
        )
    if (
        prev.for_loops != curr.for_loops
        or prev.if_chains != curr.if_chains
        or prev.while_loops != curr.while_loops
    ):
        out.append(
            f"control-flow shape differs: for prev={prev.for_loops} curr={curr.for_loops}, "
            f"if prev={prev.if_chains} curr={curr.if_chains}, "
            f"while prev={prev.while_loops} curr={curr.while_loops}"
        )
    return out


def _structural_schemas_drift(
    prev: _SchemasFingerprint, curr: _SchemasFingerprint
) -> tuple[list[str], list[str]]:
    """Return ``(structural_obs, semantic_obs)``.

    Field count delta → structural.  Same count but renamed fields → semantic
    (the *meaning* of the schema changed even though arity is preserved).
    """
    structural: list[str] = []
    semantic: list[str] = []
    prev_classes = set(prev.classes)
    curr_classes = set(curr.classes)
    added = curr_classes - prev_classes
    removed = prev_classes - curr_classes
    if added or removed:
        structural.append(
            f"schema class set differs: added={sorted(added) or '[]'} "
            f"removed={sorted(removed) or '[]'}"
        )
    for name in sorted(prev_classes & curr_classes):
        prev_fields = prev.classes[name]
        curr_fields = curr.classes[name]
        if prev_fields == curr_fields:
            continue
        if len(prev_fields) != len(curr_fields):
            structural.append(
                f"schema '{name}' field count {len(prev_fields)} → {len(curr_fields)}"
            )
        else:
            # Same arity; field-name rename → semantic.
            prev_names = sorted(f.split(":", 1)[0] for f in prev_fields)
            curr_names = sorted(f.split(":", 1)[0] for f in curr_fields)
            if prev_names != curr_names:
                semantic.append(
                    f"schema '{name}' field names renamed (same count): "
                    f"prev={prev_names} curr={curr_names}"
                )
            else:
                # Same names, different annotations → structural-ish.
                structural.append(
                    f"schema '{name}' field annotations changed: "
                    f"prev={prev_fields} curr={curr_fields}"
                )
    return structural, semantic


def _structural_descriptor_drift(
    prev: _DescriptorFingerprint, curr: _DescriptorFingerprint
) -> list[str]:
    out: list[str] = []
    if prev.inputs_arity != curr.inputs_arity:
        out.append(
            f"descriptor inputs arity {prev.inputs_arity} → {curr.inputs_arity}"
        )
    if prev.outputs_arity != curr.outputs_arity:
        out.append(
            f"descriptor outputs arity {prev.outputs_arity} → {curr.outputs_arity}"
        )
    if prev.operator_bag != curr.operator_bag:
        out.append(
            f"descriptor operator bag differs: prev={prev.operator_bag} "
            f"curr={curr.operator_bag}"
        )
    return out


# --- Top-level classification --------------------------------------------


def classify_skill_drift(
    skill_name: str,
    previous_dir: Path,
    current_dir: Path,
) -> SkillDriftVerdict:
    """Compare two compiled-package directories; return a drift verdict.

    ``previous_dir`` and ``current_dir`` are the two ``<skill>_mellea/``
    trees (NOT the skill-root directories).  The classifier looks at three
    artefacts in each: ``pipeline.py``, ``schemas.py``, ``melleafy.json``.

    If both directories are missing entirely, returns
    :attr:`DriftClass.NONE` with a note.
    """
    verdict = SkillDriftVerdict(skill=skill_name, drift_class=DriftClass.NONE)

    prev_exists = previous_dir.is_dir()
    curr_exists = current_dir.is_dir()

    if not prev_exists and not curr_exists:
        verdict.notes.append("neither previous nor current package present")
        return verdict
    if not prev_exists:
        verdict.drift_class = DriftClass.STRUCTURAL
        verdict.notes.append(
            f"previous package missing ({previous_dir}); first-time compile is "
            "treated as structural drift for review purposes"
        )
        return verdict
    if not curr_exists:
        verdict.drift_class = DriftClass.STRUCTURAL
        verdict.notes.append(
            f"current package missing ({current_dir}); compilation may have "
            "failed — investigate"
        )
        return verdict

    prev_pipe = _fingerprint_pipeline(previous_dir / "pipeline.py")
    curr_pipe = _fingerprint_pipeline(current_dir / "pipeline.py")
    prev_schemas = _fingerprint_schemas(previous_dir / "schemas.py")
    curr_schemas = _fingerprint_schemas(current_dir / "schemas.py")
    prev_desc = _fingerprint_descriptor(previous_dir / "melleafy.json")
    curr_desc = _fingerprint_descriptor(current_dir / "melleafy.json")

    # --- Semantic checks (highest class) ---
    semantic_template_drifted, template_score, template_summary = (
        _semantic_template_drift(prev_pipe.prompt_templates, curr_pipe.prompt_templates)
    )
    if semantic_template_drifted:
        verdict.pipeline_observations.append("[semantic] " + template_summary)
    else:
        verdict.pipeline_observations.append("template similarity " + template_summary)

    sym_drifted, sym_summary = _semantic_symbol_drift(
        prev_pipe.call_symbol_bag, curr_pipe.call_symbol_bag
    )
    if sym_drifted:
        verdict.pipeline_observations.append("[semantic] " + sym_summary)

    schemas_structural, schemas_semantic = _structural_schemas_drift(
        prev_schemas, curr_schemas
    )
    for line in schemas_semantic:
        verdict.schemas_observations.append("[semantic] " + line)

    # --- Structural checks ---
    pipeline_structural = _structural_pipeline_drift(prev_pipe, curr_pipe)
    for line in pipeline_structural:
        verdict.pipeline_observations.append("[structural] " + line)
    for line in schemas_structural:
        verdict.schemas_observations.append("[structural] " + line)
    descriptor_structural = _structural_descriptor_drift(prev_desc, curr_desc)
    for line in descriptor_structural:
        verdict.descriptor_observations.append("[structural] " + line)

    # --- Cosmetic vs none ---
    normalised_ast_unchanged = (
        prev_pipe.normalised_ast_hash == curr_pipe.normalised_ast_hash
    )
    descriptor_hash_unchanged = prev_desc.canonical_hash == curr_desc.canonical_hash
    schemas_classes_unchanged = prev_schemas.classes == curr_schemas.classes

    if descriptor_hash_unchanged:
        verdict.descriptor_observations.append(
            "descriptor canonical hash unchanged"
        )
    else:
        verdict.descriptor_observations.append(
            "descriptor canonical hash changed"
        )

    if normalised_ast_unchanged:
        verdict.pipeline_observations.append(
            "pipeline.py AST identical after normalisation"
        )

    # --- Final classification (highest wins) ---
    if (
        semantic_template_drifted
        or sym_drifted
        or schemas_semantic
    ):
        verdict.drift_class = DriftClass.SEMANTIC
    elif pipeline_structural or schemas_structural or descriptor_structural:
        verdict.drift_class = DriftClass.STRUCTURAL
    elif (
        normalised_ast_unchanged
        and schemas_classes_unchanged
        and descriptor_hash_unchanged
    ):
        verdict.drift_class = DriftClass.NONE
    else:
        # AST identical post-normalisation but raw bytes / descriptor hash
        # differ → cosmetic.  Or schemas identical but pipeline AST hash
        # different by transient-name-after-normalisation noise.
        verdict.drift_class = DriftClass.COSMETIC
        if not normalised_ast_unchanged:
            verdict.notes.append(
                "pipeline.py AST differs even after normalisation; classified "
                "cosmetic because no structural/semantic markers fired — review "
                "if the verdict seems wrong"
            )

    return verdict
