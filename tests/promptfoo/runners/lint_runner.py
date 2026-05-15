#!/usr/bin/env python3
"""Promptfoo exec provider — lint regression runner.

Reads JSON from stdin. The fixture YAML prompt template packages all test
vars as a JSON object via Nunjucks `dump`:

    prompts:
      - '{{ {"lint": lint, "code": code, ...} | dump | safe }}'

Outputs a single string to stdout beginning with PASS or FAIL.
"""

from __future__ import annotations

import ast
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _read_vars() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        # Accept {"vars": {...}} (promptfoo envelope) or the vars dict directly.
        return data.get("vars", data) if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _func_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_generative(decorator_list: list) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "generative")
        or (isinstance(d, ast.Attribute) and d.attr == "generative")
        for d in decorator_list
    )


# ---------------------------------------------------------------------------
# validator-soundness  (KB3 sub-check A, KB4 sub-check B)
# ---------------------------------------------------------------------------

def lint_validator_soundness(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    func_defs: dict[str, ast.FunctionDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    failures: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "validation_fn":
                continue
            val = kw.value

            # Sub-check A: raw lambda not wrapped in simple_validate
            if isinstance(val, ast.Lambda):
                failures.append(
                    f"FAIL validator-soundness[A] line {node.lineno}: "
                    "validation_fn= is a raw lambda; wrap with simple_validate() "
                    "or use a (ctx: Context, result) -> ValidationResult function"
                )
                continue

            # simple_validate(fn) call
            if isinstance(val, ast.Call) and _func_name(val.func) == "simple_validate":
                if val.args and isinstance(val.args[0], ast.Lambda):
                    body = val.args[0].body
                    # Sub-check B: vacuous lambda always returns True
                    if isinstance(body, ast.Constant) and body.value is True:
                        failures.append(
                            f"FAIL validator-soundness[B] line {node.lineno}: "
                            "vacuous validator — lambda always returns True regardless of input"
                        )
                continue

            # Named function reference — verify (ctx, result) -> ValidationResult shape
            if isinstance(val, ast.Name):
                fn_def = func_defs.get(val.id)
                if fn_def is not None:
                    args = fn_def.args.args
                    has_ctx = any(
                        isinstance(a.annotation, ast.Name) and a.annotation.id == "Context"
                        for a in args
                    )
                    if not (len(args) >= 2 and has_ctx):
                        failures.append(
                            f"FAIL validator-soundness[A] line {node.lineno}: "
                            f"validation_fn={val.id} lacks required (ctx: Context, result) signature"
                        )

    return "\n".join(failures) if failures else "PASS validator-soundness"


# ---------------------------------------------------------------------------
# instruct-has-description  (KB12 — Mellea 0.5 instruct() signature)
# ---------------------------------------------------------------------------

def lint_instruct_has_description(code: str) -> str:
    """Every m.instruct(...) call must include a description argument.

    Mellea 0.5.0's MelleaSession.instruct(description, *, ...) requires
    `description` as the first positional-or-keyword argument. Accept either
    a positional first argument OR a `description=` keyword. Anything else is
    a hard failure (would crash at runtime with TypeError).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "instruct"
            and isinstance(func.value, ast.Name)
            and func.value.id == "m"
        ):
            continue
        has_positional = len(node.args) >= 1
        has_description_kwarg = any(kw.arg == "description" for kw in node.keywords)
        if has_positional or has_description_kwarg:
            continue
        failures.append(
            f"FAIL instruct-has-description line {node.lineno}: "
            "m.instruct(...) is missing a description. Mellea 0.5+ requires "
            "`description` as the first positional argument (or as a "
            "`description=` keyword). Calls with only `format=`/"
            "`grounding_context=`/`model_options=` raise TypeError at runtime."
        )

    return "\n".join(failures) if failures else "PASS instruct-has-description"


# ---------------------------------------------------------------------------
# pipeline-entry-canonical  (run_pipeline canonical-entry contract)
# ---------------------------------------------------------------------------

import re as _re

_ENTRY_SIGNATURE_NAME_RE = _re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CANONICAL_ENTRY_NAME = "run_pipeline"


def lint_pipeline_entry_canonical(code: str, entry_signature: str | None) -> str:
    """pipeline.py must define a top-level `run_pipeline` def, and
    `melleafy.json:entry_signature` (passed via the `entry_signature` var)
    must start with `run_pipeline(`.

    Empirically observed regression: nis2-navigator pipeline.py defined
    `run_phase_2_gap_analysis`, `run_phase_3_roadmap`, `run_pipeline`. The
    smoke-check loader picked the alphabetically-first one and crashed at
    runtime when called with entry-point kwargs. The loader was patched to
    use melleafy.json:entry_signature; this lint backstops the contract.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    top_level_run_defs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("run_")
    ]

    failures: list[str] = []

    if _CANONICAL_ENTRY_NAME not in top_level_run_defs:
        failures.append(
            f"FAIL pipeline-entry-canonical: pipeline.py does not define a "
            f"top-level `{_CANONICAL_ENTRY_NAME}` function. "
            f"Found run_* defs: {', '.join(top_level_run_defs) or '(none)'}."
        )

    if entry_signature:
        match = _ENTRY_SIGNATURE_NAME_RE.match(entry_signature)
        manifest_name = match.group(1) if match else None
        if manifest_name and manifest_name != _CANONICAL_ENTRY_NAME:
            failures.append(
                f"FAIL pipeline-entry-canonical: melleafy.json:entry_signature "
                f"names `{manifest_name}`; canonical entry must be "
                f"`{_CANONICAL_ENTRY_NAME}`."
            )

    return "\n".join(failures) if failures else "PASS pipeline-entry-canonical"


# ---------------------------------------------------------------------------
# known-behaviours  (KB6/3f, KB7/3g, KB8/3h, KB11/3m)
# ---------------------------------------------------------------------------

_FORBIDDEN_PARAMS = frozenset({
    "f_args", "f_kwargs", "m", "context", "backend",
    "model_options", "strategy", "precondition_requirements", "requirements",
})

_EXTRACTION_KEYWORDS = ("extract", "do not ask", "if the")


def lint_known_behaviours(subcheck: str | None, code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    match subcheck:
        case "3f":
            return _kb6_forbidden_params(tree)
        case "3g":
            return _kb7_prefix_persona(tree)
        case "3h":
            return _kb8_watsonx(code)
        case "3m":
            return _kb11_optional_extraction(tree)
        case _:
            return f"WARN known-behaviours: unhandled subcheck '{subcheck}'"


def _kb6_forbidden_params(tree: ast.AST) -> str:
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _is_generative(node.decorator_list):
            continue
        for arg in node.args.args:
            if arg.arg in _FORBIDDEN_PARAMS:
                failures.append(
                    f"FAIL known-behaviours[3f] line {node.lineno}: "
                    f"@generative parameter '{arg.arg}' is a forbidden name"
                )
    return "\n".join(failures) if failures else "PASS known-behaviours[3f]"


def _kb7_prefix_persona(tree: ast.AST) -> str:
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "instruct"):
            continue
        for kw in node.keywords:
            if kw.arg != "prefix":
                continue
            if isinstance(kw.value, ast.Name):
                failures.append(
                    f"FAIL known-behaviours[3g] line {node.lineno}: "
                    f"prefix={kw.value.id} appears to be a config constant used as persona text; "
                    "use model_options={ModelOption.SYSTEM_PROMPT: <constant>} instead"
                )
    return "\n".join(failures) if failures else "PASS known-behaviours[3g]"


def _kb8_watsonx(code: str) -> str:
    if "watsonx" in code:
        return 'FAIL known-behaviours[3h]: "watsonx" backend string found in generated code'
    return "PASS known-behaviours[3h]"


def _kb11_optional_extraction(tree: ast.AST) -> str:
    failures = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        is_basemodel = any(
            (isinstance(b, ast.Name) and b.id == "BaseModel")
            or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
            for b in cls.bases
        )
        if not is_basemodel:
            continue

        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            ann = stmt.annotation
            is_optional = isinstance(ann, ast.Subscript) and (
                (isinstance(ann.value, ast.Name) and ann.value.id == "Optional")
                or (isinstance(ann.value, ast.Attribute) and ann.value.attr == "Optional")
            )
            if not is_optional:
                continue

            field_name = ann.value.id if isinstance(stmt.target, ast.Name) else "?"
            if isinstance(stmt.target, ast.Name):
                field_name = stmt.target.id

            val = stmt.value
            if val is None or not isinstance(val, ast.Call):
                failures.append(
                    f"FAIL known-behaviours[3m] line {stmt.lineno}: "
                    f"Optional field '{field_name}' in {cls.name} lacks Field(description=...) "
                    "with extraction guidance"
                )
                continue

            if _func_name(val.func) != "Field":
                failures.append(
                    f"FAIL known-behaviours[3m] line {stmt.lineno}: "
                    f"Optional field '{field_name}' in {cls.name} lacks Field(description=...)"
                )
                continue

            description = next(
                (kw.value.value for kw in val.keywords
                 if kw.arg == "description" and isinstance(kw.value, ast.Constant)),
                None,
            )
            if description is None:
                failures.append(
                    f"FAIL known-behaviours[3m] line {stmt.lineno}: "
                    f"Optional field '{field_name}' in {cls.name}: Field() missing description"
                )
                continue

            if not any(kw in description.lower() for kw in _EXTRACTION_KEYWORDS):
                failures.append(
                    f"FAIL known-behaviours[3m] line {stmt.lineno}: "
                    f"Optional field '{field_name}' in {cls.name}: description lacks extraction "
                    "guidance ('extract', 'do not ask', 'if the')"
                )

    return "\n".join(failures) if failures else "PASS known-behaviours[3m]"


# ---------------------------------------------------------------------------
# session-boundary  (KB5)
# ---------------------------------------------------------------------------

def lint_session_boundary(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        is_session = any(
            isinstance(item.context_expr, ast.Call)
            and _func_name(item.context_expr.func) == "start_session"
            for item in node.items
        )
        if not is_session:
            continue

        format_types: set[str] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if not (isinstance(child.func, ast.Attribute) and child.func.attr == "instruct"):
                continue
            for kw in child.keywords:
                if kw.arg == "format":
                    t = _func_name(kw.value)
                    if t:
                        format_types.add(t)

        if len(format_types) > 1:
            failures.append(
                f"FAIL session-boundary line {node.lineno}: "
                f"start_session() block uses {len(format_types)} distinct format types: "
                f"{', '.join(sorted(format_types))}. "
                "Split into separate sessions (Known Behaviour 5)."
            )

    return "\n".join(failures) if failures else "PASS session-boundary"


# ---------------------------------------------------------------------------
# cross-reference sub-check F  (Rule 3-1)
# ---------------------------------------------------------------------------

def lint_cross_reference(subcheck: str | None, code: str) -> str:
    if subcheck != "F":
        return f"WARN cross-reference: unhandled subcheck '{subcheck}'"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "run_pipeline":
            continue
        bare = [arg.arg for arg in node.args.args if arg.annotation is None]
        if bare:
            return (
                f"FAIL cross-reference[F] line {node.lineno}: "
                f"run_pipeline parameters missing type annotation: {', '.join(bare)}"
            )
        return "PASS cross-reference[F]"

    return "WARN cross-reference[F]: run_pipeline not found in provided code"


# ---------------------------------------------------------------------------
# import-soundness  (Rule 4-2)
# ---------------------------------------------------------------------------

def lint_import_soundness(code: str, mellea_api_ref: str | None) -> str:
    if not mellea_api_ref:
        return "WARN import-soundness: mellea_api_ref not provided — skipping"

    try:
        api_ref = json.loads(mellea_api_ref)
    except json.JSONDecodeError as exc:
        return f"WARN import-soundness: mellea_api_ref is not valid JSON — {exc}"

    if api_ref.get("grounding_unavailable", False):
        return "WARN import-soundness: module index unavailable — import-soundness check skipped"

    known_modules = set(api_ref.get("modules", {}).keys())

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("mellea") and module not in known_modules:
                failures.append(
                    f"FAIL import-soundness line {node.lineno}: "
                    f"'{module}' not found in mellea_api_ref.json:.modules"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mellea") and alias.name not in known_modules:
                    failures.append(
                        f"FAIL import-soundness line {node.lineno}: "
                        f"'{alias.name}' not found in mellea_api_ref.json:.modules"
                    )

    return "\n".join(failures) if failures else "PASS import-soundness"


# ---------------------------------------------------------------------------
# stdlib-arity  (Rule 4-4)
# ---------------------------------------------------------------------------

_STDLIB_SIGS: dict[str, dict[str, Any]] = {
    "simple_validate": {"min_pos": 1, "max_pos": 1, "valid_kwargs": frozenset()},
    "req":             {"min_pos": 1, "max_pos": 1, "valid_kwargs": frozenset({"validation_fn"})},
    "check":           {"min_pos": 2, "max_pos": 2, "valid_kwargs": frozenset()},
}


def _live_sig(name: str, mellea_api_ref: str | None) -> dict[str, Any] | None:
    """Look up a stdlib function's signature from mellea_api_ref.json for unknown functions.

    Returns a _STDLIB_SIGS-compatible dict, or None if lookup fails.
    Signature format in api_ref: .modules["mellea.X"]["fn"]["signature"] = "fn(a, b, *, kw=None)"
    """
    if not mellea_api_ref:
        return None
    try:
        api_ref = json.loads(mellea_api_ref)
    except json.JSONDecodeError:
        return None
    if api_ref.get("grounding_unavailable", False):
        return None
    for module_symbols in api_ref.get("modules", {}).values():
        entry = module_symbols.get(name)
        if not entry:
            continue
        sig_str = entry.get("signature", "")
        # Parse "fn(a, b, /, c, *, kw=None)" into positional count and kwarg names.
        try:
            import re
            inner = re.search(r"\(([^)]*)\)", sig_str)
            if not inner:
                return None
            params = [p.strip() for p in inner.group(1).split(",") if p.strip()]
            valid_kwargs: set[str] = set()
            min_pos = max_pos = 0
            after_star = False
            for p in params:
                if p in ("*", "/"):
                    after_star = True
                    continue
                p_name = p.split(":")[0].split("=")[0].strip().lstrip("*")
                if after_star or p.startswith("*"):
                    after_star = True
                    if "=" in p or not p.startswith("**"):
                        valid_kwargs.add(p_name)
                else:
                    max_pos += 1
                    if "=" not in p:
                        min_pos += 1
            return {"min_pos": min_pos, "max_pos": max_pos, "valid_kwargs": frozenset(valid_kwargs)}
        except Exception:
            return None
    return None


def lint_stdlib_arity(code: str, mellea_api_ref: str | None = None) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"FAIL parseable: {exc}"

    warnings: list[str] = []
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _func_name(node.func)
        if name is None:
            continue
        # Only inspect mellea.stdlib.* calls (Name nodes after an import, or Attribute nodes)
        if name not in _STDLIB_SIGS:
            # Try live lookup for unknown stdlib functions
            live = _live_sig(name, mellea_api_ref)
            if live is None:
                # Can't determine — warn only if it looks like a stdlib call
                continue
            sig = live
        else:
            sig = _STDLIB_SIGS[name]

        n_pos = len(node.args)
        if not (sig["min_pos"] <= n_pos <= sig["max_pos"]):
            failures.append(
                f"FAIL stdlib-arity line {node.lineno}: "
                f"{name}() called with {n_pos} positional arg(s) "
                f"(expected {sig['min_pos']}–{sig['max_pos']})"
            )
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in sig["valid_kwargs"]:
                failures.append(
                    f"FAIL stdlib-arity line {node.lineno}: "
                    f"{name}() has unrecognised keyword argument '{kw.arg}'"
                )

    if failures:
        return "\n".join(failures)
    if warnings:
        return "\n".join(warnings)
    return "PASS stdlib-arity"


# ---------------------------------------------------------------------------
# melleafy-json-consistency sub-check A  (Rule 5-1)
# ---------------------------------------------------------------------------

# Hard-required by the export command (stage1_validate / stage2_load in exporter.py).
# Missing any of these → export halts; lint FAIL.
_MELLEAFY_HARD_REQUIRED = frozenset({
    "manifest_version",   # stage1_validate: checked >= 1.1.0
    "entry_signature",    # stage2_load: parsed for graph wiring
    "package_name",       # stage2_load: locates the Python package directory
})

# Completeness fields — expected for a well-formed manifest but not read by the
# export command.  Missing → lint WARN only.
_MELLEAFY_SOFT_REQUIRED = frozenset({
    "source_runtime", "modality", "categories_resolved",
    "declared_env_vars", "pipeline_parameters",
})

_MIN_MANIFEST_VERSION = (1, 1, 0)


def lint_melleafy_json_consistency(subcheck: str | None, code: str) -> str:
    if subcheck != "A":
        return f"WARN melleafy-json-consistency: unhandled subcheck '{subcheck}'"
    try:
        data = json.loads(code)
    except json.JSONDecodeError as exc:
        return f"FAIL melleafy-json-consistency[A]: invalid JSON — {exc}"

    keys = set(data.keys())

    # --- Hard failures ---
    missing_hard = _MELLEAFY_HARD_REQUIRED - keys
    if missing_hard:
        return (
            f"FAIL melleafy-json-consistency[A]: "
            f"missing required field(s): {', '.join(sorted(missing_hard))}"
        )

    mv = data.get("manifest_version", "")
    try:
        mv_tuple = tuple(int(x) for x in str(mv).split(".")[:3])
        if mv_tuple < _MIN_MANIFEST_VERSION:
            req = ".".join(str(x) for x in _MIN_MANIFEST_VERSION)
            return f"FAIL melleafy-json-consistency[A]: manifest_version {mv} < required {req}"
    except (ValueError, TypeError):
        return f"FAIL melleafy-json-consistency[A]: manifest_version '{mv}' is not a valid semver string"

    # --- Soft warnings ---
    missing_soft = _MELLEAFY_SOFT_REQUIRED - keys
    if missing_soft:
        return (
            f"WARN melleafy-json-consistency[A]: "
            f"completeness field(s) absent: {', '.join(sorted(missing_soft))}"
        )

    return "PASS melleafy-json-consistency[A]"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def main() -> None:
    vars_ = _read_vars()
    lint: str = vars_.get("lint", "")
    subcheck: str | None = vars_.get("subcheck") or None
    code: str = vars_.get("code", "")
    mellea_api_ref: str | None = vars_.get("mellea_api_ref")
    entry_signature: str | None = vars_.get("entry_signature") or None

    match lint:
        case "validator-soundness":
            print(lint_validator_soundness(code))
        case "known-behaviours":
            print(lint_known_behaviours(subcheck, code))
        case "instruct-has-description":
            print(lint_instruct_has_description(code))
        case "pipeline-entry-canonical":
            print(lint_pipeline_entry_canonical(code, entry_signature))
        case "session-boundary":
            print(lint_session_boundary(code))
        case "cross-reference":
            print(lint_cross_reference(subcheck, code))
        case "import-soundness":
            print(lint_import_soundness(code, mellea_api_ref))
        case "stdlib-arity":
            print(lint_stdlib_arity(code, mellea_api_ref))
        case "melleafy-json-consistency":
            print(lint_melleafy_json_consistency(subcheck, code))
        case "":
            print("ERROR: no 'lint' key in input JSON")
        case _:
            print(f"WARN unknown lint: '{lint}'")


if __name__ == "__main__":
    main()
