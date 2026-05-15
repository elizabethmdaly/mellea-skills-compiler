"""Step 7 structural lints — Python implementations.

Implements seven lints from `.claude/commands/mellea-fy-validate.md`:

  - `bundled-asset-path-resolution` (Rule OUT-6, Rule 2.5-2)
  - `fixtures-loader-contract` (R16, Rule 4-1)
  - `runtime-defaults-bound` (C8 invariant)
  - `instruct-has-description` (KB12 — Mellea 0.5+ requires a description)
  - `pipeline-entry-canonical` (run_pipeline is the canonical entry)
  - `fixture-signature-bound` (fixture inputs keys ⊆ run_pipeline params)
  - `generative-forbidden-params` (KB6 — @generative param names)

The other lints documented in `mellea-fy-validate.md` are still applied
by the LLM during Step 7 of the slash command. These are implemented in
Python because they are AST-precise structural checks where LLM application
has proven unreliable, and they catch the exact drift modes that have caused
post-compile breakage in shipped skills (see session history).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple


_BUNDLED_DIRS = ("scripts", "references", "assets")
_ENTRY_SIGNATURE_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CANONICAL_ENTRY_NAME = "run_pipeline"


@dataclass
class LintFailure:
    file: str
    line: Optional[int]
    column: Optional[int]
    message: str
    rule_ref: Optional[str] = None


@dataclass
class LintResult:
    lint_id: str
    verdict: str  # "pass" | "fail" | "skipped"
    files_checked: int = 0
    skipped_reason: Optional[str] = None
    failures: List[LintFailure] = field(default_factory=list)


@dataclass
class LintRunResult:
    overall_verdict: str  # "pass" | "fail"
    lints: List[LintResult]
    package_path: str
    checked_at: str

    @property
    def failed(self) -> bool:
        return self.overall_verdict == "fail"


# ─── Lint: fixtures-loader-contract ───


def lint_fixtures_loader_contract(package_dir: Path) -> LintResult:
    """`<package>/fixtures/__init__.py` must export ALL_FIXTURES or FIXTURES.

    NOTE — `skipped` semantics:
    `fixtures/__init__.py` is a *mandatory* output of the deterministic
    `fixtures_writer.py` (Step 4). Its absence means the writer never ran or
    its output was discarded — historically this was masked because (a) the
    writer-renderer path resolution silently failed for out-of-tree compiles
    (see `mellea_skills.py`) and (b) this lint reported `skipped`, which the
    Step-7 aggregator counted as `pass`. The fix lives in
    `compile/mellea_skills.py` (hard-fail when writers cannot be located);
    this lint backstops it by reporting `fail` rather than `skipped` for the
    missing-file case. `skipped` is reserved for lints that genuinely do not
    apply to a given package shape (e.g. a tool-dispatch lint on a
    single_shot skill).
    """
    result = LintResult(lint_id="fixtures-loader-contract", verdict="pass")

    fixtures_init = package_dir / "fixtures" / "__init__.py"
    if not fixtures_init.exists():
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="fixtures/__init__.py",
                line=None,
                column=None,
                message=(
                    "fixtures/__init__.py not found in compiled package — this "
                    "file is mandatory and is normally produced by the "
                    "deterministic writer `.claude/melleafy/writers/fixtures_writer.py`. "
                    "Its absence usually means the writer-renderer never ran "
                    "(see compile/writer_renderer.py + the writers_repo_root "
                    "resolution in compile/mellea_skills.py). Was the spec "
                    "compiled out-of-tree with an older compiler that walked "
                    "up from the package dir instead of the installed "
                    "compiler package?"
                ),
                rule_ref="R16, Rule 4-1 (mellea-fy-fixtures.md)",
            )
        )
        return result

    result.files_checked = 1
    try:
        tree = ast.parse(fixtures_init.read_text(), filename=str(fixtures_init))
    except SyntaxError as exc:
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file=str(fixtures_init.relative_to(package_dir)),
                line=exc.lineno,
                column=exc.offset,
                message=f"SyntaxError prevents parsing fixtures/__init__.py: {exc.msg}",
                rule_ref="Rule 4-1 (mellea-fy-fixtures.md)",
            )
        )
        return result

    for node in tree.body:
        targets: List[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ("ALL_FIXTURES", "FIXTURES"):
                return result  # pass

    result.verdict = "fail"
    result.failures.append(
        LintFailure(
            file=str(fixtures_init.relative_to(package_dir)),
            line=None,
            column=None,
            message=(
                "fixtures/__init__.py does not export ALL_FIXTURES (preferred: "
                "list[Callable] of factories returning (inputs, fixture_id, description)) "
                "or FIXTURES (alternative: list[dict] with keys 'id' and 'context'). "
                "See mellea-fy-fixtures.md for the contract; under the fixtures_writer.py "
                "architecture (Step 4) this should be unreachable, so a hand-edit or a "
                "writer bypass is the likely cause."
            ),
            rule_ref="R16, Rule 4-1 (mellea-fy-fixtures.md)",
        )
    )
    return result


# ─── Lint: bundled-asset-path-resolution ───


def _is_path_dunder_file_parent(node: ast.AST) -> bool:
    """True iff node is `Path(__file__).parent`."""
    if not isinstance(node, ast.Attribute) or node.attr != "parent":
        return False
    inner = node.value
    if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Name):
        return False
    if inner.func.id != "Path" or len(inner.args) != 1:
        return False
    arg = inner.args[0]
    return isinstance(arg, ast.Name) and arg.id == "__file__"


def _is_file_rooted(node: ast.AST) -> bool:
    """Accept any expression directly rooted at __file__."""
    if isinstance(node, ast.Name) and node.id == "__file__":
        return True
    if _is_path_dunder_file_parent(node):
        return True
    if isinstance(node, ast.Call):
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == "__file__":
                return True
    return False


def _collect_div_chain(node: ast.AST) -> Tuple[ast.AST, List[ast.AST]]:
    """Unwind a chained BinOp(Div) into (leftmost, [right operands left-to-right])."""
    rights: List[ast.AST] = []
    current = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
        rights.insert(0, current.right)
        current = current.left
    return current, rights


def _starts_with_bundled_dir(s: str) -> Optional[str]:
    """If string is or starts with a bundled directory name, return that name; else None."""
    for d in _BUNDLED_DIRS:
        if s == d or s.startswith(f"{d}/") or s.startswith(f"{d}\\"):
            return d
    return None


def _node_repr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return f"<{type(node).__name__}>"


def _is_skipped_path(rel_path: str) -> bool:
    """fixtures/ are tests, intermediate/ is metadata — neither is runtime code."""
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] in ("fixtures", "intermediate") or "__pycache__" in parts


def lint_bundled_asset_path_resolution(package_dir: Path) -> LintResult:
    """Reject any path-join construction that resolves a bundled-asset path
    via anything other than `Path(__file__).parent`."""
    result = LintResult(lint_id="bundled-asset-path-resolution", verdict="pass")

    py_files: List[Path] = []
    for p in sorted(package_dir.rglob("*.py")):
        rel = p.relative_to(package_dir).as_posix()
        if not _is_skipped_path(rel):
            py_files.append(p)
    result.files_checked = len(py_files)

    seen_failures: set[Tuple[str, Optional[int], Optional[int]]] = set()

    def _record(file_rel: str, node: ast.AST, msg: str) -> None:
        key = (file_rel, getattr(node, "lineno", None), getattr(node, "col_offset", None))
        if key in seen_failures:
            return
        seen_failures.add(key)
        result.failures.append(
            LintFailure(
                file=file_rel,
                line=key[1],
                column=key[2],
                message=msg,
                rule_ref="Rule OUT-6, Rule 2.5-2",
            )
        )

    for py_file in py_files:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue  # the parseable lint is responsible

        for node in ast.walk(tree):
            # Pattern 1: BinOp(Div) chain — Path(...) / "scripts/..."
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                leftmost, rights = _collect_div_chain(node)
                bundled_hit: Optional[str] = None
                first_str: Optional[ast.Constant] = None
                for r in rights:
                    if isinstance(r, ast.Constant) and isinstance(r.value, str):
                        d = _starts_with_bundled_dir(r.value)
                        if d:
                            bundled_hit = d
                            first_str = r
                            break
                if bundled_hit is None or _is_path_dunder_file_parent(leftmost):
                    continue
                _record(
                    rel,
                    node,
                    (
                        f"Bundled asset path '{first_str.value}' is resolved via "
                        f"'{_node_repr(leftmost)}'. Bundled assets at "
                        f"<package_name>/{bundled_hit}/ MUST be resolved via "
                        f"`Path(__file__).parent / \"{bundled_hit}/<...>\"` "
                        f"(Rule OUT-6 in mellea-fy.md, Rule 2.5-2 in mellea-fy-deps.md). "
                        f"Common error: `Path(repo_root) / \"{bundled_hit}/...\"` — "
                        f"must be `Path(__file__).parent / \"{bundled_hit}\" / ...`."
                    ),
                )

            # Pattern 2: os.path.join(base, "scripts", ...)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                fn = node.func
                if (
                    isinstance(fn.value, ast.Attribute)
                    and isinstance(fn.value.value, ast.Name)
                    and fn.value.value.id == "os"
                    and fn.value.attr == "path"
                    and fn.attr == "join"
                    and node.args
                ):
                    bundled_hit = None
                    first_str = None
                    for arg in node.args[1:]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            d = _starts_with_bundled_dir(arg.value)
                            if d:
                                bundled_hit = d
                                first_str = arg
                                break
                    if bundled_hit is None or _is_file_rooted(node.args[0]):
                        continue
                    _record(
                        rel,
                        node,
                        (
                            f"Bundled asset path '{first_str.value}' (via os.path.join) "
                            f"is resolved via '{_node_repr(node.args[0])}'. "
                            f"Bundled assets MUST be resolved relative to `__file__`. "
                            f"Prefer `Path(__file__).parent / \"{bundled_hit}\" / ...`."
                        ),
                    )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: runtime-defaults-bound ───


def lint_runtime_defaults_bound(package_dir: Path) -> LintResult:
    """The compiled config.py must use the backend and model_id we instructed.

    Compares the BACKEND and MODEL_ID constants in <package>/config.py against
    the values recorded at <package>/intermediate/runtime_directive.json (which
    captures what the compile pipeline injected via the system prompt). If the
    LLM ignored the directive and emitted different values, this fails loudly.

    Skipped when the directive file is absent — usually because the package was
    compiled with an older pipeline that did not write the directive.

    NOTE — `skipped` vs `fail` semantics:
    `config.py` is a *mandatory* output of the deterministic `config_writer.py`
    (Step 3, ENFORCE mode). Its absence means the writer never ran or its
    output was discarded — historically masked because (a) the writer-renderer
    path resolution silently failed for out-of-tree compiles (see
    `mellea_skills.py`) and (b) this lint reported `skipped` for the missing
    file, which the Step-7 aggregator counted as `pass`. The primary fix is in
    `compile/mellea_skills.py` (hard-fail when writers cannot be located);
    this lint backstops it by reporting `fail` rather than `skipped` for the
    missing-config.py case. The `runtime_directive.json`-missing branch
    legitimately stays `skipped` because older packages may exist that were
    compiled before the directive was emitted at all.
    """
    result = LintResult(lint_id="runtime-defaults-bound", verdict="pass")

    directive_path = package_dir / "intermediate" / "runtime_directive.json"
    config_path = package_dir / "config.py"

    if not directive_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = (
            "intermediate/runtime_directive.json not found "
            "(probably compiled with an older pipeline version)"
        )
        return result
    if not config_path.exists():
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="config.py",
                line=None,
                column=None,
                message=(
                    "config.py not found in compiled package — this file is "
                    "mandatory and is normally produced by the deterministic "
                    "writer `.claude/melleafy/writers/config_writer.py` "
                    "(Step 3, ENFORCE mode). Its absence usually means the "
                    "writer-renderer never ran (see compile/writer_renderer.py "
                    "+ the writers_repo_root resolution in "
                    "compile/mellea_skills.py). Was the spec compiled "
                    "out-of-tree with an older compiler that walked up from "
                    "the package dir instead of the installed compiler "
                    "package?"
                ),
                rule_ref="C8 runtime defaults",
            )
        )
        return result

    try:
        directive = json.loads(directive_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result.verdict = "skipped"
        result.skipped_reason = f"could not read runtime_directive.json: {exc}"
        return result

    expected_backend = directive.get("backend")
    expected_model_id = directive.get("model_id")
    if expected_backend is None or expected_model_id is None:
        result.verdict = "skipped"
        result.skipped_reason = (
            "runtime_directive.json missing 'backend' or 'model_id' field"
        )
        return result

    try:
        tree = ast.parse(config_path.read_text(), filename=str(config_path))
    except SyntaxError as exc:
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="config.py",
                line=exc.lineno,
                column=exc.offset,
                message=f"SyntaxError prevents parsing config.py: {exc.msg}",
                rule_ref="C8 runtime defaults",
            )
        )
        return result

    found: dict[str, object] = {}
    found_lines: dict[str, int] = {}
    for node in tree.body:
        target_name: Optional[str] = None
        value_node: Optional[ast.expr] = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target_name = node.targets[0].id
            value_node = node.value
        if target_name in ("BACKEND", "MODEL_ID") and isinstance(
            value_node, ast.Constant
        ):
            found[target_name] = value_node.value
            found_lines[target_name] = getattr(node, "lineno", None)

    result.files_checked = 1

    for name, expected in (("BACKEND", expected_backend), ("MODEL_ID", expected_model_id)):
        if name not in found:
            result.failures.append(
                LintFailure(
                    file="config.py",
                    line=None,
                    column=None,
                    message=(
                        f"config.py does not define a top-level constant {name}. "
                        f"The compile pipeline instructed the LLM to emit "
                        f"{name} = {expected!r}; verify the LLM output."
                    ),
                    rule_ref="C8 runtime defaults",
                )
            )
            continue
        actual = found[name]
        if actual != expected:
            result.failures.append(
                LintFailure(
                    file="config.py",
                    line=found_lines.get(name),
                    column=None,
                    message=(
                        f"config.py:{name} = {actual!r} but the compile pipeline "
                        f"instructed {name} = {expected!r}. To change the default, "
                        f"edit .claude/data/runtime_defaults.json or recompile with "
                        f"--backend / --model-id; do not edit config.py by hand."
                    ),
                    rule_ref="C8 runtime defaults",
                )
            )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: instruct-has-description ───


_INSTRUCT_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "slots.py",
    "constrained_slots.py",
)


def _is_m_instruct_call(node: ast.AST) -> bool:
    """True iff node is a `Call` whose func is `m.instruct` (Attribute on Name 'm')."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "instruct":
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "m"


def lint_instruct_has_description(package_dir: Path) -> LintResult:
    """Every `m.instruct(...)` call must include a description argument.

    Mellea 0.5.0's `MelleaSession.instruct(description, *, ...)` requires
    `description` as the first positional-or-keyword argument. A call with
    only keyword args like `format=`, `grounding_context=`, `model_options=`
    compiles cleanly but crashes at runtime with:

        TypeError: instruct() missing 1 required positional argument: 'description'

    Accept either a positional first argument OR a `description=` keyword.
    Anything else is a hard failure.

    Scope: `pipeline.py`, `slots.py`, `constrained_slots.py`. (Other files
    in a compiled package do not typically call `m.instruct` directly.)
    """
    result = LintResult(lint_id="instruct-has-description", verdict="pass")

    files_to_check: List[Path] = []
    for fname in _INSTRUCT_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            # other lints (e.g. parseable) own SyntaxError reporting
            continue

        for node in ast.walk(tree):
            if not _is_m_instruct_call(node):
                continue
            has_positional = len(node.args) >= 1
            has_description_kwarg = any(
                kw.arg == "description" for kw in node.keywords
            )
            if has_positional or has_description_kwarg:
                continue
            result.failures.append(
                LintFailure(
                    file=rel,
                    line=getattr(node, "lineno", None),
                    column=getattr(node, "col_offset", None),
                    message=(
                        "m.instruct(...) call is missing a description. "
                        "Mellea 0.5+ requires `description` as the first "
                        "positional argument (or as a `description=` keyword). "
                        "A call with only `format=`/`grounding_context=`/"
                        "`model_options=` raises TypeError at runtime: "
                        "\"instruct() missing 1 required positional argument: "
                        "'description'\". Fix: add a description string as the "
                        "first positional argument, e.g. "
                        "`m.instruct(\"Produce a <Model> per the schema, "
                        "grounded in the provided context.\", format=...)`."
                    ),
                    rule_ref="KB12 (Mellea 0.5 instruct signature)",
                )
            )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: generative-forbidden-params (KB6) ───


# Static fallback used when `intermediate/mellea_api_ref.json:forbidden_param_names`
# is not available (e.g. older packages compiled before the grounding step
# wrote the file). Sourced from `mellea.stdlib.components.genstub._disallowed_param_names`
# via the live introspection in `compile/grounding.py`. Keep in sync with the
# `_FORBIDDEN_PARAM_NAMES_FALLBACK` in that module.
_GENERATIVE_FORBIDDEN_PARAMS_FALLBACK: frozenset = frozenset({
    "f_args",
    "f_kwargs",
    "m",
    "context",
    "backend",
    "model_options",
    "strategy",
    "precondition_requirements",
    "requirements",
})


_GENERATIVE_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "slots.py",
    "constrained_slots.py",
)


def _is_generative_decorator(decorator: ast.expr) -> bool:
    """True iff `decorator` is a `@generative` (bare or attribute) decoration."""
    if isinstance(decorator, ast.Name):
        return decorator.id == "generative"
    if isinstance(decorator, ast.Attribute):
        return decorator.attr == "generative"
    return False


def _load_forbidden_param_names(package_dir: Path) -> frozenset:
    """Return the forbidden param-name set, preferring the grounded list."""
    api_ref_path = package_dir / "intermediate" / "mellea_api_ref.json"
    if api_ref_path.exists():
        try:
            data = json.loads(api_ref_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and not data.get("grounding_unavailable", False):
            names = data.get("forbidden_param_names")
            if isinstance(names, list):
                return frozenset(n for n in names if isinstance(n, str))
    return _GENERATIVE_FORBIDDEN_PARAMS_FALLBACK


def lint_generative_forbidden_params(package_dir: Path) -> LintResult:
    """Every `@generative` function MUST avoid Mellea's reserved param names.

    Mellea's `@generative` decorator (in `mellea.stdlib.components.genstub`)
    raises `ValueError: cannot create a generative stub with disallowed
    parameter names: ['m']` at module import when the function signature
    declares any forbidden name as a parameter. The forbidden list includes
    `m` (the session, passed at call time), `context`, `backend`,
    `model_options`, `strategy`, `precondition_requirements`, `requirements`,
    `f_args`, `f_kwargs`.

    Regression target: `risk-and-issues-manager-scott-margetts` compiled with
    6 violations (every `@generative` def took `m` as its first parameter).
    All Step 7 LLM-side lints reported pass, but the smoke-check crashed at
    import time with the ValueError. The LLM's prose-level KB6 check missed
    them; lifting the check into Python catches them deterministically.

    The correct pattern (per `mellea-fy-generate.md`):
        @generative
        def extract_sentiment(text: str) -> str:
            \"\"\"...\"\"\"
            ...
        # call site passes m positionally:
        with start_session(BACKEND, MODEL_ID) as m:
            sentiment = extract_sentiment(m, text=user_input)

    Detection: AST-parse `pipeline.py`, `slots.py`, `constrained_slots.py`;
    find every top-level FunctionDef whose decorator list contains
    `@generative` (Name or Attribute); check every positional / keyword-only
    parameter name against the forbidden list (sourced from
    `intermediate/mellea_api_ref.json:forbidden_param_names`, falling back
    to a static list if the grounding file is absent).
    """
    result = LintResult(lint_id="generative-forbidden-params", verdict="pass")

    forbidden = _load_forbidden_param_names(package_dir)

    files_to_check: List[Path] = []
    for fname in _GENERATIVE_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue  # parseable lint owns SyntaxError reporting

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(
                _is_generative_decorator(d) for d in node.decorator_list
            ):
                continue
            offenders: List[str] = []
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                if arg.arg in forbidden:
                    offenders.append(arg.arg)
            if not offenders:
                continue
            result.failures.append(
                LintFailure(
                    file=rel,
                    line=getattr(node, "lineno", None),
                    column=getattr(node, "col_offset", None),
                    message=(
                        f"@generative function `{node.name}` declares forbidden "
                        f"parameter name(s): {', '.join(offenders)}. Mellea's "
                        f"`@generative` decorator reserves these names "
                        f"(`m` is the session, passed at call time; the rest "
                        f"are decorator-internal). At import the decorator "
                        f"raises `ValueError: cannot create a generative stub "
                        f"with disallowed parameter names: {offenders!r}`. "
                        f"Fix: remove the offending parameter(s) from the "
                        f"function signature; pass `m` positionally at call "
                        f"site (e.g. `with start_session(...) as m: "
                        f"{node.name}(m, ...)`)."
                    ),
                    rule_ref="KB6 (genstub forbidden param names)",
                )
            )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: generative-call-passes-session (KB6, callsite half) ───


def _collect_generative_function_names(package_dir: Path) -> set:
    """Scan slots.py + constrained_slots.py for @generative-decorated function names."""
    names: set = set()
    for fname in ("slots.py", "constrained_slots.py"):
        p = package_dir / fname
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text(), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_is_generative_decorator(d) for d in node.decorator_list):
                names.add(node.name)
    return names


def lint_generative_call_passes_session(package_dir: Path) -> LintResult:
    """Every call to a `@generative` function MUST pass a session (or context+backend).

    Mellea's `GenerativeStub.__call__` (`mellea.stdlib.components.genstub`)
    raises `TypeError: generative stub requires either a MelleaSession
    (m=...) or both a Context and Backend (context=..., backend=...) to be
    provided as the first argument(s)` when called without one of:

      1. A first positional argument (the session `m`).
      2. A `m=...` keyword argument.
      3. Both `context=...` AND `backend=...` keyword arguments.

    Regression target: `eu-ai-act-classification-werner-plutat` had
    `with start_session(BACKEND, MODEL_ID):` (no `as m`) followed by
    `extract_ai_system_profile_raw(system_description=...)` (no positional
    `m`). Step 7 LLM-side lints reported pass; smoke-check crashed at the
    first call into a generative stub. This lint catches the call-site
    half of KB6 (the definition-side half is owned by
    `generative-forbidden-params`).

    The correct pattern is:
        with start_session(BACKEND, MODEL_ID) as m:
            raw_profile = extract_ai_system_profile_raw(m, system_description=...)

    Detection: collect every `@generative` function name from `slots.py` +
    `constrained_slots.py`. AST-walk `pipeline.py`, `slots.py`,
    `constrained_slots.py` for any `Call` whose `func` is one of those
    names; assert it has at least one positional arg, OR an `m=` keyword,
    OR both `context=` and `backend=` keywords. Hard failure naming the
    file:line + the offending call.
    """
    result = LintResult(lint_id="generative-call-passes-session", verdict="pass")

    generative_names = _collect_generative_function_names(package_dir)
    if not generative_names:
        # No @generative functions defined → nothing to check; trivially passes.
        return result

    files_to_check: List[Path] = []
    for fname in _GENERATIVE_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        # Skip the FunctionDef itself when scanning slots.py — `@generative` defs
        # contain placeholder bodies (`...`) and no recursive calls.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                callee = func.id
            elif isinstance(func, ast.Attribute):
                callee = func.attr
            else:
                continue
            if callee not in generative_names:
                continue

            has_positional = len(node.args) >= 1
            kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            has_m_kw = "m" in kw_names
            has_ctx_backend = "context" in kw_names and "backend" in kw_names

            if has_positional or has_m_kw or has_ctx_backend:
                continue

            result.failures.append(
                LintFailure(
                    file=rel,
                    line=getattr(node, "lineno", None),
                    column=getattr(node, "col_offset", None),
                    message=(
                        f"Call to @generative function `{callee}(...)` does "
                        f"not pass a MelleaSession. Mellea's generative stub "
                        f"raises `TypeError: generative stub requires either "
                        f"a MelleaSession (m=...) or both a Context and "
                        f"Backend (context=..., backend=...) to be provided "
                        f"as the first argument(s)` at call time. Fix: bind "
                        f"the session with `as m` and pass it positionally, "
                        f"e.g. `with start_session(BACKEND, MODEL_ID) as m: "
                        f"{callee}(m, ...)`."
                    ),
                    rule_ref="KB6 (generative call requires session)",
                )
            )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: pipeline-entry-canonical ───


def _entry_name_from_manifest(package_dir: Path) -> Optional[str]:
    """Read `melleafy.json:entry_signature` and return the leading function name."""
    manifest = package_dir / "melleafy.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sig = data.get("entry_signature")
    if not isinstance(sig, str):
        return None
    match = _ENTRY_SIGNATURE_NAME_RE.match(sig)
    return match.group(1) if match else None


def lint_pipeline_entry_canonical(package_dir: Path) -> LintResult:
    """`pipeline.py` must expose `run_pipeline` as a top-level def AND `melleafy.json`
    must agree.

    Empirically observed regression: `nis2-navigator` pipeline.py defined three
    public functions starting with `run_*` (`run_phase_2_gap_analysis`,
    `run_phase_3_roadmap`, `run_pipeline`). The smoke-check entry-point loader
    (`toolkit/file_utils.py:load_skill_pipeline`) used `dir()` ordering, which
    is alphabetical, and picked `run_phase_2_gap_analysis` — causing a
    `TypeError` when smoke-check passed entry-point kwargs to a phase helper.

    Two related-but-independent invariants enforce together:

      1. `pipeline.py` defines a top-level `FunctionDef run_pipeline`.
      2. `melleafy.json:entry_signature` (if present) starts with `run_pipeline(`.

    The loader was patched to use `melleafy.json:entry_signature` as the
    authoritative source of truth, but this lint backstops the contract so
    misalignment is caught at compile time rather than fixture smoke-check.

    Skip cases:
      - `pipeline.py` absent → skipped (other lints own missing-file reports).
    """
    result = LintResult(lint_id="pipeline-entry-canonical", verdict="pass")

    pipeline_path = package_dir / "pipeline.py"
    if not pipeline_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = "pipeline.py not found in package"
        return result
    result.files_checked = 1

    try:
        tree = ast.parse(pipeline_path.read_text(), filename=str(pipeline_path))
    except SyntaxError as exc:
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="pipeline.py",
                line=exc.lineno,
                column=exc.offset,
                message=f"SyntaxError prevents parsing pipeline.py: {exc.msg}",
                rule_ref="pipeline entry contract",
            )
        )
        return result

    top_level_run_defs = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("run_")
    ]

    if _CANONICAL_ENTRY_NAME not in top_level_run_defs:
        result.failures.append(
            LintFailure(
                file="pipeline.py",
                line=None,
                column=None,
                message=(
                    f"pipeline.py does not define a top-level "
                    f"`{_CANONICAL_ENTRY_NAME}` function. The canonical entry "
                    f"point name is required by convention "
                    f"(see `.claude/commands/mellea-fy-generate.md`) and "
                    f"by the smoke-check loader fallback in "
                    f"`toolkit/file_utils.py:load_skill_pipeline`. Found "
                    f"top-level run_* functions: "
                    f"{', '.join(top_level_run_defs) or '(none)'}."
                ),
                rule_ref="pipeline entry contract",
            )
        )

    manifest_name = _entry_name_from_manifest(package_dir)
    if manifest_name is not None and manifest_name != _CANONICAL_ENTRY_NAME:
        result.failures.append(
            LintFailure(
                file="melleafy.json",
                line=None,
                column=None,
                message=(
                    f"melleafy.json:entry_signature names "
                    f"`{manifest_name}` as the entry function, but the "
                    f"canonical entry name is `{_CANONICAL_ENTRY_NAME}`. "
                    f"The loader at `toolkit/file_utils.py:load_skill_pipeline` "
                    f"treats the manifest as authoritative; this mismatch "
                    f"would cause the smoke-check to invoke the wrong "
                    f"function. Regenerate the package so Step 5 writes "
                    f"`entry_signature` starting with `run_pipeline(`."
                ),
                rule_ref="pipeline entry contract",
            )
        )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: fixture-signature-bound ───


def _collect_run_pipeline_params(pipeline_path: Path) -> Optional[dict]:
    """Return {'names': set[str], 'has_var_keyword': bool, 'node': FunctionDef} or None.

    None on parse failure or when `run_pipeline` is absent. `has_var_keyword`
    is True when the signature includes `**kwargs`, in which case any fixture
    keys are accepted.
    """
    try:
        tree = ast.parse(pipeline_path.read_text(), filename=str(pipeline_path))
    except SyntaxError:
        return None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == _CANONICAL_ENTRY_NAME
        ):
            args = node.args
            names: Set[str] = set()
            names.update(a.arg for a in args.posonlyargs)
            names.update(a.arg for a in args.args)
            names.update(a.arg for a in args.kwonlyargs)
            return {
                "names": names,
                "has_var_keyword": args.kwarg is not None,
                "node": node,
            }
    return None


def _extract_inputs_dict_keys(make_fn: ast.FunctionDef) -> Optional[List[str]]:
    """Walk a `make_<id>()` factory and return the string keys of its `inputs` dict.

    Pattern recognised:

        def make_X():
            inputs = {"key1": ..., "key2": ...}
            return inputs, "X", "..."

    Returns None if the function does not match this shape (e.g. inputs built
    dynamically), in which case the caller should skip the fixture rather
    than failing it.
    """
    for stmt in make_fn.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "inputs"):
            continue
        if not isinstance(stmt.value, ast.Dict):
            return None
        keys: List[str] = []
        for k in stmt.value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                return None  # non-string-literal key — skip the fixture
        return keys
    return None  # no `inputs = {...}` assignment found


def lint_fixture_signature_bound(package_dir: Path) -> LintResult:
    """Every fixture's `inputs` dict keys must be a subset of `run_pipeline`'s parameters.

    Smoke-check invokes the entry as `run_pipeline(**inputs)`. Any extra key
    raises a `TypeError` at fixture-run time. This is exactly the failure
    that surfaced in `nis2-navigator` when the loader (separately) picked a
    phase helper that didn't accept `regulatory_updates`, but the contract
    failure is independent: even with the loader fix, fixtures targeting
    `run_pipeline` MUST agree on the parameter set.

    The prose contract is already in `.claude/commands/mellea-fy-fixtures.md`:
    "The keys in every fixture's `inputs` object MUST be identical to the
    parameter names of the `run_pipeline` function in `pipeline.py`."
    This lint enforces it mechanically.

    Skip cases (verdict=skipped):
      - `pipeline.py` absent
      - `pipeline.py` does not define a `run_pipeline` function
        (pipeline-entry-canonical owns reporting that)
      - `fixtures/` absent (fixtures-loader-contract owns reporting)

    Skip per-fixture (silent, not a failure):
      - A fixture file's `make_<id>()` does not assign `inputs = {...}` as a
        constant-key dict literal (e.g. inputs built dynamically). The lint
        cannot statically verify these and defers to runtime.

    Hard failure:
      - Any fixture key that is not a parameter of `run_pipeline` and the
        signature does not include `**kwargs`.
    """
    result = LintResult(lint_id="fixture-signature-bound", verdict="pass")

    pipeline_path = package_dir / "pipeline.py"
    if not pipeline_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = "pipeline.py not found in package"
        return result

    params = _collect_run_pipeline_params(pipeline_path)
    if params is None:
        result.verdict = "skipped"
        result.skipped_reason = (
            f"`{_CANONICAL_ENTRY_NAME}` not found in pipeline.py "
            f"(see pipeline-entry-canonical lint)"
        )
        return result

    if params["has_var_keyword"]:
        # **kwargs accepts anything — nothing to check.
        result.verdict = "pass"
        return result

    fixtures_dir = package_dir / "fixtures"
    if not fixtures_dir.is_dir():
        result.verdict = "skipped"
        result.skipped_reason = "fixtures/ directory not found in package"
        return result

    fixture_files = sorted(
        p
        for p in fixtures_dir.iterdir()
        if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
    )
    result.files_checked = len(fixture_files)

    valid_params = params["names"]

    for fixture_path in fixture_files:
        rel = f"fixtures/{fixture_path.name}"
        try:
            tree = ast.parse(fixture_path.read_text(), filename=str(fixture_path))
        except SyntaxError:
            # Skip unparseable — parseable lint owns reporting.
            continue

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("make_")
            ):
                continue
            keys = _extract_inputs_dict_keys(node)
            if keys is None:
                # Couldn't statically verify (dynamic construction etc.) — skip.
                continue
            for key in keys:
                if key not in valid_params:
                    result.failures.append(
                        LintFailure(
                            file=rel,
                            line=getattr(node, "lineno", None),
                            column=getattr(node, "col_offset", None),
                            message=(
                                f"Fixture `{node.name}` returns `inputs` with "
                                f"key '{key}', but `{_CANONICAL_ENTRY_NAME}` "
                                f"does not declare a parameter of that name. "
                                f"Smoke-check invokes `{_CANONICAL_ENTRY_NAME}"
                                f"(**inputs)` and will raise "
                                f"`TypeError: got an unexpected keyword "
                                f"argument '{key}'` at fixture-run time. "
                                f"Valid parameters: "
                                f"{', '.join(sorted(valid_params)) or '(none)'}. "
                                f"Fix: align the fixture's `inputs` keys with "
                                f"the entry signature in pipeline.py."
                            ),
                            rule_ref="R16 fixture inputs ⊆ run_pipeline params",
                        )
                    )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: validator-soundness (KB3 + KB4) ───


_VALIDATOR_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "requirements.py",
    "slots.py",
    "constrained_slots.py",
)


def _validator_func_name(node: ast.expr) -> Optional[str]:
    """Return the bare name of a Call's func (Name.id or Attribute.attr)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_constant_true(node: ast.AST) -> bool:
    """True iff `node` is the literal `True` constant."""
    return isinstance(node, ast.Constant) and node.value is True


def _has_context_annotated_arg(fn_def: ast.FunctionDef) -> bool:
    """True iff any positional arg of `fn_def` is annotated as `Context`."""
    for arg in fn_def.args.args:
        ann = arg.annotation
        if isinstance(ann, ast.Name) and ann.id == "Context":
            return True
        if isinstance(ann, ast.Attribute) and ann.attr == "Context":
            return True
    return False


def _collect_local_function_defs(tree: ast.AST) -> dict:
    """Map every FunctionDef name in the module to its node (last def wins)."""
    defs: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
    return defs


def lint_validator_soundness(package_dir: Path) -> LintResult:
    """Every `validation_fn=` must be well-typed and non-vacuous (KB3 + KB4).

    Sub-check A (KB3): a `validation_fn=` argument must either be wrapped in
    `simple_validate(lambda ...)` OR be a named function whose signature is
    `(ctx: Context, result) -> ValidationResult`. A raw bare lambda passed
    as `validation_fn=lambda ...` crashes at validate time with TypeError
    because Mellea passes `(ctx, result)` to the validator.

    Sub-check B (KB4): a `simple_validate(lambda ...: True)` whose body is
    the literal constant `True` is a hard failure ("vacuous validator") —
    it cannot reject any output, so the surrounding requirement provides no
    signal to the sampling loop.

    Scope: pipeline.py, requirements.py, slots.py, constrained_slots.py.
    """
    result = LintResult(lint_id="validator-soundness", verdict="pass")

    files_to_check: List[Path] = []
    for fname in _VALIDATOR_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        local_defs = _collect_local_function_defs(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "validation_fn":
                    continue
                value = kw.value
                line = getattr(node, "lineno", None)
                col = getattr(node, "col_offset", None)

                if isinstance(value, ast.Lambda):
                    result.failures.append(
                        LintFailure(
                            file=rel,
                            line=line,
                            column=col,
                            message=(
                                "validation_fn= is a raw bare lambda. "
                                "Mellea's validation protocol passes a "
                                "(Context, result) pair to the validator; a "
                                "bare `lambda x: ...` crashes at validate "
                                "time with TypeError. Fix: wrap with "
                                "`simple_validate(...)` "
                                "(e.g. `validation_fn=simple_validate("
                                "lambda x: len(x.split()) < 100)`), or "
                                "replace with a named function "
                                "`def _check(ctx: Context, result) -> "
                                "ValidationResult: ...`."
                            ),
                            rule_ref="KB3 (validator signature)",
                        )
                    )
                    continue

                if (
                    isinstance(value, ast.Call)
                    and _validator_func_name(value.func) == "simple_validate"
                ):
                    if value.args and isinstance(value.args[0], ast.Lambda):
                        body = value.args[0].body
                        if _is_constant_true(body):
                            result.failures.append(
                                LintFailure(
                                    file=rel,
                                    line=line,
                                    column=col,
                                    message=(
                                        "Vacuous validator: "
                                        "`simple_validate(lambda ...: True)`"
                                        " always returns True regardless of "
                                        "input, so the surrounding "
                                        "requirement cannot reject any "
                                        "output. Fix: replace the lambda "
                                        "body with a real predicate (e.g. "
                                        "`lambda x: len(x.split()) < 100`) "
                                        "or remove the requirement entirely "
                                        "if no real check is intended."
                                    ),
                                    rule_ref="KB4 (no vacuous validators)",
                                )
                            )
                    continue

                if isinstance(value, ast.Name):
                    fn_def = local_defs.get(value.id)
                    if fn_def is None:
                        # Imported / out-of-module — assume OK.
                        continue
                    args = fn_def.args.args
                    if not (
                        len(args) >= 2 and _has_context_annotated_arg(fn_def)
                    ):
                        result.failures.append(
                            LintFailure(
                                file=rel,
                                line=line,
                                column=col,
                                message=(
                                    f"validation_fn={value.id} references a "
                                    f"locally-defined function whose "
                                    f"signature is not (ctx: Context, "
                                    f"result) -> ValidationResult. Mellea "
                                    f"calls the validator as `fn(ctx, "
                                    f"result)` and the first argument must "
                                    f"be annotated `Context`. Fix: change "
                                    f"the signature to `def {value.id}(ctx:"
                                    f" Context, result) -> "
                                    f"ValidationResult: ...`, OR wrap a "
                                    f"single-arg predicate in "
                                    f"`simple_validate(...)` at the call "
                                    f"site."
                                ),
                                rule_ref="KB3 (validator signature)",
                            )
                        )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: session-boundary (KB5) ───


_SESSION_BOUNDARY_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "slots.py",
    "constrained_slots.py",
)


def _is_start_session_call(node: ast.AST) -> bool:
    """True iff node is a Call whose callee is the name `start_session`."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "start_session"
    if isinstance(func, ast.Attribute):
        return func.attr == "start_session"
    return False


def _instruct_format_type_name(node: ast.Call) -> Optional[str]:
    """If `node` is `<...>.instruct(format=X)`, return X's type name (or None)."""
    if not (
        isinstance(node.func, ast.Attribute) and node.func.attr == "instruct"
    ):
        return None
    for kw in node.keywords:
        if kw.arg != "format":
            continue
        if isinstance(kw.value, ast.Name):
            return kw.value.id
        if isinstance(kw.value, ast.Attribute):
            return kw.value.attr
    return None


def lint_session_boundary(package_dir: Path) -> LintResult:
    """Each `start_session(...)` block must use at most one distinct format type (KB5).

    Mellea schema priming: once a session has generated N objects matching
    schema A, the next `m.instruct(format=B)` call in the same session keeps
    producing A-shaped output. The fix is to split the session — open a new
    `start_session()` block per distinct format type. Always-halts; no repair.
    """
    result = LintResult(lint_id="session-boundary", verdict="pass")

    files_to_check: List[Path] = []
    for fname in _SESSION_BOUNDARY_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            is_session = any(
                _is_start_session_call(item.context_expr) for item in node.items
            )
            if not is_session:
                continue

            format_types: Set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _instruct_format_type_name(child)
                if name is not None:
                    format_types.add(name)

            if len(format_types) <= 1:
                continue

            result.failures.append(
                LintFailure(
                    file=rel,
                    line=getattr(node, "lineno", None),
                    column=getattr(node, "col_offset", None),
                    message=(
                        f"start_session() block uses {len(format_types)} "
                        f"distinct format types: "
                        f"{', '.join(sorted(format_types))}. Mellea's schema "
                        f"priming means the LLM cannot reliably switch "
                        f"BaseModel schemas mid-session — after N successful "
                        f"generations of schema A, calls with format=B keep "
                        f"producing A-shaped output. Fix: split into "
                        f"separate `with start_session(...) as m:` blocks, "
                        f"one per distinct format type."
                    ),
                    rule_ref="KB5 (session schema priming)",
                )
            )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: prefix-persona (KB7) ───


_PREFIX_PERSONA_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "slots.py",
    "constrained_slots.py",
)


def _collect_config_constant_names(package_dir: Path) -> Set[str]:
    """Return the set of top-level constant names defined in `<package>/config.py`."""
    config_path = package_dir / "config.py"
    if not config_path.exists():
        return set()
    try:
        tree = ast.parse(config_path.read_text(), filename=str(config_path))
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _collect_names_imported_from_config(tree: ast.AST) -> Set[str]:
    """Return names brought in via `from config import X` or `from .config import X`."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module == "config" or module.endswith(".config"):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _is_instruct_attribute_call(node: ast.AST) -> bool:
    """True iff node is an `<x>.instruct(...)` Call (any receiver)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "instruct"


def lint_prefix_persona(package_dir: Path) -> LintResult:
    """`m.instruct(prefix=<config_constant>)` is misuse (KB7).

    `prefix=` is for structured-output continuation (e.g. `prefix='{"result":"'`)
    not persona injection. Use `model_options={ModelOption.SYSTEM_PROMPT: ...}`
    instead.
    """
    result = LintResult(lint_id="prefix-persona", verdict="pass")

    config_constants = _collect_config_constant_names(package_dir)

    files_to_check: List[Path] = []
    for fname in _PREFIX_PERSONA_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        config_imported = _collect_names_imported_from_config(tree)

        for node in ast.walk(tree):
            if not _is_instruct_attribute_call(node):
                continue
            for kw in node.keywords:
                if kw.arg != "prefix":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, str
                ):
                    continue
                if isinstance(kw.value, ast.Name):
                    name = kw.value.id
                    is_config_const = name in config_constants
                    is_config_import = name in config_imported
                    provenance = (
                        " (defined in config.py)"
                        if is_config_const
                        else (
                            " (imported from config)"
                            if is_config_import
                            else " (looks like a config constant — uppercase Name)"
                            if name.isupper()
                            else ""
                        )
                    )
                    result.failures.append(
                        LintFailure(
                            file=rel,
                            line=getattr(node, "lineno", None),
                            column=getattr(node, "col_offset", None),
                            message=(
                                f"`.instruct(prefix={name})`{provenance} uses "
                                f"the `prefix=` parameter to inject persona / "
                                f"system-prompt text. `prefix=` is for "
                                f"structured-output continuation (e.g. "
                                f"`prefix='{{\"result\":\"'`), not persona "
                                f"injection. Fix: use `model_options="
                                f"{{ModelOption.SYSTEM_PROMPT: {name}}}` "
                                f"instead (`from mellea.backends.model_options"
                                f" import ModelOption`)."
                            ),
                            rule_ref="KB7 (prefix= for persona vs SYSTEM_PROMPT)",
                        )
                    )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: no-watsonx (KB8) ───


_NO_WATSONX_LINT_FILES: Tuple[str, ...] = (
    "pipeline.py",
    "slots.py",
    "constrained_slots.py",
    "config.py",
)


def lint_no_watsonx(package_dir: Path) -> LintResult:
    """The string `"watsonx"` MUST NOT appear in any generated `.py` file (KB8).

    Mellea 0.5 deprecated the WatsonX backend; any reference crashes at
    runtime when the session opens.
    """
    result = LintResult(lint_id="no-watsonx", verdict="pass")

    files_to_check: List[Path] = []
    for fname in _NO_WATSONX_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            text = py_file.read_text()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "watsonx" in line:
                result.failures.append(
                    LintFailure(
                        file=rel,
                        line=lineno,
                        column=line.find("watsonx") + 1,
                        message=(
                            f'"watsonx" backend string found in {rel}:'
                            f"{lineno}. Mellea 0.5 deprecated the WatsonX "
                            f"backend; `start_session(\"watsonx\", ...)` "
                            f"raises at runtime and any fixture using it "
                            f"fails at session-open. Fix: change the "
                            f"backend to a supported value (e.g. "
                            f'`"ollama"`) — edit `.claude/data/'
                            f"runtime_defaults.json or recompile with "
                            f"`--backend <name>`."
                        ),
                        rule_ref="KB8 (Mellea 0.5 watsonx deprecation)",
                    )
                )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: optional-extraction-guidance (KB11) ───


_EXTRACTION_KEYWORDS: Tuple[str, ...] = ("extract", "do not ask", "if the")
_SCHEMA_NAME_SUFFIXES: Tuple[str, ...] = ("Schema", "Intent")


def _is_basemodel_subclass(cls: ast.ClassDef) -> bool:
    """True iff `cls` inherits from `BaseModel` (Name or Attribute)."""
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def _is_optional_annotation(ann: Optional[ast.expr]) -> bool:
    """True iff annotation is `Optional[...]` (Name or Attribute)."""
    if not isinstance(ann, ast.Subscript):
        return False
    value = ann.value
    if isinstance(value, ast.Name) and value.id == "Optional":
        return True
    if isinstance(value, ast.Attribute) and value.attr == "Optional":
        return True
    return False


def _collect_format_referenced_schemas(package_dir: Path) -> Set[str]:
    """Collect class names referenced as `format=<Name>` in any m.instruct call."""
    names: Set[str] = set()
    for fname in ("pipeline.py", "slots.py", "constrained_slots.py"):
        p = package_dir / fname
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text(), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not _is_m_instruct_call(node):
                continue
            for kw in node.keywords:
                if kw.arg != "format":
                    continue
                if isinstance(kw.value, ast.Name):
                    names.add(kw.value.id)
                elif isinstance(kw.value, ast.Attribute):
                    names.add(kw.value.attr)
    return names


def lint_optional_extraction_guidance(package_dir: Path) -> LintResult:
    """KB11: Every Optional field on a P2 schema MUST carry extraction guidance.

    Mellea's `format=<BaseModel>` invocation asks the model to populate every
    declared field; Optional fields default to asking the user. The fix is
    to put extraction guidance into the Field description so the model
    extracts from context instead of asking.
    """
    result = LintResult(lint_id="optional-extraction-guidance", verdict="pass")

    schemas_path = package_dir / "schemas.py"
    if not schemas_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = "schemas.py not found in package"
        return result
    result.files_checked = 1

    try:
        tree = ast.parse(schemas_path.read_text(), filename=str(schemas_path))
    except SyntaxError:
        result.verdict = "skipped"
        result.skipped_reason = "schemas.py is not parseable"
        return result

    format_referenced = _collect_format_referenced_schemas(package_dir)

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        if not _is_basemodel_subclass(cls):
            continue
        name_in_scope = (
            cls.name.endswith(_SCHEMA_NAME_SUFFIXES)
            or cls.name in format_referenced
        )
        if not name_in_scope:
            continue

        for stmt in cls.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not _is_optional_annotation(stmt.annotation):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            field_name = stmt.target.id

            val = stmt.value
            if val is None or not isinstance(val, ast.Call):
                result.failures.append(
                    LintFailure(
                        file="schemas.py",
                        line=getattr(stmt, "lineno", None),
                        column=getattr(stmt, "col_offset", None),
                        message=(
                            f"Optional field `{field_name}` on `{cls.name}` "
                            f"has no `Field(description=...)` with extraction "
                            f"guidance. Fix: add `= Field(default=None, "
                            f"description=\"Extract <X> if the user has "
                            f"stated it; otherwise null. Do not ask.\")`."
                        ),
                        rule_ref="KB11 (Optional field extraction guidance)",
                    )
                )
                continue

            callee = (
                val.func.id if isinstance(val.func, ast.Name)
                else val.func.attr if isinstance(val.func, ast.Attribute)
                else None
            )
            if callee != "Field":
                result.failures.append(
                    LintFailure(
                        file="schemas.py",
                        line=getattr(stmt, "lineno", None),
                        column=getattr(stmt, "col_offset", None),
                        message=(
                            f"Optional field `{field_name}` on `{cls.name}` "
                            f"has a non-Field default. KB11 requires "
                            f"`Field(description=...)` with extraction "
                            f"guidance."
                        ),
                        rule_ref="KB11 (Optional field extraction guidance)",
                    )
                )
                continue

            description: Optional[str] = None
            for kw in val.keywords:
                if (
                    kw.arg == "description"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    description = kw.value.value
                    break

            if description is None:
                result.failures.append(
                    LintFailure(
                        file="schemas.py",
                        line=getattr(stmt, "lineno", None),
                        column=getattr(stmt, "col_offset", None),
                        message=(
                            f"Optional field `{field_name}` on `{cls.name}`: "
                            f"`Field(...)` is missing a string `description=` "
                            f"argument. Fix: add `description=\"Extract ... "
                            f"if the user has stated it; do not ask.\"`."
                        ),
                        rule_ref="KB11 (Optional field extraction guidance)",
                    )
                )
                continue

            lower = description.lower()
            if not any(kw in lower for kw in _EXTRACTION_KEYWORDS):
                result.failures.append(
                    LintFailure(
                        file="schemas.py",
                        line=getattr(stmt, "lineno", None),
                        column=getattr(stmt, "col_offset", None),
                        message=(
                            f"Optional field `{field_name}` on `{cls.name}`: "
                            f"`Field(description=...)` lacks extraction "
                            f"guidance. Description must contain one of "
                            f"'extract', 'do not ask', 'if the' (case-"
                            f"insensitive). Current: {description!r}."
                        ),
                        rule_ref="KB11 (Optional field extraction guidance)",
                    )
                )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: run-pipeline-params-typed (Rule 3-1) ───


def lint_run_pipeline_params_typed(package_dir: Path) -> LintResult:
    """Rule 3-1: Every parameter of `pipeline.py:run_pipeline` MUST be typed.

    Bare parameter names produce untyped CLI interfaces and break
    signature-derived consumers (melleafy.json entry_signature, MCP tool
    wrapping, fixture inference).
    """
    result = LintResult(lint_id="run-pipeline-params-typed", verdict="pass")

    pipeline_path = package_dir / "pipeline.py"
    if not pipeline_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = "pipeline.py not found in package"
        return result
    result.files_checked = 1

    try:
        tree = ast.parse(pipeline_path.read_text(), filename=str(pipeline_path))
    except SyntaxError:
        result.verdict = "skipped"
        result.skipped_reason = "pipeline.py is not parseable"
        return result

    run_pipeline_node: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == _CANONICAL_ENTRY_NAME
        ):
            run_pipeline_node = node
            break

    if run_pipeline_node is None:
        result.verdict = "skipped"
        result.skipped_reason = (
            f"`{_CANONICAL_ENTRY_NAME}` not found in pipeline.py "
            f"(see pipeline-entry-canonical lint)"
        )
        return result

    args = run_pipeline_node.args
    all_named_args = (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
    )

    bare: List[ast.arg] = [a for a in all_named_args if a.annotation is None]

    if not bare:
        return result

    bare_names = ", ".join(a.arg for a in bare)
    for a in bare:
        result.failures.append(
            LintFailure(
                file="pipeline.py",
                line=getattr(a, "lineno", getattr(run_pipeline_node, "lineno", None)),
                column=getattr(a, "col_offset", None),
                message=(
                    f"`{_CANONICAL_ENTRY_NAME}` parameter `{a.arg}` has no "
                    f"type annotation. Rule 3-1 requires every parameter of "
                    f"the canonical entry point to be fully typed (e.g. "
                    f"`def {_CANONICAL_ENTRY_NAME}({a.arg}: str, ...) -> "
                    f"dict:`). Bare params break "
                    f"`melleafy.json:entry_signature` consumers and "
                    f"downstream MCP wrappers. All bare params: {bare_names}."
                ),
                rule_ref="Rule 3-1 (cross-reference sub-check F)",
            )
        )

    result.verdict = "fail"
    return result


# ─── Lint: import-soundness (Rule 4-2) ───


def _load_known_mellea_modules(package_dir: Path) -> Optional[Set[str]]:
    """Return the set of `mellea.*` module paths from the grounding file."""
    api_ref_path = package_dir / "intermediate" / "mellea_api_ref.json"
    if not api_ref_path.exists():
        return None
    try:
        data = json.loads(api_ref_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("grounding_unavailable", False):
        return None
    modules = data.get("modules")
    if not isinstance(modules, dict):
        return None
    return {k for k in modules.keys() if isinstance(k, str)}


def lint_import_soundness(package_dir: Path) -> LintResult:
    """Every `mellea.*` import path must exist in `mellea_api_ref.json:.modules`."""
    result = LintResult(lint_id="import-soundness", verdict="pass")

    known_modules = _load_known_mellea_modules(package_dir)
    if known_modules is None:
        result.verdict = "skipped"
        result.skipped_reason = (
            "intermediate/mellea_api_ref.json absent, malformed, or marked "
            "grounding_unavailable=true"
        )
        return result

    py_files: List[Path] = []
    for p in sorted(package_dir.rglob("*.py")):
        rel = p.relative_to(package_dir).as_posix()
        if not _is_skipped_path(rel):
            py_files.append(p)
    result.files_checked = len(py_files)

    for py_file in py_files:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("mellea"):
                    continue
                # `from mellea import generative` / `from mellea import
                # start_session` import re-exports from the top-level
                # package — the grounding indexes sub-modules only, so
                # accept `mellea` itself as known.
                if module == "mellea":
                    continue
                if module in known_modules:
                    continue
                result.failures.append(
                    LintFailure(
                        file=rel,
                        line=getattr(node, "lineno", None),
                        column=getattr(node, "col_offset", None),
                        message=(
                            f"`from {module} import ...` references a Mellea "
                            f"module path that does not exist in "
                            f"intermediate/mellea_api_ref.json:.modules. "
                            f"Common error: a shortened path (e.g. "
                            f"`mellea.model_options` instead of "
                            f"`mellea.backends.model_options`)."
                        ),
                        rule_ref="Rule 4-2 (import-soundness)",
                    )
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if not name.startswith("mellea"):
                        continue
                    if name in known_modules:
                        continue
                    result.failures.append(
                        LintFailure(
                            file=rel,
                            line=getattr(node, "lineno", None),
                            column=getattr(node, "col_offset", None),
                            message=(
                                f"`import {name}` references a Mellea module "
                                f"path that does not exist in "
                                f"intermediate/mellea_api_ref.json:.modules."
                            ),
                            rule_ref="Rule 4-2 (import-soundness)",
                        )
                    )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: stdlib-arity (Rule 4-4) ───


_STDLIB_STATIC_SIGS: dict = {
    "simple_validate": {
        "min_pos": 1,
        "max_pos": 1,
        "valid_kwargs": frozenset({"reason"}),
    },
    "req": {
        "min_pos": 1,
        "max_pos": 1,
        "valid_kwargs": frozenset({"validation_fn"}),
    },
    "check": {
        "min_pos": 2,
        "max_pos": 2,
        "valid_kwargs": frozenset(),
    },
}

def _extract_signature_param_block(sig_str: str) -> Optional[str]:
    """Extract the top-level (...) parameter block from a signature string.

    Tracks paren depth so nested types (e.g. `Callable[[int, str], None]`)
    do not confuse the matcher. Returns the inner content of the FIRST
    top-level `(...)` pair, or None if not found.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(sig_str):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                return sig_str[start:i]
    return None


def _split_params_bracket_aware(inner: str) -> List[str]:
    """Split a signature param list by commas at bracket depth 0.

    Handles `Literal['a', 'b']`, `Callable[[int], str]`, `Union[A, B]` etc.
    by tracking depth for `[]`, `()`, `{}` so internal commas are
    preserved within their containers.
    """
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in inner:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            chunk = "".join(buf).strip()
            if chunk:
                parts.append(chunk)
            buf = []
        else:
            buf.append(ch)
    chunk = "".join(buf).strip()
    if chunk:
        parts.append(chunk)
    return parts


def _parse_signature_string(sig_str: str) -> Optional[dict]:
    """Parse a Python signature string into (min_pos, max_pos, valid_kwargs).

    Bracket-aware splitting so complex annotations
    (`Literal[...]`, `Optional[X]`, `Union[...]`, etc.) don't break the
    parameter parse.
    """
    inner = _extract_signature_param_block(sig_str)
    if inner is None:
        return None
    inner = inner.strip()
    if not inner:
        return {"min_pos": 0, "max_pos": 0, "valid_kwargs": frozenset()}
    params_raw = _split_params_bracket_aware(inner)
    has_var = any(
        p.startswith("**") or (p.startswith("*") and p != "*")
        for p in params_raw
    )
    if has_var:
        return None
    min_pos = max_pos = 0
    valid_kwargs: set = set()
    after_kw_marker = False
    for p in params_raw:
        if p == "/":
            continue
        if p == "*":
            after_kw_marker = True
            continue
        name = p.split(":", 1)[0].split("=", 1)[0].strip()
        has_default = "=" in p
        if after_kw_marker:
            valid_kwargs.add(name)
        else:
            max_pos += 1
            if not has_default:
                min_pos += 1
            else:
                valid_kwargs.add(name)
    return {
        "min_pos": min_pos,
        "max_pos": max_pos,
        "valid_kwargs": frozenset(valid_kwargs),
    }


def _load_grounded_stdlib_sigs(package_dir: Path) -> dict:
    """Return {symbol_name: sig_dict} for parseable mellea.stdlib.* signatures."""
    api_ref_path = package_dir / "intermediate" / "mellea_api_ref.json"
    if not api_ref_path.exists():
        return {}
    try:
        data = json.loads(api_ref_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("grounding_unavailable", False):
        return {}
    sigs: dict = {}
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        return {}
    for mod_name, mod_symbols in modules.items():
        if not isinstance(mod_name, str) or not mod_name.startswith("mellea.stdlib."):
            continue
        if not isinstance(mod_symbols, dict):
            continue
        for sym_name, sym_meta in mod_symbols.items():
            if not isinstance(sym_meta, dict):
                continue
            sig_str = sym_meta.get("signature")
            if not isinstance(sig_str, str):
                continue
            parsed = _parse_signature_string(sig_str)
            if parsed is not None and sym_name not in sigs:
                sigs[sym_name] = parsed
    return sigs


def _callee_name(node: ast.Call) -> Optional[str]:
    """Return the function name for a bare-Name Call node, or None.

    Intentionally restricted to `ast.Name` callees — method calls
    (`x.foo(...)`) are NOT in scope because their grounded signatures
    include `self` as the first positional arg, which the call site does
    not pass. Conflating the two produces false positives on common
    calls like `m.instruct(...)` and `session.chat(...)`.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def lint_stdlib_arity(package_dir: Path) -> LintResult:
    """Calls to `mellea.stdlib.*` functions must match their declared arity.

    Static table (`req`, `check`, `simple_validate`) wins over grounded
    signatures because the grounded entries often use `*args, **kwargs`.
    """
    result = LintResult(lint_id="stdlib-arity", verdict="pass")

    grounded_sigs = _load_grounded_stdlib_sigs(package_dir)
    combined: dict = {**grounded_sigs, **_STDLIB_STATIC_SIGS}

    py_files: List[Path] = []
    for p in sorted(package_dir.rglob("*.py")):
        rel = p.relative_to(package_dir).as_posix()
        if not _is_skipped_path(rel):
            py_files.append(p)
    result.files_checked = len(py_files)

    for py_file in py_files:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee_name(node)
            if name is None or name not in combined:
                continue
            sig = combined[name]
            n_pos = len(node.args)
            if not (sig["min_pos"] <= n_pos <= sig["max_pos"]):
                result.failures.append(
                    LintFailure(
                        file=rel,
                        line=getattr(node, "lineno", None),
                        column=getattr(node, "col_offset", None),
                        message=(
                            f"`{name}(...)` called with {n_pos} positional "
                            f"argument(s); signature requires between "
                            f"{sig['min_pos']} and {sig['max_pos']} positional"
                            f" arg(s). Consult intermediate/"
                            f"mellea_api_ref.json:.modules."
                        ),
                        rule_ref="Rule 4-4 (stdlib-arity)",
                    )
                )
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if kw.arg not in sig["valid_kwargs"]:
                    result.failures.append(
                        LintFailure(
                            file=rel,
                            line=getattr(node, "lineno", None),
                            column=getattr(node, "col_offset", None),
                            message=(
                                f"`{name}(...)` has unrecognised keyword "
                                f"argument '{kw.arg}'. Valid keyword args: "
                                f"{sorted(sig['valid_kwargs']) or '(none)'}. "
                                f"Common typo: `validator=` instead of "
                                f"`validation_fn=` for `req()`."
                            ),
                            rule_ref="Rule 4-4 (stdlib-arity)",
                        )
                    )

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: melleafy-json-consistency (Rule 5-1 sub-check A) ───


_MELLEAFY_HARD_REQUIRED: frozenset = frozenset({
    "manifest_version",
    "entry_signature",
    "package_name",
})

_MELLEAFY_SOFT_REQUIRED: frozenset = frozenset({
    "source_runtime",
    "modality",
    "categories_resolved",
    "declared_env_vars",
    "pipeline_parameters",
})

_MIN_MANIFEST_VERSION: Tuple[int, int, int] = (1, 1, 0)


def _parse_semver(s) -> Optional[Tuple[int, int, int]]:
    """Parse 'MAJOR.MINOR.PATCH' into a 3-tuple of ints, or None on failure."""
    parts = str(s).split(".")
    if len(parts) < 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None


def lint_melleafy_json_consistency(package_dir: Path) -> LintResult:
    """melleafy.json must contain the export-command-required fields."""
    result = LintResult(lint_id="melleafy-json-consistency", verdict="pass")

    manifest_path = package_dir / "melleafy.json"
    if not manifest_path.exists():
        result.verdict = "skipped"
        result.skipped_reason = (
            "melleafy.json not found in package — the manifest is written by "
            "the export command"
        )
        return result

    result.files_checked = 1
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="melleafy.json",
                line=None,
                column=None,
                message=f"melleafy.json is not valid JSON: {exc}.",
                rule_ref="Rule 5-1 sub-check A",
            )
        )
        return result

    if not isinstance(data, dict):
        result.verdict = "fail"
        result.failures.append(
            LintFailure(
                file="melleafy.json",
                line=None,
                column=None,
                message=(
                    "melleafy.json top-level value is not a JSON object. "
                    f"Expected fields {sorted(_MELLEAFY_HARD_REQUIRED)} at minimum."
                ),
                rule_ref="Rule 5-1 sub-check A",
            )
        )
        return result

    present = set(data.keys())
    missing_hard = _MELLEAFY_HARD_REQUIRED - present
    if missing_hard:
        result.failures.append(
            LintFailure(
                file="melleafy.json",
                line=None,
                column=None,
                message=(
                    f"melleafy.json is missing hard-required field(s): "
                    f"{sorted(missing_hard)}. The export command halts "
                    f"without these. Required: "
                    f"{sorted(_MELLEAFY_HARD_REQUIRED)}."
                ),
                rule_ref="Rule 5-1 sub-check A",
            )
        )

    if "manifest_version" in present:
        mv_raw = data["manifest_version"]
        mv_tuple = (
            _parse_semver(mv_raw)
            if isinstance(mv_raw, (str, int, float))
            else None
        )
        if mv_tuple is None:
            result.failures.append(
                LintFailure(
                    file="melleafy.json",
                    line=None,
                    column=None,
                    message=(
                        f"melleafy.json:manifest_version = {mv_raw!r} does "
                        f"not parse as MAJOR.MINOR.PATCH semver."
                    ),
                    rule_ref="Rule 5-1 sub-check A",
                )
            )
        elif mv_tuple < _MIN_MANIFEST_VERSION:
            req = ".".join(str(x) for x in _MIN_MANIFEST_VERSION)
            actual = ".".join(str(x) for x in mv_tuple)
            result.failures.append(
                LintFailure(
                    file="melleafy.json",
                    line=None,
                    column=None,
                    message=(
                        f"melleafy.json:manifest_version = {actual} is below "
                        f"the minimum required version {req}."
                    ),
                    rule_ref="Rule 5-1 sub-check A",
                )
            )

    if result.failures:
        result.verdict = "fail"
        return result

    # Soft-warning entries (verdict stays pass)
    missing_soft = _MELLEAFY_SOFT_REQUIRED - present
    if missing_soft:
        result.failures.append(
            LintFailure(
                file="melleafy.json",
                line=None,
                column=None,
                message=(
                    f"WARN: melleafy.json is missing completeness field(s): "
                    f"{sorted(missing_soft)}. The export command does not "
                    f"read these, but a well-formed manifest should declare "
                    f"them."
                ),
                rule_ref="Rule 5-1 sub-check A (warning)",
            )
        )

    return result


# ─── Lint: instruct-result-parse-before-access (KB1) ───


_THUNK_CONSUMER_FUNC_NAMES: frozenset = frozenset({
    "_parse_instruct_result",
    "_safe_parse_with_fallback",
})


def _instruct_call_has_format_kwarg(node: ast.Call) -> bool:
    """True iff an `m.instruct(...)` Call has a `format=` keyword."""
    return any(kw.arg == "format" for kw in node.keywords)


def _iter_function_scopes(tree: ast.AST):
    """Yield every FunctionDef/AsyncFunctionDef + the module-level scope."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _consumer_thunk_arg_nodes(call: ast.Call, raw_thunks: Set[str]) -> Set[int]:
    """Return id(arg_node) set for thunk-Name args sitting in the first
    positional slot of a documented parser call."""
    fname = (
        call.func.id if isinstance(call.func, ast.Name)
        else call.func.attr if isinstance(call.func, ast.Attribute)
        else None
    )
    if fname not in _THUNK_CONSUMER_FUNC_NAMES:
        return set()
    if not call.args:
        return set()
    first = call.args[0]
    if isinstance(first, ast.Name) and first.id in raw_thunks:
        return {id(first)}
    return set()


def _is_model_validate_json_on_thunk_value(
    call: ast.Call, raw_thunks: Set[str]
) -> Optional[str]:
    """If call is `Model.model_validate_json(thunk.value)` for a known thunk."""
    func = call.func
    if not (
        isinstance(func, ast.Attribute) and func.attr == "model_validate_json"
    ):
        return None
    if len(call.args) != 1:
        return None
    arg = call.args[0]
    if not (isinstance(arg, ast.Attribute) and arg.attr == "value"):
        return None
    inner = arg.value
    if isinstance(inner, ast.Name) and inner.id in raw_thunks:
        return inner.id
    return None


def _stmt_assigns_from_consumer(
    stmt: ast.stmt, raw_thunks: Set[str]
) -> Optional[Tuple[str, ast.Call]]:
    """If stmt is `var = consumer(thunk, ...)`, return (var_name, call)."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    call = stmt.value
    if not isinstance(call, ast.Call):
        return None
    fname = (
        call.func.id if isinstance(call.func, ast.Name)
        else call.func.attr if isinstance(call.func, ast.Attribute)
        else None
    )
    if fname not in _THUNK_CONSUMER_FUNC_NAMES:
        return None
    if not call.args:
        return None
    first = call.args[0]
    if not (isinstance(first, ast.Name) and first.id in raw_thunks):
        return None
    return target.id, call


def _scan_scope_for_kb1(
    scope: ast.AST, rel: str, failures: List[LintFailure]
) -> None:
    """Walk one function body statement-by-statement, tracking raw thunks."""
    raw_thunks: Set[str] = set()
    body = getattr(scope, "body", [])

    def _process_block(stmts):
        nonlocal raw_thunks
        for stmt in stmts:
            new_thunk_name: Optional[str] = None
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
                and _is_m_instruct_call(stmt.value)
                and _instruct_call_has_format_kwarg(stmt.value)
            ):
                new_thunk_name = stmt.targets[0].id

            consumed = _stmt_assigns_from_consumer(stmt, raw_thunks)

            exempt_node_ids: Set[int] = set()
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    exempt_node_ids |= _consumer_thunk_arg_nodes(sub, raw_thunks)
                    thunk_in_mvj = _is_model_validate_json_on_thunk_value(
                        sub, raw_thunks
                    )
                    if thunk_in_mvj is not None:
                        arg = sub.args[0]
                        exempt_node_ids.add(id(arg))
                        exempt_node_ids.add(id(arg.value))

            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Attribute):
                    base = sub.value
                    if not (
                        isinstance(base, ast.Name) and base.id in raw_thunks
                    ):
                        continue
                    if id(sub) in exempt_node_ids:
                        continue
                    failures.append(
                        LintFailure(
                            file=rel,
                            line=getattr(sub, "lineno", None),
                            column=getattr(sub, "col_offset", None),
                            message=(
                                f"`{base.id}` is the result of "
                                f"`m.instruct(..., format=Model)` and is a "
                                f"`ComputedModelOutputThunk`, NOT a Pydantic "
                                f"model. Direct attribute access "
                                f"`{base.id}.{sub.attr}` raises AttributeError"
                                f" at runtime (KB1). Parse the thunk first: "
                                f"`parsed = _safe_parse_with_fallback("
                                f"{base.id}, <Model>, **defaults)` (preferred)"
                                f", `parsed = _parse_instruct_result("
                                f"{base.id}, <Model>)`, or `<Model>."
                                f"model_validate_json({base.id}.value)`."
                            ),
                            rule_ref="KB1 (instruct returns thunk)",
                        )
                    )

            if consumed is not None:
                consumed_var, _ = consumed
                call = stmt.value
                first_arg = call.args[0]
                if isinstance(first_arg, ast.Name):
                    raw_thunks.discard(first_arg.id)
                raw_thunks.discard(consumed_var)
            elif new_thunk_name is not None:
                raw_thunks.add(new_thunk_name)
            else:
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            raw_thunks.discard(tgt.id)

            for attr in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, attr, None)
                if isinstance(inner, list):
                    _process_block(inner)
            if isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    _process_block(handler.body)

    _process_block(body)


def lint_instruct_result_parse_before_access(package_dir: Path) -> LintResult:
    """KB1: `m.instruct(format=Model)` returns a thunk, not a Pydantic model.

    The thunk MUST be parsed before any field access. Exempt patterns:
      - `_parse_instruct_result(thunk, M)`
      - `_safe_parse_with_fallback(thunk, M, **defaults)`
      - `M.model_validate_json(thunk.value)`
    """
    result = LintResult(
        lint_id="instruct-result-parse-before-access", verdict="pass"
    )

    files_to_check: List[Path] = []
    for fname in _INSTRUCT_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue
        for scope in _iter_function_scopes(tree):
            _scan_scope_for_kb1(scope, rel, result.failures)

    # Dedup
    seen: Set[Tuple[str, Optional[int], Optional[int], str]] = set()
    deduped: List[LintFailure] = []
    for f in result.failures:
        key = (f.file, f.line, f.column, f.message[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    result.failures = deduped

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Lint: complex-schema-needs-strategy-or-fallback (KB2) ───


_KB2_COMPLEX_FIELD_THRESHOLD = 4


def _annotation_is_list(node: Optional[ast.expr]) -> bool:
    """True iff annotation is `list[...]` / `List[...]` / `Optional[list[...]]`."""
    if node is None:
        return False
    if isinstance(node, ast.Subscript):
        outer = node.value
        outer_name = (
            outer.id if isinstance(outer, ast.Name)
            else outer.attr if isinstance(outer, ast.Attribute)
            else None
        )
        if outer_name == "Optional":
            return _annotation_is_list(node.slice)
        if outer_name in ("list", "List"):
            return True
    return False


def _collect_complex_schemas(schemas_path: Path) -> dict:
    """Parse schemas.py and return {ClassName: complexity_info}."""
    schemas: dict = {}
    if not schemas_path.exists():
        return schemas
    try:
        tree = ast.parse(schemas_path.read_text(), filename=str(schemas_path))
    except SyntaxError:
        return schemas
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_basemodel = any(
            (isinstance(b, ast.Name) and b.id == "BaseModel")
            or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
            for b in node.bases
        )
        if not is_basemodel:
            continue
        field_count = 0
        has_list_field = False
        for stmt in node.body:
            if not (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            ):
                continue
            field_count += 1
            if _annotation_is_list(stmt.annotation):
                has_list_field = True
        is_complex = (
            field_count > _KB2_COMPLEX_FIELD_THRESHOLD or has_list_field
        )
        schemas[node.name] = {
            "field_count": field_count,
            "has_list_field": has_list_field,
            "is_complex": is_complex,
        }
    return schemas


def _format_kwarg_model_name(call: ast.Call) -> Optional[str]:
    for kw in call.keywords:
        if kw.arg != "format":
            continue
        v = kw.value
        if isinstance(v, ast.Name):
            return v.id
        if isinstance(v, ast.Attribute):
            return v.attr
        if isinstance(v, ast.Subscript):
            inner = v.slice
            if isinstance(inner, ast.Name):
                return inner.id
            if isinstance(inner, ast.Attribute):
                return inner.attr
        return None
    return None


def _call_has_strategy_kwarg(call: ast.Call) -> bool:
    return any(kw.arg == "strategy" for kw in call.keywords)


def _assignment_target_name(stmt: ast.stmt) -> Optional[str]:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _stmt_calls_safe_parse_with(thunk_name: str, stmt: ast.stmt) -> bool:
    """True iff stmt contains `_safe_parse_with_fallback(thunk_name, ...)`."""
    for sub in ast.walk(stmt):
        if not isinstance(sub, ast.Call):
            continue
        fname = (
            sub.func.id if isinstance(sub.func, ast.Name)
            else sub.func.attr if isinstance(sub.func, ast.Attribute)
            else None
        )
        if fname != "_safe_parse_with_fallback":
            continue
        if (
            sub.args
            and isinstance(sub.args[0], ast.Name)
            and sub.args[0].id == thunk_name
        ):
            return True
    return False


def lint_complex_schema_needs_strategy_or_fallback(
    package_dir: Path,
) -> LintResult:
    """KB2: Complex schemas in `m.instruct(format=Model)` need a strategy or fallback.

    Complex = >4 fields OR any list[...] / Optional[list[...]] field.
    Mitigation: `strategy=RepairTemplateStrategy(...)` on the call OR
    `_safe_parse_with_fallback(<thunk>, Model, **defaults)` after.
    """
    result = LintResult(
        lint_id="complex-schema-needs-strategy-or-fallback", verdict="pass"
    )

    schemas_path = package_dir / "schemas.py"
    schemas_map = _collect_complex_schemas(schemas_path)
    if not schemas_map:
        result.files_checked = 0
        return result

    files_to_check: List[Path] = []
    for fname in _INSTRUCT_LINT_FILES:
        p = package_dir / fname
        if p.exists():
            files_to_check.append(p)
    result.files_checked = len(files_to_check)

    for py_file in files_to_check:
        rel = py_file.relative_to(package_dir).as_posix()
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        scopes: List[ast.AST] = [tree]
        for sub in ast.walk(tree):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append(sub)

        for scope in scopes:
            body = getattr(scope, "body", [])
            for i, stmt in enumerate(body):
                # Skip FunctionDef bodies at the module-scope level — they
                # have their own scope iteration and otherwise would cause
                # `body[i+1:]` to be the wrong search horizon.
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                instruct_call: Optional[ast.Call] = None
                instruct_assign: Optional[ast.stmt] = None
                # Walk all Assign / AnnAssign descendants of stmt (including
                # those inside `with`, `try`, `if`, `for` bodies); pick the
                # first whose RHS is an `m.instruct(format=...)` call. This
                # lets us find both `thunk = m.instruct(...)` at the body
                # top-level AND the more common `with ...: thunk = m.instruct(...)`
                # nested pattern.
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                        rhs = sub.value
                        if (
                            isinstance(rhs, ast.Call)
                            and _is_m_instruct_call(rhs)
                            and _instruct_call_has_format_kwarg(rhs)
                        ):
                            instruct_call = rhs
                            instruct_assign = sub
                            break
                # Fallback: an unassigned `m.instruct(...)` Expression (rare).
                if instruct_call is None:
                    for sub in ast.walk(stmt):
                        if (
                            isinstance(sub, ast.Call)
                            and _is_m_instruct_call(sub)
                            and _instruct_call_has_format_kwarg(sub)
                        ):
                            instruct_call = sub
                            break
                if instruct_call is None:
                    continue
                model_name = _format_kwarg_model_name(instruct_call)
                if model_name is None:
                    continue
                info = schemas_map.get(model_name)
                if info is None or not info["is_complex"]:
                    continue
                if _call_has_strategy_kwarg(instruct_call):
                    continue

                # Get the thunk name from the actual Assign (may live inside
                # a `with` body); fall back to the outer-body statement if
                # somehow that fails.
                thunk_name = (
                    _assignment_target_name(instruct_assign)
                    if instruct_assign is not None
                    else _assignment_target_name(stmt)
                )
                satisfied = False
                if thunk_name is not None:
                    for follow in body[i + 1 :]:
                        if _stmt_calls_safe_parse_with(thunk_name, follow):
                            satisfied = True
                            break
                if satisfied:
                    continue

                reasons: List[str] = []
                if info["field_count"] > _KB2_COMPLEX_FIELD_THRESHOLD:
                    reasons.append(
                        f"{info['field_count']} fields "
                        f"(> {_KB2_COMPLEX_FIELD_THRESHOLD})"
                    )
                if info["has_list_field"]:
                    reasons.append("at least one list[...]-typed field")
                reason_str = " and ".join(reasons) or "complex schema"

                result.failures.append(
                    LintFailure(
                        file=rel,
                        line=getattr(instruct_call, "lineno", None),
                        column=getattr(instruct_call, "col_offset", None),
                        message=(
                            f"`m.instruct(..., format={model_name})` targets "
                            f"a complex schema ({reason_str}) but neither "
                            f"uses `strategy=RepairTemplateStrategy(...)` nor "
                            f"has its result parsed via "
                            f"`_safe_parse_with_fallback(<thunk>, "
                            f"{model_name}, **defaults)` in the same function"
                            f" scope. KB2: complex schemas truncate at "
                            f"`max_tokens` and raise `ValidationError: EOF "
                            f"while parsing` on plain `model_validate_json`."
                        ),
                        rule_ref="KB2 (JSON truncation on complex outputs)",
                    )
                )

    seen: Set[Tuple[str, Optional[int], Optional[int]]] = set()
    deduped: List[LintFailure] = []
    for f in result.failures:
        key = (f.file, f.line, f.column)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    result.failures = deduped

    if result.failures:
        result.verdict = "fail"
    return result


# ─── Runner ───


ALL_LINTS: Tuple[Callable[[Path], LintResult], ...] = (
    lint_fixtures_loader_contract,
    lint_bundled_asset_path_resolution,
    lint_runtime_defaults_bound,
    lint_instruct_has_description,
    lint_pipeline_entry_canonical,
    lint_fixture_signature_bound,
    lint_generative_forbidden_params,
    lint_generative_call_passes_session,
    lint_validator_soundness,
    lint_session_boundary,
    lint_prefix_persona,
    lint_no_watsonx,
    lint_optional_extraction_guidance,
    lint_run_pipeline_params_typed,
    lint_import_soundness,
    lint_stdlib_arity,
    lint_melleafy_json_consistency,
    lint_instruct_result_parse_before_access,
    lint_complex_schema_needs_strategy_or_fallback,
)


def run_lints(package_dir: Path) -> LintRunResult:
    """Run all implemented Step 7 structural lints; write step_7_report.json."""
    results: List[LintResult] = [lint_fn(package_dir) for lint_fn in ALL_LINTS]
    overall = "fail" if any(r.verdict == "fail" for r in results) else "pass"
    run_result = LintRunResult(
        overall_verdict=overall,
        lints=results,
        package_path=str(package_dir),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    intermediate = package_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": "1.0",
        "checked_at": run_result.checked_at,
        "package_path": run_result.package_path,
        "overall_verdict": run_result.overall_verdict,
        "lints": [
            {
                "lint_id": r.lint_id,
                "verdict": r.verdict,
                "files_checked": r.files_checked,
                "skipped_reason": r.skipped_reason,
                "failures": [asdict(f) for f in r.failures],
            }
            for r in results
        ],
    }
    (intermediate / "step_7_report.json").write_text(json.dumps(report, indent=2))

    return run_result
