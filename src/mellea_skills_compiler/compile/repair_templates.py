"""Per-lint repair templates (F1 follow-up).

Each ``LintFailure`` emitted by ``compile.lints`` has a well-defined fix
prescription. Today the repair loop in ``.claude/commands/mellea-fy-repair.md``
re-invokes ``/mellea-fy-generate`` (repair mode) with a generic "fix these
lint failures" instruction; Claude has to re-derive the fix on every round.

The templates here render structured, copy-paste-grade fix prescriptions
from a single ``LintFailure`` plus optional grounding (the package's
``intermediate/mellea_api_ref.json`` surface) so the repair prompt can
include the exact "where to change what, in which direction" without
Claude having to infer it.

Wave 1 (this module): the four lints that are blocking real skills in
the overnight batch:

  - ``runtime-defaults-bound``           (Rule C8 / OUT-8)
  - ``bundled-asset-path-resolution``    (Rule OUT-6)
  - ``stdlib-arity``                     (Rule 4-4)
  - ``instruct-result-parse-before-access`` (KB1)

Wave 2 will add the remaining ERROR-severity lints. ``get_repair_template``
returns ``None`` for unknown / not-yet-templated lints so the caller can
fall back to today's generic repair prompt without breaking.

NOTE — this module deliberately does NOT call into the repair loop. F1
integration (wiring these templates into the orchestrator that builds the
Claude repair prompt) is a separate task; see the forward-reference note
in ``.claude/commands/mellea-fy-repair.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from mellea_skills_compiler.compile.lints import LintFailure


# ─── Helpers ───────────────────────────────────────────────────────────


def _load_surface_json(package_dir: Optional[Path]) -> Optional[dict]:
    """Read ``intermediate/mellea_api_ref.json`` from a compiled package.

    Returns ``None`` if the file is absent, unreadable, or marked
    ``grounding_unavailable``. Callers may also pass an explicit
    ``surface_json=`` kwarg to ``get_repair_template`` to bypass this.
    """
    if package_dir is None:
        return None
    p = Path(package_dir) / "intermediate" / "mellea_api_ref.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("grounding_unavailable", False):
        return None
    return data


def _lookup_canonical_signature(
    surface_json: Optional[dict], symbol: str
) -> Optional[str]:
    """Find ``signature`` for ``symbol`` across all ``mellea.stdlib.*`` modules.

    Matches by bare name (e.g. ``check``, ``req``, ``simple_validate``).
    Returns the first signature found, or ``None``.
    """
    if not surface_json or not isinstance(surface_json, dict):
        return None
    modules = surface_json.get("modules", {})
    if not isinstance(modules, dict):
        return None
    for mod_name, mod_symbols in modules.items():
        if not isinstance(mod_name, str) or not mod_name.startswith(
            "mellea.stdlib."
        ):
            continue
        if not isinstance(mod_symbols, dict):
            continue
        sym_meta = mod_symbols.get(symbol)
        if isinstance(sym_meta, dict):
            sig = sym_meta.get("signature")
            if isinstance(sig, str):
                return sig
    return None


def _extract_callee_from_arity_message(message: str) -> Optional[str]:
    """Parse the symbol name out of a stdlib-arity failure message.

    The lint emits messages shaped like ::

        `check(...)` called with 1 positional argument(s); ...

    Returns ``"check"`` (or ``None`` if the message doesn't match).
    """
    # Cheap, deterministic: split on first backtick.
    if "`" not in message:
        return None
    after = message.split("`", 1)[1]
    if "(" not in after:
        return None
    name = after.split("(", 1)[0].strip()
    return name or None


def _extract_thunk_var_from_kb1_message(message: str) -> Optional[str]:
    """Parse the thunk variable name out of an instruct-result-parse-before-access message.

    Messages are shaped::

        `thunk` is the result of `m.instruct(..., format=Model)` and ...

    Returns ``"thunk"`` (or ``None``).
    """
    if "`" not in message:
        return None
    after = message.split("`", 1)[1]
    if "`" not in after:
        return None
    return after.split("`", 1)[0].strip() or None


# ─── Renderers ─────────────────────────────────────────────────────────


def _render_runtime_defaults_bound(
    failure: LintFailure,
    *,
    package_dir: Optional[Path] = None,
    surface_json: Optional[dict] = None,
) -> str:
    line = failure.line if failure.line is not None else "—"
    return (
        f"## Fix prescription for `runtime-defaults-bound`\n"
        f"\n"
        f"File: `{failure.file}`  \n"
        f"Line: {line}  \n"
        f"Issue: {failure.message}\n"
        f"\n"
        f"### Required change\n"
        f"\n"
        f"The `config.py` BACKEND and MODEL_ID constants MUST match the values in\n"
        f"`intermediate/runtime_directive.json`. Read that file and replicate its\n"
        f"`backend` and `model_id` values verbatim into the corresponding `config.py`\n"
        f"constants.\n"
        f"\n"
        f"If `config.py` does not exist at all, the deterministic writer\n"
        f"(`.claude/melleafy/writers/config_writer.py`, Step 3 ENFORCE mode) did\n"
        f"not run. Re-run Step 3 via `/mellea-fy-generate` with the existing\n"
        f"intermediate artefacts as grounding; do NOT hand-write `config.py`.\n"
        f"\n"
        f"### Verification\n"
        f"\n"
        f"After editing, run `lint_runtime_defaults_bound(package_dir)` against the\n"
        f"package — verdict must be `pass`.\n"
        f"\n"
        f"### Reference\n"
        f"\n"
        f"Rule OUT-8 / C8 (runtime defaults bound). The wrapper injects these values\n"
        f"via the system prompt; `intermediate/runtime_directive.json` is the\n"
        f"authoritative record of what was injected for this specific compile.\n"
    )


def _render_bundled_asset_path_resolution(
    failure: LintFailure,
    *,
    package_dir: Optional[Path] = None,
    surface_json: Optional[dict] = None,
) -> str:
    line = failure.line if failure.line is not None else "—"
    return (
        f"## Fix prescription for `bundled-asset-path-resolution`\n"
        f"\n"
        f"File: `{failure.file}`  \n"
        f"Line: {line}  \n"
        f"Issue: {failure.message}\n"
        f"\n"
        f"### Required change\n"
        f"\n"
        f"Bundled assets under `<package_dir>/scripts/`, `<package_dir>/references/`,\n"
        f"or `<package_dir>/assets/` MUST be resolved via:\n"
        f"\n"
        f"```python\n"
        f"Path(__file__).parent / \"<dir>\" / \"<file>\"\n"
        f"```\n"
        f"\n"
        f"NOT via `Path(repo_root) / ...`, NOT via a function argument named\n"
        f"`repo_root`, NOT via `os.getcwd()`, NOT via an `_PKG_DIR` constant that\n"
        f"was itself derived from anything other than `Path(__file__).parent`.\n"
        f"Rule OUT-6.\n"
        f"\n"
        f"### Refactor pattern\n"
        f"\n"
        f"```python\n"
        f"# WRONG (will FileNotFoundError on pip-install):\n"
        f"ref_text = (Path(repo_root) / \"references\" / \"owasp.md\").read_text()\n"
        f"\n"
        f"# RIGHT (works in-repo AND pip-installed):\n"
        f"_REFERENCES_DIR = Path(__file__).parent / \"references\"\n"
        f"ref_text = (_REFERENCES_DIR / \"owasp.md\").read_text()\n"
        f"```\n"
        f"\n"
        f"### Verification\n"
        f"\n"
        f"After editing, run `lint_bundled_asset_path_resolution(package_dir)`.\n"
        f"\n"
        f"### Reference\n"
        f"\n"
        f"Rule OUT-6 (companion-directory mirror, package-relative path resolution)\n"
        f"and Rule 2.5-2 (`mellea-fy-deps.md`).\n"
    )


def _render_stdlib_arity(
    failure: LintFailure,
    *,
    package_dir: Optional[Path] = None,
    surface_json: Optional[dict] = None,
) -> str:
    line = failure.line if failure.line is not None else "—"
    symbol = _extract_callee_from_arity_message(failure.message) or "<symbol>"

    # Prefer an explicit surface_json kwarg; fall back to loading it from
    # package_dir if not provided.
    sj = surface_json if surface_json is not None else _load_surface_json(
        package_dir
    )
    canonical = _lookup_canonical_signature(sj, symbol) if symbol else None
    if canonical:
        canonical_block = canonical
    else:
        canonical_block = (
            "<could not be resolved from intermediate/mellea_api_ref.json; "
            "consult the file directly under "
            ".modules.mellea.stdlib.<...> for the canonical signature>"
        )

    return (
        f"## Fix prescription for `stdlib-arity`\n"
        f"\n"
        f"File: `{failure.file}`  \n"
        f"Line: {line}  \n"
        f"Issue: {failure.message}\n"
        f"\n"
        f"### Canonical signature (from `intermediate/mellea_api_ref.json`)\n"
        f"\n"
        f"The Mellea symbol `{symbol}` has signature:\n"
        f"\n"
        f"```\n"
        f"{canonical_block}\n"
        f"```\n"
        f"\n"
        f"### Required change\n"
        f"\n"
        f"Compare your call's positional/keyword arguments against the canonical\n"
        f"signature above. Common fixes:\n"
        f"\n"
        f"- A keyword-only argument was passed positionally → make it a keyword\n"
        f"- An extra argument was passed → remove it\n"
        f"- A required argument is missing → add it (consult the docstring for\n"
        f"  expected meaning)\n"
        f"- For `check(...)`: pass `(requirement, ctx)` — two positional args.\n"
        f"  A single-arg call is the most-fired arity drift in the overnight batch.\n"
        f"\n"
        f"### Verification\n"
        f"\n"
        f"After editing, run `lint_stdlib_arity(package_dir)`. The surface JSON\n"
        f"under `mellea_api_ref.json:.modules.<...>.{symbol}` is the authoritative\n"
        f"signature record.\n"
        f"\n"
        f"### Reference\n"
        f"\n"
        f"Mellea 0.5 introspected signatures are version-pinned. Drift from the\n"
        f"surface JSON indicates the LLM hallucinated arity from training data\n"
        f"rather than reading the canonical reference.\n"
    )


def _render_instruct_result_parse_before_access(
    failure: LintFailure,
    *,
    package_dir: Optional[Path] = None,
    surface_json: Optional[dict] = None,
) -> str:
    line = failure.line if failure.line is not None else "—"
    thunk_var = _extract_thunk_var_from_kb1_message(failure.message) or "thunk"
    return (
        f"## Fix prescription for `instruct-result-parse-before-access`\n"
        f"\n"
        f"File: `{failure.file}`  \n"
        f"Line: {line}  \n"
        f"Issue: {failure.message}\n"
        f"\n"
        f"### Required change\n"
        f"\n"
        f"`m.instruct(..., format=Model)` returns a `ModelOutputThunk`. Accessing\n"
        f"attributes on the thunk directly (e.g. `{thunk_var}.severity`) raises\n"
        f"`AttributeError`. You must first parse the thunk via:\n"
        f"\n"
        f"```python\n"
        f"parsed = Model.model_validate_json({thunk_var}.value)\n"
        f"```\n"
        f"\n"
        f"Then access fields on `parsed`, not on `{thunk_var}`.\n"
        f"\n"
        f"### Refactor pattern\n"
        f"\n"
        f"```python\n"
        f"# WRONG (will AttributeError):\n"
        f"{thunk_var} = m.instruct(..., format=ScanResult)\n"
        f"severity = {thunk_var}.severity   # ← AttributeError\n"
        f"\n"
        f"# RIGHT (preferred — uses the shared helper that handles fallbacks):\n"
        f"{thunk_var} = m.instruct(..., format=ScanResult)\n"
        f"result = _safe_parse_with_fallback({thunk_var}, ScanResult, **defaults)\n"
        f"severity = result.severity   # ← OK\n"
        f"\n"
        f"# RIGHT (explicit — when you want a hard failure on parse error):\n"
        f"{thunk_var} = m.instruct(..., format=ScanResult)\n"
        f"result = ScanResult.model_validate_json({thunk_var}.value)\n"
        f"severity = result.severity   # ← OK\n"
        f"```\n"
        f"\n"
        f"### Verification\n"
        f"\n"
        f"After editing, run `lint_instruct_result_parse_before_access(package_dir)`.\n"
        f"\n"
        f"### Reference\n"
        f"\n"
        f"KB1. `mellea.stdlib.session.MelleaSession.instruct` returns\n"
        f"`ModelOutputThunk[str]`, not a `BaseModel`. The `format=` kwarg constrains\n"
        f"the LLM's output to match the schema, but parsing is the caller's\n"
        f"responsibility.\n"
    )


# ─── Registry + public entry point ─────────────────────────────────────


_TEMPLATES: Dict[str, Callable[..., str]] = {
    "runtime-defaults-bound": _render_runtime_defaults_bound,
    "bundled-asset-path-resolution": _render_bundled_asset_path_resolution,
    "stdlib-arity": _render_stdlib_arity,
    "instruct-result-parse-before-access": _render_instruct_result_parse_before_access,
}


def get_repair_template(
    lint_id: str,
    failure: LintFailure,
    *,
    package_dir: Optional[Path] = None,
    surface_json: Optional[dict] = None,
) -> Optional[str]:
    """Return a rendered fix prescription for the given lint failure.

    Parameters
    ----------
    lint_id:
        The lint identifier (matches ``LintResult.lint_id``).
    failure:
        A single ``LintFailure`` entry from ``LintResult.failures``.
    package_dir:
        Optional path to the compiled package. When provided, templates
        that need grounding (currently only ``stdlib-arity``) read
        ``<package_dir>/intermediate/mellea_api_ref.json`` to look up
        canonical signatures.
    surface_json:
        Optional already-loaded surface JSON. Takes precedence over
        ``package_dir`` for tests / callers that want to control grounding
        explicitly.

    Returns
    -------
    Rendered template string (markdown-formatted), or ``None`` if no
    template is registered for this ``lint_id``. A ``None`` return means
    "fall back to the generic repair prompt" — it is the caller's
    responsibility to handle that.
    """
    fn = _TEMPLATES.get(lint_id)
    if fn is None:
        return None
    return fn(
        failure,
        package_dir=package_dir,
        surface_json=surface_json,
    )


__all__ = ["get_repair_template"]
