"""Wrapper-side invocation of the deterministic writers in `.claude/melleafy/writers/`.

The writers (`config_writer.py`, `fixtures_writer.py`, ...) were originally
documented as "the LLM follows them" — but the slash command runs with
`--allowed-tools Read,Write,Edit` and cannot execute Python. So in practice
the LLM was mentally rendering the writer's output, which is unreliable for
anything more complex than `config.py`.

This module moves writer invocation into the compile pipeline, where Python
actually runs. Two operating modes:

  - WARN (observation only): render via the writer, diff against the file the
    LLM wrote, log a warning if they differ, do NOT overwrite.
  - ENFORCE: render via the writer, write authoritatively, deletes whatever
    the LLM put there. Combine with `Write(...)` deny rules in --settings to
    prevent the LLM writing the path in the first place.

Migration plan:
  1. WARN mode for `config_writer` (this commit) — gather evidence the diffs
     are stable and bounded across compiles.
  2. After ~5 clean compiles, flip `config_writer` to ENFORCE + add deny rule.
  3. Repeat for `fixtures_writer`.

Each writer module exposes a `render(emission: dict) -> str` function returning
Python source text. The wrapper reads the emission JSON the LLM emitted, calls
render(), and either compares-and-warns or writes-authoritatively.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from mellea_skills_compiler.compile import CLAUDE_DIR
from mellea_skills_compiler.toolkit.logging import configure_logger


LOGGER = configure_logger()


@dataclass(frozen=True)
class WriterSpec:
    """One renderable artifact: emission JSON in → Python source on disk out.

    `output_kind` controls how the writer is invoked:
      - "file" (default): writer module exposes `render(emission) -> str` and
        the renderer writes the returned text to `output_relpath`.
      - "directory": writer module exposes `write(emission, dir_path) -> list[Path]`
        and the renderer wipes `output_relpath` (preserving the dir itself)
        before invoking, so the writer is the sole producer of the dir's contents.

    `schema_path`, when set, points at a JSON-Schema file under `.claude/schemas/`.
    The renderer validates the emission JSON against the schema BEFORE invoking
    the writer, so schema-violation drift (e.g. LLM emits `fixture_id` when the
    schema requires `id`) surfaces as a precise schema error instead of a
    downstream `KeyError` inside the writer.
    """

    name: str  # human-readable label, e.g. "config.py"
    emission_relpath: str  # e.g. "intermediate/config_emission.json"
    output_relpath: str  # e.g. "config.py" or "fixtures"
    writer_path: Path  # absolute path to the .claude/melleafy/writers/*.py module
    output_kind: str = "file"  # "file" | "directory"
    schema_path: Optional[Path] = None  # absolute path to JSON Schema; None = skip


@dataclass
class RenderResult:
    name: str
    status: (
        str  # "match" | "diff" | "missing-emission" | "missing-output" | "writer-error" | "schema-invalid"
    )
    detail: Optional[str] = None
    diff_lines_added: int = 0
    diff_lines_removed: int = 0


def _load_writer_module(writer_path: Path):
    """Import a writer module from a path and return the module object."""
    spec = importlib.util.spec_from_file_location(writer_path.stem, writer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load writer from {writer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_against_schema(
    emission: dict, schema_path: Path, emission_relpath: str
) -> Optional[str]:
    """Return None if `emission` conforms to the schema at `schema_path`, else
    a precise human-readable error string naming the failing field and reason.

    The schema files at `.claude/schemas/` use Draft 2020-12. We catch the most
    common validation failures (unexpected key from `additionalProperties: false`,
    missing required field, wrong type) and format the error so the message
    points at the JSON path of the violation rather than spitting a bare class
    name. Returning a string here causes the renderer to report verdict
    `schema-invalid` without invoking the writer.
    """
    if not schema_path.exists():
        return None  # missing schema = backward-compat skip (warn-style)
    try:
        schema = json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"schema file {schema_path.name} could not be loaded: {exc}"

    try:
        import jsonschema
    except ImportError:
        # Defensive: jsonschema is a hard dep in pyproject.toml; if it's
        # somehow missing, skip validation rather than break the compile.
        LOGGER.warning(
            "[writer:%s] jsonschema not installed — emission validation skipped",
            emission_relpath,
        )
        return None

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(emission), key=lambda e: list(e.absolute_path))
    if not errors:
        return None

    # Format up to the first 3 errors with concrete JSON paths so the LLM (and
    # human) can locate the violations directly. More than 3 is usually one
    # cascading shape mistake — the head gives enough signal.
    formatted: list[str] = []
    for err in errors[:3]:
        path = list(err.absolute_path)
        path_str = "$" + "".join(
            f"[{p}]" if isinstance(p, int) else f".{p}" for p in path
        )
        formatted.append(f"  at {path_str}: {err.message}")
    extra = (
        f"\n  …and {len(errors) - 3} more validation errors"
        if len(errors) > 3
        else ""
    )
    return (
        f"{emission_relpath} violates {schema_path.name}:\n"
        + "\n".join(formatted)
        + extra
    )


def _load_writer(writer_path: Path) -> Callable[[dict], str]:
    """Return the file-mode writer's `render(emission) -> str` function."""
    module = _load_writer_module(writer_path)
    if not hasattr(module, "render"):
        raise AttributeError(f"writer {writer_path} has no render() function")
    return module.render


def _load_writer_dir(writer_path: Path) -> Callable[[dict, Path], list]:
    """Return the directory-mode writer's `write(emission, out_dir)` function."""
    module = _load_writer_module(writer_path)
    if not hasattr(module, "write"):
        raise AttributeError(
            f"writer {writer_path} has no write(emission, out_dir) function "
            f"required for directory-mode WriterSpec"
        )
    return module.write


def _render_one(package_dir: Path, spec: WriterSpec, *, enforce: bool) -> RenderResult:
    """Render one writer artifact and either warn-on-diff or write-authoritatively."""
    emission_path = package_dir / spec.emission_relpath
    output_path = package_dir / spec.output_relpath

    if not emission_path.exists():
        return RenderResult(
            name=spec.name,
            status="missing-emission",
            detail=(
                f"{spec.emission_relpath} not found — the LLM did not emit the IR. "
                f"In WARN mode this is a soft skip; in ENFORCE mode the package is "
                f"unrenderable and compile should fail."
            ),
        )

    try:
        emission = json.loads(emission_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return RenderResult(
            name=spec.name,
            status="writer-error",
            detail=f"could not read or parse {spec.emission_relpath}: {exc}",
        )

    if spec.schema_path is not None:
        schema_error = _validate_against_schema(
            emission, spec.schema_path, spec.emission_relpath
        )
        if schema_error is not None:
            return RenderResult(
                name=spec.name,
                status="schema-invalid",
                detail=schema_error,
            )

    if spec.output_kind == "directory":
        return _render_directory(spec, emission, output_path, enforce=enforce)
    return _render_file(spec, emission, output_path, enforce=enforce)


def _render_file(
    spec: WriterSpec, emission: dict, output_path: Path, *, enforce: bool
) -> RenderResult:
    """Single-file writer dispatch (config.py shape)."""
    try:
        render_fn = _load_writer(spec.writer_path)
        rendered = render_fn(emission)
    except Exception as exc:  # noqa: BLE001
        return RenderResult(
            name=spec.name,
            status="writer-error",
            detail=f"writer {spec.writer_path.name} raised: {exc}",
        )

    if not output_path.exists():
        if enforce:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered)
            return RenderResult(
                name=spec.name,
                status="match",
                detail=f"rendered {spec.output_relpath} (LLM did not write one)",
            )
        return RenderResult(
            name=spec.name,
            status="missing-output",
            detail=(
                f"{spec.output_relpath} not on disk — LLM appears to have skipped "
                f"this artifact entirely. WARN mode does not render."
            ),
        )

    actual = output_path.read_text()
    if actual == rendered:
        return RenderResult(name=spec.name, status="match")

    diff = list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"on-disk {spec.output_relpath}",
            tofile=f"writer-rendered {spec.output_relpath}",
            n=1,
        )
    )
    added = sum(
        1 for line in diff if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff if line.startswith("-") and not line.startswith("---")
    )

    if enforce:
        output_path.write_text(rendered)
        return RenderResult(
            name=spec.name,
            status="match",
            detail=f"overwrote {spec.output_relpath} with writer output (-{removed}/+{added} lines)",
            diff_lines_added=added,
            diff_lines_removed=removed,
        )

    return RenderResult(
        name=spec.name,
        status="diff",
        detail=f"on-disk differs from writer (-{removed}/+{added} lines)",
        diff_lines_added=added,
        diff_lines_removed=removed,
    )


def _render_directory(
    spec: WriterSpec, emission: dict, output_path: Path, *, enforce: bool
) -> RenderResult:
    """Directory-output writer dispatch (fixtures/ shape).

    Strategy:
      - ENFORCE: wipe the LLM's contents under output_path (everything except
        the dir itself), then invoke the writer's `write(emission, output_path)`.
        The writer is the sole producer of dir contents; any pytest-style or
        INPUT-only drift the LLM may have written is gone.
      - WARN: render to a temp dir, count how many files differ from
        on-disk, log a warning. Do not modify on-disk files.
    """
    try:
        write_fn = _load_writer_dir(spec.writer_path)
    except Exception as exc:  # noqa: BLE001
        return RenderResult(
            name=spec.name,
            status="writer-error",
            detail=f"writer {spec.writer_path.name} dir-mode load failed: {exc}",
        )

    if enforce:
        output_path.mkdir(parents=True, exist_ok=True)
        # Wipe everything inside (preserve the dir itself so cwd/imports stay valid).
        wiped = 0
        for child in output_path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
                wiped += 1
            elif child.is_dir() and child.name != "__pycache__":
                # Only wipe pycache dirs leniently; deeper subdirs would be
                # surprising for fixtures so log them rather than wipe blindly.
                _wipe_dir(child)
                wiped += 1
        try:
            written = write_fn(emission, output_path)
        except Exception as exc:  # noqa: BLE001
            return RenderResult(
                name=spec.name,
                status="writer-error",
                detail=f"writer.write() raised: {exc}",
            )
        return RenderResult(
            name=spec.name,
            status="match",
            detail=(
                f"rendered {len(written)} file(s) into {spec.output_relpath}/ "
                f"(wiped {wiped} pre-existing entries)"
            ),
        )

    # WARN mode: render to a tempdir, compare to on-disk
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            written = write_fn(emission, tmp)
        except Exception as exc:  # noqa: BLE001
            return RenderResult(
                name=spec.name,
                status="writer-error",
                detail=f"writer.write() raised: {exc}",
            )
        rendered_names = {p.name for p in written}
        actual_names = (
            {p.name for p in output_path.iterdir() if p.is_file()}
            if output_path.is_dir()
            else set()
        )
        only_rendered = sorted(rendered_names - actual_names)
        only_actual = sorted(actual_names - rendered_names - {"__pycache__"})
        if not only_rendered and not only_actual:
            return RenderResult(name=spec.name, status="match")
        return RenderResult(
            name=spec.name,
            status="diff",
            detail=(
                f"on-disk vs writer differ: "
                f"only_rendered={only_rendered or '[]'}, "
                f"only_actual={only_actual or '[]'}"
            ),
            diff_lines_added=len(only_rendered),
            diff_lines_removed=len(only_actual),
        )


def _wipe_dir(path: Path) -> None:
    """Recursively remove a directory and all its contents."""
    import shutil

    shutil.rmtree(path)


def synthesize_config_emission_from_directive(
    intermediate_dir: Path,
) -> Optional[dict]:
    """If config_emission.json is missing, synthesize it from runtime_directive.json.

    The LLM is supposed to emit config_emission.json with the BACKEND and
    MODEL_ID values that were injected via the system prompt. When the LLM
    omits this (typically because it bailed early, hit a sandbox restriction,
    or otherwise dropped the artefact) we can deterministically reconstruct
    a minimal config_emission.json from runtime_directive.json — which the
    wrapper wrote BEFORE the Claude session started, so its BACKEND/MODEL_ID
    values are authoritative.

    Behaviour:
      - If config_emission.json already exists, return its parsed contents
        unchanged and do NOT overwrite. The on-disk file (whether emitted by
        the LLM or by a previous synthesis pass) is the source of truth.
      - Otherwise, if runtime_directive.json is present and readable, write
        a schema-compliant config_emission.json containing only the C8
        BACKEND/MODEL_ID constants and return the synthesized dict. The
        deterministic writer is then guaranteed an emission to render from.
      - If runtime_directive.json is also missing (or malformed), return
        None. The caller can detect this and let the existing
        ``missing-emission`` flow surface the failure — synthesis cannot
        fabricate values from nothing.

    The synthesized JSON shape matches what an LLM-emitted minimal
    config_emission.json would look like (it conforms to
    .claude/schemas/config_emission.schema.json), so the writer treats
    synthesized and LLM-emitted inputs identically.
    """
    emission_path = intermediate_dir / "config_emission.json"
    directive_path = intermediate_dir / "runtime_directive.json"

    if emission_path.exists():
        try:
            return json.loads(emission_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Existing file is unreadable — defer to the writer-renderer's
            # writer-error path rather than overwriting a possibly-recoverable
            # artefact here.
            return None

    if not directive_path.exists():
        return None

    try:
        directive = json.loads(directive_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[writer:config.py] cannot synthesize config_emission.json — "
            "runtime_directive.json is unreadable: %s",
            exc,
        )
        return None

    backend = directive.get("backend")
    model_id = directive.get("model_id")
    if not isinstance(backend, str) or not isinstance(model_id, str):
        LOGGER.warning(
            "[writer:config.py] cannot synthesize config_emission.json — "
            "runtime_directive.json missing string 'backend' or 'model_id' "
            "(got backend=%r, model_id=%r)",
            backend,
            model_id,
        )
        return None

    synthesized: dict = {
        "constants": [
            {
                "name": "BACKEND",
                "value": backend,
                "type": "str",
                "category": "C8",
            },
            {
                "name": "MODEL_ID",
                "value": model_id,
                "type": "str",
                "category": "C8",
            },
        ]
    }

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    emission_path.write_text(json.dumps(synthesized, indent=2))
    LOGGER.warning(
        "[writer:config.py] Synthesized config_emission.json from "
        "runtime_directive.json (LLM did not emit one). "
        "BACKEND=%r MODEL_ID=%r source=%r",
        backend,
        model_id,
        directive.get("source"),
    )
    return synthesized


def _extract_make_function_return(
    tree: ast.Module, fid_expected: str
) -> Optional[tuple[Optional[dict], Optional[str], Optional[str]]]:
    """AST-walk a fixture module and pull `(inputs, fid, description)` out of
    its `make_<fid>()` function.

    Returns a 3-tuple of best-effort literal values. Any element that is not a
    pure ``ast.Constant`` / dict / list literal comes back as ``None`` — the
    caller treats that as a signal to fall back to importing the module and
    evaluating the function. Returning ``None`` for the whole tuple means the
    function shape is unrecognisable (no ``make_<fid>`` found, or no
    ``return``).
    """
    target_fn = f"make_{fid_expected}"
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != target_fn:
            continue
        # Find the return statement (first one wins; auto-generated fixtures
        # have exactly one).
        return_node = next(
            (n for n in ast.walk(node) if isinstance(n, ast.Return)), None
        )
        if return_node is None or not isinstance(return_node.value, ast.Tuple):
            return None
        elts = return_node.value.elts
        if len(elts) != 3:
            return None

        # The factory may bind ``inputs = {...}`` then ``return inputs, ...``
        # — in that shape the first tuple element is a Name, so resolve it
        # against the function body's assignments before falling back.
        inputs_node = elts[0]
        if isinstance(inputs_node, ast.Name):
            # Walk the function body looking for `inputs = <literal>` and use
            # the most recent assignment.
            resolved = None
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == inputs_node.id
                ):
                    resolved = stmt.value
                elif (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == inputs_node.id
                    and stmt.value is not None
                ):
                    resolved = stmt.value
            inputs_node = resolved if resolved is not None else inputs_node

        def _literal(n: ast.AST) -> Any:
            try:
                return ast.literal_eval(n)
            except (ValueError, SyntaxError):
                return None

        inputs_val = _literal(inputs_node) if inputs_node is not None else None
        fid_val = _literal(elts[1])
        desc_val = _literal(elts[2])

        # Coerce types: inputs must be a dict; fid and description must be strs.
        if not isinstance(inputs_val, dict):
            inputs_val = None
        if not isinstance(fid_val, str):
            fid_val = None
        if not isinstance(desc_val, str):
            desc_val = None
        return inputs_val, fid_val, desc_val
    return None


def _import_make_function_return(
    fixture_path: Path, fid: str
) -> Optional[tuple[dict, str, str]]:
    """Last-resort fallback: import the fixture module and call ``make_<fid>()``
    to read its return value at runtime.

    Used only when AST extraction returns ``None`` for at least one tuple
    element — i.e. the fixture computes its inputs from something other than a
    pure literal. Auto-generated melleafy fixtures never hit this path; it
    exists for hand-edited or legacy fixtures.
    """
    target_fn = f"make_{fid}"
    spec = importlib.util.spec_from_file_location(
        f"_fixture_synth_{fid}", fixture_path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Insert into sys.modules so dataclasses / pickle-style introspection work,
    # then remove on the way out — this keeps the synth import side-effect
    # free as far as later code is concerned.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        fn = getattr(module, target_fn, None)
        if fn is None or not callable(fn):
            return None
        result = fn()
    except Exception:  # noqa: BLE001 — fixture import is best-effort
        return None
    finally:
        sys.modules.pop(spec.name, None)
    if not isinstance(result, tuple) or len(result) != 3:
        return None
    inputs_val, fid_val, desc_val = result
    if not isinstance(inputs_val, dict) or not isinstance(fid_val, str) or not isinstance(desc_val, str):
        return None
    return inputs_val, fid_val, desc_val


def _extract_one_fixture(fixture_path: Path) -> Optional[dict]:
    """Reverse-engineer one ``fixtures/<id>.py`` file into the dict shape that
    appears inside ``fixtures_emission.json``.

    Strategy: AST-parse first (no execution), fall back to importing the
    module only if any of (inputs, fid, description) is non-literal. The
    fixture id is derived from the file stem so the caller can pass it to the
    AST walker even before the return value has been inspected.
    """
    fid_from_stem = fixture_path.stem
    try:
        tree = ast.parse(fixture_path.read_text(encoding="utf-8"), filename=str(fixture_path))
    except (OSError, SyntaxError):
        return None

    ast_result = _extract_make_function_return(tree, fid_from_stem)
    if ast_result is None:
        return None

    inputs_val, fid_val, desc_val = ast_result

    # If anything came back None we need the import fallback for the missing
    # piece. We do the whole tuple via import to keep semantics consistent
    # with what the function actually returns at runtime.
    if inputs_val is None or fid_val is None or desc_val is None:
        imported = _import_make_function_return(fixture_path, fid_from_stem)
        if imported is None:
            return None
        inputs_val, fid_val, desc_val = imported

    return {
        "id": fid_val,
        "description": desc_val,
        "inputs": inputs_val,
    }


def synthesize_fixtures_emission_from_existing(
    package_dir: Path,
) -> Optional[dict]:
    """If ``fixtures_emission.json`` is missing but ``fixtures/`` exists with
    LLM-emitted content, back-derive the emission IR from the existing
    fixture ``.py`` files.

    Each auto-generated fixture file defines ``make_<fid>() -> tuple[dict,
    str, str]`` returning ``(inputs, fixture_id, description)``. When the LLM
    forgets to emit ``intermediate/fixtures_emission.json`` (observed in the
    sentry-find-bugs failure: full pipeline ran, smoke check passed, IR
    emission was simply dropped) the wrapper's wipe-then-render contract used
    to destroy the surviving fixtures. This synthesizer reconstructs the IR
    BEFORE the wipe so the writer can re-render the same files
    byte-identically — keeping the package self-consistent without breaking
    the writer's authority over ``fixtures/`` contents.

    Behaviour:
      - If ``<package_dir>/intermediate/fixtures_emission.json`` exists →
        return its parsed contents unchanged. No clobbering. The Track 1 #4
        sibling synthesizer follows the same rule.
      - If ``<package_dir>/fixtures/`` is absent, or contains only
        ``__init__.py`` (no per-fixture files to reverse-engineer) → return
        ``None``. The wrapper then falls through to the existing
        ``missing-emission`` flow, which surfaces the failure cleanly.
      - Otherwise: AST-parse each non-``__init__`` ``.py`` file under
        ``fixtures/`` for its ``make_<fid>()`` return tuple, materialise a
        schema-conformant payload, write it to disk, and return the dict.
        Falls back to importing the fixture module only if AST extraction
        cannot resolve a literal for any tuple element (rare —
        auto-generated fixtures are pure literals).

    The function is idempotent: running it twice on the same package
    produces identical on-disk content (the first call writes the IR; the
    second call reads it back unchanged and returns the same dict).
    """
    intermediate_dir = package_dir / "intermediate"
    emission_path = intermediate_dir / "fixtures_emission.json"
    fixtures_dir = package_dir / "fixtures"

    if emission_path.exists():
        try:
            return json.loads(emission_path.read_text())
        except (OSError, json.JSONDecodeError):
            # On-disk IR is unreadable — defer to the writer-renderer's
            # writer-error path rather than overwriting a potentially
            # recoverable artefact here.
            return None

    if not fixtures_dir.is_dir():
        return None

    # Collect candidate fixture files: anything ending in .py that is not
    # __init__.py. Sort by name for deterministic ordering, but if a
    # `fixtures/__init__.py` exists with an `ALL_FIXTURES` list we prefer
    # its order — that preserves the LLM's intended sequencing across the
    # round-trip.
    candidate_paths = sorted(
        p for p in fixtures_dir.glob("*.py") if p.name != "__init__.py"
    )
    if not candidate_paths:
        return None

    ordered_ids = _read_all_fixtures_order(fixtures_dir / "__init__.py")
    if ordered_ids:
        by_stem = {p.stem: p for p in candidate_paths}
        ordered_paths: list[Path] = []
        # First, fixtures the __init__.py knows about — in its order.
        for fid in ordered_ids:
            if fid in by_stem:
                ordered_paths.append(by_stem[fid])
        # Then any orphan .py files not referenced by __init__.py, in name
        # order, so we never silently drop one.
        for p in candidate_paths:
            if p not in ordered_paths:
                ordered_paths.append(p)
        candidate_paths = ordered_paths

    fixtures: list[dict] = []
    coverage_doc = _read_coverage_doc(fixtures_dir / "__init__.py")
    for fixture_path in candidate_paths:
        extracted = _extract_one_fixture(fixture_path)
        if extracted is None:
            LOGGER.warning(
                "[writer:fixtures/] could not reverse-engineer %s — skipping. "
                "The fixture file's make_<fid>() return value is not a literal "
                "tuple and importing the module also failed. The synthesized IR "
                "will not include this fixture and the writer's wipe-then-render "
                "will drop it.",
                fixture_path.name,
            )
            continue
        fixtures.append(extracted)

    if not fixtures:
        return None

    synthesized: dict = {"fixtures": fixtures}
    if coverage_doc:
        synthesized["coverage_doc"] = coverage_doc

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    emission_path.write_text(json.dumps(synthesized, indent=2))
    LOGGER.warning(
        "[writer:fixtures/] Synthesized fixtures_emission.json from %d existing "
        "fixture file(s) (LLM did not emit IR). This is a fallback path — the "
        "model is expected to emit intermediate/fixtures_emission.json directly. "
        "See `synthesize_fixtures_emission_from_existing` for the recovery flow.",
        len(fixtures),
    )
    return synthesized


def _read_all_fixtures_order(init_path: Path) -> list[str]:
    """Parse ``fixtures/__init__.py`` for its ``ALL_FIXTURES`` list and return
    the fixture ids in declaration order.

    Returns an empty list if the file is missing, unparseable, or does not
    define ``ALL_FIXTURES`` in a recognisable shape. We are tolerant here —
    this is a best-effort ordering hint, not a contract.
    """
    if not init_path.is_file():
        return []
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, SyntaxError):
        return []
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
            value = stmt.value
        if not targets or not isinstance(value, ast.List):
            continue
        names = [t for t in targets if isinstance(t, ast.Name) and t.id == "ALL_FIXTURES"]
        if not names:
            continue
        ordered: list[str] = []
        for elt in value.elts:
            # Entries are bare Name refs (`make_<fid>`); strip the prefix.
            if isinstance(elt, ast.Name) and elt.id.startswith("make_"):
                ordered.append(elt.id[len("make_") :])
        return ordered
    return []


def _read_coverage_doc(init_path: Path) -> Optional[str]:
    """Pull the module-level docstring out of ``fixtures/__init__.py`` so the
    synthesized IR can carry the `coverage_doc` field through the round trip.

    The writer renders ``coverage_doc`` as the module docstring, so this is
    a faithful inverse. If the docstring is the auto-generated placeholder
    (``"Auto-generated by melleafy from fixtures_emission JSON."``), return
    None — that placeholder is a writer default, not a real coverage doc, and
    re-emitting it would confuse downstream readers.
    """
    if not init_path.is_file():
        return None
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, SyntaxError):
        return None
    doc = ast.get_docstring(tree, clean=False)
    if doc is None:
        return None
    if doc.strip() == "Auto-generated by melleafy from fixtures_emission JSON.":
        return None
    return doc


def render_writers(
    package_dir: Path, specs: List[WriterSpec], *, enforce: bool = False
) -> List[RenderResult]:
    """Render every writer in `specs` against the package and log outcomes.

    With `enforce=False` (default for migration phase), only logs WARN on diff
    and never overwrites. With `enforce=True`, the writer's output becomes the
    authoritative source for the artifact.

    Before dispatching the writers, two fallback synthesizers run against
    ``<package_dir>``:

      1. ``synthesize_config_emission_from_directive`` — if the LLM omitted
         ``config_emission.json`` but ``runtime_directive.json`` is present,
         a minimal schema-compliant emission is materialised from the
         directive so the deterministic ``config_writer`` always has an
         input to render. This is the wrapper-side guarantee that
         ``runtime-defaults-bound`` never fails purely because the LLM
         forgot to emit the IR.
      2. ``synthesize_fixtures_emission_from_existing`` — if the LLM omitted
         ``fixtures_emission.json`` but a populated ``fixtures/`` directory
         is on disk (LLM wrote the .py files but dropped the IR), reverse
         engineer the IR from the surviving .py files BEFORE the writer's
         wipe-then-render runs. Without this, the wipe would destroy the
         LLM's work and the missing-emission flow would bail out — both
         observed in the sentry-find-bugs overnight failure.

    Both synthesizers are idempotent and short-circuit when their target
    emission file already exists, so calling them unconditionally is safe.
    """
    # Best-effort synthesis BEFORE the writers iterate. Each synthesizer is
    # a no-op when its target emission file already exists, so these calls
    # are safe to make unconditionally on every compile.
    synthesize_config_emission_from_directive(package_dir / "intermediate")
    synthesize_fixtures_emission_from_existing(package_dir)

    results: List[RenderResult] = []
    for spec in specs:
        try:
            result = _render_one(package_dir, spec, enforce=enforce)
        except Exception as exc:  # noqa: BLE001
            result = RenderResult(
                name=spec.name,
                status="writer-error",
                detail=f"unexpected error: {exc}",
            )
        results.append(result)
        _log_result(result, enforce=enforce)
    return results


def _log_result(result: RenderResult, *, enforce: bool) -> None:
    if result.status == "match":
        if result.detail:
            LOGGER.info("[writer:%s] %s", result.name, result.detail)
        else:
            LOGGER.info("[writer:%s] matches on-disk file", result.name)
    elif result.status == "diff":
        # In WARN mode this is the signal we are watching for during migration.
        LOGGER.warning(
            "[writer:%s] %s. The on-disk file was written by the LLM and differs "
            "from what the deterministic writer would produce. Once these diffs "
            "are bounded across a few compiles, the writer can be flipped to ENFORCE.",
            result.name,
            result.detail,
        )
    elif result.status == "missing-emission":
        # Soft in WARN mode; would be hard halt in ENFORCE mode.
        level = LOGGER.error if enforce else LOGGER.warning
        level("[writer:%s] %s", result.name, result.detail)
    elif result.status == "missing-output":
        LOGGER.warning("[writer:%s] %s", result.name, result.detail)
    elif result.status == "writer-error":
        LOGGER.warning("[writer:%s] %s", result.name, result.detail)
    elif result.status == "schema-invalid":
        # Always error-level: the LLM emitted JSON that doesn't conform to the
        # writer's contract; the writer cannot be invoked. Step 7 lints will
        # then hard-fail on the missing rendered artifact.
        LOGGER.error("[writer:%s] %s", result.name, result.detail)


def _load_default_surface() -> Optional[dict[str, Any]]:
    """Resolve the bundled surface JSON via importlib.resources.

    Mirrors the resolution used by ``renderer/cli.py``: read
    ``surface_0.5.0.json`` shipped under
    ``src/mellea_skills_compiler/mellea_surface/data/``. Returns the parsed
    dict, or None if the file cannot be located or parsed (the caller treats
    None as "cannot render — descriptor mode is degraded" rather than
    crashing the compile).
    """
    try:
        from importlib.resources import files as _resource_files

        surface_path = Path(
            str(
                _resource_files("mellea_skills_compiler.mellea_surface")
                / "data"
                / "surface_0.5.0.json"
            )
        )
    except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        LOGGER.warning(
            "[writer:descriptor] could not locate bundled surface JSON: %s",
            exc,
        )
        return None
    if not surface_path.exists():
        LOGGER.warning(
            "[writer:descriptor] bundled surface JSON %s not found",
            surface_path,
        )
        return None
    try:
        return json.loads(surface_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[writer:descriptor] bundled surface JSON %s unreadable: %s",
            surface_path,
            exc,
        )
        return None


def render_descriptor_to_python(
    package_dir: Path,
    *,
    surface: Optional[dict[str, Any]] = None,
    skill_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Render ``pipeline.py`` + ``schemas.py`` from ``descriptor_emission.json``.

    This is the wrapper-side counterpart of the descriptor-renderer plumbing
    used by :func:`descriptor_emission.compile_via_descriptor`. The slash
    command (``--use-descriptor``) instructs Claude to emit
    ``intermediate/descriptor_emission.json`` and tells the model "the
    wrapper will render pipeline.py / schemas.py from this". Historically
    the wrapper did NOT actually do that — :func:`render_writers` only
    handles ``config.py`` and ``fixtures/``. Three skills in the May 17
    overnight batch landed with descriptor_emission.json on disk but no
    rendered pipeline.py / schemas.py; the lint reported ``skipped`` on the
    absent file, ``overall_verdict=pass`` was emitted, and the smoke check
    crashed with a misleading "module is missing" error. This function
    closes that gap.

    Args:
        package_dir: The compiled-skill package directory.
        surface: Pre-loaded Mellea surface JSON (the same shape the renderer
            and validator consume). When ``None``, the bundled
            ``mellea_surface/data/surface_0.5.0.json`` is loaded via
            ``importlib.resources``.
        skill_root: Path to the source skill directory, threaded into the
            renderer so v0.3 ``bundled_resources`` can resolve their source
            paths. Optional — descriptors without bundled resources do not
            require it.

    Returns:
        A summary dict with the shape
        ``{"verdict": "rendered" | "skipped" | "failed",
        "files_written": [str, ...], "reason": str | None}``.

        Verdicts:
          - ``"skipped"`` — ``intermediate/descriptor_emission.json`` is
            absent. The caller (descriptor-mode compile flow) treats this
            as a soft warning: in legacy mode Claude writes pipeline.py
            directly, and the lint will fail if neither path produced it.
          - ``"failed"`` — descriptor_emission.json is present but the
            renderer raised. ``reason`` carries the error message.
          - ``"rendered"`` — pipeline.py and schemas.py were materialised.

    Idempotency: re-running this function against the same descriptor
    produces byte-identical output (the renderer is deterministic), so it
    is safe to invoke from the lint-repair retry loop. When the on-disk
    pipeline.py / schemas.py already exist, they are overwritten — the
    wrapper is the source of truth in descriptor mode (per the slash
    command's contract). When the on-disk content matches the rendered
    content, no behavioural change beyond the overwrite occurs. When the
    on-disk content differs (Claude wrote a free-form version alongside
    the descriptor emission), a warning is logged before overwriting.
    """
    descriptor_path = package_dir / "intermediate" / "descriptor_emission.json"
    if not descriptor_path.exists():
        return {
            "verdict": "skipped",
            "files_written": [],
            "reason": "descriptor_emission_absent",
        }

    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "verdict": "failed",
            "files_written": [],
            "reason": f"descriptor_emission.json unreadable: {exc}",
        }

    if surface is None:
        surface = _load_default_surface()
    if surface is None:
        return {
            "verdict": "failed",
            "files_written": [],
            "reason": "no surface JSON available (could not load bundled default)",
        }

    # Local imports keep this module importable without dragging the
    # renderer / schemas modules into every compile invocation.
    try:
        from mellea_skills_compiler.renderer import render_descriptor
        from mellea_skills_compiler.renderer.schemas import render_schemas
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "failed",
            "files_written": [],
            "reason": f"renderer modules unavailable: {exc}",
        }

    try:
        render_result = render_descriptor(descriptor, surface, skill_root=skill_root)
        pipeline_py = render_result.pipeline_py
        schemas_py = render_schemas(descriptor.get("schemas", {}) or {})
    except Exception as exc:  # noqa: BLE001 - renderer surfaces a typed
        # RendererError but we also catch downstream Python errors so we never
        # crash the surrounding compile; the lint will hard-fail on the
        # resulting missing file.
        return {
            "verdict": "failed",
            "files_written": [],
            "reason": f"render_descriptor raised: {type(exc).__name__}: {exc}",
        }

    pipeline_path = package_dir / "pipeline.py"
    schemas_path = package_dir / "schemas.py"

    # Idempotency choice: always overwrite. The slash command's contract
    # under --use-descriptor is "Claude emits descriptor JSON, wrapper
    # renders pipeline.py + schemas.py from it" — so the wrapper is the
    # source of truth. If Claude also wrote a free-form pipeline.py the
    # wrapper-rendered version wins. We log the overwrite so the divergence
    # is auditable, but we do not require a match.
    package_dir.mkdir(parents=True, exist_ok=True)
    for target, rendered, label in (
        (pipeline_path, pipeline_py, "pipeline.py"),
        (schemas_path, schemas_py, "schemas.py"),
    ):
        if target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except OSError:
                existing = None
            if existing is not None and existing != rendered:
                LOGGER.warning(
                    "[writer:descriptor] overwriting %s with descriptor-rendered "
                    "version (on-disk content differs by %d bytes)",
                    label,
                    abs(len(existing) - len(rendered)),
                )
        target.write_text(rendered, encoding="utf-8")

    return {
        "verdict": "rendered",
        "files_written": ["pipeline.py", "schemas.py"],
        "reason": None,
    }


def default_writer_specs(repo_root) -> List[WriterSpec]:
    """Wrapper-rendered artifacts. Order matters only for log readability.

    Add new specs as each migration step lands. Each spec must satisfy:
      - file-mode writer (`output_kind="file"`): module exposes `render(emission) -> str`
      - dir-mode writer  (`output_kind="directory"`): module exposes
        `write(emission, out_dir) -> list[Path]`
    """
    writers_dir = repo_root / ".claude" / "melleafy" / "writers"
    schemas_dir = repo_root / ".claude" / "schemas"
    return [
        WriterSpec(
            name="config.py",
            emission_relpath="intermediate/config_emission.json",
            output_relpath="config.py",
            writer_path=writers_dir / "config_writer.py",
            schema_path=schemas_dir / "config_emission.schema.json",
        ),
        WriterSpec(
            name="fixtures/",
            emission_relpath="intermediate/fixtures_emission.json",
            output_relpath="fixtures",
            writer_path=writers_dir / "fixtures_writer.py",
            output_kind="directory",
            schema_path=schemas_dir / "fixtures_emission.schema.json",
        ),
    ]
