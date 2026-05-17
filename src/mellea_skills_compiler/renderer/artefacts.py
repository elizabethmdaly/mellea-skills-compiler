"""Supporting-artefact renderers — everything the descriptor compiler emits
**besides** ``pipeline.py`` and ``schemas.py``.

Per plan §5.2 every compiled descriptor package contains:

  ===================== =============================================
  File                  Renderer
  ===================== =============================================
  ``pipeline.py``       P2.A's :mod:`renderer.core` (not this module)
  ``schemas.py``        :mod:`renderer.schemas` (not this module)
  ``__init__.py``       :func:`render_init`
  ``fixtures.py``       :func:`render_fixtures`
  ``melleafy.json``     :func:`render_melleafy_json`
  ``SETUP.md``          :func:`render_setup_md`
  ``README.md``         :func:`render_readme_md`
  ===================== =============================================

Each emitter is **pure**: descriptor data in, file content out. Stable —
identical input produces byte-identical output. Time-dependent fields
(``melleafy.json:compiled_at``) accept a caller-supplied timestamp so tests
and reproducible builds can pin them.

The ``fixtures.py`` emitter intentionally produces only **boilerplate**: a
``load_fixture(name)`` helper that reads ``fixtures/<name>.json`` from the
package directory. Fixture *content* generation remains the ``/mellea-fy``
Step 4 pipeline's job (see ``examples/sentry-find-bugs/.../fixtures/`` for
what populated fixtures look like under the legacy compiler).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from mellea_skills_compiler.renderer import RendererError

__all__ = [
    "canonical_descriptor_hash",
    "render_fixtures",
    "render_init",
    "render_melleafy_json",
    "render_readme_md",
    "render_setup_md",
]


# --- Helpers ----------------------------------------------------------------


def _require_skill(descriptor: dict[str, Any]) -> dict[str, Any]:
    skill = descriptor.get("skill")
    if not isinstance(skill, dict):
        raise RendererError(
            "descriptor missing required 'skill' object "
            f"(got {type(skill).__name__})"
        )
    if not skill.get("name"):
        raise RendererError("descriptor.skill.name is required and non-empty")
    return skill


def canonical_descriptor_hash(descriptor: dict[str, Any]) -> str:
    """Return the canonical SHA-256 hash of a descriptor.

    Canonicalisation: JSON with ``sort_keys=True`` and the most compact
    separators. Same descriptor input → same digest, regardless of the
    in-memory dict's iteration order or whitespace in the source file.
    """
    canonical = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _extract_schema_class_name(slot: dict[str, Any]) -> str | None:
    """Best-effort extraction of the Python class name for an input/output slot.

    Handles the v0.1 slot shapes used in real descriptors:
      - ``{"kind": "model_ref", "ref": "#/schemas/X"}``
      - ``{"kind": "pydantic_model", "ref": "#/schemas/X"}``
      - ``{"kind": "list", "of": {"ref": "#/schemas/X"}}``
      - ``{"kind": "str" | "int" | ...}``

    Returns the bare class name, the primitive name, or ``None`` if the slot
    is shaped in a way the README emitter doesn't recognise.
    """
    schema = slot.get("schema")
    if not isinstance(schema, dict):
        return None
    if "ref" in schema and isinstance(schema["ref"], str):
        return schema["ref"].split("/")[-1]
    kind = schema.get("kind")
    if kind == "list":
        of = schema.get("of") or {}
        if isinstance(of, dict) and isinstance(of.get("ref"), str):
            return f"list[{of['ref'].split('/')[-1]}]"
    if kind in ("str", "int", "float", "bool"):
        return kind
    return None


# --- __init__.py -------------------------------------------------------------


def render_init(descriptor: dict[str, Any]) -> str:
    """Render the compiled package's ``__init__.py``.

    Re-exports the pipeline entry point. We intentionally **do not** re-export
    schema classes — the descriptor's schemas are an implementation detail of
    the compiled package; callers who need them can import from
    ``.schemas`` explicitly.

    The reference output in ``examples/sentry-find-bugs/.../__init__.py``
    re-exports a handful of schemas (``FindingsReport``, ``IssueReport``).
    That was hand-curated by the legacy compiler; P2's clean rewrite keeps
    the surface minimal until we see a real need to widen it.
    """
    _require_skill(descriptor)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="pipeline",
                names=[ast.alias(name="run_pipeline")],
                level=1,
            ),
            ast.Assign(
                targets=[ast.Name(id="__all__")],
                value=ast.List(
                    elts=[ast.Constant(value="run_pipeline")],
                    ctx=ast.Load(),
                ),
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


# --- fixtures.py -------------------------------------------------------------


def render_fixtures(descriptor: dict[str, Any], fixtures_dir: Path) -> str:
    """Render ``fixtures.py`` boilerplate — a ``load_fixture()`` helper only.

    Args:
        descriptor: The descriptor dict. The skill name is used in the
            module docstring; otherwise the descriptor is not consulted.
        fixtures_dir: The directory (relative to the compiled package) that
            holds fixture JSON files. Stored in the emitted source as the
            literal directory name (last path component) so the helper can
            resolve it relative to ``__file__``.

    Returns:
        A complete Python source string. The helper signature is
        ``load_fixture(name: str) -> dict``; it reads
        ``<package_dir>/<fixtures_dir>/<name>.json``.
    """
    skill = _require_skill(descriptor)
    fixtures_dirname = fixtures_dir.name or "fixtures"

    # Module-level docstring. We pass the *string content* to ast.Constant —
    # ast.unparse handles the surrounding quote/escape gymnastics.
    docstring_content = (
        f"Fixture-loading helpers for the compiled `{skill['name']}` skill package.\n\n"
        f"Fixtures live as JSON data files under ``{fixtures_dirname}/`` — use\n"
        ":func:`load_fixture` to read one by name. Fixture content itself is\n"
        "generated by the /mellea-fy Step 4 pipeline (or hand-authored); this\n"
        "module only provides the loading boilerplate."
    )

    module = ast.Module(
        body=[
            ast.Expr(value=ast.Constant(value=docstring_content)),
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.Import(names=[ast.alias(name="json")]),
            ast.ImportFrom(
                module="pathlib",
                names=[ast.alias(name="Path")],
                level=0,
            ),
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name="Any")],
                level=0,
            ),
            ast.Assign(
                targets=[ast.Name(id="_FIXTURES_DIR")],
                value=ast.BinOp(
                    left=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id="Path"),
                            args=[ast.Name(id="__file__")],
                            keywords=[],
                        ),
                        attr="parent",
                    ),
                    op=ast.Div(),
                    right=ast.Constant(value=fixtures_dirname),
                ),
            ),
            _build_load_fixture_fn(),
            ast.Assign(
                targets=[ast.Name(id="__all__")],
                value=ast.List(
                    elts=[ast.Constant(value="load_fixture")],
                    ctx=ast.Load(),
                ),
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _build_load_fixture_fn() -> ast.FunctionDef:
    """Build the AST for ``load_fixture(name: str) -> dict[str, Any]``.

    Body::

        path = _FIXTURES_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"fixture not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    """
    body: list[ast.stmt] = [
        ast.Expr(
            value=ast.Constant(
                value=(
                    "Load fixture `name` from the package's fixtures "
                    "directory as a dict.\n\n"
                    "Raises FileNotFoundError if the fixture does not exist."
                )
            )
        ),
        ast.Assign(
            targets=[ast.Name(id="path")],
            value=ast.BinOp(
                left=ast.Name(id="_FIXTURES_DIR"),
                op=ast.Div(),
                right=ast.JoinedStr(
                    values=[
                        ast.FormattedValue(
                            value=ast.Name(id="name"),
                            conversion=-1,
                        ),
                        ast.Constant(value=".json"),
                    ]
                ),
            ),
        ),
        ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="path"), attr="is_file"),
                    args=[],
                    keywords=[],
                ),
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="FileNotFoundError"),
                        args=[
                            ast.JoinedStr(
                                values=[
                                    ast.Constant(value="fixture not found: "),
                                    ast.FormattedValue(
                                        value=ast.Name(id="path"),
                                        conversion=-1,
                                    ),
                                ]
                            )
                        ],
                        keywords=[],
                    ),
                    cause=None,
                )
            ],
            orelse=[],
        ),
        ast.Return(
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="json"), attr="loads"),
                args=[
                    ast.Call(
                        func=ast.Attribute(value=ast.Name(id="path"), attr="read_text"),
                        args=[],
                        keywords=[ast.keyword(arg="encoding", value=ast.Constant(value="utf-8"))],
                    )
                ],
                keywords=[],
            )
        ),
    ]
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="name", annotation=ast.Name(id="str"))],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    return ast.FunctionDef(
        name="load_fixture",
        args=args,
        body=body,
        decorator_list=[],
        returns=ast.Subscript(
            value=ast.Name(id="dict"),
            slice=ast.Tuple(
                elts=[ast.Name(id="str"), ast.Name(id="Any")],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        ),
        type_params=[],
    )


# --- melleafy.json -----------------------------------------------------------


def render_melleafy_json(
    descriptor: dict[str, Any],
    mellea_version: str,
    renderer_version: str,
    compiled_at: str,
) -> str:
    """Render ``melleafy.json`` — compiled-package metadata.

    All four fields specified in plan §5.2:
      - ``mellea_version`` — the Mellea version the descriptor targeted
      - ``descriptor_hash`` — SHA-256 over the canonicalised descriptor
      - ``renderer_version`` — version of this compiler (caller-supplied)
      - ``compiled_at`` — ISO 8601 timestamp (caller-supplied)

    Determinism: ``compiled_at`` is a parameter, not a ``datetime.now()`` call.
    Same inputs → byte-identical output. The JSON itself is emitted with
    ``sort_keys=True`` and a trailing newline.

    Args:
        descriptor: The validated descriptor dict.
        mellea_version: The Mellea version string. Typically lifted from
            ``descriptor["mellea_version"]`` by the caller; passed explicitly
            so the caller can override (e.g. record the *installed* version
            distinct from the *declared* version).
        renderer_version: Semantic version of this renderer.
        compiled_at: ISO 8601 timestamp (e.g. ``"2026-05-16T12:00:00Z"``).
    """
    if not isinstance(descriptor, dict):
        raise RendererError(
            f"descriptor must be a dict, got {type(descriptor).__name__}"
        )
    if not mellea_version:
        raise RendererError("mellea_version is required and non-empty")
    if not renderer_version:
        raise RendererError("renderer_version is required and non-empty")
    if not compiled_at:
        raise RendererError("compiled_at is required and non-empty")

    payload = {
        "compiled_at": compiled_at,
        "descriptor_hash": canonical_descriptor_hash(descriptor),
        "mellea_version": mellea_version,
        "renderer_version": renderer_version,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# --- SETUP.md ----------------------------------------------------------------


def render_setup_md(descriptor: dict[str, Any]) -> str:
    """Render ``SETUP.md`` — minimal install / run instructions.

    Uses ``descriptor.skill.name`` for headings and references. Documents the
    two required environment variables every compiled package needs to talk
    to a model backend: ``MODEL_ID`` (the model name) and ``OPENAI_API_KEY``
    (the API key for the configured OpenAI-compatible endpoint).

    Output is markdown; we use raw f-strings here per scope (no Jinja, no
    Mako).
    """
    skill = _require_skill(descriptor)
    name = skill["name"]

    return f"""# Setup — `{name}`

This package was generated by the Mellea Skills Compiler from a typed
descriptor. Follow these steps to run the pipeline locally.

## 1. Install

From the skill root:

```bash
pip install -e .
```

## 2. Configure the model backend

The pipeline talks to an OpenAI-compatible endpoint. Set the two required
environment variables before invoking the pipeline:

```bash
export MODEL_ID="<your-model-id>"           # e.g. granite3.3:8b or gpt-4o-mini
export OPENAI_API_KEY="<your-api-key>"      # API key for the configured endpoint
```

If your backend exposes a non-default base URL (e.g. an Ollama proxy), also
set ``OPENAI_BASE_URL``.

## 3. Run the pipeline

```python
from {_module_name(name)} import run_pipeline

result = run_pipeline(...)
```

See ``README.md`` for the exact call signature derived from this skill's
descriptor.
"""


def _module_name(skill_name: str) -> str:
    """Convert a skill name (potentially kebab-case) into a Python-package name.

    ``find-bugs`` → ``find_bugs``. Keeps existing snake_case names intact.
    """
    return skill_name.replace("-", "_")


# --- README.md ---------------------------------------------------------------


def render_readme_md(descriptor: dict[str, Any]) -> str:
    """Render ``README.md`` — skill summary, signature, and call example.

    Pulls:
      - skill name and (optional) classification from ``descriptor.skill``
      - description from ``descriptor.skill.description`` if present
      - input parameter names and types from ``descriptor.inputs``
      - return type from ``descriptor.outputs[0]``

    The call example is constructed from the input schema only — we don't
    speculate about sensible *values* for each input, just show the
    keyword-argument shape.
    """
    skill = _require_skill(descriptor)
    name = skill["name"]
    classification = skill.get("classification")
    description = skill.get("description") or (
        f"Compiled Mellea pipeline for the `{name}` skill."
    )

    inputs = descriptor.get("inputs") or []
    outputs = descriptor.get("outputs") or []

    signature_args: list[str] = []
    for inp in inputs:
        iname = inp.get("name")
        if not iname:
            continue
        itype = _extract_schema_class_name(inp) or "Any"
        signature_args.append(f"{iname}: {itype}")
    return_type = (
        _extract_schema_class_name(outputs[0]) if outputs else None
    ) or "Any"

    signature_line = (
        f"run_pipeline({', '.join(signature_args)}) -> {return_type}"
    )

    example_kwargs = ",\n    ".join(
        f"{inp.get('name')}=..." for inp in inputs if inp.get("name")
    )
    if example_kwargs:
        call_example = f"result = run_pipeline(\n    {example_kwargs},\n)"
    else:
        call_example = "result = run_pipeline()"

    classification_line = (
        f"\n**Classification**: `{classification}`\n" if classification else ""
    )

    inputs_table = _render_inputs_table(inputs)

    return f"""# {name}

{description}
{classification_line}
## Entry point

```
{signature_line}
```

## Usage

```python
from {_module_name(name)} import run_pipeline

{call_example}
```

## Inputs
{inputs_table}
## Generated by

Mellea Skills Compiler (descriptor mode) from `{name}`'s typed descriptor.
"""


def _render_inputs_table(inputs: list[Any]) -> str:
    """Markdown table of input names + types + descriptions. Empty if no inputs."""
    rows: list[str] = []
    for inp in inputs:
        if not isinstance(inp, dict) or not inp.get("name"):
            continue
        iname = inp["name"]
        itype = _extract_schema_class_name(inp) or "Any"
        idesc = (inp.get("description") or "").strip().replace("\n", " ")
        rows.append(f"| `{iname}` | `{itype}` | {idesc} |")
    if not rows:
        return "\n(no declared inputs)\n"
    return (
        "\n| Name | Type | Description |\n"
        "| --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n"
    )
