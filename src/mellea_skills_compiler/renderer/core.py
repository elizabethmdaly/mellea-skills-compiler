"""Renderer core: RenderContext + top-level render_descriptor.

Plan §5 architecture in code form. `render_descriptor` is the production
entry point; it does not emit `schemas.py` / `fixtures.py` / etc — those
belong to P2.C. This file produces a single `pipeline.py` string plus a
source map.

The operator interface and registry live in
``mellea_skills_compiler.renderer.__init__`` (the cross-stream contract with
P2.B). P2.B's composition operators self-register on import of
``mellea_skills_compiler.renderer.operators``; this module triggers that
import lazily, and falls back to a built-in SequentialRenderer if P2.B's
package is not present (so `test_core.py` can exercise the end-to-end path
on a sequential-only descriptor without P2.B's operators package).
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from mellea_skills_compiler.renderer import (
    OperatorRenderer,
    RendererError,
    get_operator,
    register_operator,
    registered_operator_names,
)
from mellea_skills_compiler.renderer.nodes import (
    emit_call_node,
    emit_state,
    lower_type,
    resolve_symbol,
    symbol_resolves,
)
from mellea_skills_compiler.renderer.source_map import SourceMap


# --- RenderContext --------------------------------------------------------


@dataclass
class ResolvedRef:
    """Result of `RenderContext.resolve_ref(id)`. Used for repair-loop introspection."""

    id: str
    kind: str  # 'input' | 'state' | 'pipeline'


@dataclass
class RenderContext:
    """Shared mutable state for one render pass.

    Tracks:
      - imports that must be emitted at the top of pipeline.py
      - declared ids (inputs, state, pipeline nodes) for ref resolution
      - schema names that must be imported from `.schemas`
      - the source map (descriptor-node id <-> Python line span)
      - v0.3: ``is_async`` flag (skill-level async transitive transform)
      - v0.3: ``source_elements_map`` (E-id → descriptor node id, ``mapping_report.md``)
    """

    descriptor: dict[str, Any]
    surface: dict[str, Any]
    imports: dict[str, set[str]] = field(default_factory=dict)
    needs_os: bool = False
    state_ids: set[str] = field(default_factory=set)
    input_ids: set[str] = field(default_factory=set)
    pipeline_ids: set[str] = field(default_factory=set)
    schema_uses: set[str] = field(default_factory=set)
    source_map: SourceMap = field(default_factory=SourceMap)
    is_async: bool = False
    source_elements_map: dict[str, list[str]] = field(default_factory=dict)

    # --- Construction helpers ------------------------------------------

    def __post_init__(self) -> None:
        # Pre-populate declared input ids so refs validate immediately.
        for inp in self.descriptor.get("inputs", []):
            self.input_ids.add(inp["name"])

    # --- Mutation -------------------------------------------------------

    def add_import(self, module: str, name: str) -> None:
        if not module:
            return
        self.imports.setdefault(module, set()).add(name)

    def add_os_import(self) -> None:
        self.needs_os = True

    def register_id(self, id_: str, kind: str) -> None:
        # `kind` is informational — used by P2.B operators to tag loop vars
        # vs accumulators. We bucket pipeline-shaped kinds into one set; only
        # the declared 'state' and 'input' kinds get their own buckets.
        if kind == "state":
            self.state_ids.add(id_)
        elif kind == "input":
            self.input_ids.add(id_)
        else:
            # pipeline node, accumulator, loop_var, branch_result, ... all
            # become referenceable as pipeline ids.
            self.pipeline_ids.add(id_)

    def register_schema_use(self, name: str) -> None:
        if name in self.descriptor.get("schemas", {}):
            self.schema_uses.add(name)

    def record_source_elements(
        self, node_id: str, elements: list[str], descriptor_path: str = ""
    ) -> None:
        """Record ``source_elements`` metadata for descriptor traceability (v0.3).

        Consumed downstream by Step 6's ``mapping_report.md`` builder; the
        renderer itself does not emit any code change in response.
        """
        if not node_id or not elements:
            return
        self.source_elements_map[node_id] = list(elements)

    # --- Resolution -----------------------------------------------------

    def resolve_ref(self, ref_dict: dict[str, Any]) -> ast.expr:
        """Lower a ref dict ``{ref, select?}`` / ``{schema_ref}`` / ``{value}``
        into an AST expression.

        Operators call this rather than implementing their own ref-lowering;
        it's the single point of truth for "what does this descriptor ref
        mean in Python source?".
        """
        if not isinstance(ref_dict, dict):
            raise RendererError(
                f"resolve_ref expected a dict, got {type(ref_dict).__name__}"
            )
        if "ref" in ref_dict:
            # We *do not* validate the target id exists yet — composition
            # operators may forward-reference their own body's nodes during
            # render. Validation runs at descriptor-validation time (§7.1).
            node: ast.expr = ast.Name(id=ref_dict["ref"], ctx=ast.Load())
            select = ref_dict.get("select")
            if select:
                for piece in str(select).split("."):
                    node = ast.Attribute(value=node, attr=piece, ctx=ast.Load())
            return node
        if "schema_ref" in ref_dict:
            name = ref_dict["schema_ref"].split("/")[-1]
            self.register_schema_use(name)
            return ast.Name(id=name, ctx=ast.Load())
        if "value" in ref_dict:
            return ast.Constant(value=ref_dict["value"])
        raise RendererError(
            f"resolve_ref: unsupported ref shape, keys={sorted(ref_dict)}"
        )

    def lookup_ref_kind(self, ref_id: str) -> ResolvedRef:
        """Programmatic ref lookup. Throws if `ref_id` isn't declared anywhere."""
        if ref_id in self.pipeline_ids:
            return ResolvedRef(id=ref_id, kind="pipeline")
        if ref_id in self.state_ids:
            return ResolvedRef(id=ref_id, kind="state")
        if ref_id in self.input_ids:
            return ResolvedRef(id=ref_id, kind="input")
        raise RendererError(f"unresolved ref: {ref_id!r}")


# --- Built-in fallback sequential operator --------------------------------


class _FallbackSequentialRenderer(OperatorRenderer):
    """In-tree fallback used when P2.B's operators package is not installed.

    The shipped operators package (``mellea_skills_compiler.renderer.operators``)
    self-registers a richer SequentialRenderer; if that import succeeds, this
    fallback is never used. The fallback exists so `test_core.py` can run
    the renderer end-to-end on sequential-only descriptors without depending
    on P2.B's tree.
    """

    operator_name = "sequential"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        return render_body(node.get("body", []))


def _ensure_operators_registered() -> None:
    """Trigger P2.B's operator self-registration; fall back if unavailable.

    Idempotent. Called from `render_descriptor` on every render — the
    underlying registry tolerates re-registration of the same instance.
    """
    if registered_operator_names():
        return
    try:
        importlib.import_module("mellea_skills_compiler.renderer.operators")
        return
    except ImportError:
        pass
    register_operator(_FallbackSequentialRenderer())


# --- Pipeline-body emission (the recursive entry point) -------------------


def emit_pipeline_body(
    nodes: list[dict[str, Any]],
    ctx: RenderContext,
    parent_path: str = "pipeline",
) -> list[ast.stmt]:
    """Walk a list of descriptor nodes, emit statements in order.

    This is the closure point that operators re-enter via the `render_body`
    callback passed into `OperatorRenderer.render`.
    """
    stmts: list[ast.stmt] = []
    for idx, node in enumerate(nodes):
        node_path = f"{parent_path}[{idx}]"
        if not isinstance(node, dict) or "kind" not in node:
            raise RendererError(
                f"node missing 'kind' (at descriptor path: {node_path})"
            )
        if node["kind"] == "call":
            _verify_call_symbol(node, ctx, node_path)
            stmts.append(emit_call_node(node, ctx, node_path))
        elif node["kind"] == "composition":
            op_name = node.get("operator")
            if not op_name:
                raise RendererError(
                    f"composition node missing 'operator' (at descriptor path: {node_path})"
                )
            op = get_operator(op_name)
            def _recurse(body_nodes: list[dict[str, Any]], _path: str = node_path) -> list[ast.stmt]:
                return emit_pipeline_body(body_nodes, ctx, _path)
            stmts.extend(op.render(node, ctx, _recurse))
        else:
            raise RendererError(
                f"unknown node kind: {node['kind']!r} (at descriptor path: {node_path})"
            )
    return stmts


def _verify_call_symbol(node: dict[str, Any], ctx: RenderContext, node_path: str) -> None:
    """Resolve the call's symbol against the surface; fail loudly if unknown."""
    symbol = node.get("symbol")
    if not symbol:
        raise RendererError(f"call node missing 'symbol' (at descriptor path: {node_path})")
    if not symbol_resolves(symbol, ctx.surface):
        raise RendererError(
            f"unknown symbol {symbol!r} (not in surface) (at descriptor path: {node_path}) (symbol: {symbol})"
        )


# --- Top-level descriptor -> pipeline.py ----------------------------------


@dataclass
class RenderResult:
    """Output of `render_descriptor`.

    Descriptor v0.3 adds (optional, populated only when v0.3 features fire):
      - ``file_copies``: scheduled file-copy operations (``bundle`` deps +
        ``bundled_resources``). The caller (e.g. package writer) applies these
        at write-time.
      - ``package_data_dirs``: directories to add to pyproject.toml package-data.
      - ``deferred_binaries``: filenames that were detected as binary and
        deferred to v0.4 (plan §11.5).
      - ``source_elements_map``: ``descriptor_node_id`` → list of E-ids;
        consumed by Step 6's ``mapping_report.md`` builder.
    """

    pipeline_py: str
    source_map: SourceMap
    file_copies: list[Any] = field(default_factory=list)
    package_data_dirs: set[str] = field(default_factory=set)
    deferred_binaries: list[str] = field(default_factory=list)
    source_elements_map: dict[str, list[str]] = field(default_factory=dict)


def render_descriptor(
    descriptor: dict[str, Any],
    surface: dict[str, Any],
    skill_root: Any = None,
) -> RenderResult:
    """Render a descriptor to pipeline.py source + source map.

    NOTE: this does NOT emit schemas.py / fixtures.py / __init__.py /
    melleafy.json / SETUP.md / README.md — those are P2.C's responsibility.

    Descriptor-version dispatch:
      - v0.1 / v0.2: existing rendering path; ``descriptor.skill.async`` is
        not consulted; no ``dependencies[]`` / ``bundled_resources[]``
        processing.
      - v0.3+: ``skill.async`` triggers the transitive async transform;
        ``dependencies[]`` and ``bundled_resources[]`` are processed.

    The dispatch keys on ``descriptor.descriptor_version``; missing/unknown
    versions are treated as v0.1 for backward-compat (the validator should
    have rejected anything truly malformed before we get here).
    """
    _ensure_operators_registered()

    descriptor_version = str(descriptor.get("descriptor_version", "0.1"))
    is_v03_plus = descriptor_version >= "0.3"
    skill_block = descriptor.get("skill") or {}
    is_async = bool(skill_block.get("async", False)) if is_v03_plus else False

    ctx = RenderContext(descriptor=descriptor, surface=surface, is_async=is_async)

    # Pre-scan inputs/outputs for annotation imports they need (v0.3 type
    # permissiveness — RFC §3.5 + G8.4). Done up-front so the import block
    # below picks them up.
    _ensure_annotation_imports(descriptor, ctx)

    # v0.3 dependencies block (closes G2 + G5).
    deps_artefact = None
    if is_v03_plus and descriptor.get("dependencies"):
        # Local import to avoid top-level cycle with renderer/__init__.
        from mellea_skills_compiler.renderer.dependencies import render_dependencies

        deps_artefact = render_dependencies(descriptor, ctx)

    # v0.3 bundled resources (closes G3).
    bundled_artefact = None
    if is_v03_plus and descriptor.get("bundled_resources"):
        from mellea_skills_compiler.renderer.bundled import render_bundled_resources

        bundled_artefact = render_bundled_resources(descriptor, skill_root=skill_root)

    state_stmts = emit_state(descriptor.get("state", []), ctx)
    pipeline_stmts = emit_pipeline_body(descriptor.get("pipeline", []), ctx)
    return_stmt = _build_return_stmt(descriptor, ctx)

    # Build the run_pipeline function.
    args_node = _build_args_node(descriptor, deps_artefact)
    if is_async:
        # async def run_pipeline(...) per plan §11.9.
        fn: ast.stmt = ast.AsyncFunctionDef(
            name="run_pipeline",
            args=args_node,
            body=pipeline_stmts + [return_stmt],
            decorator_list=[],
            returns=_output_annotation(descriptor.get("outputs", [])),
            type_params=[],
        )
    else:
        fn = ast.FunctionDef(
            name="run_pipeline",
            args=args_node,
            body=pipeline_stmts + [return_stmt],
            decorator_list=[],
            returns=_output_annotation(descriptor.get("outputs", [])),
            type_params=[],
        )

    body: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    if ctx.needs_os:
        body.append(ast.Import(names=[ast.alias(name="os")]))
    for mod, names in sorted(ctx.imports.items()):
        body.append(
            ast.ImportFrom(
                module=mod,
                names=[ast.alias(name=n) for n in sorted(names) if n],
                level=0,
            )
        )
    # Schema imports (from sibling .schemas, emitted by P2.C).
    schemas_needed = _collect_schema_names(descriptor, ctx)
    if schemas_needed:
        body.append(
            ast.ImportFrom(
                module="schemas",
                names=[ast.alias(name=n) for n in sorted(schemas_needed)],
                level=1,  # relative: from .schemas import ...
            )
        )
    body.extend(state_stmts)
    if deps_artefact is not None:
        body.extend(deps_artefact.top_level_stmts)
    body.append(fn)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    src_no_trailing = ast.unparse(module)
    ctx.source_map.build_from_module(module, source=src_no_trailing)

    # Aggregate the v0.3 artefacts for the caller.
    file_copies: list[Any] = []
    package_data_dirs: set[str] = set()
    deferred_binaries: list[str] = []
    if deps_artefact is not None:
        file_copies.extend(deps_artefact.file_copies)
        package_data_dirs.update(deps_artefact.package_data_dirs)
    if bundled_artefact is not None:
        # Bundled resources are folder-level, so we expose them as their own
        # FileCopy-shaped objects; the caller may iterate either list.
        from mellea_skills_compiler.renderer.dependencies import FileCopyOp

        for copy in bundled_artefact.copies:
            file_copies.append(
                FileCopyOp(
                    source=copy.source_dir,
                    dest=copy.package_dir,
                    is_directory=True,
                    is_package_data=copy.is_package_data,
                )
            )
            deferred_binaries.extend(copy.deferred_binaries)
        package_data_dirs.update(bundled_artefact.package_data_dirs)

    return RenderResult(
        pipeline_py=src_no_trailing + "\n",
        source_map=ctx.source_map,
        file_copies=file_copies,
        package_data_dirs=package_data_dirs,
        deferred_binaries=deferred_binaries,
        source_elements_map=dict(ctx.source_elements_map),
    )


def _build_args_node(descriptor: dict[str, Any], deps_artefact: Any) -> ast.arguments:
    """Build the ``ast.arguments`` for ``run_pipeline()``.

    Combines descriptor inputs with v0.3 dependency-derived kwargs
    (``delegate_to_runtime`` / ``external_input``). Optional inputs get
    ``= None`` defaults to satisfy Python's "non-default after default" rule.
    """
    args: list[ast.arg] = []
    defaults: list[ast.expr] = []
    required: list[tuple[ast.arg, ast.expr | None]] = []
    optional: list[tuple[ast.arg, ast.expr]] = []

    for inp in descriptor.get("inputs", []):
        annotation = _input_annotation(inp)
        if inp.get("optional"):
            annotation = ast.BinOp(left=annotation, op=ast.BitOr(), right=ast.Constant(value=None))
            optional.append(
                (ast.arg(arg=inp["name"], annotation=annotation), ast.Constant(value=None))
            )
        else:
            required.append((ast.arg(arg=inp["name"], annotation=annotation), None))

    # Dependency-derived args. delegate_to_runtime / external_input are
    # required (no default); we tack them on as kw-only via positional after
    # existing required args. They all have annotation set.
    dep_args: list[tuple[ast.arg, ast.expr | None]] = []
    if deps_artefact is not None:
        for arg, default in zip(deps_artefact.pipeline_args, deps_artefact.pipeline_defaults):
            dep_args.append((arg, default))

    # Required first (no defaults), then optional, then required deps, then optional deps.
    # Python forbids non-default after default — so we group: required (no
    # defaults), optional (defaults), required-from-deps (no defaults if any
    # optional precedes, otherwise tack onto required). The simplest correct
    # ordering: required-inputs + required-deps + optional-inputs + optional-deps.
    required_dep_args = [(a, d) for (a, d) in dep_args if d is None]
    optional_dep_args = [(a, d) for (a, d) in dep_args if d is not None]

    for a, _ in required:
        args.append(a)
    for a, _ in required_dep_args:
        args.append(a)
    for a, d in optional:
        args.append(a)
        defaults.append(d)
    for a, d in optional_dep_args:
        args.append(a)
        if d is not None:
            defaults.append(d)

    return ast.arguments(
        posonlyargs=[],
        args=args,
        kwonlyargs=[],
        kw_defaults=[],
        defaults=defaults,
    )


# --- Helpers --------------------------------------------------------------


def _build_return_stmt(descriptor: dict[str, Any], ctx: RenderContext) -> ast.stmt:
    """Build a return statement targeting whichever node captured the first output."""
    outputs = descriptor.get("outputs", [])
    if not outputs:
        return ast.Return(value=ast.Constant(value=None))
    out_name = outputs[0]["name"]
    cap_id = _find_capture_id(descriptor.get("pipeline", []), out_name)
    if cap_id is None:
        # Fall back to the last call node in execution order.
        last = _last_call_id(descriptor.get("pipeline", []))
        if last is None:
            return ast.Return(value=ast.Constant(value=None))
        cap_id = last
    return ast.Return(value=ast.Name(id=cap_id))


def _find_capture_id(nodes: list[dict[str, Any]], out_name: str) -> str | None:
    for n in _walk(nodes):
        caps = n.get("captures") or {}
        for slot in caps.values():
            if isinstance(slot, str) and slot.endswith(f"/{out_name}"):
                return n["id"]
    return None


def _last_call_id(nodes: list[dict[str, Any]]) -> str | None:
    last: str | None = None
    for n in _walk(nodes):
        if n.get("kind") == "call":
            last = n.get("id", last)
    return last


def _walk(nodes: list[dict[str, Any]]):
    """Yield every descriptor node, recursing into composition shapes."""
    for n in nodes:
        if not isinstance(n, dict):
            continue
        yield n
        if n.get("kind") != "composition":
            continue
        yield from _walk(n.get("body", []) or [])
        for branch in (n.get("cases") or {}).values():
            yield from _walk(branch or [])
        for branch in (n.get("branches") or []):
            yield from _walk(branch or [])
        on = n.get("on")
        if isinstance(on, dict):
            yield from _walk(on.get("body", []) or [])


def _collect_schema_names(descriptor: dict[str, Any], ctx: RenderContext) -> set[str]:
    """Collect schema names referenced anywhere in the descriptor + ctx uses.

    v0.3-aware: also walks ``dict`` value-types and ``list``-of-refs.
    """
    names: set[str] = set(ctx.schema_uses)
    declared = set(descriptor.get("schemas", {}).keys())

    def _walk(sch: dict[str, Any]) -> None:
        if not isinstance(sch, dict):
            return
        ref = sch.get("ref")
        if ref:
            names.add(ref.split("/")[-1])
        of = sch.get("of")
        if isinstance(of, dict):
            _walk(of)
        vt = sch.get("value_type")
        if isinstance(vt, dict):
            _walk(vt)

    for slot in list(descriptor.get("inputs", [])) + list(descriptor.get("outputs", [])):
        _walk(slot.get("schema", {}))
    return {n for n in names if n in declared}


def _ensure_annotation_imports(descriptor: dict[str, Any], ctx: RenderContext) -> None:
    """Walk inputs/outputs and inject imports for v0.3 annotation kinds.

    - ``literal_union`` → ``from typing import Literal``
    - ``image`` / ``document`` → ``from pathlib import Path``
    - bare ``Any`` from fallbacks → ``from typing import Any``
    """
    needs_any = False
    needs_literal = False
    needs_path = False

    def _walk(sch: dict[str, Any]) -> None:
        nonlocal needs_any, needs_literal, needs_path
        if not isinstance(sch, dict):
            return
        kind = sch.get("kind")
        if kind == "literal_union":
            needs_literal = True
        elif kind in ("image", "document"):
            needs_path = True
        elif kind == "dict":
            needs_any = needs_any or (not sch.get("value_type"))
            vt = sch.get("value_type")
            if isinstance(vt, dict):
                _walk(vt)
        elif kind == "list":
            of = sch.get("of")
            if isinstance(of, dict):
                _walk(of)
            else:
                needs_any = True

    for slot in list(descriptor.get("inputs", [])) + list(descriptor.get("outputs", [])):
        _walk(slot.get("schema", {}))

    if needs_any:
        ctx.add_import("typing", "Any")
    if needs_literal:
        ctx.add_import("typing", "Literal")
    if needs_path:
        ctx.add_import("pathlib", "Path")


def _schema_annotation(schema: dict[str, Any]) -> ast.expr:
    """Lower an input/output ``schema`` dict to a Python type-annotation AST.

    Handles all v0.1/v0.2 and v0.3-permissiveness shapes (RFC §3.5):

      - scalars: ``str`` / ``int`` / ``float`` / ``bool``
      - ``ref``: bare schema-class identifier
      - ``model_ref`` / ``pydantic_model``: bare schema-class identifier
      - ``list`` with ``of: {ref}`` or ``of: str``: ``list[T]``
      - ``dict`` with ``key_type`` + ``value_type``: ``dict[K, V]``
      - ``literal_union`` with ``members: [...]``: ``Literal[...]``
      - ``image`` / ``document`` (v0.3 G8.4): ``bytes | Path``

    Falls back to ``Any`` for unknown shapes. The caller is responsible for
    registering ``Any``/``Literal``/``Path`` imports on the context.
    """
    if not isinstance(schema, dict):
        return ast.Name(id="Any")
    if "ref" in schema:
        return ast.Name(id=schema["ref"].split("/")[-1])
    kind = schema.get("kind")
    if kind == "str":
        return ast.Name(id="str")
    if kind == "int":
        return ast.Name(id="int")
    if kind == "float":
        return ast.Name(id="float")
    if kind == "bool":
        return ast.Name(id="bool")
    if kind == "list":
        of = schema.get("of")
        if isinstance(of, dict):
            inner: ast.expr = _schema_annotation(of)
        elif isinstance(of, str):
            inner = _schema_annotation({"kind": of})
        else:
            inner = ast.Name(id="Any")
        return ast.Subscript(value=ast.Name(id="list"), slice=inner)
    if kind == "dict":
        kt = schema.get("key_type", "str")
        vt = schema.get("value_type", "Any")
        key_ann = _schema_annotation({"kind": kt}) if isinstance(kt, str) else _schema_annotation(kt)
        if isinstance(vt, dict):
            val_ann = _schema_annotation(vt)
        elif isinstance(vt, str):
            val_ann = _schema_annotation({"kind": vt}) if vt in ("str", "int", "float", "bool") else ast.Name(id=vt)
        else:
            val_ann = ast.Name(id="Any")
        return ast.Subscript(
            value=ast.Name(id="dict"),
            slice=ast.Tuple(elts=[key_ann, val_ann], ctx=ast.Load()),
        )
    if kind == "literal_union":
        members = schema.get("members") or []
        return ast.Subscript(
            value=ast.Name(id="Literal"),
            slice=ast.Tuple(
                elts=[ast.Constant(value=m) for m in members],
                ctx=ast.Load(),
            ),
        )
    if kind in ("image", "document"):
        # bytes | Path — pass-through to Mellea's image/document args.
        return ast.BinOp(
            left=ast.Name(id="bytes"),
            op=ast.BitOr(),
            right=ast.Name(id="Path"),
        )
    if kind in ("model_ref", "pydantic_model") and "ref" in schema:
        return ast.Name(id=schema["ref"].split("/")[-1])
    return ast.Name(id="Any")


def _input_annotation(inp: dict[str, Any]) -> ast.expr:
    schema = inp.get("schema", {})
    return _schema_annotation(schema)


def _output_annotation(outputs: list[dict[str, Any]]) -> ast.expr:
    if not outputs:
        return ast.Constant(value=None)
    schema = outputs[0].get("schema", {})
    return _schema_annotation(schema)
