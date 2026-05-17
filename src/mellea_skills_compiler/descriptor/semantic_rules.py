"""Semantic (Python-level) validation rules for mellea-fy descriptors.

These rules sit on top of the JSON-Schema-level validator in
:mod:`.validator`. They check things JSON Schema cannot express:

* **Symbol resolution** (``R-SEM-SYMBOL``) — every ``symbol`` referenced in
  ``state[]`` and pipeline ``call`` nodes must resolve in
  ``surface["modules"]``.
* **Ref resolution** (``R-SEM-REF``) — every ``{ref}`` must resolve to a
  declared ``state.id``, ``inputs[].name``, or a prior pipeline node ``id``
  in declaration order (with loop-variable ``item_id`` scopes added by
  enclosing ``map``/``filter`` operators).
* **Schema-ref resolution** (``R-SEM-SCHEMA-REF``) — every
  ``{schema_ref: "#/schemas/X"}`` must point to a declared schema.
* **Composition operator required fields** (``R-SEM-OPERATOR``) — per-operator
  required-field check (e.g. ``map`` needs ``over``, ``body``, ``collect``,
  ``item_id``).
* **on_parse_failure default shape** (``R-SEM-FMT-DEFAULT``) — v0.2 only;
  if ``on_parse_failure.strategy == "default"``, the top-level fields of the
  ``default`` value must match the fields of the referenced ``format.schema_ref``
  schema. Best-effort: field-name match only, not deep type-checking.

Symbol resolution understands the dotted-path convention used by the
mellea surface JSON: ``module.path.Class.member`` resolves by walking
``modules[module.path].symbols[Class].members[member]``; a bare module-level
symbol resolves by ``modules[module.path].symbols[Class]``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .validator import ValidationError

# Rule identifiers (kept stable across releases so downstream tooling can pin to them).
R_SYMBOL = "R-SEM-SYMBOL"
R_REF = "R-SEM-REF"
R_SCHEMA_REF = "R-SEM-SCHEMA-REF"
R_OPERATOR = "R-SEM-OPERATOR"
R_FMT_DEFAULT = "R-SEM-FMT-DEFAULT"
R_SIGNATURE_MATCH = "R-SEM-SIGNATURE-MATCH"

# v0.3 rules (P3.5.B).
R_MODALITY_TOOL_LOOP = "R-SEM-MODALITY-TOOL-LOOP"
R_MODALITY_STREAMING = "R-SEM-MODALITY-STREAMING"
R_MODALITY_IMAGE_INPUT = "R-SEM-MODALITY-IMAGE-INPUT"
R_MODALITY_APPROVAL = "R-SEM-MODALITY-APPROVAL"
R_CALL_ARGS_MATCH_SIG = "R-SEM-CALL-ARGS-MATCH-SIG"
R_REF_SELECT_RESOLVES = "R-SEM-REF-SELECT-RESOLVES"
R_DISPOSITION_CONSISTENT = "R-SEM-DISPOSITION-CONSISTENT"
R_BUNDLED_PATH_EXISTS = "R-SEM-BUNDLED-PATH-EXISTS"
R_AGENT_LOOP_FIELDS = "R-SEM-AGENT-LOOP-FIELDS"
R_SOURCE_ELEMENT_RESOLVES = "R-SEM-SOURCE-ELEMENT-RESOLVES"
R_FIELD_TYPE_KNOWN = "R-SEM-FIELD-TYPE-KNOWN"

# Per-operator required field sets. Used as a Python-level cross-check in
# addition to the JSON Schema, because the schema's discriminated-union over
# composition nodes can yield confusing top-level "oneOf" failures.
_OPERATOR_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "sequential": ("body",),
    "branch": ("on", "cases"),
    "parallel": ("branches",),
    "map": ("over", "body", "collect", "item_id"),
    "retry_with_feedback": ("body", "max_attempts", "failure_check"),
    "filter": ("over", "predicate", "item_id"),  # v0.2 only
    "agent_loop": (
        "max_iterations",
        "state_schema",
        "termination_condition",
        "body",
    ),  # v0.3 only
    "human_approval": ("prompt", "branches"),  # v0.3 only
}

# Closed vocabulary for SchemaField.type strings — R-SEM-FIELD-TYPE-KNOWN.
_KNOWN_FIELD_TYPE_TOKENS: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "any", "Any", "model_ref", "enum_ref"}
)

# Top-level parameterised constructors accepted by the field-type vocabulary.
_KNOWN_FIELD_TYPE_PARAMETERISED: frozenset[str] = frozenset(
    {"list", "dict", "Optional", "Literal", "Union"}
)


def run_all(
    descriptor: dict[str, Any],
    schema_version: str,
    surface: dict[str, Any] | None,
    expected_signature: dict[str, Any] | None = None,
    skill_root: str | Path | None = None,
    inventory: dict[str, Any] | None = None,
) -> list[ValidationError]:
    """Run all semantic rules. Returns an empty list iff all pass.

    Args:
        descriptor: the parsed descriptor dict.
        schema_version: the descriptor schema version being validated against.
        surface: the introspected Mellea surface JSON (for R-SEM-SYMBOL).
        expected_signature: optional canonical I/O signature derived by Steps
            0-2.5. When provided, the R-SEM-SIGNATURE-MATCH rule fires;
            when ``None``, the rule is a no-op so unit tests on raw
            descriptors without intermediate artefacts still pass.
        skill_root: optional path to the skill root, for
            R-SEM-BUNDLED-PATH-EXISTS. When omitted, the rule degrades to a
            warning.
        inventory: optional parsed ``inventory.json``. When omitted,
            R-SEM-SOURCE-ELEMENT-RESOLVES degrades to a warning.
    """
    errors: list[ValidationError] = []

    if not isinstance(descriptor, dict):
        # The JSON-Schema layer will have already complained; nothing useful
        # to do here.
        return errors

    state = descriptor.get("state") if isinstance(descriptor.get("state"), list) else []
    inputs = descriptor.get("inputs") if isinstance(descriptor.get("inputs"), list) else []
    pipeline = (
        descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    )
    schemas = (
        descriptor.get("schemas") if isinstance(descriptor.get("schemas"), dict) else {}
    )

    state_ids = {
        s.get("id") for s in state if isinstance(s, dict) and isinstance(s.get("id"), str)
    }
    input_names = {
        i.get("name")
        for i in inputs
        if isinstance(i, dict) and isinstance(i.get("name"), str)
    }

    # Symbol resolution (state).
    if surface is not None:
        for idx, entry in enumerate(state):
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("symbol")
            if isinstance(symbol, str) and not _symbol_in_surface(symbol, surface):
                errors.append(
                    ValidationError(
                        path=f"/state/{idx}/symbol",
                        rule=R_SYMBOL,
                        message=(
                            f"symbol {symbol!r} does not resolve in surface "
                            f"modules"
                        ),
                    )
                )

    # Pipeline-level rules. Walk the pipeline tree carrying a scope of
    # "ids visible at this point": prior siblings + enclosing-map item_ids.
    errors.extend(
        _walk_pipeline(
            pipeline=pipeline,
            state_ids=state_ids,
            input_names=input_names,
            schemas=schemas,
            surface=surface,
            schema_version=schema_version,
            path_prefix="/pipeline",
            enclosing_ids=frozenset(),
        )
    )

    # R-SEM-SIGNATURE-MATCH (P3.5.D): compare the descriptor's I/O signature
    # against the canonical expected_signature derived from Steps 0-2.5.
    # Non-firing when no expected_signature is provided (existing tests on raw
    # descriptors without intermediate artefacts continue to pass).
    if expected_signature is not None:
        errors.extend(
            _check_expected_signature(
                descriptor=descriptor,
                expected_signature=expected_signature,
            )
        )

    # v0.3 rules: only fire when validating against the v0.3 schema. The
    # preconditions on each rule mean v0.1/v0.2 descriptors that lack the
    # relevant surface (no classification block, no dependencies block, etc.)
    # are not affected even if dispatched through these rules.
    if schema_version == "0.3":
        errors.extend(_check_field_type_known(descriptor=descriptor))
        errors.extend(_check_modality_rules(descriptor=descriptor))
        errors.extend(
            _check_dispositions(descriptor=descriptor, surface=surface)
        )
        errors.extend(
            _check_bundled_resources(
                descriptor=descriptor, skill_root=skill_root
            )
        )
        errors.extend(_check_agent_loop_fields(descriptor=descriptor))
        errors.extend(
            _check_source_elements(
                descriptor=descriptor, inventory=inventory
            )
        )
        # Layered-defence rules ship at warning severity per RFC §4.
        errors.extend(
            _check_call_args_match_sig(
                descriptor=descriptor, surface=surface
            )
        )
        errors.extend(_check_ref_select_resolves(descriptor=descriptor))

    return errors


def _walk_pipeline(
    *,
    pipeline: list[Any],
    state_ids: set[str | None],
    input_names: set[str | None],
    schemas: dict[str, Any],
    surface: dict[str, Any] | None,
    schema_version: str,
    path_prefix: str,
    enclosing_ids: frozenset[str],
) -> list[ValidationError]:
    """Walk a pipeline node array, validating each node.

    ``enclosing_ids`` carries the ids bound by surrounding scopes (loop
    variables from ``map``/``filter``, or sibling-node ids accumulated as we
    iterate the current array).
    """
    errors: list[ValidationError] = []
    visible_in_this_scope = set(enclosing_ids)
    for idx, node in enumerate(pipeline):
        node_path = f"{path_prefix}/{idx}"
        if not isinstance(node, dict):
            continue

        visible = (
            visible_in_this_scope
            | {s for s in state_ids if isinstance(s, str)}
            | {n for n in input_names if isinstance(n, str)}
        )

        if node.get("kind") == "call":
            errors.extend(
                _validate_call_node(
                    node=node,
                    node_path=node_path,
                    visible_ids=visible,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                )
            )
        elif node.get("kind") == "composition":
            errors.extend(
                _validate_composition_node(
                    node=node,
                    node_path=node_path,
                    state_ids=state_ids,
                    input_names=input_names,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                    enclosing_ids=frozenset(visible),
                )
            )

        nid = node.get("id")
        if isinstance(nid, str):
            visible_in_this_scope.add(nid)

    return errors


def _validate_call_node(
    *,
    node: dict[str, Any],
    node_path: str,
    visible_ids: set[str],
    schemas: dict[str, Any],
    surface: dict[str, Any] | None,
    schema_version: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []

    # Symbol resolution.
    symbol = node.get("symbol")
    if surface is not None and isinstance(symbol, str) and not _symbol_in_surface(symbol, surface):
        errors.append(
            ValidationError(
                path=f"{node_path}/symbol",
                rule=R_SYMBOL,
                message=f"symbol {symbol!r} does not resolve in surface modules",
            )
        )

    # bound_to ref.
    bound_to = node.get("bound_to")
    if isinstance(bound_to, dict) and isinstance(bound_to.get("ref"), str):
        target = bound_to["ref"]
        if target not in visible_ids:
            errors.append(
                ValidationError(
                    path=f"{node_path}/bound_to/ref",
                    rule=R_REF,
                    message=(
                        f"bound_to ref {target!r} does not resolve to a "
                        f"declared state.id, input name, or prior node id"
                    ),
                )
            )

    # args traversal — refs + schema_refs.
    args = node.get("args")
    if isinstance(args, dict):
        errors.extend(
            _check_argvalue_tree(
                value=args,
                path=f"{node_path}/args",
                visible_ids=visible_ids,
                schemas=schemas,
            )
        )

    # captures: each value must look like '#/outputs/<name>'. JSON-Schema
    # enforces the prefix; no extra Python check needed here.

    # v0.2-only: on_parse_failure default shape.
    if schema_version == "0.2":
        errors.extend(_check_on_parse_failure(node=node, node_path=node_path, schemas=schemas))

    return errors


def _validate_composition_node(
    *,
    node: dict[str, Any],
    node_path: str,
    state_ids: set[str | None],
    input_names: set[str | None],
    schemas: dict[str, Any],
    surface: dict[str, Any] | None,
    schema_version: str,
    enclosing_ids: frozenset[str],
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    operator = node.get("operator")
    if not isinstance(operator, str):
        return errors

    required = _OPERATOR_REQUIRED_FIELDS.get(operator)
    if required is not None:
        for fname in required:
            if fname not in node:
                errors.append(
                    ValidationError(
                        path=f"{node_path}/{fname}",
                        rule=R_OPERATOR,
                        message=(
                            f"composition operator {operator!r} requires "
                            f"field {fname!r}"
                        ),
                    )
                )

    visible = (
        set(enclosing_ids)
        | {s for s in state_ids if isinstance(s, str)}
        | {n for n in input_names if isinstance(n, str)}
    )

    # branch.on
    if operator == "branch":
        on = node.get("on")
        if isinstance(on, dict) and isinstance(on.get("ref"), str):
            if on["ref"] not in visible:
                errors.append(
                    ValidationError(
                        path=f"{node_path}/on/ref",
                        rule=R_REF,
                        message=f"branch.on ref {on['ref']!r} does not resolve",
                    )
                )
        cases = node.get("cases")
        if isinstance(cases, dict):
            for case_name, body in cases.items():
                if isinstance(body, list):
                    errors.extend(
                        _walk_pipeline(
                            pipeline=body,
                            state_ids=state_ids,
                            input_names=input_names,
                            schemas=schemas,
                            surface=surface,
                            schema_version=schema_version,
                            path_prefix=f"{node_path}/cases/{case_name}",
                            enclosing_ids=frozenset(visible),
                        )
                    )

    elif operator == "sequential":
        body = node.get("body")
        if isinstance(body, list):
            errors.extend(
                _walk_pipeline(
                    pipeline=body,
                    state_ids=state_ids,
                    input_names=input_names,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                    path_prefix=f"{node_path}/body",
                    enclosing_ids=frozenset(visible),
                )
            )

    elif operator == "parallel":
        branches = node.get("branches")
        if isinstance(branches, list):
            for b_idx, branch in enumerate(branches):
                if isinstance(branch, list):
                    errors.extend(
                        _walk_pipeline(
                            pipeline=branch,
                            state_ids=state_ids,
                            input_names=input_names,
                            schemas=schemas,
                            surface=surface,
                            schema_version=schema_version,
                            path_prefix=f"{node_path}/branches/{b_idx}",
                            enclosing_ids=frozenset(visible),
                        )
                    )

    elif operator == "map":
        over = node.get("over")
        if isinstance(over, dict) and isinstance(over.get("ref"), str):
            if over["ref"] not in visible:
                errors.append(
                    ValidationError(
                        path=f"{node_path}/over/ref",
                        rule=R_REF,
                        message=f"map.over ref {over['ref']!r} does not resolve",
                    )
                )
        item_id = node.get("item_id")
        body = node.get("body")
        if isinstance(body, list):
            inner = set(visible)
            if isinstance(item_id, str):
                inner.add(item_id)
            errors.extend(
                _walk_pipeline(
                    pipeline=body,
                    state_ids=state_ids,
                    input_names=input_names,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                    path_prefix=f"{node_path}/body",
                    enclosing_ids=frozenset(inner),
                )
            )

    elif operator == "retry_with_feedback":
        failure_check = node.get("failure_check")
        if isinstance(failure_check, dict) and isinstance(failure_check.get("ref"), str):
            # failure_check ref is allowed to point at body siblings; defer check
            # by including body ids in the scope.
            body = node.get("body")
            inner_ids = set(visible)
            if isinstance(body, list):
                for n in body:
                    if isinstance(n, dict) and isinstance(n.get("id"), str):
                        inner_ids.add(n["id"])
            if failure_check["ref"] not in inner_ids:
                errors.append(
                    ValidationError(
                        path=f"{node_path}/failure_check/ref",
                        rule=R_REF,
                        message=(
                            f"retry.failure_check ref {failure_check['ref']!r} "
                            f"does not resolve to a body node id or outer scope"
                        ),
                    )
                )
        body = node.get("body")
        if isinstance(body, list):
            errors.extend(
                _walk_pipeline(
                    pipeline=body,
                    state_ids=state_ids,
                    input_names=input_names,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                    path_prefix=f"{node_path}/body",
                    enclosing_ids=frozenset(visible),
                )
            )

    elif operator == "filter":
        # v0.2 only — JSON Schema rejects this on v0.1.
        over = node.get("over")
        if isinstance(over, dict) and isinstance(over.get("ref"), str):
            if over["ref"] not in visible:
                errors.append(
                    ValidationError(
                        path=f"{node_path}/over/ref",
                        rule=R_REF,
                        message=f"filter.over ref {over['ref']!r} does not resolve",
                    )
                )
        # No body to recurse into; the predicate is evaluated by the renderer
        # against the bound item_id, which is local-only.

    elif operator == "agent_loop":
        body = node.get("body")
        if isinstance(body, list):
            errors.extend(
                _walk_pipeline(
                    pipeline=body,
                    state_ids=state_ids,
                    input_names=input_names,
                    schemas=schemas,
                    surface=surface,
                    schema_version=schema_version,
                    path_prefix=f"{node_path}/body",
                    enclosing_ids=frozenset(visible),
                )
            )

    elif operator == "human_approval":
        branches = node.get("branches")
        if isinstance(branches, dict):
            for branch_name in ("approved", "rejected"):
                body = branches.get(branch_name)
                if isinstance(body, list):
                    errors.extend(
                        _walk_pipeline(
                            pipeline=body,
                            state_ids=state_ids,
                            input_names=input_names,
                            schemas=schemas,
                            surface=surface,
                            schema_version=schema_version,
                            path_prefix=f"{node_path}/branches/{branch_name}",
                            enclosing_ids=frozenset(visible),
                        )
                    )

    return errors


def _check_argvalue_tree(
    *,
    value: Any,
    path: str,
    visible_ids: set[str],
    schemas: dict[str, Any],
) -> list[ValidationError]:
    """Recursively walk an argument-tree, checking refs and schema_refs."""
    errors: list[ValidationError] = []
    if isinstance(value, dict):
        if isinstance(value.get("ref"), str):
            target = value["ref"]
            if target not in visible_ids:
                errors.append(
                    ValidationError(
                        path=f"{path}/ref",
                        rule=R_REF,
                        message=(
                            f"ref {target!r} does not resolve to a declared "
                            f"state.id, input name, or prior node id"
                        ),
                    )
                )
        if isinstance(value.get("schema_ref"), str):
            schema_ref = value["schema_ref"]
            schema_name = _schema_name_from_ref(schema_ref)
            if schema_name is None or schema_name not in schemas:
                errors.append(
                    ValidationError(
                        path=f"{path}/schema_ref",
                        rule=R_SCHEMA_REF,
                        message=(
                            f"schema_ref {schema_ref!r} does not point at a "
                            f"declared schema in #/schemas/"
                        ),
                    )
                )
        # Nested symbol-call (argument value of the form
        # {"symbol": "...", "args": {...}}). Recurse into args.
        if isinstance(value.get("args"), dict):
            for key, sub in value["args"].items():
                errors.extend(
                    _check_argvalue_tree(
                        value=sub,
                        path=f"{path}/args/{key}",
                        visible_ids=visible_ids,
                        schemas=schemas,
                    )
                )
        # Top-level dict-of-args (e.g. {description: {...}, format: {...}}).
        for key, sub in value.items():
            if key in ("ref", "select", "schema_ref", "value", "env", "template", "symbol", "args"):
                continue
            errors.extend(
                _check_argvalue_tree(
                    value=sub,
                    path=f"{path}/{key}",
                    visible_ids=visible_ids,
                    schemas=schemas,
                )
            )
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            errors.extend(
                _check_argvalue_tree(
                    value=sub,
                    path=f"{path}/{i}",
                    visible_ids=visible_ids,
                    schemas=schemas,
                )
            )
    return errors


def _check_on_parse_failure(
    *,
    node: dict[str, Any],
    node_path: str,
    schemas: dict[str, Any],
) -> list[ValidationError]:
    """v0.2 only: validate the on_parse_failure default's top-level shape."""
    opf = node.get("on_parse_failure")
    if not isinstance(opf, dict):
        return []
    if opf.get("strategy") != "default":
        return []
    if "default" not in opf:
        return [
            ValidationError(
                path=f"{node_path}/on_parse_failure",
                rule=R_FMT_DEFAULT,
                message=(
                    "on_parse_failure.strategy=='default' but no 'default' "
                    "value was provided"
                ),
            )
        ]

    default = opf["default"]

    # Find the format.schema_ref on the call's args.
    args = node.get("args") if isinstance(node.get("args"), dict) else {}
    format_arg = args.get("format")
    schema_ref = None
    if isinstance(format_arg, dict):
        schema_ref = format_arg.get("schema_ref")
    if not isinstance(schema_ref, str):
        return [
            ValidationError(
                path=f"{node_path}/on_parse_failure",
                rule=R_FMT_DEFAULT,
                message=(
                    "on_parse_failure.default specified but call has no "
                    "args.format.schema_ref to validate the default shape "
                    "against"
                ),
            )
        ]

    schema_name = _schema_name_from_ref(schema_ref)
    target_schema = schemas.get(schema_name) if isinstance(schema_name, str) else None
    if not isinstance(target_schema, dict):
        # schema_ref doesn't resolve; the schema_ref rule will report that
        # separately, no need to double-flag.
        return []

    if target_schema.get("kind") != "model":
        # For non-model targets (enum, primitive), we can't usefully check
        # field shape; accept as best-effort.
        return []

    declared_fields = (
        set(target_schema.get("fields", {}).keys())
        if isinstance(target_schema.get("fields"), dict)
        else set()
    )

    if not isinstance(default, dict):
        return [
            ValidationError(
                path=f"{node_path}/on_parse_failure/default",
                rule=R_FMT_DEFAULT,
                message=(
                    f"default must be an object matching schema "
                    f"{schema_name!r}; got {type(default).__name__}"
                ),
            )
        ]

    errors: list[ValidationError] = []
    extra = set(default.keys()) - declared_fields
    if extra:
        errors.append(
            ValidationError(
                path=f"{node_path}/on_parse_failure/default",
                rule=R_FMT_DEFAULT,
                message=(
                    f"default has fields not declared on schema "
                    f"{schema_name!r}: {sorted(extra)}"
                ),
            )
        )

    # Note: missing-required-fields is intentionally a warning, not an error.
    # A descriptor author may legitimately want a "partial default" that the
    # renderer fills out with Pydantic defaults / Optional[None]s.
    required_fields = {
        name
        for name, spec in (target_schema.get("fields") or {}).items()
        if isinstance(spec, dict) and not spec.get("optional", False)
    }
    missing = required_fields - set(default.keys())
    if missing:
        errors.append(
            ValidationError(
                path=f"{node_path}/on_parse_failure/default",
                rule=R_FMT_DEFAULT,
                message=(
                    f"default is missing fields declared as required on "
                    f"schema {schema_name!r}: {sorted(missing)}"
                ),
                severity="warning",
            )
        )

    return errors


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _schema_name_from_ref(schema_ref: str) -> str | None:
    """Extract ``X`` from ``"#/schemas/X"``; return None if malformed."""
    prefix = "#/schemas/"
    if not schema_ref.startswith(prefix):
        return None
    tail = schema_ref[len(prefix):]
    if "/" in tail or not tail:
        return None
    return tail


def _symbol_in_surface(symbol: str, surface: dict[str, Any]) -> bool:
    """Return True iff ``symbol`` resolves in the surface JSON.

    The surface JSON shape is::

        {
          "modules": {
            "<module.path>": {
              "symbols": {
                "<SymbolName>": {
                  "kind": "class" | "function" | ...,
                  "members": { "<member>": {...}, ... }
                }
              }
            }
          }
        }

    ``symbol`` is a dotted path. The resolver greedily picks the longest
    module-path prefix that exists in ``modules``; everything after is
    expected to be either a top-level symbol or a ``Class.member`` path.
    """
    if not isinstance(surface, dict):
        return False
    modules = surface.get("modules")
    if not isinstance(modules, dict):
        return False

    parts = symbol.split(".")
    if not parts:
        return False

    # Try every prefix length, longest first, to handle dotted module paths.
    for split_at in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split_at])
        rest = parts[split_at:]
        module = modules.get(module_path)
        if not isinstance(module, dict):
            continue
        symbols = module.get("symbols")
        if not isinstance(symbols, dict):
            continue

        # First token must be a top-level symbol in this module.
        head = rest[0]
        sym = symbols.get(head)
        if not isinstance(sym, dict):
            continue

        if len(rest) == 1:
            return True

        # Walk class members for the remaining tokens.
        current: Any = sym
        ok = True
        for token in rest[1:]:
            members = current.get("members") if isinstance(current, dict) else None
            if not isinstance(members, dict):
                ok = False
                break
            nxt = members.get(token)
            if not isinstance(nxt, dict):
                ok = False
                break
            current = nxt
        if ok:
            return True

    return False


# --------------------------------------------------------------------------- #
# R-SEM-SIGNATURE-MATCH (P3.5.D)
# --------------------------------------------------------------------------- #


def _slot_type_string(schema_ref: Any) -> str:
    """Render a descriptor Slot.schema (a SchemaRef) as a Python type string.

    Mirrors the grammar accepted by ``expected_signature.schema.json``'s
    ``signature_param.type`` field so the two sides can be compared by
    string equality.
    """
    if not isinstance(schema_ref, dict):
        return ""
    kind = schema_ref.get("kind")
    if kind in ("str", "int", "float", "bool"):
        return str(kind)
    if kind in ("model_ref", "enum_ref", "pydantic_model"):
        ref = schema_ref.get("ref")
        if isinstance(ref, str) and ref.startswith("#/schemas/"):
            return ref[len("#/schemas/"):]
        return ""
    if kind == "list":
        inner = _slot_type_string(schema_ref.get("of"))
        return f"list[{inner}]" if inner else "list"
    return str(kind) if isinstance(kind, str) else ""


def _slot_schema_ref(schema_ref: Any) -> str | None:
    """Extract the '#/schemas/X' ref from a descriptor SchemaRef when present."""
    if not isinstance(schema_ref, dict):
        return None
    if schema_ref.get("kind") in ("model_ref", "enum_ref", "pydantic_model"):
        ref = schema_ref.get("ref")
        if isinstance(ref, str) and ref.startswith("#/schemas/"):
            return ref
    if schema_ref.get("kind") == "list":
        return _slot_schema_ref(schema_ref.get("of"))
    return None


def _check_expected_signature(
    *,
    descriptor: dict[str, Any],
    expected_signature: dict[str, Any],
) -> list[ValidationError]:
    """R-SEM-SIGNATURE-MATCH — compare descriptor I/O against the locked signature.

    The locked signature is produced by /mellea-fy-map (or a future Step 2.6)
    from ``intermediate/element_mapping.json`` + ``intermediate/classification.json``
    and consumed by Step 4 (fixture generation) BEFORE the descriptor is
    emitted. This rule enforces the constraint after descriptor emission:
    any divergence is a repair-loop trigger so the descriptor's
    ``inputs``/``outputs`` (and the schemas they reference) snap back to
    the canonical shape.

    Mismatches produce structured errors naming the specific field/type
    discrepancy.
    """
    errors: list[ValidationError] = []

    expected_inputs = expected_signature.get("inputs") or []
    expected_outputs = expected_signature.get("outputs") or []
    expected_schemas = {
        s.get("name"): s
        for s in (expected_signature.get("schemas") or [])
        if isinstance(s, dict) and isinstance(s.get("name"), str)
    }

    desc_inputs = descriptor.get("inputs") if isinstance(descriptor.get("inputs"), list) else []
    desc_outputs = descriptor.get("outputs") if isinstance(descriptor.get("outputs"), list) else []
    desc_schemas = (
        descriptor.get("schemas") if isinstance(descriptor.get("schemas"), dict) else {}
    )

    # --- Inputs ----------------------------------------------------------
    expected_input_names = [
        e.get("name") for e in expected_inputs if isinstance(e, dict)
    ]
    desc_input_names = [d.get("name") for d in desc_inputs if isinstance(d, dict)]

    expected_set = set(expected_input_names)
    desc_set = set(desc_input_names)

    for name in sorted(expected_set - desc_set):
        if not isinstance(name, str):
            continue
        # Skip explicitly-optional missing inputs only when the expected
        # entry was marked optional AND not required.
        ent = next(
            (e for e in expected_inputs if isinstance(e, dict) and e.get("name") == name),
            None,
        )
        is_optional = bool(ent and ent.get("optional"))
        errors.append(
            ValidationError(
                path="/inputs",
                rule=R_SIGNATURE_MATCH,
                message=(
                    f"descriptor is missing required input {name!r} declared in "
                    f"expected_signature.inputs"
                    + ("" if not is_optional else " (marked optional)")
                ),
                severity="error" if not is_optional else "warning",
            )
        )

    for name in sorted(desc_set - expected_set):
        if not isinstance(name, str):
            continue
        errors.append(
            ValidationError(
                path="/inputs",
                rule=R_SIGNATURE_MATCH,
                message=(
                    f"descriptor declares extra input {name!r} not present in "
                    f"expected_signature.inputs (expected: {sorted(n for n in expected_set if isinstance(n, str))})"
                ),
            )
        )

    # Per-name type comparison for inputs present on both sides.
    for idx, exp in enumerate(expected_inputs):
        if not isinstance(exp, dict):
            continue
        ename = exp.get("name")
        if not isinstance(ename, str) or ename not in desc_set:
            continue
        desc_entry = next(
            (d for d in desc_inputs if isinstance(d, dict) and d.get("name") == ename),
            None,
        )
        if desc_entry is None:
            continue
        exp_type = exp.get("type") or ""
        desc_type = _slot_type_string(desc_entry.get("schema"))
        if exp_type and desc_type and exp_type != desc_type:
            errors.append(
                ValidationError(
                    path=f"/inputs/{idx}/schema",
                    rule=R_SIGNATURE_MATCH,
                    message=(
                        f"input {ename!r} type mismatch: expected_signature says "
                        f"{exp_type!r}, descriptor declares {desc_type!r}"
                    ),
                )
            )

    # --- Outputs ---------------------------------------------------------
    expected_output_names = [
        e.get("name") for e in expected_outputs if isinstance(e, dict)
    ]
    desc_output_names = [d.get("name") for d in desc_outputs if isinstance(d, dict)]

    if expected_output_names and desc_output_names:
        exp_out_set = set(expected_output_names)
        desc_out_set = set(desc_output_names)
        for name in sorted(exp_out_set - desc_out_set):
            if not isinstance(name, str):
                continue
            errors.append(
                ValidationError(
                    path="/outputs",
                    rule=R_SIGNATURE_MATCH,
                    message=(
                        f"descriptor is missing output {name!r} declared in "
                        f"expected_signature.outputs"
                    ),
                )
            )
        for name in sorted(desc_out_set - exp_out_set):
            if not isinstance(name, str):
                continue
            errors.append(
                ValidationError(
                    path="/outputs",
                    rule=R_SIGNATURE_MATCH,
                    message=(
                        f"descriptor declares extra output {name!r} not present "
                        f"in expected_signature.outputs"
                    ),
                )
            )
        for idx, exp in enumerate(expected_outputs):
            if not isinstance(exp, dict):
                continue
            ename = exp.get("name")
            if not isinstance(ename, str) or ename not in desc_out_set:
                continue
            desc_entry = next(
                (d for d in desc_outputs if isinstance(d, dict) and d.get("name") == ename),
                None,
            )
            if desc_entry is None:
                continue
            exp_type = exp.get("type") or ""
            desc_type = _slot_type_string(desc_entry.get("schema"))
            if exp_type and desc_type and exp_type != desc_type:
                errors.append(
                    ValidationError(
                        path=f"/outputs/{idx}/schema",
                        rule=R_SIGNATURE_MATCH,
                        message=(
                            f"output {ename!r} type mismatch: expected_signature "
                            f"says {exp_type!r}, descriptor declares {desc_type!r}"
                        ),
                    )
                )

    # --- Schemas ---------------------------------------------------------
    # For every schema referenced by an input or output, cross-check its
    # field set against the expected_signature.schemas entry of the same name.
    referenced_schemas: set[str] = set()
    for slot in list(desc_inputs) + list(desc_outputs):
        if not isinstance(slot, dict):
            continue
        ref = _slot_schema_ref(slot.get("schema"))
        if ref is None:
            continue
        # ref is '#/schemas/X' — extract X.
        name = ref[len("#/schemas/"):]
        if name:
            referenced_schemas.add(name)

    for schema_name in sorted(referenced_schemas):
        exp_schema = expected_schemas.get(schema_name)
        desc_schema = desc_schemas.get(schema_name)
        if exp_schema is None:
            # The expected_signature didn't declare this schema — that's fine;
            # nothing to compare. (If the descriptor invented a schema name
            # the expected_signature doesn't know about, that's caught above
            # as an "extra input/output" mismatch already.)
            continue
        if not isinstance(desc_schema, dict):
            errors.append(
                ValidationError(
                    path=f"/schemas/{schema_name}",
                    rule=R_SIGNATURE_MATCH,
                    message=(
                        f"descriptor is missing schema {schema_name!r} declared "
                        f"in expected_signature.schemas"
                    ),
                )
            )
            continue
        if exp_schema.get("kind") == "model" and desc_schema.get("kind") == "model":
            exp_fields = {
                f.get("name"): f
                for f in (exp_schema.get("fields") or [])
                if isinstance(f, dict) and isinstance(f.get("name"), str)
            }
            desc_fields = (
                desc_schema.get("fields")
                if isinstance(desc_schema.get("fields"), dict)
                else {}
            )

            exp_field_names = set(exp_fields.keys())
            desc_field_names = set(desc_fields.keys())

            for fname in sorted(exp_field_names - desc_field_names):
                ent = exp_fields[fname]
                is_optional = bool(ent.get("optional"))
                errors.append(
                    ValidationError(
                        path=f"/schemas/{schema_name}/fields/{fname}",
                        rule=R_SIGNATURE_MATCH,
                        message=(
                            f"schema {schema_name!r} is missing field {fname!r} "
                            f"declared in expected_signature.schemas"
                            + ("" if not is_optional else " (marked optional)")
                        ),
                        severity="error" if not is_optional else "warning",
                    )
                )
            for fname in sorted(desc_field_names - exp_field_names):
                errors.append(
                    ValidationError(
                        path=f"/schemas/{schema_name}/fields/{fname}",
                        rule=R_SIGNATURE_MATCH,
                        message=(
                            f"schema {schema_name!r} declares extra field "
                            f"{fname!r} not present in expected_signature.schemas"
                        ),
                    )
                )
            for fname in sorted(exp_field_names & desc_field_names):
                exp_type = exp_fields[fname].get("type") or ""
                desc_field = desc_fields[fname]
                desc_type = desc_field.get("type") if isinstance(desc_field, dict) else ""
                desc_type = desc_type or ""
                if exp_type and desc_type and exp_type != desc_type:
                    errors.append(
                        ValidationError(
                            path=f"/schemas/{schema_name}/fields/{fname}",
                            rule=R_SIGNATURE_MATCH,
                            message=(
                                f"schema {schema_name!r} field {fname!r} type "
                                f"mismatch: expected_signature says {exp_type!r}, "
                                f"descriptor declares {desc_type!r}"
                            ),
                        )
                    )
        elif exp_schema.get("kind") == "enum" and desc_schema.get("kind") == "enum":
            exp_members = list(exp_schema.get("members") or [])
            desc_members = list(desc_schema.get("members") or [])
            if exp_members and desc_members and set(exp_members) != set(desc_members):
                errors.append(
                    ValidationError(
                        path=f"/schemas/{schema_name}/members",
                        rule=R_SIGNATURE_MATCH,
                        message=(
                            f"enum {schema_name!r} members mismatch: "
                            f"expected_signature says {sorted(exp_members)!r}, "
                            f"descriptor declares {sorted(desc_members)!r}"
                        ),
                    )
                )

    return errors


# --------------------------------------------------------------------------- #
# v0.3 rules (P3.5.B)
# --------------------------------------------------------------------------- #


def _iter_pipeline_nodes(pipeline: Any):
    """Depth-first walk over every node in a pipeline tree.

    Yields ``(path, node)`` tuples where ``path`` is a JSON Pointer.
    """
    if not isinstance(pipeline, list):
        return
    stack: list[tuple[str, Any]] = []
    for idx, node in enumerate(pipeline):
        stack.append((f"/pipeline/{idx}", node))
    while stack:
        path, node = stack.pop(0)
        if not isinstance(node, dict):
            continue
        yield path, node
        kind = node.get("kind")
        operator = node.get("operator")
        if kind == "composition":
            if operator in ("sequential",):
                body = node.get("body")
                if isinstance(body, list):
                    for i, sub in enumerate(body):
                        stack.append((f"{path}/body/{i}", sub))
            elif operator == "branch":
                cases = node.get("cases")
                if isinstance(cases, dict):
                    for case_name, body in cases.items():
                        if isinstance(body, list):
                            for i, sub in enumerate(body):
                                stack.append(
                                    (f"{path}/cases/{case_name}/{i}", sub)
                                )
            elif operator == "parallel":
                branches = node.get("branches")
                if isinstance(branches, list):
                    for bi, branch in enumerate(branches):
                        if isinstance(branch, list):
                            for i, sub in enumerate(branch):
                                stack.append(
                                    (f"{path}/branches/{bi}/{i}", sub)
                                )
            elif operator == "map":
                body = node.get("body")
                if isinstance(body, list):
                    for i, sub in enumerate(body):
                        stack.append((f"{path}/body/{i}", sub))
            elif operator == "retry_with_feedback":
                body = node.get("body")
                if isinstance(body, list):
                    for i, sub in enumerate(body):
                        stack.append((f"{path}/body/{i}", sub))
            elif operator == "agent_loop":
                body = node.get("body")
                if isinstance(body, list):
                    for i, sub in enumerate(body):
                        stack.append((f"{path}/body/{i}", sub))
            elif operator == "human_approval":
                branches = node.get("branches")
                if isinstance(branches, dict):
                    for name in ("approved", "rejected"):
                        body = branches.get(name)
                        if isinstance(body, list):
                            for i, sub in enumerate(body):
                                stack.append(
                                    (f"{path}/branches/{name}/{i}", sub)
                                )


def _pipeline_has_aact_or_agent_loop(descriptor: dict[str, Any]) -> bool:
    """True iff any pipeline node is a call to ``aact`` or an ``agent_loop``."""
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    for _path, node in _iter_pipeline_nodes(pipeline):
        if node.get("kind") == "call":
            symbol = node.get("symbol")
            if isinstance(symbol, str) and symbol.endswith(".aact"):
                return True
            if isinstance(symbol, str) and symbol == "aact":
                return True
        if (
            node.get("kind") == "composition"
            and node.get("operator") == "agent_loop"
        ):
            return True
    return False


def _pipeline_has_human_approval(descriptor: dict[str, Any]) -> bool:
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    for _path, node in _iter_pipeline_nodes(pipeline):
        if (
            node.get("kind") == "composition"
            and node.get("operator") == "human_approval"
        ):
            return True
    return False


def _pipeline_terminal_calls(descriptor: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return the terminal call nodes of the pipeline.

    The "terminal" set are the last call(s) at any leaf of the pipeline tree:
    the last call node in the top-level sequence, and, recursively, the last
    call of each parallel branch / branch case. Used by R-SEM-MODALITY-STREAMING.
    """
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    terminals: list[tuple[str, dict[str, Any]]] = []

    def _last_call_of_body(body: list[Any], path: str) -> None:
        # Walk the body backward; the last call node — including last call in
        # nested composition's terminal calls — is appended.
        if not isinstance(body, list) or not body:
            return
        for idx in range(len(body) - 1, -1, -1):
            node = body[idx]
            if not isinstance(node, dict):
                continue
            sub_path = f"{path}/{idx}"
            kind = node.get("kind")
            if kind == "call":
                terminals.append((sub_path, node))
                return
            if kind == "composition":
                op = node.get("operator")
                if op in ("sequential", "map", "retry_with_feedback", "agent_loop"):
                    _last_call_of_body(node.get("body") or [], f"{sub_path}/body")
                    return
                if op == "parallel":
                    branches = node.get("branches") or []
                    for bi, br in enumerate(branches):
                        _last_call_of_body(br or [], f"{sub_path}/branches/{bi}")
                    return
                if op == "branch":
                    cases = node.get("cases") or {}
                    for case_name, body in cases.items():
                        _last_call_of_body(body or [], f"{sub_path}/cases/{case_name}")
                    return
                if op == "filter":
                    return
                if op == "human_approval":
                    branches = node.get("branches") or {}
                    for name in ("approved", "rejected"):
                        _last_call_of_body(
                            branches.get(name) or [], f"{sub_path}/branches/{name}"
                        )
                    return

    _last_call_of_body(pipeline, "/pipeline")
    return terminals


# --- R-SEM-MODALITY-* -------------------------------------------------------


def _check_modality_rules(*, descriptor: dict[str, Any]) -> list[ValidationError]:
    """Run the four R-SEM-MODALITY-* rules.

    Preconditions: none of these rules fires unless ``skill.classification`` is
    a dict (v0.3 shape). A v0.2-style string classification short-circuits.
    """
    errors: list[ValidationError] = []
    skill = descriptor.get("skill")
    if not isinstance(skill, dict):
        return errors
    classification = skill.get("classification")
    if not isinstance(classification, dict):
        return errors

    # R-SEM-MODALITY-TOOL-LOOP
    if classification.get("interaction_style") == "tool_using_loop":
        if not _pipeline_has_aact_or_agent_loop(descriptor):
            errors.append(
                ValidationError(
                    path="/skill/classification/interaction_style",
                    rule=R_MODALITY_TOOL_LOOP,
                    message=(
                        "interaction_style is 'tool_using_loop' but pipeline "
                        "contains no aact symbol nor agent_loop operator"
                    ),
                )
            )

    # R-SEM-MODALITY-STREAMING
    if classification.get("output_modality") == "streaming_text":
        terminals = _pipeline_terminal_calls(descriptor)
        if not terminals:
            errors.append(
                ValidationError(
                    path="/skill/classification/output_modality",
                    rule=R_MODALITY_STREAMING,
                    message=(
                        "output_modality is 'streaming_text' but pipeline has "
                        "no terminal call"
                    ),
                )
            )
        else:
            for path, node in terminals:
                if not bool(node.get("streaming")):
                    errors.append(
                        ValidationError(
                            path=f"{path}/streaming",
                            rule=R_MODALITY_STREAMING,
                            message=(
                                "output_modality is 'streaming_text' but "
                                "terminal call does not declare streaming: true"
                            ),
                        )
                    )

    # R-SEM-MODALITY-IMAGE-INPUT (and document)
    input_modality = classification.get("input_modality")
    if isinstance(input_modality, list):
        if "images" in input_modality:
            if not _has_input_slot_of_kind(descriptor, "image"):
                errors.append(
                    ValidationError(
                        path="/skill/classification/input_modality",
                        rule=R_MODALITY_IMAGE_INPUT,
                        message=(
                            "input_modality includes 'images' but no input "
                            "slot declares kind: 'image'"
                        ),
                    )
                )
        if "documents" in input_modality:
            if not _has_input_slot_of_kind(descriptor, "document"):
                errors.append(
                    ValidationError(
                        path="/skill/classification/input_modality",
                        rule=R_MODALITY_IMAGE_INPUT,
                        message=(
                            "input_modality includes 'documents' but no input "
                            "slot declares kind: 'document'"
                        ),
                    )
                )

    # R-SEM-MODALITY-APPROVAL
    if bool(classification.get("requires_approval_gates")):
        if not _pipeline_has_human_approval(descriptor):
            errors.append(
                ValidationError(
                    path="/skill/classification/requires_approval_gates",
                    rule=R_MODALITY_APPROVAL,
                    message=(
                        "requires_approval_gates is true but pipeline "
                        "contains no human_approval operator"
                    ),
                )
            )

    return errors


def _has_input_slot_of_kind(descriptor: dict[str, Any], kind: str) -> bool:
    inputs = descriptor.get("inputs") if isinstance(descriptor.get("inputs"), list) else []
    for slot in inputs:
        if not isinstance(slot, dict):
            continue
        schema = slot.get("schema")
        if not isinstance(schema, dict):
            continue
        if schema.get("kind") == kind:
            return True
        # Wrapped in list[...].
        if schema.get("kind") == "list":
            inner = schema.get("of")
            if isinstance(inner, dict) and inner.get("kind") == kind:
                return True
    return False


# --- R-SEM-DISPOSITION-CONSISTENT -------------------------------------------


def _check_dispositions(
    *, descriptor: dict[str, Any], surface: dict[str, Any] | None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    deps = descriptor.get("dependencies")
    if not isinstance(deps, list):
        return errors

    needs_signature = {
        "delegate_to_runtime",
        "real_impl",
        "stub",
        "mock",
        "external_input",
    }
    needs_path = {"bundle", "load_from_disk"}

    for idx, dep in enumerate(deps):
        if not isinstance(dep, dict):
            continue
        dep_path = f"/dependencies/{idx}"
        dep_id = dep.get("id", "<no-id>")
        kind = dep.get("kind")
        disposition = dep.get("disposition")
        if disposition in needs_path:
            if not isinstance(dep.get("path"), str) or not dep.get("path"):
                errors.append(
                    ValidationError(
                        path=f"{dep_path}/path",
                        rule=R_DISPOSITION_CONSISTENT,
                        message=(
                            f"dependency {dep_id!r} with disposition "
                            f"{disposition!r} is missing required field 'path'"
                        ),
                    )
                )
        if kind == "tool" and disposition in needs_signature:
            sig = dep.get("signature")
            if not isinstance(sig, str) or not sig.strip():
                errors.append(
                    ValidationError(
                        path=f"{dep_path}/signature",
                        rule=R_DISPOSITION_CONSISTENT,
                        message=(
                            f"dependency {dep_id!r} with kind='tool' and "
                            f"disposition {disposition!r} is missing "
                            f"required field 'signature'"
                        ),
                    )
                )
            elif not _is_parseable_signature(sig):
                errors.append(
                    ValidationError(
                        path=f"{dep_path}/signature",
                        rule=R_DISPOSITION_CONSISTENT,
                        message=(
                            f"dependency {dep_id!r} signature {sig!r} is not "
                            f"a parseable Python signature string "
                            f"(expected e.g. '(query: str) -> list[str]')"
                        ),
                    )
                )

        # real_impl with a symbol field: symbol must resolve in surface.
        if (
            kind == "tool"
            and disposition == "real_impl"
            and isinstance(dep.get("symbol"), str)
            and surface is not None
        ):
            if not _symbol_in_surface(dep["symbol"], surface):
                errors.append(
                    ValidationError(
                        path=f"{dep_path}/symbol",
                        rule=R_DISPOSITION_CONSISTENT,
                        message=(
                            f"dependency {dep_id!r} symbol "
                            f"{dep['symbol']!r} does not resolve in surface modules"
                        ),
                    )
                )

    return errors


_SIGNATURE_RE = re.compile(r"^\s*\(.*\)\s*(->\s*.+)?\s*$", re.DOTALL)


def _is_parseable_signature(s: str) -> bool:
    """Cheap structural check that a string looks like a Python signature.

    Accepts ``(...)`` or ``(...) -> Return``. Does not deeply parse types — the
    full parser belongs in the renderer; here we only catch obvious shape errors.
    """
    return bool(_SIGNATURE_RE.match(s))


# --- R-SEM-BUNDLED-PATH-EXISTS ---------------------------------------------


def _check_bundled_resources(
    *, descriptor: dict[str, Any], skill_root: str | Path | None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    resources = descriptor.get("bundled_resources")
    if not isinstance(resources, list):
        return errors

    root: Path | None = None
    if skill_root is not None:
        try:
            root = Path(skill_root)
        except (TypeError, ValueError):
            root = None

    for idx, entry in enumerate(resources):
        if not isinstance(entry, dict):
            continue
        path = entry.get("source_dir")
        entry_path = f"/bundled_resources/{idx}/source_dir"
        if not isinstance(path, str) or not path:
            continue
        # Reject absolute and traversal-laden paths up-front.
        if os.path.isabs(path):
            errors.append(
                ValidationError(
                    path=entry_path,
                    rule=R_BUNDLED_PATH_EXISTS,
                    message=(
                        f"bundled_resources.source_dir {path!r} is an "
                        f"absolute path; must be relative to skill root"
                    ),
                )
            )
            continue
        if ".." in Path(path).parts:
            errors.append(
                ValidationError(
                    path=entry_path,
                    rule=R_BUNDLED_PATH_EXISTS,
                    message=(
                        f"bundled_resources.source_dir {path!r} contains "
                        f"parent-directory traversal ('..'); reject"
                    ),
                )
            )
            continue
        if root is None:
            # Soft-fail: cannot verify without skill_root.
            errors.append(
                ValidationError(
                    path=entry_path,
                    rule=R_BUNDLED_PATH_EXISTS,
                    message=(
                        f"bundled_resources.source_dir {path!r} cannot be "
                        f"verified — skill_root not supplied to validator"
                    ),
                    severity="warning",
                )
            )
            continue
        target = root / path
        if not target.exists():
            errors.append(
                ValidationError(
                    path=entry_path,
                    rule=R_BUNDLED_PATH_EXISTS,
                    message=(
                        f"bundled_resources.source_dir {path!r} does not "
                        f"exist under skill root {str(root)!r}"
                    ),
                )
            )

    return errors


# --- R-SEM-AGENT-LOOP-FIELDS -----------------------------------------------


_AGENT_LOOP_REQUIRED = ("max_iterations", "state_schema", "termination_condition", "body")
_TERMINATION_PREDICATES = frozenset({"field_truthy", "field_equals", "field_in"})


def _check_agent_loop_fields(*, descriptor: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    dep_ids: set[str] = set()
    deps = descriptor.get("dependencies")
    if isinstance(deps, list):
        for dep in deps:
            if isinstance(dep, dict) and isinstance(dep.get("id"), str):
                if dep.get("kind") == "tool":
                    dep_ids.add(dep["id"])

    for path, node in _iter_pipeline_nodes(pipeline):
        if not (
            node.get("kind") == "composition"
            and node.get("operator") == "agent_loop"
        ):
            continue
        loop_id = node.get("id", "<no-id>")
        for fname in _AGENT_LOOP_REQUIRED:
            if fname not in node:
                errors.append(
                    ValidationError(
                        path=f"{path}/{fname}",
                        rule=R_AGENT_LOOP_FIELDS,
                        message=(
                            f"agent_loop {loop_id!r} is missing required "
                            f"field {fname!r}"
                        ),
                    )
                )
        termination = node.get("termination_condition")
        if isinstance(termination, dict):
            pred = termination.get("predicate")
            if pred not in _TERMINATION_PREDICATES:
                errors.append(
                    ValidationError(
                        path=f"{path}/termination_condition/predicate",
                        rule=R_AGENT_LOOP_FIELDS,
                        message=(
                            f"agent_loop {loop_id!r}.termination_condition."
                            f"predicate {pred!r} is not in "
                            f"{{field_truthy, field_equals, field_in}}"
                        ),
                    )
                )
        tools_ref = node.get("tools_ref")
        if isinstance(tools_ref, list):
            for ti, tref in enumerate(tools_ref):
                if not isinstance(tref, str):
                    continue
                if tref not in dep_ids:
                    errors.append(
                        ValidationError(
                            path=f"{path}/tools_ref/{ti}",
                            rule=R_AGENT_LOOP_FIELDS,
                            message=(
                                f"agent_loop {loop_id!r}.tools_ref entry "
                                f"{tref!r} does not resolve to a "
                                f"dependencies[] id with kind 'tool'"
                            ),
                        )
                    )
    return errors


# --- R-SEM-SOURCE-ELEMENT-RESOLVES -----------------------------------------


def _check_source_elements(
    *, descriptor: dict[str, Any], inventory: dict[str, Any] | None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []

    inventory_ids: set[str] | None = None
    if isinstance(inventory, dict):
        elements = inventory.get("elements")
        if isinstance(elements, list):
            inventory_ids = {
                e.get("id")
                for e in elements
                if isinstance(e, dict) and isinstance(e.get("id"), str)
            }

    for path, node in _iter_pipeline_nodes(pipeline):
        source_elements = node.get("source_elements")
        if not isinstance(source_elements, list):
            continue
        for si, sid in enumerate(source_elements):
            if not isinstance(sid, str):
                continue
            if inventory_ids is None:
                errors.append(
                    ValidationError(
                        path=f"{path}/source_elements/{si}",
                        rule=R_SOURCE_ELEMENT_RESOLVES,
                        message=(
                            f"source_element id {sid!r} cannot be verified — "
                            f"inventory not supplied to validator"
                        ),
                        severity="warning",
                    )
                )
                continue
            if sid not in inventory_ids:
                errors.append(
                    ValidationError(
                        path=f"{path}/source_elements/{si}",
                        rule=R_SOURCE_ELEMENT_RESOLVES,
                        message=(
                            f"source_element id {sid!r} does not resolve to "
                            f"any inventory.json element id"
                        ),
                    )
                )
    return errors


# --- R-SEM-FIELD-TYPE-KNOWN ------------------------------------------------


def _parse_field_type_token(s: str, known_schema_names: set[str]) -> bool:
    """Best-effort parser for field-type strings.

    Returns True iff ``s`` is in the closed vocabulary documented by the
    rule. Supports nested forms like ``list[str]``, ``dict[str, int]``,
    ``Optional[Foo]``, and bare model/enum references that name a declared
    schema in ``known_schema_names``.
    """
    s = s.strip()
    if not s:
        return False
    if s in _KNOWN_FIELD_TYPE_TOKENS:
        return True
    # Parameterised: <ctor>[<inside>]
    if "[" in s and s.endswith("]"):
        ctor, rest = s.split("[", 1)
        ctor = ctor.strip()
        inside = rest[:-1]
        if ctor not in _KNOWN_FIELD_TYPE_PARAMETERISED:
            return False
        # split top-level comma
        depth = 0
        parts: list[str] = []
        last = 0
        for i, ch in enumerate(inside):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(inside[last:i].strip())
                last = i + 1
        parts.append(inside[last:].strip())
        # Literal[...] members are literal values (strings/ints/bools); accept them.
        if ctor == "Literal":
            return bool(parts)
        return all(_parse_field_type_token(p, known_schema_names) for p in parts)
    # Bare identifier — must name a declared schema.
    if not s.isidentifier():
        return False
    return s in known_schema_names


def _check_field_type_known(*, descriptor: dict[str, Any]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    schemas = descriptor.get("schemas")
    if not isinstance(schemas, dict):
        return errors
    known_schema_names: set[str] = {
        sname for sname in schemas.keys() if isinstance(sname, str)
    }
    for sname, sdef in schemas.items():
        if not isinstance(sdef, dict):
            continue
        if sdef.get("kind") != "model":
            continue
        fields = sdef.get("fields")
        if not isinstance(fields, dict):
            continue
        for fname, fdef in fields.items():
            if not isinstance(fdef, dict):
                continue
            ftype = fdef.get("type")
            if not isinstance(ftype, str):
                # The JSON-Schema check already requires type to be string.
                continue
            if not _parse_field_type_token(ftype, known_schema_names):
                errors.append(
                    ValidationError(
                        path=f"/schemas/{sname}/fields/{fname}/type",
                        rule=R_FIELD_TYPE_KNOWN,
                        message=(
                            f"schema field {sname}.{fname}.type "
                            f"{ftype!r} is not in the known vocabulary "
                            f"{{str, int, float, bool, model_ref, enum_ref, "
                            f"list[T], dict[K,V], Optional[T], Literal[...], "
                            f"or a declared schema-name identifier}}"
                        ),
                    )
                )
    return errors


# --- R-SEM-CALL-ARGS-MATCH-SIG (warning) -----------------------------------


def _signature_parameters_from_surface(symbol: str, surface: dict[str, Any]) -> set[str] | None:
    """Return the set of parameter names for ``symbol``, or None if not found.

    Returns a special sentinel ``frozenset({"**kwargs"})`` when the resolved
    signature contains ``**kwargs`` — caller treats that as "any arg goes".
    Uses ``__wrapped__`` aware introspection by trusting the surface JSON's
    recorded ``signature`` (Phase 0.1 decorator-aware extraction).
    """
    member = _resolve_symbol_member(symbol, surface)
    if member is None:
        return None
    sig = member.get("signature") if isinstance(member, dict) else None
    if not isinstance(sig, str):
        return None
    # signature is like ``(self, foo: int, *, bar: str = 'x', **kwargs)``.
    # Strip outer parens (and any return annotation).
    open_paren = sig.find("(")
    if open_paren < 0:
        return None
    # Find matching close paren.
    depth = 0
    close_paren = -1
    for i in range(open_paren, len(sig)):
        if sig[i] == "(":
            depth += 1
        elif sig[i] == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
    if close_paren < 0:
        return None
    inner = sig[open_paren + 1 : close_paren]
    # Split top-level commas.
    depth = 0
    parts: list[str] = []
    last = 0
    for i, ch in enumerate(inner):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[last:i].strip())
            last = i + 1
    parts.append(inner[last:].strip())
    names: set[str] = set()
    for raw in parts:
        if not raw:
            continue
        if raw == "*" or raw == "/":
            continue
        if raw.startswith("**"):
            return frozenset({"**kwargs"})
        if raw.startswith("*"):
            # *args — skip; we're not matching positional-only args.
            continue
        # Strip leading * from positional/keyword.
        # parameter spec: ``name: type = default`` — pull the name.
        # Watch for ``self`` / ``cls`` — strip them; callers shouldn't pass.
        name_part = raw.split(":", 1)[0].split("=", 1)[0].strip()
        if name_part in ("self", "cls"):
            continue
        if name_part:
            names.add(name_part)
    return names


def _resolve_symbol_member(symbol: str, surface: dict[str, Any]) -> dict[str, Any] | None:
    """Walk the surface JSON and return the symbol's metadata dict, or None."""
    if not isinstance(surface, dict):
        return None
    modules = surface.get("modules")
    if not isinstance(modules, dict):
        return None
    parts = symbol.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split_at])
        rest = parts[split_at:]
        module = modules.get(module_path)
        if not isinstance(module, dict):
            continue
        symbols = module.get("symbols")
        if not isinstance(symbols, dict):
            continue
        head = rest[0]
        sym = symbols.get(head)
        if not isinstance(sym, dict):
            continue
        if len(rest) == 1:
            return sym
        current: Any = sym
        ok = True
        for token in rest[1:]:
            members = current.get("members") if isinstance(current, dict) else None
            if not isinstance(members, dict):
                ok = False
                break
            nxt = members.get(token)
            if not isinstance(nxt, dict):
                ok = False
                break
            current = nxt
        if ok and isinstance(current, dict):
            return current
    return None


def _check_call_args_match_sig(
    *, descriptor: dict[str, Any], surface: dict[str, Any] | None
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if surface is None:
        return errors
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    for path, node in _iter_pipeline_nodes(pipeline):
        if node.get("kind") != "call":
            continue
        symbol = node.get("symbol")
        if not isinstance(symbol, str):
            continue
        params = _signature_parameters_from_surface(symbol, surface)
        if params is None:
            # Symbol not in surface — R-SEM-SYMBOL covers that; don't double-flag.
            continue
        if "**kwargs" in params:
            continue
        args = node.get("args")
        if not isinstance(args, dict):
            continue
        for key in args.keys():
            if key not in params:
                errors.append(
                    ValidationError(
                        path=f"{path}/args/{key}",
                        rule=R_CALL_ARGS_MATCH_SIG,
                        message=(
                            f"call to {symbol!r} declares arg {key!r} not "
                            f"in its signature; valid parameters are "
                            f"{sorted(params)}"
                        ),
                        severity="warning",
                    )
                )
    return errors


# --- R-SEM-REF-SELECT-RESOLVES (warning) -----------------------------------


def _collect_node_schema_refs(
    descriptor: dict[str, Any],
) -> dict[str, str]:
    """Map node-id -> #/schemas/... ref declared via args.format.schema_ref."""
    out: dict[str, str] = {}
    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    for _path, node in _iter_pipeline_nodes(pipeline):
        if node.get("kind") != "call":
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        args = node.get("args")
        if not isinstance(args, dict):
            continue
        fmt = args.get("format")
        if isinstance(fmt, dict) and isinstance(fmt.get("schema_ref"), str):
            out[nid] = fmt["schema_ref"]
    return out


def _check_ref_select_resolves(
    *, descriptor: dict[str, Any]
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    schemas = descriptor.get("schemas") if isinstance(descriptor.get("schemas"), dict) else {}
    if not schemas:
        return errors
    node_schemas = _collect_node_schema_refs(descriptor)

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("ref"), str) and isinstance(value.get("select"), str):
                ref = value["ref"]
                select = value["select"]
                # Reject indexing syntax.
                if any(seg.isdigit() for seg in select.split(".")):
                    errors.append(
                        ValidationError(
                            path=f"{path}/select",
                            rule=R_REF_SELECT_RESOLVES,
                            message=(
                                f"select path {select!r} contains indexing "
                                f"syntax — unsupported; dot paths only in v0.3"
                            ),
                            severity="warning",
                        )
                    )
                    return
                schema_ref = node_schemas.get(ref)
                if not isinstance(schema_ref, str):
                    return
                schema_name = _schema_name_from_ref(schema_ref)
                if not schema_name or schema_name not in schemas:
                    return
                target = schemas.get(schema_name)
                if not isinstance(target, dict) or target.get("kind") != "model":
                    return
                fields = target.get("fields") or {}
                first = select.split(".", 1)[0]
                if first not in fields:
                    errors.append(
                        ValidationError(
                            path=f"{path}/select",
                            rule=R_REF_SELECT_RESOLVES,
                            message=(
                                f"select path {select!r} does not resolve to "
                                f"a declared field on schema {schema_name!r}"
                            ),
                            severity="warning",
                        )
                    )
            for k, v in value.items():
                if k in ("ref", "select", "schema_ref"):
                    continue
                _walk(v, f"{path}/{k}")
        elif isinstance(value, list):
            for i, sub in enumerate(value):
                _walk(sub, f"{path}/{i}")

    pipeline = descriptor.get("pipeline") if isinstance(descriptor.get("pipeline"), list) else []
    for path, node in _iter_pipeline_nodes(pipeline):
        if node.get("kind") == "call":
            args = node.get("args")
            if isinstance(args, dict):
                _walk(args, f"{path}/args")
    return errors
