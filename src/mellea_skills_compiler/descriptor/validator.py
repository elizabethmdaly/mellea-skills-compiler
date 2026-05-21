"""JSON-Schema-level validation of mellea-fy descriptors.

This module is the orchestration entry point. It loads the schema for the
requested descriptor version, runs ``jsonschema.Draft202012Validator``
against the descriptor, and then delegates to :mod:`.semantic_rules` for the
checks the JSON Schema cannot express (symbol resolution, ref resolution,
composition-operator-specific rules, and the v0.2-only ``on_parse_failure``
default-shape check).

Public surface:

* :func:`validate` — orchestration entry point.
* :func:`load_schema` — load a frozen schema by version.
* :class:`ValidationError` — structured error with a JSON-Pointer ``path``.
* :class:`ValidationReport` — structured outcome of :func:`validate`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSValidationError

SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("0.1", "0.2", "0.3")

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class DeprecationWarning_:
    """A structured deprecation warning emitted by the validator.

    Distinct from ``ValidationError`` so callers can decide whether to surface
    deprecations to end users or quietly log them in telemetry.

    Attributes:
        path: JSON Pointer (RFC 6901) identifying the deprecated field.
        rule: stable identifier for the deprecation (e.g. ``deprecation:capabilities_tools``).
        message: human-readable explanation.
        replacement: optional pointer to the replacement field.
        descriptor_version: the descriptor_version under which this is deprecated.
    """

    path: str
    rule: str
    message: str
    replacement: str | None = None
    descriptor_version: str | None = None


@dataclass(frozen=True)
class ValidationError:
    """A single descriptor validation failure.

    ``path`` is a JSON Pointer (RFC 6901) so consumers can highlight the
    exact location in the descriptor that triggered the error. ``rule`` is a
    short stable identifier — either ``jsonschema:<failed_keyword>`` for
    JSON-Schema-level errors or one of the rule-name constants defined in
    :mod:`.semantic_rules` for Python-level checks.
    """

    path: str
    rule: str
    message: str
    severity: Severity = "error"


class DescriptorSchemaError(Exception):
    """Raised when the pre-render descriptor schema gate rejects a descriptor.

    Distinct from generic compile-time exceptions so the CLI exit-code router
    can map this cleanly to
    :data:`mellea_skills_compiler.exit_codes.ExitCode.DESCRIPTOR_SCHEMA_FAIL`
    (exit code 13). Carries the underlying :class:`ValidationReport` so
    callers (D1 auto-repair, telemetry) can extract structured errors
    without re-parsing the message.

    The ``report`` attribute is the full :class:`ValidationReport` from
    :func:`validate`. ``str(exc)`` is a human-readable summary suitable for
    stderr — naming the JSON path of the first failing error.
    """

    def __init__(self, message: str, *, report: "ValidationReport") -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ValidationReport:
    """Structured outcome of :func:`validate`.

    Attributes:
        valid: True iff there are no errors of severity ``"error"``.
        errors: list of :class:`ValidationError`.
        descriptor_version: the ``descriptor_version`` echoed from the input,
            or ``None`` if the input did not carry one.
        schema_version: the schema version the validator was asked to enforce.
        deprecations: list of structured deprecation warnings (non-blocking).
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    descriptor_version: str | None = None
    schema_version: str = "0.2"
    deprecations: list[DeprecationWarning_] = field(default_factory=list)


def load_schema(version: str = "0.2") -> dict[str, Any]:
    """Load a frozen JSON Schema by version.

    Raises:
        ValueError: if ``version`` is not in :data:`SUPPORTED_SCHEMA_VERSIONS`.
    """
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema version {version!r}; expected one of "
            f"{SUPPORTED_SCHEMA_VERSIONS}"
        )

    schema_path = files("mellea_skills_compiler.descriptor.schemas").joinpath(
        f"descriptor.schema.v{version}.json"
    )
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_schema_version(
    descriptor: Any, requested: str | None = None
) -> str:
    """Determine the schema version to validate against.

    Dispatch logic:
    - If ``requested`` is explicitly provided, use it (validates it's supported).
    - Else, if the descriptor carries a ``descriptor_version`` string, use it
      (when supported); otherwise raise ``ValueError``.
    - Else, fall back to the package default ``"0.2"``.

    Raises:
        ValueError: if the resolved version is not in
            :data:`SUPPORTED_SCHEMA_VERSIONS`.
    """
    if requested is not None:
        if requested not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema version {requested!r}; expected one of "
                f"{SUPPORTED_SCHEMA_VERSIONS}"
            )
        return requested
    if isinstance(descriptor, dict):
        dv = descriptor.get("descriptor_version")
        if isinstance(dv, str):
            if dv not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError(
                    f"descriptor declares unsupported version {dv!r}; "
                    f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
                )
            return dv
    return "0.2"


def _path_from_jsonschema(error: JSValidationError) -> str:
    """Convert a ``jsonschema`` ``absolute_path`` into a JSON Pointer."""
    parts = []
    for segment in error.absolute_path:
        token = str(segment)
        # JSON Pointer escapes: '~' -> '~0', '/' -> '~1'.
        token = token.replace("~", "~0").replace("/", "~1")
        parts.append(token)
    if not parts:
        return ""
    return "/" + "/".join(parts)


def _jsonschema_errors(
    descriptor: dict[str, Any], schema: dict[str, Any]
) -> Iterable[ValidationError]:
    validator = Draft202012Validator(schema)
    for raw in sorted(validator.iter_errors(descriptor), key=lambda e: list(e.absolute_path)):
        yield ValidationError(
            path=_path_from_jsonschema(raw),
            rule=f"jsonschema:{raw.validator}",
            message=raw.message,
            severity="error",
        )


def validate(
    descriptor: dict[str, Any],
    schema_version: str = "0.2",
    surface: dict[str, Any] | None = None,
    expected_signature: dict[str, Any] | None = None,
    skill_root: str | Path | None = None,
    inventory: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate ``descriptor`` against a schema version + semantic rules.

    Args:
        descriptor: the descriptor as a Python ``dict`` (parsed JSON).
        schema_version: which frozen schema to use. Defaults to ``"0.2"``.
        surface: the mellea introspected surface JSON, used for symbol
            resolution. If ``None``, symbol resolution is skipped.
        expected_signature: optional canonical I/O signature derived by Steps
            0-2.5 (P3.5.D). When provided, the
            :data:`~.semantic_rules.R_SIGNATURE_MATCH` rule fires; when
            ``None``, the rule is a no-op so unit tests on raw descriptors
            without intermediate artefacts still pass.
        skill_root: optional path to the skill's root directory. Required for
            :data:`~.semantic_rules.R_BUNDLED_PATH_EXISTS` to fire as an
            error; when omitted the rule degrades to warning severity.
        inventory: optional parsed ``inventory.json``. Required for
            :data:`~.semantic_rules.R_SOURCE_ELEMENT_RESOLVES` to fire as an
            error; when omitted the rule degrades to warning severity.

    Returns:
        A :class:`ValidationReport`.
    """
    # Import here so ``validator`` and ``semantic_rules`` can grow independently
    # without provoking circular imports.
    from . import semantic_rules

    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        return ValidationReport(
            valid=False,
            errors=[
                ValidationError(
                    path="",
                    rule="validator:unsupported_schema_version",
                    message=(
                        f"schema_version {schema_version!r} not supported; "
                        f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
                    ),
                )
            ],
            descriptor_version=(
                descriptor.get("descriptor_version") if isinstance(descriptor, dict) else None
            ),
            schema_version=schema_version,
        )

    schema = load_schema(schema_version)
    errors: list[ValidationError] = list(_jsonschema_errors(descriptor, schema))

    # Even if JSON-Schema-level errors exist, run semantic rules: they
    # catch a different class of problem (symbol typos, dangling refs) and
    # surfacing them together gives a more useful report. The semantic
    # rules are written defensively against partially-malformed inputs.
    errors.extend(
        semantic_rules.run_all(
            descriptor=descriptor,
            schema_version=schema_version,
            surface=surface,
            expected_signature=expected_signature,
            skill_root=skill_root,
            inventory=inventory,
        )
    )

    deprecations = _collect_deprecations(descriptor, schema_version)

    return ValidationReport(
        valid=not any(e.severity == "error" for e in errors),
        errors=errors,
        descriptor_version=(
            descriptor.get("descriptor_version") if isinstance(descriptor, dict) else None
        ),
        schema_version=schema_version,
        deprecations=deprecations,
    )


def _collect_deprecations(
    descriptor: Any, schema_version: str
) -> list[DeprecationWarning_]:
    """Detect structured deprecation warnings for the resolved version.

    v0.3 deprecation rule (RFC §3.1 + Q6): ``capabilities.tools`` is retained
    for backward shape but v0.3 emissions should use the top-level
    ``dependencies`` block. When BOTH are present the validator emits a
    deprecation warning (non-blocking). The warning is also emitted when
    ``capabilities.tools`` is present but ``dependencies`` is not — to signal
    that the descriptor should be migrated to the canonical surface.
    """
    if schema_version != "0.3":
        return []
    if not isinstance(descriptor, dict):
        return []
    capabilities = descriptor.get("capabilities")
    if not isinstance(capabilities, dict):
        return []
    tools = capabilities.get("tools")
    if tools is None:
        return []
    has_dependencies = isinstance(descriptor.get("dependencies"), list) and bool(
        descriptor.get("dependencies")
    )
    if has_dependencies:
        message = (
            "capabilities.tools is deprecated in v0.3 and coexists with the "
            "canonical top-level dependencies[] block; prefer dependencies "
            "and remove capabilities.tools in a future emission."
        )
    else:
        message = (
            "capabilities.tools is deprecated in v0.3; replace with the "
            "top-level dependencies[] block carrying kind/disposition "
            "decisions per dependency_plan.json."
        )
    return [
        DeprecationWarning_(
            path="/capabilities/tools",
            rule="deprecation:capabilities_tools",
            message=message,
            replacement="/dependencies",
            descriptor_version="0.3",
        )
    ]


def load_schema_path(version: str = "0.2") -> Path:
    """Return the on-disk path to a frozen schema. Useful for debugging."""
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema version {version!r}")
    return Path(
        str(
            files("mellea_skills_compiler.descriptor.schemas").joinpath(
                f"descriptor.schema.v{version}.json"
            )
        )
    )
