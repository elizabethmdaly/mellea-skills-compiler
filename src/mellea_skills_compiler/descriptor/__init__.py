"""Descriptor schema validator (Phase 1.B / Phase 2.D).

Provides JSON-Schema validation plus a semantic-rules layer for the
mellea-fy descriptor language. See
``melleafy-handoff/analyses/2026-05-16-descriptor-schema-v0.2-rfc.md`` for
the source-of-truth RFC.

Public surface::

    from mellea_skills_compiler.descriptor import (
        validate,
        ValidationError,
        ValidationReport,
        load_schema,
    )

    report = validate(descriptor_dict, schema_version="0.2", surface=surface_dict)
    if not report.valid:
        for err in report.errors:
            print(err.path, err.rule, err.message)
"""
from __future__ import annotations

from .validator import (
    SUPPORTED_SCHEMA_VERSIONS,
    DeprecationWarning_,
    ValidationError,
    ValidationReport,
    load_schema,
    resolve_schema_version,
    validate,
)

__all__ = [
    "validate",
    "load_schema",
    "resolve_schema_version",
    "ValidationError",
    "ValidationReport",
    "DeprecationWarning_",
    "SUPPORTED_SCHEMA_VERSIONS",
]
