"""Mellea introspection surface — Phase 1.A of the introspection-driven descriptor compiler.

This package walks the installed Mellea package and emits a machine-readable
description (the "surface") of every public symbol. Downstream stages
(descriptor validator, renderer, prompt assembly) consume the surface to ensure
that emitted descriptors only reference real Mellea calls.

Public API:
    build_surface()        — walk mellea and return a Surface dict.
    write_surface(...)     — serialise a Surface dict to JSON (stable, sorted).
    load_surface(path)     — load a previously-emitted surface JSON.
    fetch_llms_txt()       — fetch the canonical llms.txt over HTTP.
    parse_api_urls(text)   — parse `api/*.md` rows out of an llms.txt body.
    assess_coverage(...)   — cross-reference llms.txt against a Surface.
    render_coverage_report — produce the markdown coverage report.

See `melleafy-handoff/analyses/2026-05-16-introspection-driven-descriptor-compiler-plan.md`
section 3 for the architectural context.
"""

from __future__ import annotations

from mellea_skills_compiler.mellea_surface.coverage import (
    CoverageRow,
    assess_coverage,
    fetch_llms_txt,
    parse_api_urls,
    render_coverage_report,
)
from mellea_skills_compiler.mellea_surface.introspector import (
    build_surface,
    load_surface,
    write_surface,
)
from mellea_skills_compiler.mellea_surface.types import (
    Surface,
    SurfaceMember,
    SurfaceModule,
    SurfaceSymbol,
)

__all__ = [
    "CoverageRow",
    "Surface",
    "SurfaceMember",
    "SurfaceModule",
    "SurfaceSymbol",
    "assess_coverage",
    "build_surface",
    "fetch_llms_txt",
    "load_surface",
    "parse_api_urls",
    "render_coverage_report",
    "write_surface",
]
