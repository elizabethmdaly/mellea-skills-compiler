"""llms.txt parsing + cross-reference against the introspected surface.

The ``llms.txt`` document at https://docs.mellea.ai/llms.txt is the
documentation-published view of Mellea's public surface. We cross-reference
every ``api/*.md`` row in that document against the introspected
``Surface`` so that "this module is documented but not introspectable" is
caught explicitly rather than silently passed.

The format of each api/*.md row is::

    - [mellea.foo.bar](https://docs.mellea.ai/api/foo/bar.md): description

The link text (``mellea.foo.bar``) is authoritative for the dotted-module
name; the URL is preserved for traceability but only used as a fallback
when the link text isn't a dotted module path.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Any

from mellea_skills_compiler.mellea_surface.types import CoverageRow, Surface


LLMS_TXT_URL = "https://docs.mellea.ai/llms.txt"
COVERAGE_GATE_PERCENT = 95.0

_API_LINK_RE = re.compile(
    r"^- \[(?P<text>[^\]]+)\]\((?P<url>https://docs\.mellea\.ai/api/[^)]+\.md)\)"
)


def fetch_llms_txt(url: str = LLMS_TXT_URL, timeout: int = 30) -> str:
    """Fetch the llms.txt document over HTTPS. Stdlib only — no requests."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - docs URL
        return resp.read().decode("utf-8")


def parse_api_urls(text: str) -> list[tuple[str, str]]:
    """Extract ``(link_text, url)`` pairs for every ``api/*.md`` row."""
    out: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = _API_LINK_RE.match(line)
        if not m:
            continue
        out.append((m["text"].strip(), m["url"].strip()))
    return out


def url_to_module(url: str) -> str:
    """Fallback URL → dotted-module mapping.

    Used only when the link text isn't already a ``mellea.x.y`` path. Strips
    the docs prefix/suffix and collapses ``foo/foo`` to ``foo`` (the docs
    site uses ``api/foo/foo.md`` for package-level pages).
    """
    path = url.removeprefix("https://docs.mellea.ai/api/").removesuffix(".md")
    parts = path.split("/")
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        parts = parts[:-1]
    return ".".join(parts)


def link_text_to_module(link_text: str, url: str) -> str:
    """Pick the most reliable module name for a parsed row."""
    if link_text.startswith("mellea") and " " not in link_text:
        return link_text
    return url_to_module(url)


def assess_coverage(
    surface: Surface,
    api_urls: list[tuple[str, str]],
) -> list[CoverageRow]:
    """Cross-reference the parsed llms.txt rows against the surface.

    Returns one ``CoverageRow`` per input row; ``status`` is one of
    ``found`` / ``empty`` / ``missing`` / ``import_error``.
    """
    modules = surface.get("modules") or {}
    rows: list[CoverageRow] = []
    for link_text, url in api_urls:
        mod_name = link_text_to_module(link_text, url)
        if mod_name not in modules:
            rows.append(
                CoverageRow(
                    url=url,
                    expected_module=mod_name,
                    status="missing",
                    note="module not present in introspected surface",
                )
            )
            continue

        entry = modules[mod_name]
        if entry.get("import_error"):
            rows.append(
                CoverageRow(
                    url=url,
                    expected_module=mod_name,
                    status="import_error",
                    note=str(entry["import_error"]),
                )
            )
        elif not entry.get("symbols"):
            rows.append(
                CoverageRow(
                    url=url,
                    expected_module=mod_name,
                    status="empty",
                    note="module imported but exposes no public symbols",
                )
            )
        else:
            rows.append(
                CoverageRow(
                    url=url,
                    expected_module=mod_name,
                    status="found",
                    note=f"{len(entry['symbols'])} public symbols",
                )
            )
    return rows


def coverage_summary(rows: list[CoverageRow]) -> dict[str, Any]:
    """Aggregate counts for headline numbers + per-package breakdown."""
    found = [r for r in rows if r.status == "found"]
    missing = [r for r in rows if r.status == "missing"]
    empty = [r for r in rows if r.status == "empty"]
    import_err = [r for r in rows if r.status == "import_error"]
    total = len(rows)
    pct = (len(found) / total * 100) if total else 0.0

    per_pkg: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.expected_module.startswith("mellea."):
            pkg = ".".join(r.expected_module.split(".")[:2])
        else:
            pkg = r.expected_module
        bucket = per_pkg.setdefault(
            pkg, {"found": 0, "missing": 0, "empty": 0, "import_error": 0}
        )
        bucket[r.status] += 1

    return {
        "total": total,
        "found": len(found),
        "missing": len(missing),
        "empty": len(empty),
        "import_error": len(import_err),
        "percent": pct,
        "gate_pass": pct >= COVERAGE_GATE_PERCENT,
        "per_package": per_pkg,
    }


def render_coverage_report(rows: list[CoverageRow], surface: Surface) -> str:
    """Produce the markdown coverage report shown in
    ``melleafy-handoff/kickoff/spike-outputs/coverage_report.md``.
    """
    summary = coverage_summary(rows)
    missing = [r for r in rows if r.status == "missing"]
    empty = [r for r in rows if r.status == "empty"]
    import_err = [r for r in rows if r.status == "import_error"]

    lines: list[str] = []
    lines.append("# Mellea Introspection Coverage Report")
    lines.append("")
    lines.append(f"- Mellea version: `{surface.get('mellea_version', 'unknown')}`")
    lines.append(f"- Python version: `{surface.get('python_version', 'unknown')}`")
    lines.append(
        f"- llms.txt URLs measured: **{summary['total']}** (api/*.md only)"
    )
    lines.append(
        f"- Modules introspected: **{len(surface.get('modules') or {})}**"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"**Overall coverage: {summary['percent']:.1f}%** "
        f"({summary['found']}/{summary['total']} modules from llms.txt found "
        f"in introspected surface)."
    )
    lines.append("")
    lines.append(f"- Found (>=1 public symbol): **{summary['found']}**")
    lines.append(f"- Empty (imported, no public symbols): **{summary['empty']}**")
    lines.append(f"- Missing (no matching module): **{summary['missing']}**")
    lines.append(f"- Import errors: **{summary['import_error']}**")
    lines.append("")
    lines.append(
        f"**Gate: {'PASS' if summary['gate_pass'] else 'FAIL'}** "
        f"(threshold >={COVERAGE_GATE_PERCENT:.0f}%)."
    )
    lines.append("")
    lines.append("## Per-package breakdown")
    lines.append("")
    lines.append("| Package | Found | Empty | Missing | Import error |")
    lines.append("|---|---:|---:|---:|---:|")
    for pkg in sorted(summary["per_package"]):
        b = summary["per_package"][pkg]
        lines.append(
            f"| `{pkg}` | {b['found']} | {b['empty']} | {b['missing']} | "
            f"{b['import_error']} |"
        )
    lines.append("")

    if missing:
        lines.append("## Missing modules")
        lines.append("")
        lines.append(
            "Modules listed in llms.txt that don't resolve to anything in "
            "the installed package. Likely causes: doc-only modules, "
            "renamed/moved code, or the URL->module mapping rule needs "
            "refinement."
        )
        lines.append("")
        for r in missing:
            lines.append(f"- `{r.expected_module}` <- {r.url}")
        lines.append("")

    if empty:
        lines.append("## Empty modules")
        lines.append("")
        lines.append(
            "Imported successfully but exposed no public symbols. These may "
            "be re-export shells or genuinely empty `__init__.py` files."
        )
        lines.append("")
        for r in empty:
            lines.append(f"- `{r.expected_module}` <- {r.url}")
        lines.append("")

    if import_err:
        lines.append("## Import errors")
        lines.append("")
        for r in import_err:
            lines.append(f"- `{r.expected_module}` - {r.note}")
        lines.append("")

    return "\n".join(lines) + "\n"
