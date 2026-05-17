"""argparse CLI entry point for the introspection package.

Two subcommands:

    build     - walk installed Mellea, emit surface JSON.
    coverage  - fetch llms.txt (or read a local copy), assess coverage
                against a built (or freshly-built) surface, emit a
                markdown coverage report and exit non-zero if the gate
                is missed.

Run via::

    python -m mellea_skills_compiler.mellea_surface build --out surface.json
    python -m mellea_skills_compiler.mellea_surface coverage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from mellea_skills_compiler.mellea_surface.coverage import (
    LLMS_TXT_URL,
    assess_coverage,
    coverage_summary,
    fetch_llms_txt,
    parse_api_urls,
    render_coverage_report,
)
from mellea_skills_compiler.mellea_surface.introspector import (
    build_surface,
    load_surface,
    write_surface,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mellea_skills_compiler.mellea_surface",
        description=(
            "Walk the installed Mellea package and emit a machine-readable "
            "surface description (Phase 1.A of the descriptor compiler)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser(
        "build",
        help="Walk Mellea and emit surface JSON.",
    )
    build_p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the surface JSON.",
    )
    build_p.add_argument(
        "--root",
        default="mellea",
        help="Root package to introspect (default: mellea).",
    )

    cov_p = sub.add_parser(
        "coverage",
        help="Assess llms.txt coverage against the introspected surface.",
    )
    cov_p.add_argument(
        "--surface",
        type=Path,
        default=None,
        help=(
            "Path to a previously-built surface JSON. If omitted, build the "
            "surface in-process from the installed Mellea."
        ),
    )
    cov_p.add_argument(
        "--llms-txt",
        type=Path,
        default=None,
        help=(
            "Path to a local llms.txt snapshot. If omitted, fetch from "
            f"{LLMS_TXT_URL}."
        ),
    )
    cov_p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output path for the markdown coverage report.",
    )
    cov_p.add_argument(
        "--root",
        default="mellea",
        help="Root package to introspect when --surface is not given.",
    )

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    surface = build_surface(root_package=args.root)
    out = write_surface(surface, args.out)
    n_modules = len(surface.get("modules") or {})
    print(
        f"Wrote surface for mellea {surface.get('mellea_version')} "
        f"({n_modules} modules) -> {out}",
        file=sys.stderr,
    )
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    if args.surface is not None:
        surface = load_surface(args.surface)
    else:
        surface = build_surface(root_package=args.root)

    if args.llms_txt is not None:
        text = Path(args.llms_txt).read_text(encoding="utf-8")
    else:
        text = fetch_llms_txt()

    api_urls = parse_api_urls(text)
    rows = assess_coverage(surface, api_urls)
    summary = coverage_summary(rows)
    report = render_coverage_report(rows, surface)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Wrote coverage report -> {args.report}", file=sys.stderr)
    else:
        sys.stdout.write(report)

    print(
        f"Coverage: {summary['percent']:.1f}% "
        f"({summary['found']}/{summary['total']} modules) "
        f"gate={'PASS' if summary['gate_pass'] else 'FAIL'}",
        file=sys.stderr,
    )
    return 0 if summary["gate_pass"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "coverage":
        return _cmd_coverage(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
