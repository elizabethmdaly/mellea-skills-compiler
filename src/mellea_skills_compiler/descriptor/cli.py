"""CLI entry point for descriptor validation.

Usage::

    python -m mellea_skills_compiler.descriptor validate <descriptor.json> \\
        [--schema-version 0.1|0.2] [--surface surface.json]

Exit code is 0 on success, 1 on any validation error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .validator import SUPPORTED_SCHEMA_VERSIONS, validate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mellea_skills_compiler.descriptor",
        description="Validate a mellea-fy descriptor against schema + semantic rules.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_p = subparsers.add_parser(
        "validate", help="validate a single descriptor.json"
    )
    validate_p.add_argument(
        "descriptor",
        type=Path,
        help="path to the descriptor.json to validate",
    )
    validate_p.add_argument(
        "--schema-version",
        default="0.2",
        choices=list(SUPPORTED_SCHEMA_VERSIONS),
        help="which frozen schema version to enforce (default: 0.2)",
    )
    validate_p.add_argument(
        "--surface",
        type=Path,
        default=None,
        help=(
            "path to the introspected mellea surface JSON. If omitted, "
            "symbol resolution is skipped."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "validate":
        parser.print_help()
        return 1

    try:
        descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: descriptor file not found: {args.descriptor}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: descriptor is not valid JSON: {exc}", file=sys.stderr)
        return 1

    surface = None
    if args.surface is not None:
        try:
            surface = json.loads(args.surface.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"error: surface file not found: {args.surface}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as exc:
            print(f"error: surface is not valid JSON: {exc}", file=sys.stderr)
            return 1

    report = validate(
        descriptor=descriptor,
        schema_version=args.schema_version,
        surface=surface,
    )

    if report.valid and not report.errors:
        print(
            f"OK: descriptor valid against schema v{report.schema_version} "
            f"(descriptor_version={report.descriptor_version!r})"
        )
        return 0

    for err in report.errors:
        print(
            f"[{err.severity}] {err.path}  {err.rule}: {err.message}",
            file=sys.stderr,
        )
    print(
        f"\n{len(report.errors)} issue(s); valid={report.valid}",
        file=sys.stderr,
    )
    return 0 if report.valid else 1


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    raise SystemExit(main())
