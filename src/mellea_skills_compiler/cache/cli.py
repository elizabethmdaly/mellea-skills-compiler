"""Command-line entry point: ``python -m mellea_skills_compiler.cache``.

Subcommands
-----------

``populate``  — fetch llms.txt + every linked ``api/*.md`` page.
``status``    — print a freshness report (no network).
``clear``     — remove the cache slice for the current Mellea version
                (use ``--all`` to wipe every version).

Flags
-----

``--cache-dir PATH``  override the cache root.
``--ttl-days N``      override the staleness window (status only).
``--mellea-version V``override which version slice to operate on (default:
                      currently installed Mellea).

The CLI is intentionally minimal — argparse, not typer — because this module
is invoked from the same toolchain that builds the cache; pulling in typer
would add an import-graph dependency from cache → CLI tooling that the
plan keeps separate.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from mellea_skills_compiler.cache.fetcher import (
    FetchOutcome,
    refresh,
)
from mellea_skills_compiler.cache.freshness import (
    installed_mellea_version,
    is_fresh,
)
from mellea_skills_compiler.cache.layout import (
    CacheLayout,
    resolve_cache_root,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mellea_skills_compiler.cache",
        description=(
            "Manage the local Mellea docs cache (llms.txt + per-module pages)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the cache root (default: $MELLEA_CACHE_DIR or ./.mellea-cache).",
    )
    parser.add_argument(
        "--mellea-version",
        type=str,
        default=None,
        help=(
            "Operate on this Mellea version's cache slice "
            "(default: currently installed Mellea version)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("populate", help="Fetch llms.txt + all per-module .md pages.")

    status = sub.add_parser("status", help="Report cache freshness (no network).")
    status.add_argument(
        "--ttl-days",
        type=float,
        default=None,
        help="Staleness window in days (default: 30 or $MELLEA_CACHE_TTL_DAYS).",
    )

    clear = sub.add_parser(
        "clear",
        help="Remove the cache slice for the chosen Mellea version.",
    )
    clear.add_argument(
        "--all",
        action="store_true",
        help="Remove every version slice (the entire cache root).",
    )

    return parser


def _resolve_version(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    detected = installed_mellea_version()
    if not detected:
        print(
            "error: --mellea-version not given and mellea is not installed",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return detected


def _populate(layout: CacheLayout) -> int:
    print(f"populating cache at {layout.version_dir}")

    def progress(done: int, total: int, outcome: FetchOutcome) -> None:
        status = "ok" if outcome.ok else f"FAIL ({outcome.error or outcome.status})"
        print(f"  [{done:>3}/{total}] {outcome.url}  → {status}")

    result = refresh(layout, progress=progress)
    if not result.llms_outcome.ok:
        print(
            f"error: failed to fetch {result.llms_outcome.url}: "
            f"{result.llms_outcome.error or result.llms_outcome.status}",
            file=sys.stderr,
        )
        return 1
    print(
        f"done: {result.succeeded} ok, {result.failed} failed of "
        f"{len(result.module_outcomes)} module pages"
    )
    print(f"manifest: {result.manifest_path}")
    return 0


def _status(layout: CacheLayout, ttl_days: Optional[float]) -> int:
    report = is_fresh(layout, ttl_days=ttl_days)
    label = "fresh" if report.fresh else "stale"
    print(f"{label}: {layout.version_dir}")
    if report.cached_mellea_version is not None:
        print(f"  cached mellea_version: {report.cached_mellea_version}")
    if report.installed_mellea_version is not None:
        print(f"  installed mellea_version: {report.installed_mellea_version}")
    if report.fetched_at:
        print(f"  fetched_at: {report.fetched_at}")
    if report.age_days is not None:
        print(f"  age: {report.age_days:.1f}d (ttl {report.ttl_days:.1f}d)")
    for reason in report.reasons:
        print(f"  reason: {reason}")
    # `status` exits 0 if fresh, 1 if stale — useful in CI.
    return 0 if report.fresh else 1


def _clear(layout: CacheLayout, *, wipe_all: bool) -> int:
    target = layout.cache_root if wipe_all else layout.version_dir
    if not target.exists():
        print(f"nothing to clear at {target}")
        return 0
    shutil.rmtree(target)
    print(f"removed {target}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cache_root = resolve_cache_root(args.cache_dir)
    version = _resolve_version(args.mellea_version)
    layout = CacheLayout.for_version(cache_root, version)

    if args.command == "populate":
        return _populate(layout)
    if args.command == "status":
        return _status(layout, args.ttl_days)
    if args.command == "clear":
        return _clear(layout, wipe_all=args.all)
    parser.error(f"unknown command: {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
