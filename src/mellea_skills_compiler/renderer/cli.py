"""Renderer CLI.

Two subcommands:

    python -m mellea_skills_compiler.renderer render <descriptor> <out_dir> [--surface ...]
        Renders only ``pipeline.py`` + ``source_map.json``. Useful for isolating the
        core renderer in debugging and CI smoke checks.

    python -m mellea_skills_compiler.renderer render-package <descriptor> <out_dir> [--surface ...] [--mellea-version ...]
        Renders the full compiled-skill package: ``pipeline.py``, ``schemas.py``,
        ``__init__.py``, ``fixtures.py``, ``melleafy.json``, ``SETUP.md``, ``README.md``,
        and ``source_map.json``. This is the package emitter Phase 3 invokes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from importlib import metadata as _md
from importlib.resources import files as _resource_files
from pathlib import Path

from mellea_skills_compiler.renderer import RendererError, render_descriptor
from mellea_skills_compiler.renderer.artefacts import (
    render_fixtures,
    render_init,
    render_melleafy_json,
    render_readme_md,
    render_setup_md,
)
from mellea_skills_compiler.renderer.schemas import render_schemas


# Bundled surface JSON (Phase 0.2 spike output) — ships with the package via
# ``src/mellea_skills_compiler/mellea_surface/data/`` and is located through
# :mod:`importlib.resources` so it resolves whether running from a checkout or
# an installed wheel.
_DEFAULT_SURFACE_PATH = Path(
    str(_resource_files("mellea_skills_compiler.mellea_surface") / "data" / "surface_0.5.0.json")
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mellea_skills_compiler.renderer",
        description="Render a descriptor JSON into pipeline.py + a source map.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render_p = sub.add_parser("render", help="Render a descriptor to pipeline.py")
    render_p.add_argument("descriptor", type=Path, help="Path to <name>.descriptor.json")
    render_p.add_argument("out_dir", type=Path, help="Output directory for pipeline.py / source_map.json")
    render_p.add_argument(
        "--surface",
        type=Path,
        default=_DEFAULT_SURFACE_PATH,
        help="Path to the introspected surface.json (default: Phase 0.2 spike output)",
    )

    pkg_p = sub.add_parser("render-package", help="Render the full compiled-skill package")
    pkg_p.add_argument("descriptor", type=Path)
    pkg_p.add_argument("out_dir", type=Path, help="Output directory (will be created)")
    pkg_p.add_argument(
        "--surface",
        type=Path,
        default=_DEFAULT_SURFACE_PATH,
    )
    pkg_p.add_argument(
        "--mellea-version",
        default=None,
        help="Mellea version stamped into melleafy.json (auto-detected if omitted)",
    )
    pkg_p.add_argument(
        "--compiled-at",
        default=None,
        help="ISO timestamp to stamp into melleafy.json (UTC now if omitted)",
    )

    args = parser.parse_args(argv)

    descriptor = json.loads(Path(args.descriptor).read_text(encoding="utf-8"))
    surface = json.loads(Path(args.surface).read_text(encoding="utf-8"))

    try:
        result = render_descriptor(descriptor, surface)
    except RendererError as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline.py").write_text(result.pipeline_py, encoding="utf-8")
    (out_dir / "source_map.json").write_text(
        json.dumps(result.source_map.to_json(), indent=2) + "\n",
        encoding="utf-8",
    )

    if args.command == "render":
        print(f"wrote {out_dir / 'pipeline.py'} and {out_dir / 'source_map.json'}")
        return 0

    # render-package: also emit schemas.py + __init__.py + fixtures.py + melleafy.json + SETUP.md + README.md
    fixtures_dir = out_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    (out_dir / "schemas.py").write_text(render_schemas(descriptor.get("schemas", {})), encoding="utf-8")
    (out_dir / "__init__.py").write_text(render_init(descriptor), encoding="utf-8")
    (out_dir / "fixtures.py").write_text(render_fixtures(descriptor, fixtures_dir), encoding="utf-8")
    (fixtures_dir / "__init__.py").write_text("", encoding="utf-8")

    try:
        mellea_version = args.mellea_version or _md.version("mellea")
    except Exception:
        mellea_version = args.mellea_version or "unknown"
    compiled_at = args.compiled_at or _dt.datetime.now(_dt.timezone.utc).isoformat()
    renderer_version = "p2-0.1"

    (out_dir / "melleafy.json").write_text(
        render_melleafy_json(
            descriptor,
            mellea_version=mellea_version,
            renderer_version=renderer_version,
            compiled_at=compiled_at,
        ),
        encoding="utf-8",
    )
    (out_dir / "SETUP.md").write_text(render_setup_md(descriptor), encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme_md(descriptor), encoding="utf-8")
    print(f"wrote full package to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
