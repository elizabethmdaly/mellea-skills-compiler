"""Renderer support for the descriptor v0.3 ``bundled_resources[]`` block.

Per RFC §3.2 and plan §11.5, each ``bundled_resources`` entry declares a
companion directory under the skill root that should be mirrored into the
rendered package and (optionally) included as wheel package-data.

Binary resources are **deferred to v0.4** (plan §11.5). The denylist below
covers common binary file types; matches surface a warning and a TODO
placeholder rather than being copied.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mellea_skills_compiler.renderer import RendererError


# --- Binary deferral policy ------------------------------------------------


_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".ico",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".webm",
    ".pyc", ".pyo",
    ".jar", ".class",
    ".sqlite", ".db",
})
"""Extensions considered binary; plan §11.5 defers their bundling to v0.4."""


_MAX_TEXT_SIZE_BYTES: int = 1_000_000
"""Files larger than this are also deferred (heuristic for embedded binaries)."""


def is_binary_resource(path: Path, *, max_size: int = _MAX_TEXT_SIZE_BYTES) -> bool:
    """Return True if a file should be treated as binary for bundling purposes.

    A file is binary if its extension is in :data:`_BINARY_EXTENSIONS` OR if
    the file exists and exceeds ``max_size``. Non-existent files are treated
    as text (the caller will hit a missing-file error downstream).
    """
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return True
    try:
        if path.is_file() and path.stat().st_size > max_size:
            return True
    except OSError:
        # Missing or permission-denied files are handled by the copy step.
        return False
    return False


# --- Artefact --------------------------------------------------------------


@dataclass
class BundledResourceCopy:
    """One scheduled directory-mirror operation."""

    source_dir: str
    """Path relative to the skill root."""

    package_dir: str
    """Path under the rendered package."""

    is_package_data: bool = True
    """Whether to add to pyproject.toml package-data."""

    include_glob: str | None = None
    """Optional glob restricting which files are copied."""

    deferred_binaries: list[str] = field(default_factory=list)
    """Files detected as binary and NOT copied (plan §11.5 deferral)."""


@dataclass
class BundledArtefact:
    """Aggregate output of :func:`render_bundled_resources`."""

    copies: list[BundledResourceCopy] = field(default_factory=list)
    """Scheduled directory mirror operations."""

    package_data_dirs: set[str] = field(default_factory=set)
    """Directories to declare in pyproject.toml ``[tool.setuptools.package-data]``."""

    pyproject_todo: list[str] = field(default_factory=list)
    """Per-entry TODO lines for deferred binary content; consumed by
    pyproject.toml emission as comments next to package-data entries."""


def _scan_for_binaries(
    skill_root: Path | None,
    source_dir: str,
    include_glob: str | None,
) -> list[str]:
    """Walk ``skill_root / source_dir`` and return any binary file paths.

    Returns an empty list if ``skill_root`` is unavailable (e.g. corpus
    compare from cache) — that's the validator's job, not the renderer's.
    """
    if skill_root is None:
        return []
    root = skill_root / source_dir
    if not root.is_dir():
        return []
    found: list[str] = []
    iterator = root.rglob(include_glob) if include_glob else root.rglob("*")
    for entry in iterator:
        if entry.is_file() and is_binary_resource(entry):
            try:
                found.append(str(entry.relative_to(skill_root)))
            except ValueError:
                found.append(str(entry))
    return sorted(found)


def render_bundled_resources(
    descriptor: dict[str, Any],
    skill_root: Path | None = None,
) -> BundledArtefact:
    """Translate ``descriptor.bundled_resources[]`` into a :class:`BundledArtefact`.

    Args:
        descriptor: The validated descriptor.
        skill_root: The original skill root directory (used to discover
            binary resources). May be ``None`` in cached/headless contexts;
            in that case binary detection is skipped.

    Emits a :mod:`warnings` warning for every detected binary resource so
    downstream tooling has a visible signal — actual copying is the caller's
    job (the renderer just schedules :class:`BundledResourceCopy` ops).
    """
    art = BundledArtefact()
    entries = descriptor.get("bundled_resources") or []
    if not isinstance(entries, list):
        raise RendererError(
            f"descriptor.bundled_resources must be a list, got {type(entries).__name__}"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise RendererError(
                f"each bundled_resources entry must be a dict, got {type(entry).__name__}"
            )
        source_dir = entry.get("source_dir")
        package_dir = entry.get("package_dir")
        if not source_dir or not package_dir:
            raise RendererError(
                "bundled_resources entry missing source_dir or package_dir: "
                f"{entry!r}"
            )
        is_pkg_data = entry.get("is_package_data", True)
        include_glob = entry.get("include_glob")

        deferred = _scan_for_binaries(skill_root, source_dir, include_glob)
        for binary_path in deferred:
            warnings.warn(
                f"bundled_resources: binary resource {binary_path!r} deferred to v0.4 "
                f"(plan §11.5); not copied into rendered package.",
                UserWarning,
                stacklevel=2,
            )
            art.pyproject_todo.append(
                f"# TODO: binary resource {binary_path!r} bundling deferred to v0.4"
            )

        art.copies.append(
            BundledResourceCopy(
                source_dir=source_dir,
                package_dir=package_dir,
                is_package_data=is_pkg_data,
                include_glob=include_glob,
                deferred_binaries=deferred,
            )
        )
        if is_pkg_data:
            art.package_data_dirs.add(package_dir)
    return art


def update_pyproject_package_data(
    pyproject_src: str,
    package_name: str,
    new_dirs: set[str],
) -> str:
    """Insert/extend a ``[tool.setuptools.package-data]`` block.

    Conservative line-based edit: keeps existing entries intact, appends new
    directories to the named package. Idempotent for the same input.
    """
    if not new_dirs:
        return pyproject_src
    # Build the patterns to add (one per dir, recursive glob).
    glob_patterns = sorted({f"{d}/**/*" for d in new_dirs})

    if "[tool.setuptools.package-data]" in pyproject_src:
        # Append to the existing entry for this package, or add the package.
        lines = pyproject_src.splitlines(keepends=True)
        out: list[str] = []
        in_block = False
        added = False
        for line in lines:
            stripped = line.strip()
            if stripped == "[tool.setuptools.package-data]":
                in_block = True
                out.append(line)
                continue
            if in_block and stripped.startswith("[") and stripped.endswith("]"):
                if not added:
                    out.append(f'{package_name} = {sorted(glob_patterns)!r}\n')
                    added = True
                in_block = False
            out.append(line)
        if in_block and not added:
            out.append(f'{package_name} = {sorted(glob_patterns)!r}\n')
            added = True
        return "".join(out)

    # No existing block — append one.
    block = (
        "\n[tool.setuptools.package-data]\n"
        f"{package_name} = {sorted(glob_patterns)!r}\n"
    )
    return pyproject_src + block


__all__ = [
    "BundledArtefact",
    "BundledResourceCopy",
    "is_binary_resource",
    "render_bundled_resources",
    "update_pyproject_package_data",
]
