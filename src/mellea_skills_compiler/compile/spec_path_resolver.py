"""Spec-path resolution shared by the legacy and descriptor compile branches.

Phase 3.5.A §2.1 widens the CLI's path resolver to accept the full range of
input shapes the legacy ``/mellea-fy`` slash-command workflow already handles:

* a single ``spec.md`` / ``SKILL.md`` at a skill root,
* a single ``.md`` file (any name) at any path,
* a workspace directory containing one or more ``.md`` files (with possible
  ``spec.md`` / ``SKILL.md`` precedence), plus companion files / dirs.

The legacy resolver in
:func:`mellea_skills_compiler.compile.mellea_skills._get_spec_md_path`
only recognised ``spec.md`` and ``SKILL.md`` inside a directory. This module
extends that contract without breaking it: when ``spec.md`` or ``SKILL.md``
is present, precedence remains ``SKILL.md`` → ``spec.md`` to match the
legacy behaviour byte-for-byte; otherwise we fall back to the first ``.md``
file found in the directory (sorted alphabetically for determinism).

The resolver is intentionally side-effect-free — it only returns Path
objects. Callers (CLI + slash-command argv builders) decide what to do
with the result (read text, derive a workspace directory, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "ResolvedSpec",
    "resolve_spec_path",
    "SpecPathResolutionError",
]


class SpecPathResolutionError(ValueError):
    """Raised when an input path cannot be resolved to a concrete spec.

    Subclass of ``ValueError`` so existing callers that catch ``ValueError``
    keep working. The message is operator-readable.
    """


@dataclass(frozen=True)
class ResolvedSpec:
    """Result of resolving a spec-path argument.

    Attributes:
        spec_file: The concrete ``.md`` file the compiler should treat as the
            primary spec.
        workspace_dir: The skill root / workspace directory containing the
            spec. For a single-file invocation this is ``spec_file.parent``.
        is_workspace: ``True`` when the input was a directory (multi-file
            workspace possible); ``False`` when the input was a direct ``.md``
            file path. Lets callers decide whether to mirror companion
            directories, scan for additional ``.md`` files, etc.
    """

    spec_file: Path
    workspace_dir: Path
    is_workspace: bool


def resolve_spec_path(spec_path: Path) -> ResolvedSpec:
    """Resolve a user-supplied ``spec_path`` argument.

    Accepts (in precedence order when ``spec_path`` is a directory):

    1. ``SKILL.md`` at the directory root,
    2. ``spec.md`` at the directory root,
    3. the first ``.md`` file found in the directory (sorted alphabetically).

    Raises:
        FileNotFoundError: if ``spec_path`` does not exist on disk.
        SpecPathResolutionError: if ``spec_path`` is a directory with no
            ``.md`` files inside, or a file with the wrong extension.

    The returned :class:`ResolvedSpec` always has a real on-disk file for
    ``spec_file``; callers may assume ``spec_file.read_text()`` is safe.
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"spec path does not exist: {spec_path}")

    if spec_path.is_dir():
        skill_md = spec_path / "SKILL.md"
        spec_md = spec_path / "spec.md"
        if skill_md.exists():
            return ResolvedSpec(
                spec_file=skill_md, workspace_dir=spec_path, is_workspace=True
            )
        if spec_md.exists():
            return ResolvedSpec(
                spec_file=spec_md, workspace_dir=spec_path, is_workspace=True
            )
        # Fall back: any .md file in the directory (sorted for determinism).
        md_files = sorted(
            p for p in spec_path.iterdir() if p.is_file() and p.suffix == ".md"
        )
        if md_files:
            return ResolvedSpec(
                spec_file=md_files[0], workspace_dir=spec_path, is_workspace=True
            )
        raise SpecPathResolutionError(
            "no spec.md / SKILL.md / *.md file found in skill directory: "
            f"{spec_path}"
        )

    # Direct file path.
    if spec_path.suffix != ".md":
        raise SpecPathResolutionError(
            "spec path must be a .md file or a directory containing one; got: "
            f"{spec_path}"
        )
    return ResolvedSpec(
        spec_file=spec_path, workspace_dir=spec_path.parent, is_workspace=False
    )
