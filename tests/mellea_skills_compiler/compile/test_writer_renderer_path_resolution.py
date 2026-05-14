"""Regression tests for `_resolve_writers_repo_root` in compile.mellea_skills.

Background — the bug this pins:
    The previous logic walked up from the *generated package directory* (e.g.
    `<wherever>/my_skill_mellea/`) looking for `.claude/melleafy/writers/`.
    For in-tree compiles this worked by accident because the generated
    package sat inside the compiler repo. For out-of-tree compiles (the
    standard eval-harness use case — spec lives in a separate skills repo)
    the walk fell off the top of the filesystem without finding the writers,
    `repo_root` defaulted to the package dir, `default_writer_specs` produced
    paths like `<package>/.claude/melleafy/writers/config_writer.py` that
    didn't exist, the writer subprocess failed with Errno 2, and the failure
    was swallowed as a WARNING. The compiled package shipped without config.py
    or fixtures/.

    The fix walks up from the *installed compiler package* instead. This test
    pins that behaviour and the explicit-FileNotFoundError on failure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.mellea_skills import _resolve_writers_repo_root


class TestResolveWritersRepoRoot:
    """Pin the writers-repo-root resolution semantics."""

    def test_finds_writers_at_start_dir(self):
        """If start dir itself contains .claude/melleafy/writers/, return it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".claude" / "melleafy" / "writers").mkdir(parents=True)
            assert _resolve_writers_repo_root(root) == root.resolve()

    def test_finds_writers_at_ancestor(self):
        """Walks up until .claude/melleafy/writers/ is found."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".claude" / "melleafy" / "writers").mkdir(parents=True)
            # Simulate the installed-package location: src/mellea_skills_compiler/compile/
            deep = root / "src" / "mellea_skills_compiler" / "compile"
            deep.mkdir(parents=True)
            assert _resolve_writers_repo_root(deep) == root.resolve()

    def test_raises_filenotfounderror_when_writers_absent(self, monkeypatch):
        """No `.claude/melleafy/writers/` anywhere on the ancestor chain → hard error.

        This is the case the previous walk-from-package logic silently swallowed,
        producing incomplete packages that shipped with a falsely-green Step-7
        verdict. The fix turns it into a loud FileNotFoundError.

        We monkeypatch `Path.is_dir` so the walk-up cannot escape the tmp
        subtree — otherwise the test would flake on systems where /tmp's
        ancestors happen to contain the project's real `.claude/`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp).resolve()
            start = tmp_root / "a" / "b" / "c"
            start.mkdir(parents=True)

            real_is_dir = Path.is_dir

            def constrained_is_dir(self):
                try:
                    resolved = self.resolve()
                except OSError:
                    return real_is_dir(self)
                try:
                    resolved.relative_to(tmp_root)
                except ValueError:
                    return False  # outside the sandbox → pretend absent
                return real_is_dir(self)

            monkeypatch.setattr(Path, "is_dir", constrained_is_dir)

            with pytest.raises(FileNotFoundError) as exc_info:
                _resolve_writers_repo_root(start)
            msg = str(exc_info.value)
            assert ".claude/melleafy/writers/" in msg, (
                f"Error should name the missing directory; got: {msg}"
            )

    def test_does_not_walk_up_from_generated_package_dir(self, monkeypatch):
        """Out-of-tree generated package — the OLD bug case.

        Construct: generated package lives in an independent tmp tree with
        no `.claude/` anywhere on its ancestor chain. The *correct* behaviour
        is to NOT use this directory for resolution — the caller is supposed
        to pass the installed compiler package dir. To pin this, we patch
        `Path.is_dir` to refuse to acknowledge a `.claude/melleafy/writers/`
        directory unless it sits under our synthetic 'compiler-install' root,
        then confirm: resolving from the generated-package dir raises, and
        resolving from the compiler-install dir succeeds.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp).resolve()
            # Simulated compiler install with the writers
            compiler_install = tmp_root / "compiler_repo"
            (compiler_install / ".claude" / "melleafy" / "writers").mkdir(parents=True)
            compiler_pkg_dir = compiler_install / "src" / "mellea_skills_compiler"
            compiler_pkg_dir.mkdir(parents=True)

            # Simulated out-of-tree skill spec / generated package
            skill_repo = tmp_root / "skill_repo"
            generated_pkg = skill_repo / "my_skill_mellea"
            generated_pkg.mkdir(parents=True)

            # Sanity: the compiler-install path resolves correctly.
            assert _resolve_writers_repo_root(compiler_pkg_dir) == compiler_install.resolve()

            # Patch the resolver so the walk-up from generated_pkg stops at
            # tmp_root (i.e. it can't escape into the real filesystem where
            # our actual .claude/melleafy/writers/ lives). We do this by
            # monkeypatching the walk to constrain parents to tmp_root.
            real_is_dir = Path.is_dir

            def constrained_is_dir(self):
                # Disallow matching anything outside tmp_root for this test.
                try:
                    resolved = self.resolve()
                except OSError:
                    return real_is_dir(self)
                try:
                    resolved.relative_to(tmp_root)
                except ValueError:
                    # Outside tmp_root → pretend it doesn't exist
                    return False
                return real_is_dir(self)

            monkeypatch.setattr(Path, "is_dir", constrained_is_dir)

            # Now: walking up from generated_pkg must NOT find the writers,
            # because skill_repo has no .claude/ and we've blocked escape to
            # the real filesystem.
            with pytest.raises(FileNotFoundError):
                _resolve_writers_repo_root(generated_pkg)
