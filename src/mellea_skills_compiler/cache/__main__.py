"""Allow ``python -m mellea_skills_compiler.cache`` to invoke the CLI."""

from __future__ import annotations

from mellea_skills_compiler.cache.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
