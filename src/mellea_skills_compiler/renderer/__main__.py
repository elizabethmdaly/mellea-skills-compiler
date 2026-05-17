"""Run the renderer CLI: ``python -m mellea_skills_compiler.renderer render ...``."""

from __future__ import annotations

import sys

from mellea_skills_compiler.renderer.cli import main

if __name__ == "__main__":
    sys.exit(main())
