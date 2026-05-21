"""JSON Schema files for artifacts the Step 7 lint pipeline produces.

Bundled with the package so ``importlib.resources.files(...)`` can
resolve them after ``pip install``. The build is configured to include
``*.json`` under this directory; see ``pyproject.toml`` ``[tool.pdm.build]``.
"""
