#!/usr/bin/env python3
"""Promptfoo exec provider — writer regression runner.

Reads JSON from stdin. The fixture YAML prompt template packages vars:

    prompts:
      - '{{ {"writer": writer, "emission": emission} | dump | safe }}'

Outputs the rendered writer result to stdout for promptfoo assertion.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _read_vars() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data.get("vars", data) if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _render_config(emission: str | dict) -> str:
    """Render config.py source from a config_emission JSON payload.

    Mirrors config_writer.render() inline to avoid import-path complexity.
    """
    if isinstance(emission, str):
        emission = json.loads(emission)
    lines = ["from typing import Final", ""]
    for constant in emission["constants"]:
        name: str = constant["name"]
        value: Any = constant["value"]
        py_type: str = constant["type"]
        lines.append(f"{name}: Final[{py_type}] = {value!r}")
    return "\n".join(lines) + "\n"


def main() -> None:
    vars_ = _read_vars()
    writer: str = vars_.get("writer", "")
    emission: str = vars_.get("emission", "")

    match writer:
        case "config_writer":
            try:
                print(_render_config(emission))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"ERROR config_writer: {exc}")
        case "":
            print("ERROR: no 'writer' key in input JSON")
        case _:
            print(f"WARN unknown writer: '{writer}'")


if __name__ == "__main__":
    main()
