#!/usr/bin/env python3
"""Local promptfoo-equivalent test driver for the KB/rule fixtures.

Why this exists: the promptfoo CLI is a Node.js tool. When it's not available
(no `npx` / no internet), we still need to exercise the lint/writer fixtures
to verify Known Behaviour regressions.

This script:
  1. Parses each tests/promptfoo/*.yaml fixture (subset of YAML it actually uses).
  2. Imports lint_runner and writer_runner directly via importlib.
  3. For each test entry, builds the JSON envelope, captures the runner's
     printed result, and evaluates the `assert:` rules (`contains`, `icontains`).
  4. Writes per-test logs into the supplied --out-dir, and prints a
     JSON-encoded summary on stdout (so a wrapper can ingest results).

YAML subset handled (all that the fixtures actually use):
  - mappings with literal-string keys
  - lists (block style with leading "- ")
  - scalars: plain, quoted (single/double), and block literal "|"
  - "type: contains" / "type: icontains" / "type: regex" assertions
  - "value: <string>" with quoted or plain RHS

If a fixture later uses richer YAML, switch to a real YAML parser by setting
the env var USE_REAL_YAML=1 (and have pyyaml installed in the active python).
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Minimal YAML parser (sufficient for tests/promptfoo/*.yaml)
# ---------------------------------------------------------------------------

class MiniYamlError(Exception):
    pass


def _strip_comment(line: str) -> str:
    # Strip trailing comments not inside quotes. Fixtures don't use # inside strings
    # outside block literals; we ignore comment stripping inside block-literal bodies.
    in_single = in_double = False
    out_chars: list[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out_chars.append(ch)
    return "".join(out_chars).rstrip()


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    # Block scalar markers handled by caller; here we only see flow scalars
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1]
        if s.startswith('"'):
            return inner.encode("utf-8").decode("unicode_escape")
        return inner
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    # int / float
    try:
        if "." not in s:
            return int(s)
        return float(s)
    except ValueError:
        return s


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_yaml(text: str) -> Any:
    """Tiny YAML loader for fixture files. Returns dict/list/scalar tree."""
    # Pre-process: split into lines but preserve block-literal bodies.
    raw_lines = text.splitlines()

    # Strip comment-only and trailing comments outside block literals.
    # We do a 2-pass parse: peek at every non-blank line, recurse based on indent.
    # Convert lines to (indent, content) pairs, but skip nothing yet —
    # block literals are reconstructed in-place.

    # First, an iterator with line-number tracking.
    i = 0
    n = len(raw_lines)

    def peek() -> tuple[int, str] | None:
        nonlocal i
        while i < n:
            stripped = raw_lines[i].rstrip()
            if not stripped.strip() or stripped.lstrip().startswith("#"):
                i += 1
                continue
            return i, raw_lines[i]
        return None

    def consume() -> tuple[int, str] | None:
        nonlocal i
        r = peek()
        if r is None:
            return None
        i = r[0] + 1
        return r

    def parse_block_literal(indent: int) -> str:
        """Consume lines whose indent > `indent` as a literal-block body."""
        lines: list[str] = []
        while i < n:
            ln = raw_lines[i]
            if not ln.strip():
                lines.append("")
                # don't advance past the blank without consuming
                # but include it as content
                # actually treat trailing blanks as part of block
                # — re-evaluate at end
                # advance pointer:
                # use direct increment
                # We need to advance i:
                pass
            cur_indent = _indent_of(ln)
            if ln.strip() and cur_indent <= indent:
                break
            # capture content stripping first (indent+1) spaces minimum
            # actually use cur_indent baseline = indent + 1 -> use min indent across captured lines
            lines.append(ln)
            # advance:
            # we mutate i directly
            globals_marker = None  # placeholder
            # explicit advance:
            # using nonlocal i — already defined
            # OK
            # increment
            # done.
            # (Python: assignment to nonlocal needs the keyword in enclosing fn.)
            # Already declared above.
            # Just do:
            # i += 1  — but we can't here without nonlocal
            # Refactor below.
            break
        # The flow above was overcomplicated. Reimplement cleanly:
        return ""

    # Reset and use a cleaner implementation:
    i = 0

    def line_iter():
        nonlocal i
        return i

    def parse_block_literal_clean(parent_indent: int) -> str:
        """Read consecutive lines indented more than parent_indent as a block scalar."""
        nonlocal i
        body_lines: list[str] = []
        min_indent: int | None = None
        while i < n:
            ln = raw_lines[i]
            if not ln.strip():
                body_lines.append("")
                i += 1
                continue
            cur_indent = _indent_of(ln)
            if cur_indent <= parent_indent:
                break
            if min_indent is None or cur_indent < min_indent:
                min_indent = cur_indent
            body_lines.append(ln)
            i += 1
        # trim leading indent
        if min_indent is None:
            min_indent = parent_indent + 2
        cleaned: list[str] = []
        for ln in body_lines:
            if not ln.strip():
                cleaned.append("")
            else:
                cleaned.append(ln[min_indent:] if len(ln) >= min_indent else ln.lstrip())
        # Drop trailing blank lines but keep a single trailing newline (literal style "|" keeps newline)
        while cleaned and cleaned[-1] == "":
            cleaned.pop()
        return "\n".join(cleaned) + "\n"

    def parse_value(parent_indent: int, inline: str) -> Any:
        """Parse a value that may be inline (`key: value` form) or block (next lines)."""
        inline = inline.strip()
        if inline == "":
            # block continuation — look at next non-blank line
            r = peek()
            if r is None:
                return None
            child_indent = _indent_of(r[1])
            if child_indent <= parent_indent:
                return None
            # decide list vs mapping
            content = r[1].strip()
            if content.startswith("- "):
                return parse_list(parent_indent)
            return parse_mapping(parent_indent)
        if inline == "|":
            return parse_block_literal_clean(parent_indent)
        # inline scalar
        return _parse_scalar(inline)

    def parse_list(parent_indent: int) -> list[Any]:
        nonlocal i
        items: list[Any] = []
        while True:
            r = peek()
            if r is None:
                break
            ln = r[1]
            cur_indent = _indent_of(ln)
            if cur_indent <= parent_indent:
                break
            content = ln[cur_indent:]
            if not content.startswith("- "):
                break
            # consume the line
            i += 1
            after_dash = content[2:]
            # after_dash may be either:
            #   <scalar>  ->  add scalar
            #   <key>: <value> -> start a mapping at indent = cur_indent + 2
            #   <empty>  -> nested block at deeper indent
            if not after_dash.strip():
                # nested mapping/list begins on next line
                val = parse_value(cur_indent, "")
                items.append(val)
                continue
            stripped_after = _strip_comment(after_dash).rstrip()
            if ":" in stripped_after and not (stripped_after.startswith('"') or stripped_after.startswith("'")):
                # this is a key:value pair starting an in-list mapping
                # the mapping's effective indent is cur_indent + 2
                # we need to feed the just-consumed line back as the first key
                # so build the first kv manually, then continue parsing subsequent keys
                key, _, rest = stripped_after.partition(":")
                key = key.strip()
                value = parse_value(cur_indent + 2, rest)
                d: dict[str, Any] = {key: value}
                # parse remaining keys at indent > cur_indent
                while True:
                    r2 = peek()
                    if r2 is None:
                        break
                    ln2 = r2[1]
                    ind2 = _indent_of(ln2)
                    if ind2 <= cur_indent:
                        break
                    # parse one key
                    line_content2 = _strip_comment(ln2[ind2:])
                    if line_content2.startswith("- "):
                        break
                    if ":" not in line_content2:
                        break
                    i += 1
                    k2, _, v2 = line_content2.partition(":")
                    d[k2.strip()] = parse_value(ind2, v2)
                items.append(d)
            else:
                items.append(_parse_scalar(after_dash))
        return items

    def parse_mapping(parent_indent: int) -> dict[str, Any]:
        nonlocal i
        d: dict[str, Any] = {}
        while True:
            r = peek()
            if r is None:
                break
            ln = r[1]
            cur_indent = _indent_of(ln)
            if cur_indent <= parent_indent:
                break
            content = _strip_comment(ln[cur_indent:])
            if content.startswith("- "):
                break
            if ":" not in content:
                break
            i += 1
            key, _, rest = content.partition(":")
            key = key.strip()
            d[key] = parse_value(cur_indent, rest)
        return d

    # Top level: assume mapping
    root: dict[str, Any] = {}
    while True:
        r = peek()
        if r is None:
            break
        ln = r[1]
        ind = _indent_of(ln)
        if ind != 0:
            break
        content = _strip_comment(ln)
        if content.startswith("- "):
            # top-level list
            return parse_list(-1)
        if ":" not in content:
            break
        i += 1
        key, _, rest = content.partition(":")
        key = key.strip()
        root[key] = parse_value(0, rest)
    return root


# ---------------------------------------------------------------------------
# Runner invocation
# ---------------------------------------------------------------------------

def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location(f"runner_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def invoke_runner(runner_path: Path, vars_payload: dict[str, Any]) -> str:
    """Run a runner's main() against a fixed stdin and capture stdout."""
    mod = _load_runner(runner_path)
    envelope = json.dumps({"vars": vars_payload})
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO(envelope)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mod.main()
    finally:
        sys.stdin = saved_stdin
    return buf.getvalue().rstrip("\n")


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------

def _eval_assert(rule: dict[str, Any], output: str) -> tuple[bool, str]:
    t = rule.get("type", "")
    v = rule.get("value", "")
    if t == "contains":
        return (v in output, f"contains '{v}'")
    if t == "icontains":
        return (v.lower() in output.lower(), f"icontains '{v}'")
    if t == "regex":
        return (re.search(v, output) is not None, f"regex '{v}'")
    return (False, f"unknown assert type '{t}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fixture(fixture_path: Path, runner_dir: Path, out_dir: Path) -> dict[str, Any]:
    text = fixture_path.read_text(encoding="utf-8")
    if os.environ.get("USE_REAL_YAML"):
        import yaml  # type: ignore
        doc = yaml.safe_load(text)
    else:
        doc = parse_yaml(text)

    description = doc.get("description", fixture_path.name)
    providers = doc.get("providers", [])
    runner_cmd = ""
    if isinstance(providers, list) and providers:
        cfg = providers[0].get("config", {}) if isinstance(providers[0], dict) else {}
        runner_cmd = cfg.get("command", "")

    # Map to local runner path.
    if "writer_runner" in runner_cmd:
        runner_path = runner_dir / "writer_runner.py"
    else:
        runner_path = runner_dir / "lint_runner.py"

    tests = doc.get("tests", [])
    test_results: list[dict[str, Any]] = []
    log_lines: list[str] = [f"# {fixture_path.name}", f"description: {description}", f"runner: {runner_path}", ""]

    for idx, test in enumerate(tests):
        desc = test.get("description", f"test #{idx}")
        vars_ = test.get("vars", {}) or {}
        try:
            output = invoke_runner(runner_path, vars_)
        except Exception as exc:
            output = f"RUNNER_ERROR: {type(exc).__name__}: {exc}"

        assertions = test.get("assert", []) or []
        ok_all = True
        assert_details: list[str] = []
        for rule in assertions:
            ok, label = _eval_assert(rule, output)
            assert_details.append(f"    {'OK' if ok else 'XX'}  {label}")
            if not ok:
                ok_all = False
        overall_outcome = "PASS" if ok_all else "FAIL"
        test_results.append({
            "description": desc,
            "vars_lint": vars_.get("lint", ""),
            "vars_subcheck": vars_.get("subcheck"),
            "output": output,
            "assertions_passed": ok_all,
            "outcome": overall_outcome,
        })
        log_lines.append(f"--- test {idx + 1}: {desc} ---")
        log_lines.append(f"vars: {json.dumps({k: v for k, v in vars_.items() if k != 'code' and k != 'mellea_api_ref'})}")
        log_lines.append(f"output:")
        for ln in output.splitlines() or [""]:
            log_lines.append(f"    {ln}")
        log_lines.append("assertions:")
        log_lines.extend(assert_details)
        log_lines.append(f"=> {overall_outcome}")
        log_lines.append("")

    n_pass = sum(1 for r in test_results if r["outcome"] == "PASS")
    n_fail = len(test_results) - n_pass
    log_lines.insert(3, f"summary: {n_pass}/{len(test_results)} passed, {n_fail} failed")
    log_lines.insert(4, "")

    out_log = out_dir / f"{fixture_path.stem}.log"
    out_log.write_text("\n".join(log_lines), encoding="utf-8")

    return {
        "fixture": fixture_path.name,
        "description": description,
        "n_tests": len(test_results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": test_results,
        "log_path": str(out_log),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures-dir", default="tests/promptfoo")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    runner_dir = fixtures_dir / "runners"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yamls = sorted(list(fixtures_dir.glob("kb_*.yaml")) + list(fixtures_dir.glob("rule_*.yaml")))
    summary: list[dict[str, Any]] = []
    for f in yamls:
        summary.append(run_fixture(f, runner_dir, out_dir))

    overall_log = out_dir / "_summary.json"
    overall_log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
