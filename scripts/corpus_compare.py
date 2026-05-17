"""Corpus comparison harness for Phase 3 / Phase 5.

Per `melleafy-handoff/process/regression-and-extension-policy.md` §(a):
run BOTH the legacy free-form Python compilation pipeline AND the new
descriptor IR compilation pipeline against every skill in `skills/`,
measure success rates + fixture pass rates + wall clock, write a
structured report, and produce a routing-violation count for CI.

Run:
    .venv-spike/bin/python scripts/corpus_compare.py [--skills <root>] [--out <path>]

Routing-violation definition (the gate Phase 5 checks):
    A skill whose `.melleafy-routing.json` says `"compiler": "descriptor"`
    but whose descriptor pipeline fails to compile.  Routing violations
    must be zero before Phase 5 flips the default to descriptor.

Pipeline invocation strategy
----------------------------

Legacy (default: ``live``, Phase 3.5.F):
    Both pipelines share the ``_spawn_claude`` helper in
    ``mellea_skills_compiler.compile.mellea_skills`` (P3.5.A); the
    ``live`` mode drives the real ``claude`` subprocess by calling
    ``mellea_skills.compile(..., use_descriptor=False)``.  Pass
    ``--require-live`` (the Phase-5 gate) to fail loudly if ``claude``
    cannot be spawned, rather than silently falling back to the baseline.

    ``baseline`` mode (CI fallback): when running on a headless worker
    without Claude Code installed, look for an existing
    ``examples/<skill>/<skill>_mellea/`` directory and copy it as the
    legacy reference.  Used only when ``--mode=baseline`` is passed.

    When neither a live spawn nor a baseline works the result is reported
    as ``compile_success=False`` with a descriptive ``error``.

Descriptor:
    ``mellea_skills_compiler.compile.descriptor_emission.compile_via_descriptor``
    is the in-process one-shot entry point (preserved post-P3.5.A for
    unit tests + this script).  This module imports it lazily; if the
    module isn't available the descriptor result is a graceful
    "descriptor pipeline not yet shipped" failure so the harness still
    runs end-to-end.

Structural diff is informational, not a gate — it walks both
``pipeline.py`` ASTs and renders a 3-5 line summary of differences.

Previous-vs-current drift mode (Phase 3.5 §14)
----------------------------------------------

Orthogonal to the legacy-vs-descriptor diff, pass
``--compare-against-previous --previous-dir <path-or-git-rev>`` to
recompile each skill with the *current* descriptor pipeline and compare
the output against a previous snapshot.  Each skill receives a
three-class drift verdict (cosmetic / structural / semantic) per
§14.3; the aggregated ``recompile_drift_report.md`` lands next to the
JSON output.

``--previous-dir`` accepts:

* a filesystem directory (e.g.  ``./snapshots/2026-04-01/skills``) whose
  per-skill children mirror the corpus layout, OR
* a git revision (``HEAD~1``, ``v0.5.0``, full SHA…) — files are read
  via ``git show <rev>:<path>``.

The hard-stop rule from §14.4 fires automatically when >5% of the corpus
falls into the semantic-drift class: the script exits non-zero and the
report header highlights the violation.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
DEFAULT_EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_SURFACE = REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "surface_0.5.0.json"
DEFAULT_OUTPUT = REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "corpus_comparison.json"
DEFAULT_DRIFT_REPORT = (
    REPO_ROOT / "melleafy-handoff" / "kickoff" / "spike-outputs" / "recompile_drift_report.md"
)

CompilerName = Literal["legacy", "descriptor"]
LegacyMode = Literal["baseline", "live"]
# Phase 3.5.F: alias for clarity at the CLI surface — the script-level
# --mode flag covers both legacy modes (CI baseline vs live ``claude``
# subprocess via shared ``_spawn_claude``).  ``LegacyMode`` and ``Mode``
# are intentionally equal until/unless we add further modes.
Mode = LegacyMode


class LiveSpawnUnavailable(RuntimeError):
    """Raised when ``--require-live`` is in effect but the ``claude`` CLI
    cannot be invoked (binary missing, mellea_skills module unavailable,
    etc.).  The Phase 5 gate run treats this as a hard failure."""


@dataclass
class PipelineResult:
    compiler: CompilerName
    compile_success: bool
    fixture_pass_count: int
    fixture_total_count: int
    wall_clock_seconds: float
    error: str | None = None
    package_dir: str | None = None  # absolute path to the compiled package, if any


@dataclass
class SkillRouting:
    """Read from <skill-root>/.melleafy-routing.json. Per the policy doc."""

    compiler: CompilerName
    reason: str
    decided_at: str
    tracking_rfc: str | None = None
    review_at: str | None = None


@dataclass
class SkillResult:
    name: str
    routing: SkillRouting | None
    legacy: PipelineResult | None
    descriptor: PipelineResult | None
    structural_diff_summary: str = ""

    @property
    def routing_violation(self) -> bool:
        """A 'routing violation' is: routing says descriptor, but descriptor fails to compile."""
        if not self.routing or self.routing.compiler != "descriptor":
            return False
        return not (self.descriptor and self.descriptor.compile_success)


@dataclass
class CorpusReport:
    generated_at: str
    skills_dir: str
    total_skills: int
    skills: list[SkillResult] = field(default_factory=list)

    @property
    def legacy_compile_rate(self) -> float:
        ok = sum(1 for s in self.skills if s.legacy and s.legacy.compile_success)
        return ok / self.total_skills if self.total_skills else 0.0

    @property
    def descriptor_compile_rate(self) -> float:
        ok = sum(1 for s in self.skills if s.descriptor and s.descriptor.compile_success)
        return ok / self.total_skills if self.total_skills else 0.0

    @property
    def routing_violation_count(self) -> int:
        return sum(1 for s in self.skills if s.routing_violation)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "skills_dir": self.skills_dir,
            "total_skills": self.total_skills,
            "legacy_compile_rate": round(self.legacy_compile_rate, 4),
            "descriptor_compile_rate": round(self.descriptor_compile_rate, 4),
            "compile_rate_delta_pp": round(
                (self.descriptor_compile_rate - self.legacy_compile_rate) * 100, 2
            ),
            "routing_violation_count": self.routing_violation_count,
            "skills": [
                {
                    "name": s.name,
                    "routing": asdict(s.routing) if s.routing else None,
                    "legacy": asdict(s.legacy) if s.legacy else None,
                    "descriptor": asdict(s.descriptor) if s.descriptor else None,
                    "structural_diff_summary": s.structural_diff_summary,
                    "routing_violation": s.routing_violation,
                }
                for s in self.skills
            ],
        }


# --- Skill discovery + routing read ----------------------------------------


def find_skills(skills_dir: Path) -> list[Path]:
    """Return per-skill root directories under `skills_dir`.

    A "skill directory" is any direct child directory of `skills_dir` that
    contains either ``spec.md`` or ``SKILL.md`` (the two filenames the
    legacy ``/mellea-fy`` pipeline accepts — see
    ``mellea_skills_compiler/enums.py:SpecFileFormat``).
    """
    if not skills_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(skills_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "spec.md").exists() or (p / "SKILL.md").exists():
            out.append(p)
    return out


def _spec_path_for(skill_root: Path) -> Path | None:
    """Return the spec file (spec.md preferred, then SKILL.md), or None."""
    for name in ("spec.md", "SKILL.md"):
        candidate = skill_root / name
        if candidate.exists():
            return candidate
    return None


def read_routing(skill_root: Path) -> SkillRouting | None:
    manifest = skill_root / ".melleafy-routing.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return SkillRouting(
        compiler=data["compiler"],
        reason=data["reason"],
        decided_at=data["decided_at"],
        tracking_rfc=data.get("tracking_rfc"),
        review_at=data.get("review_at"),
    )


# --- Pipeline invocations --------------------------------------------------


def _count_fixtures(package_dir: Path) -> int:
    """Best-effort: count .py fixtures in <pkg>/fixtures/ (excluding __init__.py)."""
    fixtures_dir = package_dir / "fixtures"
    if not fixtures_dir.exists():
        return 0
    return sum(1 for p in fixtures_dir.glob("*.py") if p.name != "__init__.py")


def _find_legacy_baseline(skill_root: Path, examples_dir: Path | None = None) -> Path | None:
    """Locate the pre-compiled legacy reference package for a skill.

    Looks under ``examples/<skill-name>/<package_dir>`` for any
    ``*_mellea`` directory.  Returns the first match or None.
    """
    examples_dir = examples_dir or DEFAULT_EXAMPLES_DIR
    candidate = examples_dir / skill_root.name
    if not candidate.exists():
        return None
    for child in sorted(candidate.iterdir()):
        if child.is_dir() and child.name.endswith("_mellea"):
            return child
    return None


def _claude_cli_available() -> bool:
    """Return True if the ``claude`` binary is on PATH.

    Used by ``--require-live`` to fail fast with an actionable message
    before iterating the corpus.  We intentionally don't shell out to
    ``claude --version`` here — being on PATH is enough to delegate to
    ``_spawn_claude``.
    """
    return shutil.which("claude") is not None


def compile_via_legacy(
    skill_root: Path,
    tmp_out: Path,
    *,
    mode: LegacyMode = "live",
    examples_dir: Path | None = None,
    require_live: bool = False,
) -> PipelineResult:
    """Resolve a legacy compiled package for a skill.

    Two modes:

    ``live`` (default since Phase 3.5.F)
        Invoke ``mellea_skills_compiler.compile.mellea_skills.compile``
        which spawns the ``claude`` CLI via the shared P3.5.A
        ``_spawn_claude`` helper.  Requires the ``claude`` CLI +
        Anthropic API key in the environment.  Failure (incl. import /
        claude-cli-missing) is reported as a compile failure rather than
        raised — unless ``require_live`` is True, in which case the
        caller asked us to treat live-spawn failures as fatal (Phase 5
        gate).

    ``baseline`` (CI-fallback)
        Look for an existing ``examples/<skill-name>/<skill-name>_mellea/``
        directory and copy it into ``tmp_out``.  Use this mode in CI
        environments without ``claude`` installed.  If no baseline
        exists, return ``compile_success=False`` with
        ``error="no legacy baseline available"`` — a documented
        limitation, not a corpus_compare bug.
    """
    start = time.perf_counter()
    if mode == "live":
        if require_live and not _claude_cli_available():
            raise LiveSpawnUnavailable(
                "live legacy compile requested but the `claude` binary was "
                "not found on PATH. Install Claude Code "
                "(https://docs.anthropic.com/en/docs/claude-code/quickstart) "
                "or pass --mode=baseline for CI-safe reuse of examples/."
            )
        return _compile_via_legacy_live(skill_root, tmp_out, start)

    baseline = _find_legacy_baseline(skill_root, examples_dir)
    if baseline is None:
        return PipelineResult(
            compiler="legacy",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error="no legacy baseline available",
        )

    # Copy the baseline into tmp_out (idempotent).
    dest = tmp_out
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(baseline, dest)
    return PipelineResult(
        compiler="legacy",
        compile_success=True,
        fixture_pass_count=0,  # baseline mode doesn't re-run fixtures
        fixture_total_count=_count_fixtures(dest),
        wall_clock_seconds=round(time.perf_counter() - start, 4),
        error=None,
        package_dir=str(dest.resolve()),
    )


def _compile_via_legacy_live(skill_root: Path, tmp_out: Path, start: float) -> PipelineResult:
    """Invoke the real legacy compile in-process.

    Routes through ``mellea_skills_compiler.compile.mellea_skills.compile``,
    which spawns the ``claude`` CLI via the shared ``_spawn_claude``
    helper introduced in P3.5.A.  This is the path the Phase 5 gate run
    exercises (``--mode=live --require-live``).

    Best-effort: failures (claude missing, module import error, compile
    raised) are reported as ``compile_success=False`` rather than
    propagated — except when the caller passed ``--require-live``, in
    which case the script-level driver re-raises before reaching here.
    """
    try:
        legacy = importlib.import_module("mellea_skills_compiler.compile.mellea_skills")
    except Exception as exc:  # pragma: no cover - protective
        return PipelineResult(
            compiler="legacy",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"legacy module import failed: {exc!r}",
        )

    spec_path = _spec_path_for(skill_root)
    if spec_path is None:
        return PipelineResult(
            compiler="legacy",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"no spec.md or SKILL.md found in {skill_root}",
        )

    try:
        legacy.compile(spec_path=spec_path, model=None, timeout=4500, no_run=True)
    except Exception as exc:
        return PipelineResult(
            compiler="legacy",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"legacy compile raised: {exc!r}",
        )

    # The legacy compile drops a *_mellea directory next to the spec file.
    skill_dir = spec_path.parent
    mellea_dirs = [d for d in skill_dir.iterdir() if d.is_dir() and d.name.endswith("_mellea")]
    if not mellea_dirs:
        return PipelineResult(
            compiler="legacy",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error="legacy compile produced no *_mellea/ directory",
        )

    pkg = mellea_dirs[0]
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    shutil.copytree(pkg, tmp_out)
    return PipelineResult(
        compiler="legacy",
        compile_success=True,
        fixture_pass_count=0,
        fixture_total_count=_count_fixtures(tmp_out),
        wall_clock_seconds=round(time.perf_counter() - start, 4),
        package_dir=str(tmp_out.resolve()),
    )


def compile_via_descriptor(
    skill_root: Path,
    tmp_out: Path,
    *,
    surface_path: Path | None = None,
) -> PipelineResult:
    """Invoke the descriptor IR compilation pipeline (P3.A).

    Imports ``mellea_skills_compiler.compile.descriptor_emission.compile_via_descriptor``
    lazily.  When P3.A has not landed yet (the module doesn't exist or
    doesn't export the function), the result is a graceful failure so the
    harness still runs and the routing-violation gate remains meaningful.

    The P3.A function is expected to accept ``(spec, surface, write_to=...)``
    and return a ``CompileResult`` carrying ``compile_success`` and the
    output ``package_dir``.  Field names on ``CompileResult`` are read
    defensively because P3.A is in flight; missing fields default
    sensibly.
    """
    start = time.perf_counter()
    spec_path = _spec_path_for(skill_root)
    if spec_path is None:
        return PipelineResult(
            compiler="descriptor",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"no spec.md or SKILL.md found in {skill_root}",
        )

    try:
        mod = importlib.import_module("mellea_skills_compiler.compile.descriptor_emission")
        emit_fn: Callable = getattr(mod, "compile_via_descriptor")
    except Exception as exc:
        return PipelineResult(
            compiler="descriptor",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"descriptor pipeline not yet shipped (P3.A): {exc!r}",
        )

    surface_path = surface_path or DEFAULT_SURFACE
    try:
        spec = spec_path.read_text(encoding="utf-8")
        surface = json.loads(surface_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PipelineResult(
            compiler="descriptor",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"could not read spec/surface inputs: {exc!r}",
        )

    try:
        result = emit_fn(spec, surface, write_to=tmp_out)
    except Exception as exc:
        return PipelineResult(
            compiler="descriptor",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=round(time.perf_counter() - start, 4),
            error=f"descriptor compile raised: {exc!r}",
        )

    # Translate CompileResult → PipelineResult defensively.
    return _coerce_compile_result(result, tmp_out, start)


def _coerce_compile_result(result: Any, tmp_out: Path, start: float) -> PipelineResult:
    """Read a (presumably-dataclass) CompileResult into a PipelineResult.

    The contract with P3.A is approximate while it's in flight, so we
    pull fields by name using getattr-with-default rather than a strict
    type check.
    """
    elapsed = round(time.perf_counter() - start, 4)
    success = bool(getattr(result, "compile_success", False))
    fixture_pass = int(getattr(result, "fixture_pass_count", 0) or 0)
    fixture_total = int(getattr(result, "fixture_total_count", 0) or 0)
    pkg_dir = getattr(result, "package_dir", None)
    error = getattr(result, "error", None)
    if pkg_dir is None and tmp_out.exists():
        pkg_dir = str(tmp_out.resolve())
    if fixture_total == 0 and tmp_out.exists():
        fixture_total = _count_fixtures(tmp_out)
    return PipelineResult(
        compiler="descriptor",
        compile_success=success,
        fixture_pass_count=fixture_pass,
        fixture_total_count=fixture_total,
        wall_clock_seconds=elapsed,
        error=error,
        package_dir=str(pkg_dir) if pkg_dir else None,
    )


# --- Structural diff -------------------------------------------------------


@dataclass
class _PipelineStats:
    found: bool
    line_count: int = 0
    public_signatures: list[str] = field(default_factory=list)
    instruct_calls: int = 0
    for_loops: int = 0
    if_chains: int = 0
    schema_imports: list[str] = field(default_factory=list)


def _analyse_pipeline_py(pipeline_py: Path) -> _PipelineStats:
    """Walk a compiled package's ``pipeline.py`` AST and collect summary stats."""
    if not pipeline_py.exists():
        return _PipelineStats(found=False)

    source = pipeline_py.read_text(encoding="utf-8")
    line_count = source.count("\n") + (0 if source.endswith("\n") else 1)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _PipelineStats(found=True, line_count=line_count)

    stats = _PipelineStats(found=True, line_count=line_count)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            arg_names = [a.arg for a in node.args.args]
            stats.public_signatures.append(f"{node.name}({', '.join(arg_names)})")
        if isinstance(node, ast.ImportFrom) and node.module and "schemas" in node.module:
            stats.schema_imports.extend(sorted(alias.name for alias in node.names))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # m.instruct(...) — attribute access whose attr is 'instruct'.
            # We accept any receiver; some compiled pipelines use the
            # full session reference (session.instruct) and others use
            # the looped alias (m.instruct).
            if isinstance(node.func, ast.Attribute) and node.func.attr == "instruct":
                stats.instruct_calls += 1
        elif isinstance(node, ast.For):
            stats.for_loops += 1
        elif isinstance(node, ast.If):
            stats.if_chains += 1

    # De-duplicate schema imports while preserving sort order.
    stats.schema_imports = sorted(set(stats.schema_imports))
    return stats


def structural_diff(legacy_out: Path, descriptor_out: Path) -> str:
    """Summarise structural differences between two compiled-skill packages.

    Walks both ``pipeline.py`` files via :mod:`ast` and produces a
    short bullet-point summary (3–5 lines).  Informational only —
    consumers spot-check it; the routing-violation gate is the
    machine-readable check.
    """
    legacy_py = legacy_out / "pipeline.py"
    descriptor_py = descriptor_out / "pipeline.py"

    if not legacy_py.exists() and not descriptor_py.exists():
        return "neither output present"
    if not legacy_py.exists():
        return "legacy output missing (no pipeline.py)"
    if not descriptor_py.exists():
        return "descriptor output missing (no pipeline.py)"

    leg = _analyse_pipeline_py(legacy_py)
    desc = _analyse_pipeline_py(descriptor_py)

    lines = [
        f"- signatures: legacy={leg.public_signatures or ['<none>']} "
        f"descriptor={desc.public_signatures or ['<none>']}",
        f"- instruct calls: legacy={leg.instruct_calls} descriptor={desc.instruct_calls} "
        f"(delta={desc.instruct_calls - leg.instruct_calls:+d})",
        f"- for-loops: legacy={leg.for_loops} descriptor={desc.for_loops} | "
        f"if-chains: legacy={leg.if_chains} descriptor={desc.if_chains}",
        f"- schemas imported: legacy={leg.schema_imports} descriptor={desc.schema_imports}",
        f"- line count: legacy={leg.line_count} descriptor={desc.line_count} "
        f"(delta={desc.line_count - leg.line_count:+d})",
    ]
    return "\n".join(lines)


# --- Previous-vs-current drift mode (Phase 3.5 §14.2) ---------------------


def _looks_like_git_rev(value: str) -> bool:
    """Return True if ``value`` smells like a git revision rather than a path.

    Heuristic: not an existing filesystem path AND no path separators AND
    not starting with ``./``/``../``.  Covers ``HEAD~1``, ``v0.5.0``,
    40-char SHAs, branch names — but defers when in doubt by checking
    the filesystem first.
    """
    if value in (".", ".."):
        return False
    if value.startswith(("./", "../", "/")):
        return False
    if "/" in value or "\\" in value:
        return False
    # If it's an existing path on disk, treat as a path.
    if Path(value).exists():
        return False
    return True


def _materialise_previous_dir(
    previous_ref: str,
    skill_name: str,
    workspace: Path,
) -> Path | None:
    """Resolve the "previous" compiled-package dir for ``skill_name``.

    ``previous_ref`` is one of:

    * a filesystem directory whose layout mirrors the corpus
      (``<previous_ref>/<skill_name>/<skill_name>_mellea/`` OR a
      ``<previous_ref>/<skill_name>_mellea/`` shape — both checked), OR
    * a git revision; in which case the function shells out to
      ``git archive <rev> -- <skill_name>`` extracting whichever
      ``*_mellea/`` directory exists at that rev into ``workspace``.

    Returns the resolved ``<skill>_mellea/`` directory, or ``None`` if
    the previous output does not exist on that side of the comparison.
    """
    if not _looks_like_git_rev(previous_ref):
        prev_root = Path(previous_ref)
        # Layout 1: <prev_root>/<skill>/<skill>_mellea/
        candidate_parent = prev_root / skill_name
        if candidate_parent.is_dir():
            for child in candidate_parent.iterdir():
                if child.is_dir() and child.name.endswith("_mellea"):
                    return child
        # Layout 2: <prev_root>/<skill>_mellea/
        for suffix in (f"{skill_name}_mellea", skill_name):
            candidate = prev_root / suffix
            if candidate.is_dir() and candidate.name.endswith("_mellea"):
                return candidate
        return None

    # Git-rev resolution: extract the skill's *_mellea/ tree at ``previous_ref``.
    return _extract_skill_from_git(previous_ref, skill_name, workspace)


def _extract_skill_from_git(rev: str, skill_name: str, workspace: Path) -> Path | None:
    """Use ``git archive <rev> -- <prefix>`` to extract a skill at a rev.

    Two layout candidates are tried in order:

    1. ``examples/<skill_name>/`` (the legacy committed-output location)
    2. ``skills/<skill_name>/`` (the spec dir; rarely contains the
       compiled package, but covered for completeness)

    The extraction lands under ``workspace/<rev-slug>/<skill_name>/``.
    Returns the resolved ``*_mellea/`` subdirectory or ``None``.
    """
    rev_slug = "".join(c if c.isalnum() else "_" for c in rev)[:32]
    target = workspace / f"git-{rev_slug}" / skill_name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for prefix in (f"examples/{skill_name}", f"skills/{skill_name}"):
        archive_path = target.parent / f"{skill_name}-{rev_slug}.tar"
        try:
            with archive_path.open("wb") as fh:
                proc = subprocess.run(
                    ["git", "archive", rev, "--", prefix],
                    cwd=REPO_ROOT,
                    stdout=fh,
                    stderr=subprocess.PIPE,
                    text=False,
                    check=False,
                )
            if proc.returncode != 0:
                continue
            if archive_path.stat().st_size == 0:
                continue
            # Extract the tar into ``target``.
            extract_dir = target
            extract_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["tar", "-xf", str(archive_path), "-C", str(extract_dir)],
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                continue
            # tar extracts under ``<extract_dir>/<prefix>/...``; descend.
            extracted = extract_dir / prefix
            if not extracted.is_dir():
                continue
            for child in extracted.iterdir():
                if child.is_dir() and child.name.endswith("_mellea"):
                    return child
        except FileNotFoundError:
            # `git` or `tar` not on PATH; abandon the git-rev path.
            return None
        finally:
            if archive_path.exists():
                archive_path.unlink()

    return None


def compute_skill_drift(
    skill_root: Path,
    *,
    previous_ref: str,
    recompiled_package: Path | None,
    workspace: Path,
) -> "DriftSkillSummary":
    """Compare ``previous_ref``'s snapshot of the skill against
    ``recompiled_package``.

    Returns a DriftSkillSummary containing the classified verdict plus
    the resolved previous-dir path so the report can record it.
    """
    from mellea_skills_compiler.drift import classify_skill_drift

    prev_dir = _materialise_previous_dir(previous_ref, skill_root.name, workspace)
    if prev_dir is None:
        # No previous output exists for this skill — surface as a note
        # rather than a phantom drift.
        from mellea_skills_compiler.drift import DriftClass, SkillDriftVerdict

        return DriftSkillSummary(
            verdict=SkillDriftVerdict(
                skill=skill_root.name,
                drift_class=DriftClass.STRUCTURAL,
                notes=[
                    f"no previous output resolved for {skill_root.name} at "
                    f"`{previous_ref}` (treated as first-time compile)"
                ],
            ),
            previous_dir=None,
        )

    if recompiled_package is None:
        from mellea_skills_compiler.drift import DriftClass, SkillDriftVerdict

        return DriftSkillSummary(
            verdict=SkillDriftVerdict(
                skill=skill_root.name,
                drift_class=DriftClass.STRUCTURAL,
                notes=[
                    "current descriptor compile produced no package — "
                    "compilation may have failed"
                ],
            ),
            previous_dir=prev_dir,
        )

    verdict = classify_skill_drift(skill_root.name, prev_dir, recompiled_package)
    return DriftSkillSummary(verdict=verdict, previous_dir=prev_dir)


@dataclass
class DriftSkillSummary:
    """Bundle of a drift verdict + the resolved previous-dir path."""

    verdict: Any  # SkillDriftVerdict (avoid hard import at module top)
    previous_dir: Path | None


# --- Driver ----------------------------------------------------------------


def compare_one(
    skill_root: Path,
    tmp_root: Path,
    *,
    legacy_mode: LegacyMode = "live",
    surface_path: Path | None = None,
    examples_dir: Path | None = None,
    run_descriptor: bool = True,
    require_live: bool = False,
) -> SkillResult:
    name = skill_root.name
    routing = read_routing(skill_root)

    tmp_legacy = tmp_root / f"{name}_legacy"
    tmp_descriptor = tmp_root / f"{name}_descriptor"
    tmp_legacy.parent.mkdir(parents=True, exist_ok=True)
    tmp_descriptor.parent.mkdir(parents=True, exist_ok=True)

    # Per the policy: a skill routed to legacy SHOULD NOT be tested through
    # the descriptor pipeline (avoids spurious CI failures / cost). The
    # ``run_descriptor`` knob lets the caller opt out; default is on so the
    # baseline default-routing report has data for every skill.
    legacy = compile_via_legacy(
        skill_root,
        tmp_legacy,
        mode=legacy_mode,
        examples_dir=examples_dir,
        require_live=require_live,
    )

    if run_descriptor:
        descriptor = compile_via_descriptor(
            skill_root, tmp_descriptor, surface_path=surface_path
        )
    else:
        descriptor = PipelineResult(
            compiler="descriptor",
            compile_success=False,
            fixture_pass_count=0,
            fixture_total_count=0,
            wall_clock_seconds=0.0,
            error="skipped: routed to legacy and --skip-descriptor-for-legacy-routed in effect",
        )

    diff = structural_diff(tmp_legacy, tmp_descriptor)

    return SkillResult(
        name=name,
        routing=routing,
        legacy=legacy,
        descriptor=descriptor,
        structural_diff_summary=diff,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="corpus_compare.py",
        description="Compare legacy + descriptor compilation pipelines across the skill corpus.",
    )
    parser.add_argument("--skills", type=Path, default=DEFAULT_SKILLS_DIR, help="Skills root directory.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/tmp/corpus-compare"),
        help="Where intermediate compilations land.",
    )

    # Legacy-mode selector.  ``--mode`` is the canonical name introduced in
    # Phase 3.5.F; ``--legacy-mode`` is preserved as an alias for callers
    # still wired against the older name (CI workflows, docs, the SOP).
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=("baseline", "live"),
        default=None,
        help="legacy invocation mode. live (default, Phase 3.5.F): drive the "
        "real `claude` subprocess via the shared _spawn_claude helper "
        "(requires claude CLI + creds). baseline: reuse "
        "examples/<skill>/<skill>_mellea/ (CI fallback for headless workers).",
    )
    mode_group.add_argument(
        "--legacy-mode",
        choices=("baseline", "live"),
        default=None,
        help="DEPRECATED: alias for --mode kept for backward compat.",
    )

    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Phase 5 gate: fail loudly if the `claude` CLI cannot be "
        "spawned, instead of silently degrading. Use this in the corpus "
        "run that gates the descriptor-default flip.",
    )

    parser.add_argument(
        "--surface",
        type=Path,
        default=DEFAULT_SURFACE,
        help="Path to surface_<version>.json consumed by the descriptor pipeline.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=DEFAULT_EXAMPLES_DIR,
        help="Root of pre-compiled legacy baselines (default: <repo>/examples).",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero if routing_violation_count > 0 (CI use).",
    )

    # Phase 3.5 §14 — drift detection.
    parser.add_argument(
        "--compare-against-previous",
        action="store_true",
        help="Enable the previous-vs-current drift comparison (Phase 3.5 §14.2). "
        "Requires --previous-dir.",
    )
    parser.add_argument(
        "--previous-dir",
        type=str,
        default=None,
        help="Path or git rev to compare against. Filesystem paths point at a "
        "directory whose children mirror skills/ layout; git revs are anything "
        "git can resolve (HEAD~1, v0.5.0, SHA, branch).",
    )
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=DEFAULT_DRIFT_REPORT,
        help="Path for the aggregated recompile_drift_report.md.",
    )
    parser.add_argument(
        "--mellea-version",
        type=str,
        default=None,
        help="Label written into the drift report header (informational).",
    )
    parser.add_argument(
        "--recompile-trigger",
        type=str,
        default=None,
        help="Free-text label written into the drift report header — e.g. "
        "'Mellea 0.5 → 0.6 bump', 'quarterly drift audit'.",
    )
    args = parser.parse_args(argv)

    # Resolve --mode / --legacy-mode (default: live, Phase 3.5.F).
    resolved_mode: LegacyMode = "live"
    if args.mode is not None:
        resolved_mode = args.mode  # type: ignore[assignment]
    elif args.legacy_mode is not None:
        resolved_mode = args.legacy_mode  # type: ignore[assignment]

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Phase 3.5.F: pre-flight check — if --require-live is set, fail before
    # any compile loop so the operator gets an immediate, actionable error.
    if args.require_live and resolved_mode != "live":
        print(
            "--require-live is set but --mode={mode}; pass --mode=live "
            "(or omit --mode) to use the real claude subprocess.".format(
                mode=resolved_mode
            ),
            file=sys.stderr,
        )
        return 2
    if args.require_live and not _claude_cli_available():
        print(
            "--require-live is set but the `claude` binary is not on PATH. "
            "Install Claude Code (https://docs.anthropic.com/en/docs/claude-code/quickstart) "
            "or drop --require-live for baseline-mode runs.",
            file=sys.stderr,
        )
        return 2

    if args.compare_against_previous and not args.previous_dir:
        print(
            "--compare-against-previous requires --previous-dir <path-or-git-rev>.",
            file=sys.stderr,
        )
        return 2

    skills = find_skills(args.skills)
    if not skills:
        print(f"no skills found under {args.skills}", file=sys.stderr)
        return 2

    report = CorpusReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        skills_dir=str(args.skills),
        total_skills=len(skills),
    )

    drift_summaries: list[DriftSkillSummary] = []
    drift_workspace: Path | None = None
    if args.compare_against_previous:
        drift_workspace = args.tmp_dir / "_drift"
        drift_workspace.mkdir(parents=True, exist_ok=True)

    print(f"comparing {len(skills)} skills (mode={resolved_mode})…", flush=True)
    for skill_root in skills:
        try:
            result = compare_one(
                skill_root,
                args.tmp_dir,
                legacy_mode=resolved_mode,
                surface_path=args.surface,
                examples_dir=args.examples_dir,
                require_live=args.require_live,
            )
        except LiveSpawnUnavailable as exc:
            # Hard fail mid-loop with an actionable message; --require-live
            # explicitly asked for no silent fallback.
            print(f"FATAL: {exc}", file=sys.stderr)
            return 3
        report.skills.append(result)
        leg = "ok" if result.legacy and result.legacy.compile_success else "fail"
        desc = "ok" if result.descriptor and result.descriptor.compile_success else "fail"
        flag = " !VIOLATION" if result.routing_violation else ""

        drift_tag = ""
        if args.compare_against_previous:
            assert drift_workspace is not None  # for type-narrowing
            recompiled_pkg = (
                Path(result.descriptor.package_dir)
                if result.descriptor and result.descriptor.package_dir
                else None
            )
            summary = compute_skill_drift(
                skill_root,
                previous_ref=args.previous_dir,
                recompiled_package=recompiled_pkg,
                workspace=drift_workspace,
            )
            drift_summaries.append(summary)
            drift_tag = f" drift={summary.verdict.drift_class.value}"
        print(f"  {result.name}: legacy={leg} descriptor={desc}{flag}{drift_tag}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"\nSummary: legacy compile rate {report.legacy_compile_rate*100:.0f}%, "
        f"descriptor compile rate {report.descriptor_compile_rate*100:.0f}%, "
        f"routing violations {report.routing_violation_count}"
    )
    try:
        rel = args.out.relative_to(REPO_ROOT)
    except ValueError:
        rel = args.out
    print(f"Report: {rel}")

    # --- Drift report rendering ---
    hard_stop_drift = False
    if args.compare_against_previous and drift_summaries:
        from mellea_skills_compiler.drift import DriftClass, render_drift_report

        verdicts = [s.verdict for s in drift_summaries]
        md = render_drift_report(
            verdicts,
            mellea_version=args.mellea_version,
            recompile_trigger=args.recompile_trigger,
            previous_ref=args.previous_dir,
        )
        args.drift_report.parent.mkdir(parents=True, exist_ok=True)
        args.drift_report.write_text(md, encoding="utf-8")
        try:
            drift_rel = args.drift_report.relative_to(REPO_ROOT)
        except ValueError:
            drift_rel = args.drift_report
        print(f"Drift report: {drift_rel}")

        semantic_count = sum(
            1 for v in verdicts if v.drift_class == DriftClass.SEMANTIC
        )
        total = len(verdicts)
        if total and (semantic_count / total) > 0.05:
            hard_stop_drift = True
            print(
                f"HARD STOP: semantic drift on {semantic_count}/{total} skills "
                f"({semantic_count / total * 100:.1f}% > 5% — Phase 3.5 §14.4).",
                file=sys.stderr,
            )

    if args.fail_on_violation and report.routing_violation_count > 0:
        return 1
    if hard_stop_drift:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
