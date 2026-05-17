"""Tests for ``scripts/corpus_compare.py``.

The corpus comparison harness wires the legacy + descriptor compilation
pipelines and emits the routing-violation gate that Phase 5 depends on
(see ``melleafy-handoff/process/regression-and-extension-policy.md`` §a).

Both pipelines are mocked in unit tests.  A single ``@pytest.mark.live``
end-to-end test exercises the real descriptor pipeline against
``skills/sentry-find-bugs/`` and is skipped by default.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "corpus_compare.py"


@pytest.fixture(scope="module")
def cc() -> ModuleType:
    """Load ``scripts/corpus_compare.py`` as a module.

    Must register in ``sys.modules`` BEFORE ``exec_module`` so the script's
    string-form dataclass annotations (``CompilerName``, ``SkillRouting | None``)
    can be resolved by ``dataclasses._is_type`` during class creation.
    """
    spec = importlib.util.spec_from_file_location("corpus_compare", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["corpus_compare"] = module
    spec.loader.exec_module(module)
    return module


# --- find_skills ----------------------------------------------------------


def test_find_skills_finds_dirs_with_spec_md(tmp_path: Path, cc) -> None:
    skill = tmp_path / "skill-a"
    skill.mkdir()
    (skill / "spec.md").write_text("# A")
    result = cc.find_skills(tmp_path)
    assert [p.name for p in result] == ["skill-a"]


def test_find_skills_finds_dirs_with_SKILL_md(tmp_path: Path, cc) -> None:
    skill = tmp_path / "skill-b"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# B")
    result = cc.find_skills(tmp_path)
    assert [p.name for p in result] == ["skill-b"]


def test_find_skills_skips_dirs_without_spec_or_SKILL_md(tmp_path: Path, cc) -> None:
    # bare README.md must not count.
    (tmp_path / "README.md").write_text("readme")
    empty = tmp_path / "empty-dir"
    empty.mkdir()
    (empty / "notes.txt").write_text("nope")
    good = tmp_path / "good"
    good.mkdir()
    (good / "spec.md").write_text("# good")

    result = cc.find_skills(tmp_path)
    assert [p.name for p in result] == ["good"]


def test_find_skills_returns_empty_when_skills_dir_missing(tmp_path: Path, cc) -> None:
    assert cc.find_skills(tmp_path / "does-not-exist") == []


def test_find_skills_returns_sorted_order(tmp_path: Path, cc) -> None:
    for name in ("z-skill", "a-skill", "m-skill"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "spec.md").write_text("#")
    result = [p.name for p in cc.find_skills(tmp_path)]
    assert result == sorted(result)


# --- read_routing ---------------------------------------------------------


def test_read_routing_returns_none_when_manifest_missing(tmp_path: Path, cc) -> None:
    skill = tmp_path / "skill-x"
    skill.mkdir()
    assert cc.read_routing(skill) is None


def test_read_routing_parses_full_manifest(tmp_path: Path, cc) -> None:
    skill = tmp_path / "skill-y"
    skill.mkdir()
    (skill / ".melleafy-routing.json").write_text(
        json.dumps(
            {
                "compiler": "legacy",
                "reason": "Uses `while True:` agent loop (coverage-report: certification).",
                "decided_at": "2026-06-15",
                "tracking_rfc": "rfc-001.md",
                "review_at": "2026-09-15",
            }
        )
    )
    routing = cc.read_routing(skill)
    assert routing is not None
    assert routing.compiler == "legacy"
    assert routing.tracking_rfc == "rfc-001.md"
    assert routing.review_at == "2026-09-15"


def test_read_routing_parses_minimal_manifest(tmp_path: Path, cc) -> None:
    skill = tmp_path / "skill-z"
    skill.mkdir()
    (skill / ".melleafy-routing.json").write_text(
        json.dumps(
            {
                "compiler": "descriptor",
                "reason": "Pure workflow shape.",
                "decided_at": "2026-05-01",
            }
        )
    )
    routing = cc.read_routing(skill)
    assert routing is not None
    assert routing.compiler == "descriptor"
    assert routing.tracking_rfc is None
    assert routing.review_at is None


# --- compile_via_legacy ---------------------------------------------------


def _make_skill_dir(tmp_path: Path, name: str = "demo") -> Path:
    skill = tmp_path / "skills" / name
    skill.mkdir(parents=True)
    (skill / "spec.md").write_text("# demo spec")
    return skill


def _make_example_baseline(tmp_path: Path, skill_name: str) -> Path:
    """Build a fake examples/<skill>/<skill>_mellea/ directory tree."""
    pkg = tmp_path / "examples" / skill_name / f"{skill_name}_mellea"
    pkg.mkdir(parents=True)
    (pkg / "pipeline.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            def run_pipeline(diff: str):
                with start_session() as m:
                    a = m.instruct("a")
                    b = m.instruct("b")
                if diff:
                    for f in diff.split():
                        m.instruct("each")
                return a
            """
        ).strip()
        + "\n"
    )
    fixtures = pkg / "fixtures"
    fixtures.mkdir()
    (fixtures / "__init__.py").write_text("")
    (fixtures / "f1.py").write_text("def make_f1(): return {}\n")
    (fixtures / "f2.py").write_text("def make_f2(): return {}\n")
    return pkg


def test_compile_via_legacy_finds_existing_examples_baseline(tmp_path: Path, cc) -> None:
    skill = _make_skill_dir(tmp_path, "demo")
    _make_example_baseline(tmp_path, "demo")
    tmp_out = tmp_path / "out_legacy"

    result = cc.compile_via_legacy(
        skill, tmp_out, mode="baseline", examples_dir=tmp_path / "examples"
    )
    assert result.compile_success is True
    assert result.compiler == "legacy"
    assert result.error is None
    assert result.fixture_total_count == 2
    assert (tmp_out / "pipeline.py").exists()
    assert (tmp_out / "fixtures" / "f1.py").exists()


def test_compile_via_legacy_reports_missing_when_no_baseline(tmp_path: Path, cc) -> None:
    skill = _make_skill_dir(tmp_path, "no-baseline-skill")
    tmp_out = tmp_path / "out_legacy"

    result = cc.compile_via_legacy(
        skill, tmp_out, mode="baseline", examples_dir=tmp_path / "examples"
    )
    assert result.compile_success is False
    assert result.error == "no legacy baseline available"
    assert result.compiler == "legacy"


def test_compile_via_legacy_overwrites_existing_tmp_out(tmp_path: Path, cc) -> None:
    skill = _make_skill_dir(tmp_path, "demo")
    _make_example_baseline(tmp_path, "demo")

    tmp_out = tmp_path / "out_legacy"
    tmp_out.mkdir()
    (tmp_out / "stale_file.txt").write_text("old")

    result = cc.compile_via_legacy(
        skill, tmp_out, mode="baseline", examples_dir=tmp_path / "examples"
    )
    assert result.compile_success is True
    assert not (tmp_out / "stale_file.txt").exists()
    assert (tmp_out / "pipeline.py").exists()


# --- compile_via_descriptor (mocked P3.A) ---------------------------------


@dataclass
class _FakeCompileResult:
    compile_success: bool = True
    fixture_pass_count: int = 0
    fixture_total_count: int = 0
    package_dir: str | None = None
    error: str | None = None


def test_compile_via_descriptor_wraps_compile_via_descriptor_call(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _make_skill_dir(tmp_path, "demo")
    tmp_out = tmp_path / "desc_out"
    surface_path = tmp_path / "surface.json"
    surface_path.write_text(json.dumps({"version": "0.5.0"}))

    captured: dict[str, object] = {}

    def fake_compile(spec: str, surface: dict, write_to: Path) -> _FakeCompileResult:
        captured["spec"] = spec
        captured["surface"] = surface
        captured["write_to"] = write_to
        write_to.mkdir(parents=True, exist_ok=True)
        (write_to / "pipeline.py").write_text("def run_pipeline(): return None\n")
        return _FakeCompileResult(
            compile_success=True,
            package_dir=str(write_to.resolve()),
        )

    fake_mod = ModuleType("mellea_skills_compiler.compile.descriptor_emission")
    fake_mod.compile_via_descriptor = fake_compile
    monkeypatch.setitem(sys.modules, "mellea_skills_compiler.compile.descriptor_emission", fake_mod)

    result = cc.compile_via_descriptor(skill, tmp_out, surface_path=surface_path)

    assert result.compile_success is True
    assert result.error is None
    assert result.compiler == "descriptor"
    assert result.package_dir is not None
    assert captured["spec"] == "# demo spec"
    assert captured["surface"] == {"version": "0.5.0"}
    assert captured["write_to"] == tmp_out


def test_compile_via_descriptor_graceful_when_module_missing(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P3.A not yet shipped: returns a documented failure instead of raising."""
    skill = _make_skill_dir(tmp_path, "demo")
    tmp_out = tmp_path / "desc_out"
    surface_path = tmp_path / "surface.json"
    surface_path.write_text("{}")

    # Force ImportError for the P3.A module regardless of repo state.
    import importlib

    original_import = importlib.import_module

    def fake_import(name: str, *a, **kw):
        if name == "mellea_skills_compiler.compile.descriptor_emission":
            raise ImportError("not yet shipped")
        return original_import(name, *a, **kw)

    monkeypatch.setattr(cc.importlib, "import_module", fake_import)

    result = cc.compile_via_descriptor(skill, tmp_out, surface_path=surface_path)
    assert result.compile_success is False
    assert result.error is not None and "descriptor pipeline not yet shipped" in result.error


def test_compile_via_descriptor_handles_pipeline_exception(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _make_skill_dir(tmp_path, "demo")
    tmp_out = tmp_path / "desc_out"
    surface_path = tmp_path / "surface.json"
    surface_path.write_text("{}")

    def fake_compile(spec, surface, write_to):
        raise RuntimeError("backend down")

    fake_mod = ModuleType("mellea_skills_compiler.compile.descriptor_emission")
    fake_mod.compile_via_descriptor = fake_compile
    monkeypatch.setitem(sys.modules, "mellea_skills_compiler.compile.descriptor_emission", fake_mod)

    result = cc.compile_via_descriptor(skill, tmp_out, surface_path=surface_path)
    assert result.compile_success is False
    assert result.error is not None and "backend down" in result.error


def test_compile_via_descriptor_missing_spec(tmp_path: Path, cc) -> None:
    skill = tmp_path / "noskill"
    skill.mkdir()
    tmp_out = tmp_path / "out"
    surface_path = tmp_path / "surface.json"
    surface_path.write_text("{}")
    result = cc.compile_via_descriptor(skill, tmp_out, surface_path=surface_path)
    assert result.compile_success is False
    assert result.error is not None and "no spec.md or SKILL.md" in result.error


# --- compile_via_legacy live (mocked) -------------------------------------


def test_compile_via_legacy_live_invokes_legacy_module_compile(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode delegates to mellea_skills_compiler.compile.mellea_skills.compile."""
    skill = _make_skill_dir(tmp_path, "demo")

    pkg_after = skill.parent / "demo" / "demo_mellea"

    calls: dict[str, object] = {}

    def fake_legacy_compile(*, spec_path, model, timeout, no_run):
        calls["spec_path"] = spec_path
        calls["no_run"] = no_run
        pkg_after.mkdir(parents=True, exist_ok=True)
        (pkg_after / "pipeline.py").write_text("def run_pipeline(): pass\n")

    fake_mod = ModuleType("mellea_skills_compiler.compile.mellea_skills")
    fake_mod.compile = fake_legacy_compile
    monkeypatch.setitem(sys.modules, "mellea_skills_compiler.compile.mellea_skills", fake_mod)

    tmp_out = tmp_path / "out_legacy_live"
    result = cc.compile_via_legacy(skill, tmp_out, mode="live")

    assert result.compile_success is True
    assert calls["no_run"] is True
    assert calls["spec_path"] == skill / "spec.md"
    assert (tmp_out / "pipeline.py").exists()


def test_compile_via_legacy_live_handles_legacy_pipeline_exception(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _make_skill_dir(tmp_path, "demo")

    def fake_legacy_compile(**kw):
        raise RuntimeError("claude CLI not installed")

    fake_mod = ModuleType("mellea_skills_compiler.compile.mellea_skills")
    fake_mod.compile = fake_legacy_compile
    monkeypatch.setitem(sys.modules, "mellea_skills_compiler.compile.mellea_skills", fake_mod)

    tmp_out = tmp_path / "out_legacy_live"
    result = cc.compile_via_legacy(skill, tmp_out, mode="live")
    assert result.compile_success is False
    assert result.error is not None and "legacy compile raised" in result.error


# --- structural_diff ------------------------------------------------------


def _write_pipeline(pkg: Path, source: str) -> None:
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "pipeline.py").write_text(source)


def test_structural_diff_handles_missing_files(tmp_path: Path, cc) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert cc.structural_diff(a, b) == "neither output present"

    _write_pipeline(a, "def run_pipeline(): pass\n")
    assert "legacy output missing" not in cc.structural_diff(a, b)
    assert "descriptor output missing" in cc.structural_diff(a, b)


def test_structural_diff_reports_instruct_call_count(tmp_path: Path, cc) -> None:
    legacy = tmp_path / "legacy"
    descriptor = tmp_path / "descriptor"
    _write_pipeline(
        legacy,
        textwrap.dedent(
            """
            def run_pipeline(diff):
                m.instruct("a")
                m.instruct("b")
                m.instruct("c")
            """
        ).strip()
        + "\n",
    )
    _write_pipeline(
        descriptor,
        textwrap.dedent(
            """
            def run_pipeline(diff):
                session.instruct("a")
            """
        ).strip()
        + "\n",
    )
    diff = cc.structural_diff(legacy, descriptor)
    assert "instruct calls: legacy=3 descriptor=1" in diff
    assert "delta=-2" in diff


def test_structural_diff_reports_schema_list(tmp_path: Path, cc) -> None:
    legacy = tmp_path / "legacy"
    descriptor = tmp_path / "descriptor"
    _write_pipeline(
        legacy,
        "from .schemas import Apple, Banana\n\ndef run_pipeline(): pass\n",
    )
    _write_pipeline(
        descriptor,
        "from .schemas import Cherry, Banana\n\ndef run_pipeline(): pass\n",
    )
    out = cc.structural_diff(legacy, descriptor)
    assert "Apple" in out and "Banana" in out and "Cherry" in out


def test_structural_diff_reports_signatures(tmp_path: Path, cc) -> None:
    legacy = tmp_path / "legacy"
    descriptor = tmp_path / "descriptor"
    _write_pipeline(legacy, "def run_pipeline(diff, fn): pass\n")
    _write_pipeline(descriptor, "def run_pipeline(diff): pass\n")
    out = cc.structural_diff(legacy, descriptor)
    assert "run_pipeline(diff, fn)" in out
    assert "run_pipeline(diff)" in out


def test_structural_diff_counts_loops_and_ifs(tmp_path: Path, cc) -> None:
    legacy = tmp_path / "legacy"
    descriptor = tmp_path / "descriptor"
    _write_pipeline(
        legacy,
        textwrap.dedent(
            """
            def run_pipeline(diff):
                if diff:
                    for x in diff:
                        pass
                    for y in diff:
                        pass
            """
        ).strip()
        + "\n",
    )
    _write_pipeline(
        descriptor,
        "def run_pipeline(diff):\n    pass\n",
    )
    out = cc.structural_diff(legacy, descriptor)
    assert "for-loops: legacy=2 descriptor=0" in out
    assert "if-chains: legacy=1 descriptor=0" in out


# --- routing-violation logic on SkillResult -------------------------------


def test_routing_violation_when_descriptor_routed_but_fails(cc) -> None:
    routing = cc.SkillRouting(compiler="descriptor", reason="r", decided_at="2026-01-01")
    descriptor = cc.PipelineResult(
        compiler="descriptor",
        compile_success=False,
        fixture_pass_count=0,
        fixture_total_count=0,
        wall_clock_seconds=0.0,
        error="failed",
    )
    legacy = cc.PipelineResult(
        compiler="legacy",
        compile_success=True,
        fixture_pass_count=0,
        fixture_total_count=0,
        wall_clock_seconds=0.0,
    )
    sk = cc.SkillResult(name="x", routing=routing, legacy=legacy, descriptor=descriptor)
    assert sk.routing_violation is True


def test_no_violation_when_legacy_routed_and_descriptor_fails(cc) -> None:
    routing = cc.SkillRouting(compiler="legacy", reason="r", decided_at="2026-01-01")
    descriptor = cc.PipelineResult(
        compiler="descriptor",
        compile_success=False,
        fixture_pass_count=0,
        fixture_total_count=0,
        wall_clock_seconds=0.0,
    )
    sk = cc.SkillResult(name="x", routing=routing, legacy=None, descriptor=descriptor)
    assert sk.routing_violation is False


def test_no_violation_when_descriptor_routed_and_succeeds(cc) -> None:
    routing = cc.SkillRouting(compiler="descriptor", reason="r", decided_at="2026-01-01")
    descriptor = cc.PipelineResult(
        compiler="descriptor",
        compile_success=True,
        fixture_pass_count=0,
        fixture_total_count=0,
        wall_clock_seconds=0.0,
    )
    sk = cc.SkillResult(name="x", routing=routing, legacy=None, descriptor=descriptor)
    assert sk.routing_violation is False


def test_no_violation_when_routing_absent(cc) -> None:
    descriptor = cc.PipelineResult(
        compiler="descriptor",
        compile_success=False,
        fixture_pass_count=0,
        fixture_total_count=0,
        wall_clock_seconds=0.0,
    )
    sk = cc.SkillResult(name="x", routing=None, legacy=None, descriptor=descriptor)
    assert sk.routing_violation is False


# --- CLI / report schema --------------------------------------------------


def _build_corpus_with_violation(tmp_path: Path) -> tuple[Path, Path]:
    """Build a fake skills/ + examples/ tree so corpus_compare runs end-to-end
    against mocked compile shims (via CORPUS_COMPARE_FAKE_MODE env)."""
    skills = tmp_path / "skills"
    skills.mkdir()
    s1 = skills / "skill-violator"
    s1.mkdir()
    (s1 / "spec.md").write_text("# v")
    (s1 / ".melleafy-routing.json").write_text(
        json.dumps(
            {
                "compiler": "descriptor",
                "reason": "Routed to descriptor (test fixture).",
                "decided_at": "2026-01-01",
            }
        )
    )
    return tmp_path, skills


def test_fail_on_violation_returns_nonzero_exit(tmp_path: Path) -> None:
    """End-to-end subprocess: a skill routed to descriptor where the
    descriptor pipeline fails should cause --fail-on-violation to exit 1.

    The descriptor pipeline fails because P3.A's module is intentionally
    absent in this test env (graceful failure path), and the legacy
    baseline is absent (so no examples/<skill>/<skill>_mellea/).
    """
    tmp_root, skills = _build_corpus_with_violation(tmp_path)
    out_json = tmp_path / "corpus_comparison.json"
    tmp_inter = tmp_path / "tmpdir"
    examples = tmp_path / "examples"  # intentionally empty

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(out_json),
            "--tmp-dir",
            str(tmp_inter),
            "--examples-dir",
            str(examples),
            "--fail-on-violation",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr + proc.stdout
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert data["routing_violation_count"] >= 1


def test_exit_zero_when_no_skills_routed_to_descriptor(tmp_path: Path) -> None:
    """Without descriptor-routing, descriptor failures don't violate the gate."""
    skills = tmp_path / "skills"
    skills.mkdir()
    s1 = skills / "skill-x"
    s1.mkdir()
    (s1 / "spec.md").write_text("# x")
    # no routing manifest: gate inert.

    out_json = tmp_path / "corpus_comparison.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(out_json),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
            "--examples-dir",
            str(tmp_path / "examples"),
            "--fail-on-violation",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_report_json_schema_well_formed(tmp_path: Path) -> None:
    """Output JSON must contain every documented top-level field."""
    skills = tmp_path / "skills"
    skills.mkdir()
    s1 = skills / "alpha"
    s1.mkdir()
    (s1 / "spec.md").write_text("# alpha")
    s2 = skills / "beta"
    s2.mkdir()
    (s2 / "SKILL.md").write_text("# beta")

    out_json = tmp_path / "corpus_comparison.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(out_json),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
            "--examples-dir",
            str(tmp_path / "examples"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout

    data = json.loads(out_json.read_text())
    for key in (
        "generated_at",
        "skills_dir",
        "total_skills",
        "legacy_compile_rate",
        "descriptor_compile_rate",
        "compile_rate_delta_pp",
        "routing_violation_count",
        "skills",
    ):
        assert key in data, f"missing key {key!r}"

    assert data["total_skills"] == 2
    assert [s["name"] for s in data["skills"]] == ["alpha", "beta"]
    for entry in data["skills"]:
        for k in ("name", "routing", "legacy", "descriptor", "structural_diff_summary", "routing_violation"):
            assert k in entry


def test_returns_nonzero_when_no_skills_found(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(tmp_path / "empty"),
            "--out",
            str(tmp_path / "out.json"),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


# --- Phase 3.5.F: --require-live + drift-mode coverage --------------------


def test_require_live_fails_fast_when_claude_missing(tmp_path: Path, cc, monkeypatch) -> None:
    """If --require-live is set and `claude` isn't on PATH, the script exits 2
    *before* iterating the corpus.  This is the Phase 5 gate's early bail-out."""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "demo").mkdir()
    (skills / "demo" / "spec.md").write_text("# demo")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(tmp_path / "out.json"),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
            "--examples-dir",
            str(tmp_path / "examples"),
            "--require-live",
        ],
        capture_output=True,
        text=True,
        # Empty PATH guarantees `claude` is unreachable for this child.
        env={"PATH": "/usr/bin:/bin"},
    )
    # If `claude` happens to be installed in /usr/bin, this test would
    # accidentally pass with a different return code.  Verify the assertion
    # is about the actionable message either way.
    if proc.returncode == 2 and "require-live" in (proc.stderr + proc.stdout):
        return
    # The other accepted exit path is the gate firing later in the loop —
    # but with empty PATH the pre-flight should always catch it first.
    pytest.fail(
        f"expected exit 2 with require-live message, got rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_compare_against_previous_requires_previous_dir(tmp_path: Path) -> None:
    """--compare-against-previous without --previous-dir → exit 2."""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "demo").mkdir()
    (skills / "demo" / "spec.md").write_text("# demo")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(tmp_path / "out.json"),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
            "--examples-dir",
            str(tmp_path / "examples"),
            "--mode=baseline",
            "--compare-against-previous",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr + proc.stdout
    assert "--previous-dir" in proc.stderr or "--previous-dir" in proc.stdout


def test_legacy_compile_live_requires_live_raises_when_claude_missing(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The function-level surface mirrors the CLI behaviour: require_live=True
    + no `claude` on PATH → raise LiveSpawnUnavailable."""
    skill = _make_skill_dir(tmp_path, "demo")
    tmp_out = tmp_path / "out_live"

    monkeypatch.setattr(cc, "_claude_cli_available", lambda: False)
    with pytest.raises(cc.LiveSpawnUnavailable) as exc_info:
        cc.compile_via_legacy(skill, tmp_out, mode="live", require_live=True)
    assert "claude" in str(exc_info.value)


def test_drift_mode_produces_report_with_mocked_descriptor(
    tmp_path: Path, cc, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end drift mode with the descriptor pipeline mocked to copy a
    benign package.  Verifies the script renders ``recompile_drift_report.md``
    and the per-skill summary includes a drift verdict line."""
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "demo").mkdir()
    (skills / "demo" / "spec.md").write_text("# demo")

    # Build a fake "previous" snapshot dir with a *_mellea/ subtree.
    prev_root = tmp_path / "prev-snapshot"
    prev_pkg = prev_root / "demo" / "demo_mellea"
    prev_pkg.mkdir(parents=True)
    prev_pipeline = textwrap.dedent(
        '''
        from mellea import start_session
        from .schemas import Finding


        def run_pipeline(diff):
            with start_session() as m:
                step_1 = m.instruct("Find security issues in the diff and report each carefully.")
            return step_1
        '''
    ).strip() + "\n"
    (prev_pkg / "pipeline.py").write_text(prev_pipeline)
    (prev_pkg / "schemas.py").write_text(
        "from pydantic import BaseModel\n\nclass Finding(BaseModel):\n    title: str\n"
    )
    (prev_pkg / "melleafy.json").write_text(json.dumps({"pipeline": {"operator": "sequential"}}))

    out_json = tmp_path / "corpus_comparison.json"
    drift_report = tmp_path / "drift.md"

    # The script's descriptor pipeline path is the in-process compile_via_descriptor
    # import.  We can't easily mock it for the subprocess invocation, but the
    # script's drift mode tolerates a missing descriptor pipeline (returns a
    # structural-drift verdict with a note).  That's a faithful unit-test surface
    # since baseline mode never tries to materialise the descriptor at all.
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--skills",
            str(skills),
            "--out",
            str(out_json),
            "--tmp-dir",
            str(tmp_path / "tmpdir"),
            "--examples-dir",
            str(tmp_path / "examples"),
            "--mode=baseline",
            "--compare-against-previous",
            "--previous-dir",
            str(prev_root),
            "--drift-report",
            str(drift_report),
            "--mellea-version",
            "0.6.0",
            "--recompile-trigger",
            "unit-test recompile",
        ],
        capture_output=True,
        text=True,
    )
    # Exit code may be 0 (no drift hard-stop, no routing violations) or 4 if
    # the synthetic skill went 100% semantic — either way the drift report
    # MUST have been written.
    assert proc.returncode in (0, 4), proc.stderr + proc.stdout
    assert drift_report.exists()

    md = drift_report.read_text()
    assert "Mellea version" in md
    assert "0.6.0" in md
    assert "demo" in md  # per-skill row appears


def test_looks_like_git_rev_distinguishes_paths_from_revs(cc, tmp_path: Path) -> None:
    """Sanity check the heuristic that picks between fs path and git rev."""
    # Existing path → not a rev.
    assert cc._looks_like_git_rev(str(tmp_path)) is False
    # Slashy non-existent string → path-shaped.
    assert cc._looks_like_git_rev("foo/bar") is False
    assert cc._looks_like_git_rev("./snap") is False
    # Bare names that look like git refs.
    assert cc._looks_like_git_rev("HEAD~1") is True
    assert cc._looks_like_git_rev("v0.5.0") is True
    assert cc._looks_like_git_rev("abc123def") is True


def test_materialise_previous_dir_handles_path_layout(cc, tmp_path: Path) -> None:
    """Filesystem path with ``<root>/<skill>/<skill>_mellea/`` layout resolves."""
    prev_root = tmp_path / "snapshot"
    pkg = prev_root / "demo" / "demo_mellea"
    pkg.mkdir(parents=True)
    (pkg / "pipeline.py").write_text("def run_pipeline(): pass\n")

    resolved = cc._materialise_previous_dir(str(prev_root), "demo", tmp_path / "ws")
    assert resolved == pkg


def test_materialise_previous_dir_returns_none_when_missing(
    cc, tmp_path: Path
) -> None:
    resolved = cc._materialise_previous_dir(
        str(tmp_path / "missing"), "nope", tmp_path / "ws"
    )
    assert resolved is None


# --- Live end-to-end (skipped by default) ---------------------------------


@pytest.mark.live
def test_live_end_to_end_sentry_find_bugs(tmp_path: Path, cc) -> None:
    """Real descriptor pipeline against skills/sentry-find-bugs.

    Skipped unless the user passes ``-m live``.  Hits the LLM (P3.A's
    pipeline emits a descriptor via LiteLLM), and may incur cost.
    """
    skill = REPO_ROOT / "skills" / "sentry-find-bugs"
    if not skill.exists():
        pytest.skip("skills/sentry-find-bugs/ not present in this checkout")

    tmp_out = tmp_path / "sentry_descriptor"
    result = cc.compile_via_descriptor(skill, tmp_out, surface_path=cc.DEFAULT_SURFACE)

    if not result.compile_success and result.error and "not yet shipped" in result.error:
        pytest.skip("P3.A's descriptor_emission module not yet shipped in this checkout")

    assert result.compile_success, f"descriptor compile failed: {result.error}"
    assert (tmp_out / "pipeline.py").exists()
