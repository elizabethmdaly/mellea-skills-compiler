"""Tests for ``scripts/overnight_e2e_compile.py``.

The runner is exercised end-to-end at the module level; no real ``claude``
subprocess is ever spawned (the production runner is mocked via the
``runner`` test seam on :func:`run_one_skill`).

Coverage:

* Corpus enumeration across all three spec shapes (file, workspace,
  workspace-fallback) plus the "parent dir of skills" shape.
* Failure classifier: synthetic log strings → correct class.
* Resume logic: previous summary.json consumed correctly; partial
  in-progress.txt invalidates the stale result.
* Summary aggregator: synthetic SkillResult rows → correct counts and
  by-class buckets.
* Timeout handling: a runner stub that simulates a hung subprocess →
  status=timeout, classified as ``timeout``.
* Digest rendering: top-line numbers + failure groupings.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "overnight_e2e_compile.py"


@pytest.fixture(scope="module")
def oe() -> ModuleType:
    """Load ``scripts/overnight_e2e_compile.py`` as a module.

    The script is dual-purpose (CLI + importable lib); we drive it as a
    module here to exercise the helpers directly without spawning a real
    subprocess.
    """
    spec = importlib.util.spec_from_file_location("overnight_e2e_compile", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["overnight_e2e_compile"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _make_workspace(root: Path, name: str, spec_name: str = "SKILL.md") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / spec_name).write_text(f"# {name}\n")
    return d


def test_discover_single_md_file(tmp_path: Path, oe) -> None:
    md = tmp_path / "lone.md"
    md.write_text("# lone\n")
    targets = oe.discover_targets([md])
    assert len(targets) == 1
    assert targets[0].kind == "file"
    assert targets[0].skill_id == "lone"
    assert targets[0].spec_path == md


def test_discover_workspace_with_skill_md(tmp_path: Path, oe) -> None:
    ws = _make_workspace(tmp_path, "my-skill", "SKILL.md")
    targets = oe.discover_targets([ws])
    assert len(targets) == 1
    t = targets[0]
    assert t.kind == "workspace"
    assert t.skill_id == "my-skill"
    assert t.spec_path == ws


def test_discover_workspace_with_spec_md(tmp_path: Path, oe) -> None:
    ws = _make_workspace(tmp_path, "demo", "spec.md")
    targets = oe.discover_targets([ws])
    assert targets[0].kind == "workspace"
    assert targets[0].skill_id == "demo"


def test_discover_workspace_fallback_picks_any_md(tmp_path: Path, oe) -> None:
    ws = tmp_path / "weird"
    ws.mkdir()
    (ws / "zeta.md").write_text("# z")
    (ws / "alpha.md").write_text("# a")
    targets = oe.discover_targets([ws])
    assert len(targets) == 1
    assert targets[0].kind == "workspace-fallback"
    # alpha.md should win (alphabetical) — but we mostly care that
    # discovery succeeds and the kind is correct.
    assert targets[0].resolved_file.name == "alpha.md"


def test_discover_recurses_one_level_for_parent_dirs(tmp_path: Path, oe) -> None:
    # Parent dir without a top-level spec — children ARE skills.
    _make_workspace(tmp_path, "skill-a", "SKILL.md")
    _make_workspace(tmp_path, "skill-b", "spec.md")
    targets = oe.discover_targets([tmp_path])
    ids = {t.skill_id for t in targets}
    assert ids == {"skill-a", "skill-b"}


def test_discover_skips_mellea_output_dirs(tmp_path: Path, oe) -> None:
    # ``parent/`` contains one skill + one already-compiled output dir.
    _make_workspace(tmp_path, "skill-real", "spec.md")
    compiled = tmp_path / "skill-real_mellea"
    compiled.mkdir()
    (compiled / "spec.md").write_text("# compiled — should be ignored")
    targets = oe.discover_targets([tmp_path])
    assert {t.skill_id for t in targets} == {"skill-real"}


def test_discover_raises_on_missing_path(tmp_path: Path, oe) -> None:
    with pytest.raises(FileNotFoundError):
        oe.discover_targets([tmp_path / "no-such-thing"])


def test_discover_dedupes_overlapping_corpus_paths(tmp_path: Path, oe) -> None:
    ws = _make_workspace(tmp_path, "shared", "SKILL.md")
    # The skill dir + its parent both point at the same workspace.
    targets = oe.discover_targets([ws, tmp_path])
    ids = [t.skill_id for t in targets]
    assert ids == ["shared"]  # de-duplicated


# ---------------------------------------------------------------------------
# Failure classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("log_snippet", "expected"),
    [
        ("Something exploded\n... R-SEM-CONTRACT-MISSING: field foo ...", "validator_error"),
        ("traceback: validation failure on slot 'x'", "validator_error"),
        ("RendererError: could not lower template", "renderer_error"),
        ("lower_template raised at line 42", "renderer_error"),
        ("1-shot emission produced invalid IR", "emission_error"),
        ("descriptor emission failed: schema mismatch", "emission_error"),
        ("claude exited non-zero (code 137)", "claude_subprocess_error"),
        ("Anthropic API returned 429 Too Many Requests", "claude_subprocess_error"),
        ("401 Unauthorized: bad API key", "claude_subprocess_error"),
        ("some random python traceback, no signature here", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_failure_known_signatures(oe, log_snippet: str, expected: str) -> None:
    assert oe.classify_failure(log_snippet) == expected


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def test_load_resume_results_reads_summary_json(tmp_path: Path, oe) -> None:
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "summary.json").write_text(
        json.dumps(
            {
                "skills": [
                    {"skill_id": "ok-one", "status": oe.STATUS_SUCCESS},
                    {"skill_id": "failed", "status": oe.STATUS_CRASHED},
                    {"skill_id": "ok-two", "status": oe.STATUS_SUCCESS},
                ]
            }
        )
    )
    out = oe.load_resume_results(prev)
    assert set(out.keys()) == {"ok-one", "ok-two"}


def test_load_resume_results_handles_missing_dir(tmp_path: Path, oe) -> None:
    assert oe.load_resume_results(tmp_path / "missing") == {}
    assert oe.load_resume_results(None) == {}


def test_load_resume_results_falls_back_to_per_skill_results(tmp_path: Path, oe) -> None:
    prev = tmp_path / "prev"
    (prev / "ok-one").mkdir(parents=True)
    (prev / "ok-one" / "result.json").write_text(
        json.dumps({"skill_id": "ok-one", "status": oe.STATUS_SUCCESS})
    )
    (prev / "failed").mkdir()
    (prev / "failed" / "result.json").write_text(
        json.dumps({"skill_id": "failed", "status": oe.STATUS_CRASHED})
    )
    out = oe.load_resume_results(prev)
    assert set(out.keys()) == {"ok-one"}


def test_existing_success_ids_picks_up_local_successes(tmp_path: Path, oe) -> None:
    out_dir = tmp_path / "out"
    (out_dir / "alpha").mkdir(parents=True)
    (out_dir / "alpha" / "result.json").write_text(
        json.dumps({"skill_id": "alpha", "status": oe.STATUS_SUCCESS})
    )
    (out_dir / "beta").mkdir()
    (out_dir / "beta" / "result.json").write_text(
        json.dumps({"skill_id": "beta", "status": oe.STATUS_TIMEOUT})
    )
    assert oe._existing_success_ids(out_dir) == {"alpha"}


# ---------------------------------------------------------------------------
# Summary aggregator
# ---------------------------------------------------------------------------


def _result(oe, skill_id: str, status: str, **kwargs):
    defaults = {
        "skill_id": skill_id,
        "spec_path": f"/tmp/{skill_id}",
        "kind": "workspace",
        "status": status,
        "duration_seconds": 1.0,
    }
    defaults.update(kwargs)
    return oe.SkillResult(**defaults)


def test_aggregate_summary_counts_by_status(tmp_path: Path, oe) -> None:
    rows = [
        _result(oe, "a", oe.STATUS_SUCCESS),
        _result(oe, "b", oe.STATUS_SUCCESS),
        _result(oe, "c", oe.STATUS_TIMEOUT, error_class="timeout"),
        _result(oe, "d", oe.STATUS_CRASHED, error_class="validator_error"),
        _result(oe, "e", oe.STATUS_CRASHED, error_class="validator_error"),
        _result(oe, "f", oe.STATUS_CRASHED, error_class="renderer_error"),
        _result(oe, "g", oe.STATUS_SKIPPED),
    ]
    s = oe.aggregate_summary(rows, output_dir=tmp_path, started_at="2026-05-17T00:00:00")
    assert s["total"] == 7
    assert s["counts"][oe.STATUS_SUCCESS] == 2
    assert s["counts"][oe.STATUS_TIMEOUT] == 1
    assert s["counts"][oe.STATUS_CRASHED] == 3
    assert s["counts"][oe.STATUS_SKIPPED] == 1
    assert s["by_error_class"] == {
        "timeout": 1,
        "validator_error": 2,
        "renderer_error": 1,
    }
    assert s["success_rate"] == pytest.approx(2 / 7, abs=1e-3)


def test_aggregate_summary_handles_empty_input(tmp_path: Path, oe) -> None:
    s = oe.aggregate_summary([], output_dir=tmp_path, started_at="2026-05-17T00:00:00")
    assert s["total"] == 0
    assert s["success_rate"] == 0.0
    assert s["counts"][oe.STATUS_SUCCESS] == 0


# ---------------------------------------------------------------------------
# Digest rendering
# ---------------------------------------------------------------------------


def test_render_digest_includes_topline_and_failure_table(tmp_path: Path, oe) -> None:
    rows = [
        _result(oe, "ok-a", oe.STATUS_SUCCESS, duration_seconds=200.0),
        _result(oe, "ok-slow", oe.STATUS_SUCCESS, duration_seconds=500.0),
        _result(
            oe,
            "fail-val",
            oe.STATUS_CRASHED,
            error_class="validator_error",
            error_summary="R-SEM-FOO: bad",
            exit_code=1,
        ),
        _result(oe, "fail-time", oe.STATUS_TIMEOUT, error_class="timeout", exit_code=-9),
    ]
    s = oe.aggregate_summary(rows, output_dir=tmp_path, started_at="2026-05-17T00:00:00")
    digest = oe.render_digest(s)
    assert "Overnight compile digest" in digest
    assert "Total skills:    **4**" in digest
    # Top-5 slowest successes should include the slower one first.
    assert "`ok-slow`" in digest
    # Failure-class table.
    assert "validator_error" in digest
    assert "timeout" in digest
    # Suggested actions for timeout + validator_error.
    assert "validator error" in digest.lower()


# ---------------------------------------------------------------------------
# Timeout handling (runner seam)
# ---------------------------------------------------------------------------


def _make_target(oe, tmp_path: Path, skill_id: str = "hung") -> object:
    ws = tmp_path / skill_id
    ws.mkdir()
    (ws / "spec.md").write_text("# hung\n")
    from mellea_skills_compiler.compile.spec_path_resolver import resolve_spec_path

    resolved = resolve_spec_path(ws)
    return oe._target_from(resolved, ws)


def test_run_one_skill_records_timeout(tmp_path: Path, oe) -> None:
    target = _make_target(oe, tmp_path, "hung")
    out = tmp_path / "out" / target.skill_id
    out.mkdir(parents=True)

    def hung_runner(argv, log_path, timeout):
        # Simulate: process logged something, then was killed.
        log_path.write_text("starting compile ... [hangs forever]\n")
        return -9, float(timeout), True  # exit_code, duration, timed_out=True

    result = oe.run_one_skill(
        target, out, timeout=30, runner=hung_runner
    )
    assert result.status == oe.STATUS_TIMEOUT
    assert result.error_class == "timeout"
    assert result.duration_seconds == 30.0
    assert (out / "compile.log").exists()


def test_run_one_skill_records_crash_with_classified_error(tmp_path: Path, oe) -> None:
    target = _make_target(oe, tmp_path, "bad")
    out = tmp_path / "out" / target.skill_id
    out.mkdir(parents=True)

    def crashed_runner(argv, log_path, timeout):
        log_path.write_text("RendererError: lower_template blew up on slot foo\n")
        return 1, 12.3, False

    result = oe.run_one_skill(target, out, timeout=600, runner=crashed_runner)
    assert result.status == oe.STATUS_CRASHED
    assert result.error_class == "renderer_error"
    assert result.exit_code == 1
    assert "RendererError" in (result.error_summary or "")


def test_run_one_skill_records_success_and_finds_artefacts(tmp_path: Path, oe) -> None:
    target = _make_target(oe, tmp_path, "good")
    out = tmp_path / "out" / target.skill_id
    out.mkdir(parents=True)

    # Drop a fake compiled package next to the spec so _find_artefacts picks it up.
    artefact = Path(target.spec_path) / "good_mellea"
    artefact.mkdir()
    (artefact / "pipeline.py").write_text("# fake\n")

    def good_runner(argv, log_path, timeout):
        log_path.write_text("compile ok\n")
        return 0, 30.5, False

    result = oe.run_one_skill(target, out, timeout=600, runner=good_runner)
    assert result.status == oe.STATUS_SUCCESS
    assert result.exit_code == 0
    assert result.artefact_paths == [str(artefact.resolve())]


# ---------------------------------------------------------------------------
# Full driver via main() (with mocked runner)
# ---------------------------------------------------------------------------


def test_main_dry_run_lists_targets(tmp_path: Path, oe, capsys) -> None:
    _make_workspace(tmp_path, "skill-x", "SKILL.md")
    _make_workspace(tmp_path, "skill-y", "spec.md")
    rc = oe.main(
        [
            "--corpus",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "skill-x" in captured.out
    assert "skill-y" in captured.out
    assert "Total: 2 skill" in captured.out


def test_main_errors_on_bad_corpus(tmp_path: Path, oe, capsys) -> None:
    rc = oe.main(
        [
            "--corpus",
            str(tmp_path / "nope"),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    # Logging is INFO+; the actual error appears in the batch log. We
    # just need to confirm the non-zero exit.
    assert rc == 2


def test_main_skips_when_resume_marks_success(tmp_path: Path, oe, monkeypatch) -> None:
    """End-to-end: a skill marked success in --resume-from is skipped."""
    _make_workspace(tmp_path, "alpha", "SKILL.md")
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "summary.json").write_text(
        json.dumps(
            {
                "skills": [
                    {"skill_id": "alpha", "status": oe.STATUS_SUCCESS},
                ]
            }
        )
    )

    invocations: list[list[str]] = []

    def fake_runner(argv, log_path, timeout):
        invocations.append(argv)
        log_path.write_text("should not be called\n")
        return 0, 1.0, False

    # Monkey-patch the default runner so even if the skip logic failed,
    # we'd notice via ``invocations``.
    monkeypatch.setattr(oe, "_default_runner", fake_runner)

    rc = oe.main(
        [
            "--corpus",
            str(tmp_path / "alpha"),
            "--output-dir",
            str(tmp_path / "out"),
            "--resume-from",
            str(prev),
        ]
    )
    assert rc == 0
    assert invocations == []  # never invoked
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["counts"][oe.STATUS_SKIPPED] == 1
    assert (tmp_path / "out" / "digest.md").exists()


def test_main_writes_incremental_summary_after_each_skill(
    tmp_path: Path, oe, monkeypatch
) -> None:
    _make_workspace(tmp_path, "one", "SKILL.md")
    _make_workspace(tmp_path, "two", "SKILL.md")

    written_snapshots: list[int] = []

    def runner(argv, log_path, timeout):
        log_path.write_text("ok\n")
        # After each skill, summary.json should already exist.
        s_path = tmp_path / "out" / "summary.json"
        if s_path.exists():
            data = json.loads(s_path.read_text())
            written_snapshots.append(data["total"])
        return 0, 0.5, False

    monkeypatch.setattr(oe, "_default_runner", runner)
    rc = oe.main(
        [
            "--corpus",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    # The second call should see the first skill already in summary.json.
    assert any(t >= 1 for t in written_snapshots)


def test_main_classifies_failures_and_groups_in_digest(
    tmp_path: Path, oe, monkeypatch
) -> None:
    _make_workspace(tmp_path, "vfail", "SKILL.md")
    _make_workspace(tmp_path, "rfail", "SKILL.md")

    def runner(argv, log_path, timeout):
        # Distinguish the two skills by their spec path.
        joined = " ".join(argv)
        if "vfail" in joined:
            log_path.write_text("traceback: R-SEM-BAD: not enough slots\n")
        else:
            log_path.write_text("RendererError: lower_template raised\n")
        return 1, 5.0, False

    monkeypatch.setattr(oe, "_default_runner", runner)
    rc = oe.main(
        [
            "--corpus",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["by_error_class"] == {"renderer_error": 1, "validator_error": 1}
    digest = (tmp_path / "out" / "digest.md").read_text()
    assert "renderer_error" in digest
    assert "validator_error" in digest


def test_main_invalidates_stale_in_progress(tmp_path: Path, oe, monkeypatch) -> None:
    """An in_progress.txt left from a prior crash forces a re-run."""
    _make_workspace(tmp_path, "halfdone", "SKILL.md")
    out = tmp_path / "out"
    (out / "halfdone").mkdir(parents=True)
    # Plant a stale success result + the in-progress marker.
    (out / "halfdone" / "result.json").write_text(
        json.dumps(
            {
                "skill_id": "halfdone",
                "spec_path": "x",
                "kind": "workspace",
                "status": oe.STATUS_SUCCESS,
                "duration_seconds": 0.0,
            }
        )
    )
    (out / "in_progress.txt").write_text("halfdone")

    runs: list[str] = []

    def runner(argv, log_path, timeout):
        runs.append("called")
        log_path.write_text("ok\n")
        return 0, 1.0, False

    monkeypatch.setattr(oe, "_default_runner", runner)
    rc = oe.main(
        [
            "--corpus",
            str(tmp_path / "halfdone"),
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    # Stale success result should have been discarded → compile invoked.
    assert runs == ["called"]


# ---------------------------------------------------------------------------
# _build_compile_argv — pass-through of retry-layer flags
# ---------------------------------------------------------------------------


def test_overnight_runner_passes_both_flags(tmp_path: Path, oe) -> None:
    """The orchestrator must propagate BOTH retry-layer flags when both
    are requested. Confirms the two flags compose at the orchestrator
    boundary (the two inner-CLI retry layers are independent and
    additive — see compile/mellea_skills.py)."""
    spec = tmp_path / "weather" / "spec.md"
    spec.parent.mkdir()
    spec.write_text("# weather\n")

    argv = oe._build_compile_argv(
        spec,
        python_exe="/usr/bin/python3",
        inner_timeout=600,
        resume_on_early_end=True,
        repair_on_lint_failure=True,
    )

    assert "--resume-on-early-end" in argv
    assert "--repair-on-lint-failure" in argv
    # Sanity: the spec path is the final positional arg style preserved.
    assert str(spec) in argv


def test_overnight_runner_omits_flags_by_default(tmp_path: Path, oe) -> None:
    """Default (neither flag) → neither flag string appears in the argv."""
    spec = tmp_path / "weather" / "spec.md"
    spec.parent.mkdir()
    spec.write_text("# weather\n")

    argv = oe._build_compile_argv(
        spec,
        python_exe="/usr/bin/python3",
    )

    assert "--resume-on-early-end" not in argv
    assert "--repair-on-lint-failure" not in argv
