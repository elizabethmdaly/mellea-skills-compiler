"""Tests for the per-lint repair templates (F1 wave 1).

Each of the four templates is exercised with a ``LintFailure`` built
verbatim from a real overnight-batch failure JSON, plus the negative
cases (unknown lint → None, wrong-lint-id dispatch → None).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mellea_skills_compiler.compile.lints import LintFailure
from mellea_skills_compiler.compile.repair_templates import (
    _extract_callee_from_arity_message,
    _extract_thunk_var_from_kb1_message,
    _lookup_canonical_signature,
    get_repair_template,
)


# ─── Real failures pulled from overnight_results/20260517-012930/ ───────


# From: eu-ai-act-classification-werner-plutat
RUNTIME_DEFAULTS_FAILURE = LintFailure(
    file="config.py",
    line=None,
    column=None,
    message=(
        "config.py not found in compiled package — this file is mandatory "
        "and is normally produced by the deterministic writer "
        "`.claude/melleafy/writers/config_writer.py` (Step 3, ENFORCE mode). "
        "Its absence usually means the writer-renderer never ran (see "
        "compile/writer_renderer.py + the writers_repo_root resolution in "
        "compile/mellea_skills.py). Was the spec compiled out-of-tree with "
        "an older compiler that walked up from the package dir instead of "
        "the installed compiler package?"
    ),
    rule_ref="C8 runtime defaults",
)


# From: gdpr-breach-sentinel-oliver-schmidt-prietz
BUNDLED_ASSET_FAILURE = LintFailure(
    file="loader.py",
    line=17,
    column=11,
    message=(
        "Bundled asset path 'references' is resolved via '_PKG_DIR'. "
        "Bundled assets at <package_name>/references/ MUST be resolved via "
        "`Path(__file__).parent / \"references/<...>\"` (Rule OUT-6 in "
        "mellea-fy.md, Rule 2.5-2 in mellea-fy-deps.md). Common error: "
        "`Path(repo_root) / \"references/...\"` — must be "
        "`Path(__file__).parent / \"references\" / ...`."
    ),
    rule_ref="Rule OUT-6, Rule 2.5-2",
)


# From: gdpr-privacy-notice-eu-oliver-schmidt-prietz
STDLIB_ARITY_FAILURE = LintFailure(
    file="requirements.py",
    line=91,
    column=4,
    message=(
        "`check(...)` called with 1 positional argument(s); signature "
        "requires between 2 and 2 positional arg(s). Consult intermediate/"
        "mellea_api_ref.json:.modules."
    ),
    rule_ref="Rule 4-4 (stdlib-arity)",
)


# From: nil-contract-analysis-samir-patel
INSTRUCT_PARSE_FAILURE = LintFailure(
    file="pipeline.py",
    line=251,
    column=18,
    message=(
        "`thunk` is the result of `m.instruct(..., format=Model)` and is a "
        "`ComputedModelOutputThunk`, NOT a Pydantic model. Direct attribute "
        "access `thunk.value` raises AttributeError at runtime (KB1). Parse "
        "the thunk first: `parsed = _safe_parse_with_fallback(thunk, "
        "<Model>, **defaults)` (preferred), `parsed = _parse_instruct_result("
        "thunk, <Model>)`, or `<Model>.model_validate_json(thunk.value)`."
    ),
    rule_ref="KB1 (instruct returns thunk)",
)


# A small surface JSON sample shaped like the real
# intermediate/mellea_api_ref.json — used by the stdlib-arity grounding test.
SAMPLE_SURFACE_JSON = {
    "modules": {
        "mellea.stdlib.requirements": {
            "check": {
                "signature": (
                    "check(*args, **kwargs) -> "
                    "mellea.core.requirement.Requirement"
                )
            },
            "req": {
                "signature": (
                    "req(description: str, "
                    "validation_fn: Callable | None = None, *, "
                    "output_to_bool: Callable | None = None) -> "
                    "mellea.core.requirement.Requirement"
                )
            },
        }
    }
}


# ─── Template renderings against real failures ─────────────────────────


def test_runtime_defaults_bound_template_renders_with_real_failure():
    out = get_repair_template(
        "runtime-defaults-bound", RUNTIME_DEFAULTS_FAILURE
    )
    assert out is not None
    assert "runtime-defaults-bound" in out
    assert "config.py" in out
    assert "intermediate/runtime_directive.json" in out
    assert "BACKEND" in out and "MODEL_ID" in out
    # Failure message should be interpolated verbatim.
    assert "config.py not found in compiled package" in out


def test_bundled_asset_path_resolution_template_renders_with_real_failure():
    out = get_repair_template(
        "bundled-asset-path-resolution", BUNDLED_ASSET_FAILURE
    )
    assert out is not None
    assert "bundled-asset-path-resolution" in out
    # File and line interpolated from the failure
    assert "loader.py" in out
    assert "Line: 17" in out
    # Prescription must teach the Path(__file__).parent pattern
    assert "Path(__file__).parent" in out
    assert "Rule OUT-6" in out
    # Counter-pattern is called out so Claude doesn't re-introduce it
    assert "_PKG_DIR" in out or "repo_root" in out


def test_stdlib_arity_template_renders_with_real_failure():
    out = get_repair_template(
        "stdlib-arity",
        STDLIB_ARITY_FAILURE,
        surface_json=SAMPLE_SURFACE_JSON,
    )
    assert out is not None
    assert "stdlib-arity" in out
    assert "requirements.py" in out
    assert "Line: 91" in out
    # Extracted callee
    assert "`check`" in out
    # Canonical signature is interpolated from the surface JSON
    assert "check(*args, **kwargs)" in out
    assert "mellea_api_ref.json" in out


def test_instruct_result_parse_before_access_template_renders_with_real_failure():
    out = get_repair_template(
        "instruct-result-parse-before-access", INSTRUCT_PARSE_FAILURE
    )
    assert out is not None
    assert "instruct-result-parse-before-access" in out
    assert "pipeline.py" in out
    assert "Line: 251" in out
    # Prescription must teach the parse pattern
    assert "model_validate_json" in out
    assert "_safe_parse_with_fallback" in out
    # The thunk var name from the failure message is used
    assert "thunk" in out
    assert "KB1" in out


# ─── Dispatch / unknown-lint behaviour ──────────────────────────────────


def test_get_repair_template_returns_none_for_unknown_lint():
    assert (
        get_repair_template("totally-made-up-lint", RUNTIME_DEFAULTS_FAILURE)
        is None
    )


def test_runtime_defaults_bound_dispatch_only_for_its_id():
    # Calling get_repair_template with a different lint_id does NOT render
    # the runtime-defaults template — it dispatches per lint_id.
    out = get_repair_template("some-other-lint", RUNTIME_DEFAULTS_FAILURE)
    assert out is None


def test_bundled_asset_dispatch_only_for_its_id():
    out = get_repair_template("not-bundled-asset", BUNDLED_ASSET_FAILURE)
    assert out is None


def test_stdlib_arity_dispatch_only_for_its_id():
    out = get_repair_template("not-stdlib-arity", STDLIB_ARITY_FAILURE)
    assert out is None


def test_instruct_parse_dispatch_only_for_its_id():
    out = get_repair_template("not-instruct-parse", INSTRUCT_PARSE_FAILURE)
    assert out is None


# ─── Surface-JSON integration for stdlib-arity ──────────────────────────


def test_stdlib_arity_template_uses_surface_json_kwarg():
    """When surface_json is passed explicitly, the canonical signature
    string from it appears verbatim in the rendered template."""
    out = get_repair_template(
        "stdlib-arity",
        STDLIB_ARITY_FAILURE,
        surface_json=SAMPLE_SURFACE_JSON,
    )
    assert out is not None
    # The exact signature string from SAMPLE_SURFACE_JSON
    assert (
        "check(*args, **kwargs) -> mellea.core.requirement.Requirement"
        in out
    )


def test_stdlib_arity_template_uses_package_dir_surface_json(tmp_path: Path):
    """Without surface_json kwarg, the template loads
    intermediate/mellea_api_ref.json from package_dir."""
    pkg = tmp_path / "pkg"
    (pkg / "intermediate").mkdir(parents=True)
    (pkg / "intermediate" / "mellea_api_ref.json").write_text(
        json.dumps(SAMPLE_SURFACE_JSON)
    )
    out = get_repair_template(
        "stdlib-arity",
        STDLIB_ARITY_FAILURE,
        package_dir=pkg,
    )
    assert out is not None
    assert "check(*args, **kwargs)" in out


def test_stdlib_arity_template_falls_back_when_no_surface_json():
    """No surface JSON, no package_dir — template still renders, but the
    canonical-signature block is the fallback prose."""
    out = get_repair_template("stdlib-arity", STDLIB_ARITY_FAILURE)
    assert out is not None
    assert "could not be resolved" in out
    # Symbol extraction still works even without surface JSON
    assert "`check`" in out


def test_stdlib_arity_template_handles_grounding_unavailable(tmp_path: Path):
    """If the surface file says grounding_unavailable, fall back gracefully."""
    pkg = tmp_path / "pkg"
    (pkg / "intermediate").mkdir(parents=True)
    (pkg / "intermediate" / "mellea_api_ref.json").write_text(
        json.dumps({"grounding_unavailable": True})
    )
    out = get_repair_template(
        "stdlib-arity",
        STDLIB_ARITY_FAILURE,
        package_dir=pkg,
    )
    assert out is not None
    assert "could not be resolved" in out


# ─── Helper unit tests ──────────────────────────────────────────────────


def test_extract_callee_from_arity_message():
    assert (
        _extract_callee_from_arity_message(
            "`check(...)` called with 1 positional argument(s); ..."
        )
        == "check"
    )
    assert (
        _extract_callee_from_arity_message("no backticks here") is None
    )


def test_extract_thunk_var_from_kb1_message():
    assert (
        _extract_thunk_var_from_kb1_message(
            "`thunk` is the result of `m.instruct(..., format=Model)`"
        )
        == "thunk"
    )
    assert (
        _extract_thunk_var_from_kb1_message(
            "`result_xyz` is the result of `m.instruct(...)`"
        )
        == "result_xyz"
    )


def test_lookup_canonical_signature_finds_symbol():
    assert (
        _lookup_canonical_signature(SAMPLE_SURFACE_JSON, "check")
        == "check(*args, **kwargs) -> mellea.core.requirement.Requirement"
    )


def test_lookup_canonical_signature_returns_none_for_missing():
    assert _lookup_canonical_signature(SAMPLE_SURFACE_JSON, "nonexistent") is None
    assert _lookup_canonical_signature(None, "check") is None
    assert _lookup_canonical_signature({}, "check") is None
