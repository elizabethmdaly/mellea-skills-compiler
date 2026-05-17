"""Each adversarial fixture exercises one composition operator in isolation.

The tests assert both that the rendered output ``py_compile``s and imports
under the dummy backend, AND that the operator-specific syntactic shape is
present in the rendered pipeline.py — e.g. ``for ... in ...`` for ``map``,
``ThreadPoolExecutor`` for ``parallel``, an ``if ... elif ... else`` chain
for ``branch``.
"""

from __future__ import annotations

import inspect


# --- only_sequential ----------------------------------------------------------


def test_only_sequential_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_sequential")
    fn = getattr(pkg.pipeline_module, "run_pipeline", None)
    assert callable(fn)


def test_only_sequential_has_no_composition_shapes(render_fixture) -> None:
    """No map/branch/parallel/retry — just sequential call assignments."""
    pkg = render_fixture("only_sequential")
    src = pkg.pipeline_src
    assert "ThreadPoolExecutor" not in src
    for_lines = [
        ln for ln in src.splitlines()
        if ln.strip().startswith("for ") and ln.strip().endswith(":")
    ]
    assert not for_lines, f"unexpected for-loop in sequential-only output: {for_lines}"


def test_only_sequential_emits_call_assignments(render_fixture) -> None:
    pkg = render_fixture("only_sequential")
    src = pkg.pipeline_src
    # Both call nodes' ids must appear as the LHS of an assignment.
    assert "first_call = " in src
    assert "second_call = " in src


def test_only_sequential_signature(render_fixture) -> None:
    pkg = render_fixture("only_sequential")
    sig = inspect.signature(pkg.pipeline_module.run_pipeline)
    assert list(sig.parameters) == ["query"]
    assert sig.return_annotation == "Echo"


# --- only_map_list ------------------------------------------------------------


def test_only_map_list_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_map_list")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_only_map_list_emits_for_loop_and_append(render_fixture) -> None:
    """`map` + `collect: list` emits accumulator init + for-loop + append."""
    pkg = render_fixture("only_map_list")
    src = pkg.pipeline_src
    assert "per_item_reports = []" in src
    assert "for item in " in src
    assert "per_item_reports.append(" in src


def test_only_map_list_does_not_emit_dict(render_fixture) -> None:
    pkg = render_fixture("only_map_list")
    src = pkg.pipeline_src
    assert "per_item_reports = {}" not in src


# --- only_map_dict_by_key -----------------------------------------------------


def test_only_map_dict_by_key_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_map_dict_by_key")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_only_map_dict_by_key_emits_dict_accumulator(render_fixture) -> None:
    pkg = render_fixture("only_map_dict_by_key")
    src = pkg.pipeline_src
    assert "reports_by_name = {}" in src
    # Per the operator: `reports_by_name[<last_body_id>.<key_field>] = <last_body_id>`.
    assert "reports_by_name[report_step.name] = report_step" in src


# --- only_branch --------------------------------------------------------------


def test_only_branch_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_branch")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_only_branch_emits_if_elif_else_chain(render_fixture) -> None:
    """branch with two cases + _default should emit if/elif/else."""
    pkg = render_fixture("only_branch")
    src = pkg.pipeline_src
    # The renderer expresses elif as an if inside else; ast.unparse emits "elif".
    has_if = "if triage.kind == 'high':" in src or 'if triage.kind == "high":' in src
    assert has_if, "expected `if triage.kind == \"high\":` style dispatch"
    has_elif = "elif triage.kind == 'low':" in src or 'elif triage.kind == "low":' in src
    assert has_elif, "expected `elif triage.kind == \"low\":` second case"
    # _default should produce an `else:` block.
    assert "else:" in src


def test_only_branch_binds_dispatch_in_each_case(render_fixture) -> None:
    """The branch's id (`dispatch`) must be bound to each case's last node id."""
    pkg = render_fixture("only_branch")
    src = pkg.pipeline_src
    assert "dispatch = escalate" in src
    assert "dispatch = ack" in src
    assert "dispatch = fallback" in src


# --- only_parallel ------------------------------------------------------------


def test_only_parallel_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_parallel")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_only_parallel_emits_thread_pool_executor(render_fixture) -> None:
    pkg = render_fixture("only_parallel")
    src = pkg.pipeline_src
    assert "ThreadPoolExecutor" in src
    assert "concurrent.futures" in src
    # Two branch functions.
    assert "_fanout_branch_0" in src
    assert "_fanout_branch_1" in src


def test_only_parallel_submits_and_collects(render_fixture) -> None:
    pkg = render_fixture("only_parallel")
    src = pkg.pipeline_src
    assert ".submit(" in src
    assert ".result()" in src


# --- only_retry_with_feedback -------------------------------------------------


def test_only_retry_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("only_retry_with_feedback")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_only_retry_emits_for_attempt_and_for_else(render_fixture) -> None:
    pkg = render_fixture("only_retry_with_feedback")
    src = pkg.pipeline_src
    assert "for _attempt in range(3):" in src
    # for/else: a top-level `else:` whose body is `raise RuntimeError(...)`
    assert "RuntimeError" in src
    assert "retried_answer = draft_answer" in src


def test_only_retry_check_uses_failure_check_ref(render_fixture) -> None:
    pkg = render_fixture("only_retry_with_feedback")
    src = pkg.pipeline_src
    # `if draft_answer.is_valid:` followed by `break`
    assert "if draft_answer.is_valid:" in src
    assert "break" in src
