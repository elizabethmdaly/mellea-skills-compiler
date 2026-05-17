"""Nested composition operators.

These fixtures stress the recursive ``render_body`` callback handed to each
operator's ``render`` method: a ``map`` inside a ``branch`` case body must
emit a for-loop nested inside an if/else, and a ``map`` inside a ``parallel``
branch must emit a for-loop inside the per-branch nested function.
"""

from __future__ import annotations


# --- nested_map_in_branch -----------------------------------------------------


def test_nested_map_in_branch_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("nested_map_in_branch")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_nested_map_in_branch_has_for_inside_if(render_fixture) -> None:
    """A ``map`` case body must produce a for-loop nested inside the if."""
    pkg = render_fixture("nested_map_in_branch")
    src = pkg.pipeline_src
    lines = src.splitlines()

    if_idx = next(
        (i for i, ln in enumerate(lines) if "if triage.kind == 'batch':" in ln or 'if triage.kind == "batch":' in ln),
        None,
    )
    assert if_idx is not None, "expected `if triage.kind == 'batch':` line"

    # Look for an indented for-loop after the if.
    indent = len(lines[if_idx]) - len(lines[if_idx].lstrip())
    found_inner_for = False
    for ln in lines[if_idx + 1 :]:
        if ln.strip() == "":
            continue
        cur_indent = len(ln) - len(ln.lstrip())
        if cur_indent <= indent:
            break
        if ln.strip().startswith("for ") and ln.strip().endswith(":"):
            found_inner_for = True
            break
    assert found_inner_for, "expected a for-loop nested inside the `batch` case body"


def test_nested_map_in_branch_binds_dispatch_to_map_accumulator(render_fixture) -> None:
    """The branch operator should bind `dispatch` to the map's accumulator id
    (`per_item`) in the batch case, and to the inner call id in the default
    case."""
    pkg = render_fixture("nested_map_in_branch")
    src = pkg.pipeline_src
    assert "dispatch = per_item" in src
    assert "dispatch = describe_single" in src


# --- parallel_with_inner_map --------------------------------------------------


def test_parallel_with_inner_map_renders_and_imports(render_fixture) -> None:
    pkg = render_fixture("parallel_with_inner_map")
    assert callable(getattr(pkg.pipeline_module, "run_pipeline", None))


def test_parallel_with_inner_map_emits_both_shapes(render_fixture) -> None:
    pkg = render_fixture("parallel_with_inner_map")
    src = pkg.pipeline_src
    assert "ThreadPoolExecutor" in src
    # Map inside branch 0 must produce a for-loop somewhere.
    assert "for item in " in src
    # Map accumulator init.
    assert "per_item_summary = []" in src


def test_parallel_with_inner_map_per_item_inside_branch_fn(render_fixture) -> None:
    """The map must appear inside the per-branch nested function (`_fanout_branch_0`),
    not at module/pipeline top level."""
    pkg = render_fixture("parallel_with_inner_map")
    src = pkg.pipeline_src
    fn_idx = src.find("def _fanout_branch_0")
    assert fn_idx != -1, "expected `_fanout_branch_0` nested function in parallel emit"
    fn_end_idx = src.find("def _fanout_branch_1")
    assert fn_end_idx != -1
    branch_0_body = src[fn_idx:fn_end_idx]
    assert "for item in " in branch_0_body, (
        "expected the map's for-loop to live inside the first parallel branch"
    )
