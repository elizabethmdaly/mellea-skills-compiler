"""Source-map repair attribution: rendered Python lines -> descriptor node ids.

The Phase 2.A SourceMap is the bridge that lets the repair loop (§7.5) point
at a misbehaving descriptor node from a Python traceback. These tests verify
that round-trip on real fixtures.
"""

from __future__ import annotations

import pytest


def _walk_pipeline_node_ids(nodes):
    """Yield every node id that should appear in the source map.

    Walks composition recursively; includes both the composition's own id
    and any ids inside its body / cases / branches.
    """
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if "id" in n:
            yield n["id"]
        if n.get("kind") == "composition":
            yield from _walk_pipeline_node_ids(n.get("body", []) or [])
            for case_body in (n.get("cases") or {}).values():
                yield from _walk_pipeline_node_ids(case_body or [])
            for branch in (n.get("branches") or []):
                yield from _walk_pipeline_node_ids(branch or [])


@pytest.mark.parametrize(
    "fixture_name",
    [
        "only_sequential",
        "only_map_list",
        "only_branch",
        "only_parallel",
        "only_retry_with_feedback",
        "nested_map_in_branch",
        "parallel_with_inner_map",
    ],
)
def test_every_pipeline_node_id_is_mappable(render_fixture, fixture_name) -> None:
    """At least one rendered line should reverse-map to every descriptor pipeline node."""
    pkg = render_fixture(fixture_name)
    src = pkg.pipeline_src
    source_map = pkg.render_result.source_map

    expected_ids = set(_walk_pipeline_node_ids(pkg.descriptor.get("pipeline", [])))
    assert expected_ids, "fixture has no pipeline nodes — broken fixture"

    # Collect mapped ids by walking every line of the rendered source.
    mapped_ids: set[str] = set()
    for line_no in range(1, len(src.splitlines()) + 1):
        nid = source_map.lookup(line_no)
        if nid is not None:
            mapped_ids.add(nid)

    missing = expected_ids - mapped_ids
    assert not missing, (
        f"fixture {fixture_name!r}: source map did not cover descriptor node ids {missing}"
        f"; mapped={mapped_ids}"
    )


def test_import_line_returns_none(render_fixture) -> None:
    """Lines that are not pipeline statements (e.g. ``from __future__ ...``) must
    not reverse-map to a descriptor node id."""
    pkg = render_fixture("only_sequential")
    src = pkg.pipeline_src
    source_map = pkg.render_result.source_map

    lines = src.splitlines()
    import_line_no = None
    for i, ln in enumerate(lines, start=1):
        if ln.startswith("from __future__ import annotations"):
            import_line_no = i
            break
    assert import_line_no is not None, "expected `from __future__ import annotations` at top"

    assert source_map.lookup(import_line_no) is None


def test_source_map_to_json_contains_call_node_ids(render_fixture) -> None:
    """JSON serialisation of the source map must include every call node id."""
    pkg = render_fixture("only_map_list")
    payload = pkg.render_result.source_map.to_json()
    assert isinstance(payload, list)
    assert all("descriptor_node_id" in e for e in payload)
    ids = {e["descriptor_node_id"] for e in payload}
    # The map node's body call id and the outer list_items call id must both be present.
    assert {"list_items", "report_step"} <= ids


def test_inner_node_id_wins_over_outer(render_fixture) -> None:
    """A line inside a map body that belongs to the inner call should map to
    the inner node id, not to the outer composition operator's id."""
    pkg = render_fixture("only_map_list")
    src = pkg.pipeline_src
    source_map = pkg.render_result.source_map

    target_line = None
    for i, ln in enumerate(src.splitlines(), start=1):
        if "report_step = " in ln and "model_validate_json" in ln:
            target_line = i
            break
    assert target_line is not None, "expected a `report_step = ...model_validate_json(...)` line"
    assert source_map.lookup(target_line) == "report_step"


def test_far_past_eof_returns_none(render_fixture) -> None:
    pkg = render_fixture("only_branch")
    assert pkg.render_result.source_map.lookup(1_000_000) is None
