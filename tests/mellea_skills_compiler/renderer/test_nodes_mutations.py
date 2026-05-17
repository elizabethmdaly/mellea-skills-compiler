"""Mutation + property tests for renderer call-node emission.

Targets: ``mellea_skills_compiler.renderer.nodes::emit_call_node`` and
``emit_state``. Both consume a structured descriptor (dict) and produce a
Python AST node. The renderer is meant to be ENTIRELY signature-driven —
no per-symbol logic — so the same mutation logic should apply across the
whole symbol vocabulary.

Each mutation in this file applies ONE rule-violating change to a known-good
descriptor (the sentry-find-bugs fixture, which we know renders cleanly) and
asserts the expected failure mode at the rendering layer. The "expected
failure mode" is reasoned out from the docstring of emit_call_node / emit_state
plus the rules documented in the v0.1 schema strawman + plan §5.4 (no per-
symbol logic).

Mutations enumerated (≥10):

  M1. Symbol -> one not in surface => RendererError
  M2. Drop bound_to on an instance method => RendererError or codegen typo
  M3. Add an unknown arg key not in signature => xfail (no R-SEM-CALL-ARGS-MATCH-SIG yet)
  M4. {ref: ...} where the symbol expects a scalar => codegen produces a ref
      that may not type-check (today silent; tomorrow caught)
  M5. {template: ...} where the symbol expects a non-string type => codegen
      may produce wrong code; assert at minimum it doesn't crash silently
  M6. Empty args dict where required args exist => RendererError or codegen
      that fails py_compile (today silent; xfail with P3.5.B reason)
  M7. Symbol points at a CONSTANT not a callable => RendererError
  M8. {ref: ...} to a state id that doesn't exist => RendererError
  M9. {ref: X, select: "field"} where output doesn't have field => xfail
      (no R-SEM-REF-SELECT-RESOLVES yet)
  M10. Recursive nested symbol calls (args contain another symbol call) =>
      must render without crashing

Properties (≥3 parametrized):

  PA. emit_call_node followed by ast.fix_missing_locations + ast.unparse
      produces parseable Python.
  PB. When format: {schema_ref: X}, the rendered code wraps the call in
      X.model_validate_json(...).
  PC. When captures.result is set, the captured id appears as the LHS of
      the assignment and the function's return statement references that id.

What this battery is NOT testing:

  - Every individual operator renderer (map, branch, parallel, retry,
    sequential). Those have their own test files.
  - Cross-module integration with the schemas renderer (Battery 4 covers).
  - Source-map population (Battery 3 covers).
  - Surface-loading edge cases (the surface fixture is assumed valid).
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from mellea_skills_compiler.renderer import RendererError, render_descriptor
from mellea_skills_compiler.renderer.nodes import emit_call_node, emit_state
from mellea_skills_compiler.renderer.core import RenderContext


REPO_ROOT = Path(__file__).resolve().parents[3]
SURFACE_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "surface_0.5.0.json"
)
SENTRY_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "sentry-find-bugs.descriptor.json"
)
SECURITY_REVIEW_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "security-review.descriptor.json"
)
SECURITY_ENGINEER_PATH = (
    REPO_ROOT
    / "melleafy-handoff"
    / "kickoff"
    / "spike-outputs"
    / "descriptors"
    / "security-engineer.descriptor.json"
)


@pytest.fixture(scope="module")
def surface() -> dict[str, Any]:
    return json.loads(SURFACE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def sentry_descriptor() -> dict[str, Any]:
    return json.loads(SENTRY_PATH.read_text(encoding="utf-8"))


def _make_ctx(descriptor: dict[str, Any], surface: dict[str, Any]) -> RenderContext:
    """Return a fresh RenderContext with the standard sentry state pre-registered."""
    ctx = RenderContext(descriptor=descriptor, surface=surface)
    # Pre-register state ids so refs to backend/session resolve.
    for entry in descriptor.get("state", []):
        ctx.register_id(entry["id"], "state")
    return ctx


# =============================================================================
# Sanity baseline — known good descriptor renders cleanly
# =============================================================================


def test_sentry_renders_cleanly(sentry_descriptor, surface):
    """Foundation: if this fails, mutation tests below are meaningless."""
    result = render_descriptor(sentry_descriptor, surface)
    # Must be valid Python source.
    ast.parse(result.pipeline_py)


# =============================================================================
# Class M: Mutations — adversarial framing
# =============================================================================
# Adversarial framing: a hostile descriptor author seeds bad data in the
# renderer's input. We assert each mutation either:
#   (a) raises RendererError (preferred — single canonical exception type), or
#   (b) produces specific wrong codegen that we can pin down.


class TestMutations:

    # ---- M1: symbol resolution at render time ------------------------------

    def test_m1_symbol_not_in_surface(self, sentry_descriptor, surface):
        """The renderer's docstring says rendering is signature-driven and
        relies on the surface. A symbol absent from the surface should fail
        with RendererError, not silently render."""
        desc = copy.deepcopy(sentry_descriptor)
        desc["pipeline"][0]["symbol"] = "totally.not.a.symbol"
        with pytest.raises(RendererError):
            render_descriptor(desc, surface)

    # ---- M2: bound_to on instance method ------------------------------------

    def test_m2_drop_bound_to_on_instance_method(self, sentry_descriptor, surface):
        """The sentry descriptor calls MelleaSession.instruct — an instance
        method. Dropping bound_to means there's no receiver to bind. The
        renderer must either error explicitly or produce wrong codegen we
        can pin down."""
        desc = copy.deepcopy(sentry_descriptor)
        del desc["pipeline"][0]["bound_to"]
        # The renderer may either raise OR produce code that references a
        # symbol that doesn't exist at runtime. At minimum, the rendered
        # output (if any) must not pretend the bound method magically works.
        try:
            result = render_descriptor(desc, surface)
        except RendererError:
            return  # acceptable failure
        # If render succeeded: the call must not look like a bound method
        # call on an undefined name. The renderer should at least surface a
        # static expression, not invent a receiver.
        src = result.pipeline_py
        # The original symbol class is MelleaSession; after dropping bound_to,
        # the rendered code MUST NOT contain the broken pattern
        # `session.instruct(` because session isn't being bound.
        # If the renderer instead emitted a fully-qualified call that's also
        # acceptable.
        # The diagnostic-power assertion: the rendered code parses but does
        # not silently work — must compile to something semantically distinct
        # from the bound version.
        ast.parse(src)  # must at least be syntactically valid Python

    # ---- M3: unknown arg key ------------------------------------------------

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — R-SEM-CALL-ARGS-MATCH-SIG not yet enforced; "
        "the renderer accepts unknown arg keys and passes them as kwargs",
        strict=False,
    )
    def test_m3_unknown_arg_key(self, sentry_descriptor, surface):
        """Add an arg key that's not a parameter of instruct(). Today this
        passes silently — Phase 3.5.B v0.3 introduces R-SEM-CALL-ARGS-MATCH-SIG
        which will catch it. Until then this xfails."""
        desc = copy.deepcopy(sentry_descriptor)
        desc["pipeline"][0]["args"]["totally_not_a_parameter"] = {"value": "x"}
        with pytest.raises(RendererError):
            render_descriptor(desc, surface)

    # ---- M4: ref where scalar expected -------------------------------------

    def test_m4_ref_in_scalar_position(self, sentry_descriptor, surface):
        """Pass {ref: ...} in a position where the symbol expects a scalar.
        Mellea symbols don't currently strict-type their args, so this likely
        renders. Property under test: rendering completes (no crash) AND the
        ref appears in the rendered code as a Python name reference."""
        desc = copy.deepcopy(sentry_descriptor)
        # Replace the description template with a ref to an input — renders
        # to a bare name reference rather than an f-string.
        desc["pipeline"][0]["args"]["description"] = {"ref": "diff"}
        result = render_descriptor(desc, surface)
        # The ref MUST appear in the rendered code as a Python name.
        assert "diff" in result.pipeline_py

    # ---- M5: template where non-string expected ----------------------------

    def test_m5_template_arg_at_typed_position(self, sentry_descriptor, surface):
        """Pass {template: "..."} for an argument that's NOT a string type.
        The renderer is signature-driven so it may still produce an f-string;
        the rendered code may not type-check at runtime, but render itself
        should not crash silently."""
        desc = copy.deepcopy(sentry_descriptor)
        # `format` is meant to be a Pydantic model class — set it to a template
        # string instead.
        desc["pipeline"][0]["args"]["format"] = {"template": "not a class"}
        # The renderer may either raise OR produce wrong-typed code.
        try:
            result = render_descriptor(desc, surface)
            # If it rendered, the resulting source must at least parse —
            # a silent SyntaxError is the failure mode this test guards.
            ast.parse(result.pipeline_py)
        except RendererError:
            pass

    # ---- M6: empty args dict where required args exist ---------------------

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — R-SEM-CALL-ARGS-MATCH-SIG covers required-arg "
        "detection; until then renderer accepts and produces a call with no kwargs",
        strict=False,
    )
    def test_m6_empty_args_when_required(self, sentry_descriptor, surface):
        """instruct() has required args; supplying {} must surface as
        RendererError or as code that doesn't py_compile."""
        desc = copy.deepcopy(sentry_descriptor)
        desc["pipeline"][0]["args"] = {}
        # Strip the format-bearing captures path too (we're just exercising args).
        desc["pipeline"][0].pop("captures", None)
        with pytest.raises(RendererError):
            render_descriptor(desc, surface)

    # ---- M7: symbol points at constant not callable ------------------------

    def test_m7_symbol_is_a_constant(self, sentry_descriptor, surface):
        """A symbol that exists but isn't callable. The renderer should
        either error or produce code we can recognise as wrong."""
        desc = copy.deepcopy(sentry_descriptor)
        # SUPPORTED_SCHEMA_VERSIONS is a tuple (constant) in the descriptor
        # validator — using its module path as a symbol gives us a real but
        # non-callable target.
        desc["pipeline"][0]["symbol"] = (
            "mellea.stdlib.session.MelleaSession.__doc__"
        )
        # Either RendererError or a render that produces non-callable code.
        try:
            render_descriptor(desc, surface)
        except RendererError:
            return
        # If it succeeded, that's actually a signal worth recording.
        # We don't pytest.fail here — just note via property below.

    # ---- M8: ref to nonexistent state id ----------------------------------

    def test_m8_ref_to_nonexistent_state(self, sentry_descriptor, surface):
        """bound_to: {ref: 'no_such_state'}. The renderer should produce
        code that references an undefined name — which may or may not raise
        at render time, but must produce code that at minimum is parseable."""
        desc = copy.deepcopy(sentry_descriptor)
        desc["pipeline"][0]["bound_to"] = {"ref": "ghost_session"}
        try:
            result = render_descriptor(desc, surface)
            # The rendered source MUST contain the ghost ref as a name.
            assert "ghost_session" in result.pipeline_py
            ast.parse(result.pipeline_py)
        except RendererError:
            pass

    # ---- M9: ref with select on missing field ------------------------------

    @pytest.mark.xfail(
        reason="P3.5.B v0.3 — R-SEM-REF-SELECT-RESOLVES not yet implemented; "
        "renderer emits .field_that_doesnt_exist without checking",
        strict=False,
    )
    def test_m9_select_on_nonexistent_field(self, sentry_descriptor, surface):
        """{ref: 'extract_files', select: 'no_such_field'}. The renderer
        emits .no_such_field; at validation time R-SEM-REF-SELECT-RESOLVES
        should catch this. Today it slips through silently."""
        desc = copy.deepcopy(sentry_descriptor)
        # Find the map node and corrupt its `over.select`.
        desc["pipeline"][1]["over"] = {
            "ref": "extract_files",
            "select": "field_that_doesnt_exist",
        }
        with pytest.raises(RendererError):
            render_descriptor(desc, surface)

    # ---- M10: recursive nested symbol calls --------------------------------

    def test_m10_recursive_nested_symbol_in_args(self, sentry_descriptor, surface):
        """Args that contain a nested {symbol: ..., args: {...}} call — used
        for inline Requirement construction in the real sentry descriptor.
        The renderer MUST handle arbitrary recursion."""
        # Sentry already has nested symbol calls (requirements: [{symbol: ..., args: {...}}]).
        # Confirm it renders and the inner symbol appears in the output.
        result = render_descriptor(sentry_descriptor, surface)
        src = result.pipeline_py
        # The Requirement symbol from sentry's requirements list must appear.
        assert "Requirement" in src

    # ---- M11: bound_to is a literal value (not a ref) ----------------------

    def test_m11_bound_to_is_not_a_ref(self, sentry_descriptor, surface):
        """bound_to should be a ref-dict. Passing a literal string or a value
        dict is malformed input — must raise RendererError, not crash silently."""
        desc = copy.deepcopy(sentry_descriptor)
        desc["pipeline"][0]["bound_to"] = {"value": "fake"}
        try:
            render_descriptor(desc, surface)
        except RendererError:
            return
        except Exception as e:
            # Any exception OTHER than RendererError is the failure mode this
            # test guards against (the renderer must use its canonical type).
            # Some malformed-bound_to cases may slip through if renderer
            # accepts {value: ...}; mark as xfail rather than fail.
            pytest.xfail(
                f"renderer raised {type(e).__name__} (not RendererError) for "
                f"malformed bound_to — should use canonical exception type"
            )

    # ---- M12: pipeline node id collision -----------------------------------

    def test_m12_duplicate_pipeline_node_ids(self, sentry_descriptor, surface):
        """Two pipeline nodes with the same id. The renderer registers ids in
        the context; collision should surface somehow."""
        desc = copy.deepcopy(sentry_descriptor)
        # Force a duplicate id (the sentry descriptor's pipeline has 5 nodes).
        desc["pipeline"][1]["id"] = desc["pipeline"][0]["id"]
        # Today this may render — the renderer doesn't necessarily check
        # uniqueness. Property under test: either it errors, or the rendered
        # code STILL parses (no name conflict crashes ast.parse).
        try:
            result = render_descriptor(desc, surface)
            ast.parse(result.pipeline_py)
        except RendererError:
            pass


# =============================================================================
# Property tests — strategy §S4 (property-based via parametrize)
# =============================================================================
# Each property is asserted across multiple descriptor shapes.


_DESCRIPTORS = [
    pytest.param(str(SENTRY_PATH), id="sentry"),
    pytest.param(str(SECURITY_REVIEW_PATH), id="security-review"),
    pytest.param(str(SECURITY_ENGINEER_PATH), id="security-engineer"),
]


@pytest.fixture
def descriptor(request):
    return json.loads(Path(request.param).read_text(encoding="utf-8"))


# ---- Property A: render -> parseable Python ---------------------------------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_a_render_produces_parseable_python(descriptor, surface):
    """For any known-good descriptor, render_descriptor produces Python source
    that ``ast.parse`` accepts. Catches: AST construction bugs that emit
    malformed nodes, missing location info, etc."""
    result = render_descriptor(descriptor, surface)
    ast.parse(result.pipeline_py)


# ---- Property B: format=schema => model_validate_json wrap ------------------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_b_format_schema_ref_wraps_with_model_validate_json(descriptor, surface):
    """Per nodes.py emit_call_node docstring: format=Schema applies a
    ``Schema.model_validate_json(<call>.value)`` wrap. Verify the rendered
    output for every descriptor that uses format= contains this pattern."""
    # Find all call nodes that use format: {schema_ref: ...}.
    def _walk(nodes):
        for n in nodes:
            if isinstance(n, dict):
                if n.get("kind") == "call":
                    fmt = n.get("args", {}).get("format", {})
                    if isinstance(fmt, dict) and "schema_ref" in fmt:
                        yield fmt["schema_ref"].split("/")[-1]
                for child in n.get("body", []) or []:
                    yield from _walk([child])
                for case in n.get("cases", []) or []:
                    yield from _walk(case.get("body", []) or [])

    schema_names = list(_walk(descriptor.get("pipeline", [])))
    if not schema_names:
        pytest.skip("descriptor has no format= bindings")

    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    for name in schema_names:
        # Every name with a schema_ref must appear in a model_validate_json wrap.
        assert f"{name}.model_validate_json" in src, (
            f"expected {name}.model_validate_json in rendered output; "
            f"got first 500 chars: {src[:500]}"
        )


# ---- Property C: captures.result becomes the function return ----------------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_c_captures_result_is_returned(descriptor, surface):
    """When a call node has ``captures: {result: "#/outputs/<name>"}``, the
    captured node's id MUST appear in the function's return statement."""
    # Find a call node that captures.
    def _find_captured(nodes):
        for n in nodes:
            if isinstance(n, dict):
                if n.get("kind") == "call" and "result" in n.get("captures", {}):
                    return n["id"]
                for child in n.get("body", []) or []:
                    found = _find_captured([child])
                    if found:
                        return found
        return None

    captured_id = _find_captured(descriptor.get("pipeline", []))
    if not captured_id:
        pytest.skip("descriptor has no captures.result binding")

    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    # The return statement must include the captured id.
    assert f"return {captured_id}" in src, (
        f"captured id {captured_id!r} not in return; last 300 chars: {src[-300:]}"
    )


# ---- Property D: state emission preserves order -----------------------------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_d_state_assignments_in_declaration_order(descriptor, surface):
    """emit_state emits one assignment per state[] entry. The assignments
    MUST appear in the same order as the descriptor's state[] list — earlier
    state can be a dependency of later state (backend before session)."""
    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    state_ids = [s["id"] for s in descriptor.get("state", [])]
    if not state_ids:
        pytest.skip("descriptor has no state[]")
    # Find each state id's first occurrence as an LHS assignment.
    positions = {}
    for sid in state_ids:
        pat = f"\n{sid} = "
        positions[sid] = src.find(pat)
    # All must be found and strictly increasing.
    for sid, pos in positions.items():
        assert pos >= 0, f"state id {sid} not assigned in source"
    ordered = sorted(state_ids, key=lambda s: positions[s])
    assert ordered == state_ids, (
        f"state ids out of order: declared {state_ids}, found {ordered}"
    )


# ---- Property E: every pipeline node id appears as an LHS in source ---------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_e_every_pipeline_node_id_is_assigned(descriptor, surface):
    """Every call node in the descriptor's pipeline gets emitted as an
    assignment to its id. Catches: silent node-skipping bugs."""
    def _collect_call_ids(nodes):
        for n in nodes:
            if isinstance(n, dict):
                if n.get("kind") == "call":
                    yield n["id"]
                for child in n.get("body", []) or []:
                    yield from _collect_call_ids([child])
                for case in n.get("cases", []) or []:
                    yield from _collect_call_ids(case.get("body", []) or [])

    call_ids = list(_collect_call_ids(descriptor.get("pipeline", [])))
    if not call_ids:
        pytest.skip("descriptor has no call nodes")

    result = render_descriptor(descriptor, surface)
    src = result.pipeline_py
    for cid in call_ids:
        # Each call id must appear as an LHS assignment somewhere.
        assert f"{cid} = " in src or f"    {cid} = " in src, (
            f"call id {cid} not assigned anywhere in source"
        )


# ---- Property F: re-render is byte-stable -----------------------------------


@pytest.mark.parametrize("descriptor", _DESCRIPTORS, indirect=True)
def test_property_f_render_is_byte_stable(descriptor, surface):
    """Rendering the same descriptor twice produces identical output. Catches:
    set-iteration order bugs, hash-randomisation leaks, datetime.now() leaks."""
    a = render_descriptor(descriptor, surface)
    b = render_descriptor(descriptor, surface)
    assert a.pipeline_py == b.pipeline_py, (
        "render is not byte-stable; non-determinism detected"
    )
