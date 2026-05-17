"""Composition operator plug-ins (plan §4.5).

Importing this package self-registers every operator with the renderer
registry; downstream code can then dispatch by name via
:func:`mellea_skills_compiler.renderer.get_operator`.

The 5 curated operators (plan §4.5 — extending the set requires an RFC):

- ``sequential`` — explicit pass-through of an ordered body.
- ``map`` — iterate a body over a list ref; collect to list or dict.
- ``branch`` — if/elif/else dispatch on a ref value.
- ``parallel`` — concurrent execution of multiple branches.
- ``retry_with_feedback`` — bounded retry with success/failure check.

Added in descriptor v0.3 (RFC §3.3, plan §11.2):

- ``agent_loop`` — bounded iterative agent with typed state threading.
- ``human_approval`` — prompt-and-branch approval gate with callback override.
"""

from __future__ import annotations

from mellea_skills_compiler.renderer import register_operator
from mellea_skills_compiler.renderer.operators.agent_loop import AgentLoopRenderer
from mellea_skills_compiler.renderer.operators.branch import BranchRenderer
from mellea_skills_compiler.renderer.operators.human_approval import (
    HumanApprovalRenderer,
)
from mellea_skills_compiler.renderer.operators.map_op import MapRenderer
from mellea_skills_compiler.renderer.operators.parallel import ParallelRenderer
from mellea_skills_compiler.renderer.operators.retry import RetryRenderer
from mellea_skills_compiler.renderer.operators.sequential import SequentialRenderer

# Instantiate once; registry is idempotent on same-instance re-registration.
_SEQUENTIAL = SequentialRenderer()
_MAP = MapRenderer()
_BRANCH = BranchRenderer()
_PARALLEL = ParallelRenderer()
_RETRY = RetryRenderer()
_AGENT_LOOP = AgentLoopRenderer()
_HUMAN_APPROVAL = HumanApprovalRenderer()

register_operator(_SEQUENTIAL)
register_operator(_MAP)
register_operator(_BRANCH)
register_operator(_PARALLEL)
register_operator(_RETRY)
register_operator(_AGENT_LOOP)
register_operator(_HUMAN_APPROVAL)


__all__ = [
    "AgentLoopRenderer",
    "BranchRenderer",
    "HumanApprovalRenderer",
    "MapRenderer",
    "ParallelRenderer",
    "RetryRenderer",
    "SequentialRenderer",
]
