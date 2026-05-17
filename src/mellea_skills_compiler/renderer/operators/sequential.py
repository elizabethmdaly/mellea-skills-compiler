"""``sequential`` composition operator (plan §4.5).

Sequential composition is normally **implicit**: when descriptor nodes appear
as array siblings (e.g. inside ``pipeline: [...]``) the renderer core walks
them in order. This operator handles the **explicit** form, used when the
descriptor wraps an array of nodes in a ``sequential`` composition node — for
example because the author wants to give that block a single ``id`` for
downstream reference (plan §4.5).

Required descriptor fields:
    body: list[node]  — child nodes in execution order.

Emitted shape::

    # render_body(body)[0]
    # render_body(body)[1]
    # ...

That is: the operator is a pass-through to :func:`render_body`. It anchors a
single source-map entry on the first emitted statement so the repair loop can
locate the operator in the descriptor.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Callable

from mellea_skills_compiler.renderer import OperatorRenderer
from mellea_skills_compiler.renderer.operators._support import (
    record_source,
    require_field,
)

if TYPE_CHECKING:
    from mellea_skills_compiler.renderer.core import RenderContext


class SequentialRenderer(OperatorRenderer):
    """Render an explicit ``sequential`` composition node."""

    operator_name = "sequential"

    def render(
        self,
        node: dict,
        ctx: "RenderContext",
        render_body: Callable[[list[dict]], list[ast.stmt]],
    ) -> list[ast.stmt]:
        body = require_field(node, "body", expected_type=list)
        stmts = render_body(body)  # type: ignore[arg-type]
        if stmts:
            record_source(ctx, node, stmts[0])
        return stmts
