"""Source map: rendered Python line -> originating descriptor-node id.

The renderer attaches a `descriptor_node_id` annotation to every AST statement
it emits. After `ast.unparse`, we walk the resulting Python source line-by-line
and project each statement's annotation onto the lines it occupies.

Used by §7.5 (the repair loop): when smoke-check produces a traceback, the
line number is mapped back to a descriptor node, which becomes the focal
point of the repair prompt.

Stored AST annotations are kept on a private attribute name so they do not
collide with anything the `ast` stdlib generates. Lookup is O(1) per query
once the map is built.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# Private attribute name used to tag AST statements with their originating
# descriptor-node id. Underscored to avoid collision with `ast.fix_missing_locations`.
DESCRIPTOR_NODE_ATTR = "_mellea_descriptor_node_id"


@dataclass
class SourceMapEntry:
    """One mapped statement: the descriptor node id and the Python line span."""

    descriptor_node_id: str
    start_line: int
    end_line: int
    descriptor_path: str = ""  # JSON Pointer-ish path, e.g. "pipeline[1].body[0]"

    def contains(self, line_no: int) -> bool:
        return self.start_line <= line_no <= self.end_line


@dataclass
class SourceMap:
    """Maps rendered-Python line numbers back to descriptor node ids.

    Build by tagging each emitted AST stmt with `tag_stmt(stmt, node_id)`,
    then calling `finalise(module, source)` once after `ast.unparse`.

    Lookup with `lookup(line_no)` -> descriptor_node_id or None.
    """

    entries: list[SourceMapEntry] = field(default_factory=list)
    # Cache of line -> entry for O(1) lookup after finalise.
    _line_index: dict[int, SourceMapEntry] = field(default_factory=dict)

    # --- Construction ------------------------------------------------------

    @staticmethod
    def tag_stmt(stmt: ast.stmt, descriptor_node_id: str, descriptor_path: str = "") -> ast.stmt:
        """Attach a descriptor-node id to a freshly-built AST statement.

        Returns the same stmt for convenience so callers can write
        `body.append(SourceMap.tag_stmt(ast.Assign(...), node_id))`.
        """
        setattr(stmt, DESCRIPTOR_NODE_ATTR, descriptor_node_id)
        if descriptor_path:
            setattr(stmt, DESCRIPTOR_NODE_ATTR + "_path", descriptor_path)
        return stmt

    @staticmethod
    def get_tag(stmt: ast.stmt) -> str | None:
        return getattr(stmt, DESCRIPTOR_NODE_ATTR, None)

    @staticmethod
    def get_path(stmt: ast.stmt) -> str:
        return getattr(stmt, DESCRIPTOR_NODE_ATTR + "_path", "")

    def add_entry(self, descriptor_node_id: str, start_line: int, end_line: int, descriptor_path: str = "") -> None:
        entry = SourceMapEntry(
            descriptor_node_id=descriptor_node_id,
            start_line=start_line,
            end_line=end_line,
            descriptor_path=descriptor_path,
        )
        self.entries.append(entry)
        for ln in range(start_line, end_line + 1):
            # Inner-most assignment wins (later tagged statement = nested in earlier).
            self._line_index[ln] = entry

    # --- Operator-facing API (used by P2.B's record_source helper) --------

    def record(self, *, stmt: ast.stmt | None = None, node_id: str = "", line: int = 0) -> None:
        """Tag a statement (or a line) with the originating descriptor-node id.

        Two call shapes are supported (mirrors operators/_support.record_source):

        - ``record(stmt=<ast.stmt>, node_id=<str>)`` — preferred. The tag is
          attached to the AST stmt as an attribute; `build_from_module` reads
          it back after `ast.fix_missing_locations` to compute the line span.
        - ``record(line=<int>, node_id=<str>)`` — degenerate fallback when the
          caller doesn't have the AST node any more. Records a single-line
          entry directly.

        Idempotent: re-tagging the same stmt with the same id is a no-op.
        """
        if not node_id:
            raise ValueError("source_map.record: node_id is required")
        if stmt is not None:
            SourceMap.tag_stmt(stmt, node_id)
            return
        if line > 0:
            self.add_entry(node_id, line, line)

    def build_from_module(self, module: ast.Module, source: str | None = None) -> None:
        """Project tagged statements onto rendered Python line numbers.

        The challenge: `ast.fix_missing_locations` fills missing positions
        from the nearest parent, so freshly-built AST nodes all share
        ``lineno=1``. We can't use those directly.

        Strategy: re-parse the unparsed source (produced by `ast.unparse`)
        and walk both the original `module` and the reparsed one in parallel.
        For each tagged statement in the original, copy the reparsed
        statement's real line range. This gives accurate line spans
        regardless of how ``fix_missing_locations`` filled them.

        Outer (largest-span) entries are inserted first so inner statements
        overwrite line-index slots — the deepest descriptor scope wins for
        any given line, which is what §7.5 wants.

        Args:
            module: The ast.Module that was just unparsed.
            source: The unparsed source string. If None, we compute it via
                ``ast.unparse(module)``; passing it in saves a re-unparse.
        """
        if source is None:
            source = ast.unparse(module)
        try:
            reparsed = ast.parse(source)
        except SyntaxError:
            # If the unparsed output can't reparse, we have a renderer bug —
            # don't mask it by silently skipping. Re-raise.
            raise

        collected: list[tuple[int, int, str, str]] = []
        self._walk_paired(module, reparsed, collected)

        # Outer (largest span) first so inner ones overwrite line-index slots.
        collected.sort(key=lambda r: -(r[1] - r[0]))
        for start, end, tag, path in collected:
            self.add_entry(tag, start, end, path)

    def _walk_paired(
        self,
        original: ast.AST,
        reparsed: ast.AST,
        collected: list[tuple[int, int, str, str]],
    ) -> None:
        """Walk two ASTs in lockstep, copying linenos from reparsed onto our tags."""
        # Pair body lists from both nodes that have body attrs.
        for body_attr in ("body", "orelse", "finalbody", "handlers"):
            orig_body = getattr(original, body_attr, None)
            repd_body = getattr(reparsed, body_attr, None)
            if not isinstance(orig_body, list) or not isinstance(repd_body, list):
                continue
            if len(orig_body) != len(repd_body):
                # Structural mismatch — skip this body. The renderer aims to
                # keep unparse+reparse round-trippable; if a future operator
                # emits something that doesn't round-trip exactly, we don't
                # crash, we just lose precision on its source-map entries.
                continue
            for orig_stmt, repd_stmt in zip(orig_body, repd_body):
                if isinstance(orig_stmt, ast.stmt) and isinstance(repd_stmt, ast.stmt):
                    tag = self.get_tag(orig_stmt)
                    if tag is not None:
                        start = getattr(repd_stmt, "lineno", None)
                        end = getattr(repd_stmt, "end_lineno", start)
                        if start is not None and end is not None:
                            collected.append((start, end, tag, self.get_path(orig_stmt)))
                    self._walk_paired(orig_stmt, repd_stmt, collected)

    # --- Query --------------------------------------------------------------

    def lookup(self, line_no: int) -> str | None:
        """Return the descriptor node id that produced this Python line, or None."""
        entry = self._line_index.get(line_no)
        return entry.descriptor_node_id if entry else None

    def lookup_entry(self, line_no: int) -> SourceMapEntry | None:
        return self._line_index.get(line_no)

    def to_json(self) -> list[dict]:
        """Serialisable form: list of {descriptor_node_id, start_line, end_line, descriptor_path}."""
        return [
            {
                "descriptor_node_id": e.descriptor_node_id,
                "start_line": e.start_line,
                "end_line": e.end_line,
                "descriptor_path": e.descriptor_path,
            }
            for e in self.entries
        ]
