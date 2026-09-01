"""Tool-call trace scoring.

Scores an agent's tool-use trace against the task's `expected_tools`
declaration:

  - `required`: tool names that SHOULD appear in the trace at least once.
  - `forbidden`: tool names that should NOT appear.

Plus a sequence-aware layer:
  - `required_before_edit`: required tool was called BEFORE the first
    edit/write. Calling `find_callers` after you already wrote the patch
    is theatre.
  - `edit_uses_query_result`: at least one symbol/path returned by a
    structural query (find_callers / find_dependencies / grep) appears
    in the inputs of a later edit/write call. Cheap heuristic that the
    query result was actually consumed.

All matching is substring-based on tool names so MCP-namespaced names
(`code_graph__find_callers`) still match a required entry of `find_callers`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


EDIT_TOOL_HINTS = ("write", "edit", "patch", "apply", "replace")
QUERY_TOOL_HINTS = ("find_callers", "find_dependencies", "grep", "search")


@dataclass
class TraceScore:
    agent: str
    task_id: str
    n_calls: int
    required_hit: List[str]
    required_missed: List[str]
    forbidden_hit: List[str]
    tool_counts: Dict[str, int] = field(default_factory=dict)
    required_before_edit: bool = True   # vacuously true if no required tools
    edit_uses_query_result: bool = True # vacuously true if no edits
    sequence_notes: List[str] = field(default_factory=list)

    @property
    def required_pass(self) -> bool:
        return not self.required_missed

    @property
    def forbidden_pass(self) -> bool:
        return not self.forbidden_hit

    @property
    def overall_pass(self) -> bool:
        return self.required_pass and self.forbidden_pass

    @property
    def sequence_pass(self) -> bool:
        return self.required_before_edit and self.edit_uses_query_result


def load_trace(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict) and "tool" in entry]


def _matches(expected: str, actual: str) -> bool:
    return expected in actual


def _is_edit(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in EDIT_TOOL_HINTS)


def _is_query(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in QUERY_TOOL_HINTS)


def _flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_flatten_strings(v))
    return out


def _query_result_tokens(entry: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of identifiers from a query tool's input or recorded output."""
    tokens: List[str] = []
    for blob in _flatten_strings(entry.get("input")):
        tokens.extend(t for t in blob.replace("/", " ").split() if len(t) > 2)
    for blob in _flatten_strings(entry.get("output")):
        tokens.extend(t for t in blob.replace("/", " ").split() if len(t) > 2)
    return tokens


def score_trace(
    trace: List[Dict[str, Any]],
    expected_tools: Dict[str, List[str]],
    agent: str,
    task_id: str,
) -> TraceScore:
    tool_counts: Dict[str, int] = {}
    for entry in trace:
        name = entry.get("tool") or "<unknown>"
        tool_counts[name] = tool_counts.get(name, 0) + 1

    required = list(expected_tools.get("required") or [])
    forbidden = list(expected_tools.get("forbidden") or [])

    required_hit: List[str] = []
    required_missed: List[str] = []
    for r in required:
        if any(_matches(r, name) for name in tool_counts):
            required_hit.append(r)
        else:
            required_missed.append(r)

    forbidden_hit: List[str] = []
    for f in forbidden:
        if any(_matches(f, name) for name in tool_counts):
            forbidden_hit.append(f)

    sequence_notes: List[str] = []

    first_edit_idx: int | None = None
    for i, entry in enumerate(trace):
        if _is_edit(entry.get("tool", "")):
            first_edit_idx = i
            break

    if required and first_edit_idx is not None:
        pre_edit_tools = {e.get("tool", "") for e in trace[:first_edit_idx]}
        required_before_edit = all(
            any(_matches(r, name) for name in pre_edit_tools) for r in required
        )
        if not required_before_edit:
            sequence_notes.append(
                "required tool(s) called only AFTER first edit — looks like post-hoc theatre"
            )
    else:
        required_before_edit = True

    edit_uses_query_result = True
    if first_edit_idx is not None:
        query_tokens: List[str] = []
        for entry in trace[:first_edit_idx]:
            if _is_query(entry.get("tool", "")):
                query_tokens.extend(_query_result_tokens(entry))
        query_tokens = [t for t in query_tokens if t]
        if query_tokens:
            consumed = False
            for entry in trace[first_edit_idx:]:
                if not _is_edit(entry.get("tool", "")):
                    continue
                edit_blob = " ".join(_flatten_strings(entry.get("input")))
                if any(tok in edit_blob for tok in query_tokens):
                    consumed = True
                    break
            edit_uses_query_result = consumed
            if not consumed:
                sequence_notes.append(
                    "structural query results never appear in subsequent edit inputs — possible ignored result"
                )

    return TraceScore(
        agent=agent,
        task_id=task_id,
        n_calls=len(trace),
        required_hit=required_hit,
        required_missed=required_missed,
        forbidden_hit=forbidden_hit,
        tool_counts=tool_counts,
        required_before_edit=required_before_edit,
        edit_uses_query_result=edit_uses_query_result,
        sequence_notes=sequence_notes,
    )


def score_to_dict(score: TraceScore) -> Dict[str, Any]:
    return asdict(score)
