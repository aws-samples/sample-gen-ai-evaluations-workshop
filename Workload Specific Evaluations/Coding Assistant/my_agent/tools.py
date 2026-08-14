"""Tool definitions — YOU FILL THESE IN.

Every tool you give the model must call `trace.tool(name, input)` on
invocation. Without that, notebook 07 cannot score tool-call quality
(precision@k, sequence-aware checks, etc.) for your agent.

The eval expects tool *names* to be stable. The names below match the
ones used in the prebuilt task set's `expected_tools.required` lists:

  - `read_file`      — open a file by path (and optionally line range)
  - `edit_file`      — apply an edit to a file by path
  - `run_grep`       — pattern search across the repo
  - `run_tests`      — invoke `pytest` (optional but useful on T07_*)
  - `find_callers`   — code-graph "who calls X?" (MCP server)
  - `find_dependencies` — code-graph "what does X depend on?" (MCP server)

You don't have to implement every tool. Start with read/edit/grep —
that's enough for ~6 of the 9 tasks. Add `find_callers` /
`find_dependencies` when you start working on T08 (nav-only) and the
hard tasks that benefit from structural search.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .trace import Trace


def build_toolset(repo_path: Path, trace: Trace) -> List[Any]:
    """Return a list of tools to register with your agent.

    YOU IMPLEMENT THIS. Below is a sketch using Strands' `@tool`
    decorator. Adapt to whichever framework you chose in `model.py`.

    Sketch:

        from strands import tool

        @tool
        def read_file(path: str, start_line: int = 1, end_line: int = -1) -> str:
            \"\"\"Read a file (or a range of lines).\"\"\"
            trace.tool("read_file", {"path": path, "start_line": start_line, "end_line": end_line})
            full_path = repo_path / path
            content = full_path.read_text().splitlines()
            if end_line < 0:
                end_line = len(content)
            return "\\n".join(content[start_line - 1:end_line])

        @tool
        def edit_file(path: str, old_string: str, new_string: str) -> str:
            \"\"\"Replace `old_string` with `new_string` in `path`. old_string must match exactly once.\"\"\"
            trace.tool("edit_file", {"path": path})
            full_path = repo_path / path
            text = full_path.read_text()
            count = text.count(old_string)
            if count != 1:
                return f"FAILED: old_string occurs {count} times in {path}; needs to be unique"
            full_path.write_text(text.replace(old_string, new_string, 1))
            return f"OK: edited {path}"

        # ... grep, run_tests, find_callers, find_dependencies ...

        return [read_file, edit_file]

    For the MCP-backed tools (`find_callers`, `find_dependencies`), you
    can either:
      (a) bring up the code-graph MCP server in a sidecar and call it,
          OR
      (b) implement them directly with `subprocess` / AST parsing.
    Notebook 05 walks through option (a).
    """
    raise NotImplementedError(
        "build_toolset is a stub. Open my_agent/tools.py and implement at "
        "minimum read_file / edit_file / run_grep."
    )
