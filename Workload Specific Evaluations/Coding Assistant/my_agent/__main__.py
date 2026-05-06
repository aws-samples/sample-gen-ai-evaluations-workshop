"""CLI entrypoint that the eval harness invokes.

The eval calls:

    python -m my_agent \
        --task-id <id> --tasks-file <path> --repo <path> \
        --out <diff_path> --trace-out <trace_path> [--seed <int>]

This file does the plumbing — argument parsing, task loading, diff
capture, trace serialization. You should NOT need to modify it.

Edit `agent.py` to implement the agent loop. Edit `model.py` and
`tools.py` to wire up the model and tool definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .agent import CodingAgent
from .trace import Trace


def _load_task(tasks_file: Path, task_id: str) -> Dict[str, Any]:
    doc = yaml.safe_load(tasks_file.read_text())
    for t in doc.get("tasks", []):
        if t.get("id") == task_id:
            return t
    raise SystemExit(f"task id {task_id!r} not found in {tasks_file}")


def _capture_diff(repo: Path) -> str:
    """Return a unified diff of unstaged + untracked changes in `repo`.

    Mirrors `utils.workspace.capture_diff` so the harness can run the
    agent in the workspace and pull the diff out the same way the eval
    runners do.
    """
    tracked = subprocess.run(
        ["git", "diff", "--no-color"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout

    untracked_names = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    if not untracked_names:
        return tracked

    add = subprocess.run(
        ["git", "add", "-N", "--", *untracked_names],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    if add.returncode == 0:
        full = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
        subprocess.run(
            ["git", "reset", "HEAD", "--", *untracked_names],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        return full
    return tracked


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser("my_agent")
    p.add_argument("--task-id", required=True)
    p.add_argument("--tasks-file", required=True, type=Path)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trace-out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.seed:
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    task = _load_task(args.tasks_file, args.task_id)

    trace = Trace()
    agent = CodingAgent(repo_path=args.repo, trace=trace, seed=args.seed)

    try:
        result = agent.run(task)
    except Exception as e:
        # Capture whatever diff exists at this point — the agent may have
        # made partial edits — and surface the error to stderr.
        diff = _capture_diff(args.repo)
        args.out.write_text(diff)
        args.trace_out.write_text(json.dumps(trace.to_list(), indent=2))
        print(f"agent error: {e!r}", file=sys.stderr)
        return 1

    diff = _capture_diff(args.repo)
    args.out.write_text(diff)
    args.trace_out.write_text(json.dumps(trace.to_list(), indent=2))

    # Exit code 0 = task complete, 2 = not complete (e.g. trap detected,
    # nav-only task, or agent gave up cleanly). Anything else = error.
    return 0 if result.completed else 2


if __name__ == "__main__":
    sys.exit(main())
