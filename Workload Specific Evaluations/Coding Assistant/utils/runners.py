"""Shell-out wrappers for the three coding agents under test.

All three runners take the same signature:

    run_<agent>(task: dict, workspace: Workspace, **kwargs) -> AgentOutput

so the eval loop treats them identically. Each agent's tool-call trace is
captured and scored separately downstream.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspace import Workspace, capture_diff


@dataclass
class AgentOutput:
    agent: str
    task_id: str
    diff: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    elapsed_s: float = 0.0
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    seed: int = 0
    input_tokens: Optional[int] = None   # custom agent only; None for off-the-shelf CLIs
    output_tokens: Optional[int] = None


def _task_to_prompt(task: Dict[str, Any]) -> str:
    paths_block = "\n".join("- " + p for p in task.get("affected_paths", []))
    return f"""You are working in a fresh clone of the target repo.

Implement the following task end-to-end. Use the repo's own test runner
to verify your changes. Do not introduce new top-level dependencies.
Keep changes minimal and scoped to the task.

# Task: {task['title']}

Task ID: {task['id']}

Affected paths:
{paths_block}

Issue description:

{task['issue_description']}
"""


def _seeded_env(seed: int) -> Dict[str, str]:
    """Inject seed-related env vars. Subprocess inherits the rest from os.environ."""
    env = os.environ.copy()
    env["EVAL_SEED"] = str(seed)
    # Some libraries respect PYTHONHASHSEED for reproducible iteration order.
    env["PYTHONHASHSEED"] = str(seed)
    return env


def run_claude_code(
    task: Dict[str, Any],
    workspace: Workspace,
    timeout: int = 900,
    seed: int = 0,
) -> AgentOutput:
    """Invoke `claude -p <prompt> --output-format json` inside the sandbox."""
    prompt = _task_to_prompt(task)
    started = time.time()
    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace.repo_path,
            capture_output=True, text=True,
            timeout=timeout, check=False, env=_seeded_env(seed),
        )
    except FileNotFoundError:
        return AgentOutput(
            agent="claude_code", task_id=task["id"], diff="", seed=seed,
            error="`claude` CLI not found on PATH",
        )
    except subprocess.TimeoutExpired:
        return AgentOutput(
            agent="claude_code", task_id=task["id"], seed=seed,
            diff=capture_diff(workspace.repo_path, workspace.pinned_sha),
            error=f"timeout after {timeout}s",
        )

    return AgentOutput(
        agent="claude_code", task_id=task["id"], seed=seed,
        diff=capture_diff(workspace.repo_path, workspace.pinned_sha),
        stdout=proc.stdout, stderr=proc.stderr,
        exit_code=proc.returncode,
        elapsed_s=time.time() - started,
        tool_trace=_parse_claude_trace(proc.stdout),
    )


def _parse_claude_trace(stdout: str) -> List[Dict[str, Any]]:
    """Extract tool_use blocks from Claude Code's JSON output."""
    trace: List[Dict[str, Any]] = []
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return trace
    messages = data.get("messages") if isinstance(data, dict) else None
    if not messages:
        return trace
    for m in messages:
        for block in m.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                trace.append({"tool": block.get("name"), "input": block.get("input")})
    return trace


def run_kiro(
    task: Dict[str, Any],
    workspace: Workspace,
    timeout: int = 900,
    seed: int = 0,
) -> AgentOutput:
    """Invoke `kiro-cli chat --prompt <prompt>` inside the sandbox.

    Adjust flags if your installed Kiro version differs — the workshop's
    notebook 01 prereqs cell prints your version.
    """
    prompt = _task_to_prompt(task)
    started = time.time()
    cmd = ["kiro-cli", "chat", "--prompt", prompt]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace.repo_path,
            capture_output=True, text=True,
            timeout=timeout, check=False, env=_seeded_env(seed),
        )
    except FileNotFoundError:
        return AgentOutput(
            agent="kiro", task_id=task["id"], diff="", seed=seed,
            error="`kiro-cli` not found on PATH",
        )
    except subprocess.TimeoutExpired:
        return AgentOutput(
            agent="kiro", task_id=task["id"], seed=seed,
            diff=capture_diff(workspace.repo_path, workspace.pinned_sha),
            error=f"timeout after {timeout}s",
        )

    return AgentOutput(
        agent="kiro", task_id=task["id"], seed=seed,
        diff=capture_diff(workspace.repo_path, workspace.pinned_sha),
        stdout=proc.stdout, stderr=proc.stderr,
        exit_code=proc.returncode,
        elapsed_s=time.time() - started,
    )


def run_user_agent(
    task: Dict[str, Any],
    workspace: Workspace,
    module: str,
    tasks_file: Path,
    cwd: Optional[Path] = None,
    timeout: int = 900,
    seed: int = 0,
) -> AgentOutput:
    """Invoke the user-built agent via `python -m <module>`.

    The agent must conform to the contract validated in the agent-build
    notebook: --task-id / --tasks-file / --repo / --out / --trace-out.
    Tokens are surfaced in the trace via a synthetic entry of shape
    `{"tool": "_usage", "input": {"input_tokens": ..., "output_tokens": ...}}`.
    """
    started = time.time()
    diff_path = Path(workspace.root) / f"{task['id']}.diff"
    trace_path = Path(workspace.root) / f"{task['id']}.trace.json"
    cmd = [
        sys.executable, "-m", module,
        "--task-id", task["id"],
        "--tasks-file", str(tasks_file),
        "--repo", str(workspace.repo_path),
        "--out", str(diff_path),
        "--trace-out", str(trace_path),
        "--seed", str(seed),
    ]
    run_cwd = cwd or Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.run(
            cmd, cwd=run_cwd,
            capture_output=True, text=True,
            timeout=timeout, check=False, env=_seeded_env(seed),
        )
    except subprocess.TimeoutExpired:
        return AgentOutput(
            agent=module, task_id=task["id"], seed=seed,
            diff=capture_diff(workspace.repo_path, workspace.pinned_sha),
            error=f"timeout after {timeout}s",
        )

    diff = diff_path.read_text() if diff_path.exists() else capture_diff(workspace.repo_path, workspace.pinned_sha)
    trace: List[Dict[str, Any]] = []
    if trace_path.exists():
        try:
            trace = json.loads(trace_path.read_text())
            if not isinstance(trace, list):
                trace = []
        except json.JSONDecodeError:
            trace = []

    in_tok = out_tok = None
    for entry in trace:
        if isinstance(entry, dict) and entry.get("tool") == "_usage":
            usage = entry.get("input") or {}
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")

    return AgentOutput(
        agent=module, task_id=task["id"], seed=seed,
        diff=diff,
        stdout=proc.stdout, stderr=proc.stderr,
        exit_code=proc.returncode,
        elapsed_s=time.time() - started,
        tool_trace=trace,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
