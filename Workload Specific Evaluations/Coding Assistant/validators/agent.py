"""Validate that the user's custom coding agent meets the expected CLI contract.

The workshop asks the user to build an agent that can be invoked as:

    python -m <user_module> \\
        --task-id <id> --tasks-file <path> --repo <path> \\
        --out <diff_path> --trace-out <trace_path>

and produces:
  - A unified git diff at `--out`.
  - A JSON trace of tool uses at `--trace-out` (list of {"tool": ..., "input": ...}).

This validator runs the agent against a trivial "noop" task and confirms
the contract without incurring meaningful Bedrock cost.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


NOOP_TASK = {
    "id": "NOOP_CONTRACT_CHECK",
    "title": "Contract check: no-op",
    "difficulty": "easy",
    "skills": ["smoke"],
    "affected_paths": [],
    "expected_tools": {"required": [], "forbidden": []},
    "issue_description": (
        "This is a no-op contract check. Do not modify any files. "
        "Your only job is to respond with `TASK_COMPLETE` and exit. "
        "Do not call any tools."
    ),
}


@dataclass
class AgentValidation:
    module: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    diff_exists: bool = False
    trace_exists: bool = False
    trace_is_list: bool = False
    stdout_tail: str = ""

    def report(self) -> str:
        lines = [f"Validating agent module: {self.module}"]
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}  ({self.elapsed_s:.1f}s)")
        lines.append(f"Diff file produced: {self.diff_exists}")
        lines.append(f"Trace file produced: {self.trace_exists} (list: {self.trace_is_list})")
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        if self.stdout_tail:
            lines.append("--- stdout tail ---")
            lines.append(self.stdout_tail)
        return "\n".join(lines)


def validate_agent_contract(
    module: str,
    repo_path: Path,
    cwd: Path | None = None,
    timeout: int = 180,
) -> AgentValidation:
    """Run the user's agent against a no-op task and check CLI contract."""
    result = AgentValidation(module=module, passed=False)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tasks_file = td_path / "tasks.yaml"
        diff_out = td_path / "diff.patch"
        trace_out = td_path / "trace.json"
        tasks_file.write_text(yaml.safe_dump({
            "repo": {"url": "local", "pinned_sha": "HEAD"},
            "tasks": [NOOP_TASK],
        }))

        cmd = [
            sys.executable,
            "-m",
            module,
            "--task-id",
            NOOP_TASK["id"],
            "--tasks-file",
            str(tasks_file),
            "--repo",
            str(repo_path),
            "--out",
            str(diff_out),
            "--trace-out",
            str(trace_out),
        ]
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            result.errors.append(f"Could not invoke `{sys.executable} -m {module}` — module not importable")
            return result
        except subprocess.TimeoutExpired:
            result.errors.append(f"Agent timed out after {timeout}s on a no-op task — the CLI is likely hanging")
            return result
        result.elapsed_s = time.time() - started
        result.stdout_tail = (proc.stdout[-800:] if proc.stdout else "") + (proc.stderr[-400:] if proc.stderr else "")

        if proc.returncode not in (0, 2):
            result.errors.append(
                f"unexpected exit code {proc.returncode}. Expected 0 (complete) or 2 (not complete)."
            )

        result.diff_exists = diff_out.exists()
        if not result.diff_exists:
            result.errors.append(f"--out file not created: {diff_out}")

        result.trace_exists = trace_out.exists()
        if result.trace_exists:
            try:
                trace_data = json.loads(trace_out.read_text())
                result.trace_is_list = isinstance(trace_data, list)
                if result.trace_is_list:
                    for entry in trace_data[:5]:
                        if not isinstance(entry, dict) or "tool" not in entry:
                            result.warnings.append(
                                "trace entries should be {'tool': <name>, 'input': <dict>}"
                            )
                            break
                else:
                    result.errors.append("trace file must contain a JSON list")
            except json.JSONDecodeError as e:
                result.errors.append(f"trace file is not valid JSON: {e}")
        else:
            result.warnings.append("--trace-out file not created (tool-call eval will be empty for this agent)")

    result.passed = not result.errors
    return result
