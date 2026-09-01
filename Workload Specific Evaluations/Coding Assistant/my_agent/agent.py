"""The agent loop — YOU FILL THIS IN.

The `CodingAgent.run(task)` method is what the eval calls per task.
It receives the parsed task dict (the same shape as entries in
tasks.yaml) and must:

  1. Plan an approach (read the issue_description).
  2. Use tools (read_file, edit_file, run_grep, etc.) to investigate
     and edit the repo at `self.repo_path`.
  3. Record every tool call via `self.trace.tool(name, input)`.
  4. Record cumulative token usage at the end via `self.trace.usage(...)`.
  5. Return a `RunResult` indicating whether the task is "complete".

What "complete" means by task type:

  - Normal task (is_trap=False, nav_only=False): you made code edits
    that you believe fix the described issue. completed=True.
  - Trap task (is_trap=True): you investigated and concluded the bug
    described in the issue does NOT exist in the code. Make NO edits
    and return completed=False (which becomes exit code 2).
  - Nav-only task (nav_only=True): the deliverable is your final answer
    in the model's natural-language response, not a diff. The autonomous
    eval skips these; the pair-programmer eval (notebook 06) handles them.
    For the autonomous CLI, return completed=False with no edits.

You don't need to implement the LLM call yourself — see `model.py` for
a Bedrock-backed Strands `Agent` you can plug in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .trace import Trace


@dataclass
class RunResult:
    completed: bool
    summary: str = ""


class CodingAgent:
    """Wraps your model + tools into the loop the eval calls."""

    def __init__(self, repo_path: Path, trace: Trace, seed: int = 0) -> None:
        self.repo_path = repo_path
        self.trace = trace
        self.seed = seed

        # TODO: instantiate your model here. The `model.py` module gives
        # you a Strands Agent backed by Bedrock + the tools defined in
        # `tools.py`. Something like:
        #
        #     from .model import build_strands_agent
        #     self.agent = build_strands_agent(repo_path=repo_path, trace=trace)
        #
        # Or, if you want full control, instantiate boto3 + tool registry
        # manually and write your own loop.

    def run(self, task: Dict[str, Any]) -> RunResult:
        """Run one task. YOU IMPLEMENT THIS."""
        # TODO: replace this stub with a real agent loop.
        #
        # Suggested skeleton:
        #
        #     prompt = self._build_prompt(task)
        #     response = self.agent(prompt)              # Strands invokes tools
        #     self.trace.usage(...)                      # record token usage
        #     return RunResult(completed=self._completed(task, response))
        #
        # The no-op contract task (NOOP_CONTRACT_CHECK) used by
        # validators.agent expects you to return completed=True with no
        # tool calls — handle it as a fast path if you like:
        #
        #     if task['id'] == 'NOOP_CONTRACT_CHECK':
        #         return RunResult(completed=True, summary='noop')
        if task.get("id") == "NOOP_CONTRACT_CHECK":
            return RunResult(completed=True, summary="noop fast-path")
        raise NotImplementedError(
            "CodingAgent.run is a stub. Open my_agent/agent.py and follow "
            "the TODO comments."
        )

    def _build_prompt(self, task: Dict[str, Any]) -> str:
        """Turn a task dict into the instruction string the model sees.

        TODO: tune this. The minimum useful prompt includes the
        issue_description and a reminder to make minimal, scoped edits.
        Reference: the prompts in scaffolding/prompts.md.
        """
        return (
            f"Task: {task['title']}\n\n"
            f"{task['issue_description']}\n\n"
            f"Repository root: {self.repo_path}\n"
            "Make the minimum changes needed. Stay within affected_paths. "
            "When done, say TASK_COMPLETE."
        )
