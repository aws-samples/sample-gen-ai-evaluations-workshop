"""Q&A mode runner for the pair-programmer eval.

Each agent is invoked with a single question. We capture:
  - the agent's final text answer
  - the ordered list of files it touched (from its tool trace)

The trace-derived retrieved-file list is the IR signal scored in
`validators/retrieval.py`. Off-the-shelf CLIs that don't reliably emit
a trace (Kiro) fall back to regex-extracting `path:line` mentions from
the answer text.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workspace import Workspace


# Tools whose inputs name a file we should treat as "retrieved".
FILE_INPUT_KEYS = ("file", "file_path", "path", "filename")


@dataclass
class QAOutput:
    agent: str
    task_id: str
    question: str
    answer: str
    retrieved_files: List[str] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    exit_code: int = 0
    stderr: str = ""
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


PATH_RE = re.compile(
    r"(?<![\w/])([A-Za-z0-9_./-]+\.(?:py|md|yaml|yml|toml|json|sh|cfg|ini|js|ts|tsx|jsx|go|rs))"
)


def _qa_prompt(question: str) -> str:
    return f"""Answer this question about the code in this repo. Be concise and cite
specific file paths and line numbers (path:line) when you can. If you
can't answer with confidence, say so plainly — do not guess.

Question: {question}
"""


def _extract_paths_from_input(value: Any) -> List[str]:
    """Pull file-like strings out of a tool input. Handles dicts, lists, and bare strings."""
    out: List[str] = []
    if isinstance(value, str):
        for m in PATH_RE.finditer(value):
            out.append(m.group(1))
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in FILE_INPUT_KEYS and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_extract_paths_from_input(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_extract_paths_from_input(v))
    return out


def retrieved_files_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    """Ordered, deduped file list from a tool trace."""
    seen = set()
    out: List[str] = []
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        for path in _extract_paths_from_input(entry.get("input")):
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def retrieved_files_from_text(text: str) -> List[str]:
    """Fallback: regex `path.ext` mentions from the agent's answer."""
    seen = set()
    out: List[str] = []
    for m in PATH_RE.finditer(text or ""):
        path = m.group(1)
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _parse_claude_json(stdout: str) -> tuple[str, List[Dict[str, Any]], Optional[int], Optional[int]]:
    """Pull final text + tool_use trace + token counts from `claude -p --output-format json`."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, [], None, None
    if not isinstance(data, dict):
        return stdout, [], None, None

    answer_parts: List[str] = []
    trace: List[Dict[str, Any]] = []
    in_tok = data.get("usage", {}).get("input_tokens") if isinstance(data.get("usage"), dict) else None
    out_tok = data.get("usage", {}).get("output_tokens") if isinstance(data.get("usage"), dict) else None

    for m in data.get("messages") or []:
        for block in m.get("content") or []:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "tool_use":
                trace.append({"tool": block.get("name"), "input": block.get("input")})
            elif t == "text" and m.get("role") == "assistant":
                answer_parts.append(block.get("text", ""))
    answer = "\n".join(p for p in answer_parts if p)
    if not answer:
        # Some claude versions return a top-level "result" string.
        answer = data.get("result") or stdout
    return answer, trace, in_tok, out_tok


def _run_claude_qa(question: str, workspace: Workspace, timeout: int) -> QAOutput:
    started = time.time()
    cmd = ["claude", "-p", _qa_prompt(question), "--output-format", "json"]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace.repo_path,
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except FileNotFoundError:
        return QAOutput(
            agent="claude_code", task_id=workspace.task_id, question=question,
            answer="", error="`claude` CLI not found",
        )
    except subprocess.TimeoutExpired:
        return QAOutput(
            agent="claude_code", task_id=workspace.task_id, question=question,
            answer="", error=f"timeout after {timeout}s",
        )
    answer, trace, in_tok, out_tok = _parse_claude_json(proc.stdout)
    retrieved = retrieved_files_from_trace(trace) or retrieved_files_from_text(answer)
    return QAOutput(
        agent="claude_code", task_id=workspace.task_id, question=question,
        answer=answer, retrieved_files=retrieved, tool_trace=trace,
        elapsed_s=time.time() - started, exit_code=proc.returncode,
        stderr=proc.stderr, input_tokens=in_tok, output_tokens=out_tok,
    )


def _run_kiro_qa(question: str, workspace: Workspace, timeout: int) -> QAOutput:
    started = time.time()
    cmd = ["kiro-cli", "chat", "--prompt", _qa_prompt(question)]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace.repo_path,
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except FileNotFoundError:
        return QAOutput(
            agent="kiro", task_id=workspace.task_id, question=question,
            answer="", error="`kiro-cli` not found",
        )
    except subprocess.TimeoutExpired:
        return QAOutput(
            agent="kiro", task_id=workspace.task_id, question=question,
            answer="", error=f"timeout after {timeout}s",
        )
    answer = proc.stdout
    # Kiro doesn't reliably emit a tool trace; regex-parse the answer.
    retrieved = retrieved_files_from_text(answer)
    return QAOutput(
        agent="kiro", task_id=workspace.task_id, question=question,
        answer=answer, retrieved_files=retrieved,
        elapsed_s=time.time() - started, exit_code=proc.returncode,
        stderr=proc.stderr,
    )


def _run_user_agent_qa(
    question: str,
    workspace: Workspace,
    module: str,
    cwd: Path,
    timeout: int,
) -> QAOutput:
    """Custom agent QA mode. Expects --question flag and writes an --answer-out file.

    The agent should also write its trace to --trace-out, same format as the
    autonomous mode.
    """
    started = time.time()
    answer_path = Path(workspace.root) / f"{workspace.task_id}.answer.txt"
    trace_path = Path(workspace.root) / f"{workspace.task_id}.qa.trace.json"
    cmd = [
        sys.executable, "-m", module,
        "--qa",
        "--question", question,
        "--repo", str(workspace.repo_path),
        "--answer-out", str(answer_path),
        "--trace-out", str(trace_path),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=cwd,
            capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return QAOutput(
            agent=module, task_id=workspace.task_id, question=question,
            answer="", error=f"timeout after {timeout}s",
        )

    answer = answer_path.read_text() if answer_path.exists() else proc.stdout
    trace: List[Dict[str, Any]] = []
    if trace_path.exists():
        try:
            data = json.loads(trace_path.read_text())
            if isinstance(data, list):
                trace = data
        except json.JSONDecodeError:
            trace = []
    retrieved = retrieved_files_from_trace(trace) or retrieved_files_from_text(answer)

    in_tok = out_tok = None
    for entry in trace:
        if isinstance(entry, dict) and entry.get("tool") == "_usage":
            usage = entry.get("input") or {}
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")

    return QAOutput(
        agent=module, task_id=workspace.task_id, question=question,
        answer=answer, retrieved_files=retrieved, tool_trace=trace,
        elapsed_s=time.time() - started, exit_code=proc.returncode,
        stderr=proc.stderr, input_tokens=in_tok, output_tokens=out_tok,
    )


def run_qa(
    agent: str,
    question: str,
    workspace: Workspace,
    module: Optional[str] = None,
    cwd: Optional[Path] = None,
    timeout: int = 300,
) -> QAOutput:
    """Dispatch to the right Q&A runner.

    `agent` is one of: "claude_code", "kiro", or any other string (treated
    as the module name for the user-built agent — pass `module=...` to
    override). Custom agent must implement the --qa contract above.
    """
    if agent == "claude_code":
        return _run_claude_qa(question, workspace, timeout)
    if agent == "kiro":
        return _run_kiro_qa(question, workspace, timeout)
    mod = module or agent
    return _run_user_agent_qa(
        question=question, workspace=workspace,
        module=mod, cwd=cwd or Path.cwd(), timeout=timeout,
    )
