"""Test and static-check runners against an agent-modified workspace.

Both functions are deliberately thin wrappers around the repo's own
Makefile targets (`make test-unit`, `make lint`) so the eval exercises
exactly what a developer would run locally. We rely on `uv` being on
PATH — sample-agentic-platform uses it exclusively.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    passed: bool
    summary: str          # one-line human-readable summary
    details: str          # full stdout+stderr for debugging
    violations: int = 0   # failing test count or lint violation count


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_tests(repo_path: Path, timeout: int = 600) -> CheckResult:
    """Run `uv run pytest tests/unit/` against the workspace.

    Uses the unit subset because integration tests require live AWS
    resources (Bedrock, RDS, Cognito) that the workshop doesn't provision.
    """
    proc = _run(
        ["uv", "run", "pytest", "tests/unit/", "--tb=short", "-q"],
        cwd=repo_path,
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    failed = 0
    # pytest summary line, e.g. "3 failed, 42 passed in 5.12s"
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    return CheckResult(
        passed=proc.returncode == 0,
        summary=f"pytest exit={proc.returncode} failed={failed}",
        details=output,
        violations=failed,
    )


def run_static_checks(repo_path: Path, timeout: int = 120) -> CheckResult:
    """Run `uv run ruff check src/` against the workspace."""
    proc = _run(
        ["uv", "run", "ruff", "check", "src/"],
        cwd=repo_path,
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    violations = 0
    # ruff prints "Found N errors." on failure; on success it's silent.
    m = re.search(r"Found (\d+) errors?", output)
    if m:
        violations = int(m.group(1))
    elif proc.returncode != 0:
        # Non-zero with no "Found N errors" line — count file-line entries.
        violations = len(re.findall(r"^[^:]+:\d+:\d+:", output, flags=re.MULTILINE))
    return CheckResult(
        passed=proc.returncode == 0,
        summary=f"ruff exit={proc.returncode} violations={violations}",
        details=output,
        violations=violations,
    )
