"""Workspace management for the Coding Assistant eval.

Clones a target repo at a pinned SHA into an isolated sandbox directory
for each (agent, task) run. Each sandbox is a fresh clone so agents can't
see or taint each other's work, and every run starts from the same commit
so results are reproducible.

The repo URL and pinned SHA are set by the user in notebook 02 and live
in `scaffolding/tasks/tasks.yaml`. Callers pass them explicitly — there's
no module-level default, since the whole point of this workshop is that
users swap in their own repo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Workspace:
    """A per-run scratch directory containing a fresh clone of the target repo."""

    root: Path        # directory containing the cloned repo
    repo_path: Path   # the clone itself
    pinned_sha: str   # SHA used at clone time, for `capture_diff`
    agent: str = ""
    task_id: str = ""

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def _run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _clone_name_from_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git").split("/")[-1] or "repo"


def _clone_pinned(dest: Path, url: str, sha: str) -> Path:
    """Clone `url` at `sha` into `dest/<repo-name>`."""
    repo_path = dest / _clone_name_from_url(url)
    _run(["git", "clone", "--quiet", url, str(repo_path)])
    _run(["git", "checkout", "--quiet", sha], cwd=repo_path)
    return repo_path


def create_workspace(
    repo_url: str,
    pinned_sha: str,
    agent: str = "",
    task_id: str = "",
    base_dir: Optional[Path] = None,
) -> Workspace:
    """Create a per-(agent, task) sandbox with a fresh pinned clone.

    If `CODING_ASSISTANT_EVAL_CACHE` is set, first-time clones populate
    a local cache and subsequent calls copy from that cache instead of
    hitting GitHub — this keeps full eval runs under ~1 minute per agent.
    """
    base = base_dir or Path(tempfile.mkdtemp(prefix=f"coding-eval-{agent or 'run'}-{task_id or 'task'}-"))
    base.mkdir(parents=True, exist_ok=True)
    repo_name = _clone_name_from_url(repo_url)

    cache_env = os.environ.get("CODING_ASSISTANT_EVAL_CACHE")
    if cache_env:
        cache_dir = Path(cache_env).expanduser() / repo_name
        if not cache_dir.exists():
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            _clone_pinned(cache_dir.parent, repo_url, pinned_sha)
        repo_path = base / repo_name
        shutil.copytree(cache_dir, repo_path)
        _run(["git", "checkout", "--quiet", pinned_sha], cwd=repo_path)
    else:
        repo_path = _clone_pinned(base, repo_url, pinned_sha)

    return Workspace(
        root=base, repo_path=repo_path, pinned_sha=pinned_sha,
        agent=agent, task_id=task_id,
    )


def capture_diff(repo_path: Path, pinned_sha: str) -> str:
    """Return the agent's changes vs the pinned SHA as a unified diff string."""
    _run(["git", "add", "-N", "."], cwd=repo_path, check=False)
    result = _run(["git", "diff", pinned_sha], cwd=repo_path, check=False)
    return result.stdout
