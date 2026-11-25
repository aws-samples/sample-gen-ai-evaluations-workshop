"""Helpers for the notebook `step2-run-evals.ipynb`: show the task files, scan for leaks, and run the scripts.

The scripts do the work (`harbor_env/setup.py`, `harbor run`, `harbor_env/summary.py`). These helpers
run them from the notebook, in this folder, and show their output.

    import walkthrough as wt
    job = wt.harbor_run("first-oracle", [TASK], "-a", "oracle")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
JOBS_DIR = RUN_DIR / "jobs"
ENVIRONMENT = "harbor_env.agentcore_env:AgentCoreEnvironment"
BIN = Path(sys.executable).parent  # harbor is installed next to the kernel's Python
CANARY = "CANARY-7f3a91"


# ── Files ────────────────────────────────────────────────────────────────────

def read_toml(path: Path) -> dict:
    return tomllib.loads(Path(path).read_text())


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        if default is not None:
            return default
        raise


def show_tree(root: Path) -> None:
    """Print the files of a task directory as a tree."""
    root = Path(root)
    print(root.name + "/")
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        depth = len(path.relative_to(root).parts)
        print("    " * depth + path.name + ("/" if path.is_dir() else ""))


def check_shared_files() -> None:
    """Make sure that each task's copy of a shared verifier file matches datasets/_shared/."""
    r = subprocess.run([sys.executable, "datasets/_shared/sync.py", "--check"], cwd=RUN_DIR,
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    assert r.returncode == 0, "Run `python datasets/_shared/sync.py` in step2-run-evals/, then run this cell again."


# ── Leakage scan ─────────────────────────────────────────────────────────────

def agent_visible_files(task_dir: Path) -> Iterator[Path]:
    """Files that the agent sees: the instruction and the files that build its image."""
    task_dir = Path(task_dir)
    yield task_dir / "instruction.md"
    yield from (p for p in sorted((task_dir / "environment").rglob("*")) if p.is_file())


def _scan(task_dir: Path, needles: list[str]) -> list[tuple[str, str]]:
    hits = []
    for path in agent_visible_files(task_dir):
        text = path.read_text(errors="ignore").lower()
        hits += [(str(path.relative_to(task_dir)), n) for n in needles if n and n.lower() in text]
    return hits


def leakage_scan(task_dir: Path, needles: list[str]) -> list[tuple[str, str]]:
    """Search the agent-visible files for each needle. Return (file, needle) for each hit.

    First the scan tests itself: it plants a canary in a temporary copy of the task and must find it.
    The task in the repository never changes.
    """
    with tempfile.TemporaryDirectory(prefix="leakage-canary-") as tmp:
        copy = Path(tmp) / Path(task_dir).name
        shutil.copytree(task_dir, copy)
        (copy / "environment" / "canary.txt").write_text(CANARY)
        if not _scan(copy, [CANARY]):
            raise AssertionError("The scan did not find the canary. The scan has a defect.")
    return _scan(Path(task_dir), needles)


# ── Running the scripts ──────────────────────────────────────────────────────

def _env() -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(RUN_DIR), os.environ.get("PYTHONPATH")]))}
    env.setdefault("AWS_REGION", "us-east-1")
    env.pop("AWS_BEARER_TOKEN_BEDROCK", None)  # a Bedrock API key makes bootstrap write the wrong judge model
    return env


def run(cmd: list, log: Path, tail: int = 15) -> None:
    """Run a command in step2-run-evals/, with its output in `log`. Show the time while it runs, then the end of the log."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(str(c) for c in cmd))
    start = next_note = time.time()
    with log.open("w") as f:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=RUN_DIR, env=_env(), stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            time.sleep(5)
            if time.time() >= next_note + 60:
                next_note += 60
                print(f"  ... {(time.time() - start) / 60:.0f} min, still running (log: {log.relative_to(RUN_DIR)})")
    print("\n".join(log.read_text(errors="replace").splitlines()[-tail:]))
    print(f"({(time.time() - start) / 60:.1f} min)")
    assert proc.returncode == 0, f"The command failed. See {log}."


def setup(command: str, *args: str, tail: int = 15) -> None:
    """Run `harbor_env/setup.py COMMAND ARGS`."""
    run([sys.executable, "harbor_env/setup.py", command, *args], JOBS_DIR / f"setup-{command}.log", tail=tail)


def harbor_run(job_name: str, tasks: list[Path], *agent_args: str, attempts: int = 1,
               concurrent: int | None = None) -> Path:
    """Run the tasks as one Harbor job on AgentCore. Return the job directory, jobs/<job_name>/.

    A new run replaces an old job of the same name: with an existing name, Harbor resumes the old job
    and runs nothing.
    """
    job_dir = JOBS_DIR / job_name
    if job_dir.exists():
        shutil.rmtree(job_dir)
    tasks = [Path(t).resolve() for t in tasks]
    if len(tasks) == 1:
        select = ["-p", tasks[0]]
    else:  # harbor run takes one -p: the dataset folder, and -i for each task in it
        folders = {t.parent for t in tasks}
        if len(folders) != 1:
            raise ValueError("Run the tasks of one dataset folder in a job: one job for PR_TASKS, one for CODING_TASKS.")
        (folder,) = folders
        select = ["-p", folder, *[a for t in tasks for a in ("-i", t.name)]]
    cmd = [BIN / "harbor", "run", *select,
           "-e", ENVIRONMENT, "-o", JOBS_DIR, "--job-name", job_name, "-y", "-k", str(attempts)]
    if concurrent:
        cmd += ["-n", str(concurrent)]
    run(cmd + list(agent_args), JOBS_DIR / f"{job_name}.log", tail=12)
    return job_dir


def summary(*job_dirs: Path) -> None:
    """Show `harbor_env/summary.py` for the jobs: one row for each trial."""
    r = subprocess.run([sys.executable, "harbor_env/summary.py", *map(str, job_dirs)], cwd=RUN_DIR,
                       capture_output=True, text=True)
    try:
        from IPython.display import Markdown, display
        display(Markdown(r.stdout))
    except ImportError:
        print(r.stdout)
    if r.returncode:
        print(r.stderr)


# ── Results ──────────────────────────────────────────────────────────────────

def trials(job_dir: Path) -> Iterator[tuple[str, Path, dict]]:
    """(task directory name, trial directory, result.json) for each trial of a job."""
    for result_path in sorted(Path(job_dir).glob("*/result.json")):
        result = read_json(result_path)
        yield result["task_name"].replace("/", "__"), result_path.parent, result


def trial_rewards(job_dir: Path) -> dict[str, dict]:
    """The reward of each trial, by task directory name. A trial with no reward.json has reward None."""
    rewards = {}
    for task, trial_dir, _ in trials(job_dir):
        reward = read_json(trial_dir / "verifier" / "reward.json", {})
        if not reward:
            details = read_json(trial_dir / "verifier" / "reward_details.json", {})
            reward = {"reward": None, "invalid": details.get("invalid") or ["no reward file"]}
        rewards[task] = reward
    return rewards


def grading_mode(trial_dir: Path) -> str:
    """How the environment graded the trial: `staged`, `separate`, `unsealed`, or `refused`."""
    log = Path(trial_dir) / "agentcore.jsonl"
    for line in reversed(log.read_text().splitlines() if log.exists() else []):
        event = json.loads(line)
        if event.get("event") == "grader_mode":
            return event.get("mode", "-")
    return read_json(Path(trial_dir) / "result.json", {}).get("verifier_environment_mode") or "-"


# ── Harbor view ──────────────────────────────────────────────────────────────

def start_harbor_view(ports: str = "8080-8089", wait_sec: int = 120):
    """Start `harbor view jobs` in the background. Return (process, url)."""
    JOBS_DIR.mkdir(exist_ok=True)
    log = JOBS_DIR / "harbor-view.log"
    proc = subprocess.Popen([str(BIN / "harbor"), "view", str(JOBS_DIR), "--port", ports], cwd=RUN_DIR,
                            stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    urls: list[str] = []
    for _ in range(wait_sec):
        urls = re.findall(r"https?://[\w.:-]+", log.read_text())
        if urls or proc.poll() is not None:
            break
        time.sleep(1)
    if not urls:
        print(log.read_text()[-2000:])
        raise RuntimeError(f"harbor view did not start. See {log}.")
    print(f"Harbor view runs at {urls[0]}")
    return proc, urls[0]


def stop_harbor_view(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Harbor view stopped.")
