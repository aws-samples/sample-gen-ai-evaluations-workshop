"""Copy the shared verifier files into each Harbor task, or check that the copies match.

Harbor builds or uploads each task from its own directory: a separate verifier (notebook 02) is built
from `tests/` as the Docker build context, and a shared verifier (notebook 03) gets `tests/` uploaded
into the agent's container. Neither can reach a file outside the task directory, and a symbolic link
that points outside it does not survive the copy. Thus each task keeps a full copy of its verifier
files, and this directory is the one place where you edit them:

    _shared/pr-review/tests/   -> awslabs-mcp-pr-review/tasks/*/tests/   (score.py, judge.toml, ...)
    _shared/coding/tests/      -> strands-evals-coding/tasks/*/tests/    (lib/*.py, judge/models.toml)

Files that differ between tasks (oracle.json, test.sh of the coding tasks, the hidden tests, the
rubric, the validation spec) are not here. They stay only in the task directory.

Usage, from the module directory or from anywhere:

    python datasets/_shared/sync.py           # copy the shared files into every task
    python datasets/_shared/sync.py --check   # exit 1 if a copy differs from its source
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent
DATASETS = SHARED.parent

#: Each group of shared files, and the datasets whose tasks get a copy.
GROUPS: dict[str, str] = {
    "pr-review": "awslabs-mcp-pr-review",
    "coding": "strands-evals-coding",
}


def source_files(group: str) -> list[Path]:
    """The shared files of a group, relative to the group directory."""
    root = SHARED / group
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def task_dirs(dataset: str) -> list[Path]:
    return sorted(p for p in (DATASETS / dataset / "tasks").iterdir() if (p / "task.toml").is_file())


def pairs() -> list[tuple[Path, Path]]:
    """Every (source, copy) pair."""
    out = []
    for group, dataset in GROUPS.items():
        for task in task_dirs(dataset):
            out += [(SHARED / group / rel, task / rel) for rel in source_files(group)]
    return out


def stale_copies() -> list[Path]:
    """The copies that are missing or that differ from their source."""
    return [dst for src, dst in pairs() if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False)]


def sync() -> list[Path]:
    """Copy each stale file (content and mode). Return the copies that changed."""
    changed = stale_copies()
    by_dst = {dst: src for src, dst in pairs()}
    for dst in changed:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(by_dst[dst], dst)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only check; exit 1 if a copy differs")
    args = parser.parse_args(argv)
    module = DATASETS.parent
    if args.check:
        stale = stale_copies()
        for path in stale:
            print(f"out of sync: {path.relative_to(module)}")
        if stale:
            print("Edit the file in datasets/_shared/, then run: python datasets/_shared/sync.py")
            return 1
        print(f"shared verifier files: {len(pairs())} copies in sync")
        return 0
    changed = sync()
    for path in changed:
        print(f"updated: {path.relative_to(module)}")
    print(f"shared verifier files: {len(changed)} of {len(pairs())} copies updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
