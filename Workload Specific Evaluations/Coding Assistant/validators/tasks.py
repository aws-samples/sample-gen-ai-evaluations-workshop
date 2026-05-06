"""Validate a tasks.yaml produced by the user's coding assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


REQUIRED_TASK_FIELDS = {
    "id",
    "title",
    "difficulty",
    "skills",
    "affected_paths",
    "issue_description",
    "expected_tools",
    "relevant_files",
    "qa_pairs",
}

REQUIRED_EXPECTED_TOOLS_FIELDS = {"required", "forbidden"}

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

REQUIRED_QA_FIELDS = {"q", "a", "relevant_files"}

MIN_TRAPS = 2
MIN_NAV_ONLY = 2


@dataclass
class TaskValidation:
    path: Path
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Validating {self.path}"]
        mark = "PASS" if self.passed else "FAIL"
        lines.append(f"Result: {mark}")
        if self.task_ids:
            lines.append(f"Tasks found: {', '.join(self.task_ids)}")
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        return "\n".join(lines)


def validate_tasks_file(path: Path, repo_root: Path | None = None) -> TaskValidation:
    """Check a tasks.yaml has the required structure.

    If `repo_root` is given, also checks that each task's affected_paths
    and relevant_files actually exist in the cloned target repo — a good
    gut-check on whether Claude hallucinated paths.
    """
    path = Path(path)
    result = TaskValidation(path=path, passed=False)

    if not path.exists():
        result.errors.append(f"File not found: {path}")
        return result

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        result.errors.append(f"YAML parse error: {e}")
        return result

    if not isinstance(data, dict):
        result.errors.append("Top-level YAML must be a mapping with keys `repo` and `tasks`")
        return result

    if "repo" not in data:
        result.errors.append("Missing top-level `repo` section (expects `url` and `pinned_sha`)")
    else:
        repo = data["repo"]
        for key in ("url", "pinned_sha"):
            if key not in repo:
                result.errors.append(f"Missing `repo.{key}`")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        result.errors.append("Missing or empty `tasks` list")
        return result

    if len(tasks) < 5:
        result.warnings.append(f"Only {len(tasks)} tasks — recommend 5-10 for useful coverage")
    if len(tasks) > 15:
        result.warnings.append(f"{len(tasks)} tasks — expect long eval runs; recommend <=10")

    seen_ids = set()
    n_traps = 0
    n_nav_only = 0
    for i, task in enumerate(tasks):
        ref = task.get("id") or f"task[{i}]"
        if not isinstance(task, dict):
            result.errors.append(f"{ref}: entry must be a mapping")
            continue

        missing = REQUIRED_TASK_FIELDS - set(task.keys())
        if missing:
            result.errors.append(f"{ref}: missing fields {sorted(missing)}")

        tid = task.get("id")
        if tid:
            if tid in seen_ids:
                result.errors.append(f"{ref}: duplicate task id")
            seen_ids.add(tid)
            result.task_ids.append(tid)

        diff = task.get("difficulty")
        if diff and diff not in ALLOWED_DIFFICULTIES:
            result.errors.append(
                f"{ref}: difficulty {diff!r} must be one of {sorted(ALLOWED_DIFFICULTIES)}"
            )

        paths = task.get("affected_paths")
        if paths is not None and not isinstance(paths, list):
            result.errors.append(f"{ref}: `affected_paths` must be a list")
        elif isinstance(paths, list) and repo_root is not None:
            for p in paths:
                target = Path(repo_root) / p
                if not target.exists() and "tests/" not in p:
                    result.warnings.append(f"{ref}: affected path not found in repo: {p}")

        rel = task.get("relevant_files")
        if rel is not None:
            if not isinstance(rel, list) or not rel:
                result.errors.append(
                    f"{ref}: `relevant_files` must be a non-empty list (IR gold for autonomous task)"
                )
            elif repo_root is not None:
                for p in rel:
                    target = Path(repo_root) / p
                    if not target.exists() and "tests/" not in p:
                        result.warnings.append(f"{ref}: relevant_files path not in repo: {p}")

        exp = task.get("expected_tools")
        if exp is not None:
            if not isinstance(exp, dict):
                result.errors.append(f"{ref}: `expected_tools` must be a mapping with `required` and `forbidden` lists")
            else:
                missing_exp = REQUIRED_EXPECTED_TOOLS_FIELDS - set(exp.keys())
                if missing_exp:
                    result.errors.append(
                        f"{ref}: expected_tools missing {sorted(missing_exp)}"
                    )
                for key in ("required", "forbidden"):
                    if key in exp and not isinstance(exp[key], list):
                        result.errors.append(f"{ref}: `expected_tools.{key}` must be a list")

        issue = task.get("issue_description", "")
        if isinstance(issue, str) and len(issue.strip()) < 50:
            result.warnings.append(
                f"{ref}: issue_description is very short ({len(issue.strip())} chars). "
                "Good issues read like real GitHub issues — give context."
            )

        is_trap = task.get("is_trap")
        if is_trap is not None and not isinstance(is_trap, bool):
            result.errors.append(f"{ref}: `is_trap` must be a boolean")
        if is_trap is True:
            n_traps += 1

        nav_only = task.get("nav_only")
        if nav_only is not None and not isinstance(nav_only, bool):
            result.errors.append(f"{ref}: `nav_only` must be a boolean")
        if nav_only is True:
            n_nav_only += 1

        qa = task.get("qa_pairs")
        if qa is not None:
            if not isinstance(qa, list) or not qa:
                result.errors.append(
                    f"{ref}: `qa_pairs` must be a non-empty list (need ≥1 question for pair-programmer eval)"
                )
            else:
                for j, pair in enumerate(qa):
                    pref = f"{ref}.qa_pairs[{j}]"
                    if not isinstance(pair, dict):
                        result.errors.append(f"{pref}: must be a mapping")
                        continue
                    qa_missing = REQUIRED_QA_FIELDS - set(pair.keys())
                    if qa_missing:
                        result.errors.append(f"{pref}: missing fields {sorted(qa_missing)}")
                    pair_rel = pair.get("relevant_files")
                    if pair_rel is not None and (
                        not isinstance(pair_rel, list) or not pair_rel
                    ):
                        result.errors.append(
                            f"{pref}: `relevant_files` must be a non-empty list"
                        )
                    if not isinstance(pair.get("q", ""), str) or not pair.get("q", "").strip():
                        result.errors.append(f"{pref}: `q` must be a non-empty string")
                    if not isinstance(pair.get("a", ""), str) or not pair.get("a", "").strip():
                        result.errors.append(f"{pref}: `a` must be a non-empty string")

    if n_traps < MIN_TRAPS:
        result.errors.append(
            f"Need at least {MIN_TRAPS} trap tasks (is_trap: true) for the honesty signal "
            f"— found {n_traps}"
        )
    if n_nav_only < MIN_NAV_ONLY:
        result.errors.append(
            f"Need at least {MIN_NAV_ONLY} nav-only tasks (nav_only: true) "
            f"— found {n_nav_only}"
        )

    result.passed = not result.errors
    return result
