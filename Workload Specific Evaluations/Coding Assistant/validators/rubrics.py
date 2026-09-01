"""Validate ground-truth rubric files produced by the user's coding assistant.

Rubrics are the alignment anchor for the LLM judge — if they're sloppy,
every downstream signal drifts. This validator is strict by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

import yaml

from pr_reviewer.rubric import Rubric


@dataclass
class RubricValidation:
    path: Path
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Validating {self.path.name}"]
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}")
        if self.dimensions:
            lines.append(f"Dimensions: {', '.join(self.dimensions)}")
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        return "\n".join(lines)


def validate_rubric(path: Path) -> RubricValidation:
    path = Path(path)
    result = RubricValidation(path=path, passed=False)
    if not path.exists():
        result.errors.append("file not found")
        return result

    text = path.read_text()
    try:
        rubric = Rubric.from_markdown(text, task_id=path.stem)
    except Exception as e:
        result.errors.append(f"parse error: {e}")
        return result

    result.dimensions = list(rubric.dimensions)

    if not rubric.title:
        result.errors.append("missing top-level heading (# <title>)")

    if "## Dimensions" not in text:
        result.errors.append("missing '## Dimensions' section")

    if len(rubric.dimensions) < 3:
        result.errors.append(f"only {len(rubric.dimensions)} dimensions — aim for 3-5")

    seen: Set[str] = set()
    for d in rubric.dimensions:
        if d in seen:
            result.errors.append(f"duplicate dimension: {d}")
        seen.add(d)
        body = rubric.dimension_criteria.get(d, "").strip()
        if len(body) < 30:
            result.errors.append(f"dimension '{d}' has almost no criteria ({len(body)} chars)")
        if "- " not in body:
            result.warnings.append(f"dimension '{d}' has no bullet criteria — consider adding some")

    if "scope" not in " ".join(rubric.dimensions).lower():
        result.warnings.append(
            "no scope-discipline dimension — most PR reviews should penalize drive-by edits"
        )

    if not rubric.red_flags:
        result.errors.append("missing '## Red flags' section or empty list")
    elif len(rubric.red_flags) < 2:
        result.warnings.append(f"only {len(rubric.red_flags)} red flag — most tasks have 2-4")

    result.passed = not result.errors
    return result


def validate_rubric_matches_tasks(rubric_dir: Path, tasks_yaml: Path) -> RubricValidation:
    """Check that there's exactly one rubric per non-nav-only task id.

    Nav-only tasks have no diff to review, so they don't need a rubric.
    """
    result = RubricValidation(path=Path(rubric_dir), passed=False)
    data = yaml.safe_load(Path(tasks_yaml).read_text())
    task_ids = {
        t["id"] for t in data.get("tasks", []) if not t.get("nav_only")
    }
    rubric_files = {p.stem for p in Path(rubric_dir).glob("*.md")}

    missing = task_ids - rubric_files
    extra = rubric_files - task_ids

    for m in sorted(missing):
        result.errors.append(f"task {m} has no rubric file {rubric_dir}/{m}.md")
    for e in sorted(extra):
        result.warnings.append(f"rubric {e}.md has no matching (non-nav-only) task id")

    result.passed = not result.errors
    return result
