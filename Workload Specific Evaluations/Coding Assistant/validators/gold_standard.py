"""Validate gold-standard PR review entries.

A gold-standard entry is a real merged PR diff plus a human-written
per-dimension verdict. Together they form the calibration set used to
measure how well the automated PR reviewer agrees with a human reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from pr_reviewer.rubric import Rubric


REQUIRED_FIELDS = {
    "pr_slug",
    "pr_number",
    "pr_title",
    "rubric_path",
    "diff_path",
    "human_verdicts",
}

ALLOWED_VERDICTS = {"pass", "fail"}


@dataclass
class GoldEntry:
    slug: str
    pr_number: int
    pr_title: str
    rubric_path: Path
    diff_path: Path
    human_verdicts: Dict[str, str]
    human_red_flags_hit: List[str]
    notes: str = ""

    @property
    def rubric(self) -> Rubric:
        return Rubric.from_path(self.rubric_path)

    @property
    def diff(self) -> str:
        return self.diff_path.read_text()


@dataclass
class GoldEntryValidation:
    path: Path
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entry: GoldEntry | None = None

    def report(self) -> str:
        lines = [f"Validating {self.path.name}"]
        lines.append(f"Result: {'PASS' if self.passed else 'FAIL'}")
        if self.entry:
            lines.append(f"PR #{self.entry.pr_number}: {self.entry.pr_title}")
            lines.append(f"Verdicts: {self.entry.human_verdicts}")
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARN    {w}")
        return "\n".join(lines)


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def validate_gold_entry(path: Path) -> GoldEntryValidation:
    """Validate a single gold-standard entry YAML file."""
    path = Path(path)
    result = GoldEntryValidation(path=path, passed=False)
    if not path.exists():
        result.errors.append("file not found")
        return result

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        result.errors.append(f"YAML parse error: {e}")
        return result

    if not isinstance(data, dict):
        result.errors.append("top-level must be a mapping")
        return result

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        result.errors.append(f"missing fields {sorted(missing)}")
        return result

    base = path.parent
    rubric_path = _resolve(data["rubric_path"], base)
    diff_path = _resolve(data["diff_path"], base)

    if not rubric_path.exists():
        result.errors.append(f"rubric_path not found: {rubric_path}")
    if not diff_path.exists():
        result.errors.append(f"diff_path not found: {diff_path}")

    verdicts = data.get("human_verdicts")
    if not isinstance(verdicts, dict) or not verdicts:
        result.errors.append("human_verdicts must be a non-empty mapping of dimension -> pass|fail")
    else:
        for dim, verdict in verdicts.items():
            if verdict not in ALLOWED_VERDICTS:
                result.errors.append(f"verdict for '{dim}' must be pass|fail, got {verdict!r}")

    # If rubric loads, cross-check dimensions.
    if rubric_path.exists() and isinstance(verdicts, dict):
        try:
            rubric = Rubric.from_path(rubric_path)
            rubric_dims = set(rubric.dimensions)
            verdict_dims = set(verdicts.keys())
            missing_dims = rubric_dims - verdict_dims
            extra_dims = verdict_dims - rubric_dims
            for d in sorted(missing_dims):
                result.errors.append(f"verdict missing for rubric dimension: {d}")
            for d in sorted(extra_dims):
                result.warnings.append(f"verdict has dimension not in rubric: {d}")
        except Exception as e:
            result.warnings.append(f"could not cross-check rubric dims: {e}")

    if result.errors:
        return result

    result.entry = GoldEntry(
        slug=data["pr_slug"],
        pr_number=data["pr_number"],
        pr_title=data["pr_title"],
        rubric_path=rubric_path,
        diff_path=diff_path,
        human_verdicts=dict(verdicts),
        human_red_flags_hit=list(data.get("human_red_flags_hit", [])),
        notes=data.get("notes", ""),
    )
    result.passed = True
    return result


def load_gold_set(gold_dir: Path) -> List[GoldEntry]:
    """Load all valid gold-standard entries from a directory."""
    entries: List[GoldEntry] = []
    for p in sorted(Path(gold_dir).glob("*.yaml")):
        v = validate_gold_entry(p)
        if v.passed and v.entry:
            entries.append(v.entry)
    return entries
