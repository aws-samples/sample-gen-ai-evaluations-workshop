"""Rubric loader + default review dimensions.

A Rubric is a structured view of a ground-truth review checklist. The
expected on-disk format is the markdown used in `tasks/ground_truth/*.md`:
an `## Dimensions` section with `### <name>` subsections followed by
bullet criteria, and an optional `## Red flags` section.

When no rubric is supplied (CI usage on an arbitrary PR), the reviewer
falls back to `DEFAULT_DIMENSIONS` — a generic pass/fail review across
correctness, scope, tests, and security/secrets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


DEFAULT_DIMENSIONS: List[str] = [
    "correctness",
    "scope_discipline",
    "test_coverage",
    "security",
]


def _slugify(name: str) -> str:
    """Turn a heading like `correctness — retrieval` into `correctness_retrieval`."""
    cleaned = re.sub(r"[^\w\s-]", " ", name)  # drop punctuation, dashes
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


@dataclass
class Rubric:
    """Parsed view of a ground-truth review rubric."""

    task_id: Optional[str] = None
    title: Optional[str] = None
    dimensions: List[str] = field(default_factory=list)
    dimension_criteria: dict = field(default_factory=dict)  # name -> raw text
    red_flags: List[str] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def from_markdown(cls, text: str, task_id: Optional[str] = None) -> "Rubric":
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else None

        dims: List[str] = []
        dim_criteria: dict = {}
        # ### <n>. <name>  OR  ### <name>
        # Name captures the full heading after the optional number, then is
        # slugified so compound names like "correctness — retrieval" become
        # "correctness_retrieval" (distinct from "correctness_append").
        for m in re.finditer(
            r"^###\s+(?:\d+\.\s*)?(.+?)\s*$",
            text,
            flags=re.MULTILINE,
        ):
            raw_name = m.group(1).strip()
            name = _slugify(raw_name)
            if not name:
                continue
            start = m.end()
            next_section = re.search(r"^##+\s+", text[start:], flags=re.MULTILINE)
            end = start + next_section.start() if next_section else len(text)
            body = text[start:end].strip()
            # If duplicate (e.g. two "correctness" headings with the same
            # slug), de-dup with a suffix so both are preserved.
            final_name = name
            suffix = 2
            while final_name in dim_criteria:
                final_name = f"{name}_{suffix}"
                suffix += 1
            dims.append(final_name)
            dim_criteria[final_name] = body

        red_flags: List[str] = []
        rf_match = re.search(r"##\s+Red flags[\s\S]*?(?=^##\s|\Z)", text, flags=re.MULTILINE)
        if rf_match:
            red_flags = [
                line.strip("- ").strip()
                for line in rf_match.group(0).splitlines()
                if line.strip().startswith("-")
            ]

        return cls(
            task_id=task_id,
            title=title,
            dimensions=dims,
            dimension_criteria=dim_criteria,
            red_flags=red_flags,
            raw=text,
        )

    @classmethod
    def from_path(cls, path: Path) -> "Rubric":
        text = Path(path).read_text()
        return cls.from_markdown(text, task_id=Path(path).stem)

    @classmethod
    def default(cls) -> "Rubric":
        return cls(
            task_id=None,
            title="Generic PR review",
            dimensions=list(DEFAULT_DIMENSIONS),
            dimension_criteria={d: "" for d in DEFAULT_DIMENSIONS},
            red_flags=[
                "Introduces a new top-level dependency without justification",
                "Commits secrets or credentials",
                "Disables tests or linters instead of fixing them",
            ],
            raw="",
        )
