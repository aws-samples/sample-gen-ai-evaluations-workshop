"""Calibrate the automated PR reviewer against a gold-standard set.

For each gold entry (real PR + human verdict), run the PR reviewer with
the same rubric and compare dimension-by-dimension agreement. Report:

  - overall agreement rate
  - per-dimension agreement
  - specific disagreements (with both verdicts + reasons) so the user
    can decide whether the rubric needs tightening

When agreement < MIN_AGREEMENT_FOR_RELEASE, we tell the user not to trust
the automated reviewer yet and iterate on the rubric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from pr_reviewer import review
from validators.gold_standard import GoldEntry


MIN_AGREEMENT_FOR_RELEASE = 0.80


@dataclass
class EntryComparison:
    slug: str
    pr_number: int
    pr_title: str
    agreement_rate: float
    per_dimension: List[Dict[str, Any]]       # list of {dim, human, auto, agree, auto_reason}
    overall_human: str
    overall_auto: str
    overall_agree: bool


@dataclass
class CalibrationReport:
    entries: List[EntryComparison] = field(default_factory=list)

    @property
    def total_dimensions(self) -> int:
        return sum(len(e.per_dimension) for e in self.entries)

    @property
    def matching_dimensions(self) -> int:
        return sum(
            sum(1 for row in e.per_dimension if row["agree"])
            for e in self.entries
        )

    @property
    def agreement_rate(self) -> float:
        total = self.total_dimensions
        return (self.matching_dimensions / total) if total else 0.0

    def to_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for e in self.entries:
            for row in e.per_dimension:
                rows.append({
                    "pr": e.slug,
                    "dimension": row["dim"],
                    "human": row["human"],
                    "auto": row["auto"],
                    "agree": row["agree"],
                    "auto_reason": row["auto_reason"],
                })
        return pd.DataFrame(rows)

    def disagreements(self) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        return df[~df["agree"]].reset_index(drop=True)

    def summary(self) -> str:
        total = self.total_dimensions
        match = self.matching_dimensions
        rate = self.agreement_rate
        release_verdict = "READY" if rate >= MIN_AGREEMENT_FOR_RELEASE else "NOT READY"
        lines = [
            f"Gold-standard entries: {len(self.entries)}",
            f"Dimensions compared: {total}",
            f"Dimensions in agreement: {match} ({rate:.0%})",
            f"Overall-verdict agreement: "
            f"{sum(1 for e in self.entries if e.overall_agree)}/{len(self.entries)}",
            f"Release status: {release_verdict} (threshold {MIN_AGREEMENT_FOR_RELEASE:.0%})",
        ]
        return "\n".join(lines)


def calibrate(entries: List[GoldEntry]) -> CalibrationReport:
    report = CalibrationReport()
    for entry in entries:
        rubric = entry.rubric
        diff = entry.diff
        result = review(diff, rubric=rubric)

        auto_dims = {d.name: d for d in result.dimensions}
        per_dim: List[Dict[str, Any]] = []
        for dim, human in entry.human_verdicts.items():
            auto_entry = auto_dims.get(dim)
            auto = auto_entry.verdict if auto_entry else "missing"
            per_dim.append({
                "dim": dim,
                "human": human,
                "auto": auto,
                "agree": human == auto,
                "auto_reason": auto_entry.reason if auto_entry else "",
            })
        agree_count = sum(1 for row in per_dim if row["agree"])
        rate = (agree_count / len(per_dim)) if per_dim else 0.0
        human_overall = "pass" if all(v == "pass" for v in entry.human_verdicts.values()) else "fail"

        report.entries.append(EntryComparison(
            slug=entry.slug,
            pr_number=entry.pr_number,
            pr_title=entry.pr_title,
            agreement_rate=rate,
            per_dimension=per_dim,
            overall_human=human_overall,
            overall_auto=result.overall,
            overall_agree=(human_overall == result.overall),
        ))
    return report
