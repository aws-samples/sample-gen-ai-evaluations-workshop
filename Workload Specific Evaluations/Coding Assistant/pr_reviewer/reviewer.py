"""Aligned PR reviewer using Claude on Bedrock.

The reviewer takes a diff plus an optional rubric (from ground_truth/*.md)
and produces binary pass/fail verdicts per dimension, each with a one-line
justification. The rubric is the alignment anchor: without it, the reviewer
falls back to a generic default so it can still run in CI on arbitrary PRs.

Design notes:
 - Binary per-dimension verdicts per workshop convention.
 - Sonnet for the judge (quality matters). Nova Micro is too weak for
   code review nuance based on internal comparison runs.
 - Output is JSON for programmatic use; a markdown view is available via
   ReviewResult.to_markdown() for CI PR comments.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import boto3

from .rubric import Rubric


JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MAX_DIFF_CHARS = 80_000  # truncate huge diffs to stay within context


SYSTEM_PROMPT = """You are a senior software engineer performing a code review.

You will be given:
  1. A unified diff representing a proposed change.
  2. A rubric with named review dimensions and criteria.
  3. A list of red flags (any one triggers overall fail).

For each dimension, return a binary verdict: "pass" or "fail". No middle
ground. If you're uncertain, return "fail" with a one-line reason. Also
return a list of any red flags you observed.

Output STRICT JSON with this shape — no prose, no markdown fences:

{
  "dimensions": [
    {"name": "<dim>", "verdict": "pass" | "fail", "reason": "<one line>"}
  ],
  "red_flags_hit": ["<flag>"],
  "overall": "pass" | "fail"
}

Overall is "pass" iff every dimension is "pass" AND no red flags were hit.
"""


@dataclass
class ReviewDimension:
    name: str
    verdict: str  # "pass" | "fail"
    reason: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass
class ReviewResult:
    overall: str
    dimensions: List[ReviewDimension]
    red_flags_hit: List[str] = field(default_factory=list)
    rubric_title: Optional[str] = None
    raw_response: str = ""

    @property
    def passed(self) -> bool:
        return self.overall == "pass"

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "dimensions": [asdict(d) for d in self.dimensions],
            "red_flags_hit": self.red_flags_hit,
            "rubric_title": self.rubric_title,
        }

    def to_markdown(self) -> str:
        lines = [f"# PR Review — {self.overall.upper()}"]
        if self.rubric_title:
            lines.append(f"_Rubric: {self.rubric_title}_\n")
        lines.append("| Dimension | Verdict | Reason |")
        lines.append("|---|---|---|")
        for d in self.dimensions:
            mark = "PASS" if d.passed else "FAIL"
            lines.append(f"| {d.name} | {mark} | {d.reason} |")
        if self.red_flags_hit:
            lines.append("\n**Red flags hit:**")
            for f in self.red_flags_hit:
                lines.append(f"- {f}")
        return "\n".join(lines)


def _bedrock_client():
    import os
    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client("bedrock-runtime", region_name=region)


def _build_user_prompt(diff: str, rubric: Rubric) -> str:
    dim_block = "\n".join(
        f"### {name}\n{rubric.dimension_criteria.get(name, '(no detail)')}"
        for name in rubric.dimensions
    )
    red_flags_block = "\n".join(f"- {f}" for f in rubric.red_flags) or "(none)"
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n\n[DIFF TRUNCATED]"
    return f"""# Rubric
Title: {rubric.title or "(untitled)"}

## Dimensions
{dim_block}

## Red flags
{red_flags_block}

# Diff
```diff
{diff}
```
"""


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model output, tolerant of stray prose."""
    # Fenced block first.
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    payload = m.group(1) if m else None
    if payload is None:
        # Fall back to the first balanced-looking JSON object.
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON found in judge output: {text[:200]}")
        payload = text[start:]
    return json.loads(payload)


def review(
    diff: str,
    rubric: Optional[Rubric] = None,
    model_id: str = JUDGE_MODEL_ID,
) -> ReviewResult:
    """Judge a diff against a rubric. Returns a binary per-dimension verdict."""
    rubric = rubric or Rubric.default()
    user_prompt = _build_user_prompt(diff, rubric)

    bedrock = _bedrock_client()
    response = bedrock.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0},
    )
    raw = response["output"]["message"]["content"][0]["text"]
    parsed = _extract_json(raw)

    dims = [
        ReviewDimension(
            name=d["name"],
            verdict=d.get("verdict", "fail"),
            reason=d.get("reason", ""),
        )
        for d in parsed.get("dimensions", [])
    ]
    return ReviewResult(
        overall=parsed.get("overall", "fail"),
        dimensions=dims,
        red_flags_hit=parsed.get("red_flags_hit", []),
        rubric_title=rubric.title,
        raw_response=raw,
    )


def _load_rubric_from_cli(args: argparse.Namespace) -> Optional[Rubric]:
    if args.rubric:
        return Rubric.from_path(Path(args.rubric))
    return None


def _diff_from_cli(args: argparse.Namespace) -> str:
    if args.diff:
        return Path(args.diff).read_text()
    if args.repo and args.base:
        # Used in CI: diff HEAD against a base ref.
        result = subprocess.run(
            ["git", "diff", args.base, "HEAD"],
            cwd=args.repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --diff <path>, --repo <path> + --base <ref>, or pipe a diff on stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(prog="pr_reviewer")
    parser.add_argument("--diff", help="Path to a unified diff file")
    parser.add_argument("--repo", help="Repo path (used with --base)")
    parser.add_argument("--base", help="Base ref to diff against, e.g. origin/main")
    parser.add_argument("--rubric", help="Path to a ground-truth rubric markdown file")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--model", default=JUDGE_MODEL_ID)
    args = parser.parse_args()

    rubric = _load_rubric_from_cli(args)
    diff = _diff_from_cli(args)
    result = review(diff, rubric=rubric, model_id=args.model)
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.to_markdown())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
