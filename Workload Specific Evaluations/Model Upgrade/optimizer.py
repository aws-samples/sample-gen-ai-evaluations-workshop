"""LLM-authored prompt mutation for the hill climb.

The optimizer model (model_config.OPTIMIZER_MODEL) proposes K candidate system
prompts, each a MUTATION of the current incumbent, explicitly informed by the
judge's reasoning on where the incumbent lost points. This is what makes the search
hill climbing rather than random: the optimizer is told the failures and asked to
fix them without regressing what already works.

The notebook injects a `complete(prompt) -> str` callable bound to the fixed
optimizer model (routed through the same Gateway seam). This module is pure string
plumbing so it is unit-testable offline.
"""

from __future__ import annotations

import json
import re
from typing import Callable

META_PROMPT = """\
You are optimizing the SYSTEM PROMPT of a travel-booking assistant so it scores
higher on an automated multi-turn evaluation. You will propose {k} distinct
candidate system prompts.

The assistant is scored on:
- session-level GOAL COMPLETION (did it finish the user's task correctly), and
- binary rubric checks: never invent a flight not in search results; never go
  off-topic; always use valid YYYY-MM-DD dates; confirm details before booking.

CURRENT BEST SYSTEM PROMPT (the incumbent):
\"\"\"
{incumbent}
\"\"\"

WHERE THE INCUMBENT LOSES POINTS (judge reasoning on failing cases):
{reasoning}

Write {k} improved candidate system prompts. Each candidate must:
- keep everything that already works (confirm-before-book, flights+hotels scope);
- directly address the failures above with concrete, testable instructions;
- stay a SYSTEM PROMPT (instructions to the assistant), not a description;
- differ meaningfully from each other — explore different fixes, do not paraphrase.

Return ONLY a JSON array of {k} strings, nothing else. Example:
["candidate prompt one", "candidate prompt two"]
"""


def build_meta_prompt(incumbent: str, judge_reasoning: list[str], k: int) -> str:
    reasoning_block = (
        "\n".join(f"- {r}" for r in judge_reasoning)
        if judge_reasoning
        else "- (no specific failures recorded; push for robustness and clarity)"
    )
    return META_PROMPT.format(k=k, incumbent=incumbent.strip(), reasoning=reasoning_block)


def _parse_candidates(text: str, k: int) -> list[str]:
    """Parse the optimizer's JSON array; tolerant of fenced code blocks."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list):
            out = [str(p).strip() for p in arr if str(p).strip()]
            if out:
                return out[:k]
    except json.JSONDecodeError:
        pass
    # Fallback: split on blank lines if the model ignored the JSON instruction.
    chunks = [c.strip() for c in re.split(r"\n\s*\n", cleaned) if c.strip()]
    return chunks[:k]


def make_mutate_fn(complete: Callable[[str], str]):
    """Return a MutateFn for hillclimb.run_hill_climb, bound to the optimizer model.

    `complete` is a single-call text completion against the FIXED optimizer model,
    routed through the Gateway. The returned function matches hillclimb.MutateFn.
    """

    def mutate(incumbent: str, judge_reasoning: list[str], k: int) -> list[str]:
        meta = build_meta_prompt(incumbent, judge_reasoning, k)
        raw = complete(meta)
        candidates = _parse_candidates(raw, k)
        # Guarantee K candidates: if the optimizer under-delivered, pad with the
        # incumbent so a generation is never short (it simply won't beat itself).
        while len(candidates) < k:
            candidates.append(incumbent)
        return candidates

    return mutate
