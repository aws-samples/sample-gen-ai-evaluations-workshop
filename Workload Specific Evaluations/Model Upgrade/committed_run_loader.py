"""Load the committed hill-climbing run into hillclimb types.

Lets the notebook read the SAME objects (CandidateScore / Generation /
HillClimbResult) whether it ran live or is reading the committed run, so Sections
6-8 render identically either way.
"""

from __future__ import annotations

import json
from pathlib import Path

from hillclimb import CandidateScore, Generation, HillClimbResult

_COMMITTED = Path(__file__).resolve().parent / "data" / "committed_run"


def _candidate(d: dict, prompt_key: str = "prompt_label") -> CandidateScore:
    return CandidateScore(
        prompt=d.get("prompt", d.get(prompt_key, "")),
        accuracy=d["accuracy"],
        criterion_pass_rates=d.get("criterion_pass_rates", {}),
        judge_reasoning=d.get("judge_reasoning", []),
        latency_p50_s=d.get("latency_p50_s"),
        latency_p95_s=d.get("latency_p95_s"),
        cost_usd_per_session=d.get("cost_usd_per_session"),
        n_sessions=d.get("n_sessions", 0),
    )


def load_committed_baseline() -> dict:
    """Return {'source': CandidateScore, 'target_unoptimized': CandidateScore, 'meta': {...}}."""
    data = json.loads((_COMMITTED / "baseline.json").read_text())
    res = data["results"]
    return {
        "source": _candidate(res["source"]),
        "target_unoptimized": _candidate(res["target_unoptimized"]),
        "meta": {k: v for k, v in data.items() if k != "results"},
    }


def load_committed_hill_climb() -> HillClimbResult:
    """Return a HillClimbResult reconstructed from the committed generations."""
    data = json.loads((_COMMITTED / "generations.json").read_text())
    baseline = CandidateScore(
        prompt=data.get("winner", {}).get("prompt", "<seed>"),
        accuracy=data["baseline_accuracy"],
    )
    # Baseline prompt is the SEED, not the winner; keep its accuracy, blank the text.
    baseline = CandidateScore(prompt="<seed prompt>", accuracy=data["baseline_accuracy"])
    generations = [
        Generation(
            index=g["index"],
            candidates=[_candidate(c) for c in g["candidates"]],
        )
        for g in data["generations"]
    ]
    return HillClimbResult(
        baseline=baseline,
        generations=generations,
        stopped_reason=data["stopped_reason"],
    )
