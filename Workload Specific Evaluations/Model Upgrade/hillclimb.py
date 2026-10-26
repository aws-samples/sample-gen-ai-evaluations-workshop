"""Notebook-driven hill-climbing prompt optimizer for a model migration.

The control loop lives here so the lead notebook stays thin and readable. It is
deliberately NOT a managed service: every step is inspectable, and the mutation
step is *informed* by the judge's reasoning on where the current best prompt
loses points, so this is hill climbing rather than random search.

Seams the notebook wires in (so this file has no AWS dependency and is unit-testable):
  - `mutate_fn(incumbent, judge_reasoning, k)  -> list[str]`     propose K candidate prompts
  - `fanout_score_fn(prompt)                   -> CandidateScore` run a prompt across AgentCore
                                                                  sessions and score it

`run_hill_climb` only orchestrates: mutate → fan out+score → select → check stop rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class CandidateScore:
    """Scores for one candidate prompt, as returned by the AgentCore evaluators.

    `accuracy` is the headline session-level goal-completion rate (0..1) that the
    hill climb optimizes. `criterion_pass_rates` and `judge_reasoning` are the
    per-criterion signal used to explain *why* a score moved and to steer the next
    mutation. Latency and cost ride along so the recommendation covers all three axes.
    """

    prompt: str
    accuracy: float                                   # session-level goal-completion rate
    criterion_pass_rates: dict[str, float] = field(default_factory=dict)
    judge_reasoning: list[str] = field(default_factory=list)
    latency_p50_s: float | None = None                # per-session, whole multi-turn conversation
    latency_p95_s: float | None = None
    cost_usd_per_session: float | None = None
    n_sessions: int = 0


@dataclass
class Generation:
    index: int
    candidates: list[CandidateScore]

    @property
    def best(self) -> CandidateScore:
        return max(self.candidates, key=lambda c: c.accuracy)


@dataclass
class HillClimbResult:
    baseline: CandidateScore                          # the seed prompt, scored on the target model
    generations: list[Generation]
    stopped_reason: str

    @property
    def best(self) -> CandidateScore:
        best = self.baseline
        for gen in self.generations:
            if gen.best.accuracy > best.accuracy:
                best = gen.best
        return best

    @property
    def accuracy_gain(self) -> float:
        return self.best.accuracy - self.baseline.accuracy


class MutateFn(Protocol):
    def __call__(self, incumbent: str, judge_reasoning: list[str], k: int) -> list[str]: ...


class FanoutScoreFn(Protocol):
    def __call__(self, prompt: str) -> CandidateScore: ...


def run_hill_climb(
    *,
    seed_prompt: str,
    mutate_fn: MutateFn,
    fanout_score_fn: FanoutScoreFn,
    candidates_per_generation: int,
    max_generations: int,
    min_accuracy_gain_to_continue: float,
    on_generation: Callable[[Generation], None] | None = None,
) -> HillClimbResult:
    """Climb from `seed_prompt` toward the highest-accuracy prompt on the target model.

    Stops when a generation's best accuracy improves over the previous generation's
    best by less than `min_accuracy_gain_to_continue`, or `max_generations` is reached.
    """
    baseline = fanout_score_fn(seed_prompt)
    generations: list[Generation] = []
    incumbent = baseline
    stopped_reason = "max_generations reached"

    for gen_index in range(1, max_generations + 1):
        reasoning = _losing_reasons(incumbent)
        proposals = mutate_fn(incumbent.prompt, reasoning, candidates_per_generation)
        scored = [fanout_score_fn(p) for p in proposals]
        generation = Generation(index=gen_index, candidates=scored)
        generations.append(generation)
        if on_generation is not None:
            on_generation(generation)

        gen_best = generation.best
        gain = gen_best.accuracy - incumbent.accuracy
        if gen_best.accuracy > incumbent.accuracy:
            incumbent = gen_best
        if gain < min_accuracy_gain_to_continue:
            stopped_reason = (
                f"accuracy gain {gain:+.3f} < threshold "
                f"{min_accuracy_gain_to_continue:.3f} at generation {gen_index}"
            )
            break

    return HillClimbResult(
        baseline=baseline, generations=generations, stopped_reason=stopped_reason
    )


def _losing_reasons(candidate: CandidateScore, max_reasons: int = 12) -> list[str]:
    """The judge reasoning most worth feeding the optimizer: failures first.

    Prefers reasoning tied to the lowest-passing criteria so the next mutation
    targets where the incumbent actually loses points.
    """
    if not candidate.criterion_pass_rates:
        return candidate.judge_reasoning[:max_reasons]
    weakest = sorted(candidate.criterion_pass_rates.items(), key=lambda kv: kv[1])
    weak_names = {name for name, _ in weakest[: max(1, len(weakest) // 2)]}
    prioritized = [r for r in candidate.judge_reasoning if any(w in r for w in weak_names)]
    rest = [r for r in candidate.judge_reasoning if r not in prioritized]
    return (prioritized + rest)[:max_reasons]
