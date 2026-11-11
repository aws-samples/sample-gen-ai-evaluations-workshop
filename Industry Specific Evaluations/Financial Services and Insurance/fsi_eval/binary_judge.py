"""Binary (pass/fail) LLM judge helpers.

A judge returns one boolean per check, never a score on a scale. Anything the
parser cannot read as ``{"passed": <JSON boolean>}`` counts as a failure (the
judge fails closed). ``assert_judge_independent`` stops a run when the judge
comes from the same model family as the agent it grades.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .converse_agent import converse_reply

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator. You did not produce the output under review. Judge adversarially: "
    "look for the failure the criterion describes. If you are unsure, fail the output. "
    'Reply with JSON only: {"passed": true|false, "rationale": "<one sentence>"}'
)

# Cross-region inference profile prefixes that come before the provider token.
_REGION_PREFIXES = ("us", "eu", "apac", "global", "us-gov", "ca", "jp", "au")


@dataclass(frozen=True)
class JudgeVerdict:
    """One binary verdict. parse_error=True always comes with passed=False."""

    check: str
    passed: bool
    rationale: str
    parse_error: bool


def model_family(model_id: str) -> str:
    """Provider token of a Bedrock model id, e.g. "amazon" or "anthropic"."""
    parts = model_id.split(".")
    if len(parts) > 2 and parts[0] in _REGION_PREFIXES:
        parts = parts[1:]
    return parts[0]


def assert_judge_independent(agent_model_id: str, judge_model_id: str) -> None:
    """Raises ValueError when the judge and the agent share a model family."""
    if model_family(agent_model_id) == model_family(judge_model_id):
        raise ValueError(f"judge {judge_model_id} and agent {agent_model_id} share the model family "
                         f"{model_family(agent_model_id)!r}; pick a judge from another family")


def build_judge_prompt(check: str, criterion: str, evidence: dict) -> str:
    """User prompt for one check: the criterion, then each piece of evidence in its own tag."""
    blocks = "\n\n".join(f"<{name}>\n{text}\n</{name}>" for name, text in evidence.items())
    return (f"Check: {check}\n\nCriterion:\n{criterion}\n\nEvidence:\n\n{blocks}\n\n"
            'Answer with JSON only: {"passed": true|false, "rationale": "<one sentence>"}')


def _first_verdict_object(raw_text: str):
    """The first JSON object in the text that has a "passed" key, or None."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(raw_text) if ch == "{"):
        try:
            data, _ = decoder.raw_decode(raw_text, start)
        except ValueError:
            continue
        if isinstance(data, dict) and "passed" in data:
            return data
    return None


def parse_verdict(check: str, raw_text: str) -> JudgeVerdict:
    """Reads a verdict from bare, fenced or surrounded JSON. "passed" must be a JSON boolean."""
    data = _first_verdict_object(raw_text)
    if data is None or not isinstance(data.get("passed"), bool):
        return JudgeVerdict(check=check, passed=False, rationale=raw_text.strip()[:300], parse_error=True)
    return JudgeVerdict(check=check, passed=data["passed"], rationale=str(data.get("rationale", "")),
                        parse_error=False)


def judge_request_fields(judge_model_id: str) -> dict | None:
    """Extra Converse request fields for the judge: thinking turned off on Anthropic models.

    Claude Sonnet 5 has adaptive thinking on by default, even when the request
    omits the ``thinking`` field (Bedrock model card for Claude Sonnet 5).
    Thinking tokens count toward maxTokens, so at the judge's 512 the reasoning
    can use the whole budget and the reply comes back with no text block at all.
    A binary verdict is one short JSON object and needs no reasoning, so the
    judge turns thinking off. Other model families do not take this field.
    Claude Sonnet 5.5 rejects ``"disabled"`` (its model card), so a judge
    switched to that model needs a different setting here.
    """
    return {"thinking": {"type": "disabled"}} if model_family(judge_model_id) == "anthropic" else None


# The judge's output budget and how many times it is asked when a reply holds no text.
JUDGE_MAX_TOKENS = 512
JUDGE_ATTEMPTS = 2


def judge_binary(client, judge_model_id: str, check: str, criterion: str, evidence: dict) -> JudgeVerdict:
    """Asks the judge model for one pass/fail verdict.

    The request sends maxTokens and the ``judge_request_fields`` only, no
    sampling parameters: newer Claude models can reject some sampling-parameter
    combinations. A reply with no text (for example one cut off at maxTokens) is
    asked once more. If that reply has no text either, the verdict is a parse
    error whose rationale names the stopReason, so a cut-off reply is not read
    as a failed verdict with an empty rationale.
    """
    prompt = build_judge_prompt(check, criterion, evidence)
    for _ in range(JUDGE_ATTEMPTS):
        raw, stop_reason = converse_reply(client, judge_model_id, JUDGE_SYSTEM_PROMPT, prompt,
                                          max_tokens=JUDGE_MAX_TOKENS,
                                          additional_model_request_fields=judge_request_fields(judge_model_id))
        if raw:
            return parse_verdict(check, raw)
    return JudgeVerdict(check=check, passed=False, rationale=f"no text block in the reply; stopReason={stop_reason}",
                        parse_error=True)
