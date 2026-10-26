"""AgentCore custom evaluators for the Model Upgrade module.

Scoring reused from the Chatbot module, expressed as AgentCore custom evaluators so
the agents-on-Runtime-behind-Gateway path scores sessions natively:

  - goal_success: SESSION-level binary goal completion against a per-scenario
    `expected_assertion` (the headline accuracy number the hill climb optimizes).
  - binary rubric checks (F1 no invented flight, F6 no off-topic, F7 valid dates):
    OUTPUT/TRACE-level pass/fail, each returning its judge REASONING so the
    optimizer can target where the current best prompt loses points.

The judge model is held FIXED (model_config.JUDGE_MODEL) across source and target so
the score compares the two AGENT models, not two judges.

NOTE: the exact AgentCore evaluation SDK decorator/argument surface
(`custom_code_based_evaluator`, `custom_llm_as_judge_evaluator`, EvaluatorInput /
EvaluatorOutput fields) can vary by `bedrock-agentcore` version. These handlers are
written against the surface used by the AgentCore module's
`response_format_evaluator.py`; pin + verify against your installed version before a
real evaluation job.
"""

from __future__ import annotations

import json
from typing import Any

# The committed-run path and the notebook can import the rubric text + assertion
# logic without the AgentCore SDK present; the SDK is only needed to register the
# handlers as live evaluators.

RUBRIC_F1_NO_INVENTED_FLIGHTS = (
    "Binary check: does the agent reference ANY flight that was NOT returned by "
    "search_flights in this conversation (a flight number or airline not in the "
    "search results)? PASS (1.0) if every flight the agent mentioned appeared in "
    "search_flights output. FAIL (0.0) if the agent mentioned or booked any flight "
    "not present in search_flights output."
)
RUBRIC_F6_NO_OFF_TOPIC = (
    "Binary check: did the agent answer an off-topic request (writing code, giving "
    "medical/legal advice, or any topic outside travel booking)? PASS (1.0) if the "
    "agent stayed within travel booking and politely redirected any off-topic "
    "request. FAIL (0.0) if the agent produced any off-topic content."
)
RUBRIC_F7_DATE_FORMAT = (
    "Binary check: did the agent invoke any booking tool (book_flight or book_hotel) "
    "with a date argument not in YYYY-MM-DD format, or an impossible calendar date? "
    "PASS (1.0) if every booking tool call used a valid YYYY-MM-DD date. FAIL (0.0) "
    "otherwise."
)

BINARY_RUBRICS = {
    "F1_no_invented_flights": RUBRIC_F1_NO_INVENTED_FLIGHTS,
    "F6_no_off_topic": RUBRIC_F6_NO_OFF_TOPIC,
    "F7_date_format": RUBRIC_F7_DATE_FORMAT,
}


# --- Live AgentCore registration (optional import) --------------------------
# Guarded so this module imports cleanly offline (committed-run path) even when the
# AgentCore evaluation SDK is not installed.
try:
    from bedrock_agentcore.evaluation.custom_code_based_evaluators import (  # type: ignore
        EvaluatorInput,
        EvaluatorOutput,
        custom_code_based_evaluator,
    )

    _SDK = True
except Exception:  # pragma: no cover - SDK absent in offline/committed-run mode
    _SDK = False


def _extract_assertion(evaluator_input: "Any") -> str | None:
    """Pull the per-scenario expected_assertion carried in session metadata."""
    meta = getattr(evaluator_input, "metadata", None) or {}
    return meta.get("expected_assertion")


if _SDK:

    @custom_code_based_evaluator()
    def goal_success(evaluator_input: "EvaluatorInput", context) -> "EvaluatorOutput":
        """SESSION-level binary goal completion against the scenario assertion.

        This handler delegates the natural-language judgement to the fixed judge
        model via the notebook's judge helper (passed in through context), keeping
        the assertion text (human-authored, auditable) as the pass criterion. The
        returned EvaluatorOutput.explanation is the judge REASONING the hill climb
        feeds back into the next mutation.
        """
        assertion = _extract_assertion(evaluator_input)
        if not assertion:
            return EvaluatorOutput(
                value=0.0,
                label="Fail",
                explanation="No expected_assertion found on the session metadata.",
            )
        judge = context.get("judge")  # notebook injects a fixed-judge callable
        verdict = judge(session_spans=evaluator_input.session_spans, assertion=assertion)
        return EvaluatorOutput(
            value=1.0 if verdict["passed"] else 0.0,
            label="Pass" if verdict["passed"] else "Fail",
            explanation=verdict["reasoning"],
        )

    def make_binary_rubric_evaluator(name: str, rubric: str):
        """Factory: one OUTPUT-level binary evaluator per rubric, named for reporting."""

        @custom_code_based_evaluator(name=name)
        def _rubric_eval(evaluator_input: "EvaluatorInput", context) -> "EvaluatorOutput":
            judge = context.get("judge")
            verdict = judge(session_spans=evaluator_input.session_spans, rubric=rubric)
            return EvaluatorOutput(
                value=1.0 if verdict["passed"] else 0.0,
                label="Pass" if verdict["passed"] else "Fail",
                explanation=verdict["reasoning"],
            )

        return _rubric_eval

    BINARY_RUBRIC_EVALUATORS = {
        name: make_binary_rubric_evaluator(name, rubric)
        for name, rubric in BINARY_RUBRICS.items()
    }
else:  # offline / committed-run: expose names only
    BINARY_RUBRIC_EVALUATORS = {}
