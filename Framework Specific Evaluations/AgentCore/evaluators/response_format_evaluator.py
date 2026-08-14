import json
from typing import Any

from bedrock_agentcore.evaluation.custom_code_based_evaluators import (
    EvaluatorInput,
    EvaluatorOutput,
    custom_code_based_evaluator,
)


REQUIRED_TOP_LEVEL_KEYS = {"answer", "cities", "tools_used"}
RESPONSE_ATTRIBUTE_KEYS = (
    "gen_ai.response.content",
    "gen_ai.completion",
    "gen_ai.output.messages",
    "output",
)


def _candidate_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "output"):
            candidate = _candidate_text(value.get(key))
            if candidate:
                return candidate
    if isinstance(value, list):
        for item in reversed(value):
            candidate = _candidate_text(item)
            if candidate:
                return candidate
    return None


def extract_response_text(session_spans: list[dict[str, Any]]) -> str | None:
    for span in reversed(session_spans):
        attributes = span.get("attributes", {})
        for key in RESPONSE_ATTRIBUTE_KEYS:
            if key in attributes:
                candidate = _candidate_text(attributes[key])
                if candidate:
                    return candidate
    return None


def validate_response_schema(response_text: str) -> tuple[bool, str]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return False, "The final response is not valid JSON."

    if not isinstance(payload, dict):
        return False, "The final response must be a JSON object."

    missing = REQUIRED_TOP_LEVEL_KEYS - payload.keys()
    if missing:
        return False, f"Missing required keys: {', '.join(sorted(missing))}."
    if not isinstance(payload["answer"], str):
        return False, "answer must be a string."
    if not isinstance(payload["cities"], list):
        return False, "cities must be a list."
    if not isinstance(payload["tools_used"], list) or not all(
        isinstance(item, str) for item in payload["tools_used"]
    ):
        return False, "tools_used must be a list of strings."
    return True, "The final response matches the required JSON schema."


@custom_code_based_evaluator()
def handler(evaluator_input: EvaluatorInput, context) -> EvaluatorOutput:
    response_text = extract_response_text(evaluator_input.session_spans)
    if response_text is None:
        return EvaluatorOutput(
            value=0.0,
            label="Fail",
            explanation="No final response text was found in the supplied spans.",
        )

    passed, explanation = validate_response_schema(response_text)
    return EvaluatorOutput(
        value=1.0 if passed else 0.0,
        label="Pass" if passed else "Fail",
        explanation=explanation,
    )

