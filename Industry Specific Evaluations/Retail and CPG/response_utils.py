"""Parse model JSON and validate binary judge results before scoring."""

import json
import re
from collections.abc import Iterable
from typing import Any


def parse_json_object(
    response: str, *, context: str = "Model response"
) -> dict[str, Any]:
    """Accept a JSON object, optionally wrapped in Markdown or explanatory text.

    Malformed JSON is rejected, not repaired. Including the entire outer
    object prevents a truncated response from being mistaken for a valid
    nested object.
    """
    text = response.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = re.search(r"[\{\[]", text)
        if start is None:
            raise ValueError(
                f"{context}: no JSON object found. Response preview: {text[:400]!r}"
            ) from None
        closing = "}" if start.group() == "{" else "]"
        candidate = text[start.start() : text.rfind(closing) + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{context}: invalid or incomplete JSON ({error.msg}). "
                f"Response preview: {text[:400]!r}"
            ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{context}: expected a JSON object, got {type(parsed).__name__}. "
            f"Response preview: {text[:400]!r}"
        )
    return parsed


def parse_judge_response(
    response: str,
    expected_keys: Iterable[str],
    *,
    context: str = "Judge response",
) -> dict[str, str]:
    """Require every expected check and a lowercase pass/fail value for each."""
    score = parse_json_object(response, context=context)
    expected = set(expected_keys)
    missing = expected - score.keys()
    unexpected = score.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"{context}: incorrect judge fields; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}. Response preview: {response[:400]!r}"
        )
    invalid = {
        key: value
        for key, value in score.items()
        if not isinstance(value, str) or value not in ("pass", "fail")
    }
    if invalid:
        raise ValueError(
            f"{context}: each check must be 'pass' or 'fail'; "
            f"invalid values={invalid!r}."
        )
    return score
