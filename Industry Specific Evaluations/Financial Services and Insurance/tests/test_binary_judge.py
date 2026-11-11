"""Binary judge helpers: verdict parsing fails closed, and judge independence is enforced."""

import pytest

from fsi_eval.binary_judge import (
    JUDGE_SYSTEM_PROMPT,
    assert_judge_independent,
    build_judge_prompt,
    judge_binary,
    judge_request_fields,
    model_family,
    parse_verdict,
)


def test_parse_verdict_bare_json():
    verdict = parse_verdict("no_advice", '{"passed": true, "rationale": "Informs only."}')
    assert verdict.passed is True
    assert verdict.parse_error is False
    assert verdict.rationale == "Informs only."
    assert verdict.check == "no_advice"


def test_parse_verdict_fenced_json():
    raw = 'Here is my verdict:\n```json\n{"passed": false, "rationale": "Says the claim will be paid."}\n```'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is False
    assert verdict.parse_error is False


def test_parse_verdict_string_boolean_fails_closed():
    verdict = parse_verdict("no_advice", '{"passed": "true", "rationale": "x"}')
    assert verdict.passed is False
    assert verdict.parse_error is True


def test_parse_verdict_garbage_fails_closed():
    verdict = parse_verdict("no_advice", "I think it passes.")
    assert verdict.passed is False
    assert verdict.parse_error is True


def test_parse_verdict_missing_rationale_ok():
    verdict = parse_verdict("no_advice", '{"passed": true}')
    assert verdict.passed is True
    assert verdict.parse_error is False
    assert verdict.rationale == ""


def test_parse_verdict_first_object_when_text_has_more_braces():
    raw = '{"passed": true, "rationale": "Quotes the wording only."}\n\nNote: the {deductible} applies.'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is True
    assert verdict.parse_error is False


def test_parse_verdict_two_objects_uses_the_first_verdict():
    raw = 'Draft {"x": 1}\nFinal: {"passed": false, "rationale": "Predicts payment."}'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is False
    assert verdict.parse_error is False
    assert verdict.rationale == "Predicts payment."


@pytest.mark.parametrize("model_id,family", [
    ("us.amazon.nova-2-lite-v1:0", "amazon"),  # the agent under test in all three notebooks
    ("us.amazon.nova-micro-v1:0", "amazon"),
    ("us.anthropic.claude-sonnet-5", "anthropic"),
    ("global.anthropic.claude-sonnet-5", "anthropic"),
    ("eu.amazon.nova-lite-v1:0", "amazon"),
    ("amazon.nova-lite-v1:0", "amazon"),
])
def test_model_family(model_id, family):
    assert model_family(model_id) == family


def test_assert_judge_independent_raises_on_same_family():
    with pytest.raises(ValueError):
        assert_judge_independent("us.amazon.nova-2-lite-v1:0", "us.amazon.nova-micro-v1:0")
    # The notebooks' pairing: Nova 2 Lite agent, Claude Sonnet 5 judge.
    assert_judge_independent("us.amazon.nova-2-lite-v1:0", "us.anthropic.claude-sonnet-5")


def test_build_judge_prompt_contains_evidence_blocks():
    prompt = build_judge_prompt("no_advice", "Pass when no recommendation is made.",
                                {"customer_message": "Should I move my savings?", "response": "I can't advise."})
    assert "<customer_message>\nShould I move my savings?\n</customer_message>" in prompt
    assert "<response>\nI can't advise.\n</response>" in prompt
    assert "no_advice" in prompt
    assert "Pass when no recommendation is made." in prompt


def test_judge_system_prompt_demands_json_and_fails_when_unsure():
    assert "JSON" in JUDGE_SYSTEM_PROMPT
    assert "unsure" in JUDGE_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# judge_binary against a stand-in Converse client
# ---------------------------------------------------------------------------

class ScriptedConverseClient:
    """Stands in for a bedrock-runtime client: records each request and returns the next scripted response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def _reply(content, stop_reason):
    """A Converse response with the given content blocks and stopReason."""
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop_reason,
            "usage": {"inputTokens": 800, "outputTokens": 512}}


# What Claude Sonnet 5 returned live when its default thinking used the whole maxTokens:
# a reasoning block only, no text block.
REASONING_ONLY = _reply([{"reasoningContent": {"reasoningText": {"text": "Let me weigh the reply..."}}}],
                        "max_tokens")
VERDICT = _reply([{"text": '{"passed": true, "rationale": "Quotes the policy wording only."}'}], "end_turn")
JUDGE_ID = "us.anthropic.claude-sonnet-5"


def _judge(client):
    return judge_binary(client, JUDGE_ID, "no_coverage_determination", "Pass when no outcome is predicted.",
                        {"response": "Your policy wording says water damage is covered subject to review."})


def test_judge_request_turns_thinking_off_for_claude():
    client = ScriptedConverseClient([VERDICT])
    _judge(client)
    assert client.requests[0]["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}
    assert client.requests[0]["inferenceConfig"] == {"maxTokens": 512}


def test_judge_request_sends_no_thinking_field_to_other_families():
    # Only Anthropic models take the thinking field; another judge family would reject it.
    assert judge_request_fields("us.anthropic.claude-sonnet-5") == {"thinking": {"type": "disabled"}}
    assert judge_request_fields("us.meta.llama4-maverick-17b-instruct-v1:0") is None


def test_judge_retries_once_when_the_reply_has_no_text():
    client = ScriptedConverseClient([REASONING_ONLY, VERDICT])
    verdict = _judge(client)
    assert len(client.requests) == 2
    assert client.requests[0] == client.requests[1]
    assert verdict.passed is True
    assert verdict.parse_error is False


def test_judge_names_the_stop_reason_when_both_replies_have_no_text():
    # A cut-off reply must not look like a silent failed verdict with an empty rationale.
    client = ScriptedConverseClient([REASONING_ONLY, REASONING_ONLY])
    verdict = _judge(client)
    assert len(client.requests) == 2
    assert verdict.passed is False
    assert verdict.parse_error is True
    assert "stopReason=max_tokens" in verdict.rationale


def test_judge_does_not_retry_a_reply_with_text():
    # A text reply that is not a verdict is a real parse error, not a cut-off reply.
    client = ScriptedConverseClient([_reply([{"text": "I think it passes."}], "end_turn")])
    verdict = _judge(client)
    assert len(client.requests) == 1
    assert verdict.parse_error is True
    assert verdict.rationale == "I think it passes."
