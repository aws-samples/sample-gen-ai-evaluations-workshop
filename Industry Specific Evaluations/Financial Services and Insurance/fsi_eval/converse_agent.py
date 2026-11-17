"""A thin Amazon Bedrock Converse wrapper with a tool-use loop.

``run_conversation`` plays a list of customer turns against one agent
configuration. Every tool the model asks for is executed through the recording
gateway in ``systems_of_record``, so the conversation leaves an inspectable
state store behind. ``converse_text`` is a single call without tools, used by
judges and classifiers.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable

import boto3
from botocore.config import Config

from .systems_of_record import Session, StateStore, SystemsOfRecord

# Nova models emit their reasoning in <thinking> tags inside text blocks.
_THINKING = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def make_bedrock_client(region: str | None = None):
    """Bedrock runtime client with timeouts and adaptive retries for evaluation runs."""
    return boto3.client(
        "bedrock-runtime",
        region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(connect_timeout=5, read_timeout=120, retries={"max_attempts": 6, "mode": "adaptive"}),
    )


def greedy_decoding_fields(model_id: str) -> dict | None:
    """Extra request fields for greedy decoding (topK=1), recommended for Nova tool use."""
    return {"inferenceConfig": {"topK": 1}} if "amazon.nova" in model_id else None


@dataclass(frozen=True)
class AgentConfig:
    """What the agent under test is: model, instructions and loop limits."""

    name: str  # behaviour name, e.g. "session_bound_identity"
    model_id: str
    system_prompt: str
    max_tool_rounds: int = 6
    max_tokens: int = 1024
    temperature: float = 0.0
    additional_model_request_fields: dict | None = None
    # Earlier turns the model sees on each call; None = the whole conversation.
    # A short window stands in for the context limits of a production deployment.
    history_turns: int | None = None


@dataclass
class TurnRecord:
    """One customer turn and the agent's reply to it."""

    turn_index: int
    user_text: str
    assistant_text: str  # user-visible text, <thinking> removed; "" means a silent turn
    raw_assistant_text: str
    tool_call_seqs: list
    started_monotonic: float
    finished_monotonic: float
    model_calls: int
    stop_reason: str  # Bedrock stopReason, "max_tool_rounds", "model_error", or "scripted"
    scripted: bool


@dataclass
class ConversationResult:
    """Everything one conversation produced: turns, the gateway's state store and the messages."""

    case_id: str
    config_name: str
    session: Session
    turns: list
    store: StateStore
    messages: list = field(default_factory=list)


def visible_text(content_blocks: list) -> tuple:
    """Returns (visible, raw) text of a message; visible drops <thinking> blocks."""
    raw = "\n".join(block["text"] for block in content_blocks if "text" in block).strip()
    visible = _THINKING.sub("", raw).strip()
    return visible, raw


def tool_results_for(content_blocks: list, systems: SystemsOfRecord, session: Session, turn_index: int) -> list:
    """Executes every toolUse block through the gateway and returns Converse toolResult blocks."""
    results = []
    for block in content_blocks:
        if "toolUse" not in block:
            continue
        use = block["toolUse"]
        result = systems.execute(session, use["name"], use.get("input") or {}, turn_index)
        results.append({"toolResult": {
            "toolUseId": use["toolUseId"],
            "content": [{"json": result}],
            "status": "error" if result.get("status") == "error" else "success",
        }})
    return results


def scripted_exchange(user_text: str, assistant_text: str) -> list:
    """Two Converse messages for a pre-written exchange; no model call is made."""
    return [{"role": "user", "content": [{"text": user_text}]},
            {"role": "assistant", "content": [{"text": assistant_text}]}]


def _append_user_content(messages: list, content: list) -> int:
    """Adds user content, merging into the last message when it is already a user message.

    Converse needs roles to alternate; a turn cut off at max_tool_rounds ends on
    a user toolResult message. Returns the index of the message that holds the content.
    """
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"].extend(content)
    else:
        messages.append({"role": "user", "content": list(content)})
    return len(messages) - 1


def messages_in_window(messages: list, turn_starts: list, history_turns: int | None) -> list:
    """The messages a model sees: the current turn plus ``history_turns`` earlier turns.

    ``turn_starts`` holds the message index where each turn began. ``None``
    keeps the whole conversation.
    """
    if history_turns is None or history_turns + 1 >= len(turn_starts):
        return messages
    window = messages[turn_starts[-(history_turns + 1)]:]
    first = window[0]
    # A toolResult whose toolUse fell outside the window would be rejected by Converse.
    kept = [block for block in first["content"] if "toolResult" not in block]
    if len(kept) != len(first["content"]):
        window = [{"role": first["role"], "content": kept}] + window[1:]
    return window


def _converse(client, agent: AgentConfig, messages: list, tool_config: dict) -> dict:
    """One Converse call with the agent's settings."""
    request = {
        "modelId": agent.model_id,
        "system": [{"text": agent.system_prompt}],
        "messages": messages,
        "toolConfig": tool_config,  # sent on every call: required once history holds toolUse blocks
        "inferenceConfig": {"maxTokens": agent.max_tokens, "temperature": agent.temperature},
    }
    if agent.additional_model_request_fields:
        request["additionalModelRequestFields"] = agent.additional_model_request_fields
    return client.converse(**request)


def run_conversation(client, agent: AgentConfig, systems: SystemsOfRecord, session: Session,
                     turns: list, case_id: str) -> ConversationResult:
    """Plays ``turns`` against the agent and returns the conversation record.

    ``turns`` is a list of ``{"user": str, "scripted_assistant": str | None}``.
    Live turns call Converse in a tool-use loop of up to ``max_tool_rounds``
    rounds; scripted turns are appended without a model call. Every live
    message is passed to ``systems.persist_message`` (the log sink). The full
    history is kept in the result; each call sends only the agent's window.
    """
    tool_config = systems.tool_config()
    messages: list = []
    turn_starts: list = []  # message index where each turn began
    records = []
    for turn_index, turn in enumerate(turns):
        user_text = turn["user"]
        started = time.monotonic()

        # Scripted padding: history only, no model call and no gateway activity.
        if turn.get("scripted_assistant") is not None:
            exchange = scripted_exchange(user_text, turn["scripted_assistant"])
            turn_starts.append(_append_user_content(messages, exchange[0]["content"]))
            messages.append(exchange[1])
            records.append(TurnRecord(turn_index, user_text, turn["scripted_assistant"],
                                      turn["scripted_assistant"], [], started, time.monotonic(), 0,
                                      "scripted", True))
            continue

        systems.persist_message(session, "user", user_text)
        turn_starts.append(_append_user_content(messages, [{"text": user_text}]))
        first_seq = len(systems.store.tool_calls)
        visible_parts, raw_parts = [], []
        model_calls, rounds = 0, 0
        while True:
            try:
                window = messages_in_window(messages, turn_starts, agent.history_turns)
                response = _converse(client, agent, window, tool_config)
            except client.exceptions.ModelErrorException:
                # Nova can emit an invalid tool-use sequence. The turn ends with no reply,
                # which the checks then count as a silent turn: it is an agent failure.
                stop_reason = "model_error"
                break
            model_calls += 1
            message = response["output"]["message"]
            messages.append(message)
            visible, raw = visible_text(message["content"])
            if visible:
                visible_parts.append(visible)
            if raw:
                raw_parts.append(raw)
            stop_reason = response["stopReason"]
            if stop_reason != "tool_use":
                break
            # Tool round: execute through the gateway and send the results back.
            _append_user_content(messages, tool_results_for(message["content"], systems, session, turn_index))
            rounds += 1
            if rounds >= agent.max_tool_rounds:
                stop_reason = "max_tool_rounds"
                break

        assistant_text = "\n".join(visible_parts).strip()
        systems.persist_message(session, "assistant", assistant_text)
        seqs = [call.seq for call in systems.store.tool_calls[first_seq:]]
        records.append(TurnRecord(turn_index, user_text, assistant_text, "\n".join(raw_parts), seqs,
                                  started, time.monotonic(), model_calls, stop_reason, False))
    return ConversationResult(case_id=case_id, config_name=agent.name, session=session, turns=records,
                              store=systems.store, messages=messages)


def converse_reply(client, model_id: str, system_prompt: str, user_text: str, max_tokens: int = 512,
                   temperature: float | None = None, additional_model_request_fields: dict | None = None) -> tuple:
    """Single-shot Converse call without tools; returns (visible text, stopReason) of the reply.

    The stopReason tells a caller why a reply holds no text, for example
    "max_tokens" when a model's reasoning used the whole output budget.
    """
    inference = {"maxTokens": max_tokens}
    if temperature is not None:
        inference["temperature"] = temperature
    request = {"modelId": model_id, "system": [{"text": system_prompt}],
               "messages": [{"role": "user", "content": [{"text": user_text}]}],
               "inferenceConfig": inference}
    if additional_model_request_fields:
        request["additionalModelRequestFields"] = additional_model_request_fields
    response = client.converse(**request)
    return visible_text(response["output"]["message"]["content"])[0], response.get("stopReason", "")


def converse_text(client, model_id: str, system_prompt: str, user_text: str, max_tokens: int = 512,
                  temperature: float | None = None, additional_model_request_fields: dict | None = None) -> str:
    """Single-shot Converse call without tools; returns the visible text of the reply."""
    return converse_reply(client, model_id, system_prompt, user_text, max_tokens, temperature,
                          additional_model_request_fields)[0]


def map_in_order(function: Callable, items: Iterable, max_workers: int = 4) -> list:
    """Applies ``function`` to every item on a thread pool and returns results in input order.

    Conversations are independent (one SystemsOfRecord each), so they can run
    side by side to keep notebook run time down.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(function, items))
