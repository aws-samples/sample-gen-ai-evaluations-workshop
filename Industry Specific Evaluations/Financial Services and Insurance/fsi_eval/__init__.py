"""Shared code for the FSI evaluation notebooks.

Three parts, one module each:

* ``systems_of_record``: mock systems of record behind a recording tool
  gateway, with an inspectable state store.
* ``converse_agent``: a thin Amazon Bedrock Converse wrapper with a tool-use loop.
* ``binary_judge``: pass/fail LLM judge helpers that fail closed.
"""

from .systems_of_record import (
    SERVICING_TOOLS,
    GatewaySettings,
    Session,
    StateStore,
    SystemsOfRecord,
    ToolCall,
    ToolDefinition,
    WriteRecord,
    is_explicit_assent,
    load_fixtures,
    queue_routing_policy,
    redact_pii,
    write_handoff_record,
)
from .converse_agent import (
    AgentConfig,
    ConversationResult,
    TurnRecord,
    converse_text,
    greedy_decoding_fields,
    make_bedrock_client,
    map_in_order,
    messages_in_window,
    run_conversation,
    scripted_exchange,
    tool_results_for,
    visible_text,
)
from .binary_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    assert_judge_independent,
    build_judge_prompt,
    judge_binary,
    model_family,
    parse_verdict,
)
