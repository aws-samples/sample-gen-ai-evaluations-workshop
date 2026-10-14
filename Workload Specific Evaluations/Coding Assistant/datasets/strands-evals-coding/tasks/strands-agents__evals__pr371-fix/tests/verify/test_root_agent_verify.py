"""Hidden verifier for strands-agents/evals PR 371: the root agent of a flat multi-agent trace.

The agent does not see this file. The verifier copies it into tests/ after the agent finishes.

The tests use only the public API (TraceExtractor and Trace). They check which agent the tool-level
inputs are anchored to: its user prompt, its tools, and the agent_span_id of tool spans without an owner.

Test names start with test_f2p_ (fail on the base, must pass after the fix) or test_p2p_ (pass on the
base, must still pass after the fix).
"""

from datetime import datetime, timedelta, timezone

from strands_evals.extractors.trace_extractor import TraceExtractor
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
    UserMessage,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _info(span_id, start, end, parent=None, tz=True):
    start_time, end_time = T0 + timedelta(seconds=start), T0 + timedelta(seconds=end)
    if not tz:
        start_time, end_time = start_time.replace(tzinfo=None), end_time.replace(tzinfo=None)
    return SpanInfo(session_id="s", trace_id="t", span_id=span_id, parent_span_id=parent,
                    start_time=start_time, end_time=end_time)


def _agent(span_id, start, end, prompt="", response="", tools=(), parent=None, tz=True):
    return AgentInvocationSpan(
        span_info=_info(span_id, start, end, parent=parent, tz=tz),
        user_prompt=prompt,
        agent_response=response,
        available_tools=[ToolConfig(name=t) for t in tools],
    )


def _orphan_tool(name, start, end):
    """A tool span with no parent and no agent_span_id. It falls back to the root agent."""
    return ToolExecutionSpan(
        span_info=_info(f"tool-{name}", start, end),
        tool_call=ToolCall(name=name, arguments={}),
        tool_result=ToolResult(content="ok"),
    )


COORDINATOR_PROMPT = "Weather in New York and London, then calculate the difference."


def _coordinator(start=0, end=20):
    return _agent("coord", start, end, prompt=COORDINATOR_PROMPT, response="18F difference.",
                  tools=("ask_weather", "calculate"))


def _sub_agent(span_id="sub", start=5, end=10, prompt="Weather in London"):
    return _agent(span_id, start, end, prompt=prompt, response="50F", tools=("get_weather",))


def _tool_inputs(spans):
    trace = Trace(spans=spans, trace_id="t", session_id="s")
    return TraceExtractor(EvaluationLevel.TOOL_LEVEL).extract(Session(traces=[trace], session_id="s"))


def _first_prompt(tool_input):
    first = tool_input.session_history[0]
    assert isinstance(first, UserMessage)
    return first.content[0].text


def test_f2p_sub_agent_listed_first_is_not_root():
    inputs = _tool_inputs([_sub_agent(), _coordinator(), _orphan_tool("calculate", 11, 12)])
    assert len(inputs) == 1
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT


def test_f2p_orphan_tool_is_owned_by_coordinator():
    inputs = _tool_inputs([_sub_agent(), _coordinator(), _orphan_tool("calculate", 11, 12)])
    assert inputs[0].tool_execution_details.agent_span_id == "coord"


def test_f2p_orphan_tool_gets_coordinator_tools():
    inputs = _tool_inputs([_sub_agent(), _coordinator(), _orphan_tool("calculate", 11, 12)])
    assert [t.name for t in inputs[0].available_tools] == ["ask_weather", "calculate"]


def test_f2p_coordinator_listed_last_of_three():
    spans = [_sub_agent("sub_a", 2, 6, "Weather in London"), _sub_agent("sub_b", 3, 7, "Weather in New York"),
             _coordinator(), _orphan_tool("calculate", 11, 12)]
    inputs = _tool_inputs(spans)
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT
    assert inputs[0].tool_execution_details.agent_span_id == "coord"


def test_f2p_all_agents_empty_earliest_parentless_is_root():
    spans = [_agent("late", 5, 6, tools=("late_tool",)), _agent("early", 0, 10, tools=("early_tool",)),
             _orphan_tool("x", 7, 8)]
    inputs = _tool_inputs(spans)
    assert inputs[0].tool_execution_details.agent_span_id == "early"


def test_f2p_mixed_naive_and_aware_start_times():
    spans = [_sub_agent(start=5, end=10), _agent("coord", 0, 20, prompt=COORDINATOR_PROMPT, tz=False,
                                                  tools=("ask_weather", "calculate")),
             _orphan_tool("calculate", 11, 12)]
    inputs = _tool_inputs(spans)
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT


def test_p2p_coordinator_listed_first():
    inputs = _tool_inputs([_coordinator(), _sub_agent(), _orphan_tool("calculate", 11, 12)])
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT
    assert inputs[0].tool_execution_details.agent_span_id == "coord"


def test_p2p_agent_with_content_beats_earlier_empty_agent():
    spans = [_agent("empty", 0, 30), _coordinator(start=1), _orphan_tool("calculate", 11, 12)]
    inputs = _tool_inputs(spans)
    assert inputs[0].tool_execution_details.agent_span_id == "coord"


def test_p2p_parentless_agent_beats_earlier_child_agent():
    spans = [_agent("child", 0, 5, prompt="child task", parent="elsewhere"), _coordinator(start=1),
             _orphan_tool("calculate", 11, 12)]
    inputs = _tool_inputs(spans)
    assert inputs[0].tool_execution_details.agent_span_id == "coord"


def test_p2p_nested_trace_tool_keeps_its_parent_agent():
    sub = _agent("sub", 5, 10, prompt="Weather in London", parent="coord", tools=("get_weather",))
    tool = ToolExecutionSpan(
        span_info=_info("tool-london", 6, 7, parent="sub"),
        tool_call=ToolCall(name="get_weather", arguments={"city": "London"}),
        tool_result=ToolResult(content="50F"),
    )
    inputs = _tool_inputs([sub, _coordinator(), tool])
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT
    assert inputs[0].tool_execution_details.agent_span_id == "sub"


def test_p2p_single_agent_trace():
    inputs = _tool_inputs([_coordinator(), _orphan_tool("calculate", 11, 12)])
    assert _first_prompt(inputs[0]) == COORDINATOR_PROMPT
