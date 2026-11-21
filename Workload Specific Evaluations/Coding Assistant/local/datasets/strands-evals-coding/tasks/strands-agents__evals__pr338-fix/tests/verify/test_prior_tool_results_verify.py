"""Hidden verifier for strands-agents/evals PR 338: earlier tool results in tool-level session history.

The agent does not see this file. The verifier copies it into tests/ after the agent finishes.

The tests use only the public extractor API. They do not check where in session_history a result is,
or how the results are grouped: only which tool executions a tool-level input can see.

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
    ToolExecution,
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
    return SpanInfo(session_id="s", span_id=span_id, parent_span_id=parent, start_time=start_time, end_time=end_time)


def _tool(name, start, end, result=None, tz=True):
    return ToolExecutionSpan(
        span_info=_info(name, start, end, parent="agent", tz=tz),
        tool_call=ToolCall(name=name, arguments={"step": name}, tool_call_id=f"id-{name}"),
        tool_result=ToolResult(content=result or f"result of {name}", tool_call_id=f"id-{name}"),
    )


def _agent(prompt, start=0, end=100, span_id="agent", response="done"):
    return AgentInvocationSpan(
        span_info=_info(span_id, start, end),
        user_prompt=prompt,
        agent_response=response,
        available_tools=[ToolConfig(name="a"), ToolConfig(name="b"), ToolConfig(name="c")],
    )


def _extract(*traces):
    session = Session(session_id="s", traces=list(traces))
    return {i.tool_execution_details.tool_call.name: i for i in TraceExtractor(EvaluationLevel.TOOL_LEVEL).extract(session)}


def _seen(tool_input):
    """Return the names of the tool executions in the session history of a tool-level input."""
    names = []
    for item in tool_input.session_history:
        for execution in item if isinstance(item, list) else [item]:
            if isinstance(execution, ToolExecution):
                names.append(execution.tool_call.name)
    return names


def _sequential_trace():
    """One user turn. The agent calls a, then b, then c, one after the other."""
    return Trace(trace_id="t1", session_id="s", spans=[
        _agent("plan the trip"), _tool("a", 1, 2), _tool("b", 3, 4), _tool("c", 5, 6),
    ])


def test_f2p_second_call_sees_first_result():
    inputs = _extract(_sequential_trace())
    assert "a" in _seen(inputs["b"])


def test_f2p_third_call_sees_both_earlier_results():
    inputs = _extract(_sequential_trace())
    assert sorted(_seen(inputs["c"])) == ["a", "b"]


def test_f2p_result_content_is_kept():
    trace = Trace(trace_id="t1", session_id="s", spans=[
        _agent("add then multiply"), _tool("a", 1, 2, result="42"), _tool("b", 3, 4),
    ])
    inputs = _extract(trace)
    results = [
        e.tool_result.content
        for item in inputs["b"].session_history
        for e in (item if isinstance(item, list) else [item])
        if isinstance(e, ToolExecution) and e.tool_call.name == "a"
    ]
    assert results == ["42"]


def test_f2p_unsorted_span_list_uses_time_order():
    trace = Trace(trace_id="t1", session_id="s", spans=[
        _agent("plan the trip"), _tool("c", 5, 6), _tool("a", 1, 2), _tool("b", 3, 4),
    ])
    inputs = _extract(trace)
    assert sorted(_seen(inputs["c"])) == ["a", "b"]
    assert _seen(inputs["b"]) == ["a"]
    assert _seen(inputs["a"]) == []


def test_f2p_result_that_ends_when_next_call_starts_is_seen():
    trace = Trace(trace_id="t1", session_id="s", spans=[_agent("go"), _tool("a", 1, 3), _tool("b", 3, 4)])
    assert _seen(_extract(trace)["b"]) == ["a"]


def test_f2p_naive_and_aware_timestamps():
    trace = Trace(trace_id="t1", session_id="s", spans=[
        _agent("go"), _tool("a", 1, 2, tz=False), _tool("b", 3, 4),
    ])
    assert _seen(_extract(trace)["b"]) == ["a"]


def test_f2p_second_turn_sees_its_own_earlier_call():
    first = Trace(trace_id="t1", session_id="s", spans=[_agent("turn one", 0, 10), _tool("a", 1, 2)])
    second = Trace(trace_id="t2", session_id="s", spans=[
        _agent("turn two", 20, 30, span_id="agent2"), _tool("b", 21, 22), _tool("c", 23, 24),
    ])
    assert sorted(_seen(_extract(first, second)["c"])) == ["a", "b"]


def test_p2p_first_call_sees_no_tool_results():
    assert _seen(_extract(_sequential_trace())["a"]) == []


def test_p2p_call_never_sees_itself_or_later_calls():
    inputs = _extract(_sequential_trace())
    assert "b" not in _seen(inputs["b"]) and "c" not in _seen(inputs["b"])
    assert "c" not in _seen(inputs["c"])


def test_p2p_parallel_calls_do_not_see_each_other():
    trace = Trace(trace_id="t1", session_id="s", spans=[_agent("go"), _tool("a", 1, 4), _tool("b", 2, 3)])
    inputs = _extract(trace)
    assert _seen(inputs["a"]) == []
    assert _seen(inputs["b"]) == []


def test_p2p_previous_turn_results_are_kept():
    first = Trace(trace_id="t1", session_id="s", spans=[_agent("turn one", 0, 10), _tool("a", 1, 2)])
    second = Trace(trace_id="t2", session_id="s", spans=[_agent("turn two", 20, 30, span_id="agent2"), _tool("b", 21, 22)])
    assert _seen(_extract(first, second)["b"]) == ["a"]


def test_p2p_user_prompt_comes_first():
    inputs = _extract(_sequential_trace())
    for name in ("a", "b", "c"):
        first = inputs[name].session_history[0]
        assert isinstance(first, UserMessage) and first.content[0].text == "plan the trip"


def test_p2p_one_input_per_tool_call():
    session = Session(session_id="s", traces=[_sequential_trace()])
    assert len(TraceExtractor(EvaluationLevel.TOOL_LEVEL).extract(session)) == 3
