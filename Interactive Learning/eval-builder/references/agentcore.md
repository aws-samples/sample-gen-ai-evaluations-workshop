# AgentCore — reference

## What it evaluates

Agents deployed on **AgentCore Runtime**, using AWS-native observability. Evaluation flows from
three layers without instrumenting agent code: (1) **tool selection accuracy** extracted from
CloudWatch traces, (2) **binary LLM-as-judge** quality on the agent's response, and (3) **native
AgentCore evaluators** via the Evaluations API. Each invocation is tagged with a unique session ID
(≥33 chars) so its trace is retrievable from CloudWatch.

## Metrics (definitions + how computed)

- **Tool precision / recall / F1** — from tools extracted out of the CloudWatch trace vs. expected
  tools for the test case. Same formulas as tool-calling.
- **Quality checks (binary)** — helpfulness, accuracy, clarity, completeness, each judged pass/fail
  by an LLM (see below), never on a 1–5 scale.
- **Native evaluator verdicts** — `Builtin.Helpfulness`, `Builtin.Accuracy` return a value + label
  per session; treat as pass/fail via a documented threshold if you fold them into the aggregate.

Logs live at `/aws/bedrock-agentcore/runtimes/{agent_id}-{qualifier}`; filter by session ID.

## Binary judge template(s)

**Binary, one failure mode per judge**, returning `{"passed": true|false, "reason": "..."}`. Replaces
the 1–5 multi-dimensional rubric — one judge per quality dimension.

```python
import boto3, json
bedrock = boto3.client("bedrock-runtime")
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # from eval_config.yaml

def _judge(query: str, response: str, question: str) -> dict:
    prompt = f"""Evaluate this agent response.
Query: {query}
Response: {response}

{question}
Return ONLY JSON: {{"passed": true|false, "reason": "<one sentence>"}}"""
    result = bedrock.invoke_model(
        modelId=JUDGE_MODEL,
        body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 500,
                         "messages": [{"role": "user", "content": prompt}]}))
    text = json.loads(result["body"].read())["content"][0]["text"]
    try:
        return json.loads(text[text.find("{"):text.rfind("}") + 1])
    except json.JSONDecodeError:
        return {"passed": False, "reason": f"unparseable judge output: {text[:200]}"}

def judge_helpfulness(query, response) -> dict:
    """FAIL if the response does not help the user accomplish their goal."""
    return _judge(query, response,
        "Question: Does the response actually help the user accomplish their goal? "
        "PASS only if it is genuinely helpful.")

def judge_accuracy(query, response) -> dict:
    """FAIL if the response contains a factual error."""
    return _judge(query, response,
        "Question: Is the response factually correct with no errors? "
        "PASS only if every claim is accurate.")

def judge_clarity(query, response) -> dict:
    """FAIL if the response is confusing or hard to follow."""
    return _judge(query, response,
        "Question: Is the response clear and easy to understand? "
        "PASS only if it is unambiguous and well-structured.")

def judge_completeness(query, response) -> dict:
    """FAIL if the response omits part of what was asked."""
    return _judge(query, response,
        "Question: Does the response fully address every part of the request? "
        "PASS only if nothing asked-for is missing.")
```

Deterministic tool-selection gate (binary, $0 once the trace is fetched):

```python
def check_tool_selection(expected: list, actual: list) -> dict:
    """FAIL if extracted tool usage doesn't match expected (F1 < 1.0)."""
    exp, act = set(expected), set(actual)
    if not exp and not act:
        return {"passed": True, "reason": "no tools expected or called"}
    tp = len(exp & act)
    prec = tp / len(act) if act else 0.0
    rec = tp / len(exp) if exp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"passed": f1 == 1.0, "reason": f"tool F1={f1:.2f} (expected {sorted(exp)}, got {sorted(act)})"}
```

## Aggregation

- **Per-check pass rate** for `tool_selection`, `helpfulness`, `accuracy`, `clarity`, `completeness`.
- **All-checks-pass rate** = share of sessions where every check passes.
- Optionally include native-evaluator verdicts as additional binary checks (threshold on `value`).

```python
CHECKS = ["tool_selection", "helpfulness", "accuracy", "clarity", "completeness"]

def aggregate(cases: list[dict]) -> dict:
    n = len(cases)
    per_check = {c: sum(case["verdicts"][c] for case in cases) / n for c in CHECKS}
    all_pass = sum(1 for case in cases if all(case["verdicts"].values())) / n
    return {"per_check_pass_rate": per_check, "all_checks_pass_rate": all_pass, "n": n}
```

## Code pattern

Invoke the deployed agent with a unique session ID, then extract tool usage from CloudWatch:

```python
import boto3, json, uuid

agentcore = boto3.client("bedrock-agentcore")
logs = boto3.client("logs")

def invoke(agent_arn: str, prompt: str, qualifier: str = "DEFAULT") -> tuple[str, str]:
    session_id = f"eval-session-{uuid.uuid4()}"  # must be >= 33 chars
    resp = agentcore.invoke_agent_runtime(
        agentRuntimeArn=agent_arn, runtimeSessionId=session_id,
        payload=json.dumps({"prompt": prompt}), qualifier=qualifier)
    return session_id, resp["response"].read().decode("utf-8")

def get_tool_calls(session_id: str, agent_arn: str, qualifier: str = "DEFAULT") -> list:
    agent_id = agent_arn.split("/")[-1]
    log_group = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{qualifier}"
    events = logs.filter_log_events(logGroupName=log_group, filterPattern=session_id)["events"]
    tools = set()
    for e in events:
        try:
            data = json.loads(e["message"])
            for msg in data.get("body", {}).get("output", {}).get("messages", []):
                content = json.loads(msg.get("content", {}).get("content", ""))
                for item in content:
                    if "toolUse" in item:
                        tools.add(item["toolUse"]["name"])
        except (json.JSONDecodeError, KeyError):
            continue
    return list(tools)
```

Native evaluators (highest-level layer — retrieves session spans and scores them for you):

```python
from bedrock_agentcore_starter_toolkit import Evaluation

def native_eval(agent_arn: str, session_id: str) -> dict:
    agent_id = agent_arn.split("/")[-1]
    results = Evaluation().run(agent_id=agent_id, session_id=session_id,
                               evaluators=["Builtin.Helpfulness", "Builtin.Accuracy"])
    return {r.evaluator_name: {"value": r.value, "label": r.label}
            for r in results.get_successful_results()}
```

Per test case: `invoke → get_tool_calls → check_tool_selection + the four binary judges` (and,
optionally, `native_eval`) → assemble the case verdicts → `aggregate`.

## Adaptation notes

- Substitute the user's `agent_arn`, `qualifier`, region, and `JUDGE_MODEL` (from `eval_config.yaml`).
- Session IDs must be **≥33 chars** — keep the `eval-session-{uuid4}` shape.
- Log-group path is fixed as `/aws/bedrock-agentcore/runtimes/{agent_id}-{qualifier}`; `agent_id` is
  the last ARN segment. Adjust only if the user's qualifier differs from `DEFAULT`.
- Build test cases as `(query, expected_tools, category)`; include a no-tool case (e.g. a greeting →
  `expected_tools == []`).
- The trace can lag the invocation — add a short retry/wait before `get_tool_calls` if logs are empty.
- Native evaluators require AgentCore Evaluations access; if unavailable, drop that layer and keep
  the CloudWatch + binary-judge layers.
