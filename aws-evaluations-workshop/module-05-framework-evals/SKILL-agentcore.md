---
name: AgentCore Evaluations
description: Deploy agents to AgentCore Runtime and evaluate them using CloudWatch log analysis, LLM-as-judge scoring, and the native Evaluations API
---

# AgentCore Runtime Evaluations

Build an end-to-end agent evaluation pipeline on AWS AgentCore Runtime. You will deploy a Strands-based agent, invoke it to generate traces, extract tool usage from CloudWatch logs, run LLM-as-judge quality assessments, and execute built-in evaluators via the AgentCore Evaluations API.

> **Note:** This module uses `strands-agents` for the deployed agent. Strands is under active development — APIs may change between versions.

## What This Does NOT Cover

- Agent design patterns or prompt engineering
- AgentCore pricing or capacity planning
- Custom evaluator authoring (uses built-in evaluators only)
- Multi-agent orchestration evaluation

## Prerequisites

- AWS account with Bedrock and AgentCore access
- CloudWatch Logs read access
- A deployed AgentCore agent (or willingness to deploy one)

## Learning Objectives

1. Deploy a Strands-based agent to AgentCore Runtime and confirm READY status
2. Invoke the deployed agent with session IDs that generate CloudWatch traces
3. Extract tool usage from CloudWatch logs and compute selection accuracy
4. Run LLM-as-judge evaluation producing structured multi-dimensional scores
5. Execute AgentCore native evaluators via the Evaluations API

## Setup

```bash
pip install strands-agents strands-agents-bedrock bedrock-agentcore bedrock-agentcore-starter-toolkit boto3
```

**Project structure:**
```
agentcore-evals/
├── citysearch.py          # Agent entrypoint
├── requirements.txt       # Runtime dependencies
└── evaluate.py            # Evaluation harness
```

---

### Section 1: Deploy and Configure AgentCore Runtime

**Concept:** AgentCore Runtime manages infrastructure for your agent — you provide a Python entrypoint decorated with `@app.entrypoint`, and the starter toolkit handles containerization, ECR publishing, and endpoint creation. The `configure()` + `launch()` pattern abstracts IAM roles, ECR repos, and Dockerfiles into two calls. You poll `status()` until READY before invoking.

**Build:**

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore_starter_toolkit import Runtime
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from boto3.session import Session

# --- Agent definition ---
app = BedrockAgentCoreApp()

@tool
def web_search(topic: str) -> str:
    """Search for information about a topic."""
    from ddgs import DDGS
    results = DDGS(timeout=5).text(topic, max_results=5)
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)

model = BedrockModel(model_id="us.amazon.nova-micro-v1:0")
agent = Agent(model=model, tools=[web_search])

@app.entrypoint
def handler(prompt: str) -> str:
    return str(agent(prompt))

# --- Deploy ---
region = Session().region_name
runtime = Runtime()
runtime.configure(
    entrypoint="citysearch.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name="citysearch"
)
launch_result = runtime.launch(auto_update_on_conflict=True)
agent_arn = launch_result.agent_arn
```

---

### Section 2: Invoke and Extract Tool Usage from Logs

**Concept:** Each invocation requires a unique session ID (≥33 chars) that tags the execution trace in CloudWatch. AgentCore logs every tool invocation under `/aws/bedrock-agentcore/runtimes/{agent_id}-{qualifier}`. By filtering logs for a session ID, you extract which tools were called — enabling tool selection accuracy evaluation without custom instrumentation.

**Build:**

```python
import boto3, json, uuid

client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Invoke
session_id = f"eval-session-{uuid.uuid4()}"
response = client.invoke_agent_runtime(
    agentRuntimeArn=agent_arn,
    runtimeSessionId=session_id,
    payload=json.dumps({"prompt": "What is the population of Seattle?"}),
    qualifier="DEFAULT"
)
result = response['response'].read().decode('utf-8')

# Extract tool calls from CloudWatch
def get_tool_calls(session_id: str, agent_arn: str, qualifier: str = "DEFAULT") -> list:
    logs_client = boto3.client('logs')
    agent_id = agent_arn.split('/')[-1]
    log_group = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{qualifier}"
    response = logs_client.filter_log_events(
        logGroupName=log_group, filterPattern=session_id
    )
    tools = set()
    for event in response['events']:
        try:
            data = json.loads(event['message'])
            messages = data.get('body', {}).get('output', {}).get('messages', [])
            for msg in messages:
                content = json.loads(msg.get('content', {}).get('content', ''))
                for item in content:
                    if 'toolUse' in item:
                        tools.add(item['toolUse']['name'])
        except (json.JSONDecodeError, KeyError):
            continue
    return list(tools)

# Compute tool selection accuracy
def tool_selection_score(expected: list, actual: list) -> dict:
    if not expected and not actual:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    expected_set, actual_set = set(expected), set(actual)
    intersection = expected_set & actual_set
    precision = len(intersection) / len(actual_set) if actual_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

### Section 3: LLM-as-Judge Quality Evaluation

**Concept:** For dimensions that can't be computed deterministically (helpfulness, accuracy, clarity), you use a stronger model as an evaluator. The pattern: construct a rubric prompt, send the agent's response to the judge model, parse structured scores. This gives you multi-dimensional quality metrics per invocation.

**Build:**

```python
import boto3, json

bedrock = boto3.client('bedrock-runtime')

def evaluate_quality(query: str, response: str, model_id: str) -> dict:
    prompt = f"""Evaluate this agent response on a 1-5 scale for each metric.
Query: {query}
Response: {response}

Metrics: HELPFULNESS, ACCURACY, CLARITY, COMPLETENESS
Respond with ONLY a JSON object: {{"helpfulness": N, "accuracy": N, "clarity": N, "completeness": N}}"""

    result = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    text = json.loads(result['body'].read())['content'][0]['text']
    return json.loads(text[text.find('{'):text.rfind('}')+1])
```

---

### Section 4: Native Evaluations API

**Concept:** The AgentCore Evaluations API provides built-in evaluators (`Builtin.Helpfulness`, `Builtin.Accuracy`) that automatically retrieve session spans from CloudWatch and score them. This is the highest-level evaluation primitive — one call replaces manual log retrieval + custom scoring.

**Build:**

```python
from bedrock_agentcore_starter_toolkit import Evaluation

eval_client = Evaluation()
agent_id = agent_arn.split('/')[-1]

results = eval_client.run(
    agent_id=agent_id,
    session_id=session_id,
    evaluators=["Builtin.Helpfulness", "Builtin.Accuracy"]
)

for result in results.get_successful_results():
    print(f"{result.evaluator_name}: {result.value:.2f} ({result.label})")
```

---

### Section 5: Full Evaluation Harness

**Concept:** Combine all three approaches (CloudWatch log extraction, LLM-as-judge, native evaluators) into a single test harness that runs multiple test cases and produces an aggregate scorecard. This is the pattern you reuse for any AgentCore agent.

**Build:**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TestCase:
    query: str
    expected_tools: List[str]
    category: str

test_cases = [
    TestCase("Hello!", [], "greeting"),
    TestCase("Population of NYC?", ["web_search"], "factual"),
    TestCase("Area of Chicago in sq miles?", ["web_search"], "factual"),
]

results = []
for tc in test_cases:
    session_id = f"eval-session-{uuid.uuid4()}"
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": tc.query}),
        qualifier="DEFAULT"
    )
    text = response['response'].read().decode('utf-8')
    actual_tools = get_tool_calls(session_id, agent_arn)
    tool_score = tool_selection_score(tc.expected_tools, actual_tools)
    quality = evaluate_quality(tc.query, text, "us.anthropic.claude-sonnet-4-20250514-v1:0")
    results.append({"query": tc.query, "tool_f1": tool_score["f1"], **quality})

# Aggregate
import statistics
print(f"Mean Tool F1: {statistics.mean(r['tool_f1'] for r in results):.2f}")
print(f"Mean Accuracy: {statistics.mean(r['accuracy'] for r in results):.1f}/5")
```

---

## Challenges

Build a regression test suite that runs nightly against your deployed agent and alerts on score degradation. Your harness should cover at least 10 test cases across 3+ categories, combine all three evaluation layers, and produce a summary report.

**Assessment criteria:**

- Deploy an agent to AgentCore Runtime and confirm READY status
- Extract tool usage from CloudWatch logs and compute precision/recall/F1
- Run LLM-as-judge evaluation producing structured quality scores
- Execute AgentCore native evaluators (`Builtin.Helpfulness`, `Builtin.Accuracy`) and interpret results
- Combine all methods into a repeatable test harness with aggregate scoring

## Wrap-Up

You built a complete AgentCore evaluation pipeline covering three layers:

| Layer | Method | What It Measures |
|-------|--------|-----------------|
| Operational | CloudWatch log extraction | Tool selection accuracy |
| Quality | LLM-as-judge | Helpfulness, accuracy, clarity |
| Native | AgentCore Evaluations API | Built-in composite scores |

**Key takeaway:** AgentCore's native observability means you never instrument your agent code for evaluation — traces flow automatically, and the Evaluations API consumes them directly.

See **CHALLENGE-deep-dive.md** for the Module 05 extension challenge.
