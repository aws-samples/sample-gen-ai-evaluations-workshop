---
name: AgentCore Evaluations
description: Use when learners want to evaluate AI agents deployed on Amazon Bedrock AgentCore Runtime using predefined metrics, CloudWatch log-based tool evaluation, and the native AgentCore Evaluations API.
---

# Evaluating Agents with Amazon Bedrock AgentCore

In this skill, you build a complete evaluation pipeline for an AI agent running on Amazon Bedrock AgentCore Runtime. Unlike single-turn LLM evaluations, agent evaluation is inherently multi-step: an agent reasons, selects tools, executes actions, and synthesizes results across multiple turns before producing a final answer. AgentCore provides managed infrastructure that captures this entire workflow as observable traces, enabling you to evaluate not just the final response but every decision the agent made along the way.

You will deploy a city search agent to AgentCore Runtime, evaluate its responses using LLM-as-Judge with custom quality metrics, analyze tool usage through CloudWatch log queries, and run native AgentCore Evaluations with built-in evaluators like `Builtin.Helpfulness` and `Builtin.ToolSelectionAccuracy`. By the end, you will have a repeatable harness for assessing agent quality across multiple dimensions.

## Prerequisites

- Completed Module 03 (Agentic Metrics) — familiarity with agent concepts and tool use
- AWS credentials configured with permissions for Bedrock, AgentCore, CloudWatch, ECR, and IAM
- Access to Amazon Bedrock foundation models (Claude Sonnet, Nova Micro)
- Python 3.9+ with `boto3`, `strands-agents`, and `bedrock-agentcore-starter-toolkit` installed

## Learning Objectives

- Deploy a Strands-based agent to AgentCore Runtime and verify it is ready for invocation
- Construct an LLM-as-Judge evaluation that scores agent responses on helpfulness, accuracy, clarity, professionalism, and completeness
- Extract tool usage data from CloudWatch logs using session-filtered queries and compute tool selection precision/recall
- Run AgentCore native evaluations using `Builtin.Helpfulness` and `Builtin.ToolSelectionAccuracy` evaluators against session spans
- Interpret evaluation scores, labels, and explanations to identify areas for agent improvement

## Setup

Install required packages and configure AWS clients:

```python
!pip install strands-agents strands-agents-tools bedrock-agentcore bedrock-agentcore-starter-toolkit boto3 pandas ddgs
```

```python
import boto3
import json
import uuid
import time
from datetime import datetime
from boto3.session import Session

boto_session = Session()
region = boto_session.region_name

bedrock = boto3.client('bedrock-runtime')
agentcore_client = boto3.client('bedrock-agentcore', region_name='us-east-1')
```

Deploy the city search agent to AgentCore Runtime:

```python
from bedrock_agentcore_starter_toolkit import Runtime

agentcore_runtime = Runtime()
configure_response = agentcore_runtime.configure(
    entrypoint="citysearch.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name="citysearch"
)

launch_result = agentcore_runtime.launch(auto_update_on_conflict=True)
citysearch_agent_arn = launch_result.agent_arn
```

Verify the agent reaches READY status before proceeding:

```python
status_response = agentcore_runtime.status()
status = status_response.endpoint['status']
end_status = ['READY', 'CREATE_FAILED', 'DELETE_FAILED', 'UPDATE_FAILED']
while status not in end_status:
    time.sleep(10)
    status_response = agentcore_runtime.status()
    status = status_response.endpoint['status']
assert status == 'READY', f"Agent not ready: {status}"
```

### Section 1: LLM-as-Judge Agent Evaluation

When you evaluate an agent, you need to assess more than whether the answer is correct. An agent's response quality spans multiple dimensions: Was it helpful? Was it accurate? Was it clear and professional? A single metric cannot capture this. LLM-as-Judge evaluation uses a separate model (the "judge") to score agent responses against defined criteria, providing nuanced multi-dimensional assessment that mirrors how a human evaluator would rate the output.

This approach is particularly valuable for agent systems because agents produce free-form responses that cannot be evaluated with simple string matching. The judge model can reason about whether the response addressed the user's intent, whether the information is factually grounded, and whether the tone is appropriate — all in a single evaluation pass.

The key insight is that you define the evaluation criteria explicitly in the judge prompt, making the evaluation reproducible and auditable. Different use cases may weight different dimensions: a customer-facing agent might prioritize professionalism and clarity, while a research agent might prioritize accuracy and completeness.

**Build: Create test cases and an LLM-as-Judge evaluator**

Define test cases with expected behaviors and evaluation criteria:

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class TestCase:
    id: str
    query: str
    category: str
    expected_tools: List[str]
    expected_criteria: Dict[str, Any]
    description: str

@dataclass
class EvaluationResult:
    test_case_id: str
    query: str
    response: str
    metrics: Dict[str, float]
    response_time: float
    success: bool
    tool_calls: List[str] = None

TEST_CASES = [
    TestCase(
        id="basic_greeting",
        query="Hi, I need help with finding information about cities",
        category="basic_inquiry",
        expected_tools=[],
        expected_criteria={"should_be_polite": True, "should_ask_for_details": True},
        description="Basic greeting and help request"
    ),
    TestCase(
        id="city_population_search",
        query="What is the population of Seattle?",
        category="population_inquiry",
        expected_tools=["web_search"],
        expected_criteria={"should_provide_population": True, "should_be_accurate": True},
        description="City population information request"
    ),
    TestCase(
        id="city_area_search",
        query="How large is Los Angeles in square miles?",
        category="area_inquiry",
        expected_tools=["web_search"],
        expected_criteria={"should_provide_area": True, "should_be_clear": True},
        description="City area information request"
    )
]
```

Build the LLM-as-Judge evaluation function that scores responses on five dimensions:

```python
EVALUATOR_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

async def evaluate_response_quality(query: str, response: str, criteria: Dict[str, Any]) -> Dict[str, float]:
    evaluation_prompt = f"""
    You are an expert evaluator for city search AI agents. Evaluate the following response on a scale of 1-5 for each metric.

    Customer Query: {query}
    Agent Response: {response}

    Evaluate on these metrics (1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent):

    1. HELPFULNESS: Does the response address the user's needs?
    2. ACCURACY: Is the information factually correct?
    3. CLARITY: Is the response clear and well-structured?
    4. PROFESSIONALISM: Does the response maintain appropriate tone?
    5. COMPLETENESS: Does the response fully address all aspects of the query?

    Expected criteria: {json.dumps(criteria, indent=2)}

    Respond with ONLY a JSON object:
    {{"helpfulness": <score>, "accuracy": <score>, "clarity": <score>, "professionalism": <score>, "completeness": <score>}}
    """

    response_obj = bedrock.invoke_model(
        modelId=EVALUATOR_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": evaluation_prompt}]
        })
    )

    result = json.loads(response_obj['body'].read())
    content = result['content'][0]['text']
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    scores = json.loads(content[start_idx:end_idx])
    return {k: v for k, v in scores.items() if k != "reasoning"}
```

### Section 2: CloudWatch Log-Based Tool Evaluation

Knowing whether an agent produced a good final answer is only half the story. You also need to know whether the agent used the right tools at the right time. Did it call `web_search` when it should have? Did it avoid unnecessary tool calls for simple greetings? Tool selection accuracy directly impacts latency, cost, and reliability — an agent that calls tools unnecessarily wastes time and money, while one that skips needed tools produces hallucinated answers.

AgentCore Runtime automatically logs all agent execution traces to CloudWatch, including every tool invocation. By querying these logs filtered by session ID, you can extract exactly which tools were called during a specific interaction. This gives you ground truth for tool usage evaluation without any additional instrumentation in your agent code.

The evaluation pattern compares expected tools (defined in your test cases) against actual tools (extracted from logs) using precision and recall metrics. This F1-based scoring penalizes both unnecessary tool calls (low precision) and missed tool calls (low recall), giving you a balanced view of tool selection quality.

**Build: Extract tool calls from CloudWatch and compute tool accuracy**

```python
def extract_agent_log_name(arn):
    return arn.split('/')[-1]

def extract_tool_calls_from_agentcore_observability(session_id, log_group_name, agent_qualifier, log_group_prefix='/aws/bedrock-agentcore/runtimes'):
    logs_client = boto3.client('logs')
    log_group_name = f"{log_group_prefix}/{log_group_name}-{agent_qualifier}"

    response = logs_client.filter_log_events(
        logGroupName=log_group_name,
        filterPattern=session_id
    )
    logs_list = [event['message'] for event in response['events']]
    return extract_logs_for_session(logs_list)

def extract_tooluse_from_log(log_message):
    tools = []
    try:
        log_data = json.loads(log_message)
        messages = log_data.get('body', {}).get('output', {}).get('messages', [])
        for message in messages:
            content = message.get('content', {}).get('content', '')
            if content:
                content_data = json.loads(content)
                for item in content_data:
                    if 'toolUse' in item:
                        tools.append(item['toolUse']['name'])
    except (json.JSONDecodeError, KeyError):
        pass
    return tools

def extract_logs_for_session(logs_list):
    all_tools = []
    for log in logs_list:
        tools = extract_tooluse_from_log(log)
        all_tools.extend(tools)
    return list(set(all_tools))
```

Compute tool usage effectiveness using precision/recall:

```python
def evaluate_tool_usage(expected_tools: List[str], actual_tools: List[str]) -> float:
    if not expected_tools:
        return 5.0 if not actual_tools else 3.0

    if not actual_tools:
        return 0.0

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    precision = len(expected_set.intersection(actual_set)) / len(actual_set) if actual_set else 0
    recall = len(expected_set.intersection(actual_set)) / len(expected_set) if expected_set else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1 * 5  # Scale to 0-5
```

Run a single end-to-end test case combining response quality and tool evaluation:

```python
async def invoke_agent(query: str) -> Dict[str, Any]:
    start_time = time.time()
    payload = json.dumps({"prompt": query})
    session_id = f"eval-session-{uuid.uuid4()}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=citysearch_agent_arn,
        runtimeSessionId=session_id,
        payload=payload,
        qualifier="DEFAULT"
    )

    response_text = response['response'].read().decode('utf-8')
    log_group_name = extract_agent_log_name(citysearch_agent_arn)
    tool_calls = extract_tool_calls_from_agentcore_observability(session_id, log_group_name, "DEFAULT")

    return {
        "response": response_text,
        "success": True,
        "tool_calls": tool_calls,
        "response_time": time.time() - start_time
    }

# Test with a single case
demo_test = TEST_CASES[1]
agent_result = await invoke_agent(demo_test.query)
quality_scores = await evaluate_response_quality(demo_test.query, agent_result["response"], demo_test.expected_criteria)
tool_score = evaluate_tool_usage(demo_test.expected_tools, agent_result["tool_calls"])

print(f"Quality scores: {quality_scores}")
print(f"Tool usage score: {tool_score}/5.0")
print(f"Tools called: {agent_result['tool_calls']}")
```

### Section 3: Native AgentCore Evaluations API

The previous sections showed you how to build custom evaluation logic. But AgentCore also provides a native Evaluations API with built-in evaluators that analyze agent execution traces directly. This is powerful because the built-in evaluators have access to the full execution trace — every LLM call, every tool invocation, every intermediate reasoning step — not just the final response. They can assess whether the agent's reasoning process was sound, not just whether the output looks correct.

The `Builtin.Helpfulness` evaluator assesses whether the agent's response genuinely helps the user accomplish their goal. The `Builtin.ToolSelectionAccuracy` evaluator examines each tool call in the trace and determines whether it was appropriate given the user's query and the available tools. These evaluators return scores (0.0–1.0), categorical labels (like "Very Helpful" or "Yes"/"No"), and detailed explanations of their reasoning.

The workflow is: invoke the agent (generating traces in CloudWatch), wait for log propagation, then call the Evaluation API with the session ID. The API retrieves the spans internally and runs the evaluators against them. This makes it straightforward to integrate into automated testing pipelines.

**Build: Run native AgentCore evaluations with built-in evaluators**

Invoke the agent for multiple test cases to generate traces:

```python
import pandas as pd
import re

# Load test cities
city_df = pd.read_csv("city_pop.csv")
city_df['city_clean'] = city_df['city'].apply(lambda x: re.sub(r'\[\w+\]', '', x).strip())

NUM_TEST_CASES = 5
test_cities = city_df.head(NUM_TEST_CASES)

session_ids = {}
for idx, row in test_cities.iterrows():
    city = row['city_clean']
    state = row['state']
    city_key = f"{city}, {state}"
    session_id = f"eval-session-{uuid.uuid4()}"
    session_ids[city_key] = session_id

    query = f"What is the population and area of {city}, {state}?"
    payload = json.dumps({"prompt": query})

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=citysearch_agent_arn,
        runtimeSessionId=session_id,
        payload=payload,
        qualifier="DEFAULT"
    )
    response_body = response['response'].read()
    print(f"✓ {city_key} — session: {session_id}")

# Wait for CloudWatch log propagation
print("Waiting 30s for log propagation...")
time.sleep(30)
```

Run AgentCore Evaluations using the starter toolkit:

```python
from bedrock_agentcore_starter_toolkit import Evaluation

eval_client = Evaluation()
agent_id = citysearch_agent_arn.split('/')[-1]

all_results = []
for city_key, session_id in session_ids.items():
    results = eval_client.run(
        agent_id=agent_id,
        session_id=session_id,
        evaluators=["Builtin.Helpfulness", "Builtin.ToolSelectionAccuracy"]
    )

    successful = results.get_successful_results()
    for result in successful:
        print(f"  {result.evaluator_name}: score={result.value:.2f}, label={result.label}")
        all_results.append({
            'city': city_key,
            'evaluator': result.evaluator_name,
            'score': result.value,
            'label': result.label
        })
```

Analyze aggregate results:

```python
results_df = pd.DataFrame(all_results)
avg_by_evaluator = results_df.groupby('evaluator')['score'].agg(['mean', 'min', 'max', 'count'])
print(avg_by_evaluator)

overall_avg = results_df['score'].mean()
print(f"\nOverall Average Score: {overall_avg:.3f}")
```

## Challenges

**Challenge: Build a comprehensive agent evaluation report**

Create an evaluation script that:
1. Defines at least 5 diverse test cases spanning different query categories (greetings, factual lookups, multi-part questions)
2. Runs both custom LLM-as-Judge evaluation AND native AgentCore evaluations for each test case
3. Computes aggregate metrics including per-category breakdowns and response time percentiles
4. Identifies the weakest evaluation dimension and proposes a system prompt improvement to address it
5. Re-runs evaluation after the prompt change and compares before/after scores

**Assessment criteria:**

1. Script runs without errors and produces evaluation results for all test cases
2. Uses LLM-as-Judge with at least 3 quality dimensions from Section 1
3. Extracts tool usage from CloudWatch logs and computes precision/recall from Section 2
4. Runs native AgentCore evaluations with both `Builtin.Helpfulness` and `Builtin.ToolSelectionAccuracy` from Section 3
5. Produces a summary table comparing before/after scores with at least one measurable improvement
6. Learner can explain their approach and why they chose specific test cases

---

## Deep-Dive Challenge

AgentCore is an **agent-focused** framework — it treats the system as a workflow with multiple steps, tool calls, and state transitions. You evaluate process quality in addition to final output. This deep-dive pushes you beyond notebook-level usage into advanced observability and failure analysis patterns.

### Workflow

| Stage | What you implement |
|---|---|
| Agent instrumentation | Capture traces/spans for a multi-step agent workflow |
| Step-level metrics | Metrics per agent step (tool selection accuracy, retrieval quality, reasoning correctness) |
| End-to-end metrics | Task completion, total latency, cost |
| Failure analysis | Identify where and why the agent fails (wrong tool, bad retrieval, hallucination) |
| Observability | Dashboard or structured log output showing per-step and aggregate health |

### "Beyond" Examples for AgentCore

- Custom metric plugin; cross-step metric correlation (retrieval quality → answer quality)
- Custom CW dashboard with alarms; anomaly detection on metric trends; automated alerting
- A/B evaluation of two agent configs; latency-quality tradeoff analysis; cost-per-quality-point metric

### Scoring Rubric

| Tier | Points | Criteria |
|---|---|---|
| **Functional** | 60-69 | Complete workflow runs end-to-end; uses only notebook-level features; results are valid |
| **Extended** | 70-84 | Adds 1 capability not in notebook; clear justification for the extension |
| **Advanced** | 85-94 | Adds 2+ capabilities; demonstrates iteration (before/after comparison); addresses a real evaluation gap |
| **Exceptional** | 95-100 | Novel approach; production-quality output (CI-ready, dashboarded, or automated); teaches the reviewer something new |

### Assessment Criteria

| Criterion | Weight | Description |
|---|---|---|
| Complete workflow execution | 25% | All stages implemented and runnable; produces valid output |
| Beyond-notebook features | 25% | Number and quality of capabilities not covered in source notebook |
| Justification & analysis | 20% | Why each metric/feature was chosen; what evaluation gap it addresses |
| Iteration evidence | 15% | Before/after comparison showing the pipeline caught or improved something |
| "What was left out" | 10% | Identifies limitations; names what they'd need to cover them |
| Code quality & documentation | 5% | Readable, commented, reproducible |

### Tips

1. **Start with the notebook** — get it running, then extend one piece at a time.
2. **Define your "beyond" early** — decide what you're adding before you start coding.
3. **Document as you go** — capture why you chose each metric and what gap it fills.
4. **Show iteration** — run your eval, change something, re-run, and compare results. This is the strongest signal of understanding.
5. **Name your limitations** — the rubric rewards honesty about what's missing.

---

## Wrap-Up

You have built a multi-layered evaluation pipeline for an agent-focused system. You can now assess agent quality across response dimensions (helpfulness, accuracy, clarity), tool selection behavior (precision/recall from CloudWatch logs), and end-to-end performance (native AgentCore evaluators against full execution traces).

The key distinction of agent evaluation versus simple LLM evaluation is that you must evaluate the entire workflow — not just the final output. An agent might produce a correct answer through incorrect reasoning, or use the wrong tools but get lucky. The combination of response quality scoring, tool usage analysis, and trace-based evaluation gives you confidence that your agent is performing well for the right reasons.

For a deeper challenge that combines these techniques with other framework evaluations, see [CHALLENGE-deep-dive.md](./CHALLENGE-deep-dive.md).

**Suggested next module:** Module 06 — Continuous Evaluation Pipelines, where you integrate these evaluation techniques into automated CI/CD workflows.
