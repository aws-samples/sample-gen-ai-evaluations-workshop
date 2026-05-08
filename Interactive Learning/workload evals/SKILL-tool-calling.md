---
name: Tool Calling Evaluation
description: Help me evaluate tool-calling agents without executing real tools, using static analysis, schema validation, mock tools, LLM-as-judge, and synthetic simulation
---

# Evaluating Tool-Calling Agents Without Real Tool Execution

Tool calling is the core capability that separates an agent from a chatbot — but real tool calls are expensive, slow, non-deterministic, and can have side effects. This module teaches you to rigorously evaluate an agent's tool-calling behavior without executing real tools, using six approaches ordered from cheapest to most expensive.

## Prerequisites

- AWS account with Bedrock access (us-west-2)
- Python 3.10+
- Familiarity with boto3 and the Bedrock Converse API
- Completed SKILL-guardrails or equivalent understanding of evaluation harness patterns

## Learning Objectives

By the end of this module, you will be able to:

1. Decompose tool-calling evaluation into three independent decisions (whether, which, what parameters)
2. Compute precision, recall, and F1 metrics on pre-recorded tool-calling traces at zero cost
3. Validate tool parameters against JSON Schema to catch hallucination programmatically
4. Run live model evaluation with mock tool responses using the Bedrock Converse API tool loop
5. Apply LLM-as-Judge with structured rubrics to assess subjective tool-calling quality
6. Build multi-turn and fully synthetic evaluations using scripted scenarios and Strands simulators

## Setup

```bash
pip install boto3 pandas jsonschema strands-agents strands-agents-evals --quiet
```

```python
import boto3
import json
import re
import pandas as pd
from botocore.config import Config
from collections import defaultdict
import jsonschema

bedrock = boto3.client(
    'bedrock-runtime',
    config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 2})
)

AGENT_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
```

Load the evaluation data (provided in `data/`):

```python
with open('data/tool_definitions.json') as f:
    tool_defs = json.load(f)

with open('data/test_cases.json') as f:
    test_cases = json.load(f)

with open('data/recorded_traces.json') as f:
    recorded_traces = json.load(f)

tc_lookup = {tc['id']: tc for tc in test_cases}

print(f"Tools available: {len(tool_defs['tools'])}")
print(f"Test cases: {len(test_cases)}")
print(f"Pre-recorded traces: {len(recorded_traces)}")
```

The scenario is a customer support agent with 6 tools: `lookup_order`, `initiate_return`, `check_inventory`, `update_shipping_address`, `get_customer_profile`, and `escalate_to_human`. Pre-recorded traces include both correct and intentionally incorrect behaviors for testing.

## Section 1: The Three Decisions and Evaluation Pyramid

**Concept:** Every tool call involves three model decisions that can be evaluated independently:

1. **Whether** to call a tool (vs. responding conversationally)
2. **Which** tool to call (tool selection)
3. **What** parameters to pass (parameter generation)

These map to an evaluation pyramid — six approaches ordered by cost:

| Layer | Approach | Cost | Best For |
|-------|----------|------|----------|
| 1 | Static Trajectory Analysis | $0 | Production logs, regression testing |
| 2 | Schema Validation | $0 | Parameter hallucination, type errors |
| 3 | Mock Tool Evaluation | $ | Live model decisions safely |
| 4 | LLM-as-Judge | $$ | Subjective quality, edge cases |
| 5 | Multi-Turn Simulation | $$$ | End-to-end goal achievement |
| 6 | Strands ToolSimulator + ActorSimulator | $$$$ | Fully synthetic stateful simulation |

The key insight: **separate what you're evaluating from what you're executing.** You're evaluating the model's *decisions*. You don't need real tools for that.

**Explore the scenario:**

```python
print("Available Tools")
for tool in tool_defs['tools']:
    required = tool['parameters'].get('required', [])
    print(f"  {tool['name']}: {tool['description'][:80]}")
    print(f"    Required params: {required}")

print("\nTest Case Distribution")
categories = defaultdict(list)
for tc in test_cases:
    categories[tc['category']].append(tc['id'])
for cat, ids in categories.items():
    print(f"  {cat}: {len(ids)} cases")
```

## Section 2: Static Analysis and Schema Validation (Cost: $0)

**Concept:** The cheapest evaluation runs entirely on pre-recorded traces — no API calls needed. Static trajectory analysis computes tool selection metrics (precision, recall, F1, call/no-call accuracy, parameter accuracy, sequence match). Schema validation catches parameter hallucination by validating against JSON Schema definitions. Together, these are your first line of defense — run them on every trace, every time.

**Build the static analyzer:**

```python
def analyze_trace(trace, test_case):
    """Analyze a pre-recorded trace against ground truth. No API calls needed."""
    results = {
        'trace_id': trace['trace_id'],
        'label': trace['label'],
    }

    expected_tools = set(test_case['expected_tools'])
    actual_tools = set(trace['tools_called'])

    tp = len(expected_tools & actual_tools)
    fp = len(actual_tools - expected_tools)
    fn = len(expected_tools - actual_tools)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results['tool_precision'] = round(precision, 3)
    results['tool_recall'] = round(recall, 3)
    results['tool_f1'] = round(f1, 3)
    results['call_decision_correct'] = test_case['should_call_tool'] == (len(trace['tools_called']) > 0)
    results['sequence_match'] = trace['tools_called'] == test_case['expected_tools']

    param_scores = []
    for tool_name, expected_params in test_case['expected_params'].items():
        if tool_name in trace.get('params_used', {}):
            actual_params = trace['params_used'][tool_name]
            if expected_params:
                matching = sum(
                    1 for k, v in expected_params.items()
                    if k in actual_params and actual_params[k] == v
                )
                param_scores.append(matching / len(expected_params))
        elif expected_params:
            param_scores.append(0.0)

    results['param_accuracy'] = round(
        sum(param_scores) / len(param_scores), 3
    ) if param_scores else None

    return results
```

**Run static analysis:**

```python
trace_results = []
for trace in recorded_traces:
    tc = tc_lookup.get(trace['test_case_id'])
    if tc:
        trace_results.append(analyze_trace(trace, tc))

df_traces = pd.DataFrame(trace_results)
print(df_traces.to_string(index=False))

print(f"\nCall/No-Call Accuracy: {df_traces['call_decision_correct'].mean():.1%}")
print(f"Average Tool F1:      {df_traces['tool_f1'].mean():.3f}")
print(f"Sequence Match Rate:  {df_traces['sequence_match'].mean():.1%}")
```

**Build the schema validator:**

```python
def validate_tool_call(tool_name, params, tool_defs):
    """Validate tool parameters against JSON Schema."""
    tool_schema = next(
        (t for t in tool_defs['tools'] if t['name'] == tool_name), None
    )
    if not tool_schema:
        return {'tool': tool_name, 'valid': False, 'errors': [f"Unknown tool: {tool_name}"]}

    schema = tool_schema['parameters']
    errors = []
    error_types = []

    validator = jsonschema.Draft7Validator(schema)
    for error in validator.iter_errors(params):
        errors.append(error.message)
        error_types.append(error.validator)

    if 'order_id' in params:
        if not re.match(r'^ORD-\d{5}$', str(params['order_id'])):
            errors.append(f"order_id '{params['order_id']}' doesn't match pattern ORD-XXXXX")
            error_types.append('pattern_violation')

    return {'tool': tool_name, 'valid': len(errors) == 0, 'errors': errors, 'error_types': error_types}
```

**Run schema validation on all traces:**

```python
validation_results = []
for trace in recorded_traces:
    for tool_name in trace['tools_called']:
        params = trace['params_used'].get(tool_name, {})
        result = validate_tool_call(tool_name, params, tool_defs)
        result['trace_id'] = trace['trace_id']
        validation_results.append(result)
        status = "PASS" if result['valid'] else "FAIL"
        print(f"  {trace['trace_id']} | {tool_name:25s} | {status}")
        if not result['valid']:
            for err in result['errors']:
                print(f"    -> {err}")

total = len(validation_results)
valid = sum(1 for r in validation_results if r['valid'])
print(f"\nSchema Compliance: {valid}/{total} ({valid/total:.0%})")
```

Three different failure modes caught for free: unnecessary tool calls (wrong call/no-call decision), missing parameters (tool F1 fine but param accuracy fails), and hallucinated parameters (schema pattern violation on fabricated values like `order_id: "POLICY"`).

## Section 3: Mock Tool Evaluation (Cost: $)

**Concept:** Static analysis only evaluates recorded traces. Mock tool evaluation runs the actual model live — it makes real decisions about whether, which, and what — but returns pre-defined mock responses instead of executing real tools. No side effects occur. The Bedrock Converse API's `toolConfig` parameter provides tool definitions; when the model returns `toolUse` blocks, you intercept them, return mock `toolResult` messages, and continue until the model gives a final text response.

**Build mock responses:**

```python
MOCK_RESPONSES = {
    "lookup_order": lambda params: {
        "order_id": params.get("order_id", "ORD-00000"),
        "status": "delivered",
        "items": ["Wireless Headphones", "Phone Case"],
        "delivered_date": "2026-04-01",
        "total": "$89.99",
    },
    "initiate_return": lambda params: {
        "return_auth": "RET-77201",
        "instructions": "Print the prepaid label and drop off at any UPS location within 14 days.",
        "refund_estimate": "5-7 business days after receipt",
    },
    "check_inventory": lambda params: {
        "product_id": params.get("product_id", "UNKNOWN"),
        "available": True,
        "quantity": 156,
        "warehouse": params.get("warehouse_region", "us-east"),
    },
    "update_shipping_address": lambda params: {
        "status": "updated",
        "order_id": params.get("order_id", "ORD-00000"),
        "note": "Address updated successfully.",
    },
    "get_customer_profile": lambda params: {
        "customer_id": params.get("customer_id", "UNKNOWN"),
        "name": "Jane Smith",
        "tier": "Gold",
        "total_orders": 47,
    },
    "escalate_to_human": lambda params: {
        "ticket": "ESC-3301",
        "estimated_wait": "5 minutes",
        "priority": params.get("priority", "medium"),
    },
}
```

**Convert tool definitions to Converse API format:**

```python
def to_converse_tool_config(tool_defs):
    """Convert tool definitions to Bedrock Converse API toolConfig format."""
    tools = []
    for tool in tool_defs['tools']:
        tools.append({
            "toolSpec": {
                "name": tool['name'],
                "description": tool['description'],
                "inputSchema": {"json": tool['parameters']}
            }
        })
    return {"tools": tools}

tool_config = to_converse_tool_config(tool_defs)
```

**Build the mock tool evaluation loop:**

```python
SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce platform.
Use the available tools to look up information and take actions.
Only call tools when the customer's request requires accessing backend systems.
For general questions, greetings, or policy inquiries, respond directly without using tools."""

def run_with_mock_tools(user_input, max_turns=10):
    """Run Bedrock Converse API with mock tool responses. Returns full trajectory."""
    messages = [{"role": "user", "content": [{"text": user_input}]}]
    trajectory = []

    for turn in range(max_turns):
        response = bedrock.converse(
            modelId=AGENT_MODEL,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            toolConfig=tool_config
        )

        stop_reason = response['stopReason']
        assistant_content = response['output']['message']['content']
        messages.append({"role": "assistant", "content": assistant_content})

        if stop_reason == 'end_turn':
            final_text = "".join(
                block['text'] for block in assistant_content if 'text' in block
            )
            break

        if stop_reason == 'tool_use':
            tool_results = []
            for block in assistant_content:
                if 'toolUse' in block:
                    tool_call = block['toolUse']
                    tool_name = tool_call['name']
                    tool_params = tool_call['input']

                    trajectory.append({'tool': tool_name, 'params': tool_params, 'turn': turn})

                    mock_fn = MOCK_RESPONSES.get(tool_name)
                    mock_result = mock_fn(tool_params) if mock_fn else {"error": f"Unknown tool: {tool_name}"}

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_call['toolUseId'],
                            "content": [{"json": mock_result}],
                            "status": "success"
                        }
                    })
            messages.append({"role": "user", "content": tool_results})
    else:
        final_text = "[Max turns reached]"

    return {
        'trajectory': trajectory,
        'tools_called': [t['tool'] for t in trajectory],
        'params_used': {t['tool']: t['params'] for t in trajectory},
        'final_output': final_text,
    }
```

**Run all test cases and evaluate:**

```python
for tc in test_cases:
    result = run_with_mock_tools(tc['input'])

    expected_tools = set(tc['expected_tools'])
    actual_tools = set(result['tools_called'])
    tp = len(expected_tools & actual_tools)
    fp = len(actual_tools - expected_tools)
    fn = len(expected_tools - actual_tools)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    should_call = tc['should_call_tool']
    did_call = len(result['tools_called']) > 0
    status = "PASS" if f1 == 1.0 and (should_call == did_call) else "FAIL"

    print(f"  {tc['id']} [{status}] F1={f1:.2f} Call={'OK' if should_call == did_call else 'WRONG'}")
```

Mock tool evaluation tests the model's live decisions — not just recorded behavior. It catches regressions when you change prompts or swap models, and it's safe to run in CI because no real tools execute.

## Section 4: LLM-as-Judge for Tool Calling (Cost: $$)

**Concept:** Programmatic metrics catch clear-cut errors, but some quality aspects require judgment: Was tool selection *appropriate* even if technically valid? Was the sequence *efficient*? Was the final response *helpful* given the tool results? An LLM judge evaluates these subjective dimensions using a structured rubric, scoring on 5 dimensions (tool selection, parameter quality, sequence logic, efficiency, response quality) on a 1–5 scale.

**Build the judge:**

```python
def parse_llm_json(text):
    """Robustly extract JSON from LLM output."""
    if '```' in text:
        text = text.split('```')[1]
        if text.startswith('json'):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Failed to parse judge response", "raw": text[:300]}


def judge_trajectory(test_case, result):
    """Use an LLM judge to evaluate tool-calling trajectory quality."""
    tool_descriptions = "\n".join([
        f"- {t['name']}: {t['description']}" for t in tool_defs['tools']
    ])
    trajectory_str = json.dumps(result['trajectory'], indent=2) if result['trajectory'] else "(no tools called)"

    judge_prompt = f"""You are an expert evaluator of AI agent tool-calling behavior.

<available_tools>
{tool_descriptions}
</available_tools>

<user_request>
{test_case['input']}
</user_request>

<expected_tools>
{json.dumps(test_case['expected_tools'])}
</expected_tools>

<actual_trajectory>
{trajectory_str}
</actual_trajectory>

<final_response>
{result['final_output'][:500]}
</final_response>

Evaluate on 5 dimensions (score each 1-5):
1. **Tool Selection**: Right tools for the task?
2. **Parameter Quality**: Correct, complete, well-formed?
3. **Sequence Logic**: Sensible, efficient order?
4. **Efficiency**: Minimum tools needed, no waste?
5. **Response Quality**: Final response addresses the user's request?

Return ONLY valid JSON:
{{
    "tool_selection": {{"score": <1-5>, "reason": "<brief>"}},
    "parameter_quality": {{"score": <1-5>, "reason": "<brief>"}},
    "sequence_logic": {{"score": <1-5>, "reason": "<brief>"}},
    "efficiency": {{"score": <1-5>, "reason": "<brief>"}},
    "response_quality": {{"score": <1-5>, "reason": "<brief>"}},
    "overall_assessment": "<one sentence>"
}}"""

    response = bedrock.converse(
        modelId=JUDGE_MODEL,
        messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
        system=[{"text": "You are an evaluation expert. Return only valid JSON."}]
    )
    judge_text = response['output']['message']['content'][0]['text']
    return parse_llm_json(judge_text)
```

**Run the judge on interesting test cases:**

```python
judge_test_ids = ["tc-001", "tc-002", "tc-004", "tc-005", "tc-008", "tc-009", "tc-012"]

for tc_id in judge_test_ids:
    tc = tc_lookup[tc_id]
    result = run_with_mock_tools(tc['input'])
    scores = judge_trajectory(tc, result)

    if 'error' not in scores:
        dims = ['tool_selection', 'parameter_quality', 'sequence_logic', 'efficiency', 'response_quality']
        avg_score = sum(scores[d]['score'] for d in dims) / len(dims)
        print(f"\n  {tc_id} - {tc['name']}")
        for dim in dims:
            print(f"    {dim:20s}: {scores[dim]['score']}/5 - {scores[dim]['reason']}")
        print(f"    Average: {avg_score:.1f}/5")
```

The LLM judge catches issues programmatic metrics miss: an agent that calls `lookup_order` before `initiate_return` gets a higher sequence logic score (good practice) even though both orderings produce correct F1. It also flags when a technically-correct response is unhelpful or confusing to the user.

## Section 5: Multi-Turn and Synthetic Simulation (Cost: $$$ – $$$$)

**Concept:** Real agent conversations are multi-turn. A customer asks about an order, follows up with a return request, then escalates. Multi-turn simulation tests whether the agent maintains context and makes correct tool decisions across a full session. The Strands Evals SDK takes this further with fully synthetic evaluation: `ToolSimulator` uses an LLM to generate realistic, stateful tool responses (no hardcoded mocks), and `ActorSimulator` generates goal-driven user behavior (no scripted turns). Combined: no real tools, no real users, no scripted scenarios — yet it tests realistic interactions.

**Scripted multi-turn evaluation:**

```python
def run_multi_turn_scenario(scenario):
    """Run a multi-turn conversation with mock tools, evaluating each turn."""
    messages = []
    turn_results = []

    for i, turn in enumerate(scenario['turns']):
        messages.append({"role": "user", "content": [{"text": turn['user']}]})
        turn_trajectory = []

        for _ in range(5):
            response = bedrock.converse(
                modelId=AGENT_MODEL, messages=messages,
                system=[{"text": SYSTEM_PROMPT}], toolConfig=tool_config
            )
            stop_reason = response['stopReason']
            assistant_content = response['output']['message']['content']
            messages.append({"role": "assistant", "content": assistant_content})

            if stop_reason == 'end_turn':
                break
            if stop_reason == 'tool_use':
                tool_results = []
                for block in assistant_content:
                    if 'toolUse' in block:
                        tool_call = block['toolUse']
                        turn_trajectory.append({'tool': tool_call['name'], 'params': tool_call['input']})
                        mock_fn = MOCK_RESPONSES.get(tool_call['name'])
                        mock_result = mock_fn(tool_call['input']) if mock_fn else {"error": "Unknown"}
                        tool_results.append({"toolResult": {
                            "toolUseId": tool_call['toolUseId'],
                            "content": [{"json": mock_result}], "status": "success"
                        }})
                messages.append({"role": "user", "content": tool_results})

        actual_tools = set(t['tool'] for t in turn_trajectory)
        expected_tools = set(turn['expected_tools'])
        tp = len(expected_tools & actual_tools)
        fp = len(actual_tools - expected_tools)
        fn = len(expected_tools - actual_tools)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        turn_results.append({'turn': i + 1, 'f1': round(f1, 3), 'actual': [t['tool'] for t in turn_trajectory]})

    return turn_results


scenario = {
    "name": "Order lookup then return",
    "turns": [
        {"user": "Can you check the status of my order ORD-48291?", "expected_tools": ["lookup_order"]},
        {"user": "I want to return the headphones. They're defective.", "expected_tools": ["initiate_return"]},
    ]
}

for tr in run_multi_turn_scenario(scenario):
    print(f"  Turn {tr['turn']}: F1={tr['f1']} Tools={tr['actual']}")
```

**Fully synthetic evaluation with Strands ToolSimulator + ActorSimulator:**

```python
from strands import Agent
from strands.models import BedrockModel
from strands_evals.simulation import ToolSimulator
from strands_evals import ActorSimulator, Case
from pydantic import BaseModel, Field
from typing import Optional

SIMULATOR_MODEL = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    boto_client_config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 2})
)

simulator = ToolSimulator(model=SIMULATOR_MODEL)

class OrderResponse(BaseModel):
    order_id: str = Field(description="The order ID")
    status: str = Field(description="Order status: pending, shipped, delivered, cancelled")
    items: list[str] = Field(description="Items in the order")
    total: str = Field(description="Order total")

class ReturnResponse(BaseModel):
    return_auth: str = Field(description="Return authorization number")
    instructions: str = Field(description="Return shipping instructions")
    refund_estimate: str = Field(default="5-7 business days")

ORDER_STATE = "order_management"

@simulator.tool(output_schema=OrderResponse, share_state_id=ORDER_STATE,
    initial_state_description="E-commerce order database. Orders use format ORD-XXXXX.")
def lookup_order(order_id: str) -> dict:
    """Look up order details by order ID."""
    pass

@simulator.tool(output_schema=ReturnResponse, share_state_id=ORDER_STATE,
    initial_state_description="Return processing system. Auth numbers use format RET-XXXXX.")
def initiate_return(order_id: str, reason: str, items: list[str] = None) -> dict:
    """Start a return process for a specific order."""
    pass
```

**Run a synthetic case:**

```python
case = Case(
    name="order-return-flow",
    input="Hi, can you check the status of my order ORD-48291?",
    metadata={
        "task_description": "Customer checks order status, discovers an issue, and initiates a return",
        "category": "multi_step"
    }
)

simulated_tools = [simulator.get_tool(name) for name in simulator.list_tools()]
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=simulated_tools,
    model=BedrockModel(model_id=AGENT_MODEL, boto_client_config=Config(read_timeout=60)),
    callback_handler=None
)

user_sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=4)
user_message = case.input

while user_sim.has_next():
    agent_response = agent(user_message)
    print(f"  Agent: {str(agent_response)[:100]}...")
    user_result = user_sim.act(str(agent_response))
    user_message = str(user_result.structured_output.message)
    print(f"  User: {user_message[:100]}...")
```

**When to use each:**

| Dimension | Hardcoded Mocks (Section 3) | ToolSimulator (this section) |
|-----------|----------------------------|------------------------------|
| Determinism | Fully deterministic | Non-deterministic (LLM-generated) |
| Cost per call | $0 | $$ (LLM call per tool invocation) |
| Cross-tool consistency | Manual | Automatic via StateRegistry |
| Best for | CI gates, regression tests | Exploratory testing, edge-case discovery |

## Challenges

### Challenge: Design an Evaluation Pipeline for a New Agent

You have a **travel booking agent** with these tools: `search_flights`, `book_flight`, `search_hotels`, `book_hotel`, `get_traveler_profile`, `cancel_booking`. Design and implement an evaluation pipeline that:

1. Selects ≥3 approaches from the pyramid and justifies why each is appropriate for this domain
2. Writes ≥3 test cases with ground truth (including at least one no-tool case and one multi-tool case)
3. Implements the evaluation for your chosen approaches
4. Identifies at least one gap — what failure mode could slip through your chosen combination?

**Constraint:** The travel domain has a unique challenge the customer support scenario didn't: some tool calls are *irreversible* (booking charges money). How does this affect which evaluation approaches you prioritize?

**Assessment criteria:**
1. Selects ≥3 approaches with cost/coverage reasoning specific to the travel domain (not generic justification)
2. Produces ≥3 test cases with correct ground truth labels, expected tools, and expected parameters
3. Implements at least one programmatic evaluator (static or schema) and one model-based evaluator (mock or judge) that produce metrics
4. Identifies ≥1 specific gap in the chosen approach combination and explains what failure mode it misses

For a cross-module integrative challenge, see `CHALLENGE-capstone.md`.

## Wrap-Up

**Key takeaways:**
- Tool-calling evaluation decomposes into three independent decisions: whether, which, and what parameters
- The evaluation pyramid provides six approaches at increasing cost — use them in combination
- Static analysis + schema validation are free and catch three distinct failure modes (wrong selection, missing params, hallucinated params)
- Mock tool evaluation tests live model behavior safely in CI
- LLM-as-Judge adds subjective quality assessment that programmatic metrics miss
- Strands ToolSimulator + ActorSimulator enables fully synthetic, stateful evaluation without scripted scenarios

**Key metrics to track:**
- Tool Selection F1 — the single most important metric
- Call/No-Call Accuracy — over-calling wastes resources and confuses users
- Schema Compliance — cheapest defense against parameter hallucination
- Goal Success Rate — the ultimate measure for multi-turn agents

**This module does NOT cover:**
- Real tool execution testing (integration tests with live backends)
- Tool definition design (how to write good tool schemas)
- Agent framework comparison (Strands vs. LangChain vs. others)
- Production monitoring and alerting on tool-calling metrics

**Next steps:**
- Adapt the mock tool layer to your own agent's tools
- Build a ground truth test suite from your production logs
- Set up automated evaluation in your CI/CD pipeline
- Explore `CHALLENGE-capstone.md` for an integrative challenge combining evaluation approaches across modules
