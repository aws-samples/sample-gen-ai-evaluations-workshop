---
name: "Agentic Metrics"
description: "Evaluate multi-step AI agents by designing reusable metrics for tool use, latency, and accuracy across agent execution traces"
---

In this module you build an evaluation framework for AI agents that use tools across multiple steps. You will implement a custom evaluation function that scores agent outputs against ground truth, stand up a Strands SDK agent with web-search tools, and measure tool-routing accuracy across a structured dataset. By the end, you can design metrics that apply at different steps of an agent workflow and articulate what your framework does not yet cover.

## Prerequisites

- Completed Module 01 (Evaluation Fundamentals) and Module 02 (LLM-as-Judge)
- Python 3.10+ with `strands-agents` and `strands-agents-builder` installed
- AWS credentials configured with Bedrock model access (Claude Sonnet)
- Familiarity with JSON trace formats and basic statistics (percent error)

## Learning Objectives

- Implement an evaluation function that extracts structured outputs and computes quantitative error against ground truth
- Configure and invoke a Strands SDK agent with tool-decorated functions to generate execution traces
- Measure tool selection accuracy across a multi-case dataset using direct tool call recording
- Design reusable metrics that apply at multiple agent steps with step-appropriate thresholds
- Identify evaluation gaps in an agent framework and specify the data needed to close them

## Setup

1. Navigate to the module directory:

```bash
cd 03-agentic-metrics/
```

2. Verify data files are present:

```bash
ls data/raw_traces.json data/labeled_traces.json
ls city_pop.csv test_cases.json
```

3. Confirm Strands SDK is available:

```python
from strands import Agent
from strands.models.bedrock import BedrockModel
print("Strands SDK ready")
```

4. Set your model configuration:

```python
model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514",
    region_name="us-west-2"
)
```

### Section 1: Building a Custom Evaluation Function

**Concept**

Before you can evaluate an agent, you need a function that takes raw agent output and produces a score. Agents rarely return clean numbers — they embed answers in natural language, XML tags, or JSON. Your evaluation function must parse that output, compare it to ground truth, and capture operational metadata like token count and latency.

The `evaluate_city_guess` function (Cell 8 of the source notebook) demonstrates this pattern. It extracts a population estimate from XML-tagged agent output, computes percent error against a known value from `city_pop.csv`, and records how many tokens the agent consumed and how long it took. This three-part structure — parse, score, annotate — is the foundation for every agentic metric you will build.

Why does this matter? Without structured extraction, you cannot compare agent runs. Without ground truth, you cannot quantify drift. Without operational metadata, you cannot distinguish a correct-but-slow agent from a correct-and-fast one.

**Build**

Extract and run the evaluation function from Cell 8. This function expects the agent to return population guesses wrapped in `<answer>` tags:

```python
import re
import time
import csv

# Load ground truth from city_pop.csv
ground_truth = {}
with open("city_pop.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row["city"].lower()] = int(row["population"])

def evaluate_city_guess(agent_output: str, city: str, token_count: int, latency: float) -> dict:
    """Extract XML-tagged population guess, compute % error vs ground truth."""
    # Extract answer from XML tags
    match = re.search(r"<answer>([\d,]+)</answer>", agent_output)
    if not match:
        return {"city": city, "parsed": False, "error_pct": None,
                "token_count": token_count, "latency_s": latency}

    guess = int(match.group(1).replace(",", ""))
    actual = ground_truth[city.lower()]
    error_pct = abs(guess - actual) / actual * 100

    return {
        "city": city,
        "parsed": True,
        "guess": guess,
        "actual": actual,
        "error_pct": round(error_pct, 2),
        "token_count": token_count,
        "latency_s": round(latency, 3)
    }

# Test with a synthetic output
result = evaluate_city_guess("<answer>750,000</answer>", "seattle", token_count=340, latency=2.1)
print(result)
```

Verify that `error_pct` is computed correctly and that all metadata fields are populated.

### Section 2: Strands Agent Scaffold with Tool-Decorated Functions

**Concept**

An evaluation function is only useful if you have agent outputs to evaluate. The Strands SDK provides a minimal agent framework: you configure a `BedrockModel`, define Python functions decorated with `@tool`, and pass them to an `Agent` instance. When the agent runs, it decides which tools to call, in what order, and how to synthesize results.

Cells 10–16 of the source notebook build a city-population-lookup agent with two tools: `web_search` (retrieves search results) and `get_page` (fetches page content). The agent receives a query like "What is the population of Denver?" and must return an `<answer>`-tagged response. Each invocation produces a trace: the sequence of tool calls, intermediate outputs, total tokens, and wall-clock time.

This scaffold matters because it gives you control over trace generation. Pre-built traces are useful for offline evaluation, but generating fresh traces lets you test new queries, measure variance across runs, and capture metadata that pre-built data may lack.

**Build**

Stand up the agent from Cells 10–16:

```python
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands import tool

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514",
    region_name="us-west-2"
)

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Simplified stub — replace with real search in production
    return f"Search results for: {query}"

@tool
def get_page(url: str) -> str:
    """Fetch the content of a web page."""
    return f"Page content from: {url}"

agent = Agent(
    model=model,
    tools=[web_search, get_page],
    system_prompt=(
        "You are a research assistant. When asked about city populations, "
        "use your tools to find the answer, then respond with the population "
        "wrapped in <answer></answer> XML tags. Example: <answer>500,000</answer>"
    )
)

# Generate a trace
start = time.time()
response = agent("What is the population of Seattle, Washington?")
latency = time.time() - start

print(f"Response: {response}")
print(f"Latency: {latency:.2f}s")
```

Run the agent against at least 2 cities from `city_pop.csv` and feed each response through `evaluate_city_guess` from Section 1.

### Section 3: Tool Selection Accuracy Framework

**Concept**

Beyond output correctness, you need to evaluate whether an agent picks the right tool for a given task. A calculator question routed to a code interpreter may still produce a correct answer, but it reveals a routing inefficiency that compounds in production — wrong tools cost more tokens, add latency, and increase failure probability.

Cells 33–34 of the source notebook introduce a tool selection framework: a 20-case dataset spanning five tools (`calculator`, `file_read`, `current_time`, `file_write`, `code_interpreter`) and an agent configured with `record_direct_tool_call=True`. This flag captures which tool the agent selects first, before execution, letting you compare the agent's routing decision against the expected tool for each case.

This framework matters because tool routing is the first decision point in any multi-tool agent. If routing is wrong, downstream evaluation is meaningless — you are measuring the wrong tool's output quality.

**Build**

Load the test cases and configure the tool-selection agent from Cells 33–34:

```python
import json

# Load the 20-case tool selection dataset
with open("test_cases.json", "r") as f:
    test_cases = json.load(f)

# Preview structure
print(f"Total cases: {len(test_cases)}")
print(f"Sample: {json.dumps(test_cases[0], indent=2)}")

# Configure agent with direct tool call recording
tool_eval_agent = Agent(
    model=model,
    tools=[web_search, get_page],  # Extend with calculator, file tools as needed
    record_direct_tool_call=True
)

# Evaluate tool routing accuracy
correct = 0
results = []
for case in test_cases:
    response = tool_eval_agent(case["query"])
    selected_tool = response.tool_calls[0]["name"] if response.tool_calls else None
    is_correct = selected_tool == case["expected_tool"]
    correct += int(is_correct)
    results.append({
        "query": case["query"],
        "expected": case["expected_tool"],
        "selected": selected_tool,
        "correct": is_correct
    })

accuracy = correct / len(test_cases) * 100
print(f"Tool routing accuracy: {accuracy:.1f}%")
```

Examine which tool categories have the lowest routing accuracy and hypothesize why.

## Challenges

**Challenge: Evaluate a Multi-Step Agent**

The notebook provides three reusable assets for this challenge:

1. **Pre-built trace data** — `data/raw_traces.json` contains 100 synthetic restaurant-booking agent conversations with TOOL_CALL entries. `data/labeled_traces.json` adds success/failure state labels to the same traces.
2. **Agent scaffold** — The notebook's Strands SDK agent pattern (BedrockModel config → `Agent` with `@tool`-decorated functions like `web_search` and `get_page` → `evaluate_city_guess` evaluation function) can be extracted and run to generate fresh traces.
3. **Tool selection framework** — A 20-case dataset testing agent tool routing across `calculator`, `file_read`, `current_time`, `file_write`, and `code_interpreter` with `record_direct_tool_call=True`.

**Choose one path:**

**Path A — Use existing traces:** Work with the 100 pre-built restaurant-booking traces in `data/raw_traces.json`. These have multi-step TOOL_CALL sequences you can evaluate immediately without running an agent.

**Path B — Generate fresh traces:** Extract the notebook's Strands agent scaffold (city population lookup agent with `web_search` + `get_page` tools), run it against 5+ queries from `city_pop.csv`, and capture execution traces including tool calls, latency, and token counts.

**Then, for either path:**

1. **Define 3 evaluation metrics** applicable to the agent workflow. At least one metric must be reusable across 2+ steps (e.g., latency, confidence threshold, format compliance).
2. **Implement all 3 metrics** and score your traces.
3. **Demonstrate metric reuse** — show the same metric applied at different steps with step-appropriate thresholds or interpretations.
4. **Identify what was left out** — name 2 metrics or evaluation dimensions your framework does *not* cover, explain *why* they matter, and describe what data you'd need to implement them.

**Source assets table:**

| Asset | Location | What it provides |
|---|---|---|
| `data/raw_traces.json` | `03-agentic-metrics/data/` | 100 restaurant-booking agent traces with TOOL_CALL entries |
| `data/labeled_traces.json` | `03-agentic-metrics/data/` | Same 100 traces + `last_success_state` and `first_failure_state` labels |
| `city_pop.csv` | `03-agentic-metrics/` | Ground truth dataset (city, state, population, land_area) for agent evaluation |
| `test_cases.json` | `03-agentic-metrics/` | 2 test cases with query, expected output, category for tool selection eval |
| Agent scaffold (Cells 10-16) | Notebook | Strands `Agent` with `web_search` + `get_page` tools, `BedrockModel` config |
| `evaluate_city_guess` (Cell 8) | Notebook | Eval function: extracts XML-tagged values, computes % error vs ground truth, captures token count + latency |
| Tool selection framework (Cells 33-34) | Notebook | 20-case dataset + agent with `record_direct_tool_call=True` for tool routing accuracy |

**Assessment criteria (weighted):**

| Criterion | Weight | Excellent | Adequate |
|---|---|---|---|
| Trace source | 10% | Path B: generates clean traces with all steps, timing, outputs. Path A: correctly parses and structures pre-built traces | Traces loaded/generated but missing metadata |
| Metric design & justification | 25% | Metrics clearly motivated, mapped to agent failure modes | Metrics defined but weakly justified |
| Implementation quality | 25% | Runs, produces scores, handles edge cases | Runs but brittle or hardcoded |
| Metric reuse across steps | 20% | Same metric at 2+ steps with adapted thresholds and interpretation | Reuse attempted but superficial |
| "What was left out" analysis | 15% | Identifies meaningful gaps with data requirements and priority | Lists gaps without depth |
| Use of notebook assets | 5% | Leverages multiple provided assets (traces + ground truth + eval functions) | Uses only one asset |

**Tips:**

- The SKILL doc's `evaluate_city_guess` function is the reference implementation for Section 1
- For Section 2, stub tools are fine — the evaluation targets the agent's output format and routing, not tool correctness
- `record_direct_tool_call=True` captures the agent's first tool choice before execution — this is what you compare against ground truth
- Pre-built traces in `data/raw_traces.json` contain restaurant-booking agent conversations with TOOL_CALL entries if you prefer offline evaluation for Section 3

## Wrap-Up

In this module you built three components of an agent evaluation framework: a structured evaluation function that parses and scores agent output, a Strands SDK agent scaffold that generates traceable multi-step executions, and a tool selection accuracy framework that measures routing decisions independently of output quality.

Key takeaways:
- Agent evaluation requires metrics at multiple levels — output correctness, operational cost, and routing accuracy are complementary, not interchangeable
- Reusable metrics (latency, format compliance) applied with step-appropriate thresholds give you coverage without metric proliferation
- Knowing what your framework does *not* measure is as important as what it does

**Feedback:** What metric was hardest to implement? Which evaluation gap surprised you most? Share your findings with your workshop cohort.

**Next module:** Module 04 builds on these metrics to construct automated evaluation pipelines that run on every agent deployment.
