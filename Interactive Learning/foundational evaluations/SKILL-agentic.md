---
name: agentic-metrics
description: Build evaluation metrics for multi-step agents. Activate when asked to "evaluate agent performance", "measure tool selection accuracy", "build agent metrics", "assess multi-step traces", or "create an evaluation framework for agents".
---

# Evaluating Multi-Step Agents

Build metrics that measure agent performance across accuracy, tool selection, and resource efficiency — then demonstrate how those metrics reuse across different evaluation scenarios.

## Prerequisites
- Completion of quality-metrics module (concepts: ground truth comparison, structured output parsing, error rate calculation)
- Source notebook: `../../Foundational Evaluations/04-agentic-metrics/01-Agentic-Metrics.ipynb`
- Data files: `data/raw_traces.json`, `data/labeled_traces.json`, `city_pop.csv`, `test_cases.json`
- AWS services: Amazon Bedrock (Nova Micro/Lite/Pro, Claude Haiku/Sonnet)

## Learning Objectives
By the end of this module, you will:
- Implement a ground-truth evaluation function that extracts structured outputs and computes percent error
- Build a tool selection accuracy metric using expected-vs-actual comparison
- Design a multi-city consistency metric that aggregates per-case results into summary statistics
- Reuse evaluation functions across single-model, multi-model, and multi-city scenarios
- Identify gaps in an evaluation framework and propose extensions

## Setup

```python
from strands import Agent, tool
from strands.models import BedrockModel
from botocore.config import Config
import pandas as pd, re, json, statistics, random

# Timeout configs for evaluation scenarios
quick_config = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 0})
longer_config = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 1})

# Ground truth dataset
gold_standard = pd.read_csv('city_pop.csv')
gold_standard['population'] = gold_standard['population'].astype(str).str.replace(',', '').astype(float)
gold_standard['land_area_mi2'] = gold_standard['land_area_mi2'].astype(str).str.replace(',', '').astype(float)
```

## Section 1: Ground-Truth Accuracy Metric

**Concept:** Agent evaluation starts with a measurable baseline. By requiring structured output (XML tags) from the agent, you enable automated comparison against known-correct values. This pattern — structured extraction → numeric comparison → percent error — is the foundation of every accuracy metric in this module.

**Build:**

```python
def evaluate_city_guess(city, state, chatbot_response, dataset):
    """Compare agent's structured output against ground truth dataset.
    
    Extracts <pop> and <area> XML tags from response, matches against
    dataset, returns percent errors + performance metrics.
    """
    final_msg = chatbot_response.message['content'][0]['text']
    guessed_pop = int(re.search(r'<pop>(.*?)</pop>', final_msg).group(1))
    guessed_area = float(re.search(r'<area>(.*?)</area>', final_msg).group(1))

    # Extract agent loop metrics from built-in observability
    total_tokens = chatbot_response.metrics.accumulated_usage['totalTokens']
    total_time = sum(chatbot_response.metrics.cycle_durations)
    tool_calls = sum(
        chatbot_response.metrics.tool_metrics[t].call_count
        for t in chatbot_response.metrics.tool_metrics
    )

    # Match against ground truth (handles Wikipedia annotations like [c])
    mask = (dataset['city'].str.replace(r'\[.*\]', '', regex=True)
            .str.strip().str.lower() == city.strip().lower()) & \
           (dataset['state'].str.upper() == state.upper())
    row = dataset[mask].iloc[0]

    pop_error = abs(row['population'] - guessed_pop) / row['population'] * 100
    area_error = abs(row['land_area_mi2'] - guessed_area) / row['land_area_mi2'] * 100

    return {
        'population_error_percent': round(pop_error, 2),
        'area_error_percent': round(area_error, 2),
        'total_tokens': total_tokens,
        'total_time': total_time,
        'tool_calls': tool_calls
    }
```

Source: Cell 8 of `03-Agentic-Metrics.ipynb` — full version includes city/state validation and warning handling.

## Section 2: Agent-as-Evaluator Pattern

**Concept:** Wrapping an evaluation function as a `@tool` lets a meta-agent orchestrate comparisons across models. The evaluator agent calls `eval_model` for each model, handles failures with retries, and compiles results. This pattern separates "what to measure" (the tool) from "how to orchestrate" (the agent).

**Build:**

```python
@tool
def eval_model(model_name: str) -> str:
    """Evaluate a single model on a standard city query.
    Returns formatted accuracy + performance metrics."""
    chatbot_model = BedrockModel(model_id=model_name, boto_client_config=quick_config)
    chatbot = Agent(tools=[web_search, get_page], model=chatbot_model, callback_handler=None)
    
    prompt = """How many people live in Phoenix, AZ, and what's the area in square miles?
    Include your answer in 'pop' and 'area' XML tags (numbers only)."""
    
    response = chatbot(prompt)
    result = evaluate_city_guess("Phoenix", "AZ", response, gold_standard)
    return (f"Population error: {result['population_error_percent']}%\n"
            f"Area error: {result['area_error_percent']}%\n"
            f"Tokens: {result['total_tokens']} | Time: {result['total_time']:.2f}s | "
            f"Tool calls: {result['tool_calls']}")
```

Source: Cells 10-16 — the multi-model evaluator agent uses this tool to compare Nova Micro/Lite/Pro and Claude Haiku/Sonnet.

## Section 3: Multi-Case Consistency Metric

**Concept:** A single test case can't reveal consistency issues. By evaluating the same model across multiple cities (randomly sampled), you measure variance — not just accuracy. High variance signals that the model is unreliable even when its average looks good.

**Build:**

```python
def evaluate_multiple_cities(model_name, num_cities=3):
    """Run evaluation across N random cities, return aggregate stats."""
    MAJOR_CITIES = [
        ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
        ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA")
    ]
    test_cities = random.sample(MAJOR_CITIES, num_cities)
    results = []

    for city, state in test_cities:
        chatbot = Agent(
            tools=[web_search, get_page, calculate],
            model=BedrockModel(model_id=model_name, boto_client_config=quick_config),
            callback_handler=None
        )
        prompt = f"How many people live in {city}, {state}...? Include <pop> and <area> tags."
        try:
            response = chatbot(prompt)
            results.append(evaluate_city_guess(city, state, response, gold_standard))
        except Exception as e:
            print(f"✗ Failed {city}, {state}: {e}")

    if results:
        return {
            'avg_population_error': round(statistics.mean(
                [r['population_error_percent'] for r in results]), 2),
            'avg_area_error': round(statistics.mean(
                [r['area_error_percent'] for r in results]), 2),
            'total_tokens': sum(r['total_tokens'] for r in results),
            'total_tool_calls': sum(r['tool_calls'] for r in results)
        }
```

## Section 4: Tool Selection Accuracy

**Concept:** Beyond answer quality, agents must select the RIGHT tool for each task. A tool selection metric compares expected tool usage (from a labeled dataset) against actual invocations. This catches models that "know the answer" but bypass tools, or that call irrelevant tools.

**Build:**

```python
# Labeled test dataset — each case specifies expected tool
# Source: test_cases.json + inline dataset from Cell 33
dataset = [
    {"id": 1, "input": "What is 234 + 876?", "expected_tool": "calculator"},
    {"id": 4, "input": "Read the contents of notes.txt", "expected_tool": "file_read"},
    {"id": 10, "input": "Run Python code: print(2+3)", "expected_tool": "code_interpreter"},
    {"id": 13, "input": "What is the capital of France?", "expected_tool": "none"},
    # ... 20 cases total across calculator, file_read, file_write, code_interpreter, none
]

def measure_tool_selection(agent, dataset):
    """Evaluate tool selection accuracy across labeled test cases."""
    results = []
    for case in dataset:
        response = agent(case["input"])
        used_tools = [
            name for name, metric in response.metrics.tool_metrics.items()
            if metric.call_count > 0
        ]
        results.append({
            "expected": case["expected_tool"],
            "actual": used_tools,
            "correct": case["expected_tool"] in used_tools or 
                      (case["expected_tool"] == "none" and not used_tools)
        })
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    return accuracy, results
```

Source: Cells 33-34 of `03-Agentic-Metrics.ipynb` — full version includes 20 test cases and detailed per-case reporting.

## Section 5: Metric Reuse Across Evaluation Scenarios

**Concept:** The power of well-designed metrics is reuse. `evaluate_city_guess` appears in three contexts: single-model baseline (Section 1), multi-model comparison via agent orchestration (Section 2), and multi-city consistency (Section 3). The same function, different orchestration patterns. This is the key insight: separate measurement from orchestration.

**Build:**

```python
# Reuse pattern: same metric, three orchestration modes

# Mode 1: Direct call (single baseline)
result = evaluate_city_guess("New York", "NY", response, gold_standard)

# Mode 2: Wrapped as @tool for agent orchestration
@tool
def eval_model(model_name: str) -> str:
    # ... calls evaluate_city_guess internally
    
# Mode 3: Loop for consistency measurement  
def evaluate_multiple_cities(model_name, num_cities=3):
    # ... calls evaluate_city_guess per city

# The metric function never changes — only the caller does.
# This enables: add new orchestration patterns without touching measurement logic.
```

## Challenges

### Challenge: Evaluate a Multi-Step Agent

Choose **Path A** (existing traces) or **Path B** (generate fresh). Define 3 metrics, implement, demonstrate reuse, identify 2 gaps.

**Path A — Existing Traces:** Use `data/raw_traces.json` and `data/labeled_traces.json` to define metrics that evaluate the multi-turn agent conversations. Traces contain tool calls, errors, and recovery patterns.

**Path B — Generate Fresh:** Build a Strands agent (using the scaffold from Cells 10-16), run it against `test_cases.json`, capture traces, then evaluate.

**Assessment criteria (weighted):**

| Criterion | Weight |
|---|---|
| Trace source | 10% |
| Metric design & justification | 25% |
| Implementation quality | 25% |
| Metric reuse across steps | 20% |
| "What was left out" analysis | 15% |
| Use of notebook assets | 5% |

**Available assets:**
- `data/raw_traces.json` — multi-turn agent conversations with tool calls and errors
- `data/labeled_traces.json` — same traces with quality labels
- `city_pop.csv` — ground truth for accuracy comparison
- `test_cases.json` — labeled queries with expected tools/outputs
- Cell 8: `evaluate_city_guess` — reference implementation for structured evaluation
- Cells 10-16: Agent scaffold with multi-model orchestration
- Cells 33-34: Tool selection framework with labeled dataset

## Wrap-Up

**Key takeaways:**
- Structured output (XML tags) enables automated accuracy measurement against ground truth
- Wrapping metrics as `@tool` lets agents orchestrate complex evaluation workflows
- The same metric function reuses across single, multi-model, and multi-city scenarios — separate measurement from orchestration

**What this does NOT cover:**
- LLM-as-a-Judge qualitative evaluation (shown in notebook §12 but not taught here)
- Semantic similarity scoring for free-text outputs
- Continuous evaluation pipelines or A/B testing frameworks
- Cost optimization strategies beyond token counting
- Real-time monitoring or alerting on metric degradation

**Next steps:**
- Module 04 (Guardrails): Apply evaluation patterns to safety and compliance constraints
- Module 05 (PromptFoo): Integrate metrics into automated testing frameworks
