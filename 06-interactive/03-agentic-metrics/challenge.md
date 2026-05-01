# Challenge: Agentic Evaluation Metrics

Build an evaluation framework for multi-step AI agents — scoring output accuracy, measuring tool selection, and designing reusable metrics across agent execution traces.

## Prerequisites

- [SKILL: Agentic Metrics](./SKILL.md)
- Source notebook: [03-Agentic-Metrics.ipynb](../../03-agentic-metrics/03-Agentic-Metrics.ipynb)
- Data files: [city_pop.csv](../../03-agentic-metrics/city_pop.csv), [test_cases.json](../../03-agentic-metrics/test_cases.json), [data/raw_traces.json](../../03-agentic-metrics/data/raw_traces.json), [data/labeled_traces.json](../../03-agentic-metrics/data/labeled_traces.json)

## Exercise 1: Build a Custom Evaluation Function

Write an evaluation function that extracts a structured answer from agent output, compares it to ground truth, and captures operational metadata. Use the `city_pop.csv` dataset as ground truth.

**Success criteria:**
- Function parses an `<answer>` XML tag from agent output and extracts a numeric population value
- Computes percent error against the matching city in `city_pop.csv`
- Returns a dict with: `city`, `parsed` (bool), `guess`, `actual`, `error_pct`, `token_count`, `latency_s`
- Handles missing/malformed tags gracefully (returns `parsed=False` with `error_pct=None`)
- Test with at least 3 synthetic agent outputs (one well-formed, one malformed, one missing tags) and print results

## Exercise 2: Stand Up a Strands Agent and Generate Traces

Configure a Strands SDK agent with `@tool`-decorated functions (`web_search`, `get_page`) and run it against cities from `city_pop.csv`. Feed each response through your evaluation function from Exercise 1.

**Success criteria:**
- Agent is configured with `BedrockModel` and a system prompt instructing `<answer>` tag output format
- At least two `@tool` functions are defined and passed to the `Agent` constructor
- Agent is invoked for at least 3 cities from `city_pop.csv`, capturing response text, latency, and token count per invocation
- Each response is scored by the Exercise 1 evaluation function
- Print a results table: city, guess, actual, error_pct, latency_s

## Exercise 3: Measure Tool Selection Accuracy

Load the `test_cases.json` dataset and evaluate whether an agent routes queries to the correct tool. Compute overall routing accuracy and per-category accuracy.

**Success criteria:**
- Load all test cases from `test_cases.json` and print the total count and tool categories present
- Configure an agent with `record_direct_tool_call=True` (or manually capture the first tool call name)
- Run each test case through the agent and compare the selected tool against `expected_tool`
- Compute and print overall accuracy (%) and per-category accuracy breakdown
- Identify the lowest-accuracy category and hypothesize why the agent struggles with it

## Exercise 4: Design Reusable Cross-Step Metrics

Using either the pre-built traces in `data/raw_traces.json` or fresh traces from Exercise 2, define and implement 3 evaluation metrics. At least one metric must be reusable across 2+ agent steps with step-appropriate thresholds.

**Success criteria:**
- Three distinct metrics implemented as functions (e.g., latency threshold, format compliance, output accuracy)
- At least one metric is applied at two different steps with different thresholds (e.g., latency < 500ms for tool selection, latency < 3000ms for full response)
- Each metric function accepts a trace/result dict and returns a score or pass/fail
- Run all metrics across your traces and print a summary: metric name, step, pass rate
- Identify 2 evaluation dimensions your framework does NOT cover, explain why they matter, and describe what data you'd need to implement them

## Tips

- The SKILL doc's `evaluate_city_guess` function is the reference implementation for Exercise 1
- For Exercise 2, stub tools are fine — the evaluation targets the agent's output format and routing, not tool correctness
- `record_direct_tool_call=True` captures the agent's first tool choice before execution — this is what you compare against ground truth
- Pre-built traces in `data/raw_traces.json` contain restaurant-booking agent conversations with TOOL_CALL entries if you prefer offline evaluation for Exercise 4
