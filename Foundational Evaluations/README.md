# Foundational Evaluations

These four modules form the core of the workshop. They should be completed in order, as each builds on concepts from the previous one. Together they give you a complete evaluation toolkit: measuring operational health, assessing output quality, discovering what's going wrong, and evaluating agent behavior at scale.

---

## Module Sequence

| # | Module | Core Question |
|---|--------|---------------|
| 01 | [Operational Metrics](01-operational-metrics/) | Is my system running well? (cost, latency, throughput) |
| 02 | [Quality Metrics](02-quality-metrics/) | Are the outputs actually good? (accuracy, completeness, reliability) |
| 03 | [Understanding Failures](03-understanding-failures/) | What's going wrong and what should I measure? (trace review, problem discovery) |
| 04 | [Agentic Metrics](04-agentic-metrics/) | Is my agent making good decisions? (tool selection, routing, multi-step evaluation) |

---

## 01 — Operational Metrics

**Directory:** `01-operational-metrics/`

Teaches you to measure and monitor the operational health of LLM applications on Amazon Bedrock. This is the foundation — before you can evaluate quality, you need to know whether your system is running within acceptable cost and performance bounds.

### What's Inside

| File | Purpose |
|------|---------|
| `01-Operational-Metrics.ipynb` | Main notebook covering cost, latency, TTFT/TTLT, and CloudWatch integration |
| `data/emails/` | Sample email documents for the summarization use case |
| `images/` | CloudWatch dashboard screenshots for reference |

### Key Concepts

- **Cost per request**: Calculate real-time costs from token counts using model-specific pricing. Track spending across models to make informed selection decisions.
- **Time to First Token (TTFT) vs. Time to Last Token (TTLT)**: TTFT drives perceived responsiveness — a user who sees text appearing in 300ms feels the system is fast even if the full response takes 4 seconds. TTLT determines throughput capacity. The gap between them reveals generation speed.
- **Streaming measurement**: Use `converse_stream` with `time.time()` around the iteration loop for reliable TTFT/TTLT. The built-in `response['metrics']['latencyMs']` only gives end-to-end latency without streaming granularity.
- **CloudWatch custom metrics**: Publish to a custom namespace with model ID dimensions so you can slice data per-model. Use `Sum` for cost (not Average), `Average` for latency, and `notBreaching` for `TreatMissingData` on latency alarms.
- **Multi-model comparison**: Run the same workload across Nova Lite, Nova Pro, and Claude to see the cost/latency/quality tradeoff empirically rather than guessing.

### Best Practices

- Always publish metrics on both success and failure paths — a synthetic load script that only tracks happy-path latency will miss throttling-induced spikes.
- Verify alarms fire under synthetic load before trusting them in production. If an alarm doesn't fire during a load test, your thresholds are too lenient.

---

## 02 — Quality Metrics

**Directory:** `02-quality-metrics/`

Teaches three complementary approaches to measuring output quality: programmatic testing against ground truth, LLM-as-a-Judge evaluation, and judge calibration. The critical insight is that you need all three — programmatic tests catch objective errors, judges catch subjective quality issues, and calibration ensures your judges are trustworthy.

### What's Inside

| File | Purpose |
|------|---------|
| `01_LLM_as_Judge_analysis.ipynb` | Programmatic testing + single-judge evaluation |
| `02_LLM_as_Jury_evaluation_analysis.ipynb` | Multi-judge agreement and confidence intervals |
| `03_Evaluating_your_Judge.ipynb` | Calibrate and validate judges against human labels |
| `city_pop.csv` | Ground truth dataset (US cities with population and land area) |
| `judge_benchmark.jsonl` | Human-labeled benchmark for judge calibration |
| `utils.py` | Shared evaluation utilities |

### Key Concepts

- **Programmatic testing**: When you have ground truth (known-correct answers), use exact comparison first. It's free, instant, and deterministic. Only reach for LLM-as-Judge when the output is free-form and can't be verified programmatically.
- **LLM-as-a-Judge**: Use a powerful model to score responses on multiple dimensions (accuracy, completeness, analytical quality). Each judge should evaluate a single dimension preferably with a yes/no answer.
- **Judge calibration**: A judge is just another prompt — it needs its own test set. Split human-labeled data into few-shot examples, dev set (for tuning), and test set (for final measurement). Measure TPR and TNR to quantify judge accuracy as an estimator of real model error.

### Best Practices

- **One failure mode per judge.** Don't ask a single judge prompt to check for three things at once — it will be unreliable. Build three focused evaluators that each do one thing well.
- **Binary pass/fail over numeric scales.** 1-5 Likert scales introduce inter-rater disagreement and cluster around the middle. Binary pass/fail forces clearer criteria and is easier to act on.
- **Evaluate your judge before trusting it.** Run it against human-labeled examples and measure agreement. A judge that agrees with humans 60% of the time is worse than a coin flip for borderline cases.
- **Repeatability testing.** Run the same judge on the same inputs multiple times. If verdicts flip between runs, your judge prompt needs tightening.

---

## 03 — Understanding Failures

**Directory:** `03-understanding-failures/`

Fills the gap between knowing *how* to evaluate (Module 02) and knowing *what* to evaluate. Before building automated evaluators, you need to understand what's actually going wrong. This module teaches a systematic process for reading agent traces, discovering failure patterns, and deciding whether to fix problems directly or build evaluation infrastructure.

### What's Inside

| File | Purpose |
|------|---------|
| `01_Discovering_Failure_Patterns.ipynb` | Complete workflow: trace review → grouping → prioritization → fixing → evaluator design |
| `data/raw_traces.json` | 100 restaurant booking agent conversations (shared with Module 04) |

### Key Concepts

- **Review before building evaluators.** You can't measure what you don't understand. Reading traces first tells you what to measure. Many teams skip this step and build evaluators for problems that don't exist while missing the ones that do.
- **LLM-assisted labeling is a starting point, not a substitute.** The LLM hasn't seen the traces — it works from your notes. If your notes are vague, the groupings will be vague. The value comes from your understanding of the failures, not from automating the labeling.
- **First-failure focus.** When reviewing a trace, identify the root cause rather than chasing downstream effects. An agent that confirms a cancellation after the tool returned an error has one problem (ignoring tool results), not two.
- **Frequency × severity = priority.** A problem that happens often and misleads users is more urgent than a rare cosmetic issue. Combine both dimensions into a single ranking.
- **Fix before you evaluate.** Many failures disappear with a 5-minute prompt edit. A prompt fix that eliminates a problem is always better than a multi-day evaluator build. Save LLM-as-Judge for problems that can't be fixed at the source.
- **Persona simulation for validation.** For chatbots, after editing a prompt, replay traces against the improved agent using a simulated user that follows the original conversation script. This lets you see before/after behavior at the exact failure point.

### Best Practices

- **Start with the cheapest fix.** The decision flowchart: Can you fix it with a prompt change? → Do that. Can you catch it with a simple string check? → Do that. Only build an LLM-as-Judge evaluator for problems that require judgment.
- **Make trace review a recurring practice.** Failure patterns change as you update prompts, switch models, or add features. A 30-minute weekly trace review catches more issues than any automated pipeline running unmonitored.
- **One problem → one question → one evaluator.** Each problem category should map to exactly one yes/no question. If you find yourself writing a question that covers multiple failure modes, split it.
- **The problems you discover here become your evaluation criteria.** This module's output feeds directly into Module 02's judge-building techniques and Module 04's agentic metrics.

---

## 04 — Agentic Metrics

**Directory:** `04-agentic-metrics/`

Teaches you to evaluate AI agents that use tools across multiple steps. Agents introduce evaluation challenges that don't exist for single-turn LLMs: tool selection accuracy, multi-step reasoning, operational efficiency, and the distinction between getting the right answer and getting it the right way.

### What's Inside

| File | Purpose |
|------|---------|
| `01-Agentic-Metrics.ipynb` | Main notebook: evaluation functions, agent scaffold, tool selection framework |
| `city_pop.csv` | Ground truth dataset for agent evaluation |
| `test_cases.json` | Tool selection test cases (query → expected tool mapping) |
| `data/raw_traces.json` | Pre-built restaurant booking agent traces (shared with Module 03) |
| `data/labeled_traces.json` | Same traces with success/failure state labels |
| `data/synthetic_queries_for_analysis.csv` | Synthetic queries for analysis |
| `images/architecture.png` | Architecture diagram |

### Key Concepts

- **Three levels of agent evaluation**: Output correctness (did it get the right answer?), operational efficiency (how many tokens/how much time?), and routing accuracy (did it pick the right tool?). These are complementary, not interchangeable.
- **Structured output extraction**: Agents rarely return clean numbers. Your evaluation function must parse output (from XML tags, JSON, or natural language), compare to ground truth, and capture operational metadata. The pattern is always: parse → score → annotate.
- **Tool selection accuracy**: The first decision point in any multi-tool agent. If routing is wrong, downstream evaluation is meaningless. Use `record_direct_tool_call=True` to capture which tool the agent selects before execution, then compare against expected tools using precision/recall.
- **Reusable metrics across steps**: A metric like latency or format compliance can be applied at multiple agent steps with step-appropriate thresholds. This gives you coverage without metric proliferation.
- **The distinction between correct and correct-for-the-right-reasons**: An agent might return the right answer by luck (e.g., computing manually instead of using the calculator tool). Tool selection evaluation catches this — measuring the process, not just the outcome.

### Best Practices

- **Evaluate tool routing independently of output quality.** A calculator question routed to a code interpreter may still produce a correct answer, but it reveals a routing inefficiency that compounds in production.
- **Use ground truth datasets for objective scoring.** The `city_pop.csv` pattern — known values you can compute percent error against — is the gold standard for factual accuracy evaluation.
- **Capture operational metadata alongside quality scores.** Token count and latency per evaluation let you distinguish a correct-but-slow agent from a correct-and-fast one.
- **Know what your framework doesn't measure.** Identifying evaluation gaps (and specifying what data you'd need to close them) is as important as the metrics you do implement.
- **Test multiple cities/queries for consistency.** Single-case tests hide variance. Multi-case evaluation reveals whether good performance is reliable or lucky.

---

## Shared Patterns Across All Modules

These principles recur throughout the foundational modules and apply to every evaluation you'll build in the rest of the workshop:

1. **Start simple, add complexity only when needed.** Programmatic checks before LLM judges. Single judges before juries. Prompt fixes before evaluation infrastructure.

2. **Ground truth is gold.** Whenever you have known-correct answers, use them. They're free, deterministic, and unambiguous. LLM-as-Judge is for when ground truth doesn't exist.

3. **Evaluate your evaluators.** Judges need calibration. Metrics need validation. An untested evaluator gives you false confidence — worse than no evaluator at all.

4. **Operational metrics are the foundation.** You can't evaluate quality if your system is throttled, erroring, or too slow to complete requests. Always establish operational health first.

5. **Read the traces.** No amount of automated evaluation replaces understanding what your system actually does. Regular trace review surfaces problems that metrics miss.

6. **Binary decisions over scales.** Pass/fail is clearer, more consistent, and more actionable than 1-5 ratings. When you need nuance, use multiple binary evaluators rather than one multi-point scale.

7. **Frequency × severity = priority.** Not all problems are worth solving. Focus evaluation effort on failures that happen often and cause real harm.
