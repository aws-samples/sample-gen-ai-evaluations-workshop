# Challenge: Framework Deep-Dive

Pick ONE framework from this module. Implement a complete evaluation workflow
that goes beyond its entire guided module: custom metrics, CI integration, or
multi-model comparison.

This is a **deep-dive**, not a bake-off. You are not comparing frameworks against each other — you are pushing a single framework to its limits.

---

## Eval-Focused vs Agent-Focused

Your chosen framework falls into one of two paradigms. Know which one you're in — it determines what "beyond" means.

**Eval-focused** (PromptFoo, DSPy): Treat the model as a function — input→output. Evaluate response quality across test cases. "Beyond" = richer assertions, CI integration, automated pass/fail gates.

**Agent-focused** (AgentCore, Strands): Treat the system as a workflow — multiple steps, tool calls, state transitions. Evaluate process quality in addition to final output. "Beyond" = richer failure evidence, calibrated automation, production monitoring, or controlled improvement experiments.

---

## Complete Workflow: Eval-Focused Frameworks

If you chose PromptFoo or DSPy, implement all five stages:

| Stage | What You Implement |
|---|---|
| Test suite definition | 10+ test cases with inputs, expected outputs, and edge cases |
| Metric configuration | 3+ metrics (at least 1 custom beyond built-in) |
| Execution | Run eval suite against a live model endpoint |
| Analysis | Score distribution, failure clustering, threshold tuning |
| Iteration | Modify prompts or config based on results, re-run, show improvement |

**PromptFoo specifics:** Your SKILL covered `promptfooconfig.yaml`, CSV datasets, and `__expected` assertions. Go beyond exact-match — implement custom JS/Python assertion providers, use `--grader` for LLM-judged assertions, or wire `promptfoo eval` into a CI script with exit-code gates.

**DSPy specifics:** Your SKILL covered `BootstrapFewShot`, `city_metric`, and `ChainOfThought`. Go beyond single-metric optimization — compose metrics (numeric + faithfulness), compare optimized vs unoptimized across multiple task types, or implement a custom teleprompter that selects demos differently.

---

## Complete Workflow: Agent-Focused Frameworks

If you chose AgentCore or Strands, implement all five stages:

| Stage | What You Implement |
|---|---|
| Agent instrumentation | Capture traces/spans for a multi-step agent workflow |
| Step-level metrics | Observable tool choice, parameters, state transitions, span latency, and cost |
| End-to-end metrics | Output quality, task completion, total latency, and total cost |
| Failure analysis | Identify wrong tools, bad parameters, lost state, incorrect outputs, or latency/cost regressions |
| Observability | Dashboard or structured log output showing per-step and aggregate health |

**AgentCore specifics:** Your SKILL covered all six guided notebooks as one
progression: an instrumented Runtime, session/trace/tool-call levels, built-in
on-demand evaluation, reference inputs, curated datasets, focused custom
evaluators, judge calibration, batch gates, sampled online evaluation, and
simulation. Go beyond all six by implementing one production-quality extension:
a calibrated release gate with evaluator-version tracking, a monitored online
configuration with drift review, an insights-to-regression workflow, or a
promotion and rollback controller that combines evaluation effect size with
latency and cost limits.

**Strands specifics:** Your SKILL covered `Case`, `OutputEvaluator`, `TrajectoryEvaluator`, custom `Evaluator` subclasses, and `Experiment`. Go beyond — compare multiple agents on the same cases, score tool-use efficiency (fewest calls to correct answer), track coherence across multi-turn conversations, or implement `run_evaluations_async()` for CI-scale test suites.

---

## "Beyond" Examples by Framework

| Framework | Guided Module Covered | "Beyond" Examples |
|---|---|---|
| PromptFoo | Basic prompt comparison | Custom JS/Python assertion provider; multi-model tournament; CI integration with pass/fail gates |
| DSPy | Basic optimization | Custom teleprompter; evaluation of optimized vs. unoptimized across 3+ tasks; metric composition |
| AgentCore | Runtime, traces, built-ins, datasets, custom evaluators, monitoring, simulation, and optimization | Calibrated release gate; online evaluator drift review; insights converted into regression cases; automated promotion and rollback using quality plus operational limits |
| Strands | Basic agent eval | Multi-agent comparison; tool-use efficiency scoring; conversation-level coherence tracking |

---

## Scoring Rubric

| Tier | Points | Criteria |
|---|---|---|
| **Functional** | 60-69 | Complete workflow runs end-to-end; uses only notebook-level features; results are valid |
| **Extended** | 70-84 | Adds 1 capability not in notebook; clear justification for the extension |
| **Advanced** | 85-94 | Adds 2+ capabilities; demonstrates iteration (before/after comparison); addresses a real evaluation gap |
| **Exceptional** | 95-100 | Novel approach; production-quality output (CI-ready, dashboarded, or automated); teaches the reviewer something new |

---

## Assessment Criteria (Weighted)

| Criterion | Weight | Description |
|---|---|---|
| Complete workflow execution | 25% | All stages implemented and runnable; produces valid output |
| Beyond-guided-module features | 25% | Number and quality of capabilities not covered in the full guided module |
| Justification & analysis | 20% | Why each metric/feature was chosen; what evaluation gap it addresses |
| Iteration evidence | 15% | Before/after comparison showing the pipeline caught or improved something |
| "What was left out" | 10% | Identifies limitations; names what they'd need to cover them |
| Code quality & documentation | 5% | Readable, commented, reproducible |

---

## Deliverables

1. **Working code** — a notebook or script that runs end-to-end against a live endpoint
2. **Results summary** — scores, distributions, or dashboards showing what you measured
3. **Iteration log** — at least one before/after comparison (changed prompt, config, or metric → different result)
4. **Reflection** — 3-5 sentences: what evaluation gap did you address, what's still missing, what would you add with more time

---

## Getting Started

1. Pick your framework
2. Re-read the corresponding SKILL in this module
3. Reproduce the guided module baseline (this is your "Functional" tier floor)
4. Identify one thing the guided module did not operationalize that would make the evaluation more useful
5. Implement it, measure the difference, document why it matters
