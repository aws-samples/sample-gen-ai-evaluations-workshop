# Challenge: Framework Deep-Dive (Option B)

This is **Option B: Deep-dive into ONE framework**. You are NOT doing a bake-off or comparison across frameworks. Instead, pick a single framework and implement a complete evaluation workflow that goes beyond what the notebook covered.

Before starting, ensure you have completed at least one SKILL file: [SKILL-promptfoo.md](./SKILL-promptfoo.md), [SKILL-agentcore.md](./SKILL-agentcore.md), [SKILL-strands.md](./SKILL-strands.md), or [SKILL-dspy.md](./SKILL-dspy.md).

---

## Eval-Focused vs Agent-Focused Distinction

**Eval-focused** (PromptFoo, DSPy): Treat the model as a function — input→output. Evaluate response quality across test cases. "Beyond" = richer assertions, CI integration, automated pass/fail gates.

**Agent-focused** (AgentCore, Strands, CW evals): Treat the system as a workflow — multiple steps, tool calls, state transitions. Evaluate process quality in addition to final output. "Beyond" = cross-step correlation, observability dashboards, failure mode analysis.

---

## The Challenge

**Challenge:** Pick one framework. Implement a complete evaluation workflow that goes beyond what the notebook covered: custom metrics, CI integration, or multi-model comparison.

---

## Workflow: Eval-Focused Frameworks

| Stage | What learner implements |
|---|---|
| Test suite definition | 10+ test cases with inputs, expected outputs, and edge cases |
| Metric configuration | 3+ metrics (at least 1 custom beyond built-in) |
| Execution | Run eval suite against a live model endpoint |
| Analysis | Score distribution, failure clustering, threshold tuning |
| Iteration | Modify prompts or config based on results, re-run, show improvement |

## Workflow: Agent-Focused Frameworks

| Stage | What learner implements |
|---|---|
| Agent instrumentation | Capture traces/spans for a multi-step agent workflow |
| Step-level metrics | Metrics per agent step (tool selection accuracy, retrieval quality, reasoning correctness) |
| End-to-end metrics | Task completion, total latency, cost |
| Failure analysis | Identify where and why the agent fails (wrong tool, bad retrieval, hallucination) |
| Observability | Dashboard or structured log output showing per-step and aggregate health |

---

## "Beyond" Examples

| Framework | Notebook covered | "Beyond" examples |
|---|---|---|
| PromptFoo | Basic prompt comparison | Custom JS/Python assertion provider; multi-model tournament; CI integration with pass/fail gates |
| DSPy | Basic optimization | Custom teleprompter; evaluation of optimized vs. unoptimized across 3+ tasks; metric composition |
| AgentCore Metrics | Predefined metrics | Custom metric plugin; cross-step metric correlation (retrieval quality → answer quality) |
| CW Agent Evals | Basic CloudWatch logging | Custom CW dashboard with alarms; anomaly detection on metric trends; automated alerting |
| Strands | Basic agent eval | Multi-agent comparison; tool-use efficiency scoring; conversation-level coherence tracking |
| AgentCore Runtime | Runtime metric capture | A/B evaluation of two agent configs; latency-quality tradeoff analysis; cost-per-quality-point metric |

---

## Scoring Rubric

| Tier | Points | Criteria |
|---|---|---|
| **Functional** | 60-69 | Complete workflow runs end-to-end; uses only notebook-level features; results are valid |
| **Extended** | 70-84 | Adds 1 capability not in notebook; clear justification for the extension |
| **Advanced** | 85-94 | Adds 2+ capabilities; demonstrates iteration (before/after comparison); addresses a real evaluation gap |
| **Exceptional** | 95-100 | Novel approach; production-quality output (CI-ready, dashboarded, or automated); teaches the reviewer something new |

---

## Assessment Criteria

| Criterion | Weight | Description |
|---|---|---|
| Complete workflow execution | 25% | All stages implemented and runnable; produces valid output |
| Beyond-notebook features | 25% | Number and quality of capabilities not covered in source notebook |
| Justification & analysis | 20% | Why each metric/feature was chosen; what evaluation gap it addresses |
| Iteration evidence | 15% | Before/after comparison showing the pipeline caught or improved something |
| "What was left out" | 10% | Identifies limitations; names what they'd need to cover them |
| Code quality & documentation | 5% | Readable, commented, reproducible |

---

## Tips for Getting Started

1. **Pick the framework you know best** — depth matters more than breadth here.
2. **Start with the notebook** — get it running, then extend one piece at a time.
3. **Define your "beyond" early** — decide what you're adding before you start coding.
4. **Document as you go** — capture why you chose each metric and what gap it fills.
5. **Show iteration** — run your eval, change something, re-run, and compare results. This is the strongest signal of understanding.
6. **Name your limitations** — the rubric rewards honesty about what's missing.
