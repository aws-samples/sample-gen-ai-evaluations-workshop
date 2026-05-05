# 07 Understanding Failures

## Overview

Before you can build good evaluators, you need to know what problems to look for. This module teaches you how to systematically review agent traces, spot what's going wrong, and decide whether to fix problems directly or build automated checks. It answers the practical question that comes up between measuring quality (Module 02) and evaluating agent behavior (Module 03): **"What should I actually be measuring?"**

Module 02 teaches you *how* to measure quality with techniques like LLM-as-a-Judge. Module 03 shows you *how* to evaluate agent behavior at scale. This module fills the gap between them — it gives you a hands-on process for figuring out *what* to measure in the first place. You'll read through real agent conversations, write down what went wrong, group similar problems together, and then take action: fix the easy ones immediately and turn the rest into evaluator criteria.

The module uses traces from a restaurant booking agent — the same agent and data you'll work with in Module 03. This creates a continuous thread across the workshop: you discover the problems here, then build the metrics to track them there.

## Notebooks

| Notebook | Description |
|---|---|
| **01_Discovering_Failure_Patterns** | Walk through agent traces, write notes on failures (with LLM-assisted suggestions), group problems, prioritize by impact, fix the top issue with a prompt change using persona simulation, and bridge one problem into an evaluator prompt |

## What You'll Learn

This hands-on workshop covers a complete workflow for understanding how your agent fails before building evaluation infrastructure:

> **⚠️ A note on LLM-assisted labeling**
>
> This notebook uses LLMs to suggest labels at several points: individual trace notes, problem category groupings, and severity ratings. These suggestions are included as a convenience to speed up the workflow and show you what LLM-assisted annotation looks like in practice. **They are not a substitute for human judgment.**
>
> We strongly recommend that all labels — trace notes, category assignments, and severity ratings — undergo human review. In both this workshop and real workloads, the LLM's suggestions are a starting point. You should read the traces yourself, verify the LLM's observations, and correct anything it gets wrong. The value of this process comes from *your* understanding of the failures, not from automating the labeling.

### 1. Reviewing Traces and Writing Notes
Read through agent conversations one at a time and write down what went wrong:
- **Trace Display**: View conversations with clear turn numbers, role labels, and highlighted tool call results
- **LLM-Assisted Labeling**: Each trace gets an LLM-suggested note that you can accept or override with your own observation
- **First-Failure Focus**: Learn to identify the root cause rather than chasing downstream effects

### 2. Grouping Similar Problems Together
Use LLM assistance to organize your notes into actionable problem categories:
- **LLM-Assisted Clustering**: Let Claude suggest initial groupings from your notes
- **Human Refinement**: Review, merge, split, and rename categories until they're specific enough to act on
- **Frequency Counting**: See which problems show up most often across traces

### 3. Prioritizing What to Fix First
Rank problems by how often they occur and how much damage they cause:
- **Severity Rating**: Classify each problem from cosmetic to critical
- **Priority Scoring**: Combine frequency and severity into a single ranking
- **Decision Flowchart**: A practical guide for choosing between prompt fixes, simple checks, and LLM-as-Judge evaluators

### 4. Fixing Problems Directly
Fix the highest-priority problem by editing the agent's system prompt:
- **Root Cause Identification**: Find the gap in the prompt that causes the failure
- **Persona Simulation**: Replay traces against the improved agent using a simulated user that follows the original conversation script
- **Before/After Comparison**: See how the agent's behavior changes at the exact point where it previously failed
- **Cost-Effective Fixes**: See that many failures disappear with a 5-minute prompt edit — no evaluation infrastructure needed

### 5. Bridging into Evaluator Design
Turn a discovered problem into a binary pass/fail evaluator prompt:
- **Binary Question Writing**: Convert a problem category into a single yes/no question
- **Judge Prompt Sketching**: Draft an evaluator prompt that uses your question to assess traces
- **Connection to Module 02**: See how the problems you found here become the criteria for the LLM-as-Judge evaluators taught in Module 02

## Connections to Other Modules

This module sits at a key point in the workshop — it connects what you've already learned to what comes next:

- **Module 01 — Operational Metrics** ([01-Operational-Metrics.ipynb](../01-operational-metrics/01-Operational-Metrics.ipynb)): Operational anomalies like high latency or unusual token counts are signals for *where* to look during trace review. If a set of conversations shows unusually high token usage, that's a good place to start reading.

- **Module 02 — Quality Metrics** ([01_LLM_as_Judge_analysis.ipynb](../02-quality-metrics/01_LLM_as_Judge_analysis.ipynb), [03_Evaluating_your_Judge.ipynb](../02-quality-metrics/03_Evaluating_your_Judge.ipynb)): The failure modes you discover here become the criteria for LLM-as-Judge evaluators. The binary pass/fail pattern used in the evaluator bridge section of this module is taught in depth in Module 02's judge calibration notebook.

- **Module 03 — Agentic Metrics** ([03-Agentic-Metrics.ipynb](../03-agentic-metrics/03-Agentic-Metrics.ipynb)): This module uses the same restaurant booking agent traces that Module 03 uses for agentic evaluation. The problem categories you build here directly inform what metrics to track there.

## Shared Data

The `data/` directory contains `raw_traces.json` — restaurant booking agent conversations also used in Module 03 (Agentic Metrics). This is intentional: you discover the problems in this module, then build the metrics to track them in Module 03. Both modules follow the same agent through the same conversations, giving you a continuous narrative across the workshop.

## Prerequisites

- AWS account with Amazon Bedrock access
- Access to Claude on Amazon Bedrock in your region
- Python 3.10+ with `strands-agents` and `boto3` libraries
- Recommended: completion of Module 01 (Operational Metrics) and Module 02 (Quality Metrics) — the concepts from those modules are referenced throughout

## Key Takeaways

By completing this module, you will:
- Know how to review agent traces systematically and write useful notes about what went wrong
- Be able to group similar problems into categories that are specific enough to act on
- Understand how to prioritize problems by frequency and severity so you focus on what matters most
- See that many failures can be fixed with a simple prompt change — before building any evaluation infrastructure
- Know how to turn a discovered problem into a binary pass/fail evaluator prompt, connecting your findings to the evaluation techniques in Module 02
