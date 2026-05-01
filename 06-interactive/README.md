# Interactive / Agentic Mode

This directory contains an AI-powered tutor that reads the workshop modules and presents them as guided, hands-on challenges. Instead of passively reading notebooks, you work through exercises with an AI assistant that checks your understanding, provides hints when you're stuck, and adapts to your pace.

## What It Is

An AI tutor skill file that instructs your AI assistant (Kiro, Claude, or similar) to act as an evaluations workshop facilitator. The agent reads the relevant module's notebooks and SKILL docs, then guides you through challenge exercises — asking you to write code, explain concepts, and debug configurations rather than lecturing at you.

## Prerequisites

Same as the main workshop:

- AWS account with Amazon Bedrock model access (Nova Lite, Nova Pro, or Claude 3.7 Sonnet)
- Python 3.10+ with boto3 installed
- IAM permissions for CloudWatch and Bedrock Runtime (Module 01)
- Node.js installed (Module 05 — promptfoo)
- Familiarity with Jupyter notebooks

## How to Use

1. Point your AI assistant at the skill file: `06-interactive/claude/kiro.md`
2. Tell the agent which module you want to work through (e.g., "Let's start Module 01")
3. The agent will read the relevant notebooks and SKILL docs, then present challenges one at a time
4. Write code, run it, and share your results — the agent will check your work and guide you forward

## Module Challenges

| Module | Challenge File | Topics Covered |
|--------|---------------|----------------|
| 01 — Operational Metrics | [01-operational-metrics/challenge.md](./01-operational-metrics/challenge.md) | CloudWatch custom metrics, dashboards, alarms, TTFT/TTLT |
| 02 — Quality Metrics | [02-quality-metrics/challenge.md](./02-quality-metrics/challenge.md) | LLM-as-Judge, LLM-as-Jury, agreement rates, confidence intervals |
| 03 — Agentic Metrics | [03-agentic-metrics/challenge.md](./03-agentic-metrics/challenge.md) | Agent evaluation functions, tool selection accuracy, Strands SDK |
| 04 — Workload Evals | [04-workload-evals/challenge.md](./04-workload-evals/challenge.md) | RAG retrieval metrics, faithfulness, guardrails, unified pipelines |
| 05 — Framework Evals | [05-framework-evals/challenge.md](./05-framework-evals/challenge.md) | Promptfoo YAML configs, assertions, multi-provider comparison |

## Facilitator Guide

### Recommended Delivery Order

Module 01 is recommended before Module 02. Module 02's SKILL.md lists Module 01 as a prerequisite, though the concepts are not strictly dependent.

```
Day 1:  Module 01 (Operational)  →  Module 02 (Quality)
Day 2:  Module 03 (Agentic)     →  Module 04 (Workload) — pick 2 SKILLs
Day 3:  Module 05 (Framework)   →  Capstone / Deep-Dive Challenge
```

### Time Estimates

| Module | Content | Hands-on | Challenge | Total |
|---|---|---|---|---|
| 01 — Operational Metrics | 45 min | 60 min | — | ~2 hrs |
| 02 — Quality Metrics | 45 min | 75 min | — | ~2 hrs |
| 03 — Agentic Metrics | 45 min | 60 min | — | ~2 hrs |
| 04 — Workload (2 SKILLs + capstone) | 60 min | 90 min | 60 min | ~3.5 hrs |
| 05 — Framework (2 SKILLs + deep-dive) | 60 min | 90 min | 60 min | ~3.5 hrs |

### Common Learner Issues

| Issue | Fix |
|---|---|
| `AccessDeniedException` on Bedrock | Model access not enabled — Bedrock console → Model access → Request |
| `ThrottlingException` in Module 01 | Reduce concurrency or switch to Nova Lite |
| `ModuleNotFoundError: strands_agents` | `pip install strands-agents` (hyphen, not underscore) |
| `ModuleNotFoundError: ddgs` | `pip install duckduckgo-search` (package name differs from import name) |
| PromptFoo `command not found` | `npm install -g promptfoo` — requires Node.js 18+ |
| ChromaDB sqlite3 version error | `pip install pysqlite3-binary` |
| CloudWatch metrics not appearing | 1–2 min propagation delay; verify namespace spelling |
| DSPy optimization hangs | Reduce `max_bootstrapped_demos` to 2 for workshop |