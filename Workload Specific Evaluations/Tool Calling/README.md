# 04-08 Evaluating Tool-Calling Agents Without Real Tool Execution

## Overview

This module tackles one of the hardest problems in agentic AI evaluation: **how do you rigorously evaluate an agent's tool-calling behavior without actually executing the tools?**

Real tool calls are expensive, slow, non-deterministic, and can have side effects (sending emails, modifying databases, charging credit cards). Yet tool calling is the core capability that separates an agent from a chatbot. If you can't evaluate it safely and cheaply, you can't iterate on it.

This module uses the **Amazon Bedrock Converse API** directly (no agent framework required) to demonstrate five progressively sophisticated evaluation approaches.

## The Core Insight

Every tool call involves three model decisions:

1. **Whether** to call a tool (vs. responding conversationally)
2. **Which** tool to call (tool selection)
3. **What** parameters to pass (parameter generation)

Each can be evaluated independently, at different cost points, with different techniques. You don't need real tools to evaluate any of them — just like you don't need a real chess board to evaluate whether a player chose a good move.

## The Evaluation Pyramid

| Layer | Approach | Cost | Speed | What It Catches |
|-------|----------|------|-------|-----------------|
| 1 | Static Trajectory Analysis | $0 | Instant | Tool selection errors, sequence problems, param mismatches |
| 2 | Schema Validation | $0 | Instant | Invalid params, missing fields, type errors, hallucinated values |
| 3 | Mock Tool Evaluation | $ | Seconds | Live decision errors, context-dependent mistakes |
| 4 | LLM-as-Judge | $$ | Seconds | Subjective quality, appropriateness, efficiency |
| 5 | Multi-Turn Simulation | $$$ | Minutes | End-to-end goal achievement, multi-step reasoning |
| 6 | Strands ToolSimulator + ActorSimulator | $$$$ | Minutes | Fully synthetic, stateful LLM-powered simulation |

## What You'll Learn

### Approach 1: Static Trajectory Analysis (Offline, $0)
Evaluate pre-recorded agent traces without running any model. Parse tool call sequences from recorded traces and compute trajectory metrics: tool selection precision/recall/F1, sequence correctness, parameter accuracy, and call/no-call decision accuracy.

### Approach 2: Schema-Based Parameter Validation (Offline, $0)
Validate generated parameters against JSON Schema definitions. Catches type errors, missing required fields, invalid enum values, and pattern violations — all without any LLM calls.

### Approach 3: Mock Tool Evaluation (Live Agent, Fake Tools, $)
Run the agent live using the Bedrock Converse API, but replace real tool execution with mock responses. The model makes real decisions about which tools to call and what parameters to pass, but no real side effects occur.

### Approach 4: LLM-as-Judge Trajectory Evaluation ($$)
Use a judge model to evaluate trajectory quality across multiple dimensions: tool selection appropriateness, parameter quality, sequence logic, efficiency, and response quality.

### Approach 5: Multi-Turn Simulation ($$$)
Simulate realistic multi-turn conversations where the agent handles follow-up questions and complex workflows, then evaluate the complete session trajectory.

### Approach 6: Strands ToolSimulator + ActorSimulator ($$$$)
The most sophisticated approach. Uses the Strands Evals SDK's `ToolSimulator` (LLM-powered, stateful tool response generation) combined with `ActorSimulator` (LLM-powered, goal-driven user simulation) to create a fully synthetic evaluation environment. Neither real tools nor real users are needed — the agent's decision-making is the only real thing under test.

## Key Metrics

| Metric | What It Measures | Approaches |
|--------|-----------------|------------|
| Tool Selection F1 | Did the agent pick the right tools? | 1, 3 |
| Schema Compliance | Are generated parameters valid JSON Schema? | 2, 3 |
| Parameter Accuracy | Do params match expected values? | 1, 3 |
| Call/No-Call Accuracy | Does the agent know when NOT to call? | 1, 3 |
| Sequence Match | Is the tool order correct? | 1, 3 |
| Trajectory Quality | LLM-judged overall quality (1-5) | 4 |
| Goal Success Rate | Did the user achieve their objective? | 5, 6 |
| State Consistency | Are tool responses consistent across calls? | 6 |

## File Structure

```
04-08-tool-calling/
├── README.md                              # This file
├── 04-08-Tool-Calling-Evaluation.ipynb    # Main evaluation notebook
├── requirements.txt                       # Python dependencies
└── data/
    ├── tool_definitions.json              # Tool schemas for the evaluation scenario
    ├── test_cases.json                    # Evaluation test cases with ground truth
    └── recorded_traces.json               # Pre-recorded agent traces for offline analysis
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have AWS credentials configured with access to Amazon Bedrock:
   - Claude Sonnet 4 for both the agent under test and LLM-as-Judge

3. Open `04-08-Tool-Calling-Evaluation.ipynb` and work through the approaches in order.

## Prerequisites

- AWS account with Amazon Bedrock access
- Python 3.10+
- Familiarity with the Bedrock Converse API is helpful but not required

## Key Takeaways

By completing this module, you will:
- Understand five distinct approaches to evaluating tool-calling agents without real execution
- Know when to use each approach based on cost, speed, and depth tradeoffs
- Be able to compute programmatic metrics (F1, schema compliance, sequence match) on traces
- Build a mock tool layer for safe, repeatable agent testing with the Bedrock Converse API
- Use LLM-as-Judge for subjective trajectory quality assessment
- Have a reusable evaluation pipeline applicable to any tool-calling agent
