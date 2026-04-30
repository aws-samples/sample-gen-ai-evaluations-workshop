# 04-11 Multi-Turn Chatbot Evaluations

## Overview

This module demonstrates how to evaluate multi-turn conversational AI systems using **Strands Agents Evals** and **DeepEval** with Amazon Bedrock. Real users don't interact with chatbots in single turns — they ask follow-ups, change direction, express frustration, and circle back. Evaluating these dynamic patterns requires more than static test cases with fixed inputs and expected outputs.

You'll build a travel booking assistant, simulate realistic multi-turn conversations, and evaluate them across multiple dimensions: conversation quality, context retention, goal completion, tool usage, and behavioral compliance.

## What You'll Learn

- Why multi-turn evaluation is fundamentally harder than single-turn evaluation
- How to build a multi-turn chatbot agent with Strands Agents
- How to simulate realistic conversations with Strands `ActorSimulator` and DeepEval `ConversationSimulator`
- How to evaluate conversations using Strands Evals trace-level and session-level evaluators
- How to evaluate conversations using DeepEval's multi-turn metrics
- How to generate synthetic multi-turn test cases with Strands `ExperimentGenerator`
- How to use `ToolSimulator` for fully simulated end-to-end conversations
- How to build an end-to-end multi-turn evaluation pipeline

## Notebooks

| Notebook | Description | Sections Covered | Time |
|----------|-------------|------------------|------|
| `04-11-01-intro-and-setup.ipynb` | Introduction to multi-turn evaluation concepts, environment setup, and building the travel booking assistant | 1, 2, 3 | ~15 min |
| `04-11-02-strands-simulation.ipynb` | Multi-turn conversation simulation with Strands `ActorSimulator` — creating test cases, generating actor profiles, running goal-oriented conversations, and capturing traces | 4a | ~20 min |
| `04-11-03-deepeval-simulation.ipynb` | Multi-turn conversation simulation with DeepEval `ConversationSimulator` — defining scenarios with `ConversationalGolden`, writing the model callback, and designing diverse user personas | 4b | ~20 min |
| `04-11-04-deepeval-metrics.ipynb` | DeepEval multi-turn evaluation metrics | 5a–5e | ~30 min |
| `04-11-05-strands-evaluators.ipynb` | Strands Evals trace-level evaluators (Coherence, Faithfulness, Relevance, Helpfulness), session-level evaluators (GoalSuccessRate, Interactions), and combining multiple evaluators | 6a–6c | ~25 min |
| `04-11-06-synthetic-data.ipynb` | Generating synthetic multi-turn test cases with Strands `ExperimentGenerator` and DeepEval scenarios, plus scaling strategies | 7a, 7b, 7c | ~20 min |
| `04-11-07-tool-simulation.ipynb` | Using Strands `ToolSimulator` for LLM-powered tool responses, combining `ActorSimulator` + `ToolSimulator` for fully simulated end-to-end conversations | 8 | ~20 min |
| `04-11-08-e2e-pipeline.ipynb` | End-to-end multi-turn evaluation pipeline combining all techniques, best practices, and summary | 9, 10, 11 | ~20 min |

## File Structure

```
04-11-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── 04-11-01-intro-and-setup.ipynb
├── 04-11-02-strands-simulation.ipynb
├── 04-11-03-deepeval-simulation.ipynb          # Ester
├── 04-11-04-deepeval-metrics.ipynb             # Ester
├── 04-11-05-strands-evaluators.ipynb
├── 04-11-06-synthetic-data.ipynb
├── 04-11-07-tool-simulation.ipynb
└── 04-11-08-e2e-pipeline.ipynb
```

## Prerequisites

- AWS account with Amazon Bedrock access (us-east-1)
- Access to Claude Sonnet 4 (travel booking agent) and Amazon Nova Micro (evaluation judge) models in Bedrock
- Python 3.10+
- AWS CLI configured with valid credentials (`aws configure`)

## Getting Started

1. Ensure your AWS credentials are configured: `aws configure`
2. Install dependencies: `pip install -r requirements.txt`
3. Start with `04-11-01-intro-and-setup.ipynb` and work through the notebooks in order

## Resources

- [Strands Evals SDK](https://github.com/strands-agents/evals)
- [Strands Evaluators Docs](https://strandsagents.com/docs/user-guide/evals-sdk/evaluators/)
- [DeepEval Multi-Turn Metrics](https://deepeval.com/guides/guides-multi-turn-evaluation-metrics)
- [DeepEval Multi-Turn Simulation](https://deepeval.com/guides/guides-multi-turn-simulation)
- [Anthropic — Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
