# 04-11 Multi-Turn Chatbot Evaluations

## Overview

This module demonstrates how to evaluate multi-turn conversational AI systems using **Strands Agents Evals** and **DeepEval** with Amazon Bedrock. Real users don't interact with chatbots in single turns. They ask follow-ups, change direction, express frustration, and circle back. Evaluating these dynamic patterns requires more than static test cases with fixed inputs and expected outputs.

You'll build a travel booking assistant, simulate realistic multi-turn conversations with it, and evaluate those conversations against domain-specific pass/fail criteria derived from real failure modes.

## Approach

This module follows evaluation practices that have emerged as standard in the industry for production GenAI systems:

- **Evaluate against failure modes, not abstract qualities.** Generic metrics like "helpfulness" or "coherence" measure things that sound meaningful but rarely predict whether a system works for your users. We derive evaluators from concrete failure modes observed in the travel booking domain.
- **Prefer binary pass/fail over numeric scales.** 1-5 Likert scales introduce inter-rater disagreement (is this a 3 or a 4?) and tend to cluster around the middle. Binary pass/fail forces clearer criteria, produces more consistent labels, and is easier to act on.
- **Generate synthetic data along structured dimensions.** Asking an LLM for "some test queries" yields repetitive, low-coverage outputs. Defining dimensions (intent, complexity, user mood, edge cases) and combining them into tuples produces controlled, diverse coverage tied to the behaviors you care about.

## What You'll Learn

- Why multi-turn evaluation is fundamentally harder than single-turn evaluation
- How to build a multi-turn chatbot agent with Strands Agents
- How to simulate realistic conversations with Strands `ActorSimulator` and DeepEval `ConversationSimulator`
- How to design custom binary evaluators from the travel booking domain's real failure modes
- How to generate synthetic test cases from structured dimensions rather than free-form prompts
- How to use `ToolSimulator` for fully simulated end-to-end conversations
- How to build an end-to-end multi-turn evaluation pipeline that reports pass rate per failure mode

## Notebooks

| Notebook | Description | Sections Covered | Time |
|----------|-------------|------------------|------|
| `04-11-01-intro-and-setup.ipynb` | Introduction to multi-turn evaluation concepts, approach, environment setup, and building the travel booking assistant | 1, 2, 3 | ~15 min |
| `04-11-02-strands-simulation.ipynb` | Multi-turn conversation simulation with Strands `ActorSimulator`: creating test cases, generating actor profiles, running goal-oriented conversations, and capturing traces | 4a | ~20 min |
| `04-11-03-deepeval-simulation.ipynb` | Multi-turn conversation simulation with DeepEval `ConversationSimulator`: defining scenarios with `ConversationalGolden`, writing the model callback, and designing diverse user personas | 4b | ~20 min |
| `04-11-04-deepeval-metrics.ipynb` | Custom binary DeepEval metrics. `ConversationalGEval` provides pass/fail criteria and `ConversationalDAGMetric` for decision-tree logic, built from the travel booking domain's real failure modes | 5a-5e | ~30 min |
| `04-11-05-strands-evaluators.ipynb` | Custom binary Strands evaluators. `OutputEvaluator` provides pass/fail rubrics and `GoalSuccessRateEvaluator` in assertion mode. Shows how to derive evaluators from failure modes and validate them against human labels | 6a-6c | ~25 min |
| `04-11-06-synthetic-data.ipynb` | Dimension-driven synthetic data generation. Define failure hypotheses, derive dimensions, write seed tuples manually, then scale with two-step tuple-to-query generation | 7a, 7b, 7c | ~20 min |
| `04-11-07-tool-simulation.ipynb` | Using Strands `ToolSimulator` for LLM-powered tool responses, combining `ActorSimulator` + `ToolSimulator` for fully simulated end-to-end conversations | 8 | ~20 min |
| `04-11-08-e2e-pipeline.ipynb` | End-to-end multi-turn evaluation pipeline covering dimensions → simulation → custom binary evaluators → pass rate per failure mode | 9, 10, 11 | ~20 min |

## File Structure

```
04-11-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── img/
│   └── deepeval_sim.png
├── 04-11-01-intro-and-setup.ipynb
├── 04-11-02-strands-simulation.ipynb
├── 04-11-03-deepeval-simulation.ipynb
├── 04-11-04-deepeval-metrics.ipynb
├── 04-11-05-strands-evaluators.ipynb
├── 04-11-06-synthetic-data.ipynb
├── 04-11-07-tool-simulation.ipynb
└── 04-11-08-e2e-pipeline.ipynb
```

## Prerequisites

- AWS account with Amazon Bedrock access (us-east-1)
- Access to Claude Sonnet 4.5 (travel booking agent and judge) and Amazon Nova Micro (user simulator) models in Bedrock
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
- [Anthropic - Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
