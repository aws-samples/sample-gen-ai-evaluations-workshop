# Interactive Learning Mode

This directory contains AI-tutored interactive lessons for the AWS Evaluations
Workshop. Each SKILL file is a lesson plan that an AI coding assistant uses to
teach through questions, small implementation steps, and applied challenges.

## How to Use

1. Configure your AI coding tool to read [AGENTS.md](AGENTS.md).
2. Pick a skill from the map below.
3. Tell the assistant, "Teach me about [topic]."
4. Work through one question at a time; the assistant checks your reasoning
   before advancing.

## Skill Map

### Foundational Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-operational](foundational%20evaluations/SKILL-operational.md) | CloudWatch metrics, dashboards, and alarms for LLM monitoring |
| [SKILL-quality](foundational%20evaluations/SKILL-quality.md) | Binary LLM-as-a-Judge checks and judge calibration |
| [SKILL-understanding-failures](foundational%20evaluations/SKILL-understanding-failures.md) | Trace review, failure-pattern discovery, and evaluator criteria |
| [SKILL-agentic](foundational%20evaluations/SKILL-agentic.md) | Agent trace evaluation, tool-selection metrics, and metric reuse |

### Workload-Specific Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-structured-data](workload%20evals/SKILL-structured-data.md) | Document extraction accuracy scoring |
| [SKILL-guardrails](workload%20evals/SKILL-guardrails.md) | Bedrock Guardrails filters, grounding, alignment, and evaluation |
| [SKILL-rag-evaluation](workload%20evals/SKILL-rag-evaluation.md) | Retrieval and generation evaluation across text and multimodal RAG |
| [SKILL-speech-reasoning](workload%20evals/SKILL-speech-reasoning.md) | Speech-to-speech and automated reasoning verification |
| [SKILL-tool-calling](workload%20evals/SKILL-tool-calling.md) | Tool selection, parameter, trajectory, and simulation evaluation |
| [SKILL-chatbot](workload%20evals/SKILL-chatbot.md) | Multi-turn simulation, binary evaluators, and synthetic test data |
| [SKILL-red-teaming](workload%20evals/SKILL-red-teaming.md) | Adversarial testing for LLM apps, guardrails, RAG, and agents |
| [SKILL-multiagent-context](workload%20evals/SKILL-multiagent-context.md) | Shared-context quality across hub-spoke and peer-to-peer agents |
| [CHALLENGE-capstone](workload%20evals/CHALLENGE-capstone.md) | **Capstone:** integrate several workload evaluation techniques |

### Framework-Specific Evaluations

| Skill | What You'll Build |
|---|---|
| [SKILL-promptfoo](framework%20evals/SKILL-promptfoo.md) | Promptfoo YAML configuration, assertions, and model comparisons |
| [SKILL-agentcore](framework%20evals/SKILL-agentcore.md) | Run, observe, evaluate, and improve one AgentCore agent through built-ins, ground truth, custom evaluators, monitoring, and simulation |
| [SKILL-strands](framework%20evals/SKILL-strands.md) | Strands Evals cases, experiments, trajectory checks, and custom evaluators |
| [SKILL-dspy](framework%20evals/SKILL-dspy.md) | DSPy signatures, metrics, and prompt optimization loops |
| [SKILL-mlflow](framework%20evals/SKILL-mlflow.md) | MLflow experiment tracking, scorer composition, and run comparison |
| [CHALLENGE-deep-dive](framework%20evals/CHALLENGE-deep-dive.md) | **Deep-dive:** extend one framework beyond its guided module |

## Prerequisites

- AWS account with Amazon Bedrock model access
- Python 3.10 or later
- AWS credentials configured
- An AI coding assistant such as Claude Code, Kiro, or Codex

Individual skills list their additional dependencies. AgentCore requires
Node.js 20 or later, `uv`, and the current `@aws/agentcore` CLI.

## Dependencies

Complete the foundational skills in order. Workload and framework skills can
then be selected independently. See [curriculum.md](curriculum.md) for the full
learning path and the six-notebook AgentCore progression.

## Contributing a Skill

Use [meta/SKILL-BUILDER.md](meta/SKILL-BUILDER.md) to generate or revise a
lesson from source notebooks. Validate changes with:

```bash
bash meta/validate_skills.sh path/to/SKILL-name.md
```
