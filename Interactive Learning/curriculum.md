# AWS Evaluations Workshop Curriculum

## Overview

This workshop teaches systematic evaluation for generative AI applications on
AWS. Learners first build a shared foundation, then choose workload-specific or
framework-specific modules based on what they are building.

## Learning Path

Complete the four foundational modules in order:

```text
01 Operational Metrics
        |
02 Quality Metrics
        |
03 Understanding Failures
        |
04 Agentic Metrics
        |
        +--> Workload-Specific Evaluations
        |
        +--> Framework-Specific Evaluations
```

The workload and framework branches are independent. Choose modules from either
branch after the foundations.

## Foundational Modules

| Order | Topic | Source notebook(s) | Interactive skill |
|---:|---|---|---|
| 01 | Operational Metrics | `../Foundational Evaluations/01-operational-metrics/01-Operational-Metrics.ipynb` | `foundational evaluations/SKILL-operational.md` |
| 02 | Quality Metrics | `../Foundational Evaluations/02-quality-metrics/01_LLM_as_Judge_analysis.ipynb`, `03_Evaluating_your_Judge.ipynb` | `foundational evaluations/SKILL-quality.md` |
| 03 | Understanding Failures | `../Foundational Evaluations/03-understanding-failures/01_Discovering_Failure_Patterns.ipynb` | `foundational evaluations/SKILL-understanding-failures.md` |
| 04 | Agentic Metrics | `../Foundational Evaluations/04-agentic-metrics/01-Agentic-Metrics.ipynb` | `foundational evaluations/SKILL-agentic.md` |

## Workload-Specific Modules

| Topic | Source directory | Interactive skill |
|---|---|---|
| Intelligent Document Processing | `../Workload Specific Evaluations/Intelligent Document Processing/` | `workload evals/SKILL-structured-data.md` |
| Guardrails | `../Workload Specific Evaluations/Guardrails/` | `workload evals/SKILL-guardrails.md` |
| Basic and Multimodal RAG | `../Workload Specific Evaluations/Basic RAG/`, `MultiModal RAG/` | `workload evals/SKILL-rag-evaluation.md` |
| Speech and Automated Reasoning | `../Workload Specific Evaluations/Speech to Speech/`, `Automated Reasoning Evaluations/` | `workload evals/SKILL-speech-reasoning.md` |
| Tool Calling | `../Workload Specific Evaluations/Tool Calling/` | `workload evals/SKILL-tool-calling.md` |
| Multi-Turn Chatbot | `../Workload Specific Evaluations/Chatbot/` | `workload evals/SKILL-chatbot.md` |
| Red Teaming | `../Workload Specific Evaluations/Red Teaming/` | `workload evals/SKILL-red-teaming.md` |
| Multi-Agent Shared Context | `../Workload Specific Evaluations/Multiagent Shared Context Evaluation/` | `workload evals/SKILL-multiagent-context.md` |

The workload capstone is `workload evals/CHALLENGE-capstone.md`.

## Framework-Specific Modules

| Framework | Source notebook(s) | Interactive skill |
|---|---|---|
| Promptfoo | `../Framework Specific Evaluations/Prompt Foo/01 Promptfoo basic.ipynb` | `framework evals/SKILL-promptfoo.md` |
| Amazon Bedrock AgentCore | Six-notebook progression described below | `framework evals/SKILL-agentcore.md` |
| Strands Evals | `../Framework Specific Evaluations/Strands/01 Strands Evals.ipynb` | `framework evals/SKILL-strands.md` |
| DSPy | `../Framework Specific Evaluations/DSPy/01 DSPy Prompt Optimization.ipynb` | `framework evals/SKILL-dspy.md` |
| MLflow | `../Framework Specific Evaluations/MLflow/01 Mlflow Evaluation.ipynb` | `framework evals/SKILL-mlflow.md` |
| DeepEval | `../Framework Specific Evaluations/DeepEval/01 DeepEval RAG Evaluation.ipynb` | Source workshop module; no Interactive Learning skill yet |

The framework extension is `framework evals/CHALLENGE-deep-dive.md`.

## AgentCore Learning Progression

AgentCore is one cumulative module, not separate metrics and Runtime modules.
Every notebook reuses the deterministic `CityAnalyst` Runtime deployed in
notebook 01.

| Order | Notebook | Learning checkpoint |
|---:|---|---|
| 01 | `01-agentcore-foundations.ipynb` | Connect Runtime, Observability, Evaluations, CLI, and SDK; deploy once and inspect session/trace/tool spans |
| 02 | `02-built-in-on-demand-evaluations.ipynb` | Select built-ins by failure mode and level; run targeted CLI and SDK evaluations |
| 03 | `03-ground-truth-and-datasets.ipynb` | Add expected responses, assertions, trajectories, and curated single-turn/multi-turn datasets |
| 04 | `04-custom-evaluators.ipynb` | Build focused binary LLM judges and deterministic code evaluators; calibrate against human labels |
| 05 | `05-batch-and-online-evaluation.ipynb` | Add historical batch scoring, fail-closed release gates, and sampled online monitoring |
| 06 | `06-simulation-and-optimization.ipynb` | Simulate harder users, analyze failures, review recommendations, and validate changes through controlled experiments |

Notebooks 01-04 form the core path. Notebooks 05-06 are production and advanced
extensions.

### AgentCore Mental Model

| Layer | Responsibility |
|---|---|
| Runtime | Hosts and scales the agent loop supplied by the developer |
| Observability | Emits OpenTelemetry-compatible sessions, traces, model spans, and tool spans |
| Evaluations | Scores behavior at `SESSION`, `TRACE`, or `TOOL_CALL` level |
| AgentCore CLI | Manages project configuration, deployment, traces, evaluators, datasets, and jobs |
| Python SDK | Automates targeted evaluations, dataset runners, simulation, and custom evaluator logic |

## Prerequisites

- AWS account with Amazon Bedrock model access
- Python 3.10 or later
- AWS CLI configured with credentials
- An AI coding assistant that can read `Interactive Learning/AGENTS.md`

AgentCore additionally requires:

- Node.js 20 or later
- `uv`
- Permissions for AgentCore, CloudFormation, IAM, CloudWatch, X-Ray, and Bedrock
  model invocation

## Installation

Core workshop environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install boto3 pandas matplotlib numpy jupyterlab
```

AgentCore environment:

```bash
python -m pip uninstall -y bedrock-agentcore-starter-toolkit
npm install -g @aws/agentcore
python -m pip install \
  -r "Framework Specific Evaluations/AgentCore/requirements.txt"
```

Some Python environments may expose a conflicting `agentcore` command through
`bedrock-agentcore-starter-toolkit`. Removing it ensures the workshop uses the
Node.js CLI.

## AWS Configuration

```bash
aws configure
aws sts get-caller-identity
aws bedrock list-foundation-models \
  --query "modelSummaries[0].modelId"
```

Use workshop or development accounts for modules that create resources. Scope
permissions and resource ARNs for production rather than copying broad workshop
roles.
