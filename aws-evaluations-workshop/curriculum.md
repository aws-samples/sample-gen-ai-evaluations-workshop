# AWS Evaluations Workshop

## Overview

This workshop teaches you how to evaluate Large Language Model (LLM) applications on AWS using Amazon Bedrock. You'll progress from basic operational metrics through quality assessment, agentic evaluation, workload-specific testing, and framework-integrated evaluation pipelines. Designed for ML engineers, solution architects, and developers building production LLM applications who need systematic evaluation approaches.

## Module Dependency Map

```
Module 01: Operational Metrics ─────────────────┐
  (no prerequisites)                             │
                                                 ├──▶ Module 04: Workload-Specific
Module 02: Quality Metrics ─────────────────────┤     (recommends 01 + 02)
  (no prerequisites)                             │
                                                 │
Module 03: Agentic Metrics ─────────────────────┼──▶ Module 05: Framework-Specific
  (recommends Module 01)                         │     (recommends 01 + 02 + 03)
                                                 │
                                                 │
  ┌──────────────────────────────────────────────┘
  │
  ▼
Modules 04 & 05 are independent of each other.
Pick based on your workload or framework.
```

## Module Summaries

| Module | Title | Description | Notebooks | Key Skills |
|--------|-------|-------------|-----------|------------|
| 01 | Operational Metrics | Measure cost, latency (TTFT/TTLT), throttling, and throughput for Bedrock models using email summarization | 01-Operational-Metrics.ipynb | Token cost analysis, latency profiling, throughput measurement, model comparison |
| 02 | Quality Metrics | Evaluate response quality using LLM-as-Judge and LLM-as-Jury patterns against ground truth | 01_LLM_as_Judge_analysis.ipynb, 02_LLM_as_Jury_evaluation_analysis.ipynb | Programmatic testing, LLM-as-Judge setup, multi-judge consensus, evaluation visualization |
| 03 | Agentic Metrics | Evaluate agent performance, tool execution reliability, and resource efficiency | 03-Agentic-Metrics.ipynb | Agent accuracy measurement, tool selection analysis, token/latency tracking, reliability testing |
| 04 | Workload-Specific Evaluations | Apply evaluation techniques to specific workloads: structured data extraction, guardrails, RAG, multimodal RAG, speech-to-speech, and automated reasoning | 04-01 through 04-06 (11 notebooks) | Precision/recall/F1 for extraction, guardrail cost tracking, RAG component evaluation, multimodal assessment, speech pipeline evaluation, reasoning verification |
| 05 | Framework-Specific Evaluations | Integrate evaluations into frameworks: PromptFoo, AgentCore, Strands, and DSPy | 05-01 through 05-05 (8 notebooks) | PromptFoo test harness, AgentCore metrics with CloudWatch, Strands Evals SDK, runtime evaluation, DSPy prompt optimization |

## Module 04 Breakdown

| Sub-module | Topic | Notebook(s) |
|------------|-------|-------------|
| 04-01 | Structured Data Extraction | 04-01-Simple-structured-data-evaluation.ipynb |
| 04-02 | Guardrails | 04-02-01 through 04-02-06 (5 notebooks: filters, grounding, alignment, operational, evaluation) |
| 04-03 | Basic RAG | 04-03-Basic-RAG-Evaluation.ipynb |
| 04-04 | Multimodal RAG | 04-04-01-Multimodal-RAG.ipynb |
| 04-05 | Speech-to-Speech | s2s_entire_eval_pipeline.ipynb |
| 04-06 | Automated Reasoning | 04-06-01-automated-reasoning-evaluation.ipynb |

## Module 05 Breakdown

| Sub-module | Topic | Notebook(s) |
|------------|-------|-------------|
| 05-01 | PromptFoo | 05-01-Promptfoo-basic.ipynb |
| 05-02 | AgentCore | 05-02-01-Agentic-Metrics-AgentCore.ipynb, 05-02-02-Agent-and-tool-evals-with-cwlogs.ipynb |
| 05-03 | Strands | 05-03-Strands-Evals.ipynb |
| 05-04 | AgentCore Runtime | 05-04-AgentCore-Runtime-Evals.ipynb |
| 05-05 | DSPy | 05-05-DSPy-Prompt-Optimization.ipynb |

## IAM Permissions Required

All modules combined require these IAM actions:

```
bedrock:InvokeModel
bedrock:InvokeModelWithResponseStream
bedrock:CreateGuardrail
bedrock:TagResource
bedrock:ApplyGuardrail
cloudwatch:PutMetricData
logs:FilterLogEvents
ecr:DescribeRepositories
```

Attach these to your workshop IAM role or user. For least-privilege, scope `bedrock:*` actions to specific model ARNs in your region.

## Setup Instructions

### Prerequisites

- AWS account with Amazon Bedrock model access enabled (Claude, Titan)
- Python 3.10+
- AWS CLI configured with credentials

### Installation

```bash
# Clone the repository
git clone <repo-url> && cd evals-workshop

# Install core dependencies
pip install boto3 pandas matplotlib numpy

# Module-specific installs
pip install promptfoo          # Module 05-01
pip install strands-agents     # Module 05-03
pip install strands-agents-evals
pip install dspy               # Module 05-05
```

### AWS Configuration

```bash
aws configure
# Set your default region (us-east-1 or us-west-2 recommended for Bedrock)

# Verify Bedrock access
aws bedrock list-foundation-models --query "modelSummaries[0].modelId"
```

### Running Notebooks

```bash
pip install jupyter
jupyter notebook
```

Navigate to any module directory and open the notebook. Each module is self-contained — start with Module 01 or 02 if new to LLM evaluation.
