# AWS Evaluations Workshop

A hands-on workshop covering end-to-end evaluation strategies for LLM-powered applications on Amazon Bedrock. Participants progress from basic operational metrics through quality and agentic evaluations, then apply these techniques to real-world workloads and popular agent frameworks.

## Module Dependency Map

```
Module 01 (Operational Metrics) ──┐
                                  ├──→ Module 04 (Workload-Specific)
Module 02 (Quality Metrics) ──────┤
         │                        ├──→ Module 05 (Framework-Specific)
         └──→ Module 03 (Agentic) ┘
```

- **Module 01**: No prerequisites
- **Module 02**: No prerequisites (benefits from Module 01)
- **Module 03**: Requires Module 02 (quality metrics concepts)
- **Module 04**: Requires Modules 01–03
- **Module 05**: Requires Modules 01–03

## Module 01: Operational Metrics

**Notebooks:** `01-Operational-Metrics.ipynb`

Measure and analyze key operational metrics for LLMs in Amazon Bedrock using an email summarization use case. Covers cost metrics (token usage/pricing), latency (TTFT vs TTLT), throttling, throughput, and publishing custom metrics to CloudWatch.

**Key skills:** Token cost analysis, latency profiling, CloudWatch custom metrics, model comparison

## Module 02: Quality Metrics

**Notebooks:** `01_LLM_as_Judge_analysis.ipynb`, `02_LLM_as_Jury_evaluation_analysis.ipynb`

Evaluate LLM output quality using programmatic testing against ground truth and LLM-as-a-Judge methodology. The second notebook extends to multi-judge (jury) evaluation for more robust quality assessment on US cities demographic analysis.

**Key skills:** LLM-as-a-Judge, LLM-as-a-Jury, programmatic accuracy testing, evaluation visualization

## Module 03: Agentic Metrics

**Notebooks:** `03-Agentic-Metrics.ipynb`

Comprehensive evaluation of agent performance across four dimensions: accuracy against ground truth datasets, tool selection and execution success rates, resource efficiency (tokens, latency, cycles), and reliability across test scenarios.

**Key skills:** Agent accuracy evaluation, tool execution analysis, resource efficiency metrics, reliability testing

## Module 04: Workload-Specific Evaluations

### 04-01 Intelligent Document Processing
**Notebook:** `04-01-Simple-structured-data-evaluation.ipynb`

Evaluate structured data extraction models using precision, recall, and F1-score at field and document levels for invoice processing.

### 04-02 Guardrails
**Notebooks:** `04-02-01-filters.ipynb`, `04-02-02-grounding.ipynb`, `04-02-03-alignment.ipynb`, `04-02-04-operational.ipynb`, `04-02-06-evaluation.ipynb`

Evaluate Bedrock Guardrails across content filtering, grounding checks, alignment policies, and operational metrics for a city government chatbot scenario.

### 04-03 Basic RAG
**Notebook:** `04-03-Basic-RAG-Evaluation.ipynb`

Evaluate a RAG system's retrieval (recall@k, precision@k) and end-to-end generation quality using LLM-as-a-Judge with ChromaDB as the vector store.

### 04-04 MultiModal RAG
**Notebook:** `04-04-01-Multimodal-RAG.ipynb`

Enhanced RAGAS implementation combining traditional retrieval metrics with custom multimodal metrics using ImageBind embeddings for text, vision, and audio.

### 04-05 Speech-to-Speech
**Notebook:** `s2s_entire_eval_pipeline.ipynb`

End-to-end evaluation pipeline for speech-to-speech applications: automated Playwright testing, session annotation, and conversation flow evaluation.

### 04-06 Automated Reasoning
**Notebook:** `04-06-01-automated-reasoning-evaluation.ipynb`

Evaluate Automated Reasoning Checks in Bedrock Guardrails — verifying LLM outputs against formal policy rules using SMT solver validation with 7 result types.

**Module 04 key skills:** Domain-specific evaluation design, guardrail testing, RAG metrics, multimodal evaluation, speech pipeline testing, formal verification

## Module 05: Framework-Specific Evaluations

### 05-01 PromptFoo
**Notebook:** `05-01-Promptfoo-basic.ipynb`

Set up and run evaluations using the PromptFoo CLI with Amazon Bedrock for an email classification system.

### 05-02 AgentCore
**Notebooks:** `05-02-01-Agentic-Metrics-AgentCore.ipynb`, `05-02-02-Agent-and-tool-evals-with-cwlogs.ipynb`

Deploy a Strands agent to AgentCore Runtime and evaluate with multi-dimensional quality assessment, tool usage analysis, and CloudWatch log-based metrics.

### 05-03 Strands
**Notebook:** `05-03-Strands-Evals.ipynb`

Use the Strands Agents Evaluation SDK with OutputEvaluator, TrajectoryEvaluator, and custom evaluators for structured agent evaluation.

### 05-04 AgentCore Runtime
**Notebook:** `05-04-AgentCore-Runtime-Evals.ipynb`

Deploy agents to AgentCore Runtime and evaluate using the native AgentCore Evaluations API with built-in evaluators (Helpfulness, ToolSelectionAccuracy).

### 05-05 DSPy
**Notebook:** `05-05-DSPy-Prompt-Optimization.ipynb`

Use DSPy for programmatic prompt optimization — replacing manual prompt engineering with metric-driven optimization on Bedrock.

**Module 05 key skills:** Framework integration, CI/CD evaluation pipelines, agent deployment, prompt optimization

## IAM Permissions Required

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

## Setup Instructions

### AWS Configuration

```bash
aws configure
# Set region to us-east-1 or us-west-2 (Bedrock model availability)
# Ensure IAM role/user has the permissions listed above
# Enable model access in the Bedrock console for Claude and Nova models
```

### Python Dependencies

```bash
pip install boto3 pandas numpy

# Module 03 - Agentic Metrics
pip install strands-agents duckduckgo-search beautifulsoup4

# Module 04 - Workload-Specific
pip install chromadb llama-index faiss-cpu  # RAG modules
pip install playwright                      # Speech-to-Speech

# Module 05 - Framework-Specific
npm install -g promptfoo                    # 05-01
pip install bedrock-agentcore               # 05-02, 05-04
pip install strands-agents strands-agents-evals  # 05-03
pip install dspy                            # 05-05
```

### Verify Setup

```bash
aws bedrock list-foundation-models --query "modelSummaries[?modelId=='anthropic.claude-3-sonnet-20240229-v1:0']"
python -c "import boto3; print(boto3.client('bedrock-runtime').meta.region_name)"
```
