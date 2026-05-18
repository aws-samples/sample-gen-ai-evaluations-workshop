# DeepEval

## Overview

[DeepEval](https://deepeval.com/) is an open-source framework for evaluating LLM applications, built around four design principles: local-first, pytest-native, trace-aware, and composable. It ships with 50+ built-in metrics spanning RAG, agents, tool use, multi-turn conversations, safety, and multimodal applications, and supports custom metrics in natural language when the defaults don't fit.

This module covers LLM evaluation with DeepEval, starting with RAG pipeline evaluation and expanding to other use cases over time.

## Notebooks

| Notebook | Description | Time |
|---|---|---|
| `01 DeepEval RAG Evaluation.ipynb` | End-to-end RAG evaluation | ~60 min |

## What You'll Learn

- Structure an evaluation as `LLMTestCase` objects and score them with metrics
- Use `AmazonBedrockModel` so any model can serve as the DeepEval judge
- Evaluate retrieval quality separately from generation quality
- Mix built-in RAG metrics with custom G-Eval metrics for failure modes the built-ins don't cover
- Cross-reference retrieval and generation scores to diagnose where failures originate
- Compare candidate models and read the judge's reasons on failures

## Metrics

| Stage | Criterion | Metric |
|---|---|---|
| Retrieval | Relevant | `ContextualRelevancyMetric` |
| Retrieval | Sufficient | `ContextualRecallMetric` |
| Generation | On-topic | `AnswerRelevancyMetric` |
| Generation | Grounded | `FaithfulnessMetric` |
| Generation | Factually correct | `Correctness` (G-Eval) |
| Generation | Complete | `Completeness` (G-Eval) |

## Dataset

`data/aws_qa.json` — 20 examples covering conceptual AWS topics across compute, storage, networking, identity, and operations. Each example has a question, a grounding passage, a reference answer, and a source URL.

## Models

| Role | Model |
|---|---|
| Candidate #1 | `us.amazon.nova-2-lite-v1:0` |
| Candidate #2 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Judge | `us.anthropic.claude-sonnet-4-6` |

## Prerequisites

- AWS account with access to the three Bedrock models listed above
- Python 3.10+
- AWS CLI configured (`aws configure` or `aws login`)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
aws sts get-caller-identity
```

Then open `01 DeepEval RAG Evaluation.ipynb` and run the cells in order (~60 min including ~5 min of model call time).

## Resources

- [DeepEval documentation](https://deepeval.com/docs/introduction)
- [DeepEval metrics reference](https://deepeval.com/docs/metrics-introduction)
- [G-Eval](https://deepeval.com/docs/metrics-llm-evals)
- [DeepEval RAG evaluation guide](https://deepeval.com/guides/guides-rag-evaluation)
- [Amazon Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
