# 05-05: Evaluating and Optimizing Prompts with DSPy

## Overview

This module demonstrates how to use [DSPy](https://dspy.ai/) to programmatically evaluate and optimize prompts using Amazon Bedrock. Instead of hand-writing prompts, you declare **what** you want (a typed signature), define a metric that scores quality, and let an optimizer find the best prompts automatically.

The use case is meeting transcript summarization: given a meeting transcript, extract key decisions, action items, and a concise summary.

## What is DSPy?

DSPy is a framework for programming — not prompting — language models. Core concepts:

- **Signatures**: Typed input/output contracts (like function signatures for LLMs)
- **Modules**: Composable building blocks (`Predict`, `ChainOfThought`, custom subclasses)
- **Metrics**: Functions that score output quality
- **Optimizers**: Algorithms that find the best prompts by compiling from examples and metrics

The key idea: **compile, don't write**. The optimizer generates instructions and few-shot examples that outperform hand-written prompts.

## What You'll Learn

1. Configure DSPy with Amazon Bedrock
2. Define typed signatures for structured outputs
3. Build a metric to score summarization quality
4. Run systematic evaluation with `dspy.Evaluate`
5. Optimize prompts automatically with `BootstrapFewShot`
6. Compare baseline vs optimized performance

## Prerequisites

- Python 3.11+
- AWS credentials with Bedrock access
- Model access: `global.anthropic.claude-haiku-4-5-20251001-v1:0`

## Estimated Time

- Guided sections (1–9): 30–60 minutes
- ToDo sections (10–15): additional 30 minutes

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate sample data (optional — pre-generated data included)

```bash
python utils/generate_data.py
```

### 3. Run the notebook

Open `05-05-DSPy-Prompt-Optimization.ipynb` and follow along.

## Files

| File | Description |
|------|-------------|
| `05-05-DSPy-Prompt-Optimization.ipynb` | Main workshop notebook |
| `requirements.txt` | Python dependencies |
| `data/meetings.json` | Sample meeting transcripts with expected outputs |
| `utils/generate_data.py` | Script to generate synthetic meeting data via Bedrock |
| `README.md` | This documentation |

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [DSPy Signatures Guide](https://dspy.ai/learn/programming/signatures/)
- [DSPy Optimizers Guide](https://dspy.ai/learn/optimization/optimizers/)
