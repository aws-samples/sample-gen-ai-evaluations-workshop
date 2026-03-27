# 05-05: Evaluating and Optimizing Prompts with DSPy

## Overview

This module demonstrates how to use [DSPy](https://dspy.ai/) to programmatically evaluate and optimize prompts using Amazon Bedrock. Instead of hand-writing prompts, you declare **what** you want (a typed signature), define a metric that scores quality, and let an optimizer find the best prompts automatically.

The use case is **city Q&A**: given a dataset of ~300 US cities with population and land area data, build an LLM system that answers factual questions accurately — scored against ground-truth values.

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
3. Build a numeric accuracy metric (% error with 5%/10% thresholds)
4. Run systematic evaluation with `dspy.Evaluate`
5. Optimize prompts automatically with `BootstrapFewShot`
6. Add reasoning with `ChainOfThought`
7. Build custom modules and optimize them end-to-end
8. Enhance metrics with LLM-as-judge faithfulness checks
9. Compare all approaches in a summary table

## Prerequisites

- Python 3.11+
- AWS credentials with Bedrock access
- Model access: `global.anthropic.claude-haiku-4-5-20251001-v1:0`

## Estimated Time

45–90 minutes (15 sections)

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `05-05-DSPy-Prompt-Optimization.ipynb` and follow along.

## Files

| File | Description |
|------|-------------|
| `05-05-DSPy-Prompt-Optimization.ipynb` | Main workshop notebook (15 sections) |
| `requirements.txt` | Python dependencies |
| `city_pop.csv` | US city population and land area data (~300 cities) |
| `README.md` | This documentation |

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [DSPy Signatures Guide](https://dspy.ai/learn/programming/signatures/)
- [DSPy Optimizers Guide](https://dspy.ai/learn/optimization/optimizers/)
