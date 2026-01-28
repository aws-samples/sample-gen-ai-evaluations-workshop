# 05-03 Strands Agents Evaluations

## Overview

This module demonstrates how to use the Strands Agents Evaluation SDK (`strands-agents-evals`) to evaluate agent performance. We use the same web search agent from Module 3 (Agentic Metrics) and evaluate it using Strands Evals' built-in evaluators.

## What is Strands Evals?

Strands Evals is a comprehensive evaluation framework for AI agents that provides:

- **LLM-as-Judge Evaluation**: Uses powerful models to assess response quality against custom rubrics
- **Trajectory Evaluation**: Assesses whether agents use the right tools in the right order
- **Experiment Management**: Organize test cases, run evaluations, and track results over time
- **Custom Evaluators**: Build domain-specific evaluation logic

## Key Evaluation Components

1. **Case**: A test case containing input, expected output, expected trajectory, and metadata
2. **Experiment**: A collection of test cases with associated evaluators
3. **Evaluators**: Assess agent responses against specific criteria
   - `OutputEvaluator`: Evaluates response quality using custom rubrics
   - `TrajectoryEvaluator`: Evaluates tool usage patterns and sequences
   - Custom evaluators for domain-specific needs

## Evaluation Approaches Covered

### Output Evaluation
Uses LLM-as-judge to compare agent output against expected values based on a custom rubric. Useful for assessing:
- Factual accuracy
- Response completeness
- Format compliance

### Trajectory Evaluation
Assesses the tools used by the agent during task execution:
- Correct tool selection
- Logical tool sequence
- Efficiency (avoiding unnecessary tools)

### Custom Evaluation
Create domain-specific evaluators for specialized needs:
- XML format validation
- Business rule compliance
- Safety checks

## Getting Started

Navigate to `05-03-Strands-Evals.ipynb` which demonstrates:
1. Setting up a Strands agent with web search tools
2. Creating test cases with the `Case` class
3. Using `OutputEvaluator` for response quality assessment
4. Using `TrajectoryEvaluator` for tool usage evaluation
5. Building custom evaluators
6. Running experiments and analyzing results

## Prerequisites

```bash
pip install strands-agents strands-agents-tools strands-agents-evals
pip install ddgs
```

Before executing this notebook, ensure access to Claude 3.5 Haiku in your configured AWS region.

## Resources

- [Strands Evals Documentation](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/)
- [Strands Agents GitHub](https://github.com/strands-agents/strands-agents)
- [Evaluators Reference](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/)
