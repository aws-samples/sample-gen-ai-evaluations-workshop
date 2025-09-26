# Agentic AI Evaluation Framework

A comprehensive evaluation framework for AI agents using built-in observability features and custom evaluation metrics.

## Overview

This framework demonstrates multiple evaluation techniques for AI agents across four key dimensions:

- **Agent Performance**: Measuring accuracy using ground truth datasets
- **Tool Execution**: Analyzing tool selection and execution success rates  
- **Resource Efficiency**: Token usage, latency, and cycle duration analysis
- **Agent Reliability**: Consistency across multiple test scenarios

## Quick Start

### Prerequisites

```bash
pip install strands-agents strands-agents-tools ddgs
```

### Basic Usage

```python
from strands import Agent, tool
from strands.models import BedrockModel

# Create agent with tools
agent = Agent(tools=[web_search, get_page], model=BedrockModel("us.amazon.nova-micro-v1:0"))

# Run evaluation
response = agent("How many people live in Phoenix, AZ?")
result = evaluate_city_guess("Phoenix", "AZ", response, dataset)
```

## Evaluation Methods

### 1. Ground Truth Validation

Uses structured city population dataset (`city_pop.csv`) to validate agent responses:

- Population accuracy measurement
- Area estimation validation  
- Percent error calculations
- Multi-city consistency testing

### 2. Performance Metrics

Leverages built-in observability to track agent performance:

```python
# Access comprehensive metrics
total_tokens = response.metrics.accumulated_usage['totalTokens']
execution_time = sum(response.metrics.cycle_durations)
tool_calls = sum(metric.call_count for metric in response.metrics.tool_metrics.values())
```

### 3. Tool Selection Accuracy

Evaluates correct tool usage across different task types:

- Mathematical calculations → `calculator`
- File operations → `file_read`/`file_write`  
- Code execution → `code_interpreter`
- Knowledge queries → No tools required

### 4. LLM-as-a-Judge

Uses stronger models to evaluate response quality:

- Accuracy assessment
- Relevance scoring
- Completeness evaluation
- Tool usage appropriateness

## Key Features

### Multi-Model Comparison

```python
# Evaluate multiple models
models = [
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0", 
    "us.anthropic.claude-3-haiku-20240307-v1:0"
]

for model in models:
    result = eval_model(model)
```

### Batch Evaluation

```python
# Test multiple cities for consistency
results = evaluate_multiple_cities("us.amazon.nova-micro-v1:0", num_cities=3)
print(f"Average population error: {results['avg_population_error']}%")
```

### Structured Output Validation

Uses XML tags for programmatic evaluation:

```xml
<pop>1680992</pop>
<area>302.6</area>
```

## Files Structure

```
├── 03-Agentic-Metrics.ipynb    # Main evaluation notebook
├── city_pop.csv                # Ground truth dataset
├── dataset.json                # Tool selection test cases
├── evaluation_results.json     # LLM judge results
└── README.md                   # This file
```

## Evaluation Results

### Model Performance Patterns

- **Larger models** (Claude, Nova Pro): Better accuracy, higher token costs
- **Smaller models** (Nova Micro): Faster execution, less reliable with complex instructions
- **Tool selection accuracy**: Varies significantly between model families

### Key Metrics Tracked

| Metric | Description | Usage |
|--------|-------------|-------|
| Population Error % | Accuracy vs ground truth | Model comparison |
| Area Error % | Geographic data accuracy | Consistency testing |
| Total Tokens | Resource consumption | Cost optimization |
| Execution Time | Response latency | Performance tuning |
| Tool Calls | Efficiency measurement | Workflow optimization |

## Advanced Features

### Custom Evaluation Functions

```python
def evaluate_city_guess(city, state, response, dataset):
    """Comprehensive city data evaluation with error calculations"""
    # Extract structured outputs
    # Calculate percent errors
    # Gather performance metrics
    return evaluation_results
```

### Observability Integration

```python
def display_metrics(result):
    """Detailed performance analysis using built-in metrics"""
    summary = result.metrics.get_summary()
    # Tool performance analysis
    # Token usage breakdown  
    # Cycle duration tracking
```


