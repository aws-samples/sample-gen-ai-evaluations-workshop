# 02 Model Quality Metrics

## Overview

This module demonstrates advanced evaluation techniques for LLM applications using programmatic testing, LLM-as-a-Judge methodology, and judge calibration. Using a US cities demographic dataset, you'll learn to systematically evaluate model accuracy, analytical depth, and response quality — and then validate that your evaluators themselves are reliable.

## Notebooks

| Notebook | Description |
|---|---|
| **01_LLM_as_Judge_analysis** | Programmatic testing against ground truth + LLM-as-a-Judge evaluation |
| **02_LLM_as_Jury_evaluation_analysis** | Multi-judge agreement analysis using the LLM-as-Jury pattern |
| **03_Evaluating_your_Judge** | Calibrate and validate an LLM judge against human-labeled benchmarks |

## What You'll Learn

This hands-on workshop covers three complementary evaluation approaches:

### 1. Programmatic Testing
Objective verification methods for factual accuracy:
- **Ground Truth Validation**: Compare model responses against known dataset values
- **Structured Response Parsing**: Extract and verify specific data points from model outputs
- **Automated Accuracy Scoring**: Calculate success rates across different query types

### 2. LLM-as-a-Judge Evaluation
Qualitative assessment using AI evaluators:
- **Multi-dimensional Scoring**: Assess accuracy, completeness, and analytical quality
- **Question Type Classification**: Categorize queries by complexity and requirements
- **Detailed Feedback Generation**: Receive specific improvement recommendations

### 3. Evaluating Your Judge
Validate that your automated evaluators are trustworthy:
- **Binary Pass/Fail Calibration**: One failure mode per judge, no rating scales
- **Human-Labeled Benchmarks**: Split labeled data into few-shot / dev / test sets
- **TPR/TNR Measurement**: Quantify judge accuracy as an estimator of real model error
- **Repeatability Testing**: Verify the judge produces stable verdicts across runs
- **Open-Coding Failure Analysis**: Data-driven identification of judge failure patterns


## Evaluation Methodologies

### Programmatic Verification
- **Exact Match Testing**: Verify specific population figures and geographic data
- **Calculation Validation**: Check mathematical operations like population density
- **Comparison Logic**: Validate ranking and comparison responses

### Judge-Based Assessment
- **Structured Rubrics**: Consistent evaluation criteria across all responses
- **Context-Aware Scoring**: Consider dataset characteristics and formatting nuances
- **Improvement Recommendations**: Actionable feedback for response enhancement


## Getting Started

Run the notebooks in order:

1. **01_LLM_as_Judge_analysis** — Establish programmatic and judge-based evaluation baselines
2. **02_LLM_as_Jury_evaluation_analysis** — Compare multiple judges for agreement and confidence
3. **03_Evaluating_your_Judge** — Calibrate your judge against human labels before trusting it in production

**Prerequisites:**
- AWS account with Amazon Bedrock access
- Access to Claude Sonnet 4.6 and Nova models in your region
- Python 3.10+ with boto3 library

## Key Takeaways

By completing this module, you will:
- Understand three essential LLM evaluation methodologies for production applications
- Know when to use programmatic vs. judge-based evaluation approaches
- Be able to calibrate and validate LLM judges against human-labeled ground truth
- Build reusable evaluation frameworks that can be adapted to your specific use cases
