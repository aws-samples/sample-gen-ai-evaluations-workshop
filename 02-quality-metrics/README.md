# 02-01 Model Quality Metrics

## Overview

This module demonstrates advanced evaluation techniques for LLM applications using both programmatic testing and LLM-as-a-Judge methodology. Using a US cities demographic dataset, you'll learn to systematically evaluate model accuracy, analytical depth, and response quality across different question types and complexity levels.

## What You'll Learn

This hands-on workshop covers two complementary evaluation approaches:

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

Navigate to the `02-Quality_Metrics.ipynb` notebook which demonstrates:
- Loading and exploring the US cities demographic dataset
- Implementing programmatic testing with ground truth validation
- Setting up LLM-as-a-Judge evaluation pipelines
- Analyzing results across question types and complexity levels

**Prerequisites:**
- AWS account with Amazon Bedrock access
- Access to Claude 3.7 Sonnet and Nova models in your region
- Python 3.10+ with boto3 library

## Key Takeaways

By completing this module, you will:
- Understand two essential LLM evaluation methodologies for production applications
- Understand when to use programmatic vs. judge-based evaluation approaches
- Gain hands-on experience with evaluation challenges and data complexities
- Build reusable evaluation frameworks that can be adapted to your specific use cases
