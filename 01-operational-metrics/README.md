# 01 Operational Metrics

## Overview

This module teaches you to measure and analyze key operational metrics for Large Language Models (LLMs) within Amazon Bedrock. Understanding these metrics is crucial for optimizing application performance, managing costs effectively, and understanding the trade-offs between latency and accuracy in production environments.

## What You'll Learn

This hands-on workshop covers essential operational metrics that every production LLM application needs to monitor:

### 1. Cost Metrics
Track and optimize your LLM spending:
- **Token Usage Analysis**: Monitor input and output token consumption across different models
- **Cost Calculation**: Calculate real-time costs based on current Bedrock pricing
- **Model Cost Comparison**: Compare cost efficiency across Nova Lite, Nova Pro, and Claude models
- **Custom CloudWatch Metrics**: Publish cost metrics for monitoring and alerting

### 2. Latency Metrics
Measure response times and optimize user experience:
- **End-to-End Latency**: Total time from request to complete response
- **Server-Side Timing**: Leverage Bedrock's built-in latency measurements
- **Throughput Analysis**: Calculate tokens per second for different models
- **Performance Comparison**: Benchmark latency across model families

### 3. Time to First Token (TTFT) vs Time to Last Token (TTLT)
Understand streaming performance characteristics:
- **TTFT Measurement**: How quickly models begin generating responses
- **TTLT Analysis**: Total time to complete response generation
- **Inter-token Latency**: Measure consistency of token generation speed
- **Streaming Optimization**: Optimize for perceived responsiveness vs total throughput

### 4. Real-World Use Case: Email Summarization
Apply metrics to a practical business scenario:
- **Multi-Model Comparison**: Evaluate Nova Lite, Nova Pro, and Claude 3.7 Sonnet
- **Performance vs Quality Trade-offs**: Balance speed, cost, and output quality
- **Production Readiness**: Assess which models meet your specific requirements

## Key Metrics Covered

- **Cost per Request**: Track spending across different models and use cases
- **Latency Distribution**: Understand response time patterns and outliers
- **Tokens per Second**: Measure generation speed for throughput planning
- **Time to First Token**: Optimize for perceived responsiveness
- **Error Rates**: Monitor throttling and failure patterns

## Getting Started

Navigate to the `01-Operational-Metrics.ipynb` notebook which demonstrates:
- Setting up cost tracking and calculation functions
- Measuring latency with both synchronous and streaming APIs
- Implementing TTFT/TTLT measurement with precise timing
- Building CloudWatch dashboards for operational monitoring
- Comparing model performance on email summarization tasks

**Prerequisites:**
- AWS account with Amazon Bedrock access
- Access to Nova Lite, Nova Pro, and Claude 3.7 Sonnet models in your region
- Python 3.10+ with boto3 library
- CloudWatch permissions for custom metrics publishing

## CloudWatch Integration

This module includes examples of:
- Publishing custom operational metrics to CloudWatch
- Creating dashboards for real-time monitoring
- Setting up alerts for cost and performance thresholds
- Visualizing metrics across different models and time periods

## Key Takeaways

By completing this module, you will:
- Understand the fundamental operational metrics needed for production LLM applications
- Learn to measure and optimize cost, latency, and throughput
- Gain hands-on experience with Bedrock's built-in metrics and custom CloudWatch integration
- Build a framework for ongoing operational monitoring and optimization
- Make data-driven decisions about model selection based on performance requirements

This foundation in operational metrics is essential before moving on to quality and agentic evaluations in subsequent modules.