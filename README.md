![Workshop Structure](https://github.com/aws-samples/sample-gen-ai-evaluations-workshop/blob/main/evals%20workshop.png?v=2 "Evals are so cool!")
# Generative AI Evaluations Workshop

This workshop teaches systematic approaches to evaluating Generative AI workloads for production use. You'll learn to build evaluation frameworks that go beyond basic metrics to ensure reliable model performance while optimizing cost and performance.

[Click here for a slide deck](https://d2ot4ns4zf41bm.cloudfront.net/slides/Gen+AI+Evals+Workshop.pptx) which covers the basics of evaluations and includes an overview of this workshop.

## How to use this repository

We strongly recommend going in order through the Foundational Evaluations modules (01–04). These cover the core of generative AI evaluations which will be critical in all workloads. After that, please feel free to select any of the workload- and framework-specific modules in any order, according to what is most relevant to you.

## What You'll Learn

### [Foundational Evaluations](Foundational%20Evaluations/) - Do all of these in order!
- 01 Operational Metrics: evaluate how your workload is running in terms of cost and performance.
- 02 Quality Metrics: evaluate and tune the quality of your results.
- 03 Understanding Failures: discover failure patterns by reading agent traces.
- 04 Agentic Metrics: evaluate your agents and use agents for evaluation.

### Optional Modules - Do any of these in any order!
- [Workload Specific Evaluations](Workload%20Specific%20Evaluations/)
  - Intelligent Document Processing
  - Guardrails
  - Basic RAG
  - MultiModal RAG
  - Speech to Speech
  - Automated Reasoning Evaluations
  - Tool Calling
  - Chatbot
  - Red Teaming
  - Multiagent Shared Context Evaluation
  
- [Framework Specific Evaluations](Framework%20Specific%20Evaluations/)
  - PromptFoo
  - Strands Evaluations
  - AgentCore Evaluations
  - AgentCore Runtime Evals
  - DSPy Prompt Optimization
  - MLflow

- [Interactive Learning Mode](Interactive%20Learning/README.md): guided challenges, exercises, and real-time feedback with an interactive tutor.

## Choose Your Learning Mode

This workshop supports two ways to learn:

- **Traditional**: Work through the Jupyter notebooks in the Foundational Evaluations modules sequentially at your own pace, then pick from Workload and Framework modules.
- **Interactive**: Use the interactive tutor in `Interactive Learning/` for guided challenges, exercises, and real-time feedback.

You can also combine both: work through a module's notebooks first, then use the interactive tutor to test your understanding before moving on.

## Prerequisites

- AWS account with Amazon Bedrock enabled
- Basic Python and ML familiarity
- No prior AI evaluation experience required

## Getting Started

1. Clone the repository
2. Configure AWS credentials
3. Work through the Foundational Evaluations modules (01–04), using the [interactive tutor](Interactive%20Learning/README.md) to test your understanding after each one
4. Review the Workload Specific and Framework Specific modules, choose any in any order.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
