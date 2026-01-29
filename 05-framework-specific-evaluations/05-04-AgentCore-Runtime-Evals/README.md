# 05-04: AgentCore Runtime Evaluations

## Overview

This workshop module demonstrates how to deploy a Strands-based AI agent to **AgentCore Runtime** and evaluate its performance using the native **AgentCore Evaluations** API with built-in evaluators.

### What is AgentCore Runtime?

**AgentCore Runtime** is an AWS managed service for deploying and running AI agents with built-in observability. It provides:

- **Managed Infrastructure**: Deploy agents without managing servers or containers
- **Built-in Observability**: Automatic logging of agent execution traces to CloudWatch
- **Scalable Invocation**: Invoke agents via API with session management
- **Easy Deployment**: Streamlined deployment using the starter toolkit

### What is AgentCore Evaluations?

**AgentCore Evaluations** provides the `evaluate()` API for assessing agent performance using built-in evaluators:

- **Builtin.Helpfulness**: Assesses response quality and usefulness
- **Builtin.ToolSelectionAccuracy**: Evaluates appropriate tool usage
- **Objective Scoring**: Numeric scores, labels, and detailed explanations
- **Token Usage Tracking**: Monitor evaluation costs

## Prerequisites

Before running this notebook, ensure you have:

### AWS Permissions

Your AWS credentials must have permissions for:
- Amazon Bedrock (model access)
- AgentCore Runtime (deploy, invoke, destroy)
- AgentCore Evaluations (evaluate API)
- CloudWatch Logs (read logs)
- ECR (push container images)
- IAM (create execution roles, if using auto-create)

### Model Access

Enable access to the following models in Amazon Bedrock:
- `us.amazon.nova-micro-v1:0` (or your preferred model)

### Environment

- Python 3.9 or higher
- AWS CLI configured with appropriate credentials
- Docker (for container builds)


## Running the Notebook

### Step 1: Install Dependencies

Run the first code cell to install required packages:
```bash
pip install strands-agents strands-agents-tools
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit
pip install ddgs boto3 pandas
```

### Step 2: Create and Test the Agent

The notebook guides you through:
1. Defining a `web_search` tool using DuckDuckGo
2. Configuring BedrockModel with Amazon Nova Micro
3. Creating a Strands Agent with a tour guide persona
4. Testing the agent locally before deployment

### Step 3: Deploy to AgentCore Runtime

The deployment process:
1. Creates `citysearch.py` with BedrockAgentCoreApp integration
2. Configures deployment settings (ECR, IAM roles, region)
3. Launches the agent to AgentCore Runtime
4. Verifies the agent reaches READY status

**Expected time**: 2-5 minutes for deployment

### Step 4: Generate Evaluation Data

The notebook:
1. Loads test cases from `city_pop.csv`
2. Generates unique session IDs for each invocation
3. Invokes the agent for each test case
4. Waits for CloudWatch log propagation

### Step 5: Run Evaluations

Using the AgentCore Evaluations API:
1. Retrieves session spans from CloudWatch
2. Parses spans into the required format
3. Calls `evaluate()` with built-in evaluators
4. Extracts scores, labels, and explanations

### Step 6: Analyze Results

The notebook provides:
- Score summaries by evaluator
- Average scores across test cases
- Interpretation guidance
- Recommendations for improvement

### Step 7: Cleanup

**Important**: Always run cleanup to avoid charges:
1. Destroy the deployed agent
2. Verify deletion
3. Review additional resources to clean up

## Expected Outputs

### Evaluation Results

Each evaluation returns:
- **Score**: 0.0 - 1.0 (higher is better)
- **Label**: PASS, FAIL, or other categorical result
- **Explanation**: Detailed reasoning for the score
- **Token Usage**: Input and output token counts

### Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Excellent performance |
| 0.7 - 0.9 | Good with minor issues |
| 0.5 - 0.7 | Needs improvement |
| 0.0 - 0.5 | Requires significant work |

## Files in This Module

| File | Description |
|------|-------------|
| `05-04-AgentCore-Runtime-Evals.ipynb` | Main workshop notebook |
| `requirements.txt` | Python dependencies |
| `city_pop.csv` | Test data with city information |
| `README.md` | This documentation |

## Additional Resources

- [AgentCore Runtime Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-runtime.html)
- [AgentCore Evaluations Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-evaluations.html)
- [Strands Agents Framework](https://github.com/strands-agents/strands-agents)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

## Troubleshooting

### Agent Deployment Issues

- **Timeout during launch**: Check CodeBuild logs in AWS Console
- **ECR push failure**: Verify Docker is running and AWS credentials are valid
- **Role creation failure**: Ensure IAM permissions for role creation

### Evaluation Issues

- **No logs found**: Wait longer for CloudWatch propagation (30-60 seconds)
- **Log group not found**: Verify the agent was invoked successfully
- **API errors**: Check that session spans are in the correct format