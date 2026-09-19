# Amazon Bedrock AgentCore: Run, Observe, Evaluate, Improve

This module presents AgentCore evaluation as an end-to-end workflow. You will build one deterministic agent, deploy it once, inspect its traces, evaluate specific sessions, add ground truth, create focused custom evaluators, monitor sampled traffic, and then use simulation and optimization as an optional advanced loop.

The examples use a fixed city-facts dataset rather than live web search. That makes expected responses and tool trajectories stable enough for regression testing.

## Learning Path

Complete the notebooks in order. Each notebook reuses the `CityAnalyst` runtime created in notebook 01.

| Notebook | Focus | Typical time | AWS resources |
|---|---|---:|---|
| [01 - AgentCore Foundations](01-agentcore-foundations.ipynb) | Runtime, observability, traces, and one shared deployment | 35-50 min | Runtime and supporting CDK resources |
| [02 - Built-in On-Demand Evaluations](02-built-in-on-demand-evaluations.ipynb) | Built-in evaluators through the CLI and Python SDK | 30-45 min | Evaluation model calls |
| [03 - Ground Truth and Curated Datasets](03-ground-truth-and-datasets.ipynb) | Reference inputs, curated scenarios, and dataset runners | 40-60 min | Runtime invocations and evaluation calls |
| [04 - Custom Evaluators](04-custom-evaluators.ipynb) | Focused LLM judges, deterministic code evaluators, and judge calibration | 45-60 min | Optional custom evaluator resources |
| [05 - Batch and Online Evaluation](05-batch-and-online-evaluation.ipynb) | Batch regression, CI gates, and sampled production monitoring | 35-50 min | Optional batch and online evaluation resources |
| [06 - Simulation and Optimization](06-simulation-and-optimization.ipynb) | Simulated users, insights, recommendations, config bundles, and A/B tests | 35-60 min | Optional preview and optimization resources |

Notebooks 01-04 form the core path. Notebooks 05-06 are production and advanced extensions.

## Mental Model

| Layer | Responsibility |
|---|---|
| AgentCore Runtime | Hosts and scales an agent loop that you own. This module uses a Strands agent and the HTTP protocol. |
| AgentCore Observability | Emits OpenTelemetry-compatible sessions, traces, model spans, and tool spans to CloudWatch. |
| AgentCore Evaluations | Scores instrumented session, trace, or tool-call behavior with built-in or custom evaluators. |
| AgentCore CLI | Manages the project, local development, deployment, traces, evaluators, datasets, and evaluation jobs. |
| Python SDK | Automates targeted evaluations, curated scenario runners, simulation, and custom evaluator logic. |

AgentCore Runtime and AgentCore Harness are different choices. Runtime hosts agent code and orchestration that you provide. Harness is a managed, configuration-driven agent loop. The evaluation concepts in this module apply to both, but the hands-on agent uses Runtime so learners can see the application and tool boundaries directly.

## Repository Layout

```text
AgentCore/
├── README.md
├── requirements.txt
├── 01-agentcore-foundations.ipynb
├── 02-built-in-on-demand-evaluations.ipynb
├── 03-ground-truth-and-datasets.ipynb
├── 04-custom-evaluators.ipynb
├── 05-batch-and-online-evaluation.ipynb
├── 06-simulation-and-optimization.ipynb
├── agentcore/                         # AgentCore CLI project configuration
├── app/CityAnalyst/                   # Shared deterministic Runtime agent
├── data/                              # Evaluation and calibration fixtures
├── evaluators/                        # Custom code-based evaluator
├── src/                               # Notebook helpers and cleanup
└── generated/                         # Local results and traces; gitignored
```

## Prerequisites

- Node.js 20 or later
- Python 3.10 or later
- `uv`
- AWS credentials for a workshop or development account
- Access to the configured Amazon Bedrock model
- Permissions to bootstrap and deploy AWS CDK resources and to use AgentCore, CloudWatch, X-Ray, and Bedrock

Install the AgentCore CLI and verify the installation:

```bash
npm install -g @aws/agentcore@0.27.0
hash -r
agentcore --version
aws sts get-caller-identity
aws configure get region
```

If the Region command is empty, set `AWS_REGION` or configure a default Region before continuing.

Create a notebook environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Before the first deployment, bootstrap the AWS account and Region. CDK bootstrap is a one-time setup for each account and Region:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region)}}

cd agentcore/cdk
npm ci --no-audit --no-fund
./node_modules/.bin/cdk bootstrap "aws://${ACCOUNT_ID}/${REGION}"
cd ../..
```

Run all commands from this directory. Start with `01-agentcore-foundations.ipynb`.

## Reproducibility

The city data in `app/CityAnalyst/data/city_facts.json` is a workshop fixture, not a claim of current demographic data. The agent has three deterministic tools:

- `lookup_city`
- `compare_cities`
- `calculate_density`

This creates meaningful test cases for tool selection, tool parameters, expected responses, and multi-turn behavior without making results depend on a search engine or a changing web page.

## Feature Status

- On-demand and online AgentCore Evaluations are the mainline path.
- Dataset evaluation, including predefined and simulated scenarios, is public preview as of August 14, 2026.
- Batch evaluation is public preview and does not currently emit CloudTrail events. Do not use it for workflows that require a complete API audit trail.
- Insights is public preview. Recommendations, configuration bundles, and A/B testing are fast-moving optional extensions. Notebook 06 shows how to verify the installed CLI surface before use.
- Built-in evaluator names can change. The notebooks show the ten evaluator IDs recognized by AgentCore CLI `0.27.0` and ask learners to verify their installed version.

## Cost, Security, and Cleanup

Evaluation calls invoke judge models. Dataset and simulation runs also invoke the agent. Online evaluation adds judge calls for the sampled portion of live traffic.

Use `100%` online sampling only for the small, controlled traffic generated by this workshop. Start production monitoring around `1-5%`, then adjust using traffic volume, risk, and cost data.

Traces and evaluation results can contain prompts, responses, and tool parameters. In production:

- scrub or avoid sensitive data before telemetry export
- encrypt CloudWatch log groups and evaluation resources with customer-managed KMS keys where required
- set log retention
- scope IAM permissions to the runtime, evaluator, and log resources
- review cross-region inference against data-residency requirements

Run the cleanup helper after the module:

```bash
python src/cleanup.py --yes
```

The helper temporarily deploys an empty project specification so CloudFormation removes resources managed by this module, then restores the checked-in workshop configuration. Verify any externally managed Lambda functions, KMS keys, retained log groups, or manually created resources separately.

## Official References

- [AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)
- [Evaluation types](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations-types.html)
- [Dataset evaluation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/dataset-evaluations.html)
- [AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)
- [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html)
- [AgentCore CLI](https://github.com/aws/agentcore-cli)
- [AgentCore Python SDK](https://github.com/aws/bedrock-agentcore-sdk-python)
