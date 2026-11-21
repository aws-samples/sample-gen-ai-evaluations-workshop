# Coding Assistant Evaluations

This module shows how to evaluate coding agents. The PR review datapoints come from the PR history of [awslabs/mcp](https://github.com/awslabs/mcp). We made them with [sample-code-eval-harvester](https://github.com/aws-samples/sample-code-eval-harvester). The coding datapoints come from bug fixes in [strands-agents/evals](https://github.com/strands-agents/evals), in the format of [Senior SWE-Bench](https://hub.harborframework.com/datasets/snorkel-ai/senior-swe-bench-v2026.06/latest?tab=tasks). [Codex](https://github.com/openai/codex) on Amazon Bedrock is the agent under test. You run the datapoints in [Harbor](https://github.com/laude-institute/harbor) on Docker.

Harbor is not necessary. But its task format keeps the agent files and the verifier files separate. Many benchmarks and tutorials use this format.

## Approach

1. **Roll back** the repository to the parent commit of a PR change.
2. **Seal** the repository. The agent cannot find the answer.
3. **Use the PR review or the merged git patch as the oracle.** The oracle is the gold standard answer. For a coding task, hidden tests check the fix of the agent.

In this workshop, the tasks run on local Docker. In practice, run them on a compute substrate, for example Amazon EC2 or Lambda MicroVMs.

## Choose a mode

| Mode | Where the tasks run | You need | Start |
|---|---|---|---|
| **Baseline: Docker and Harbor** | Local Docker, with Harbor | Docker, Python 3.12, Codex CLI, GitHub CLI, AWS credentials | The notebooks below |
| **AgentCore** (pre-release, for testers) | Amazon Bedrock AgentCore Runtime, one session per trial. Images are built on AWS CodeBuild | Python 3.10 and `boto3>=1.44.7`, AWS credentials that may create IAM, ECR, S3, CodeBuild, and AgentCore resources. No Docker | [agentcore/README.md](agentcore/README.md) |
| **Local** (coming on this branch) | Your machine only | Not decided yet | Not available yet |

## Notebooks

| Notebook | Content | Time |
|----------|---------|------|
| [01 intro and setup.ipynb](01%20intro%20and%20setup.ipynb) | The approach, the Harbor task format, how tasks run, and the setup checks | ~20 min |
| [02 PR agent eval.ipynb](02%20PR%20agent%20eval.ipynb) | Examine a prepared reject datapoint and approve datapoint. Select the metrics. Run Codex as a PR review agent. | ~45 min |
| [03 coding agent eval.ipynb](03%20coding%20agent%20eval.ipynb) | Examine two bug-fix datapoints. Grade the fix as Senior SWE-Bench does: hidden tests, a validation agent, a task rubric, a taste judge, and patch bloat. Run Codex as a coding agent that writes the fix, and optionally Claude Code on the same tasks. | ~60 min |
| [04 Replay recorded runs (no Docker).ipynb](04%20Replay%20recorded%20runs%20(no%20Docker).ipynb) | Grade and examine recorded Codex and Claude Code runs of notebooks 02 and 03. No Docker, no AWS, no tokens. Start here if you do not have Docker. | ~15 min |

## File Structure

```
Coding Assistant/
├── README.md
├── requirements.txt
├── env_setup.py        # PATH repair and .env loading for the Jupyter kernel
├── coding_eval.py      # the helpers that notebooks 02 and 03 share (see below)
├── model_config.py     # all model IDs of the module: the agent under test and the judges
├── codex-bedrock.toml  # Codex settings for the agent under test (Bedrock provider)
├── .env.example        # copy to .env: AWS_REGION and AWS_PROFILE
├── .gitignore
├── 01 intro and setup.ipynb
├── 02 PR agent eval.ipynb
├── 03 coding agent eval.ipynb
├── 04 Replay recorded runs (no Docker).ipynb
├── datasets/
│   ├── _shared/                 # one source for the verifier files that are the same in each task
│   │   ├── sync.py              # copies them into each task; --check reports a copy that differs
│   │   ├── pr-review/tests/     # score.py, judge.toml, test.sh, rubric.md, Dockerfile (notebook 02)
│   │   └── coding/tests/        # lib/*.py, judge/models.toml (notebook 03)
│   ├── awslabs-mcp-pr-review/   # the PR review datapoints for notebook 02
│   │   ├── candidates/          # the PR data and the labels that each datapoint comes from
│   │   ├── rubric.md            # how to set a severity
│   │   ├── risk-map.toml        # path prefixes to risk levels
│   │   ├── metric.py            # the dataset metric: the mean of each reward key
│   │   └── tasks/               # the Harbor tasks (one reject, one approve)
│   ├── recorded-runs/           # recorded Harbor jobs of notebooks 02 and 03, for notebook 04
│   └── strands-evals-coding/    # the coding datapoints for notebook 03
│       └── tasks/               # the Harbor tasks (two bug fixes, each with its own reward stack)
├── harvest_env/        # a Harbor Docker environment that enforces the network allowlist on Docker Desktop
├── tests_local/        # checks of the module code that need no Docker or AWS: python -m pytest tests_local
└── jobs/               # the notebooks make this directory: Harbor results
```

### Shared code

- **`coding_eval.py`** has the notebook helpers: the leakage scan (it plants its canary in a temporary copy of the task), the image build (it skips images that exist; `force=True` builds them again), the git seal check, an optional run-time egress check, the AWS credential refresh, the Region check, `harbor_run`, `trial_rewards`, and `harbor view`. The notebooks keep the explanations.
- **`datasets/_shared/`** has the verifier files that are the same in every task of a dataset. Harbor builds or uploads a task only from its own directory, and a symbolic link that points outside it does not work. Thus each task keeps a full copy. Edit the file in `datasets/_shared/`, then run `python datasets/_shared/sync.py`. Notebooks 02 and 03 warn if a copy differs from its source.

## Prerequisites

- **Docker.** Install it first: [Docker Desktop](https://docs.docker.com/get-docker/) (macOS, Windows) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux). Docker needs approximately 15 GB of free disk space for the task images. Each image has a clone of the repository, its Python dependencies, and Codex. The images of notebook 03 are approximately 4 GB each. If a build fails with `At least one invalid signature was encountered`, the Docker disk is full. Run `docker builder prune` to remove the build cache.
- Python 3.12 or later. Harbor needs this version.
- [Codex CLI](https://github.com/openai/codex) (`npm install -g @openai/codex`), configured for Amazon Bedrock.
- AWS credentials with access to the Codex model and the judge models on Amazon Bedrock in `us-east-1`. The model IDs are in [model_config.py](model_config.py): `CODEX_MODEL_ID` (the agent under test), `CLAUDE_CODE_MODEL_ID` (the optional second agent), `JUDGE_MODEL_ID` (notebook 02), and `VERIFIER_MODEL_ID` (notebook 03).
- [GitHub CLI](https://cli.github.com/), logged in with `gh auth login`.
- git.

## Time, Disk, and Cost

| | Approximately |
|---|---|
| **Disk** | 15 GB for the Docker images: four task images of 3.5 GB each, and the proxy |
| **Time** | 1.5 to 2.5 minutes to build each image one time. Then 30 to 45 minutes for notebooks 02 and 03, with one trial for each task |
| **Amazon Bedrock cost** | $1 to $2 for notebooks 02 and 03, with one trial for each task: about $0.25 for notebook 02 and $0.75 for notebook 03. Most of it is for the agent model. Each more trial for each task adds about the same again |

We measured these values in us-east-1 in November 2026, with the model IDs in `model_config.py`. Most input tokens are cached, so a run uses millions of tokens but costs little. Prices change, and they are different in each Region: see [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).

## AWS Credentials for the Agent

Harbor gives your AWS credentials to Codex in the task container, and to the verifier. The network allowlist lets the container reach only Amazon Bedrock. But the agent under test can use the credentials for each Bedrock call that they allow.

- Use a role that can only invoke the models in `model_config.py`: the agent model and the judge model. Do not use administrator credentials.
- Use temporary credentials, for example from AWS IAM Identity Center, not long-term access keys.

## Getting Started

1. Install Docker and start it.
2. Install the dependencies: `pip install -r requirements.txt`
3. Open [01 intro and setup.ipynb](01%20intro%20and%20setup.ipynb). Do the setup checks.
4. Do the notebooks in sequence.

## Troubleshooting

**The notebook stops with `AWS_REGION is ..., but the tasks allow only the Bedrock endpoints in us-east-1`.** Each `task.toml` allows only the Bedrock endpoints of one Region. A different Region, for example from your shell, makes every trial fail. Set `AWS_REGION=us-east-1` in `.env`. Notebooks 02 and 03 make `.env` from `.env.example` if it does not exist. To use a different Region, change `allowed_hosts` in each `task.toml` and the region in `codex-bedrock.toml`.

**The setup checks say `docker`, `gh`, `codex`, or `harbor` is not on PATH, but the tool is installed.** On macOS, an editor that you open from the Dock or Finder does not read your shell profile (`~/.zshrc`). The Jupyter kernel then gets a minimal `PATH`. Each notebook runs `from env_setup import fix_path; fix_path()` at the start. This copies the `PATH` from your login shell into the kernel. If a check still fails, run that cell again, or restart the kernel and run the notebook from the top. You can also open the editor from a terminal (for example, `code .`).

## Resources

- [Harbor](https://github.com/laude-institute/harbor)
- [Code Review Bench](https://github.com/withmartian/code-review-benchmark)
- [Senior SWE-Bench](https://senior-swe-bench.snorkel.ai/blog/2026-06-16-how-it-works)

