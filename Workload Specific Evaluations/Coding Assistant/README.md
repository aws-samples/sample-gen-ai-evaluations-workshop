# Coding Assistant Evaluations

This module shows how to evaluate coding agents. The datapoints come from the PR history of [awslabs/mcp](https://github.com/awslabs/mcp). We made the PR review datapoints with [sample-code-eval-harvester](https://github.com/aws-samples/sample-code-eval-harvester). [Codex](https://github.com/openai/codex) on Amazon Bedrock is the agent under test. You run the datapoints in [Harbor](https://github.com/laude-institute/harbor) on Docker.

Harbor is not necessary. But its task format keeps the agent files and the verifier files separate. Many benchmarks and tutorials use this format.

## Approach

1. **Roll back** the repository to the parent commit of a PR change.
2. **Seal** the repository. The agent cannot find the answer.
3. **Use the PR review or the merged git patch as the oracle.** The oracle is the gold standard answer.

In this workshop, the tasks run on local Docker. In practice, run them on a compute substrate, for example Amazon EC2 or Lambda MicroVMs.

## Notebooks

| Notebook | Content | Time |
|----------|---------|------|
| [01 intro and setup.ipynb](01%20intro%20and%20setup.ipynb) | The approach, the Harbor task format, how tasks run, and the setup checks | ~20 min |
| [02 PR agent eval.ipynb](02%20PR%20agent%20eval.ipynb) | Examine a prepared reject datapoint and approve datapoint. Select the metrics. Run Codex as a PR review agent. | ~45 min |
| [03 coding agent eval.ipynb](03%20coding%20agent%20eval.ipynb) | Coming soon: evaluate a coding agent that writes the fix. | — |

## File Structure

```
Coding Assistant/
├── README.md
├── requirements.txt
├── env_setup.py
├── codex-bedrock.toml   # Codex settings for the agent under test (Bedrock provider)
├── .gitignore
├── 01 intro and setup.ipynb
├── 02 PR agent eval.ipynb
├── 03 coding agent eval.ipynb   # coming soon
├── datasets/
│   └── awslabs-mcp-pr-review/   # the PR review datapoints for notebook 02
│       ├── candidates/          # the PR data and the labels that each datapoint comes from
│       ├── rubric.md            # how to set a severity
│       ├── risk-map.toml        # path prefixes to risk levels
│       ├── metric.py            # the dataset metric: the mean of each reward key
│       └── tasks/               # the Harbor tasks (one reject, one approve)
├── harvest_env/        # a Harbor Docker environment that enforces the network allowlist on Docker Desktop
└── jobs/               # the notebooks make this directory: Harbor results
```

## Prerequisites

- **Docker.** Install it first: [Docker Desktop](https://docs.docker.com/get-docker/) (macOS, Windows) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux). Docker needs approximately 15 GB of free disk space for the task images. Each image has a clone of the repository, the Python dependencies of one server, and Codex. If a build fails with `At least one invalid signature was encountered`, the Docker disk is full. Run `docker builder prune` to remove the build cache.
- Python 3.12 or later. Harbor needs this version.
- [Codex CLI](https://github.com/openai/codex) (`npm install -g @openai/codex`), configured for Amazon Bedrock.
- AWS credentials with access to the Codex model and the judge model (Claude Haiku 4.5) on Amazon Bedrock in `us-east-1`.
- [GitHub CLI](https://cli.github.com/), logged in with `gh auth login`.
- git.

## Getting Started

1. Install Docker and start it.
2. Install the dependencies: `pip install -r requirements.txt`
3. Open [01 intro and setup.ipynb](01%20intro%20and%20setup.ipynb). Do the setup checks.
4. Do the notebooks in sequence.

## Troubleshooting

**The setup checks say `docker`, `gh`, `codex`, or `harbor` is not on PATH, but the tool is installed.** On macOS, an editor that you open from the Dock or Finder does not read your shell profile (`~/.zshrc`). The Jupyter kernel then gets a minimal `PATH`. Each notebook runs `from env_setup import fix_path; fix_path()` at the start. This copies the `PATH` from your login shell into the kernel. If a check still fails, run that cell again, or restart the kernel and run the notebook from the top. You can also open the editor from a terminal (for example, `code .`).

## Resources

- [Harbor](https://github.com/laude-institute/harbor)
- [Code Review Bench](https://github.com/withmartian/code-review-benchmark)
- [Senior SWE-Bench](https://senior-swe-bench.snorkel.ai/blog/2026-06-16-how-it-works)

