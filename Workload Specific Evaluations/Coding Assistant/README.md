# Coding Assistant Evaluations

Compare coding agents (Codex, Claude Code, OpenCode, ...) and models on Amazon Bedrock, on tasks made from real pull requests: PR review (find the defects in a change) and bug fix (fix an issue, graded with hidden tests and LLM judges). Each trial runs in its own Amazon Bedrock AgentCore Runtime session with [Harbor](https://github.com/harbor-framework/harbor).

The module has two steps. Each folder has the scripts for its step; each notebook explains its step and runs the scripts in order.

| Step | Folder | Notebook |
|------|--------|----------|
| 1. Build tasks from your pull requests | [`step1-build-tasks/`](step1-build-tasks/) | [`step1-build-tasks.ipynb`](step1-build-tasks.ipynb) |
| 2. Run the tasks with any agent and model, and grade them | [`step2-run-evals/`](step2-run-evals/) | [`step2-run-evals.ipynb`](step2-run-evals.ipynb) |

Step 2 has four tasks made before the workshop, so you can start with it. To reuse a step, take its folder: [`step2-run-evals/README.md`](step2-run-evals/README.md) runs the same steps from a terminal.
