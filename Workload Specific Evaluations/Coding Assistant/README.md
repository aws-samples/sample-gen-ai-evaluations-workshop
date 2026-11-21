# Coding Assistant Evaluations

This module evaluates coding agents (Codex on Amazon Bedrock) on PR review and bug-fix tasks made from real pull requests. It has three ways to run the same tasks, one folder each. Each folder is complete by itself: pick one and work only inside it.

| Folder | Where the tasks run | You need |
|---|---|---|
| [docker/](docker/) | Local Docker, with Harbor | Docker, Python 3.12, Codex CLI, GitHub CLI, AWS credentials |
| [local/](local/) (preview) | Your Mac only: a sandboxed agent (`srt`) and a `uv` environment per trial. Apple silicon only | `uv`, AWS CLI v2, `git`, and an IAM user key with Bedrock access |
| [agentcore/](agentcore/) (pre-release) | Amazon Bedrock AgentCore Runtime, one session per trial. Images are built on AWS CodeBuild | Python 3.11 and `boto3>=1.44.7`, AWS credentials that may create IAM, ECR, S3, CodeBuild, and AgentCore resources |

- **Docker and Harbor:** notebooks 01 to 04. Start with [docker/README.md](docker/README.md).
- **Local (Mac):** a runner, a test ladder and a walkthrough notebook. Start with [local/README.md](local/README.md).
- **AgentCore:** one command from an empty account, and a walkthrough notebook. Start with [agentcore/README.md](agentcore/README.md).

For measured setup time, cost, speed, rewards, and security gates of the three, see [docs/lite-modes-comparison.md](docs/lite-modes-comparison.md).
