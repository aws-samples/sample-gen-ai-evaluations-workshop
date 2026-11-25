"""Bedrock model IDs of the judges.

`harbor_env/setup.py bootstrap` gives the grader role these models. Each task.toml passes
JUDGE_MODEL_ID / VERIFIER_MODEL_ID from your shell to its verifier, with the same IDs as defaults:
if you change a model here, export the same ID when you run `harbor run`, and run `bootstrap` again.
The agent's model is Harbor's `-m`, not set here.
"""

# PR-review tasks: decides if an agent finding and an oracle finding describe the same defect.
JUDGE_MODEL_ID = "us.anthropic.claude-haiku-5-5"

# Coding tasks: the validation agent, the validation judge, the rubric judge, and the taste judge.
VERIFIER_MODEL_ID = "us.anthropic.claude-haiku-5-5"
