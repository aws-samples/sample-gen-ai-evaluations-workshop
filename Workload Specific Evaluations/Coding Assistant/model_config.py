"""
Centralised Bedrock model IDs for all Coding Assistant notebooks.

Edit this file to change models across every notebook in this module,
then restart your notebook kernel and re-run from the top.

Notebooks import this file with:

    from model_config import CODEX_MODEL_ID, JUDGE_MODEL_ID, VERIFIER_MODEL_ID

The judge models also reach the Harbor tasks. The notebooks put JUDGE_MODEL_ID and
VERIFIER_MODEL_ID in the environment, and each task.toml passes them to its verifier
([verifier.env]). Each task.toml also has a default, so a task still runs without the
notebooks (for example, with `harbor run` in a terminal). The verifier records the
model that it used.
"""

# ---------------------------------------------------------------------------
# Agent under test
# ---------------------------------------------------------------------------

# Codex on Amazon Bedrock, through the Mantle endpoint (bedrock-mantle.<region>.api.aws).
# Notebooks 01, 02, and 03.
CODEX_MODEL_ID = "openai.gpt-6-sol"

# ---------------------------------------------------------------------------
# Judges (they run in the Harbor verifier and call bedrock-runtime.<region>.amazonaws.com)
# ---------------------------------------------------------------------------

# PR review tasks (notebook 02): decides if an agent finding and an oracle finding
# describe the same defect.
JUDGE_MODEL_ID = "us.anthropic.claude-haiku-5-5"

# Coding tasks (notebook 03): the validation agent, the validation judge, the task
# rubric judge, and the taste judge.
VERIFIER_MODEL_ID = "us.anthropic.claude-haiku-5-5"
