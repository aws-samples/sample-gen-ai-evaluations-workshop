"""
Centralised Bedrock model IDs for all Foundational Evaluations notebooks.

Edit this file to change models across every notebook in this module,
then restart your notebook kernel and re-run from the top.

Notebooks import this file from their subdirectory with:

    import sys
    sys.path.append("..")
    from model_config import DEFAULT_MODEL_ID, JUDGE_MODEL_ID
"""

# ---------------------------------------------------------------------------
# Primary model IDs
# ---------------------------------------------------------------------------

# Default model — generation, agents, and general workloads.
# Claude Sonnet 5 (US cross-region inference profile).
# Model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-5"

# Judge model — LLM-as-judge evaluations. Kept separate so you can use a
# stronger (or cheaper) judge without changing the model under evaluation.
JUDGE_MODEL_ID = DEFAULT_MODEL_ID

# Small, fast model used as the "model under evaluation" in 04-agentic-metrics
# (deliberately weak so the evaluation has failures to find).
CHATBOT_MODEL_ID = "us.amazon.nova-micro-v1:0"

# ---------------------------------------------------------------------------
# Model comparison sets
# ---------------------------------------------------------------------------

# On-Demand pricing (per 1M tokens) for the models compared in
# 01-operational-metrics. Source: https://aws.amazon.com/bedrock/pricing/
# Note: Claude Sonnet 5 standard rate is $3/$15 (an introductory rate of
# $2/$10 applies through August 31, 2026).
MODEL_PRICING = {
    "us.amazon.nova-2-lite-v1:0": {"input": 0.33, "output": 2.75},
    "us.amazon.nova-pro-v1:0": {"input": 0.80, "output": 3.20},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.00, "output": 5.00},
    DEFAULT_MODEL_ID: {"input": 3.00, "output": 15.00},
}

# Small models used in the quick multi-city comparison in 04-agentic-metrics.
QUICK_COMPARISON_MODELS = [
    "us.amazon.nova-2-lite-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

# Friendly names used in comparison prompts and result tables.
MODEL_NAMES = {
    CHATBOT_MODEL_ID: "Nova Micro",
    "us.amazon.nova-2-lite-v1:0": "Nova 2 Lite",
    "us.amazon.nova-pro-v1:0": "Nova Pro",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5",
    DEFAULT_MODEL_ID: "Claude Sonnet 5",
}
