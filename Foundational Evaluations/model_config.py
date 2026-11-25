"""
Centralised Bedrock model IDs for the workshop.

This is the one place to change the default and judge models. Edit it, then
restart your notebook kernel and re-run from the top.

Foundational Evaluations notebooks import this file from their subdirectory:

    import sys
    sys.path.append("..")
    from model_config import DEFAULT_MODEL_ID, JUDGE_MODEL_ID

Notebooks in other modules (for example "Workload Specific Evaluations/Tool
Calling") sit two folders below the repository root and import it like this:

    import os, sys
    sys.path.append(os.path.abspath(os.path.join("..", "..", "Foundational Evaluations")))
    from model_config import DEFAULT_MODEL_ID, JUDGE_MODEL_ID

A module that keeps its own model_config.py (for prompts or a model-comparison
set) should read DEFAULT_MODEL_ID and JUDGE_MODEL_ID from this file instead of
repeating the IDs.

Request parameters: Claude Sonnet 5 rejects ``temperature`` and ``top_p`` with a
ValidationException, so notebooks that use these models must not send them.
"""

# ---------------------------------------------------------------------------
# Primary model IDs
# ---------------------------------------------------------------------------

# Default model — generation, agents, and general workloads.
# Claude Sonnet 5 (US cross-region inference profile).
# Model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-5"

# Judge model — LLM-as-judge evaluations. It defaults to DEFAULT_MODEL_ID to keep
# workshop costs low, and it is a separate setting so you can change the judge
# without changing the model under evaluation. A different or stronger judge is not
# automatically better: self-preference bias shows up on some workloads and not on
# others. Measure it. Validate the judge against human labels
# (02-quality-metrics/03_Evaluating_your_Judge.ipynb), then change the judge prompt
# or this model only if the results show you need to. A cheaper judge, for example
# HAIKU_MODEL_ID, is worth testing the same way.
JUDGE_MODEL_ID = DEFAULT_MODEL_ID

# Small, fast model used as the "model under evaluation" in 04-agentic-metrics
# (deliberately weak so the evaluation has failures to find).
CHATBOT_MODEL_ID = "us.amazon.nova-micro-v1:0"

# Named smaller models. Notebooks that compare models, or that deliberately use
# a small, cheap model (for example as a guardrail-checked agent), import these
# names instead of repeating the IDs.
NOVA_2_LITE_MODEL_ID = "us.amazon.nova-2-lite-v1:0"
NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ---------------------------------------------------------------------------
# Model comparison sets
# ---------------------------------------------------------------------------

# On-Demand pricing (per 1M tokens) for the models compared in
# 01-operational-metrics. Source: https://aws.amazon.com/bedrock/pricing/
# Note: Claude Sonnet 5 standard rate is $3/$15 (an introductory rate of
# $2/$10 applies through August 31, 2026).
MODEL_PRICING = {
    NOVA_2_LITE_MODEL_ID: {"input": 0.33, "output": 2.75},
    NOVA_PRO_MODEL_ID: {"input": 0.80, "output": 3.20},
    HAIKU_MODEL_ID: {"input": 1.00, "output": 5.00},
    DEFAULT_MODEL_ID: {"input": 3.00, "output": 15.00},
}

# Small models used in the quick multi-city comparison in 04-agentic-metrics.
QUICK_COMPARISON_MODELS = [
    NOVA_2_LITE_MODEL_ID,
    HAIKU_MODEL_ID,
]

# Friendly names used in comparison prompts and result tables.
MODEL_NAMES = {
    CHATBOT_MODEL_ID: "Nova Micro",
    NOVA_2_LITE_MODEL_ID: "Nova 2 Lite",
    NOVA_PRO_MODEL_ID: "Nova Pro",
    HAIKU_MODEL_ID: "Claude Haiku 4.5",
    DEFAULT_MODEL_ID: "Claude Sonnet 5",
}
