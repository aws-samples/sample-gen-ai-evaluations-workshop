"""Centralised Bedrock model configuration for Retail and CPG notebooks.

Edit this file to change the model, then restart the notebook kernel and
rerun from the top. EVAL_MODEL_ID, when set, overrides the default below.

Notebooks in this folder import the configuration with:

    from model_config import DEFAULT_MODEL_ID as MODEL_ID
"""

import os

# Claude Sonnet 4 (US cross-region inference profile, usable from us-east-1).
# Choose a Claude model compatible with the notebook's Anthropic Messages body.
DEFAULT_MODEL_ID = os.environ.get(
    "EVAL_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
