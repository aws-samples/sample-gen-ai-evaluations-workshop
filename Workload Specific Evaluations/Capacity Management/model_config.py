"""
Model IDs, pricing, and simulated quota tables for the Capacity Management module.

Edit this file to change models or limits across the whole notebook, then
restart the kernel and re-run from the top.
"""

REGION = "us-east-1"

# ---------------------------------------------------------------------------
# The candidate portfolio
# ---------------------------------------------------------------------------
# Five text models spanning three providers. A capacity portfolio built from a
# single provider tends to share quota pools and correlated outages, so
# spreading across providers is part of the point.
#
# Note that gpt-oss-120b is served ON_DEMAND and therefore has no `us.`
# cross-region inference profile prefix, while the other four do. That
# inconsistency is normal and worth handling explicitly in your own config.

NOVA_2_LITE = "us.amazon.nova-2-lite-v1:0"
CLAUDE_HAIKU_45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_SONNET_5 = "us.anthropic.claude-sonnet-5"
GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
GPT_56_LUNA = "us.openai.gpt-5.6-luna"

CANDIDATE_MODELS = [
    NOVA_2_LITE,
    CLAUDE_HAIKU_45,
    CLAUDE_SONNET_5,
    GPT_OSS_120B,
    GPT_56_LUNA,
]

MODEL_NAMES = {
    NOVA_2_LITE: "Nova 2 Lite",
    CLAUDE_HAIKU_45: "Claude Haiku 4.5",
    CLAUDE_SONNET_5: "Claude Sonnet 5",
    GPT_OSS_120B: "gpt-oss-120b",
    GPT_56_LUNA: "GPT-5.6 Luna",
}

MODEL_PROVIDERS = {
    NOVA_2_LITE: "Amazon",
    CLAUDE_HAIKU_45: "Anthropic",
    CLAUDE_SONNET_5: "Anthropic",
    GPT_OSS_120B: "OpenAI (open weight)",
    GPT_56_LUNA: "OpenAI",
}

# ---------------------------------------------------------------------------
# On-Demand pricing, USD per 1M tokens, us-east-1
# ---------------------------------------------------------------------------
# Verify current rates at https://aws.amazon.com/bedrock/pricing/ before you
# rely on these for a real decision. Prices change.
MODEL_PRICING = {
    NOVA_2_LITE: {"input": 0.33, "output": 2.75},
    CLAUDE_HAIKU_45: {"input": 1.00, "output": 5.00},
    CLAUDE_SONNET_5: {"input": 3.00, "output": 15.00},
    GPT_OSS_120B: {"input": 0.15, "output": 0.60},
    GPT_56_LUNA: {"input": 0.20, "output": 1.20},
}

# ---------------------------------------------------------------------------
# Simulated quotas (requests per minute)
# ---------------------------------------------------------------------------
# These are ARTIFICIAL and deliberately tiny. Real Bedrock quotas are orders of
# magnitude higher, so reaching them in a workshop would take thousands of
# requests and real money. Shrinking the numbers makes throttling observable in
# seconds without changing any of the mechanics.

# Uniform limits keep the arithmetic obvious in the routing demos: spend the
# burst of 6, then earn one more request every 10 seconds (6 per minute).
DEMO_QUOTAS = {m: 6 for m in CANDIDATE_MODELS}

# Effectively unlimited. Used for the quality-evaluation sweeps, which measure
# accuracy rather than capacity — throttling those would just make the notebook
# slow without teaching anything new.
EVAL_QUOTAS = {m: 10_000 for m in CANDIDATE_MODELS}

# Example per-model limits for the projection section. Also ARTIFICIAL, and small
# for the same reason as DEMO_QUOTAS: we want the limits to be easy to reach so
# the routing behaviour is visible inside a workshop.
#
# The *shape* is deliberate. On Bedrock, smaller and cheaper models generally
# carry the higher default request quotas while frontier models carry the tighter
# ones, so the two cheapest models here get four times the ceiling of the rest.
# Check the current defaults for your own account and Region with:
#
#     aws service-quotas list-aws-default-service-quotas --service-code bedrock \
#       --region us-east-1 --query \
#       'Quotas[?contains(QuotaName, `requests per minute`)].[QuotaName,Value]'
#
# Two things to know when you do. Real ceilings are orders of magnitude higher
# than these, and RPM is not enforced for every model — several are governed by
# token quotas alone. A token bucket behaves the same whichever dimension binds
# first, which is why modelling RPM only still teaches the right lesson.
PRODUCTION_QUOTAS_EXAMPLE = {
    NOVA_2_LITE: 100,
    CLAUDE_HAIKU_45: 100,
    CLAUDE_SONNET_5: 100,
    GPT_OSS_120B: 400,
    GPT_56_LUNA: 400,
}

# ---------------------------------------------------------------------------
# Advanced Prompt Optimization
# ---------------------------------------------------------------------------
# AdvPO accepts up to 5 target models per job, which is exactly our portfolio,
# and allowed judge models are a fixed short list.
ADVPO_JUDGE_MODEL = "anthropic.claude-sonnet-4-6"
ADVPO_INPUT_VERSION = "bedrock-2026-05-14"

# Advanced Prompt Optimization does not accept the same model identifiers as the
# runtime Converse API, and the difference is not intuitive:
#
#   Nova and Anthropic  -> AdvPO wants the `us.` inference profile ID
#                          (the bare `amazon.*` / `anthropic.*` ID is rejected)
#   OpenAI frontier     -> AdvPO wants the BARE model ID `openai.gpt-5.6-luna`
#                          (the `us.` / `global.` profile ID is rejected, even
#                          though Converse requires the profile and rejects the
#                          bare ID)
#
# So for GPT-5.6 Luna the two APIs accept mutually exclusive identifiers. Get it
# wrong and the job fails validation with the generic message
# "No inference API is accessible for model ...".
#
# Verified empirically against CreateAdvancedPromptOptimizationJob in us-east-1.
ADVPO_MODEL_IDS = {
    NOVA_2_LITE: NOVA_2_LITE,
    CLAUDE_HAIKU_45: CLAUDE_HAIKU_45,
    CLAUDE_SONNET_5: CLAUDE_SONNET_5,
    GPT_OSS_120B: GPT_OSS_120B,
    GPT_56_LUNA: "openai.gpt-5.6-luna",
}

# Reverse map, for reading AdvPO results back onto runtime model IDs.
RUNTIME_MODEL_IDS = {v: k for k, v in ADVPO_MODEL_IDS.items()}


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one invocation, from measured token counts."""
    p = MODEL_PRICING[model_id]
    return (input_tokens / 1_000_000) * p["input"] + \
           (output_tokens / 1_000_000) * p["output"]


def friendly(model_id: str) -> str:
    return MODEL_NAMES.get(model_id, model_id)
