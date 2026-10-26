"""Central configuration for the Model Upgrade module.

Every model ID, price, region, and fan-out knob lives here so that pointing the
module at a different source/target model or scaling the run up to your own data
is a one-line change, never a notebook rewrite.

The agent never talks to these backends directly: all inference is routed through
a single AgentCore Gateway inference target (see gateway/). These IDs tell the
Gateway which backend serves each candidate and tell the cost math which price to
apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field


AWS_REGION = "us-east-1"


@dataclass(frozen=True)
class ModelSpec:
    """One end of the migration. `provider` selects the Gateway inference target.

    `provider` doubles as the Gateway inference-target NAME, so the agent's
    `model` field (`{provider}/{model_id}`) routes to the matching target:
      - "bedrock"       -> the Bedrock inference target (default live path)
      - "openai"        -> the OpenAI first-party inference target (example)
      - "azure"         -> the Azure OpenAI inference target (example)
    """

    label: str
    model_id: str
    provider: str  # gateway inference-target name: "bedrock" | "openai" | "azure"
    price_in_per_1m: float  # USD per 1M input tokens
    price_out_per_1m: float  # USD per 1M output tokens

    @property
    def gateway_model(self) -> str:
        """The value the agent puts in the request `model` field.

        The Gateway routes by this string. We use the target-qualified form
        `{targetName}/{modelId}` (e.g. `openai/gpt-4.1`,
        `bedrock/anthropic.claude-sonnet-5-5`) so routing is unambiguous even when
        two targets could match the same bare ID. The agent calls ONE endpoint and
        only this string changes between source and target.
        """
        return f"{self.provider}/{self.model_id}"


# NOTE on prices: the per-1M-token numbers below are PLACEHOLDERS used only for the
# illustrative cost math. On-demand prices vary by region and change over time —
# verify each against its live source before you trust a cost figure (the Bedrock
# model card's Pricing section, or the provider's pricing page for 1P/Azure).

# --- The migration pair -----------------------------------------------------
# DEFAULT LIVE PATH: both ends on Amazon Bedrock, so anyone with only Bedrock
# access can run the whole module end to end. We migrate an OpenAI-on-Bedrock
# model from one version to the next: GPT-5.6 Luna -> GPT-6 Luna.
#
# SOURCE: current production model.
SOURCE_MODEL = ModelSpec(
    label="GPT-5.6 Luna (Amazon Bedrock)",
    model_id="us.openai.gpt-5.6-luna",
    provider="bedrock",
    price_in_per_1m=0.20,   # PLACEHOLDER — verify against the Bedrock model card
    price_out_per_1m=1.20,  # PLACEHOLDER
)

# TARGET: upgrade candidate (the newer version).
TARGET_MODEL = ModelSpec(
    label="GPT-6 Luna (Amazon Bedrock)",
    model_id="us.openai.gpt-6-luna",
    provider="bedrock",
    price_in_per_1m=0.25,   # PLACEHOLDER — verify against the Bedrock model card
    price_out_per_1m=1.50,  # PLACEHOLDER
)

# --- Example: migrate FROM (or TO) a non-Bedrock model ----------------------
# Many customers run their current model on OpenAI first-party or Azure OpenAI and
# want to migrate to (or from) Bedrock. The Gateway's single inference seam makes
# that a config change, not an agent-code change: add the matching inference target
# (gateway/target-openai.json or gateway/target-azure-openai.json) and point one
# end of the migration at it. These are examples — the live test uses the Bedrock
# pair above. To use one, set SOURCE_MODEL (or TARGET_MODEL) to it and provision the
# corresponding gateway target with its API key credential provider.
EXAMPLE_SOURCE_OPENAI_1P = ModelSpec(
    label="GPT-4.1 (OpenAI first-party)",
    model_id="gpt-4.1",
    provider="openai",       # -> gateway/target-openai.json (endpoint https://api.openai.com)
    price_in_per_1m=2.00,    # PLACEHOLDER — verify against https://openai.com/api/pricing/
    price_out_per_1m=8.00,   # PLACEHOLDER
)
EXAMPLE_SOURCE_AZURE_OPENAI = ModelSpec(
    label="GPT-4.1 (Azure OpenAI)",
    model_id="gpt-4.1",      # your Azure *deployment name*; see target-azure-openai.json
    provider="azure",        # -> gateway/target-azure-openai.json (your Azure endpoint)
    price_in_per_1m=2.00,    # PLACEHOLDER — verify against your Azure OpenAI pricing
    price_out_per_1m=8.00,   # PLACEHOLDER
)

# Judge model for the AgentCore custom evaluators. Kept small/cheap and held FIXED
# across source and target so the score is comparable — you are comparing the two
# agent models, not two judges. (The Chatbot module uses Nova Micro as judge.)
JUDGE_MODEL = ModelSpec(
    label="Amazon Nova Micro (judge)",
    model_id="amazon.nova-micro-v1:0",
    provider="bedrock",
    price_in_per_1m=0.035,
    price_out_per_1m=0.14,
)

# Model that PROPOSES candidate prompts during hill climbing (the "optimizer").
# Reads the judge's reasoning on the current best prompt's failures and mutates
# toward them. Uses the target model so candidates are written by the model they
# will run on.
OPTIMIZER_MODEL = ModelSpec(
    label="GPT-6 Luna (prompt optimizer)",
    model_id="us.openai.gpt-6-luna",
    provider="bedrock",
    price_in_per_1m=0.25,
    price_out_per_1m=1.50,
)


# --- Fan-out and hill-climbing knobs ----------------------------------------
# These are the SMALL RUNNABLE DEFAULTS. Turn them up to run on your own data at
# production fan-out — this is the intended way to use the module in anger.
@dataclass(frozen=True)
class RunConfig:
    candidates_per_generation: int = 3      # K prompts proposed each round
    sessions_per_candidate: int = 10        # concurrent AgentCore sessions per candidate
    scenarios_per_session: int = 5          # gold scenarios each session runs
    max_generations: int = 3                # hill-climbing rounds before the hard stop
    # Stop when a generation's best accuracy improves by less than this over the
    # previous generation's best (the "not significantly increasing" rule).
    min_accuracy_gain_to_continue: float = 0.02
    # Use the committed full run instead of invoking anything live.
    use_committed_run: bool = True

    @property
    def sessions_per_generation(self) -> int:
        return self.candidates_per_generation * self.sessions_per_candidate


RUN = RunConfig()


# Example of a turned-up configuration (what you'd run on real data):
PRODUCTION_RUN_EXAMPLE = RunConfig(
    candidates_per_generation=5,
    sessions_per_candidate=10,
    scenarios_per_session=10,
    max_generations=8,
    min_accuracy_gain_to_continue=0.01,
    use_committed_run=False,
)


# --- The single Gateway seam ------------------------------------------------
import json as _json
from pathlib import Path as _Path


def gateway_inference_base_url() -> str:
    """Read the resolved Gateway `/inference/v1` base_url written by setup_gateway.py.

    The agent builds ONE OpenAI-compatible client against this base_url and selects
    the backend purely by the request `model` field (ModelSpec.gateway_model). If the
    gateway is not provisioned yet, this raises with a clear instruction.
    """
    resolved = _Path(__file__).resolve().parent / "gateway" / ".resolved.json"
    if not resolved.exists():
        raise RuntimeError(
            "Gateway not provisioned. Run `python gateway/setup_gateway.py create` "
            "first (or set use_committed_run=True to read the committed run offline)."
        )
    return _json.loads(resolved.read_text())["inference_base_url"]

