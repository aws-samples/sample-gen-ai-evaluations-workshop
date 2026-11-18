"""
Centralised model IDs and prompts for the metrics notebooks.
Edit this file to change prompts or to override the models for this module.

The agent and judge models come from the workshop's shared config,
Foundational Evaluations/model_config.py. Change them there to update every
module at once.
"""

import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------

# This folder's own file is also called model_config.py, so load the shared one
# by path instead of with a plain import (which would find this file again).
_SHARED_PATH = Path(__file__).resolve().parents[2] / "Foundational Evaluations" / "model_config.py"
_spec = importlib.util.spec_from_file_location("workshop_shared_model_config", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

# Agent model — used by all spoke/peer agents and the hub coordinator.
# To compare models in this module only, assign a different model ID here.
AGENT_MODEL_ID = _shared.DEFAULT_MODEL_ID

# Judge model — used by LLM-as-judge for semantic evaluation.
# For higher accuracy (at the cost of longer evaluation times), assign a
# stronger model ID here.
JUDGE_MODEL_ID = _shared.JUDGE_MODEL_ID

# Embedding model — used for C2 alignment in peer-to-peer notebook
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


# ---------------------------------------------------------------------------
# Hub-and-Spoke Prompts (Notebook 1)
# ---------------------------------------------------------------------------

FLIGHT_PROMPT = (
    "You are a flight booking assistant. Help find flights, make reservations, "
    "and answer questions about airlines, routes, and pricing. Be specific with "
    "prices and schedules."
)

HOTEL_PROMPT = (
    "You are a hotel booking assistant. Help find hotels, make reservations, "
    "and answer questions about accommodations, amenities, and pricing. "
    "Be specific with prices."
)

ITINERARY_PROMPT = (
    "You are an itinerary planner. Read the flight and hotel information from "
    "conversation history and create a cohesive day-by-day travel itinerary. "
    "Reference specific flight times, hotel names, and prices from the prior "
    "agents' outputs."
)

HUB_PROMPT = """
You are a travel planning coordinator. You delegate to specialized agents:
- flight_booking_assistant: for flight queries
- hotel_booking_assistant: for hotel queries
- itinerary_assistant: for building the final itinerary (call LAST, after flight + hotel)

For complete trip requests, call flight first, then hotel, then itinerary.
Keep messages short. Ask max 2 questions per turn.
"""


# ---------------------------------------------------------------------------
# Peer-to-Peer Prompts (Notebook 2) — Restaurant Industry
# ---------------------------------------------------------------------------

MARKET_TRENDS_PROMPT = (
    "You are a Market Trends Analyst specialising in the restaurant and food service industry. "
    "Analyse market trends, competitive landscape, market size, growth rates, and emerging "
    "opportunities for the given topic. Be specific with numbers, brand names, and market "
    "segments. Keep your analysis concise (2-3 paragraphs)."
)

CUSTOMER_INSIGHTS_PROMPT = (
    "You are a Customer Insights Analyst specialising in the restaurant and food service industry. "
    "Analyse customer segments, dining behaviour, pain points, and demand patterns. Reference "
    "specific demographics and behavioural data. Build on any prior market analysis available "
    "in context. Keep your analysis concise (2-3 paragraphs)."
)

STRATEGY_SYNTH_PROMPT = (
    "You are a Strategy Synthesizer specialising in the restaurant and food service industry. "
    "Read the market trends and customer insights from your colleagues, then produce a unified "
    "strategic recommendation. Reference specific findings from both prior analyses. Provide "
    "actionable recommendations with priorities. Keep it concise (2-3 paragraphs)."
)

PEER_CONFIGS = [
    ("market_trends",      MARKET_TRENDS_PROMPT),
    ("customer_insights",  CUSTOMER_INSIGHTS_PROMPT),
    ("strategy_synth",     STRATEGY_SYNTH_PROMPT),
]
