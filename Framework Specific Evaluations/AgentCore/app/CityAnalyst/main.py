import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from botocore.config import Config
from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.models.bedrock import BedrockModel


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "data" / "city_facts.json"
SYSTEM_PROMPT_PATH = APP_ROOT / "system_prompt.txt"
MODEL_ID = os.getenv("CITY_AGENT_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
MAX_SESSION_AGENTS = 128

app = BedrockAgentCoreApp()
logger = app.logger


def _load_city_index() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {
        (item["city"].casefold(), item["state"].upper()): item
        for item in payload["cities"]
    }


CITY_INDEX = _load_city_index()
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _city_record(city: str, state: str) -> dict[str, Any] | None:
    item = CITY_INDEX.get((city.strip().casefold(), state.strip().upper()))
    if item is None:
        return None
    population = int(item["population"])
    land_area = float(item["land_area_mi2"])
    return {
        **item,
        "density_per_mi2": round(population / land_area, 1),
    }


@tool
def lookup_city(city: str, state: str) -> dict[str, Any]:
    """Look up one city in the fixed workshop dataset.

    Args:
        city: Full US city name.
        state: Two-letter US state abbreviation.
    """
    record = _city_record(city, state)
    if record is None:
        return {
            "found": False,
            "city": city.strip(),
            "state": state.strip().upper(),
            "message": "City is not present in the workshop dataset.",
        }
    return {"found": True, **record}


@tool
def compare_cities(
    city_a: str,
    state_a: str,
    city_b: str,
    state_b: str,
) -> dict[str, Any]:
    """Compare population, land area, and density for two workshop cities.

    Args:
        city_a: First full city name.
        state_a: First two-letter state abbreviation.
        city_b: Second full city name.
        state_b: Second two-letter state abbreviation.
    """
    first = _city_record(city_a, state_a)
    second = _city_record(city_b, state_b)
    missing = [
        f"{city}, {state.upper()}"
        for city, state, record in (
            (city_a, state_a, first),
            (city_b, state_b, second),
        )
        if record is None
    ]
    if missing:
        return {
            "found": False,
            "missing": missing,
            "message": "One or more cities are not present in the workshop dataset.",
        }
    return {
        "found": True,
        "cities": [first, second],
        "population_difference": abs(first["population"] - second["population"]),
        "land_area_difference_mi2": round(
            abs(first["land_area_mi2"] - second["land_area_mi2"]),
            1,
        ),
        "denser_city": (
            f"{first['city']}, {first['state']}"
            if first["density_per_mi2"] > second["density_per_mi2"]
            else f"{second['city']}, {second['state']}"
        ),
    }


@tool
def calculate_density(population: int, land_area_mi2: float) -> dict[str, float]:
    """Calculate people per square mile from supplied numeric values.

    Args:
        population: Non-negative population count.
        land_area_mi2: Positive land area in square miles.
    """
    if population < 0:
        raise ValueError("population must be non-negative")
    if land_area_mi2 <= 0:
        raise ValueError("land_area_mi2 must be positive")
    return {"density_per_mi2": round(population / land_area_mi2, 1)}


class CityFact(BaseModel):
    city: str
    state: str
    population: int | None = None
    land_area_mi2: float | None = None
    density_per_mi2: float | None = None


class CityResponse(BaseModel):
    answer: str = Field(description="Concise answer for the user")
    cities: list[CityFact] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


def _build_agent() -> Agent:
    model_config = Config(
        retries={"total_max_attempts": 5, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=120,
    )
    model = BedrockModel(
        model_id=MODEL_ID,
        max_tokens=1000,
        temperature=0,
        boto_client_config=model_config,
    )
    return Agent(
        name="CityAnalyst",
        description="Answers questions using a fixed city-facts dataset.",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[lookup_city, compare_cities, calculate_density],
        callback_handler=None,
        conversation_manager=NullConversationManager(),
        trace_attributes={
            "workshop.module": "agentcore-evaluations",
            "workshop.agent": "CityAnalyst",
        },
    )


_session_agents: OrderedDict[str, Agent] = OrderedDict()
_session_lock = threading.Lock()


def _get_session_agent(session_id: str) -> Agent:
    with _session_lock:
        if session_id in _session_agents:
            _session_agents.move_to_end(session_id)
            return _session_agents[session_id]
        if len(_session_agents) >= MAX_SESSION_AGENTS:
            _session_agents.popitem(last=False)
        agent = _build_agent()
        _session_agents[session_id] = agent
        return agent


def _extract_prompt(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("payload.prompt must be a non-empty string")
    return prompt.strip()


@app.entrypoint
async def invoke(payload, context):
    prompt = _extract_prompt(payload)
    session_id = getattr(context, "session_id", "local-session")
    logger.info("Invoking CityAnalyst", extra={"session_id": session_id})

    result = await _get_session_agent(session_id).invoke_async(
        prompt,
        structured_output_model=CityResponse,
    )
    if result.structured_output is None:
        raise RuntimeError("Agent did not return the required structured output")
    return result.structured_output.model_dump(mode="json")


if __name__ == "__main__":
    app.run()

