---
name: Chatbot Multi-Turn Evaluation
description: Help me evaluate a multi-turn chatbot, simulate realistic conversations, build custom binary evaluators from failure modes, generate dimension-driven test data, and assemble an end-to-end evaluation pipeline
---

# Evaluating Multi-Turn Chatbots

Single-turn evaluation follows a simple pattern: input, output, judge. Multi-turn conversations break this model because turn five is not just a response to message five — it is a response to the entire trajectory of messages one through four. This module teaches you to evaluate multi-turn chatbots systematically: simulate realistic conversations at scale, build custom binary evaluators tied to concrete failure modes, generate test data along structured dimensions, and wire it all into a repeatable pipeline that reports pass rate per failure mode.

## Prerequisites

- AWS account with Bedrock access (us-east-1)
- Access to Claude Sonnet 4.5 and Amazon Nova Micro in Bedrock
- Python 3.10+
- Familiarity with boto3 and the Bedrock Converse API
- Completed SKILL-operational or equivalent understanding of LLM invocation

## Learning Objectives

By the end of this module, you will be able to:

1. Identify common multi-turn failure modes and explain why they require different evaluation approaches than single-turn systems
2. Simulate goal-driven multi-turn conversations using Strands ActorSimulator and DeepEval ConversationSimulator
3. Build custom binary pass/fail evaluators from domain-specific failure modes using both frameworks
4. Generate synthetic test cases along structured dimensions derived from failure hypotheses
5. Assemble a full evaluation pipeline: dimensions → simulation → custom evaluators → pass-rate-per-failure-mode reporting

## Setup

```bash
pip install strands-agents strands-agents-evals deepeval boto3
```

```python
import boto3
import json
from strands import Agent, tool
from datetime import date

# Verify Bedrock access
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
for model_id in ['us.anthropic.claude-sonnet-4-5-20250929-v1:0', 'amazon.nova-micro-v1:0']:
    resp = bedrock.converse(
        modelId=model_id,
        messages=[{'role': 'user', 'content': [{'text': 'Say hello in one word.'}]}],
        inferenceConfig={'maxTokens': 10}
    )
    print(f"OK {model_id}: {resp['output']['message']['content'][0]['text']}")
```

Build the travel booking agent — the evaluation target throughout this module:

```python
bookings: dict = {}
booking_counter = 0

@tool
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> str:
    """Search for available flights between two cities."""
    flights = [
        {"flight": "AA101", "airline": "American Airlines", "departs": "08:00", "arrives": "20:00", "price": 450, "class": "Economy"},
        {"flight": "BA202", "airline": "British Airways", "departs": "11:30", "arrives": "23:45", "price": 620, "class": "Economy"},
        {"flight": "UA303", "airline": "United Airlines", "departs": "14:00", "arrives": "02:15", "price": 890, "class": "Business"},
    ]
    trip = f"round-trip (return: {return_date})" if return_date else "one-way"
    lines = [f"Flights from {origin} to {destination} on {departure_date} ({trip}):"]
    for f in flights:
        lines.append(f" {f['flight']} | {f['airline']} | {f['departs']}-{f['arrives']} | ${f['price']} | {f['class']}")
    return "\n".join(lines)

@tool
def search_hotels(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search for available hotels in a city."""
    hotels = [
        {"name": "Grand Plaza Hotel", "stars": 5, "price_per_night": 320, "amenities": "Pool, Spa, Restaurant"},
        {"name": "City Center Inn", "stars": 3, "price_per_night": 95, "amenities": "Free WiFi, Breakfast"},
        {"name": "Boutique Stay", "stars": 4, "price_per_night": 180, "amenities": "Gym, Bar, Concierge"},
    ]
    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    lines = [f"Hotels in {city} | {check_in} to {check_out} ({nights} nights, {guests} guest(s)):"]
    for h in hotels:
        total = h["price_per_night"] * nights
        lines.append(f" {h['name']} ({h['stars']}*) | ${h['price_per_night']}/night (${total} total) | {h['amenities']}")
    return "\n".join(lines)

@tool
def book_flight(flight_number: str, passenger_name: str, origin: str, destination: str, travel_date: str) -> str:
    """Book a flight for a passenger."""
    global booking_counter
    booking_counter += 1
    ref = f"FLT-{booking_counter:04d}"
    bookings[ref] = {"type": "flight", "flight": flight_number, "passenger": passenger_name,
                     "route": f"{origin} -> {destination}", "date": travel_date, "status": "Confirmed"}
    return f"Flight booked! Ref: {ref} | {flight_number} | {origin} -> {destination} on {travel_date} | Passenger: {passenger_name}"

@tool
def book_hotel(hotel_name: str, guest_name: str, city: str, check_in: str, check_out: str) -> str:
    """Book a hotel room for a guest."""
    global booking_counter
    booking_counter += 1
    ref = f"HTL-{booking_counter:04d}"
    bookings[ref] = {"type": "hotel", "hotel": hotel_name, "guest": guest_name,
                     "city": city, "check_in": check_in, "check_out": check_out, "status": "Confirmed"}
    return f"Hotel booked! Ref: {ref} | {hotel_name} in {city} | {check_in} to {check_out} | Guest: {guest_name}"

@tool
def get_bookings() -> str:
    """Retrieve all current bookings."""
    if not bookings:
        return "No bookings found."
    lines = ["Current bookings:"]
    for ref, b in bookings.items():
        if b["type"] == "flight":
            lines.append(f" [{ref}] Flight {b['flight']} | {b['route']} on {b['date']} | {b['passenger']} | {b['status']}")
        else:
            lines.append(f" [{ref}] Hotel {b['hotel']} in {b['city']} | {b['check_in']} to {b['check_out']} | {b['guest']} | {b['status']}")
    return "\n".join(lines)

@tool
def cancel_booking(booking_ref: str) -> str:
    """Cancel an existing booking."""
    if booking_ref not in bookings:
        return f"Booking {booking_ref} not found."
    bookings[booking_ref]["status"] = "Cancelled"
    return f"Booking {booking_ref} has been cancelled."

ALL_TOOLS = [search_flights, search_hotels, book_flight, book_hotel, get_bookings, cancel_booking]

SYSTEM_PROMPT = (
    "You are a helpful travel booking assistant. You help users search for flights and hotels, "
    "make bookings, view existing reservations, and cancel bookings. "
    "Always confirm details with the user before completing a booking. "
    "Use today's date as context when dates are not fully specified."
)

agent = Agent(system_prompt=SYSTEM_PROMPT, tools=ALL_TOOLS, callback_handler=None)
print(f"Agent ready with {len(ALL_TOOLS)} tools.")
```

## Section 1: Multi-Turn Failure Modes and Evaluation Strategy

**Concept:** Multi-turn conversations introduce failure modes that don't exist in single-turn systems. The search space is unpredictable (you can't predefine what the "correct" exchange looks like five turns deep), and quality is cumulative (one weak turn can derail everything that follows).

Common multi-turn failure modes for a travel booking agent:

| Category | Failure | Example |
|----------|---------|---------|
| Context | Forgetting earlier state | Asking for a name provided four turns ago |
| Context | Self-contradiction | Recommending a flight, then saying it's unavailable |
| Response | Booking without confirmation | Skipping the "shall I proceed?" step |
| Response | Inventing data | Referencing a flight not in search results |
| Flow | Infinite clarification loops | Asking the same question repeatedly |
| Flow | Topic drift | User asked about flights, agent discusses travel tips |

The evaluation strategy uses three granularities:

| Level | Question | Evaluator Type |
|-------|----------|---------------|
| Turn-level | Was this specific response acceptable? | Binary rubric per turn |
| Session-level | Did the full conversation achieve its goal? | Assertion-based pass/fail |
| Output-level | Is this response well-formed against domain rules? | Domain rubric check |

The workflow: **identify failure modes → define scenarios → simulate → evaluate with custom binary criteria → analyze pass rates → iterate**.

> Source: `01 intro and setup.ipynb`

## Section 2: Simulating Multi-Turn Conversations

**Concept:** Manual testing doesn't scale, and historical replay becomes irrelevant when you change the agent. Simulation generates realistic multi-turn conversations from structured scenarios, giving you reproducible, scalable, forward-looking evaluation data.

Two frameworks provide simulation: Strands `ActorSimulator` (goal-driven with persona generation) and DeepEval `ConversationSimulator` (scenario-driven with custom models).

**Strands ActorSimulator** — generates a consistent persona from a test case, tracks goal completion, and adapts user behavior to agent responses:

```python
from strands_evals import Case, ActorSimulator
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

# Define a test case with task description for persona generation
case = Case(
    name="tokyo-business-trip",
    input="I need to plan a business trip to Tokyo. I'll be there October 15-19, 2025.",
    expected_trajectory=["search_flights", "book_flight", "search_hotels", "book_hotel"],
    expected_assertion=(
        "The agent searched for flights to Tokyo, booked one, "
        "searched for hotels for Oct 15-19, and booked one. "
        "The user has confirmed booking references for both."
    ),
    metadata={
        "task_description": (
            "Plan a complete business trip to Tokyo: search for flights, "
            "compare options, book a flight, then search for hotels in Tokyo "
            "for 4 nights (Oct 15-19), and book a hotel."
        ),
    }
)

# Setup telemetry for trace capture
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

# Create simulator from case (auto-generates user persona)
simulator = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=8)

# Create agent with tracing
sim_agent = Agent(
    system_prompt=SYSTEM_PROMPT, tools=ALL_TOOLS,
    trace_attributes={"session.id": case.session_id},
    callback_handler=None,
)

# Run conversation loop
user_message = case.input
all_spans = []

while simulator.has_next():
    memory_exporter.clear()
    agent_response = sim_agent(user_message)
    all_spans.extend(list(memory_exporter.get_finished_spans()))
    user_result = simulator.act(str(agent_response))
    user_message = str(user_result.structured_output.message)

# Map spans to session for evaluation
mapper = StrandsInMemorySessionMapper()
session = mapper.map_to_session(all_spans, session_id=case.session_id)
print(f"Conversation complete. Session has {len(session.invocations)} invocations.")
```

**DeepEval ConversationSimulator** — uses `ConversationalGolden` scenarios with custom Bedrock models for user simulation:

```python
from deepeval.dataset import ConversationalGolden
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn, ToolCall
from deepeval.models import DeepEvalBaseLLM

class BedrockNovaMicro(DeepEvalBaseLLM):
    """Amazon Nova Micro as the user simulator model."""
    MODEL_ID = "amazon.nova-micro-v1:0"

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        super().__init__(model=self.MODEL_ID)

    def load_model(self):
        return boto3.client("bedrock-runtime", region_name=self.region)

    def get_model_name(self) -> str:
        return self.MODEL_ID

    def _invoke(self, prompt: str, schema=None) -> str:
        system_text = "You are a helpful assistant. Respond only in valid JSON." if schema else None
        if schema:
            json_schema = schema.model_json_schema()
            prompt = f"{prompt}\n\nReturn ONLY valid JSON matching this schema:\n{json.dumps(json_schema, indent=2)}"
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        body = {"messages": messages, "inferenceConfig": {"maxTokens": 4096, "temperature": 0.0}}
        if system_text:
            body["system"] = [{"text": system_text}]
        response = self.model.converse(modelId=self.MODEL_ID, **body)
        text = response["output"]["message"]["content"][0]["text"]
        if schema:
            return schema.model_validate_json(text)
        return text

    def generate(self, prompt: str, schema=None) -> str:
        return self._invoke(prompt, schema)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self._invoke(prompt, schema)

nova_micro = BedrockNovaMicro()

# Define scenarios
goldens = [
    ConversationalGolden(
        scenario="Solo traveler searching for a cheap one-way flight and booking the most affordable option",
        expected_outcome="User finds a flight and receives a confirmed booking reference",
        user_description="Budget-conscious traveler who prioritizes low cost over comfort"
    ),
    ConversationalGolden(
        scenario="User with multiple bookings who wants to review them before cancelling one",
        expected_outcome="User retrieves bookings, selects one to cancel, and verifies the rest are intact",
        user_description="Cautious user who double-checks details before making changes"
    ),
]

# Agent callback for DeepEval
def agent_callback(input: str) -> Turn:
    response = agent(input)
    text = response.message["content"][0]["text"] if response.message.get("content") else str(response)
    tools = [ToolCall(name=tm.tool["name"], input_parameters=tm.tool.get("input"))
             for tm in response.metrics.tool_metrics.values()] or None
    return Turn(role="assistant", content=text, tools_called=tools)

# Run simulation
sim = ConversationSimulator(model_callback=agent_callback, simulator_model=nova_micro, async_mode=False)
test_cases = sim.simulate(conversational_goldens=goldens, max_user_simulations=10)
print(f"Simulated {len(test_cases)} conversations.")
```

> Source: `02 strands simulation.ipynb`, `03 deepeval simulation.ipynb`

## Section 3: Custom Binary Evaluators from Failure Modes

**Concept:** Generic numeric metrics ("helpfulness: 0.75") are not actionable. Custom binary evaluators say "this response invented a flight not in search results — FAIL." They are actionable, auditable, stable across judges, and tied to real user impact. The pattern: enumerate failure modes → write one binary criterion per mode → run as pass/fail.

**DeepEval ConversationalGEval** — each metric is a single yes/no question with `threshold=1.0` for strict binary semantics:

```python
from deepeval.metrics import ConversationalGEval, ConversationalDAGMetric
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode, ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode, ConversationalVerdictNode,
)
from deepeval.test_case import TurnParams
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig

# Claude Sonnet 4.5 as the evaluation judge (same wrapper pattern as BedrockNovaMicro)
class BedrockClaudeSonnet45(DeepEvalBaseLLM):
    MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        super().__init__(model=self.MODEL_ID)

    def load_model(self):
        return boto3.client('bedrock-runtime', region_name=self.region)

    def get_model_name(self) -> str:
        return self.MODEL_ID

    def _invoke(self, prompt: str, schema=None) -> str:
        system_text = 'You are a helpful assistant. Respond only in valid JSON.' if schema else None
        if schema:
            json_schema = schema.model_json_schema()
            prompt = (f'{prompt}\n\nReturn ONLY valid JSON matching this schema. '
                      f'No markdown fences, no commentary. JSON only.\n'
                      f'{json.dumps(json_schema, indent=2)}')
        messages = [{'role': 'user', 'content': [{'text': prompt}]}]
        body = {'messages': messages, 'inferenceConfig': {'maxTokens': 8192, 'temperature': 0.0}}
        if system_text:
            body['system'] = [{'text': system_text}]
        response = self.model.converse(modelId=self.MODEL_ID, **body)
        text = response['output']['message']['content'][0]['text']
        if schema:
            return schema.model_validate_json(text)
        return text

    def generate(self, prompt: str, schema=None) -> str:
        return self._invoke(prompt, schema)

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self._invoke(prompt, schema)

judge_model = BedrockClaudeSonnet45()

# One binary metric per failure mode
metric_no_invented = ConversationalGEval(
    name='F1_no_invented_flights',
    criteria=(
        'Binary check: does the assistant mention or book any flight number not returned '
        'by search_flights earlier in the conversation? '
        'PASS (1.0) if every flight referenced came from search results. '
        'FAIL (0.0) if any flight was invented.'
    ),
    model=judge_model, threshold=1.0,
)

metric_confirmed = ConversationalGEval(
    name='F2_confirmed_before_booking',
    criteria=(
        'Binary check: if a booking happened, did the assistant explicitly ask the user '
        'to confirm details and receive acknowledgement BEFORE producing the booking? '
        'PASS if confirmation preceded every booking, or no booking happened. '
        'FAIL if any booking was produced without prior user confirmation.'
    ),
    model=judge_model, threshold=1.0,
)

metric_retention = ConversationalGEval(
    name='F3_remembers_state',
    criteria=(
        'Binary check: did the assistant ask the user to repeat a fact already provided '
        '(name, destination, date, or booking reference)? '
        'PASS if all earlier facts were carried forward. '
        'FAIL if the assistant forgot any earlier-provided fact.'
    ),
    model=judge_model, threshold=1.0,
)

# Run metrics
results = evaluate(
    test_cases=test_cases,
    metrics=[metric_no_invented, metric_confirmed, metric_retention],
    async_config=AsyncConfig(run_async=False),
)
```

**ConversationalDAGMetric** — for ordered checks where later criteria only apply if earlier ones pass:

```python
def make_task_then_tone_metric():
    """Task-completion gates tone evaluation."""
    tone_node = ConversationalNonBinaryJudgementNode(
        criteria="Classify the assistant's overall tone.",
        children=[
            ConversationalVerdictNode(verdict='Rude', score=0),
            ConversationalVerdictNode(verdict='Playful', score=5),
            ConversationalVerdictNode(verdict='Professional', score=10),
        ],
    )
    task_gate = ConversationalBinaryJudgementNode(
        criteria="Did the assistant satisfy all user requests?",
        children=[
            ConversationalVerdictNode(verdict=False, score=0),
            ConversationalVerdictNode(verdict=True, child=tone_node),
        ],
    )
    root = ConversationalTaskNode(
        instructions='Summarize the conversation before evaluating.',
        output_label='Summary',
        evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
        children=[task_gate],
    )
    dag = DeepAcyclicGraph(root_nodes=[root])
    return ConversationalDAGMetric(name='TaskThenTone', model=judge_model, dag=dag)
```

**Strands GoalSuccessRateEvaluator** — session-level binary evaluation using assertions:

```python
from strands_evals import Experiment
from strands_evals.evaluators import GoalSuccessRateEvaluator, OutputEvaluator

# Session-level: did the full conversation achieve its goal?
goal_evaluator = GoalSuccessRateEvaluator(
    model_id='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    region_name='us-east-1'
)

experiment = Experiment(
    cases=[case],
    evaluators=[goal_evaluator],
)
experiment_results = experiment.run(session=session)
print(f"Goal success: {experiment_results[0].score}")

# Output-level: binary rubric per failure mode
no_hallucination_rubric = """
Evaluate the agent's FINAL response against this criterion:
Does the response reference any flight number, hotel name, or booking reference
that was NOT produced by a tool call earlier in the conversation?
Return PASS if all references are grounded in tool outputs.
Return FAIL if any reference is invented.
"""

output_evaluator = OutputEvaluator(
    rubric=no_hallucination_rubric,
    model_id='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    region_name='us-east-1'
)
```

> Source: `04 deepeval metrics.ipynb`, `05 strands evaluators.ipynb`

## Section 4: Dimension-Driven Synthetic Test Data

**Concept:** "Ask an LLM for test queries" produces repetitive, narrow coverage. Dimension-driven generation starts from failure hypotheses, derives structured dimensions, and combines them into tuples that give controlled, deliberate coverage of the behaviors you want to probe.

The workflow: **failure hypotheses → dimensions → seed tuples → LLM-scaled generation → natural-language queries**.

```python
# Step 1: Failure hypotheses for the travel booking agent
# H1: Agent invents flights not in search results
# H2: Agent skips confirmation when user sounds decisive
# H3: Agent forgets state across many turns or topic switches
# H4: Agent handles past/malformed dates poorly
# H5: Agent struggles with budget constraints
# H6: Agent cancels the wrong booking when user is ambiguous
# H7: Agent breaks scope on unrelated requests

# Step 2: Derive dimensions from hypotheses
DIMENSIONS = {
    'IntentType': ['search', 'book', 'review', 'cancel', 'modify'],
    'Complexity': ['single_leg', 'multi_step', 'pivot'],
    'UserMood': ['neutral', 'impatient', 'price_sensitive', 'uncertain'],
    'EdgeCase': ['none', 'past_date', 'invalid_city', 'ambiguous_ref', 'off_topic'],
}

total = 1
for vals in DIMENSIONS.values():
    total *= len(vals)
print(f"Dimension space: {total} possible tuples")

# Step 3: Hand-write seed tuples (validate the schema works)
SEED_TUPLES = [
    ('search', 'single_leg', 'neutral', 'none'),
    ('book', 'multi_step', 'price_sensitive', 'none'),
    ('cancel', 'single_leg', 'uncertain', 'ambiguous_ref'),
    ('book', 'pivot', 'uncertain', 'none'),
    ('book', 'single_leg', 'impatient', 'past_date'),
    ('search', 'single_leg', 'neutral', 'invalid_city'),
    ('book', 'single_leg', 'neutral', 'off_topic'),
]

# Step 4: Two-step LLM generation (tuple → natural language query)
TUPLE_TO_QUERY_PROMPT = """Convert this test specification into a natural first message
from a user to a travel booking assistant.

Specification:
- Intent: {intent}
- Complexity: {complexity}
- User mood: {mood}
- Edge case: {edge}

Write ONE realistic opening message. No explanation, just the message."""

def generate_query(intent, complexity, mood, edge):
    prompt = TUPLE_TO_QUERY_PROMPT.format(intent=intent, complexity=complexity, mood=mood, edge=edge)
    resp = bedrock.converse(
        modelId='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        messages=[{'role': 'user', 'content': [{'text': prompt}]}],
        inferenceConfig={'maxTokens': 200, 'temperature': 0.7},
    )
    return resp['output']['message']['content'][0]['text']

# Generate queries from seed tuples
generated_cases = []
for intent, complexity, mood, edge in SEED_TUPLES:
    query = generate_query(intent, complexity, mood, edge)
    generated_cases.append(Case(
        name=f"{intent}-{complexity}-{mood}-{edge}",
        input=query,
        metadata={"intent": intent, "complexity": complexity, "mood": mood, "edge": edge},
    ))
    print(f"  [{intent}/{mood}/{edge}] {query[:80]}...")
```

> Source: `06 synthetic data.ipynb`

## Section 5: End-to-End Evaluation Pipeline

**Concept:** The full pipeline wires everything together: dimension-driven cases → multi-turn simulation → custom binary evaluators → pass-rate-per-failure-mode reporting. This is the shape of a production evaluation pipeline you can run in CI on every agent change.

**ToolSimulator** for fully simulated conversations (no real tools, no real users):

```python
from strands_evals.simulation.tool_simulator import ToolSimulator
from pydantic import BaseModel, Field

simulator = ToolSimulator()

class FlightSearchResponse(BaseModel):
    flights: list[dict] = Field(description='Available flights with airline, number, price, class')
    origin: str
    destination: str

@simulator.tool(output_schema=FlightSearchResponse, share_state_id='travel_bookings',
    initial_state_description='Airline system with flights to major cities. Prices $300-$1200.')
def search_flights(origin: str, destination: str, departure_date: str, return_date: str = None) -> dict:
    """Search for available flights."""
    pass  # LLM generates the response

# Register all tools similarly, then combine with ActorSimulator:
# ActorSimulator (fake user) → Agent (real, under test) → ToolSimulator (fake tools)
```

**The full pipeline** — dimensions → simulation → evaluation → reporting:

```python
from strands_evals import Case, Experiment, ActorSimulator
from strands_evals.evaluators import GoalSuccessRateEvaluator, OutputEvaluator
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

JUDGE_MODEL = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

# Define evaluators for each failure mode
evaluators = {
    'F1_no_hallucination': OutputEvaluator(
        rubric="Does the response reference any data not produced by a tool call? PASS if grounded, FAIL if invented.",
        model_id=JUDGE_MODEL, region_name='us-east-1'),
    'F2_confirmation': OutputEvaluator(
        rubric="If a booking occurred, was explicit user confirmation obtained first? PASS if yes or no booking. FAIL otherwise.",
        model_id=JUDGE_MODEL, region_name='us-east-1'),
    'F3_goal_success': GoalSuccessRateEvaluator(
        model_id=JUDGE_MODEL, region_name='us-east-1'),
}

# Run pipeline for each generated case
results_by_mode = {name: [] for name in evaluators}

for case in generated_cases:
    # Reset state
    bookings.clear()
    booking_counter = 0

    # Simulate conversation
    sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=8)
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    exporter = telemetry.in_memory_exporter

    eval_agent = Agent(
        system_prompt=SYSTEM_PROMPT, tools=ALL_TOOLS,
        trace_attributes={"session.id": case.session_id},
        callback_handler=None,
    )

    user_msg = case.input
    spans = []
    while sim.has_next():
        exporter.clear()
        resp = eval_agent(user_msg)
        spans.extend(list(exporter.get_finished_spans()))
        user_result = sim.act(str(resp))
        user_msg = str(user_result.structured_output.message)

    session = StrandsInMemorySessionMapper().map_to_session(spans, session_id=case.session_id)

    # Evaluate against each failure mode
    for name, evaluator in evaluators.items():
        experiment = Experiment(cases=[case], evaluators=[evaluator])
        result = experiment.run(session=session)
        results_by_mode[name].append(result[0].score)

# Report: pass rate per failure mode
print("\n=== Pass Rate Per Failure Mode ===")
for name, scores in results_by_mode.items():
    pass_rate = sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0
    print(f"  {name:25s} {pass_rate:6.0%} ({sum(1 for s in scores if s >= 1.0)}/{len(scores)})")
```

> Source: `07 tool simulation.ipynb`, `08 e2e pipeline.ipynb`

## Challenges

### Challenge: Evaluate a Customer Support Agent

Apply the full evaluation workflow to a **customer support agent** (not travel booking). The agent handles order status inquiries, returns/refunds, and product questions.

1. Write ≥3 failure hypotheses specific to customer support (e.g., "agent promises refunds it can't authorize")
2. Derive ≥3 dimensions with closed value sets from those hypotheses
3. Write ≥5 seed tuples covering core workflows and edge cases
4. Implement ≥2 custom binary evaluators: one session-level (GoalSuccessRateEvaluator with assertion) and one output-level (OutputEvaluator with domain rubric)
5. Run a simulation pipeline that produces a pass-rate-per-failure-mode table

**Constraint:** At least one failure hypothesis must involve multi-turn context retention (the agent forgetting order details across turns), and at least one must involve scope boundaries (the agent being asked something outside its domain).

**Assessment criteria:**
1. Produces ≥3 failure hypotheses that are specific to customer support (not generic)
2. Dimensions have ≥3 values each and map clearly to the hypotheses
3. Seed tuples cover at least one edge case and one multi-step workflow
4. Both evaluators produce binary pass/fail verdicts with clear criteria
5. Pipeline runs end-to-end and produces a pass-rate table with ≥2 failure modes reported

For a larger challenge integrating evaluation across multiple workload types, see `CHALLENGE-capstone.md`.

## Wrap-Up

**Key takeaways:**
- Multi-turn evaluation requires simulation because manual testing doesn't scale and historical replay becomes irrelevant after agent changes
- Custom binary evaluators (one per failure mode) are more actionable than generic numeric metrics
- Dimension-driven test generation gives controlled coverage tied to specific failure hypotheses
- The production pipeline shape is: dimensions → simulation → custom evaluators → pass-rate-per-failure-mode

**This module does NOT cover:**
- Single-turn evaluation (see SKILL-operational)
- Guardrail configuration (see SKILL-guardrails)
- Framework-specific evaluation setup (see SKILL-promptfoo, SKILL-dspy)
- Human labeling workflows for judge calibration

**Next steps:**
- Apply this pipeline to your own chatbot or agent
- Grow the scenario set: every production incident becomes a new test case + evaluator
- Integrate into CI/CD: run the same pipeline on every agent change to catch regressions
