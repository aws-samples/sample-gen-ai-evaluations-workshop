---
name: understanding-failures
description: Systematically discover, categorize, and prioritize agent failure patterns from traces before building evaluators. Activate when asked to "review agent traces", "find failure patterns", "categorize agent problems", "prioritize what to fix", or "turn failures into evaluators".
---

# Understanding Failures: Discovering Patterns Before Building Evaluators

Systematically review agent traces to discover what's going wrong, group similar failures into actionable categories, prioritize by impact, fix what you can fix cheaply, and bridge remaining problems into automated evaluators — so you build the right checks instead of guessing.

## Prerequisites
- Completion of Module 01 (concepts: operational metrics, trace collection, latency tracking)
- Source notebook: `../../Foundational Evaluations/03-understanding-failures/01_Discovering_Failure_Patterns.ipynb`
- AWS services: Amazon Bedrock (Claude Sonnet)
- Python libraries: strands, json, collections

## Learning Objectives
By the end of this module, you will:
- Read agent traces and write specific, actionable notes about what went wrong
- Use LLM assistance to group individual observations into named problem categories
- Prioritize problem categories using a frequency × severity scoring system
- Fix the highest-priority problem with a targeted prompt change and verify via replay
- Convert a remaining problem category into a binary pass/fail evaluator question

## Setup

```python
from strands import Agent
from strands.models import BedrockModel
import json
import sys
import io
from collections import Counter
from IPython.display import display, Markdown

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
bedrock_model = BedrockModel(model_id=MODEL_ID)

# Load the restaurant booking agent traces
with open("data/raw_traces.json") as f:
    raw_traces = json.load(f)

print(f"Loaded {len(raw_traces)} raw traces")
```

## Section 1: Reading Traces and Writing Notes

**Concept:** Before you can measure failures, you need to observe them. Reading traces one at a time and writing notes about what went wrong is the foundation — everything that follows (grouping, prioritizing, fixing) depends on what you notice here. Focus on the **first failure** in each trace, because downstream symptoms are usually consequences of the initial problem.

**Build:**

```python
def display_trace(trace, trace_number=None):
    """Display a single trace with numbered turns and role labels.

    Args:
        trace: A dict with 'conversation_id' and 'messages' list.
        trace_number: Optional display number for the header.
    """
    conversation_id = trace.get("conversation_id", "(unknown)")
    messages = trace.get("messages", [])

    lines = []
    if trace_number is not None:
        lines.append(f"━━━ Trace {trace_number} ━━━")
    lines.append(f"**Conversation ID:** `{conversation_id}`")
    lines.append("")

    if not messages:
        lines.append("(no messages in this trace)")
        display(Markdown("\n".join(lines)))
        return

    for i, msg in enumerate(messages, 1):
        role = msg.get("role", None)
        content = msg.get("content", None)

        if content is not None and content.startswith("TOOL_CALL["):
            label = "⚠️ Tool Result"
        elif role == "user":
            label = "👤 User"
        elif role == "agent":
            label = "🤖 Agent"
        else:
            label = f"({role or 'missing role'})"

        display_content = content if content is not None else "(missing content)"
        lines.append(f"**Turn {i} — {label}**")
        lines.append(f"> {display_content}")
        lines.append("")

    display(Markdown("\n".join(lines)))
```

Display a trace and write a note about what went wrong:

```python
display_trace(raw_traces[0], trace_number=1)

# Write your note — aim for this level of specificity:
note_1 = {
    "conversation_id": raw_traces[0]["conversation_id"],
    "note": "Agent confirmed reservation but the GetBookingDetails tool returned an error — agent acted as if the booking went through when it had no connection to the database"
}
```

Use an LLM to suggest notes for traces you've already read:

```python
def suggest_note(trace):
    """Ask the LLM to suggest a note about what went wrong in a trace.

    Read the trace yourself first — this gives you a starting point
    to accept or correct.

    Args:
        trace: A dict with 'conversation_id' and 'messages' list.

    Returns:
        A string containing the LLM's suggested note.
    """
    note_agent = Agent(
        model=bedrock_model,
        system_prompt=(
            "You are helping label AI agent traces. "
            "Identify the first thing that went wrong — it could be a tool error, "
            "bad reasoning by the agent, a misunderstanding, or anything else. "
            "Write ONE or TWO sentences, max 50 words. Be specific and concise."
        )
    )

    trace_text = f"Conversation ID: {trace['conversation_id']}\n\n"
    for i, msg in enumerate(trace.get("messages", []), 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        trace_text += f"Turn {i} — {role}: {content}\n"

    prompt = f"What is the first thing that went wrong in this trace?\n\n{trace_text}"

    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        response = note_agent(prompt)
    finally:
        sys.stdout = _stdout

    return str(response).strip()
```

Collect notes for all traces:

```python
all_notes = []
for i, trace in enumerate(raw_traces[:10], 1):
    display_trace(trace, trace_number=i)
    suggestion = suggest_note(trace)
    note = {
        "conversation_id": trace["conversation_id"],
        "note": suggestion  # Replace with your own if the LLM got it wrong
    }
    all_notes.append(note)
    print(f"  Note: {note['note']}\n")

print(f"\nTotal notes collected: {len(all_notes)}")
```

## Section 2: Grouping Problems into Categories

**Concept:** Individual notes become actionable when you group them into categories. A good category name is specific enough that someone unfamiliar with the traces can immediately understand the problem (e.g., "Agent confirms action succeeded when the tool call failed" — not "bad response"). The LLM helps organize your observations, but it works from your notes — vague notes produce vague groupings.

**Build:**

```python
grouping_agent = Agent(
    model=bedrock_model,
    system_prompt="You are helping a workshop participant analyze notes about an AI agent's failures. Be specific and actionable in your suggestions."
)

GROUPING_PROMPT = """Here are notes from reviewing {count} conversations with a restaurant booking agent.
Each note describes something that went wrong or seemed concerning.

Notes:
{formatted_notes}

Based on these notes, suggest 3-6 distinct problem categories. For each category provide:
1. A specific, actionable name (e.g., "Agent confirms action succeeded when the tool call failed" — NOT vague names like "bad response")
2. A one-sentence definition
3. Which conversation IDs belong to this category

Format your response as a numbered list."""

formatted_notes = "\n".join(
    f"- [{n['conversation_id'][:12]}...] {n['note']}" for n in all_notes
)

response = grouping_agent(
    GROUPING_PROMPT.format(count=len(all_notes), formatted_notes=formatted_notes)
)
display(Markdown(str(response)))
```

Finalize your categories — merge, split, or rename the LLM's suggestions:

```python
problem_categories = {
    "Agent confirms action succeeded when the tool call failed": [
        "393b219a-ff56-4edb-a494-ab12c3fa1248",
        "0f186b8e-bf58-44ab-a117-a57ed6141c80",
        # Add conversation IDs that match this pattern
    ],
    "Agent fabricates details not present in any tool response": [
        "faf64bb8-4a60-4946-adf8-bd50468bc3b5",
        # Add more...
    ],
    # Add more categories as needed (aim for 3-6 total)
}

print("Problem Category Frequencies:\n")
for category, conv_ids in sorted(problem_categories.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {len(conv_ids)}x  {category}")
```

## Section 3: Prioritizing with Frequency × Severity

**Concept:** Not all problems are equally urgent. A problem that happens often AND misleads users is more urgent than a rare cosmetic issue. Priority = frequency × severity weight. This gives you a ranked list so you know exactly where to focus effort first — start with the cheapest fix for the highest-priority problem.

**Build:**

```python
SEVERITY_WEIGHTS = {
    "Critical": 4,  # User is misled or harmed
    "High": 3,      # Task fails silently
    "Medium": 2,    # Poor experience but task completes
    "Low": 1        # Cosmetic
}

severity_ratings = {
    "Agent confirms action succeeded when the tool call failed": "Critical",
    "Agent fabricates details not present in any tool response": "Critical",
    # Assign severity to each of your categories
}

def compute_priorities(problem_categories, severity_ratings):
    """Compute priority scores as frequency × severity weight.

    Args:
        problem_categories: Dict mapping category name to list of conversation IDs.
        severity_ratings: Dict mapping category name to severity level string.

    Returns:
        List of dicts sorted by priority_score descending.

    Raises:
        ValueError: If keys in problem_categories and severity_ratings don't match.
    """
    if not problem_categories:
        return []

    cat_keys = set(problem_categories.keys())
    sev_keys = set(severity_ratings.keys())
    if cat_keys != sev_keys:
        missing = cat_keys - sev_keys
        extra = sev_keys - cat_keys
        parts = []
        if missing:
            parts.append(f"Missing severity ratings for: {missing}")
        if extra:
            parts.append(f"Extra severity ratings for: {extra}")
        raise ValueError("; ".join(parts))

    results = []
    for category, conv_ids in problem_categories.items():
        severity = severity_ratings[category]
        weight = SEVERITY_WEIGHTS[severity]
        frequency = len(conv_ids)
        results.append({
            "category": category,
            "frequency": frequency,
            "severity": severity,
            "weight": weight,
            "priority_score": frequency * weight,
        })

    results.sort(key=lambda x: x["priority_score"], reverse=True)
    return results

priorities = compute_priorities(problem_categories, severity_ratings)

print("Priority Ranking:\n")
print(f"{'#':<3} {'Category':<60} {'Freq':>4} {'Severity':<9} {'Score':>5}")
print("-" * 85)
for i, p in enumerate(priorities, 1):
    cat = p['category'][:59] + '…' if len(p['category']) > 60 else p['category']
    print(f"{i:<3} {cat:<60} {p['frequency']:>4} {p['severity']:<9} {p['priority_score']:>5}")
```

Decision flowchart for what to do with each problem:

```
Problem discovered
  └─ Can I fix it by changing the prompt?
       ├─ YES → Fix it now. Add a regression test. Done.
       └─ NO → Is it frequent enough to justify building a checker?
              ├─ YES → Build an LLM-as-Judge evaluator (see Module 02).
              └─ NO → Log it. Revisit if frequency increases.
```

## Section 4: Fixing the Top Problem with a Prompt Change

**Concept:** The cheapest fix is often the best fix. A 5-minute prompt edit that eliminates a problem beats days of evaluator-building. The "hallucinated success" pattern — agent confirms actions that failed — happens because the prompt says nothing about handling tool errors. Adding explicit error-handling instructions fixes it at the source.

**Build:**

```python
ORIGINAL_SYSTEM_PROMPT = """You are a helpful restaurant booking assistant. You help customers with:
- Making new reservations
- Cancelling existing reservations
- Looking up booking details
- Modifying reservations

When helping customers, be polite and professional. Collect all necessary information
including the customer's name, party size, preferred date and time, and any special requests.

If you need to use tools to look up or modify bookings, do so proactively.
Always confirm the details with the customer before finalizing any changes.

Available tools:
- GetBookingDetails: Look up reservation information
- CreateBooking: Make a new reservation
- DeleteBooking: Cancel an existing reservation
- GetCustomerProfile: Retrieve customer information
- GenBookingArgs: Generate booking parameters from conversation context"""

# The gap: nothing about what to do when a tool call FAILS.
# The fix: add explicit error-handling instructions.

improved_prompt = ORIGINAL_SYSTEM_PROMPT + """

If a tool call returns an error, inform the customer that the action could not be completed
and explain what happened. Never confirm an action that failed. If you cannot complete a
request due to a tool error, apologize and suggest the customer try again later or contact
the restaurant directly."""

print("Added to prompt:")
print(improved_prompt[len(ORIGINAL_SYSTEM_PROMPT):])
```

Replay a trace with the improved prompt to verify the fix:

```python
def create_simulated_user(original_trace):
    """Create a simulated user that follows the original trace's script.

    If the agent diverges (e.g., reports an error instead of confirming success),
    the simulated user responds naturally rather than following the script.

    Args:
        original_trace: The original trace dict with 'messages' list.

    Returns:
        A Strands Agent configured as the simulated user.
    """
    trace_script = "\n".join(
        f"Turn {i} — {m.get('role', '?')}: {m.get('content', '')}"
        for i, m in enumerate(original_trace.get('messages', []), 1)
    )

    persona_prompt = f"""You are simulating a customer in a restaurant booking conversation.
Here is how the original conversation went:

{trace_script}

Rules:
- Follow the customer's messages from the original trace as closely as possible.
- If the agent responds similarly to the original, send the next customer message.
- If the agent says something DIFFERENT (e.g., reports an error), respond naturally.
- Keep responses short and natural (1-2 sentences).
- When the conversation reaches a natural end, output exactly: [END]"""

    return Agent(model=bedrock_model, system_prompt=persona_prompt)


def replay_conversation(booking_agent, original_trace, max_turns=20):
    """Replay a trace using persona simulation against the booking agent.

    Tool results from the original trace are injected at the same points,
    so both agents face the same environment. The only difference is the prompt.

    Args:
        booking_agent: The Strands Agent to test.
        original_trace: The original trace dict.
        max_turns: Maximum turns before stopping.

    Returns:
        A list of dicts with 'role' and 'content' for the replayed conversation.
    """
    sim_user = create_simulated_user(original_trace)

    tool_results = [
        m['content'] for m in original_trace.get('messages', [])
        if m.get('content', '').startswith('TOOL_CALL[')
    ]
    tool_idx = 0

    first_user_msgs = [
        m['content'] for m in original_trace.get('messages', [])
        if m.get('role') == 'user'
    ]
    if not first_user_msgs:
        return []

    conversation = []
    user_msg = first_user_msgs[0]
    conversation.append({'role': 'user', 'content': user_msg})
    _stdout = sys.stdout

    for turn in range(max_turns):
        agent_input = user_msg
        if tool_idx < len(tool_results):
            agent_input += f"\n\n[System: Tool result: {tool_results[tool_idx]}]"
            conversation.append({'role': 'tool', 'content': tool_results[tool_idx]})
            tool_idx += 1

        sys.stdout = io.StringIO()
        try:
            agent_response = booking_agent(agent_input)
        finally:
            sys.stdout = _stdout

        agent_text = str(agent_response).strip()
        conversation.append({'role': 'agent', 'content': agent_text})

        sys.stdout = io.StringIO()
        try:
            user_response = sim_user(f"The agent said: {agent_text}")
        finally:
            sys.stdout = _stdout

        user_text = str(user_response).strip()
        if '[END]' in user_text:
            break

        conversation.append({'role': 'user', 'content': user_text})
        user_msg = user_text

    return conversation


# Replay a trace that has a tool error
tool_error_traces = [
    t for t in raw_traces
    if any("TOOL_CALL[" in msg.get("content", "") and "Error" in msg.get("content", "")
           for msg in t.get("messages", []))
]

replay_agent = Agent(model=bedrock_model, system_prompt=improved_prompt)
replayed = replay_conversation(replay_agent, tool_error_traces[0])

print("Replayed conversation with improved prompt:\n")
for msg in replayed:
    role = msg['role']
    emoji = '👤' if role == 'user' else '🤖' if role == 'agent' else '⚠️'
    print(f"{emoji} {msg['content']}\n")
```

## Section 5: Bridging into Evaluator Design

**Concept:** Not every problem can be fixed with a prompt change. For persistent failure modes, you need automated checking — an evaluator that reviews traces and flags failures without a human reading every conversation. The bridge: take a problem category, write it as a single binary yes/no question specific enough that two readers would give the same answer, then test it as a judge prompt. This is the entry point to Module 02's full evaluator design workflow.

**Build:**

```python
# Pick a problem category that CAN'T be fixed with a prompt change.
# Write it as a single yes/no question a judge can answer from a trace.

binary_question = "Did the agent fabricate specific details (like table numbers, refund amounts, or booking confirmations) without data from a tool call?"

JUDGE_PROMPT_TEMPLATE = """You are evaluating a conversation between a user and a restaurant booking agent.

Review the conversation below and answer this question:
{binary_question}

Respond with a JSON object:
{{"verdict": "PASS" or "FAIL", "reasoning": "one sentence explanation"}}

Conversation:
{trace_text}"""

judge_agent = Agent(
    model=bedrock_model,
    system_prompt="You are a precise evaluator. Answer exactly as instructed."
)

# Test against a few traces to see if the question produces useful verdicts
for trace in raw_traces[:3]:
    trace_text = "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in trace["messages"]
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        binary_question=binary_question,
        trace_text=trace_text
    )

    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        response = judge_agent(prompt)
    finally:
        sys.stdout = _stdout

    print(f"Trace {trace['conversation_id'][:12]}...")
    print(f"Judge says: {response}\n---")
```

The pattern: one problem → one binary question → one evaluator. If your question covers multiple failure modes, split it. Three focused evaluators that each do one thing well beat one confused evaluator trying to check three things at once.

## Challenges

### Challenge: Full Failure Discovery Pipeline on a New Domain

Given a set of agent traces from a **different domain** (e.g., a customer support agent, a code review assistant, or a travel planning agent), run the complete failure discovery pipeline:

1. Read traces and write notes about what went wrong
2. Group notes into ≥3 problem categories with specific, actionable names
3. Assign severity ratings and compute priority scores
4. For the top-priority problem: determine whether it's fixable with a prompt change
   - If yes: write the improved prompt and explain what gap you're filling
   - If no: write a binary pass/fail question for an LLM-as-Judge evaluator
5. For a second problem (different from #4): do the opposite approach — if you fixed #4 with a prompt, write a judge question for #5, and vice versa

**Assessment criteria:**
- Produces ≥3 problem categories with specific names (not "bad response" or "error handling")
- Each category has ≥2 conversation IDs assigned to it
- Priority scores are computed correctly (frequency × severity weight)
- Prompt fix addresses a specific gap identified in the original prompt (not generic additions)
- Binary question is answerable by reading a single trace — two readers would give the same verdict
- Learner explains why one problem is prompt-fixable and the other requires an evaluator

## Wrap-Up

**Key takeaways:**
- Read traces before building evaluators — you can't measure what you don't understand
- Fix problems before automating detection — many failures disappear with a prompt change
- LLMs assist but don't replace observation — the LLM grouped your notes, but couldn't have written them
- One problem → one question → one evaluator — keep each check focused on a single failure mode
- Revisit regularly — failure patterns change as you update prompts, switch models, or add features

**Connections to other modules:**
- Module 01 (Operational Metrics): Anomalies like high latency or unusual token counts signal where to start trace review
- Module 02 (Quality Metrics): The binary pass/fail pattern sketched in Section 5 is taught in depth in Module 02's judge calibration notebook
- Module 04 (Agentic Metrics): These same restaurant booking traces are used for agentic evaluation — the problem categories you built here directly inform what metrics to track

**Next steps:**
- For a cross-module challenge integrating failure discovery with evaluator design and operational monitoring, see [CHALLENGE-capstone.md](../CHALLENGE-capstone.md)
