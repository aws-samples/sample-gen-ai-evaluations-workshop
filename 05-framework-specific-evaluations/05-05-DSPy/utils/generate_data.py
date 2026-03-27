"""Generate synthetic meeting transcripts with expected summaries via Bedrock.

Produces 15 meeting examples across different meeting types and saves to data/meetings.json.
"""

import json
import os

import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "meetings.json")

MEETING_TYPES = [
    "daily standup — engineering team discussing blockers and progress",
    "sprint planning — team estimating and assigning user stories",
    "design review — discussing API design for a new microservice",
    "retrospective — team reflecting on what went well and what to improve",
    "product roadmap — PM presenting Q2 priorities to stakeholders",
    "incident postmortem — reviewing a production outage root cause",
    "hiring debrief — panel discussing a candidate after interviews",
    "budget review — finance and engineering discussing cloud costs",
    "customer feedback review — support team sharing top user complaints",
    "architecture decision — debating monolith vs microservices migration",
    "onboarding sync — manager and new hire planning first 30 days",
    "security review — team discussing vulnerability scan results",
    "release planning — coordinating a major version release timeline",
    "cross-team sync — frontend and backend teams aligning on integration",
    "quarterly business review — leadership reviewing OKR progress",
]


def generate_meeting(client, meeting_type: str, index: int) -> dict:
    """Generate one synthetic meeting transcript with expected outputs."""
    prompt = f"""Generate a realistic meeting transcript and its expected summary outputs.

Meeting type: {meeting_type}

Requirements:
- Transcript should be 150-250 words with 3-5 speakers using first names only
- Include realistic dialogue with filler words and natural speech patterns
- The meeting should contain 2-3 clear decisions and 2-4 action items

Return a JSON object with exactly these fields:
- "transcript": the full meeting transcript
- "expected_decisions": list of 2-3 key decisions made
- "expected_action_items": list of 2-4 action items with owners
- "expected_summary": a 2-3 sentence summary of the meeting

Return ONLY valid JSON, no markdown fences."""

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )

    body = json.loads(response["body"].read())
    text = body["content"][0]["text"].strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove opening fence line
        text = text.rsplit("```", 1)[0]  # remove closing fence
    data = json.loads(text)
    data["id"] = f"meeting_{index:02d}"
    data["meeting_type"] = meeting_type.split("—")[0].strip()
    return data


def main():
    client = boto3.client("bedrock-runtime")
    examples = []

    for i, meeting_type in enumerate(MEETING_TYPES):
        print(f"Generating {i + 1}/{len(MEETING_TYPES)}: {meeting_type.split('—')[0].strip()}")
        try:
            example = generate_meeting(client, meeting_type, i)
            examples.append(example)
        except Exception as e:
            print(f"  Failed: {e}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nGenerated {len(examples)} examples → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
