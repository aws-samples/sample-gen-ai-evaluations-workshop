"""
Shared evaluation helpers used by both notebooks.

Functions only — no classes. Keeps the notebooks focused on
orchestration logic while reusing the common bits.
"""

import json
import math
import boto3

from model_config import EMBEDDING_MODEL_ID


# ---------------------------------------------------------------------------
# Shared memory formatting
# ---------------------------------------------------------------------------

def format_memory(memory: list) -> str:
    """Format a shared memory list as a readable string for agent prompts.

    Each entry is a dict with 'agent', 'content' (and optionally 'role').
    Output looks like:
        [flight] Assistant: Found Delta DL123, $650...
        [hotel]  Assistant: Marriott Marquis, $220/night...
    """
    if not memory:
        return ""
    parts = []
    for entry in memory:
        agent = entry.get("agent", "unknown")
        content = entry.get("content", "")
        role = entry.get("role", "")
        prefix = f"[{agent}]"
        if role:
            prefix += f" {role.title()}:"
        parts.append(f"{prefix} {content}")
    return "\n".join(parts)


def print_memory(memory: list, preview_chars: int = 200):
    """Print shared memory contents for inspection in notebooks."""
    print(f"Memory has {len(memory)} entries:\n")
    for i, entry in enumerate(memory):
        role = entry.get("role", "-")
        content = entry.get("content", "")
        print(f"  [{i}] agent={entry.get('agent', '?')}, role={role}")
        print(f"      {content[:preview_chars]}...\n")


# ---------------------------------------------------------------------------
# Embeddings (for C2 alignment in peer-to-peer)
# ---------------------------------------------------------------------------

def get_titan_embedding(text: str, region: str = "us-west-2") -> list:
    """Get embedding vector from Bedrock Titan Embeddings V2."""
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({"inputText": text[:8000]})  # Titan limit
    resp = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID, body=body,
        contentType="application/json", accept="application/json")
    return json.loads(resp["body"].read())["embedding"]


def cosine_similarity(a: list, b: list) -> float:
    """Pure-Python cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def compute_embeddings(responses: dict, region: str = "us-west-2") -> dict:
    """Compute Titan embeddings for each response. Returns {name: vector}."""
    embeddings = {}
    for name, text in responses.items():
        if text:
            embeddings[name] = get_titan_embedding(text, region=region)
            print(f"Embedded {name}: {len(embeddings[name])} dims")
    return embeddings


def c2_alignment_report(embeddings: dict, label: str = "") -> str:
    """Render pairwise cosine similarities as a markdown report."""
    names = list(embeddings.keys())
    title = f"### C2 Alignment ({label})" if label else "### C2 Alignment"
    lines = [title, ""]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = cosine_similarity(embeddings[names[i]], embeddings[names[j]])
            lines.append(f"- {names[i]} ↔ {names[j]}: **{sim:.3f}**")
    return "\n".join(lines)
