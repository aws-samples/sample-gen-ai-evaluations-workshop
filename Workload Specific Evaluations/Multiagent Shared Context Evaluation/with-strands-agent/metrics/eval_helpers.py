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


# ---------------------------------------------------------------------------
# Visualization helpers (matplotlib)
# ---------------------------------------------------------------------------

def plot_context_metrics_radar(collector, session_label: str = "Session"):
    """Radar chart of average LLM-judge scores across all agents."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["context_freshness", "handoff_completeness", "context_utilization",
               "write_accuracy", "redundant_context"]
    labels = ["Freshness", "Handoff\nComplete", "Context\nUtil.", "Write\nAccuracy", "Low\nRedundancy"]

    scores = []
    for key in metrics:
        vals = []
        for t in collector.turns:
            for r in t.agent_calls:
                s = r.judge_scores.get(key, {}).get("score")
                if s is not None:
                    vals.append(s)
        scores.append(sum(vals) / len(vals) if vals else 0)

    # Add state consistency
    sc_vals = [t.state_consistency.get("score", 0) for t in collector.turns if t.state_consistency.get("score")]
    scores.append(sum(sc_vals) / len(sc_vals) if sc_vals else 0)
    labels.append("State\nConsist.")
    metrics.append("state_consistency")

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, scores_plot, alpha=0.25, color="#2196F3")
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="#2196F3")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=8)
    ax.set_title(f"Context Quality — {session_label}", size=14, pad=20)
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_latency_breakdown(collector, session_label: str = "Session"):
    """Stacked bar chart: memory read / write / reasoning time per agent call."""
    import matplotlib.pyplot as plt
    import numpy as np

    agents, reads, writes, reasoning = [], [], [], []
    for turn in collector.turns:
        for rec in turn.agent_calls:
            label = f"T{turn.turn_number}:{rec.agent_name}"
            agents.append(label)
            reads.append(rec.memory_read_latency)
            writes.append(rec.memory_write_latency)
            coord = rec.memory_read_latency + rec.memory_write_latency
            reasoning.append(max(rec.total_agent_latency - coord, 0))

    x = np.arange(len(agents))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.2), 5))
    ax.bar(x, reads, width, label="Memory Read", color="#4CAF50")
    ax.bar(x, writes, width, bottom=reads, label="Memory Write", color="#FF9800")
    bottoms = [r + w for r, w in zip(reads, writes)]
    ax.bar(x, reasoning, width, bottom=bottoms, label="Reasoning", color="#2196F3")

    ax.set_ylabel("Seconds")
    ax.set_title(f"Latency Breakdown — {session_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_session_comparison(collector_a, collector_b,
                            label_a: str = "Session 1", label_b: str = "Session 2"):
    """Side-by-side bar chart comparing average scores between two sessions."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["context_freshness", "handoff_completeness", "context_utilization",
               "write_accuracy", "redundant_context"]
    labels = ["Freshness", "Handoff", "Ctx Util", "Write Acc", "Low Redund"]

    def avg(collector, key):
        vals = []
        for t in collector.turns:
            for r in t.agent_calls:
                s = r.judge_scores.get(key, {}).get("score")
                if s is not None:
                    vals.append(s)
        return sum(vals) / len(vals) if vals else 0

    scores_a = [avg(collector_a, k) for k in metrics]
    scores_b = [avg(collector_b, k) for k in metrics]

    # Add state consistency
    def avg_sc(collector):
        vals = [t.state_consistency.get("score", 0) for t in collector.turns if t.state_consistency.get("score")]
        return sum(vals) / len(vals) if vals else 0

    scores_a.append(avg_sc(collector_a))
    scores_b.append(avg_sc(collector_b))
    labels.append("Consistency")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(x - width / 2, scores_a, width, label=label_a, color="#2196F3")
    bars_b = ax.bar(x + width / 2, scores_b, width, label=label_b, color="#FF5722")

    ax.set_ylabel("Score (1-5)")
    ax.set_title("Session Comparison — Context Quality Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5)
    ax.legend()

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def plot_coordination_overhead(collector, session_label: str = "Session"):
    """Bar chart showing input vs output token usage per agent call."""
    import matplotlib.pyplot as plt
    import numpy as np

    agents, input_toks, output_toks = [], [], []

    for turn in collector.turns:
        for rec in turn.agent_calls:
            label = f"T{turn.turn_number}:{rec.agent_name}"
            agents.append(label)
            input_toks.append(rec.reasoning_input_tokens)
            output_toks.append(rec.reasoning_output_tokens)

    if not agents:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No token data recorded", ha="center", va="center", fontsize=14)
        ax.set_title(f"Token Usage — {session_label}")
        return fig

    x = np.arange(len(agents))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.2), 5))
    ax.bar(x, input_toks, width, label="Input Tokens", color="#2196F3")
    ax.bar(x, output_toks, width, bottom=input_toks, label="Output Tokens", color="#FF9800")

    ax.set_ylabel("Tokens")
    ax.set_title(f"Token Usage per Agent Call — {session_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_c2_heatmap(embeddings: dict, label: str = ""):
    """Heatmap of pairwise cosine similarities between agent responses."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(embeddings.keys())
    n = len(names)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = cosine_similarity(embeddings[names[i]], embeddings[names[j]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    color="black" if matrix[i][j] > 0.5 else "white", fontsize=11)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    title = f"C2 Alignment Heatmap — {label}" if label else "C2 Alignment Heatmap"
    ax.set_title(title)
    plt.tight_layout()
    return fig
