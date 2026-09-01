"""Information-retrieval scoring for context retrieval.

Finding the right code is an IR problem. Given a question:

  - the agent's trace yields an ORDERED list of files it touched
    (read_file, grep, MCP find_callers, etc.) — the "retrieved" set.
  - the task's qa_pair has a `relevant_files` list — the "relevant" set.

Standard IR metrics apply: precision@k, recall@k, MRR. Path comparison
is normalised (trailing/leading slashes, case-sensitive on POSIX).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Sequence


def _normalise(path: str) -> str:
    return path.strip().lstrip("./").rstrip("/")


def _dedupe_preserve_order(paths: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        np = _normalise(p)
        if not np or np in seen:
            continue
        seen.add(np)
        out.append(np)
    return out


@dataclass
class RetrievalResult:
    agent: str
    task_id: str
    question: str
    retrieved: List[str]   # ordered, deduped
    relevant: List[str]    # gold set, normalised
    precision_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    extras: dict = field(default_factory=dict)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    head = retrieved[:k]
    if not head:
        return 0.0
    rel_set = {_normalise(r) for r in relevant}
    hits = sum(1 for p in head if _normalise(p) in rel_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel_set = {_normalise(r) for r in relevant}
    if not rel_set:
        return 0.0
    head = {_normalise(p) for p in retrieved[:k]}
    hits = len(head & rel_set)
    return hits / len(rel_set)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel_set = {_normalise(r) for r in relevant}
    for i, p in enumerate(retrieved, start=1):
        if _normalise(p) in rel_set:
            return 1.0 / i
    return 0.0


def score_retrieval(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    agent: str,
    task_id: str,
    question: str,
    k_precision: int = 5,
    k_recall: int = 10,
) -> RetrievalResult:
    retrieved = _dedupe_preserve_order(retrieved)
    relevant_norm = [_normalise(r) for r in relevant]
    return RetrievalResult(
        agent=agent,
        task_id=task_id,
        question=question,
        retrieved=retrieved,
        relevant=relevant_norm,
        precision_at_5=precision_at_k(retrieved, relevant_norm, k_precision),
        recall_at_10=recall_at_k(retrieved, relevant_norm, k_recall),
        mrr=mrr(retrieved, relevant_norm),
    )


def to_dict(r: RetrievalResult) -> dict:
    return asdict(r)
