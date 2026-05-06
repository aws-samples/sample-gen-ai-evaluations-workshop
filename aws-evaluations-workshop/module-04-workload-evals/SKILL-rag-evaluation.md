---
name: RAG Evaluation
description: Teach learners to evaluate RAG systems using retrieval quality metrics, faithfulness scoring, relevance assessment, and multimodal document handling
---

# RAG System Evaluation

Evaluate Retrieval-Augmented Generation systems end-to-end — from measuring whether the right documents are retrieved, to scoring whether generated answers are faithful to context, to extending evaluation across text, vision, and audio modalities.

## Prerequisites

- Completed Module 01 (operational metrics) and Module 02 (quality metrics with LLM-as-a-Judge)
- Working AWS credentials with Bedrock access (Titan Embed Text V2, Claude Haiku)
- Python environment with `chromadb`, `pandas`, `numpy`, `boto3`
- Familiarity with embeddings and vector similarity concepts

## Learning Objectives

By the end of this module, learners will be able to:

1. **Calculate** precision@k, recall@k, and NDCG@k for a retrieval pipeline against a labeled validation set
2. **Implement** an LLM-as-a-Judge rubric that scores RAG responses on faithfulness, completeness, and context relevancy
3. **Design** a multimodal retrieval evaluation that compares text-only, vision-only, and combined strategies using shared embedding spaces
4. **Interpret** aggregate retrieval metrics (MAP, MRR) to diagnose whether poor RAG output stems from retrieval failure or generation failure

## Setup

```python
import boto3
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

# Initialize Bedrock client
config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
session = boto3.Session()
region = session.region_name or 'us-east-1'

bedrock = session.client(
    service_name="bedrock-runtime",
    region_name=region,
    config=config
)

# Embedding model for retrieval experiments
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'
```

## Section 1: Retrieval Quality Metrics

**Concept:** A RAG system can fail at two stages: retrieval (wrong documents) or generation (wrong answer from right documents). Retrieval metrics isolate the first failure mode. Precision@k asks "of the top-k results, how many are relevant?" Recall@k asks "of all relevant documents, how many appeared in top-k?" NDCG@k penalizes relevant documents that appear lower in the ranked list.

Without these metrics, you cannot distinguish "the model hallucinated" from "the model never saw the right context."

**Build:**

```python
class IRMetricsCalculator:
    @staticmethod
    def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        return len(set(relevant) & set(retrieved_k)) / k if k > 0 else 0

    @staticmethod
    def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        return len(set(relevant) & set(retrieved_k)) / len(relevant) if relevant else 0

    @staticmethod
    def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(retrieved_k) if item in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0

    def calculate_metrics(self, df: pd.DataFrame, k_values=[1, 3, 5]) -> pd.DataFrame:
        for k in k_values:
            df[f'precision@{k}'] = df.apply(
                lambda row: self.precision_at_k(row['relevant_ids'], row['retrieved_ids'], k), axis=1)
            df[f'recall@{k}'] = df.apply(
                lambda row: self.recall_at_k(row['relevant_ids'], row['retrieved_ids'], k), axis=1)
            df[f'ndcg@{k}'] = df.apply(
                lambda row: self.ndcg_at_k(row['relevant_ids'], row['retrieved_ids'], k), axis=1)
        return df
```

**Key insight:** recall@5 is often the most important single metric — it tells you whether the answer *exists* in the context window the model will see.

## Section 2: Faithfulness and End-to-End Scoring

**Concept:** Once retrieval delivers context, the generation model must use it faithfully. End-to-end evaluation uses LLM-as-a-Judge with a structured rubric to score: (1) context utilization — does the answer use only provided context? (2) completeness — are all question aspects addressed? (3) faithfulness — no fabricated details? (4) context relevancy — was the retrieved context sufficient? (5) clarity — is the answer understandable?

Scoring each criterion as 0 or 1 produces a 0–5 composite that pinpoints *which* aspect failed.

**Build:**

```python
RUBRIC_SYSTEM_PROMPT = """You are an expert judge evaluating RAG applications.
Score each criterion 0 or 1:
1. Context Utilization: Uses only provided context, no external details?
2. Completeness: Addresses all key elements from available context?
3. Conciseness: Efficiently worded without redundancy?
4. Context Relevancy: Retrieved context sufficient for a correct answer?
5. Clarity: Easy to understand and follow?
Present evaluation in <thinking> tags, total score in <score> tags."""

RUBRIC_USER_PROMPT = """Evaluate this RAG response:
Question: {query_text}
Ground Truth: {ground_truth}
Generated Answer: {llm_response}
Context Used: {context}

Score each criterion 0 or 1. Total in <score></score> tags."""

def evaluate_rag_response(query, ground_truth, llm_response, context) -> int:
    """Returns 0-5 composite score from LLM judge."""
    messages = [{"role": "user", "content": [{"text": RUBRIC_USER_PROMPT.format(
        query_text=query, ground_truth=ground_truth,
        llm_response=llm_response, context=context
    )}]}]
    response = bedrock.converse(
        modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        messages=messages,
        system=[{"text": RUBRIC_SYSTEM_PROMPT}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 1024}
    )
    text = response['output']['message']['content'][0]['text']
    import re
    match = re.search(r'<score>(\d+)</score>', text)
    return int(match.group(1)) if match else 0
```

## Section 3: Aggregate Diagnostics — MAP and MRR

**Concept:** Individual query metrics are noisy. Aggregate metrics reveal systemic patterns. Mean Average Precision (MAP) averages precision across all relevant documents for each query, then averages across queries — it rewards systems that rank *all* relevant docs highly. Mean Reciprocal Rank (MRR) measures how quickly the *first* relevant result appears — critical for single-answer use cases.

Together: high MRR + low MAP = "finds one good doc but misses others." Low MRR + high recall = "relevant docs exist but aren't ranked first."

**Build:**

```python
def mean_average_precision(df: pd.DataFrame) -> float:
    """MAP across all queries. Expects 'relevant_ids' and 'retrieved_ids' columns."""
    aps = []
    for _, row in df.iterrows():
        relevant = set(row['relevant_ids'])
        retrieved = row['retrieved_ids']
        hits, running_sum = 0, 0.0
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                hits += 1
                running_sum += hits / i
        aps.append(running_sum / len(relevant) if relevant else 0)
    return np.mean(aps)

def mean_reciprocal_rank(df: pd.DataFrame) -> float:
    """MRR — how quickly does the first relevant doc appear?"""
    rrs = []
    for _, row in df.iterrows():
        relevant = set(row['relevant_ids'])
        for i, doc in enumerate(row['retrieved_ids'], 1):
            if doc in relevant:
                rrs.append(1.0 / i)
                break
        else:
            rrs.append(0.0)
    return np.mean(rrs)
```

## Section 4: Multimodal Retrieval Evaluation

**Concept:** Text-only retrieval fails when the answer lives in visual or audio content. Multimodal evaluation compares retrieval strategies (text-only, vision-only, audio-only, combined) using a shared embedding space. Models like ImageBind create unified vectors where semantically similar content across modalities clusters together — enabling cross-modal retrieval and fair comparison.

The evaluation pattern: run the same queries against each strategy, compute the same metrics (precision, recall, NDCG), then compare. The strategy that wins depends on the content type — text excels for factual lookup, vision for scene identification, audio for speaker/tone matching.

**Build:**

```python
import faiss
from sklearn.preprocessing import normalize

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index from normalized embeddings for cosine similarity search."""
    normalized = normalize(embeddings, axis=1).astype('float32')
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index

def compare_retrieval_strategies(
    query_embedding: np.ndarray,
    indices: Dict[str, faiss.IndexFlatIP],
    ground_truth_ids: List[int],
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """Compare precision/recall across modality-specific indices."""
    results = {}
    query_norm = normalize(query_embedding.reshape(1, -1)).astype('float32')
    for strategy_name, index in indices.items():
        distances, retrieved_ids = index.search(query_norm, k)
        retrieved = retrieved_ids[0].tolist()
        results[strategy_name] = {
            'precision@k': len(set(ground_truth_ids) & set(retrieved)) / k,
            'recall@k': len(set(ground_truth_ids) & set(retrieved)) / len(ground_truth_ids),
            'top_similarity': float(distances[0][0])
        }
    return results
```

**Key insight:** If text-only recall is low but multimodal recall is high, your knowledge base contains information that text embeddings cannot capture — visual diagrams, audio explanations, or cross-modal relationships.

## Challenges

1. **Retrieval diagnosis:** Given a RAG system with MAP=0.3 and MRR=0.8, explain what this means about the system's behavior and propose two specific changes to improve MAP without hurting MRR.

2. **Rubric design:** Create a custom evaluation rubric for a domain-specific RAG system (e.g., medical, legal, or code documentation) that adds domain-appropriate criteria beyond the generic 5-point rubric.

3. **Multimodal comparison:** Run the same 10 queries against text-only and multimodal retrieval strategies. Identify which query types benefit most from multimodal and explain why.

## Wrap-Up

You can now evaluate RAG systems at every stage — retrieval quality, generation faithfulness, and cross-modal effectiveness. The key diagnostic pattern: start with retrieval metrics to confirm the right documents are found, then use LLM-as-a-Judge to assess generation quality, then compare modality strategies to find coverage gaps.

This module does **NOT** cover: fine-tuning embedding models, reranker evaluation, chunk size optimization experiments, or production monitoring dashboards.

**Next:** Apply these evaluation patterns in the Module 04 capstone challenge — see `CHALLENGE-capstone.md` for a multi-component RAG evaluation task that combines retrieval scoring, faithfulness assessment, and strategy comparison into a single diagnostic workflow.
