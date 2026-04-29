---
name: RAG Evaluation
description: Use when learners need to evaluate retrieval-augmented generation systems using retrieval quality metrics, faithfulness scoring, relevance scoring, and multimodal document handling.
---

In this skill, you build a complete RAG evaluation pipeline that measures both retrieval quality and end-to-end answer generation. You start by implementing information retrieval metrics like precision@k, recall@k, and NDCG@k to assess whether your vector search returns the right documents. You then construct an LLM-as-a-Judge evaluation rubric that scores generated answers on faithfulness, relevance, and context utilization. Finally, you extend your evaluation to multimodal RAG systems that handle text, images, and audio using unified embedding spaces.

## Prerequisites

- Completion of Module 01 (evaluation fundamentals) and Module 02 (LLM-as-a-Judge patterns)
- Working knowledge of Python, pandas, and numpy
- Familiarity with vector databases and embedding models
- AWS account with Amazon Bedrock access configured

## Learning Objectives

- Implement precision@k, recall@k, and NDCG@k scoring functions to measure retrieval quality against a curated validation dataset
- Construct an LLM-as-a-Judge evaluation rubric that scores RAG responses on context utilization, completeness, conciseness, context relevancy, and clarity
- Compute faithfulness and relevance metrics for end-to-end RAG system outputs using ground truth comparisons
- Evaluate multimodal retrieval systems across text, vision, and audio modalities using unified embeddings and FAISS indices

## Setup

Install required dependencies and initialize clients:

```python
import boto3
from botocore.config import Config
import chromadb
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
session = boto3.Session()
region = session.region_name or 'us-east-1'

bedrock = session.client(
    service_name="bedrock-runtime",
    region_name=region,
    config=config
)

chroma_client = chromadb.PersistentClient(path="./data/chroma")
```

Prepare a validation dataset with queries and their known relevant document IDs. Each entry maps a `query_text` to a list of `relevant_doc_ids` representing ground truth:

```python
eval_df = pd.read_csv('./data/eval-datasets/1_embeddings_validation.csv')
eval_df = eval_df.rename(columns=lambda x: x.strip())
eval_df = eval_df.dropna(how='all')
eval_df['relevant_doc_ids'] = eval_df['relevant_doc_ids'].astype(str)
```

### Section 1: Retrieval Quality Metrics

Retrieval is the foundation of any RAG system. If your retrieval step returns irrelevant documents, no amount of prompt engineering will save the final answer. Unlike LLMs or embedding models, there are no public leaderboards for your specific RAG configuration — your chunk size, embedding model, and retrieval strategy create a unique system that must be evaluated empirically.

The core question is: given a user query, does the retrieval step surface the documents that actually contain the answer? Precision@k tells you what fraction of the top-k results are relevant. Recall@k tells you what fraction of all relevant documents appear in the top-k. NDCG@k accounts for ranking position — relevant documents ranked higher contribute more to the score.

These metrics together paint a complete picture. High recall with low precision means you're retrieving too much noise. High precision with low recall means you're missing relevant content. NDCG reveals whether your ranking puts the best documents first.

**Build: Implement the IRMetricsCalculator**

Create a class that computes retrieval metrics given relevant and retrieved document lists:

```python
class IRMetricsCalculator:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def precision_at_k(relevant, retrieved, k):
        retrieved_k = retrieved[:k]
        return len(set(relevant) & set(retrieved_k)) / k if k > 0 else 0

    @staticmethod
    def recall_at_k(relevant, retrieved, k):
        retrieved_k = retrieved[:k]
        return len(set(relevant) & set(retrieved_k)) / len(relevant) if len(relevant) > 0 else 0

    @staticmethod
    def dcg_at_k(relevant, retrieved, k):
        retrieved_k = retrieved[:k]
        dcg = 0
        for i, item in enumerate(retrieved_k):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)
        return dcg

    @staticmethod
    def ndcg_at_k(relevant, retrieved, k):
        dcg = IRMetricsCalculator.dcg_at_k(relevant, retrieved, k)
        idcg = IRMetricsCalculator.dcg_at_k(relevant, relevant, k)
        return dcg / idcg if idcg > 0 else 0

    def calculate_metrics(self, k_values=[1, 3, 5, 10]):
        for k in k_values:
            self.df[f'precision@{k}'] = self.df.apply(lambda row: self.precision_at_k(
                json.loads(row['relevant_doc_ids']),
                json.loads(row['retrieved_doc_ids']), k), axis=1)
            self.df[f'recall@{k}'] = self.df.apply(lambda row: self.recall_at_k(
                json.loads(row['relevant_doc_ids']),
                json.loads(row['retrieved_doc_ids']), k), axis=1)
            self.df[f'ndcg@{k}'] = self.df.apply(lambda row: self.ndcg_at_k(
                json.loads(row['relevant_doc_ids']),
                json.loads(row['retrieved_doc_ids']), k), axis=1)
        return self.df
```

Run the retrieval task against your validation dataset and compute aggregate metrics:

```python
runner = RetrievalTaskRunner(eval_df, retrieval_task)
results_df = runner.run()

# Summarize results
summary = {
    'Mean Precision@5': results_df['precision@5'].mean(),
    'Mean Recall@5': results_df['recall@5'].mean(),
    'Mean NDCG@5': results_df['ndcg@5'].mean(),
    '% Queries with Relevant Doc in Top 5': (results_df['precision@5'] > 0).mean() * 100
}
print(pd.DataFrame(summary.items(), columns=['Metric', 'Value']))
```

### Section 2: Faithfulness Scoring

Once retrieval delivers relevant context, the next failure mode is the generation step producing answers that aren't grounded in that context. Faithfulness measures whether the generated answer uses only information present in the retrieved passages — without hallucinating facts or introducing external knowledge.

This matters because a RAG system that hallucinates defeats its own purpose. Users trust RAG answers because they believe the system is citing real sources. An unfaithful answer erodes that trust and can cause real harm in domains like legal, medical, or financial applications.

The LLM-as-a-Judge approach works well here: you provide the context, the question, and the generated answer to a judge model, then ask it to verify each claim against the source material. A binary score per criterion (0 or 1) keeps evaluation crisp and actionable.

**Build: Create the evaluation rubric and client**

Define the evaluation rubric that scores faithfulness alongside other quality criteria:

```python
RUBRIC_SYSTEM_PROMPT = """You are an expert judge evaluating RAG applications.
Evaluation Criteria (Score either 0 or 1 for each, total score is the sum):
1. Context Utilization: Does the answer use only information from the context?
2. Completeness: Does the answer address all key elements of the question?
3. Conciseness: Does the answer avoid unnecessary redundancy?
4. Context Relevancy: Is the context sufficient to answer like the ground truth?
5. Clarity: Is the answer easy to understand and follow?"""

RUBRIC_USER_PROMPT = """Evaluate the following RAG response:

Question: {query_text}
Ground Truth: {ground_truth}
Generated answer: {llm_response}
Context: {context}

For each criterion, assign 0 or 1. Present evaluation in <thinking></thinking> tags.
Include total score in <score></score> tags."""
```

Implement the evaluation client that extracts scores:

```python
class EvaluationClient:
    def __init__(self, bedrock_client, user_prompt, system_prompt, model_id, hyper_params):
        self.client = bedrock_client
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.model_id = model_id
        self.hyper_params = hyper_params

    def extract_score_and_thinking(self, llm_output):
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', llm_output, re.DOTALL)
        score_match = re.search(r'<score>(.*?)</score>', llm_output, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else "No thinking found"
        score = float(score_match.group(1)) if score_match else None
        return score, thinking

    def evaluate(self, df):
        df = df.copy()
        # Build prompts and call model for each row
        for _, row in df.iterrows():
            prompt = self.user_prompt.format(**row)
            # ... call bedrock and extract scores
        return df
```

### Section 3: Relevance Scoring

Relevance scoring closes the loop between retrieval and generation. Where faithfulness asks "did the answer stay within the context?", relevance asks "did the context actually help answer the question?" A system can be perfectly faithful to irrelevant context and still produce a useless answer.

End-to-end relevance evaluation requires ground truth answers. You compare the generated response against a known-good answer to determine whether the retrieval + generation pipeline produced something useful. This catches cases where retrieval returns tangentially related documents that don't contain the actual answer.

The combination of retrieval metrics (Section 1), faithfulness (Section 2), and relevance scoring gives you a complete diagnostic view. Low retrieval scores point to embedding or chunking problems. Low faithfulness points to prompt or model problems. Low relevance despite good retrieval points to context window or prompt formatting issues.

**Build: Run end-to-end evaluation with ground truth**

Set up the RAG client and run the full pipeline:

```python
RAG_SYSTEM_PROMPT = """You are an AI assistant specialized in RAG.
Use only information from the provided context. Be concise and accurate.
Place your response in <response></response> tags."""

RAG_USER_PROMPT = """Answer using only the provided context:
<query>{query_text}</query>
<context>{context}</context>"""

HAIKU_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

# Generate RAG responses
rag_client = RAGClient(
    bedrock, RAG_USER_PROMPT, RAG_SYSTEM_PROMPT, HAIKU_ID,
    {"temperature": 0.5, "maxTokens": 2096}, retrieval_task
)
rag_df = rag_client.process(eval_df)

# Evaluate with LLM-as-a-Judge
eval_client = EvaluationClient(
    bedrock, RUBRIC_USER_PROMPT, RUBRIC_SYSTEM_PROMPT, HAIKU_ID,
    {"temperature": 0.7, "maxTokens": 4096}
)
results_df = eval_client.evaluate(rag_df)

# Summarize
mean_score = results_df['grade'].astype(float).mean()
print(f"Mean E2E Score: {mean_score:.2f} / 5.0")
print(f"Score Distribution:\n{results_df['grade'].value_counts().sort_index()}")
```

### Section 4: Multimodal RAG Evaluation

Real-world documents contain more than text. PDFs have diagrams, knowledge bases include images, and media archives contain audio. A text-only retrieval system misses visual and auditory signals that could identify the correct content. Multimodal evaluation extends your metrics to measure whether the system leverages all available modalities effectively.

The key insight from multimodal evaluation is that different modalities excel at different tasks. Text embeddings capture conceptual meaning but miss visual context. Vision embeddings identify scenes and objects but can't parse dialogue. Audio embeddings capture tone, music, and sound effects. A unified embedding space (like ImageBind) maps all modalities into the same vector space, enabling cross-modal retrieval and comparison.

Evaluating multimodal RAG requires testing each modality independently and in combination. You measure whether adding modalities improves retrieval accuracy, and whether the system correctly identifies which modality provides the strongest signal for each query.

**Build: Evaluate retrieval across modalities**

Create embeddings and FAISS indices for multimodal retrieval:

```python
import torch
import faiss
from sklearn.preprocessing import normalize

# Assume ImageBind model is loaded and embeddings are created
# embeddings = {'text': np.array(...), 'vision': np.array(...), 'audio': np.array(...)}

def create_separate_indices(embeddings):
    """Create FAISS indices for each modality."""
    indices = {}
    normalized = {}
    for modality, emb in embeddings.items():
        norm_emb = normalize(emb, axis=1).astype('float32')
        index = faiss.IndexFlatIP(norm_emb.shape[1])
        index.add(norm_emb)
        indices[modality] = index
        normalized[modality] = norm_emb
    return indices, normalized

def create_multimodal_indices(normalized_embeddings):
    """Create combined indices by averaging modality embeddings."""
    combined = {}
    # Full multimodal: average all three
    full = np.mean([
        normalized_embeddings['text'],
        normalized_embeddings['vision'],
        normalized_embeddings['audio']
    ], axis=0)
    full_norm = normalize(full, axis=1).astype('float32')
    index = faiss.IndexFlatIP(full_norm.shape[1])
    index.add(full_norm)
    combined['full_multimodal'] = index
    return combined, {'full_multimodal': full_norm}
```

Compare retrieval strategies and compute metrics per modality:

```python
strategies = ['text_only', 'vision_only', 'audio_only', 'full_multimodal']
results = {}

for strategy in strategies:
    # Run retrieval with the appropriate index
    # Compute precision@1, MRR@5, NDCG@5
    results[strategy] = {
        'precision@1': precision_score,
        'mrr@5': mrr_score,
        'ndcg@5': ndcg_score
    }

comparison_df = pd.DataFrame(results).T
print(comparison_df)
# Identify which modality performs best for each metric
print(f"\nBest strategy: {comparison_df.mean(axis=1).idxmax()}")
```

## Challenges

**Challenge: Build a RAG evaluation dashboard for a new domain**

Select a document corpus of your choice (at least 20 documents). Build a complete evaluation pipeline that:
1. Creates a validation dataset with at least 15 query/relevant-document pairs
2. Implements retrieval metrics and runs at least two experiments with different configurations (e.g., different chunk sizes or embedding models)
3. Implements an LLM-as-a-Judge rubric with at least 4 evaluation criteria
4. Produces a summary comparison showing which configuration performs best

**Assessment criteria:**

- Pipeline runs without errors end-to-end
- Retrieval metrics (precision@k, recall@k, NDCG@k) are correctly computed and aggregated
- LLM-as-a-Judge rubric includes faithfulness and relevance criteria with structured score extraction
- At least two configurations are compared with clear summary metrics
- Results include both retrieval-level and end-to-end evaluation scores
- Learner can explain why one configuration outperforms another and what they would try next

## Wrap-Up

You built a multi-layered RAG evaluation system that measures retrieval quality, answer faithfulness, end-to-end relevance, and multimodal effectiveness. The key insight is that RAG evaluation requires examining each component independently — a failure in retrieval looks different from a failure in generation, and the fix is different too.

These techniques form the foundation for the Module 04 capstone challenge. See `CHALLENGE-capstone.md` for a comprehensive exercise that combines RAG evaluation with other workload-specific evaluation patterns from this module.

**Feedback prompt:** Which metric was most surprising in what it revealed about your system's performance? What would you change first based on your evaluation results?

**Next steps:** Apply these evaluation patterns to your own RAG system, or explore the structured data evaluation skill for complementary techniques.
