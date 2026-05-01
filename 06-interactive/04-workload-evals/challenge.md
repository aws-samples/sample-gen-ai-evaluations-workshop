# Challenge: Workload-Specific Evaluations

Build a RAG evaluation pipeline that measures retrieval quality, answer faithfulness, and end-to-end relevance. Combine these with guardrails validation into a unified evaluation report.

## Prerequisites

- [SKILL: RAG Evaluation](../../module-04-workload-evals/SKILL-rag-evaluation.md)
- [SKILL: Guardrails](../../module-04-workload-evals/SKILL-guardrails.md)
- [CHALLENGE: Capstone](../../module-04-workload-evals/CHALLENGE-capstone.md)
- Source notebooks: [04-03-Basic-RAG](../../04-workload-specific-evaluations/04-03-Basic-RAG/), [04-02-Guardrails](../../04-workload-specific-evaluations/04-02-Guardrails/)

## Exercise 1: Implement Retrieval Quality Metrics

Write functions for precision@k, recall@k, and NDCG@k. Create a small validation dataset (at least 8 queries with known relevant document IDs) and compute aggregate retrieval scores.

**Success criteria:**
- `precision_at_k(relevant, retrieved, k)` returns the fraction of top-k results that are relevant
- `recall_at_k(relevant, retrieved, k)` returns the fraction of all relevant docs found in top-k
- `ndcg_at_k(relevant, retrieved, k)` computes DCG/IDCG with log2 discounting
- Validation dataset has at least 8 query/relevant-doc-ID pairs stored as a list of dicts or DataFrame
- Print a summary table with mean precision@3, recall@3, NDCG@3, and precision@5, recall@5, NDCG@5
- Explain what high recall + low precision would indicate about your retrieval configuration

## Exercise 2: Build an LLM-as-Judge Faithfulness Scorer

Create an evaluation rubric and scoring function that judges whether a RAG-generated answer is faithful to its retrieved context. The rubric should score on at least 4 criteria.

**Success criteria:**
- Rubric prompt includes at least 4 criteria: Context Utilization, Completeness, Conciseness, and Clarity (or equivalents)
- Each criterion is scored 0 or 1 (binary) with the judge providing reasoning in `<thinking>` tags and a total in `<score>` tags
- Scoring function calls Bedrock, parses the structured output, and returns per-criterion scores plus total
- Evaluate at least 5 question/context/answer triples (at least one should be deliberately unfaithful)
- Print results showing which criteria the unfaithful answer fails on

## Exercise 3: End-to-End RAG Evaluation

Wire together retrieval + generation + evaluation into a single pipeline. Given a query, retrieve context, generate an answer, and score it.

**Success criteria:**
- Pipeline function accepts a query, retrieves top-k documents (simulated or real), generates an answer via Bedrock, and scores it with the Exercise 2 rubric
- Run the pipeline on at least 5 queries that have ground-truth answers
- Compare generated answers against ground truth using your faithfulness scorer
- Print a summary: query, retrieval precision@3, faithfulness score (out of 4), and whether the answer matches ground truth
- Identify which pipeline stage (retrieval vs. generation) is the bottleneck for your lowest-scoring queries

## Exercise 4: Unified Evaluation Report with Guardrails

Add a guardrails validation stage to your pipeline and produce a unified report that combines retrieval metrics, faithfulness scores, and guardrails pass/fail results.

**Success criteria:**
- Guardrails stage checks generated answers against at least one policy (e.g., no PII, no harmful content) using Bedrock Guardrails or a custom check function
- Pipeline runs all three stages: retrieval eval → faithfulness eval → guardrails check
- Unified report is a printed Markdown table with columns: Query, Precision@3, Faithfulness Score, Guardrails Pass/Fail, Overall Pass/Fail
- Overall Pass/Fail requires: precision@3 > 0.5 AND faithfulness ≥ 3/4 AND guardrails pass
- Run on at least 5 queries and print the report
- Explain which metric you would prioritize in a production deployment and why

## Tips

- The SKILL doc's `IRMetricsCalculator` class is a reference for Exercise 1 — but implement the functions yourself first
- For Exercise 2, use `temperature=0.1` on the judge model for consistent scoring
- If you don't have a real vector store, simulate retrieval by pre-assigning document IDs to queries
- The capstone challenge doc shows the full pipeline architecture diagram — use it as a reference for Exercise 4
