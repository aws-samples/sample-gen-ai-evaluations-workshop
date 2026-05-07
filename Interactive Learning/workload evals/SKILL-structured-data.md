---
name: Structured Data Extraction Evaluation
description: Evaluate structured data extraction accuracy using field-level classification, precision, recall, and F1-score metrics for document processing systems
---

# Evaluating Structured Data Extraction

Automated document processing systems extract fields like invoice numbers, vendor names, and totals from unstructured documents. But how accurate are those extractions? This skill teaches you to build a field-level evaluation framework that classifies predictions, computes precision/recall/F1, and surfaces per-field weaknesses in extraction pipelines.

## Prerequisites

- Python 3.10+
- `json` and `collections` standard library modules
- Familiarity with precision/recall concepts (helpful but not required)

## Learning Objectives

By the end of this skill, you will:

1. Classify extraction predictions into five categories (TP, TN, FA, FD, FN) by comparing against ground truth
2. Compute precision, recall, F1-score, and accuracy at both field and document levels
3. Visualize per-field metrics to identify extraction weaknesses
4. Interpret metric trade-offs to guide extraction system improvements

## Setup

```python
import json
import collections

# Ground truth: what the documents actually contain
ground_truth_data = [
    {'DocumentID': 1, 'invoice_number': 'INV-001', 'vendor': 'Acme Corp', 'total': 1500.00},
    {'DocumentID': 2, 'invoice_number': 'INV-002', 'vendor': '', 'total': 2200.50},
    {'DocumentID': 3, 'invoice_number': '', 'vendor': 'SuperTech', 'total': None},
    {'DocumentID': 4, 'invoice_number': 'INV-004', 'vendor': 'Global Services', 'total': 3750.00},
    {'DocumentID': 5, 'invoice_number': 'INV-005', 'vendor': 'Mega Industries', 'total': 1200.75}
]

# Predictions: what our extraction system produced
prediction_data = [
    {'DocumentID': 1, 'invoice_number': 'INV-001', 'vendor': 'Acme Corp', 'total': 1500.00},
    {'DocumentID': 2, 'invoice_number': 'INV-002', 'vendor': '', 'total': 2250.00},
    {'DocumentID': 3, 'invoice_number': 'INV-003', 'vendor': '', 'total': None},
    {'DocumentID': 4, 'invoice_number': '', 'vendor': 'Global Svcs', 'total': 3750.00},
    {'DocumentID': 5, 'invoice_number': '', 'vendor': '', 'total': None}
]
```

## Section 1: Classification Framework

**Concept:** Before computing metrics, each predicted field must be classified against ground truth. Five categories capture every possible outcome in structured extraction:

| Category | Condition | Example |
|----------|-----------|---------|
| True Positive (TP) | Both have values, values match | Predicted "INV-001", truth is "INV-001" |
| True Negative (TN) | Both empty | Predicted empty, truth is empty |
| False Alarm (FA) | Prediction has value, truth is empty | Predicted "INV-003", truth is empty |
| False Discovery (FD) | Both have values, values differ | Predicted "Global Svcs", truth is "Global Services" |
| False Negative (FN) | Prediction empty, truth has value | Predicted empty, truth is "INV-005" |

The distinction between FA and FD matters: FA means the system hallucinated a field that shouldn't exist, while FD means it found the right field but extracted the wrong value.

**Build:**

```python
def compare_and_classify(ground_truth_data, prediction_data):
    fields = set().union(*(set(doc.keys()) for doc in ground_truth_data + prediction_data)) - {'DocumentID'}
    results = {field: {'TP': 0, 'TN': 0, 'FA': 0, 'FD': 0, 'FN': 0} for field in fields}

    def is_empty(value):
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        return False

    def values_match(val1, val2):
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) < 0.01
        return val1 == val2

    for gt_doc, pred_doc in zip(ground_truth_data, prediction_data):
        assert gt_doc['DocumentID'] == pred_doc['DocumentID'], "Document IDs don't match"
        for field in fields:
            gt_value = gt_doc.get(field, None)
            pred_value = pred_doc.get(field, None)
            gt_empty = is_empty(gt_value)
            pred_empty = is_empty(pred_value)

            if not gt_empty and not pred_empty:
                if values_match(gt_value, pred_value):
                    results[field]['TP'] += 1
                else:
                    results[field]['FD'] += 1
            elif gt_empty and pred_empty:
                results[field]['TN'] += 1
            elif gt_empty and not pred_empty:
                results[field]['FA'] += 1
            elif not gt_empty and pred_empty:
                results[field]['FN'] += 1

    return results

evaluation_results = compare_and_classify(ground_truth_data, prediction_data)
print(json.dumps(evaluation_results, indent=4))
```

## Section 2: Computing Evaluation Metrics

**Concept:** Raw classification counts become actionable through four metrics, each answering a different question:

- **Precision** = TP / (TP + FA + FD) — "When the system extracts a value, how often is it correct?"
- **Recall** = TP / (TP + FN + FD) — "Of all values that should be extracted, how many did we get right?"
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — balanced single metric
- **Accuracy** = (TP + TN) / (TP + TN + FA + FD + FN) — overall correctness including empty fields

Note that FD (wrong value) penalizes both precision AND recall. This is intentional: extracting the wrong value is worse than extracting nothing, because downstream systems may act on incorrect data.

**Build:**

```python
def calculate_metrics(evaluation_results):
    def _safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0.0

    def _metrics_from_counts(counts):
        TP, TN = counts.get('TP', 0), counts.get('TN', 0)
        FA, FD, FN = counts.get('FA', 0), counts.get('FD', 0), counts.get('FN', 0)
        precision = _safe_divide(TP, TP + FA + FD)
        recall = _safe_divide(TP, TP + FN + FD)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        accuracy = _safe_divide(TP + TN, TP + TN + FA + FD + FN)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy, 'counts': counts}

    metrics = {field: _metrics_from_counts(counts) for field, counts in evaluation_results.items()}

    overall_counts = collections.Counter()
    for counts in evaluation_results.values():
        overall_counts.update(counts)
    metrics['overall'] = _metrics_from_counts(dict(overall_counts))

    return metrics

metrics_results = calculate_metrics(evaluation_results)
print(json.dumps(metrics_results, indent=4))
```

## Section 3: Field-Level Analysis and Visualization

**Concept:** Aggregate metrics hide per-field problems. A system with 80% overall F1 might have 95% on invoice numbers but 40% on vendor names. Field-level breakdown reveals which extraction targets need prompt tuning, additional training data, or post-processing rules.

**Build:**

```python
def visualize_metrics(metrics_results):
    def bar(percentage, width=20):
        filled = int(percentage * width / 100)
        return '█' * filled + '░' * (width - filled)

    overall = metrics_results['overall']
    print("\n--- Overall Metrics ---")
    print(f"Precision  {overall['precision']*100:6.2f}%  {bar(overall['precision']*100)}")
    print(f"Recall     {overall['recall']*100:6.2f}%  {bar(overall['recall']*100)}")
    print(f"F1 Score   {overall['f1']*100:6.2f}%  {bar(overall['f1']*100)}")
    print(f"Accuracy   {overall['accuracy']*100:6.2f}%  {bar(overall['accuracy']*100)}")

    print("\n--- Field-Level Breakdown ---")
    print(f"{'Field':<20} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Visual'}")
    print("-" * 70)
    for field, m in metrics_results.items():
        if field == 'overall':
            continue
        f1_pct = m['f1'] * 100
        print(f"{field:<20} {m['precision']*100:6.2f}% {m['recall']*100:6.2f}% {f1_pct:6.2f}% {bar(f1_pct)}")

visualize_metrics(metrics_results)
```

## Section 4: Interpreting Results for System Improvement

**Concept:** Metrics point to specific failure modes that map to different remediation strategies:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Low precision, high recall | System over-extracts (hallucinations) | Tighten extraction prompts, add validation |
| High precision, low recall | System under-extracts (misses) | Broaden extraction scope, add examples |
| High FA on one field | System finds values where none exist | Add "field not present" as valid output |
| High FD on one field | Partial/fuzzy matches | Improve normalization or matching tolerance |

**Build:**

```python
def diagnose_weaknesses(metrics_results, threshold=0.7):
    issues = []
    for field, m in metrics_results.items():
        if field == 'overall':
            continue
        counts = m['counts']
        if m['precision'] < threshold and m['recall'] >= threshold:
            issues.append(f"{field}: Low precision ({m['precision']:.2f}) — over-extracting. FA={counts['FA']}, FD={counts['FD']}")
        elif m['recall'] < threshold and m['precision'] >= threshold:
            issues.append(f"{field}: Low recall ({m['recall']:.2f}) — missing values. FN={counts['FN']}")
        elif m['f1'] < threshold:
            issues.append(f"{field}: Low F1 ({m['f1']:.2f}) — both precision and recall weak")
    return issues

for issue in diagnose_weaknesses(metrics_results):
    print(f"⚠️  {issue}")
```

## Challenges

### Challenge 1: Custom Matching Logic

The current `values_match` function uses exact string matching and numeric tolerance. Extend it to handle:
- Case-insensitive comparison for text fields
- Abbreviation matching (e.g., "Corp" ↔ "Corporation")
- Date format normalization (e.g., "2024-01-15" ↔ "Jan 15, 2024")

**Assessment criteria:**
1. Code runs without errors on the existing dataset
2. Implements at least two matching strategies with field-type awareness
3. Handles edge cases (None values, mixed types, empty strings)
4. Explain how fuzzy matching affects the precision/recall trade-off

### Challenge 2: Multi-Document Trend Analysis

Given a time-series of extraction batches (e.g., weekly runs), build a function that tracks metric drift over time and flags fields whose F1 drops below a configurable threshold.

**Assessment criteria:**
1. Code runs on synthetic multi-batch data
2. Correctly identifies degrading fields across batches
3. Handles missing fields in some batches gracefully
4. Explain what metric drift indicates about the extraction system

### Capstone

For the integrative challenge combining structured data evaluation with other Module 04 skills, see [CHALLENGE-capstone.md](./CHALLENGE-capstone.md).

## Wrap-Up

**Key takeaways:**
- Five classification categories (TP, TN, FA, FD, FN) form the foundation of extraction evaluation
- Precision measures extraction correctness; recall measures completeness; F1 balances both
- Field-level analysis reveals which extraction targets need improvement
- Diagnostic patterns map metric symptoms to specific remediation strategies

**This skill does NOT cover:**
- LLM prompt engineering for extraction (covered in extraction-focused skills)
- Nested/hierarchical document structures (lists, tables within documents)
- Fuzzy matching algorithms (Levenshtein, token-set-ratio)
- Production monitoring and alerting on metric drift
- Bedrock model invocation for extraction (this skill focuses on evaluation, not extraction)

**Next steps:**
- Apply this framework to your own extraction pipeline outputs
- Combine with RAG evaluation metrics (SKILL-rag-evaluation.md) for end-to-end assessment
- Tackle the [Module 04 Capstone](./CHALLENGE-capstone.md) to integrate evaluation across workload types
