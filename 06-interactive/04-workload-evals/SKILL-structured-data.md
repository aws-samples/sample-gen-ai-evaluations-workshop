---
name: Structured Data Extraction Evaluation
description: Evaluate structured data extraction accuracy using field-level classification, precision/recall/F1 metrics, and visual performance analysis for document processing systems.
---

In this skill, you build a complete evaluation pipeline for structured data extraction — the kind used in Intelligent Document Processing (IDP) systems that pull invoice numbers, vendor names, and totals from business documents. You will classify extraction results into five categories (true positive, true negative, false alarm, false discovery, false negative), compute precision/recall/F1/accuracy metrics at both field and document levels, and visualize performance to identify weak spots. By the end, you can measure whether an extraction system is production-ready.

## Prerequisites

- Python 3.10+ with `json` and `collections` standard library modules
- Familiarity with basic Python data structures (lists, dictionaries)
- Understanding of what structured data extraction means (pulling key-value fields from documents)

## Learning Objectives

- Classify extraction predictions into five outcome categories (TP, TN, FA, FD, FN) by comparing against ground truth data
- Compute precision, recall, F1-score, and accuracy metrics from classification counts
- Analyze field-level metric breakdowns to identify which extraction fields underperform
- Visualize evaluation results to communicate system performance to stakeholders

## Setup

Create a working directory and Python file for this skill:

```bash
mkdir -p idp-evaluation && cd idp-evaluation
touch structured_data_eval.py
```

Ensure Python 3.10+ is available:

```bash
python3 --version
```

No external packages are required — this skill uses only the Python standard library (`json`, `collections`).

### Section 1: Preparing Ground Truth and Prediction Data

When evaluating an extraction system, you need paired datasets: what the system *should* have extracted (ground truth) and what it *actually* extracted (predictions). Without this pairing, you cannot measure accuracy. In real IDP pipelines, ground truth comes from human-annotated documents, while predictions come from your model's output.

The structure of this data matters. Each document has an identifier and a set of fields. Fields can contain values, be empty strings, or be null — each combination tells a different story about system behavior. A prediction that finds "INV-003" when ground truth is empty is a fundamentally different error than finding "Global Svcs" when the correct value is "Global Services."

Understanding these five scenarios — correct match, correct empty, false find, wrong value, and missed value — is the foundation for all metrics that follow.

**Build: Create sample extraction data**

Create ground truth and prediction datasets representing five invoice documents with three fields each. This data encodes all five extraction scenarios:

```python
import json

ground_truth_data = [
    {'DocumentID': 1, 'invoice_number': 'INV-001', 'vendor': 'Acme Corp', 'total': 1500.00},
    {'DocumentID': 2, 'invoice_number': 'INV-002', 'vendor': '', 'total': 2200.50},
    {'DocumentID': 3, 'invoice_number': '', 'vendor': 'SuperTech', 'total': None},
    {'DocumentID': 4, 'invoice_number': 'INV-004', 'vendor': 'Global Services', 'total': 3750.00},
    {'DocumentID': 5, 'invoice_number': 'INV-005', 'vendor': 'Mega Industries', 'total': 1200.75}
]

prediction_data = [
    {'DocumentID': 1, 'invoice_number': 'INV-001', 'vendor': 'Acme Corp', 'total': 1500.00},  # All true positives
    {'DocumentID': 2, 'invoice_number': 'INV-002', 'vendor': '', 'total': 2250.00},  # invoice_number: TP, vendor: TN, total: FD
    {'DocumentID': 3, 'invoice_number': 'INV-003', 'vendor': '', 'total': None},  # invoice_number: FA, vendor: FN, total: TN
    {'DocumentID': 4, 'invoice_number': '', 'vendor': 'Global Svcs', 'total': 3750.00},  # invoice_number: FN, vendor: FD, total: TP
    {'DocumentID': 5, 'invoice_number': '', 'vendor': '', 'total': None}  # All false negatives
]

print(json.dumps(ground_truth_data, indent=4))
print(json.dumps(prediction_data, indent=4))
```

Run this and verify you see both datasets printed. Document 1 shows perfect extraction; Document 5 shows complete failure. The documents in between show partial success with different error types.

### Section 2: Classifying Predictions at Field Level

Before computing any metric, you must classify every individual field prediction into one of five categories. This classification step is where domain logic lives — how do you decide if two values "match"? For numbers, you might allow a small tolerance. For strings, you might need exact match or fuzzy matching depending on your use case.

The five categories form a complete partition: every field comparison falls into exactly one bucket. True Positive means both have values and they match. True Negative means both are empty. False Alarm means the system hallucinated a value where none exists. False Discovery means both have values but they differ. False Negative means the system missed a value that exists.

This classification is the most critical step because all downstream metrics are derived from these counts. An error here propagates through your entire evaluation.

**Build: Implement field-level classification**

```python
def compare_and_classify_field_predictions(ground_truth_data, prediction_data):
    """
    Compare ground truth and prediction data at field level and count metrics.
    
    Args:
        ground_truth_data: List of dictionaries containing ground truth values
        prediction_data: List of dictionaries containing prediction values
    
    Returns:
        Dictionary with counts of TP, TN, FA, FD, FN for each field type
    """
    # Get all unique field names as a union from both datasets, excluding DocumentID
    fields = set().union(*(set(doc.keys()) for doc in ground_truth_data + prediction_data)) - {'DocumentID'}
    
    # Initialize counters for each field
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

evaluation_results = compare_and_classify_field_predictions(ground_truth_data, prediction_data)
print("Field-level count:")
print(json.dumps(evaluation_results, indent=4))
```

Verify the output shows: `invoice_number` has 2 TP, 1 FA, 2 FN; `vendor` has 1 TP, 1 TN, 1 FD, 2 FN; `total` has 2 TP, 1 TN, 1 FD, 1 FN.

### Section 3: Computing Precision, Recall, F1, and Accuracy

Raw classification counts tell you *what* happened, but metrics tell you *how well* the system performs. Precision answers "when the system extracts a value, how often is it correct?" Recall answers "of all values that should be extracted, how many did the system find?" F1 balances both into a single number for model comparison.

These metrics behave differently depending on your use case. A financial document system might prioritize precision (never extract a wrong amount) while a discovery system might prioritize recall (never miss a relevant document). The F1 score is useful when you need a single number to compare models, but always examine precision and recall separately to understand *why* a model underperforms.

Accuracy includes true negatives, making it valuable when correctly identifying empty fields matters — for example, in legal documents where asserting a field is absent has consequences.

**Build: Implement metrics calculation**

```python
import collections

def calculate_metrics(evaluation_results):
    """
    Calculates precision, recall, F1-score, and accuracy from evaluation results
    for individual fields and for an overall summary.
    """
    def _safe_divide(numerator, denominator):
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return 0.0

    def _calculate_metrics_from_counts(counts):
        TP = counts.get('TP', 0)
        TN = counts.get('TN', 0)
        FA = counts.get('FA', 0)
        FD = counts.get('FD', 0)
        FN = counts.get('FN', 0)
        
        precision = _safe_divide(TP, (TP + FA + FD))
        recall = _safe_divide(TP, (TP + FN + FD))
        f1 = _safe_divide((2 * precision * recall), (precision + recall))
        accuracy = _safe_divide((TP + TN), (TP + FA + FD + TN + FN))
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'counts': counts
        }

    metrics = {
        field: _calculate_metrics_from_counts(counts)
        for field, counts in evaluation_results.items()
    }
    
    overall_counts = collections.Counter()
    for counts in evaluation_results.values():
        overall_counts.update(counts)
    
    metrics['overall'] = _calculate_metrics_from_counts(overall_counts)
    
    return metrics

metrics_results = calculate_metrics(evaluation_results)
print(json.dumps(metrics_results, indent=4))
```

Verify overall metrics: precision ≈ 62.5%, recall ≈ 41.7%, F1 = 50%, accuracy ≈ 46.7%. The low recall indicates the system is missing too many values.

### Section 4: Visualizing Performance for Analysis

Numbers alone make it hard to spot patterns across fields. Visualization transforms metric tables into actionable insights — you can immediately see which fields lag behind and by how much. In production systems, these visualizations feed into dashboards that trigger alerts when extraction quality degrades.

The two views serve different audiences: overall metrics give executives a system health summary, while field-level breakdowns give engineers the detail needed to prioritize improvements. A visual bar scaled to 100% makes relative performance instantly comparable without mental arithmetic.

**Build: Create text-based metric visualization**

```python
def visualize_metrics(metrics_results):
    """Create a visual representation of the evaluation metrics."""
    overall = metrics_results['overall']
    
    def create_visual_bar(percentage, width=20):
        filled = int(percentage * width / 100)
        return '█' * filled + '░' * (width - filled)
    
    print("\n--- Overall Metrics ---")
    print("Metric               Value Visual                ")
    print("--------------------------------------------------")
    
    prec = overall['precision'] * 100
    recall = overall['recall'] * 100
    f1 = overall['f1'] * 100
    accuracy = overall['accuracy'] * 100
    
    print(f"Precision           {prec:6.2f}% {create_visual_bar(prec)}  ")
    print(f"Recall              {recall:6.2f}% {create_visual_bar(recall)}  ")
    print(f"F1 Score            {f1:6.2f}% {create_visual_bar(f1)}  ")
    print(f"Accuracy            {accuracy:6.2f}% {create_visual_bar(accuracy)}  ")
    
    print("\n\n=== FIELD-LEVEL METRICS ===\n")
    print("Field                    TP     FA     TN     FN     FD     Prec   Recall       F1      Acc     Visual                ")
    print("-----------------------------------------------------------------------------------------------------------------------------")
    
    for field, metrics in metrics_results.items():
        if field == 'overall':
            continue
        
        c = metrics['counts']
        tp, fa, tn, fn, fd = c['TP'], c['FA'], c['TN'], c['FN'], c['FD']
        prec = metrics['precision'] * 100
        recall = metrics['recall'] * 100
        f1 = metrics['f1'] * 100
        accuracy = metrics['accuracy'] * 100
        
        visual = create_visual_bar(f1)
        print(f"{field:20s} {tp:6d} {fa:6d} {tn:6d} {fn:6d} {fd:6d} {prec:8.2f}% {recall:6.2f}% {f1:9.2f}% {accuracy:6.2f}%    {visual}  ")

visualize_metrics(metrics_results)
```

The output shows `vendor` has the lowest F1 (33%) — this field needs the most improvement. The visual bars make this immediately obvious without reading numbers.

## Challenges

**Challenge: Evaluate a New Document Set**

Using the evaluation pipeline you built, evaluate a new set of 8 invoice documents with 4 fields (add a `date` field). Your ground truth and predictions should include at least one example of each classification category (TP, TN, FA, FD, FN) across the new field set. Identify which field has the worst recall and propose a hypothesis for why.

**Assessment criteria:**

1. Code runs without errors and produces classification counts, metrics, and visualization
2. Uses the five-category classification framework (TP, TN, FA, FD, FN) correctly for all fields including the new `date` field
3. Handles edge cases: null values, empty strings, numeric tolerance, and missing fields
4. Produces both field-level and overall metrics with correct formulas
5. Identifies the lowest-recall field with a written explanation of potential causes
6. Learner can explain their approach and why specific predictions were classified into each category

## Wrap-Up

You built a structured data evaluation pipeline covering the full cycle: data preparation → classification → metrics computation → visualization. The key insight is that raw accuracy hides important details — field-level precision and recall reveal *where* and *how* your system fails, which is what you need to improve it.

This skill focused on flat field extraction with exact matching. Real IDP systems face additional complexity: nested structures, fuzzy matching requirements, and format normalization. The [Module 04 Capstone Challenge](CHALLENGE-capstone.md) integrates this structured data evaluation with other workload-specific evaluation techniques into a comprehensive assessment pipeline.

**Next steps:** Apply these evaluation patterns to your own document extraction use case, or continue to the capstone challenge to combine structured data evaluation with other workload-specific techniques from Module 04.
