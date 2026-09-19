# Industry Specific Evaluations

Generic evaluation patterns teach you how to measure AI systems. Industry-specific modules teach you what to measure.

Every module in this directory applies the same foundational techniques from the core workshop (binary pass/fail judges, precision/recall, calibration) to domain-specific data shapes and failure modes that general-purpose evals miss.

## Why this exists

Most evaluation patterns in this workshop apply regardless of industry. A chatbot eval works whether the chatbot answers questions about insurance or about pet food. But certain workloads have failure modes that only surface when the data has industry-specific structure:

- A product catalog enrichment pipeline can hallucinate allergen claims. Getting "gluten free" wrong has regulatory consequences that "slightly off tone" does not.
- A demand sensing agent can be directionally correct but poorly calibrated. The difference between 0.7 confidence and 0.9 confidence determines whether a human reviews the decision or the system acts autonomously.
- A supply chain chatbot might answer correctly 95% of the time but disclose supplier pricing in the other 5%. Generic retrieval metrics won't catch that.

These modules give teams a starting point that looks like their actual workload, not a toy example they have to mentally translate.

## Modules

| Module | Domain | What it evaluates |
|--------|--------|-------------------|
| [Retail and CPG](Retail%20and%20CPG/) | Consumer packaged goods, grocery, retail | Catalog enrichment accuracy, demand forecast calibration, supply chain chatbot guardrails |

## How to use

Complete the [Foundational Evaluations](../Foundational%20Evaluations/) first. These modules assume familiarity with binary judges, precision/recall, and the LLM-as-Judge pattern. Then pick the industry module closest to your workload.

Each module includes:
- A README with scenario descriptions, evaluation criteria, and success thresholds
- A Jupyter notebook with runnable evaluation code using Amazon Bedrock
