# Challenge: Framework-Specific Evaluations with Promptfoo

Set up a complete promptfoo evaluation pipeline — prompt templates, test cases with assertions, YAML configuration, and multi-provider comparison — for a classification task using Amazon Bedrock.

## Prerequisites

- [SKILL: Promptfoo Evaluation Framework](./SKILL-promptfoo.md)
- [CHALLENGE: Deep-Dive](./CHALLENGE-deep-dive.md)
- Source notebooks: [05-01-Prompt-Foo](../../05-framework-specific-evaluations/05-01-Prompt-Foo/)

## Exercise 1: Create a Prompt Template and Test Dataset

Write a Python prompt function for a text classification task of your choice (not email classification — pick a new domain like support ticket routing, product categorization, or sentiment analysis). Create a CSV test dataset with at least 10 labeled examples.

**Success criteria:**
- `prompts.py` contains a function that accepts a single input variable and returns a fully-formatted classification prompt
- Prompt constrains the model to output only a category label (no explanations)
- `dataset.csv` has at least 10 rows with an input column and an `__expected` column
- Dataset covers at least 3 distinct categories with at least 2 examples per category
- Print the prompt output for one sample input to verify formatting

## Exercise 2: Configure and Run a Multi-Provider Evaluation

Write a `promptfooconfig.yaml` that references your prompt function, configures at least 2 Bedrock providers, and points to your test dataset. Run the evaluation.

**Success criteria:**
- YAML file has `description`, `prompts` (referencing `prompts.py:function_name`), `providers` (at least 2 Bedrock models with labels and region config), and `tests` (referencing `dataset.csv`)
- `promptfoo eval --no-cache --no-progress-bar` runs without errors
- Results show pass/fail counts per provider
- Print overall accuracy per provider and identify which model performs better

## Exercise 3: Add Custom Assertions

Extend your test cases with assertion types beyond exact match. Add at least 3 test cases that use `contains`, `not-contains`, or regex-based assertions to validate output format and content.

**Success criteria:**
- At least 3 test cases in `dataset.csv` or a separate YAML test file use non-exact assertions
- At least one `contains` assertion verifies the output includes a required keyword
- At least one `not-contains` or `icontains-none` assertion verifies the output excludes unwanted content (e.g., explanations, apologies)
- Re-run `promptfoo eval` and show that the new assertions produce meaningful pass/fail results
- Explain why format assertions matter in addition to correctness assertions

## Exercise 4: Analyze Results and Iterate

Examine your evaluation results, identify failure patterns, modify your prompt to address the most common failure mode, and re-run to show improvement.

**Success criteria:**
- Identify the category or assertion type with the lowest pass rate from Exercise 2/3 results
- Modify `prompts.py` to address the failure (e.g., add few-shot examples, clarify category definitions, add output format constraints)
- Re-run `promptfoo eval` with the updated prompt
- Print a before/after comparison: category, old pass rate, new pass rate
- Explain what you changed in the prompt and why it helped (or didn't)

## Tips

- Use `promptfoo --version` to verify installation before starting
- The SKILL doc's email classification example shows the exact YAML structure — adapt it for your domain
- `promptfoo share` generates a shareable URL with an interactive results viewer
- For Exercise 3, promptfoo supports assertion types like `contains`, `icontains`, `not-contains`, `is-json`, `regex`, and `javascript` — see promptfoo docs for the full list
- Keep your prompt function simple — classification prompts that constrain output format tend to evaluate more cleanly
