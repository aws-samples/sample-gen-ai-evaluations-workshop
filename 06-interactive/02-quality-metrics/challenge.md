# Challenge: Quality Metrics with LLM-as-Judge

Build LLM-as-Judge and LLM-as-Jury evaluation systems, measure inter-judge agreement, and recommend which approach to use for different evaluation scenarios.

## Prerequisites

- [SKILL: LLM Quality Metrics](./SKILL.md)
- Source notebooks: [01_LLM_as_Judge_analysis.ipynb](../../02-quality-metrics/01_LLM_as_Judge_analysis.ipynb), [02_LLM_as_Jury_evaluation_analysis.ipynb](../../02-quality-metrics/02_LLM_as_Jury_evaluation_analysis.ipynb), [03_Evaluating_your_Judge.ipynb](../../02-quality-metrics/03_Evaluating_your_Judge.ipynb)

## Exercise 1: Build a Structured LLM-as-Judge

Write a judge prompt template and evaluation function that scores LLM responses on a multi-dimensional rubric. The judge should evaluate responses to customer support questions for a retail company.

**Success criteria:**
- Judge prompt includes at least 3 scoring dimensions (e.g., Correctness, Completeness, Relevance) each scored 1–5
- Prompt constrains the judge to output structured scores (e.g., XML tags or JSON) with a brief justification per dimension
- Evaluation function calls Bedrock with `temperature=0.1`, parses the structured output, and returns a dict of dimension scores
- Run the judge on at least 5 different question/answer pairs and print a summary table of scores per dimension

## Exercise 2: Calculate Judge Reliability

Implement a judge reliability scoring function that profiles a judge's behavior across multiple evaluations. Measure consistency (low variance), discrimination (uses the full score range), and centrality (avoids extreme bias).

**Success criteria:**
- Function accepts a list of scores from a single judge and returns a reliability score (0–1) with component breakdowns
- Consistency component uses coefficient of variation (lower CV = higher consistency)
- Discrimination component measures how many unique score values the judge uses relative to the scale
- Run the same 5 question/answer pairs through the judge 3 times each and compute reliability from the 15 scores
- Print reliability score and explain whether this judge is trustworthy

## Exercise 3: Build an LLM-as-Jury System

Extend your single-judge into a jury of 3 judges using different models or temperature settings. Aggregate scores using bootstrap confidence intervals.

**Success criteria:**
- Jury uses at least 2 distinct Bedrock models (e.g., Nova Pro and Claude Haiku) or the same model at different temperatures
- Each jury member scores the same 5 responses on the same rubric from Exercise 1
- Bootstrap confidence interval function resamples jury scores 1000 times and returns mean, 95% CI lower, and 95% CI upper
- Print a comparison table: response ID, single-judge score, jury mean, CI width
- Identify which responses have the widest confidence intervals and explain why

## Exercise 4: Agreement Rate Analysis and Recommendation

Calculate pairwise agreement rates across your jury members for each scoring dimension. Produce a written recommendation on when to use single-judge vs. jury evaluation.

**Success criteria:**
- Agreement rate function computes pairwise agreement (scores within ±1) for each dimension across all jury members
- Print agreement rates per dimension and identify which dimensions are most/least contested
- Written recommendation (3–5 sentences) addresses: cost tradeoff, when jury adds value, which dimensions need jury consensus
- Recommendation references specific agreement rate numbers from your results

## Tips

- Use `temperature=0.1` for judges to maximize consistency; use `temperature=0.7` only if you want to test variance
- The SKILL doc's `bootstrap_confidence_interval` function is a good starting point for Exercise 3
- Wide confidence intervals signal subjective dimensions — these are where jury evaluation adds the most value
- Keep your test dataset consistent across all exercises so results are comparable
