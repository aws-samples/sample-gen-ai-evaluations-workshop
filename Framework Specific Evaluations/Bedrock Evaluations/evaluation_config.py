"""Settings for the Bedrock Evaluations module.

The generator model comes from the workshop's shared config,
Foundational Evaluations/model_config.py. The evaluator (judge) model is set here
because Bedrock evaluation jobs accept only the evaluator models listed in
https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html#evaluation-judge-supported
and the shared JUDGE_MODEL_ID is not on that list.
"""

# The evaluator model Bedrock uses as the judge, for built-in and custom metrics.
# Claude Haiku 4.5 (US cross-region inference profile, the same ID as HAIKU_MODEL_ID
# in the shared config), to keep the default cost low. It must be on both
# supported-evaluator lists in the page above (built-in metrics and custom metrics).
# A cheaper judge is a choice to test, not to assume: the notebook measures this
# judge against a programmatic check (TPR and TNR). Change the model only if those
# results show you need to.
EVALUATOR_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Prefix for the S3 bucket, IAM role, and evaluation jobs the notebook creates,
# so the cleanup cell can find them.
RESOURCE_PREFIX = "genai-evals-workshop"
