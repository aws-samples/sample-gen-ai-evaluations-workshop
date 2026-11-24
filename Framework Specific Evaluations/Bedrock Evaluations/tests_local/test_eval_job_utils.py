"""Offline checks of eval_job_utils. They need no AWS credentials.

Run from the module directory: python -m pytest tests_local
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_job_utils as u  # noqa: E402


def test_trust_policy_limits_the_role_to_bedrock_jobs_in_this_account_and_region():
    statement = u._trust_policy("123456789012", "us-east-1")["Statement"][0]
    assert statement["Principal"] == {"Service": "bedrock.amazonaws.com"}
    assert statement["Condition"]["StringEquals"] == {"aws:SourceAccount": "123456789012"}
    assert statement["Condition"]["ArnLike"]["aws:SourceArn"] == "arn:aws:bedrock:us-east-1:123456789012:evaluation-job/*"


def test_permissions_policy_scopes_s3_to_the_workshop_bucket():
    policy = u._permissions_policy("123456789012", "genai-evals-workshop-123456789012-us-east-1")
    s3 = next(s for s in policy["Statement"] if s["Sid"] == "ReadDatasetWriteResults")
    assert s3["Resource"] == [
        "arn:aws:s3:::genai-evals-workshop-123456789012-us-east-1",
        "arn:aws:s3:::genai-evals-workshop-123456789012-us-east-1/*",
    ]
    assert "s3:DeleteObject" not in s3["Action"]
    bedrock = next(s for s in policy["Statement"] if s["Sid"] == "InvokeEvaluatorModel")
    assert all(r.startswith("arn:aws:bedrock:") for r in bedrock["Resource"])


def test_scores_by_metric_reads_scores_and_explanations():
    record = {
        "inputRecord": {"prompt": "q"},
        "automatedEvaluationResult": {"scores": [
            {"metricName": "Builtin.Correctness", "result": 1.0,
             "evaluatorDetails": [{"modelIdentifier": "m", "explanation": "right"}]},
            {"metricName": "number_within_5_percent", "result": 0.0},
        ]},
    }
    assert u.scores_by_metric(record) == {
        "Builtin.Correctness": {"result": 1.0, "explanation": "right"},
        "number_within_5_percent": {"result": 0.0, "explanation": ""},
    }


def test_scores_by_metric_handles_a_record_without_results():
    assert u.scores_by_metric({"inputRecord": {"prompt": "q"}}) == {}
