# Amazon Bedrock Evaluations

**Managed LLM-as-a-judge evaluation jobs, validated against ground truth**

The other modules run their judges in the notebook. Amazon Bedrock also runs evaluations as a managed service: you put a dataset in Amazon S3, choose metrics, and an evaluation job runs a judge model over every record, writes per-record scores and explanations to S3, and shows a report in the Bedrock console.

This module runs one LLM-as-a-judge **model evaluation job** with your own inference responses ("bring your own inference responses"), so the same job can grade any application or model, on Bedrock or elsewhere. It combines three built-in metrics with a custom binary metric, then checks the managed judge against a programmatic ground-truth check, using the TPR and TNR method from [Foundational Evaluations, 02 Quality Metrics](../../Foundational%20Evaluations/02-quality-metrics/).

## What you'll learn

1. When a managed evaluation job fits better than a judge in your notebook
2. The prompt dataset format, including `referenceResponse`, `category` and `modelResponses`
3. Built-in metrics (`Builtin.Correctness`, `Builtin.Completeness`, `Builtin.FollowingInstructions`) and a custom pass/fail metric
4. A least-privilege service role and S3 bucket for evaluation jobs
5. Reading per-record results and measuring the judge's TPR and TNR

## Notebook

| Notebook | Time | Cost |
|---|---|---|
| [01 Bedrock LLM-as-a-Judge Evaluation Jobs](01%20Bedrock%20LLM-as-a-Judge%20Evaluation%20Jobs.ipynb) | About 25 minutes, most of it waiting for the job | Well under $1 |

## Files

| File | Purpose |
|---|---|
| `evaluation_config.py` | The evaluator model and the resource name prefix |
| `eval_job_utils.py` | Creates and removes the S3 bucket and IAM role, waits for a job, reads results |
| `tests_local/` | Offline tests of `eval_job_utils.py`: `python -m pytest tests_local` |

## Models

- **Generator:** `NOVA_2_LITE_MODEL_ID` from [`Foundational Evaluations/model_config.py`](../../Foundational%20Evaluations/model_config.py). The notebook generates the responses itself; the job only grades them.
- **Evaluator:** `EVALUATOR_MODEL_ID` in `evaluation_config.py`, Claude Haiku 4.5 by default to keep costs low. The notebook measures how well it agrees with a programmatic check; change it only if those results show you need a different judge. Evaluation jobs accept only the [supported evaluator models](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html#evaluation-judge-supported), and a custom metric needs a model on the custom-metrics list too. That is why this module does not use the workshop's shared `JUDGE_MODEL_ID`.

## AWS resources and permissions

The notebook creates, and its last section deletes:

- An S3 bucket, `genai-evals-workshop-<account>-<region>`, with all public access blocked
- An IAM role, `genai-evals-workshop-bedrock-eval-<region>`, that only Bedrock evaluation jobs in your account and Region can assume. Its inline policy allows reading and writing that bucket and invoking Bedrock models.
- One evaluation job per run, named `genai-evals-workshop-city-qa-<timestamp>`

Your own credentials need, in addition to `bedrock:InvokeModel`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateEvaluationJob", "bedrock:GetEvaluationJob",
        "bedrock:ListEvaluationJobs", "bedrock:BatchDeleteEvaluationJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket", "s3:PutBucketPublicAccessBlock", "s3:DeleteBucket",
        "s3:PutObject", "s3:GetObject", "s3:ListBucket", "s3:DeleteObject"
      ],
      "Resource": ["arn:aws:s3:::genai-evals-workshop-*", "arn:aws:s3:::genai-evals-workshop-*/*"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole", "iam:GetRole", "iam:TagRole", "iam:PutRolePolicy",
        "iam:ListRolePolicies", "iam:DeleteRolePolicy", "iam:DeleteRole", "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::*:role/genai-evals-workshop-*"
    }
  ]
}
```

## Setup

```bash
pip install -r requirements.txt
```

The notebook uses `AWS_REGION`, or `us-east-1` when it is not set.

## Learn more

- [Evaluate model performance using another LLM as a judge](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html)
- [Prompt datasets for judge-based evaluation jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-prompt-datasets-judge.html)
- [Create a prompt for a custom metric](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-custom-metrics-prompt-formats.html)
- [Service role permissions for judge-based evaluation jobs](https://docs.aws.amazon.com/bedrock/latest/userguide/judge-service-roles.html)
