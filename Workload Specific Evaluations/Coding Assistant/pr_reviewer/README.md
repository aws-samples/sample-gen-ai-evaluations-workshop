# pr_reviewer — aligned PR reviewer

A small, dependency-light PR reviewer that uses Claude on Amazon Bedrock
to produce binary pass/fail verdicts against a review rubric. Born as the
LLM judge in the Coding Assistant eval workshop; useful in CI on its own.

## Why "aligned"

Generic LLM-as-judge prompts drift because "looks good" is underspecified.
This reviewer takes a **rubric** — a structured checklist of dimensions
and red flags — as input. The rubric is the alignment anchor between
human review standards and the LLM's verdict. The workshop's
`tasks/ground_truth/*.md` files are examples of good rubrics.

## Installation

Part of the workshop. From the `Coding Assistant/` directory:

```bash
pip install boto3   # only required dep beyond the Python stdlib
```

AWS credentials must be configured with access to Bedrock
(`us.anthropic.claude-sonnet-4-5-20250929-v1:0`).

## Usage — library

```python
from pathlib import Path
from pr_reviewer import review, Rubric

rubric = Rubric.from_path(Path("tasks/ground_truth/T01_remove_stray_prints.md"))
diff = Path("agent_output.diff").read_text()
result = review(diff, rubric=rubric)

print(result.overall)           # "pass" or "fail"
for d in result.dimensions:
    print(d.name, d.verdict, "—", d.reason)
```

## Usage — CLI

```bash
# Review a produced diff against a specific rubric
python -m pr_reviewer --diff out.diff --rubric tasks/ground_truth/T01_remove_stray_prints.md

# Review a local repo's current HEAD against a base ref (CI-style)
python -m pr_reviewer --repo . --base origin/main --format markdown

# Pipe a diff in
git diff origin/main...HEAD | python -m pr_reviewer --format json
```

Exit code is `0` on overall pass, `1` on fail — making it easy to gate CI.

## GitHub Actions

A drop-in workflow is provided at
`../.github-action-example/pr-review.yml`. It:

1. Checks out the PR branch.
2. Runs `python -m pr_reviewer --repo . --base origin/${{ github.base_ref }}`.
3. Posts the markdown review as a PR comment.
4. Fails the check if the review fails — so the verdict blocks merge.

Copy it into your repo's `.github/workflows/`, adjust the rubric path
(or let it fall back to the generic default rubric), and wire up AWS
credentials via your preferred method (OIDC recommended).

## Rubric format

The reviewer parses the markdown rubric format used in the workshop's
`tasks/ground_truth/`:

```markdown
# <title>

## Dimensions

### 1. correctness
- criterion
- criterion

### 2. scope_discipline
- criterion

## Red flags
- red flag 1
- red flag 2
```

If no rubric is supplied, a built-in generic rubric is used that covers
correctness, scope, test coverage, and security.
