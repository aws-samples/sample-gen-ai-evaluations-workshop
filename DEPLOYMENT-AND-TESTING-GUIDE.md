# Deployment & Testing Guide — AWS Evaluations Workshop

## Table of Contents
- [1. Deployment](#1-deployment)
- [2. Testing Workshop Content](#2-testing-workshop-content)
- [3. Facilitator Guide](#3-facilitator-guide)
- [4. CI/CD Integration](#4-cicd-integration)

---

## 1. Deployment

### Deployment Targets

| Environment | Best for | Notes |
|---|---|---|
| **SageMaker Studio** | Instructor-led | Pre-built Jupyter; attach IAM role to domain. Recommended. |
| **Cloud9** | Self-paced | Good for PromptFoo (npm available); resize EBS to 30 GB |
| **Local** | Experienced builders | Requires `aws configure` and Python 3.10+ |

### IAM Permissions

Attach this policy to the learner role (or Studio execution role):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
      "bedrock:Converse", "bedrock:ConverseStream",
      "bedrock:CreateGuardrail", "bedrock:TagResource", "bedrock:ApplyGuardrail",
      "bedrock:ListGuardrails", "bedrock:GetGuardrail",
      "cloudwatch:PutMetricData",
      "cloudwatch:PutDashboard", "cloudwatch:PutMetricAlarm",
      "logs:FilterLogEvents", "ecr:DescribeRepositories"
    ],
    "Resource": "*"
  }]
}
```

> **Note:** `PutDashboard` and `PutMetricAlarm` extend the base curriculum.md list; they are needed by Module 01 dashboard exercises.

### Bedrock Model Access

Enable in the Bedrock console (us-east-1 or us-west-2):

- **Anthropic Claude 3.7 Sonnet** — primary model, all modules
- **Anthropic Claude 3 Sonnet** — Module 02 jury member
- **Amazon Nova Lite / Nova Pro** — Modules 01, 04, 05

Verify: `aws bedrock list-foundation-models --query "modelSummaries[?contains(modelId,'claude')].modelId"`

### Environment Setup

```bash
# Core (all modules)
pip install boto3 pandas numpy

# Module 02 — Quality Metrics
pip install matplotlib seaborn scipy

# Module 03 — Agentic Metrics
pip install strands-agents duckduckgo-search beautifulsoup4

# Module 04 — Workload Evaluations
pip install llama-index faiss-cpu chromadb python-dotenv PyPDF2

# Module 05 — Framework Evaluations
pip install strands-agents strands-agents-tools strands-agents-evals "dspy>=3.1,<4" bedrock-agentcore bedrock-agentcore-starter-toolkit duckduckgo-search

# Requires Node.js 18+
npm install -g promptfoo
```

### Distributing Notebooks

SKILL.md files are teaching content; source notebooks are the runnable code. Place notebooks alongside SKILLs in each module directory. For SageMaker Studio, clone the repo via a lifecycle config. For Cloud9, use `git clone` in the bootstrap script.

---

## 2. Testing Workshop Content

### SKILL.md Structural Validation

Every SKILL file must have: (1) YAML frontmatter with `name` and `description`, (2) sections for Prerequisites, Learning Objectives, and Setup, (3) at least one fenced code block.

```bash
#!/usr/bin/env bash
# scripts/validate_skills.sh
set -euo pipefail; ERRORS=0
for f in $(find . -name 'SKILL*.md'); do
  head -1 "$f" | grep -q '^---$'         || { echo "FAIL: $f no frontmatter"; ((ERRORS++)); }
  for s in "Prerequisites" "Learning Objectives" "Setup"; do
    grep -q "^## $s" "$f"                || { echo "FAIL: $f missing '## $s'"; ((ERRORS++)); }
  done
  grep -q '```python\|```bash' "$f"      || { echo "FAIL: $f no code blocks"; ((ERRORS++)); }
  lines=$(wc -l < "$f")
  [ "$lines" -gt 500 ] && echo "WARN: $f has $lines lines (over 500 limit)"
done
[ "$ERRORS" -eq 0 ] && echo "All SKILL files valid." || { echo "$ERRORS error(s)"; exit 1; }
```

### Challenge Validation

```bash
for f in $(find . -name 'CHALLENGE*.md'); do
  grep -q "Assessment Criteria" "$f"                              || echo "FAIL: $f missing Assessment Criteria"
  grep -q 'Assessment criteria\|Assessment Criteria\|Criterion' "$f" || echo "FAIL: $f missing scoring rubric"
done
```

### Code Snippet Smoke Tests

Extract and run the first Python block from each SKILL to catch import/API errors:

```bash
for f in $(find . -name 'SKILL*.md'); do
  sed -n '/^```python$/,/^```$/{ /^```/d; p; }' "$f" | head -30 > /tmp/snippet.py
  [ -s /tmp/snippet.py ] && timeout 30 python /tmp/snippet.py 2>&1 || echo "WARN: $f"
done
```

> Full validation requires running the source notebooks end-to-end.

### Smoke Test Checklist

| Module | Quick check |
|---|---|
| 01 — Operational Metrics | `python -c "import boto3; boto3.client('cloudwatch').list_metrics(Namespace='BedrockEvals')"` |
| 02 — Quality Metrics | `python -c "import pandas, scipy, seaborn, boto3; print('OK')"` |
| 03 — Agentic Metrics | `python -c "from strands import Agent; print('OK')"` |
| 04 — Workload-Specific | `python -c "import chromadb, llama_index; print('OK')"` |
| 05 — Framework-Specific | `promptfoo --version && python -c "import dspy; print('OK')"` |

### Verifying the Dependency Map

```bash
grep -h "Completed\|Completion of\|Requires" $(find . -name 'SKILL*.md') | sort -u
```

Compare against the map in `curriculum.md`: Module 03 → requires 01 + 02; Modules 04 & 05 → require 01–03.

---

## 3. Facilitator Guide

### Recommended Delivery Order

Module 01 is recommended before Module 02. Module 02's SKILL.md lists Module 01 as a prerequisite, though the concepts are not strictly dependent.

```
Day 1:  Module 01 (Operational)  →  Module 02 (Quality)
Day 2:  Module 03 (Agentic)     →  Module 04 (Workload) — pick 2 SKILLs
Day 3:  Module 05 (Framework)   →  Capstone / Deep-Dive Challenge
```

### Time Estimates

| Module | Content | Hands-on | Challenge | Total |
|---|---|---|---|---|
| 01 — Operational Metrics | 45 min | 60 min | — | ~2 hrs |
| 02 — Quality Metrics | 45 min | 75 min | — | ~2 hrs |
| 03 — Agentic Metrics | 45 min | 60 min | — | ~2 hrs |
| 04 — Workload (2 SKILLs + capstone) | 60 min | 90 min | 60 min | ~3.5 hrs |
| 05 — Framework (2 SKILLs + deep-dive) | 60 min | 90 min | 60 min | ~3.5 hrs |

### Common Learner Issues

| Issue | Fix |
|---|---|
| `AccessDeniedException` on Bedrock | Model access not enabled — Bedrock console → Model access → Request |
| `ThrottlingException` in Module 01 | Reduce concurrency or switch to Nova Lite |
| `ModuleNotFoundError: strands_agents` | `pip install strands-agents` (hyphen, not underscore) |
| `ModuleNotFoundError: ddgs` | `pip install duckduckgo-search` (package name differs from import name) |
| PromptFoo `command not found` | `npm install -g promptfoo` — requires Node.js 18+ |
| ChromaDB sqlite3 version error | `pip install pysqlite3-binary` |
| CloudWatch metrics not appearing | 1–2 min propagation delay; verify namespace spelling |
| DSPy optimization hangs | Reduce `max_bootstrapped_demos` to 2 for workshop |

> **Note:** SKILL-agentcore.md covers both the AgentCore Metrics (05-02) and AgentCore Runtime Evals (05-04) content from the source notebooks.

### Assessing Challenge Submissions

1. **Run it** — must execute end-to-end without errors (baseline).
2. **Score the rubric** — use the weighted table in each CHALLENGE file.
3. **Module 04 capstone** — verify all pipeline stages (guardrails, RAG, custom metric, orchestration, report) produce results and the 5 assessment criteria are met.
4. **Module 05 deep-dive** — look for "beyond notebook" features and the "what was left out" self-assessment (10% of rubric).

---

## 4. CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/validate-content.yml
name: Validate Workshop Content
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate SKILL structure
        run: bash scripts/validate_skills.sh
      - name: Validate challenges
        run: |
          for f in $(find . -name 'CHALLENGE*.md'); do
            grep -q "Assessment Criteria" "$f" || { echo "FAIL: $f"; exit 1; }
            grep -q 'Assessment criteria\|Assessment Criteria\|Criterion' "$f" || { echo "FAIL: $f"; exit 1; }
          done
      - name: Check internal links
        run: |
          for f in $(find . -name '*.md'); do
            grep -oP '\[.*?\]\(\K[^)#]+' "$f" | grep -v '^http' | while read link; do
              [ -f "$(dirname "$f")/$link" ] || echo "BROKEN: $f -> $link"
            done
          done
      - name: Verify curriculum references
        run: |
          for dir in module-*/; do
            grep -q "${dir%/}" curriculum.md || { echo "FAIL: $dir missing from curriculum.md"; exit 1; }
          done
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-skills
        name: Validate SKILL files
        entry: bash scripts/validate_skills.sh
        language: system
        files: 'SKILL.*\.md$'
        pass_filenames: false
```

```bash
pip install pre-commit && pre-commit install
```

### CodePipeline Alternative

For AWS-native CI, use CodeBuild with `aws/codebuild/standard:7.0`:

```yaml
# buildspec.yml
version: 0.2
phases:
  build:
    commands:
      - bash scripts/validate_skills.sh
```
