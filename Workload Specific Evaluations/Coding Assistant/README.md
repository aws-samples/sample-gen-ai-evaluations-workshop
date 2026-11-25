# Coding Assistant Evaluations

Compare coding agents (Codex, Claude Code, OpenCode, ...) and models on Amazon Bedrock, on tasks made from real pull requests:

- **PR review:** the agent reviews a change and reports defects; it is graded against the review the change really got. `pr4615` (the reviewer blocked it) and `pr4121` (approved), from `awslabs/mcp`.
- **Bug fix:** the agent fixes a bug in a repository rolled back to before the fix; it is graded with hidden tests, a validation agent, and LLM judges, as Senior SWE-Bench does it. `pr371` and `pr338`, from `strands-agents/evals`.

[Harbor](https://github.com/harbor-framework/harbor) 0.26.0 runs the agent and the grader. Each trial runs in its own Amazon Bedrock AgentCore Runtime session (one microVM) behind a firewall, so nothing runs on your machine except Harbor itself. The tasks are in `datasets/`, in the Harbor task format.

**Status: pre-release.** All numbers below come from one AWS account. See [Known limits](#known-limits) before you rely on a score.

## Prerequisites

| Need | Detail |
|---|---|
| Python 3.12 and [uv](https://docs.astral.sh/uv/) | Harbor 0.26.0 needs Python 3.12 or later |
| An AWS account | One where you may create IAM roles, ECR, S3, CodeBuild, and AgentCore runtimes ([Permissions](#permissions)). Use a sandbox account |
| Region | `us-east-1` |
| Models | The models you want to test, available in your account. The judges use `us.anthropic.claude-haiku-5-5` (`model_config.py`) |

## Quick start (about 25 minutes)

Run everything from this folder. In every new terminal, set the region and `PYTHONPATH` again.

```bash
uv venv --clear --python 3.12 .venv && uv pip install --python .venv/bin/python 'harbor==0.26.0' 'boto3>=1.44.7'
export AWS_REGION=us-east-1 PYTHONPATH=$PWD     # Harbor imports harbor_env from here
unset AWS_BEARER_TOKEN_BEDROCK                  # a Bedrock API key makes bootstrap write the wrong judge model
# optional: export P2_PREFIX=evals-me-          # resource name prefix (default evals-hb-); set it before bootstrap
```

**1. Bootstrap** (seconds). It creates the IAM roles, the ECR repository, the S3 bucket, and the CodeBuild project. The agent role may call only the models you name here, so name every model you plan to test:

```bash
.venv/bin/python harbor_env/setup.py bootstrap --agent-model openai.gpt-6-sol \
    --agent-model us.anthropic.claude-sonnet-5-5 --agent-model us.amazon.nova-2-lite-v1:0
```

**2. Build the images** (about 10 minutes). Without this step, the first trial of a task builds its image.

```bash
.venv/bin/python harbor_env/setup.py prebuild \
    datasets/awslabs-mcp-pr-review/tasks/awslabs__mcp__pr4121-approve \
    datasets/strands-evals-coding/tasks/strands-agents__evals__pr371-fix
```

**3. Check the grader:** run the bug-fix task with the reference answer (`oracle`, must score 1) and with an agent that does nothing (`nop`, must score 0).

```bash
T=datasets/strands-evals-coding/tasks/strands-agents__evals__pr371-fix
.venv/bin/harbor run -p $T -e harbor_env.agentcore_env:AgentCoreEnvironment -a oracle -o jobs --job-name first-oracle -y
.venv/bin/harbor run -p $T -e harbor_env.agentcore_env:AgentCoreEnvironment -a nop -o jobs --job-name first-nop -y
```

**4. Run a real agent** on both kinds of task:

```bash
CODEX=(-a codex -m openai.gpt-6-sol --skip-model-parsing-and-provider-mapping --ak config=codex-bedrock.toml --ak disable_web_search=true --ae AWS_REGION=us-east-1)
.venv/bin/harbor run -p datasets/awslabs-mcp-pr-review/tasks/awslabs__mcp__pr4121-approve \
    -e harbor_env.agentcore_env:AgentCoreEnvironment "${CODEX[@]}" -o jobs --job-name first-review -y
.venv/bin/harbor run -p $T -e harbor_env.agentcore_env:AgentCoreEnvironment "${CODEX[@]}" -o jobs --job-name first-fix -y
```

**5. Read the results:**

```bash
.venv/bin/python harbor_env/summary.py jobs/first-oracle jobs/first-nop jobs/first-review jobs/first-fix
.venv/bin/harbor view jobs     # a local web viewer: each trial's commands, messages, and changed files
```

| Job | `grading` | What we saw |
|---|---|---|
| `first-oracle` | `staged` | `reward 1.0` |
| `first-nop` | `staged` | `reward 0.0` (`fail_to_pass 0.0`, `pass_to_pass 1.0`) |
| `first-review` | `separate` | The verdict varies between runs: Codex usually approves (passed), sometimes blocks the change with one `high` finding (failed). A failed review is a result, not a setup error |
| `first-fix` | `staged` | Usually `reward 1.0`; sometimes 0.0 when its fix breaks a hidden test |

No row should show an exception, `unsealed`, `refused`, or a `MISSING grader_mode` warning. When you are done, [clean up](#clean-up).

A job name can be used once: with the same `--job-name`, Harbor resumes the old job and runs nothing. Use a new name, or delete `jobs/<name>`.

## Choose a harness and a model

The harness is Harbor's `-a`, the model is Harbor's `-m`. To test a model, name it in `bootstrap --agent-model` (with every model you already use: the agent role allows only those), then pass it to `-m`. An `openai.*` ID is called on `bedrock-mantle`; any other ID must be an inference profile in your account. For example, to also test Nova Pro:

```bash
.venv/bin/python harbor_env/setup.py bootstrap --agent-model openai.gpt-6-sol --agent-model us.anthropic.claude-sonnet-5-5 \
    --agent-model us.amazon.nova-2-lite-v1:0 --agent-model us.amazon.nova-pro-v1:0
```

Each harness needs its own flags to work inside the sandbox: its model endpoint, and any side model it calls pointed at the same model (the agent may call only the named models), with update checks and downloads off (there is no internet). Copy these into your shell and change the model:

```bash
CODEX=(-a codex -m openai.gpt-6-sol --skip-model-parsing-and-provider-mapping
  --ak config=codex-bedrock.toml --ak disable_web_search=true --ae AWS_REGION=us-east-1)
M=us.anthropic.claude-sonnet-5-5
CLAUDE=(-a claude-code -m $M --skip-model-parsing-and-provider-mapping --ae CLAUDE_CODE_USE_BEDROCK=1
  --ae ANTHROPIC_DEFAULT_HAIKU_MODEL=$M --ae ANTHROPIC_SMALL_FAST_MODEL=$M
  --ae CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 --ae DISABLE_AUTOUPDATER=1 --ae AWS_REGION=us-east-1)
N=us.amazon.nova-2-lite-v1:0
OPENCODE=(-a opencode -m amazon-bedrock/$N --ae AWS_REGION=us-east-1
  --ae OPENCODE_DISABLE_MODELS_FETCH=1 --ae OPENCODE_DISABLE_AUTOUPDATE=1 --ae OPENCODE_DISABLE_LSP_DOWNLOAD=1
  --ae OPENCODE_DISABLE_DEFAULT_PLUGINS=1 --ae OPENCODE_DISABLE_CLAUDE_CODE=1
  --ak "opencode_config={\"small_model\":\"amazon-bedrock/$N\",\"autoupdate\":false,\"share\":\"disabled\"}")
```

| Harness | Models | Tested with |
|---|---|---|
| Codex | OpenAI models on `bedrock-mantle` (Responses API) | `openai.gpt-6-sol`, all four tasks |
| Claude Code | Claude models on `bedrock-runtime` | `us.anthropic.claude-sonnet-5-5`, `pr4121` and `pr371` |
| OpenCode | Bedrock Converse models (Nova, Kimi, GLM, Claude, ...) | `us.amazon.nova-2-lite-v1:0`, `pr4121` and `pr371` |

- Do not pass AWS keys with `--ae`: the environment drops them and gives the agent its own short-lived credentials.
- Harbor's warning `could not determine or estimate the inference endpoint for agent 'codex'` is safe to ignore.
- **Another Harbor agent** (Aider, Goose, ...): install it in `harbor_env/image/layer.Dockerfile` (the agent has no internet), and add its model endpoint to the task's `allowed_hosts`. Not tested. Pi was tried and does not work yet: it asks Nova 2 Lite for more output tokens than the model allows.

## Compare them

An agent's result varies between runs, so run each harness and model a few times: `-k 3 -n 3` runs 3 trials at once. `-p datasets/awslabs-mcp-pr-review/tasks` runs every task in that folder.

```bash
P=datasets/awslabs-mcp-pr-review/tasks/awslabs__mcp__pr4121-approve
E=harbor_env.agentcore_env:AgentCoreEnvironment
.venv/bin/harbor run -p $P -e $E "${CODEX[@]}" -k 3 -n 3 -o jobs --job-name cmp-codex -y
.venv/bin/harbor run -p $P -e $E "${CLAUDE[@]}" -k 3 -n 3 -o jobs --job-name cmp-claude -y
.venv/bin/harbor run -p $P -e $E "${OPENCODE[@]}" -k 3 -n 3 -o jobs --job-name cmp-opencode -y
.venv/bin/python harbor_env/summary.py jobs/cmp-*      # one table for all of them
```

## Read the results

**Did it pass?** In `summary.py` and in Harbor's table:

- **Bug fix:** `reward` 1 is a pass. It needs every check: the hidden tests (`verifiers_pass`), the validation agent's tests (`validation_score`), the rubric judge (`rubric_score` > 0.5), patch size (`patch_bloat` < 2, the agent's patch divided by the reference patch), and two taste judges (`practice_alignment`, `relative_taste` ≥ 0.6). `correct` 1 means the fix works, whatever the judges said. `agent_tests_*` describe the tests the agent wrote and do not count.
- **PR review:** `passed` 1 is a pass: the agent's verdict (it blocks when one of its findings is `high`) matches the real one, and a blocked change also needs every high defect found (`coverage_required` 1.0). No findings file is no review (`submitted` 0, verdict `not submitted`): the agent failed or put its review in its reply instead of the file, so check its log. On the approve task there is nothing to find, so `coverage_required` is 1.0 for everyone: read `passed`.

**Why not?** The sub-scores above say which check failed. `.venv/bin/harbor view jobs` shows what the agent did, and the trial's `verifier/` folder has the grader's files.

**No reward** is neither a pass nor a fail: the grader marked the trial invalid (`verifier/grading_invalid.txt` or `reward_details.json` says why), or the results did not reach your machine. Harbor's own table counts a missing value as 0.

The `grading` column should be `staged` (bug fix, [graded in stages](#how-it-works)) or `separate` (PR review, graded in its own microVM). `unsealed` means the task's `test.sh` has no `--stage`, so its reward is not isolated from the agent's code; `refused` means the trial was not graded.

## Bring your own tasks

Any Harbor task runs here if:

- its `environment/Dockerfile` is Debian- or Ubuntu-based and installs every dependency at build time: the agent and the grader have no internet, only the task's `allowed_hosts`;
- its compressed image is at most 10 GB;
- it is a single-step task, and its agent and verifier do not run as root;
- its `tests/test.sh` writes `/logs/verifier/reward.json` (or `reward.txt`).

Without `--stage`, grading runs in one stage (`unsealed`) and the reward is not isolated from the agent's code. For staged grading, use the sample tasks' `tests/test.sh` as a model. PR-review tasks can be made with [eval-harvest](https://github.com/aws-samples/sample-code-eval-harvester).

## Clean up

```bash
.venv/bin/python harbor_env/setup.py teardown     # runtimes, images, CodeBuild, IAM roles, S3 bucket, log groups (about 4 min)
.venv/bin/python harbor_env/setup.py resources    # must print empty lists
```

Teardown deletes only the resources this tool names for its prefix (`$P2_PREFIX`, default `evals-hb-`). It prints nothing while it waits for the runtimes to go. If a Harbor process hangs or is killed, `setup.py stop-sessions <trial dirs>` stops the sessions it left. Left on purpose: AgentCore's service-linked role `AWSServiceRoleForBedrockAgentCoreRuntimeIdentity`, your own role, and the local `jobs/` folder (agent and judge transcripts: read them before you share them).

## Cost and time

Measured in one account, in `us-east-1`. Dollar cost was not measured: it comes from model calls (agent and judges), CodeBuild, ECR storage, and AgentCore sessions.

| What | Time |
|---|---|
| `bootstrap` | about 15 s |
| Image build | verifier image about 45 s; review task image 2 to 3 min; bug-fix task image about 6 min |
| Runtime deploy, once per image | 3 to 4 min |
| `prebuild` of two tasks (3 images, in parallel) | about 10 min |
| Session start of a trial | 5 to 12 s |
| A review trial (Codex) | 2 to 4 min |
| A bug-fix trial (Codex) | 4 to 7 min, of which grading 2 to 3 min |
| `teardown` | about 4 min |

## Permissions

The person who runs `setup.py` and `harbor run` needs the actions below. `CreateRole` with `PutRolePolicy` and `PassRole` is close to administrator access: scope the IAM actions and `PassRole` to `role/evals-*`, and consider a permissions boundary. A role with exactly these actions, scoped to one prefix, ran the quick start and teardown with no AccessDenied; whether every action is needed was not checked.

| Service | Actions |
|---|---|
| IAM | `CreateRole`, `GetRole`, `TagRole`, `UpdateAssumeRolePolicy`, `PutRolePolicy`, `PassRole`, `DeleteRolePolicy`, `DeleteRole`, `ListRoles`, `ListRolePolicies`, `ListAttachedRolePolicies`, `DetachRolePolicy`; `CreateServiceLinkedRole` (first deploy in an account; condition `iam:AWSServiceName = runtime-identity.bedrock-agentcore.amazonaws.com`) |
| STS | `AssumeRole` on `<prefix>agent-role` |
| ECR | `CreateRepository`, `TagResource`, `DescribeRepositories`, `DescribeImages`, `DeleteRepository`, `BatchDeleteImage` |
| S3 | `CreateBucket`, `ListBucket`, `ListBucketVersions`, `PutBucketPublicAccessBlock`, `PutBucketTagging`, `PutLifecycleConfiguration`, `PutObject`, `PutObjectTagging`, `DeleteObject`, `DeleteObjectVersion`, `DeleteBucket`, `ListAllMyBuckets` |
| CloudWatch Logs | `CreateLogGroup`, `PutRetentionPolicy`, `TagResource`, `GetLogEvents`, `DescribeLogGroups`, `DeleteLogGroup` |
| CodeBuild | `CreateProject`, `UpdateProject`, `BatchGetProjects`, `StartBuild`, `BatchGetBuilds`, `DeleteProject`, `ListProjects` |
| Bedrock | `GetInferenceProfile` |
| AgentCore | `CreateAgentRuntime`, `UpdateAgentRuntime`, `GetAgentRuntime`, `ListAgentRuntimes`, `DeleteAgentRuntime`, `TagResource`, `CreateAgentRuntimeEndpoint`, `CreateWorkloadIdentity`, `DeleteAgentRuntimeEndpoint`, `DeleteWorkloadIdentity`, `InvokeAgentRuntimeCommand`, `StopRuntimeSession` |

The roles that `bootstrap` creates, and what they may do:

- `<prefix>agent-role` (the agent): only the `--agent-model` IDs, as `bedrock-mantle:CreateInference` for `openai.*` models and `bedrock:InvokeModel*` on the inference profile and its foundation models for the others. `setup.py policies --agent-model <ID>` prints it.
- `<prefix>grader-role` (the AgentCore execution role, used for grading): `bedrock:InvokeModel` on the judge models, ECR pull on one repository, and log writes to AgentCore runtime log groups.
- `<prefix>codebuild-role`: builds the images.

## How it works

- **Sandbox.** The agent runs as an unprivileged user in an empty network namespace whose only exit is a proxy that allows the task's `allowed_hosts` (no GitHub, PyPI, or metadata service). Its credentials, in a root-only file, can call only the agent models. The hidden tests and the reference solution are not in its image, and the repository has one commit and no remote. `-a harbor_env.probe_agent:ProbeAgent` checks all of this from the agent's seat (its output: `agent/probe-host-copy.txt`).
- **PR-review grading** runs in a new microVM built from the task's `tests/`, after the agent's session stops. It reads the agent's findings file as data.
- **Bug-fix grading** runs in the agent's microVM, in sealed stages:
  1. Every agent process is killed (no grading if one survives), and the credentials switch to the grader role.
  2. Root removes compiled Python caches, refuses a nested git repository, and records the agent's patch.
  3. `test.sh --stage untrusted` (the hidden tests, the agent's tests, the validation agent: everything that runs the agent's code) runs as the agent's user.
  4. Every agent process is killed again. If the repository changed, the trial is invalid.
  5. Root copies that stage's regular files, never a reward file.
  6. `test.sh --stage trusted` (the judges and the final score, which only read files) runs as a separate `grader` user on those copies. The judges see root's patch.
  7. Root rebuilds `/logs/verifier`; only the trusted stage's `reward.json` can become the reward.
- **Refused:** multi-step tasks, tasks whose agent or verifier runs as root, and a microVM without kernel link protection.
- **Runtimes** use AgentCore platform version V2 (images up to 10 GB; an image over 2 GB gets more build memory).

## Known limits

- **Pre-release:** tested with the four sample tasks, in one AWS account, in `us-east-1`, with Harbor 0.26.0. Dollar cost was not measured.
- **Isolation of test results:** the reward file and the judges are isolated from the agent's code, but the hidden-test and validation results (`junit.xml`, `validation.json`) are produced in the same stage as the agent's code. The scores assume the agent under test acts in good faith.
- **Designed for evaluating coding agents:** the sandbox isolates each trial's network and credentials, but it is not a security boundary for running untrusted code.
- **Trial folders** hold the agent's and the judges' transcripts and logs. Review them before you share them.

## Modify and verify

If you change the environment, the graders, or the tasks, check these:

- Unit tests (no AWS): `uv pip install --python .venv/bin/python pytest && AWS_REGION=us-east-1 .venv/bin/python -m pytest -q -p no:cacheprovider harbor_env/tests`
- The task graders share files in `datasets/_shared/`: edit them there, then run `python3 datasets/_shared/sync.py` (and `--check`).
- On AWS, after a change to `harbor_env/`, `agentcore_aws.py`, `image/`, or a grader: `oracle` (1) and `nop` (0) on all four tasks; the probe; the grading isolation test agent on `pr371` (`-a harbor_env.tests.tamper_agent:TamperAgent`, four variants: `--ak keep_fds=true`, none, `--ak stage_out=invalid`, `--ak stage_out=changed`; each ends with reward 0 or no reward); one trial per harness; then `teardown` and `resources`.
- Before a release, check the quick start with a role that has only the [Permissions](#permissions) above.
