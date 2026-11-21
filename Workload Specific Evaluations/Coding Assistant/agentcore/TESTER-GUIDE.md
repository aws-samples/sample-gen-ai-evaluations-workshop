# Tester guide: AgentCore mode

You are testing a way to run the Coding Assistant module on AWS only. This guide says what you need, what it costs, how to run a first test, how to clean up, what is known to be weak, and what feedback we want. [README.md](README.md) has the four steps and the design.

## 1. Prerequisites

| Need | Detail |
|---|---|
| Python | 3.10 or later (tested with 3.12) |
| boto3 | **1.44.7 or later**. Version 1.43.74 has no `platformVersion` for AgentCore. The runner checks this first and stops with `p2: boto3 ... is too old` before any AWS call (an older runner failed only at `deploy`, after the 2 to 6 minute image build). The default `python3` of your machine may be an old one: use the venv's `.venv/bin/python`. The venv line in the README installs the right version |
| AWS account | One where you may create IAM roles, ECR, S3, CodeBuild, and AgentCore runtimes. Use a sandbox account, not a shared production one |
| Region | `us-east-1` only (set in `p2_v2.py`, and in the module's task files) |
| Credentials | In the terminal: `aws sso login`, `AWS_PROFILE=...`, or access keys. The runner uses the standard boto3 credential chain |
| Models | Codex model `openai.gpt-6-sol` on the Bedrock Mantle endpoint (the agent), and `us.anthropic.claude-haiku-5-5` (the judges). Both IDs are read from `../model_config.py`. Your account must be allowed to use them in `us-east-1` |
| Not needed | Docker, Harbor, Node, the Codex CLI, the AWS CLI (only `scripts/teardown.sh` uses it), CloudFormation |

## 2. Permissions the caller needs

The permissions below are what you, the person who runs `p2_v2.py`, need. They are taken from the API calls in `p2_v2.py` (`bootstrap`, `build`, `deploy`, `run`, `cleanup`) and from `scripts/teardown.sh`. **They were not tested with a restricted caller.** All runs so far used an administrator-level caller. If a call is denied, send us the action name from the error (see the feedback template). An account that has no AgentCore service-linked role yet may also need `iam:CreateServiceLinkedRole`.

| Service | Actions | Used by |
|---|---|---|
| IAM | `CreateRole`, `GetRole`, `TagRole`, `UpdateAssumeRolePolicy`, `PutRolePolicy`, `PassRole`, `DeleteRolePolicy`, `DeleteRole` | bootstrap, deploy, cleanup. Roles: `<prefix>codebuild-role`, `<prefix>grader-role`, `<prefix>agent-role` |
| STS | `AssumeRole` on `<prefix>agent-role` (the runner mints the agent phase credentials) | bootstrap, run |
| ECR | `CreateRepository`, `TagResource`, `DescribeRepositories`, `DescribeImages`, `DeleteRepository`, `BatchDeleteImage` | bootstrap, all, cleanup |
| S3 | `CreateBucket`, `ListBucket`, `PutBucketPublicAccessBlock`, `PutBucketTagging`, `PutLifecycleConfiguration`, `PutObject`, `PutObjectTagging`, then `DeleteObject`, `DeleteBucket`, `ListAllMyBuckets` | bootstrap, build, teardown.sh |
| CloudWatch Logs | `CreateLogGroup`, `PutRetentionPolicy`, `TagResource`, `GetLogEvents`, `DescribeLogGroups`, `DeleteLogGroup` | bootstrap, build (failure tail), teardown.sh |
| CodeBuild | `CreateProject`, `UpdateProject`, `BatchGetProjects`, `StartBuild`, `BatchGetBuilds`, `DeleteProject`, `ListProjects` | bootstrap, build, cleanup |
| Bedrock | `GetInferenceProfile` (optional: without it a built-in list of the judge profile's regions is used) | bootstrap |
| AgentCore control plane | `CreateAgentRuntime`, `UpdateAgentRuntime`, `GetAgentRuntime`, `ListAgentRuntimes`, `DeleteAgentRuntime`, `TagResource` | deploy, all, cleanup |
| AgentCore data plane | `InvokeAgentRuntimeCommand`, `StopRuntimeSession` | run |
| Resource Groups Tagging | `tag:GetResources` | teardown.sh check |

The caller does not call the Codex or judge models itself. The two roles that the runner creates do:

- `<prefix>agent-role`: `bedrock-mantle:CreateInference` for the Codex model only.
- `<prefix>grader-role` (the AgentCore execution role): `bedrock:InvokeModel` for the judge model only, ECR pull on one repository, and log writes.

`p2_v2.py policies` prints the three policies that bootstrap installs, with the account id masked.

## 3. Cost and time

All numbers are from the runs of the authors, us-east-1, list prices, with +-30 % uncertainty on the dollar figures.

| What | Time | Cost |
|---|---|---|
| First `all` (bootstrap, build, deploy) | 14 s bootstrap, 2 to 6 min build (images build in parallel; pr4121 about 2 min, pr371 about 6 min), 10 s deploy | CodeBuild about $0.12 per image (ARM, large) |
| One PR review trial (`pr4121`, `pr4615`) | about 40 s | about $0.13 (the agent) |
| One coding trial (`pr371`, `pr338`) | 6 to 7 min (agent 3 to 4 min, grader 1.5 to 2.5 min) | about $0.30 to $0.45 (the agent) plus $0.03 to $0.04 (the judges) |
| Many trials at once | 12 or 24 trials at once took 10 % to 20 % longer than one trial. 30 at once ran without throttling | scales with the number of trials |
| Idle, between runs | none | the ECR images (about 7 GB for four tasks): about $0.02 per day. Runtimes cost nothing while idle |
| A first test | about 5 min | **under $0.50** (`all pr4121:1`) |
| A full first run (`all pr371:3 pr4121:3 pr4615:3`) | about 13 min (images build in parallel, the longest takes 6 min; then the trials run at once) | about $3 to $4 |

## 4. Smoke test first

The cheapest rung is one PR review trial. It builds one image, deploys one runtime, runs the agent for about 40 seconds, and grades it.

```bash
.venv/bin/python p2_v2.py all pr4121:1
```

The build is the slow step: no output of its own for the first 2 to 6 minutes except `<task> build started`. If it fails, the runner prints the reason and exits 1 (see section 6, Failures).

Expected (4 min 17 s on the first run of the authors): the log lines `bootstrap done in ~13s, 8 steps`, then `pr4121 build SUCCEEDED`, `pr4121 runtime READY`, and a table with one row. The row has `coverage_required` 1 and `coverage_all` 1 in nearly every trial (this approve task passed in 21 of 21 earlier trials), and the `errors` column is empty. A file `jobs/<time>/summary.md` has the same table. Then try one coding trial: `all pr371:1` (it builds a second image, 6 min).

Two checks that cost almost nothing and prove the grader and the roles:

```bash
.venv/bin/python p2_v2.py run pr371:1 --agent empty        # no agent: an unchanged repository must score reward 0
.venv/bin/python scripts/scope_test_v2.py                  # grader role: judge model ok, other models AccessDenied; agent role: no bedrock-runtime access
```

## 5. Clean up and check that nothing is left

```bash
PYTHON=.venv/bin/python scripts/teardown.sh                # default prefix evals-p2b-; give another as the first argument
```

It runs `p2_v2.py cleanup` (runtimes, ECR repository with its images, CodeBuild project, three roles), then deletes the S3 bucket and the log groups of the prefix, then lists what is left. Every list in the last section must be empty, with two exceptions. A runtime shows `DELETING` for a few minutes before it is gone (run `p2_v2.py status` again later). And the tag query lists every resource with the tag, including resources of other prefixes and the runtimes that are still deleting; look for your prefix. If your shell may not delete S3 objects or log groups, run `scripts/teardown.sh evals-p2b- --print` and give the printed commands to someone who may.

Everything the runner creates is tagged `Project=evals-streamline-spike` and has the name prefix (`evals-p2b-`, or your `--prefix`). To check by hand:

```bash
aws resourcegroupstaggingapi get-resources --tag-filters Key=Project,Values=evals-streamline-spike --region us-east-1 --query 'ResourceTagMappingList[].ResourceARN'
aws iam list-roles --query 'Roles[?starts_with(RoleName, `evals-p2`)].RoleName'
```

The `Expiry` tag is a date two weeks out (or `$P2_EXPIRY`). Nothing deletes a resource on that date. Only the S3 build contexts under `p2/` expire after 3 days, and the log groups keep 3 days.

## 6. Known limits

- **Public network mode.** The runtimes use `networkMode: PUBLIC`, with no VPC. The only control over what the agent can reach is the in-session firewall (empty network namespace, allowlist proxy). Outside the firewall, the microVM has internet access.
- **Scoped roles.** Each phase has a role for one model. The grader role is the AgentCore execution role, so the grader phase runs with it. The verifier still runs agent-written code as an unprivileged user holding the judge-only credentials. Changing `JUDGE_MODEL_ID` or `VERIFIER_MODEL_ID` in `model_config.py` changes the policy only at the next `bootstrap` (run `p2_v2.py bootstrap`; `all` bootstraps only when something is missing).
- **Tampering.** A patch that plants a force-pass hook inside the package source (tamper "H") is not blocked by the optional `--variant hard`. The validation agent and its judge catch it in the tests so far (reward 0 or an invalid trial), but only one fake agent per tamper was tried. The default path (the module's `tests/test.sh`) also lets a planted `conftest.py` through stage 1.
- **The firewall is not a certified boundary.** 43 scripted probes and two LLM red-team runs (7 and 8 turns), all with honest QA framing, found no break-out. A third run was stopped by the Bedrock content filter. Get an expert review before you run an agent that is trying to escape.
- **Scale.** 30 trials at once ran without throttling and with 2 to 7 % of sessions failing at start or read-back (`RuntimeClientError` 500, truncated archive); those trials show as invalid or with an `errors` entry. Larger batches were not tested. The `--variant par` option adds 2 more concurrent judge calls per trial and was not tested at scale.
- **No fresh-account test.** All runs used one account that had AWS prerequisites from earlier work. Bootstrap from an empty prefix was tested there, not in a different account, and not with a restricted caller (see section 2).
- **PR review `pr4615` (reject).** It passes about 12 % of the time with Codex (23 of 185 trials, 95 % interval 8 % to 18 %), at 1, 5, and 30 trials at once. That is judge and agent variance, not an environment bug: no difference between this mode and the Docker mode was found that would explain a different rate (the Docker mode had 0 of 3 trials passing). Do not report a low pass rate on this task as a bug, report a rate that differs a lot from 12 % over 20 or more trials.
- **Coding rewards.** Judge fields (`relative_taste`, `validation_score`) vary between runs of the same patch: one reward 0.0 in a trial came from a hidden test that the agent's patch broke, another from `relative_taste` 0.2.
- **Models and Region.** One agent (Codex) and `us-east-1`. The Claude Code swap of the module's notebooks is not part of this mode.
- **Failures.** A failed image build prints the last 80 log lines, then `p2: image build failed: <task>: ERROR: ... (BUILD phase)`, exits 1, and nothing is deployed or run (checked with a deliberately broken Dockerfile; the failing build takes about 2 minutes to report). `build` and `deploy` also exit 1 on failure. `run` on a task that is not deployed says `p2: no AgentCore runtime ... Run 'deploy <task>' or 'all' first`. Ctrl-C during `run` stops the sessions and exits with code 130 (tested with fake sessions only). `bootstrap` can be interrupted and re-run at any point. `cleanup` continues past a runtime that is still CREATING, UPDATING or DELETING, prints `FAILED runtime ...`, and exits 1: wait a few minutes (a delete takes 3 to 7 minutes) and run it again. A command that runs longer than its timeout (1500 s for the agent) returns exit code -1 in `result.json` (`agent_exit`), with no entry in the `errors` column. Not tested: a caller with restricted permissions, `cleanup` while a runtime is CREATING.

## 7. What feedback we want

Things we cannot see from our own account:

1. Did `all pr4121:1` work the first time, from a clean checkout? Where did it fail, and what was the error text?
2. Which IAM permissions were missing for your caller (section 2)? Which were not needed?
3. Wall time of bootstrap, of the image builds, and of a trial in your account, and the Region/account type (sandbox, shared, enterprise with SCPs).
4. Anything the guide got wrong or left out. Steps that you had to guess.
5. Trials that are invalid, slow, or throttled at your batch size.
6. Anything left behind after `scripts/teardown.sh`.
7. Whether you would use this mode over the Docker/Harbor mode, and why not.

Send it as an issue on the repository or as a message to the owner, with this template:

```markdown
**Mode:** AgentCore (agentcore/)    **Branch/commit:** <sha>
**Caller:** <sso role / IAM user type; admin or restricted; SCPs?>    **Region:** us-east-1
**Python / boto3:** <3.x.y / 1.44.x>    **OS:** <macOS/Linux/Windows>

**Command(s) run:**
<paste, with the prefix>

**Result:** <worked / failed at step N / worked with changes>
**Wall time:** bootstrap <s>, build <min per task>, deploy <s>, trials <s>
**Error text (account ids removed):**
<paste>

**Missing IAM actions (from AccessDenied):**
<action names>

**Trial table** (jobs/<time>/summary.md, ids removed):
<paste>

**Left after teardown.sh:**
<paste the lists, or "nothing">

**README / guide problems:**
<what was wrong or missing>

**Other:**
```

Remove account ids, access keys, and session tokens before you send anything. The `jobs/` folder holds agent transcripts and judge transcripts; read it before you share any of it.
