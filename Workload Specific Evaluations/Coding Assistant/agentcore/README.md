# Coding Assistant on Amazon Bedrock AgentCore Runtime (test mode)

This folder runs the datapoints of the Coding Assistant module without Docker, Harbor, or CloudFormation on your machine. One command creates what it needs in your AWS account, builds the task images on AWS CodeBuild, runs each trial in its own Amazon Bedrock AgentCore Runtime session, and prints a summary table. The runner reads the tasks and verifiers from `../datasets/`, the model IDs from `../model_config.py`, and the Codex settings from `../codex-bedrock.toml`. It uses `../coding_eval.py` only to recognize the module folder. Nothing in the module is copied or changed.

**Status: pre-release, for testers.** Read [TESTER-GUIDE.md](TESTER-GUIDE.md) first. It has the permissions, the cost, the smoke test, the cleanup, the known limits, and the feedback template.

## The four steps

Run them from this folder (`Workload Specific Evaluations/Coding Assistant/agentcore`).

```bash
python3 -m venv .venv && .venv/bin/pip install "boto3>=1.44.7"     # 1. Python 3.10 or later
# 2. MANUAL: put AWS credentials in the terminal (aws sso login, or AWS_PROFILE, or access keys); the Region is us-east-1
.venv/bin/python p2_v2.py all pr4121:1                               # 3. smoke test: bootstrap + build + deploy + 1 PR review trial (about 5 min)
.venv/bin/python p2_v2.py all pr371:3 pr4121:3 pr4615:3              # 4. a real run: three trials of each task, all at once
```

`all` creates what is missing, builds the images that are missing (2 to 6 minutes, in parallel), deploys one runtime per task, runs every trial at the same time, and writes `jobs/<time>/summary.md` with one row per trial. A second `all` on the same account starts its trials after about 10 seconds. When you are done, run [scripts/teardown.sh](scripts/teardown.sh).

More than 30 trials: run waves of 30 (recipe in [TESTER-GUIDE.md](TESTER-GUIDE.md) section 3). A trial with no reward is listed under the summary with its reason; rerun it with `run`.

The tasks are `pr4121` (PR review, approve), `pr4615` (PR review, reject), `pr371` and `pr338` (coding, bug fix). The agent under test is Codex on Amazon Bedrock. The judges are the ones of the module.

## What it does

```
your machine                         your AWS account (us-east-1)
------------                         ----------------------------------------------------------
p2_v2.py bootstrap  ---- API ---->   ECR repo, S3 bucket, CodeBuild project, 3 IAM roles (scoped)
p2_v2.py build      ---- zip ---->   S3 -> CodeBuild (arm64) -> ECR image = the task's own Dockerfile + a small layer
p2_v2.py deploy     ---- API ---->   one AgentCore runtime per task (PUBLIC network mode, no VPC)
p2_v2.py run        ---- API ---->   one session = one microVM per trial:

                                     +--------------------------------------------------------------+
                                     | 1. agent phase   Codex runs as user `agent` in an EMPTY      |
                                     |    network namespace. Its only way out is a relay on         |
                                     |    localhost to a unix socket. The allowlist proxy behind    |
                                     |    the socket forwards to the Codex model endpoint only.     |
                                     |    Credentials: the agent role (that model only).            |
                                     | 2. phase switch  agent processes are killed; the proxy        |
                                     |    allowlist changes to the judge endpoint; the hidden tests |
                                     |    are copied in (they are root-only until now)              |
                                     | 3. grader phase  the module's own tests/test.sh runs.        |
                                     |    Credentials: the grader role (the judge model only)       |
                                     +--------------------------------------------------------------+
                  <---- tar ---    /logs/verifier/reward.json, agent logs, artifacts
```

Four points:

- **One session for each trial.** AgentCore starts a fresh microVM (2 vCPU, 8 GB) for the session, and the session ends when the trial ends. Many trials run at the same time, so a batch takes as long as its slowest trial.
- **A firewall inside the session.** The microVM gives root no network-admin capability, so nftables and iptables do not load. Instead the agent runs in an empty network namespace, owned by its own user namespace. A relay inside that namespace and an allowlist proxy outside it (unix socket) are the only route out. `image/start.sh` and `image/proxy.py` are the whole firewall.
- **Two phases, two roles.** The agent phase has credentials for the Codex model and nothing else. The grader phase has credentials for the judge model and nothing else. The hidden tests are in the image (root, mode 0700) and become readable only after the agent's processes are killed.
- **No images on your machine.** The image of a task is the task's own `environment/Dockerfile`, unchanged, plus `image/layer.Dockerfile`. CodeBuild builds it and pushes it to your ECR repository.

## Files

```
agentcore/
├── README.md          this file
├── TESTER-GUIDE.md    prerequisites, permissions, cost, smoke test, cleanup, limits, feedback template
├── p2_v2.py           the runner (bootstrap, build, deploy, run, all, status, policies, cleanup)
├── image/             the files added to every task image
│   ├── layer.Dockerfile, layer_variants.Dockerfile, buildspec.yml
│   ├── main.py        the HTTP shim AgentCore needs on port 8080 (/ping, /invocations)
│   ├── start.sh, proxy.py, as-agent       the in-session firewall and the way to run a command as the agent
│   └── variants/      the optional grader variants (--variant par|hard), coding tasks only
└── scripts/
    ├── teardown.sh        cleanup of one prefix, and a check that nothing is left
    ├── scope_test_v2.py   negative test of the scoped roles, from inside a session
    └── test_runner_v2.py  unit tests of the runner (archive extraction, session-start retry, summary); no AWS call
```

`p2_v2.py` finds the module by itself: this folder is inside it. Run from elsewhere, give `--module "<repo>/Workload Specific Evaluations/Coding Assistant"` or set `P2_MODULE`. Results go to `jobs/` here (ignored by git).

Other commands: `p2_v2.py policies` prints the three IAM policies it installs. `p2_v2.py status` shows the runtimes. `p2_v2.py run pr371:2 --agent empty` runs a trial with no agent (the verifier then scores an unchanged repository, a check that the grader works). `--prefix NAME` (or `$P2_PREFIX`, default `evals-p2b-`) gives a second, independent copy of every resource.
