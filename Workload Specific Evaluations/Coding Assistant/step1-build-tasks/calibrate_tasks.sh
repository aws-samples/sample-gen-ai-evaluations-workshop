#!/bin/bash
# Step 1, calibration: run each new task with Harbor's nop agent (changes nothing) and its oracle agent
# (applies the real change) on AgentCore, then let coding_tasks.py compare the two runs: a test that fails
# without the change and passes with it is fail-to-pass; a test that passes in both is pass-to-pass.
# Then run nop and oracle once more: a usable task scores 0 with nop and 1 with oracle.
#
#   step1-build-tasks/calibrate_tasks.sh <task dir> [<task dir> ...]
#
# It uses Step 2's AgentCore environment, so do Step 2's setup first (the .venv with Harbor, and
# `harbor_env/setup.py bootstrap`). Run from the Coding Assistant folder.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '7p' "$0" | sed 's/^# *//' >&2; exit 2; }
export PYTHONPATH=$PWD AWS_REGION=${AWS_REGION:-us-east-1}
unset AWS_BEARER_TOKEN_BEDROCK
E=harbor_env.agentcore_env:AgentCoreEnvironment
JOBS=${JOBS:-step1-build-tasks/work/jobs}

# One image per task, built once on CodeBuild (this is the slow part: several minutes per task).
.venv/bin/python harbor_env/setup.py prebuild "$@" > /dev/null

run() {  # <task> <agent> <job name>
  rm -rf "$JOBS/$3"
  .venv/bin/harbor run -p "$1" -e $E -a "$2" -o "$JOBS" --job-name "$3" -y > /dev/null 2>&1 || true
}

for TASK in "$@"; do
  NAME=$(basename "$TASK")
  echo "== $NAME"
  # Before calibration the task's grader records every test's status and gives no reward, so Harbor
  # reports these two trials without one. That is expected.
  run "$TASK" nop "$NAME-cal-nop"
  run "$TASK" oracle "$NAME-cal-oracle"
  .venv/bin/python step1-build-tasks/coding_tasks.py calibrate "$TASK" --nop "$JOBS/$NAME-cal-nop" --oracle "$JOBS/$NAME-cal-oracle" \
    || { echo "   not a usable task"; continue; }
  run "$TASK" nop "$NAME-nop"
  run "$TASK" oracle "$NAME-oracle"
  .venv/bin/python step1-build-tasks/check_gate.py "$JOBS/$NAME-nop" "$JOBS/$NAME-oracle"
done
