#!/bin/bash
# Step 1, calibration: run each new task with Harbor's nop agent (changes nothing) and its oracle agent
# (applies the real change) on AgentCore, then let coding_tasks.py compare the two runs: a test that fails
# without the change and passes with it is fail-to-pass; a test that passes in both is pass-to-pass.
# Then run nop and oracle once more: a usable task scores 0 with nop and 1 with oracle.
#
#   step1-build-tasks/calibrate_tasks.sh <task dir> [<task dir> ...]
#
# It uses Step 2's AgentCore environment, so do Step 2's setup first: in step2-run-evals/, its .venv with
# Harbor and `harbor_env/setup.py bootstrap` (see step2-run-evals/README.md). Run from the Coding Assistant
# folder. STEP2 names Step 2's folder; by default step2-run-evals/, or this folder if harbor_env/ is here.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '7p' "$0" | sed 's/^# *//' >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${STEP2:-}" ]; then
  if [ -d step2-run-evals/harbor_env ]; then STEP2=step2-run-evals; else STEP2=.; fi
fi
STEP2=$(cd "$STEP2" && pwd)
[ -x "$STEP2/.venv/bin/harbor" ] && [ -d "$STEP2/harbor_env" ] \
  || { echo "no Step 2 setup in $STEP2: make its .venv with Harbor and run bootstrap first" >&2; exit 2; }
TASKS=()
for TASK in "$@"; do TASKS+=("$(cd "$TASK" && pwd)"); done
JOBS=$(mkdir -p "${JOBS:-$HERE/work/jobs}" && cd "${JOBS:-$HERE/work/jobs}" && pwd)

# Harbor and harbor_env run from Step 2's folder, as its README runs them.
cd "$STEP2"
export PYTHONPATH=$PWD AWS_REGION=${AWS_REGION:-us-east-1}
unset AWS_BEARER_TOKEN_BEDROCK
E=harbor_env.agentcore_env:AgentCoreEnvironment

# One image per task, built once on CodeBuild (this is the slow part: several minutes per task).
.venv/bin/python harbor_env/setup.py prebuild "${TASKS[@]}" > /dev/null

run() {  # <task> <agent> <job name>
  rm -rf "${JOBS:?}/$3"
  .venv/bin/harbor run -p "$1" -e $E -a "$2" -o "$JOBS" --job-name "$3" -y > /dev/null 2>&1 || true
}

for TASK in "${TASKS[@]}"; do
  NAME=$(basename "$TASK")
  echo "== $NAME"
  # Before calibration the task's grader records every test's status and gives no reward, so Harbor
  # reports these two trials without one. That is expected.
  run "$TASK" nop "$NAME-cal-nop"
  run "$TASK" oracle "$NAME-cal-oracle"
  .venv/bin/python "$HERE/coding_tasks.py" calibrate "$TASK" --nop "$JOBS/$NAME-cal-nop" --oracle "$JOBS/$NAME-cal-oracle" \
    || { echo "   not a usable task"; continue; }
  run "$TASK" nop "$NAME-nop"
  run "$TASK" oracle "$NAME-oracle"
  .venv/bin/python "$HERE/check_gate.py" "$JOBS/$NAME-nop" "$JOBS/$NAME-oracle"
done
