#!/bin/bash
# Step 1: coding tasks from a repository's merged pull requests, with eval-harvest.
#
#   step1-build-tasks/build_coding_tasks.sh <git URL or local clone> <PR number> [<PR number> ...]
#
# For each pull request, eval-harvest writes one Harbor task: the repository at the commit before the
# change, the tests the pull request added or changed (hidden from the agent until grading), and the
# change itself as the reference solution. The tasks grade in stages (tests/test.sh --stage), the way
# Step 2 runs every coding task. Run from the Coding Assistant folder.
#
# Settings (environment variables):
#   PROFILE      how the repository's tests run (default: step1-build-tasks/coding-profile.toml)
#   DATASET      where the tasks go (default: step1-build-tasks/work/dataset); they land in $DATASET/tasks/
#   EVAL_HARVEST where eval-harvest comes from (default: the aws-samples repository)
set -euo pipefail
[ $# -ge 2 ] || { sed -n '3,4p' "$0" | sed 's/^# *//' >&2; exit 2; }
SOURCE=$1
shift
PROFILE=${PROFILE:-step1-build-tasks/coding-profile.toml}
DATASET=${DATASET:-step1-build-tasks/work/dataset}
EVAL_HARVEST=${EVAL_HARVEST:-git+https://github.com/aws-samples/sample-code-eval-harvester}
eval_harvest() { uvx --quiet --python 3.12 --from "$EVAL_HARVEST" eval-harvest "$@"; }

mkdir -p "$DATASET"
cp "$PROFILE" "$DATASET/coding-profile.toml"

# The repository: a local clone as it is, or a fresh clone of a URL (eval-harvest reads git history
# and never fetches).
if [ -d "$SOURCE/.git" ]; then
  CLONE=$SOURCE
else
  CLONE=step1-build-tasks/work/repos/$(basename "$SOURCE" .git)
  [ -d "$CLONE/.git" ] || git clone --quiet "$SOURCE" "$CLONE"
fi

for PR in "$@"; do
  echo "== PR #$PR"
  # A pull request that cannot be a coding task (no test change, only tests, no runnable test file)
  # is refused with the reason; the others still run.
  eval_harvest code --pr "$PR" --clone "$CLONE" --dataset "$DATASET" || echo "   skipped: see the reason above"
done
echo
echo "tasks in $DATASET/tasks:"
ls "$DATASET/tasks" 2>/dev/null | sed 's/^/  /'
echo "next: step1-build-tasks/calibrate_tasks.sh $DATASET/tasks/<task>   (finds which hidden tests the change makes pass)"
