#!/bin/bash
# Step 1: coding tasks from a repository's merged pull requests.
#
#   step1-build-tasks/build_coding_tasks.sh <git URL or local clone> <PR number> [<PR number> ...]
#
# For each pull request, coding_tasks.py (in this folder, standard-library Python) writes one Harbor task:
# the repository at the commit before the change, the tests the pull request added or changed (hidden from
# the agent until grading), and the change itself as the reference solution. The tasks grade in stages
# (tests/test.sh --stage), the way harbor_env runs every coding task. Run from the Coding Assistant folder.
#
# Settings (environment variables):
#   PROFILE   how the repository's tests run (default: step1-build-tasks/coding-profile.toml)
#   DATASET   where the tasks go (default: step1-build-tasks/work/dataset); they land in $DATASET/tasks/
#   PYTHON    a Python 3.11 or later (default: Step 2's .venv/bin/python, else python3)
set -euo pipefail
[ $# -ge 2 ] || { sed -n '4p' "$0" | sed 's/^# *//' >&2; exit 2; }
SOURCE=$1
shift
PROFILE=${PROFILE:-step1-build-tasks/coding-profile.toml}
DATASET=${DATASET:-step1-build-tasks/work/dataset}
PYTHON=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "needs Python 3.11 or later (tomllib): make Step 2's .venv, or set PYTHON" >&2; exit 2; }

# The repository: a local clone as it is, or a fresh clone of a URL (the builder reads git history and
# never calls the GitHub API).
if [ -d "$SOURCE/.git" ]; then
  CLONE=$SOURCE
else
  CLONE=step1-build-tasks/work/repos/$(basename "$SOURCE" .git)
  [ -d "$CLONE/.git" ] || git clone --quiet "$SOURCE" "$CLONE"
fi

ARGS=()
for PR in "$@"; do ARGS+=(--pr "$PR"); done
# A pull request that cannot be a coding task (no test change, only tests, no runnable test file) is
# refused with the reason; the others are still written.
"$PYTHON" step1-build-tasks/coding_tasks.py build --clone "$CLONE" --profile "$PROFILE" --out "$DATASET" "${ARGS[@]}" || true
echo
echo "tasks in $DATASET/tasks:"
ls "$DATASET/tasks" 2>/dev/null | sed 's/^/  /'
echo "next: step1-build-tasks/calibrate_tasks.sh $DATASET/tasks/<task>   (finds which hidden tests the change makes pass)"
