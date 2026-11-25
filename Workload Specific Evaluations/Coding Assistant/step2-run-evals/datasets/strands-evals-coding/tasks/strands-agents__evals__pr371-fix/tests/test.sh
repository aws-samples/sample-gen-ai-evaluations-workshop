#!/bin/bash
# Verifier entrypoint: the reward stack of a coding task, after Senior SWE-Bench. Harbor runs it in the
# agent's container after the agent stops, and reads only /logs/verifier/reward.json.
#
#   0. Save the agent's change.
#   1. Pre-written verifiers: the hidden tests and the base unit test suite.        (gate)
#   2. Agent-written tests: do the agent's own tests pass on its fix, fail on base?  (reported)
#   3. Validation agent: tests adapted to the change, from the validation spec.       (gate)
#   4. Task rubric judge, taste judge, and patch bloat.                               (thresholds)
#   5. Aggregate.
#
#   test.sh                      every stage in order, in /logs/verifier
#   test.sh --stage untrusted    stages 0-3 (they run code that the agent influenced); writes $OUT
#   test.sh --stage trusted      stages 4-5 (they read files and the patch as data); copies $IN, writes $OUT
# IN and OUT default to /logs/verifier.
set -u
MODE=all
[ "${1:-}" = --stage ] && MODE=${2:-}
case "$MODE" in all|untrusted|trusted) ;; *) echo "usage: test.sh [--stage untrusted|trusted]" >&2; exit 2 ;; esac
if [ "$MODE" = all ]; then L=/logs/verifier; IN=$L; else L=${OUT:-/logs/verifier}; IN=${IN:-/logs/verifier}; fi
export VERIFIER_LOGS="$L"
mkdir -p "$L"
HIDDEN=test_root_agent_verify.py
BASE=$(tr -d '[:space:]' < /var/lib/task_base_ref)
stage() { echo "== $1" | tee -a "$L/stages.log"; }

if [ "$MODE" != trusted ]; then
: > "$L/stages.log"
cd /work/repo || exit 0

stage "0. save the agent's change"
git config user.email verifier@localhost; git config user.name verifier
git add -A >/dev/null 2>&1
git -c core.hooksPath=/dev/null commit -q --no-verify -m "agent changes" >/dev/null 2>&1
AGENT=$(git rev-parse HEAD)
git diff "$BASE" "$AGENT" > "$L/agent.patch"
git diff --stat "$BASE" "$AGENT" > "$L/agent.stat"

stage "1. pre-written verifiers"
# Restore the base test files, so that the agent cannot change the tests that pass on the base. Then add
# the hidden tests and run all of them.
git checkout "$BASE" -- tests 2>/dev/null
# Base versions of the files that pytest or Python load on their own (the agent's tree comes back below).
git diff --name-only "$BASE" "$AGENT" \
  | grep -E '(^|/)(conftest\.py|sitecustomize\.py|usercustomize\.py|pytest\.ini|\.pytest\.ini|tox\.ini|setup\.cfg)$|\.pth$' \
  | while IFS= read -r f; do git checkout "$BASE" -- "$f" 2>/dev/null || rm -f "$f"; done
# pytest settings from the base pyproject.toml, not the agent's.
CFG=${BASE_PYTEST_CONFIG:-}
if [ -z "$CFG" ] || [ ! -f "$CFG" ]; then
  CFG=/tmp/base-pytest-config/pyproject.toml
  mkdir -p /tmp/base-pytest-config && git show "$BASE:pyproject.toml" > "$CFG" 2>/dev/null
fi
CFG_ARGS=()
grep -q '^\[tool\.pytest\.ini_options\]' "$CFG" 2>/dev/null && CFG_ARGS=(-c "$CFG" --rootdir /work/repo)
cp "/tests/verify/$HIDDEN" tests/
BASE_TESTS=$(git ls-tree -r --name-only "$BASE" tests/ | grep -E '/test_[^/]*\.py$')
PYTHONNOUSERSITE=1 timeout 900 python -m pytest -p no:cacheprovider ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} -q -o junit_family=xunit1 \
  --junitxml="$L/junit.xml" $BASE_TESTS "tests/$HIDDEN" > "$L/pytest.log" 2>&1
tail -1 "$L/pytest.log" | tee -a "$L/stages.log"
# Put the agent's tree back for the next stages.
rm -f "tests/$HIDDEN"
git checkout "$AGENT" -- . 2>/dev/null
git reset -q "$AGENT"

cd /tests/lib || exit 0
stage "2. agent-written tests"
python agent_tests.py "$BASE" "$AGENT" >> "$L/stages.log" 2>&1
stage "3. validation agent"
python validate.py 2>> "$L/validation.stderr" | tee -a "$L/stages.log"
fi
[ "$MODE" = untrusted ] && exit 0

if [ "$MODE" = trusted ]; then
  # The untrusted stage's files, minus the names this stage writes.
  [ "$IN" = "$L" ] || cp -r "$IN"/. "$L"/
  for f in rubric.json taste.json bloat.json rubric.stderr taste.stderr rubric_judge_transcript.jsonl \
           taste_judge_transcript.jsonl reward.json reward.txt reward_details.json failed_tests.txt; do rm -f "$L/$f"; done
  # Import nothing from the agent's source tree.
  SITE=$(echo "${VIRTUAL_ENV:-/opt/venv}"/lib/python3*/site-packages)
  python() { PYTHONPATH="$SITE" PYTHONNOUSERSITE=1 command python -S -s "$@"; }
fi
cd /tests/lib || exit 0
stage "4. rubric judge, taste judge, patch bloat"
python judges.py rubric 2>> "$L/rubric.stderr" | tee -a "$L/stages.log"
python judges.py taste 2>> "$L/taste.stderr" | tee -a "$L/stages.log"
python judges.py bloat | tee -a "$L/stages.log"
stage "5. aggregate"
python aggregate.py "${HIDDEN%.py}" | tee -a "$L/stages.log"
exit 0
