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
set -u
L=/logs/verifier
mkdir -p "$L"
: > "$L/stages.log"
HIDDEN=test_prior_tool_results_verify.py
BASE=$(tr -d '[:space:]' < /var/lib/task_base_ref)
cd /work/repo || exit 0
stage() { echo "== $1" | tee -a "$L/stages.log"; }

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
cp "/tests/verify/$HIDDEN" tests/
BASE_TESTS=$(git ls-tree -r --name-only "$BASE" tests/ | grep -E '/test_[^/]*\.py$')
timeout 900 python -m pytest -p no:cacheprovider -q -o junit_family=xunit1 \
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
stage "4. rubric judge, taste judge, patch bloat"
python judges.py rubric 2>> "$L/rubric.stderr" | tee -a "$L/stages.log"
python judges.py taste 2>> "$L/taste.stderr" | tee -a "$L/stages.log"
python judges.py bloat | tee -a "$L/stages.log"
stage "5. aggregate"
python aggregate.py "${HIDDEN%.py}" | tee -a "$L/stages.log"
exit 0
