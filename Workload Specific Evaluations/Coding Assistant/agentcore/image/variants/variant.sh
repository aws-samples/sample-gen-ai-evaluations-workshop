#!/bin/bash
# The stages of tests/test.sh (datasets/_shared/coding/tests/test.sh), split into steps so that orchestrate.sh can run stage 1 elsewhere
# and the LLM stages concurrently. The module's test.sh is not edited; this is a copy of its commands. Runs as the `agent` uid.
#   variant.sh stage0 | s1 | rest | aggregate      env: P2_PAR (concurrent LLM stages), P2_HARD (stage 1 on the prepared tree)
set -u
L=/logs/verifier
HIDDEN=$(grep -m1 '^HIDDEN=' /tests/test.sh | cut -d= -f2)   # the hidden test file name, as the task's own test.sh sets it
BASE=$(tr -d '[:space:]' < /var/lib/task_base_ref)
P2_PAR=${P2_PAR:-0}; P2_HARD=${P2_HARD:-0}
cd /work/repo || exit 0

case "${1:-}" in
stage0)
  : > "$L/stages.log"
  echo "== 0. save the agent's change" | tee -a "$L/stages.log"
  git config user.email verifier@localhost; git config user.name verifier
  git add -A >/dev/null 2>&1
  git -c core.hooksPath=/dev/null commit -q --no-verify -m "agent changes" >/dev/null 2>&1
  AGENT=$(git rev-parse HEAD)
  git diff "$BASE" "$AGENT" > "$L/agent.patch"
  git diff "$BASE" "$AGENT" --binary > "$L/stage1.patch"
  git diff --stat "$BASE" "$AGENT" > "$L/agent.stat"
  ;;
s1)
  # stage 1: /work/repo is the prepared tree here (private mount namespace)
  echo "tree: owner=$(stat -c %U /work/repo) hard=$P2_HARD" > "$L/p2_s1_tree.txt"
  if [ "$P2_HARD" = 1 ]; then
    BASE_TESTS=$(cat /opt/verify/base_tests.txt)
  else
    git checkout "$BASE" -- tests 2>/dev/null
    cp "/tests/verify/$HIDDEN" tests/
    BASE_TESTS=$(git ls-tree -r --name-only "$BASE" tests/ | grep -E '/test_[^/]*\.py$')
  fi
  timeout 900 python -m pytest -p no:cacheprovider -q -o junit_family=xunit1 \
    --junitxml="$L/junit.xml" $BASE_TESTS "tests/$HIDDEN" > "$L/pytest.log" 2>&1
  tail -1 "$L/pytest.log" > "$L/.o1"
  ;;
rest)
  AGENT=$(git rev-parse HEAD)
  cd /tests/lib || exit 0
  python agent_tests.py "$BASE" "$AGENT" > "$L/.o2" 2>&1
  s3() { python validate.py 2>> "$L/validation.stderr" > "$L/.o3"; }
  s4r() { python judges.py rubric 2>> "$L/rubric.stderr" > "$L/.o4r"; }
  s4t() { python judges.py taste 2>> "$L/taste.stderr" > "$L/.o4t"; }
  if [ "$P2_PAR" = 1 ]; then s3 & s4r & s4t & wait; else s3; s4r; s4t; fi
  python judges.py bloat > "$L/.o4b"
  ;;
aggregate)
  {
    echo "== 1. pre-written verifiers"; cat "$L/.o1"
    echo "== 2. agent-written tests"; cat "$L/.o2"
    echo "== 3. validation agent"; cat "$L/.o3"
    echo "== 4. rubric judge, taste judge, patch bloat"; cat "$L/.o4r" "$L/.o4t" "$L/.o4b"
    echo "== 5. aggregate"
  } >> "$L/stages.log"
  cd /tests/lib || exit 0
  python aggregate.py "${HIDDEN%.py}" | tee -a "$L/stages.log"
  ;;
esac
exit 0
