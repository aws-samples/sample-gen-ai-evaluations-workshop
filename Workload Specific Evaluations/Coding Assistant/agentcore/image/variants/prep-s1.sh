#!/bin/bash
# Root. Prepares the tree stage 1 will see at /work/repo (bind-mounted in a private mount namespace by orchestrate.sh).
#  default (P2_HARD=0): a copy of the agent's tree (the V-par isolation: stage 2 changes the real tree while stage 1 runs).
#  P2_HARD=1: copy of the sealed repo (/opt/hidden/sealed-repo, base commit), then the agent's patch applied with git apply, MINUS
#             the paths below; the hidden test file added; the tree is root-owned and read-only for the agent.
# Root never runs git inside the agent's repository (its .git/config can name programs to run); it only reads $L/stage1.patch as data.
set -eu
L=/logs/verifier; V=/opt/verify
HIDDEN=$(grep -m1 '^HIDDEN=' /tests/test.sh | cut -d= -f2)   # the hidden test file name, as the task's own test.sh sets it
BASE=$(tr -d '[:space:]' < /var/lib/task_base_ref)
rm -rf $V; install -d -m 0755 -o root -g root $V
if [ "${P2_HARD:-0}" != 1 ]; then
  cp -a /work/repo $V/repo
  exit 0
fi
cp -a /opt/hidden/sealed-repo $V/repo
cd $V/repo
[ -f $L/stage1.patch ] && [ ! -L $L/stage1.patch ] || { echo "no stage1.patch"; exit 1; }
cp $L/stage1.patch $V/stage1.patch
# Excluded from stage 1 (reported, never silent): everything under tests/ (stage 1 restores the base tests anyway), and anything pytest or
# Python loads implicitly: conftest.py, sitecustomize/usercustomize, .pth files, pytest/tox/setup.cfg config files.
EXC=(--exclude='tests/*' --exclude='*conftest.py' --exclude='*sitecustomize.py' --exclude='*usercustomize.py' --exclude='*.pth'
     --exclude='pytest.ini' --exclude='*/pytest.ini' --exclude='.pytest.ini' --exclude='tox.ini' --exclude='*/tox.ini' --exclude='setup.cfg' --exclude='*/setup.cfg')
: > $V/all.txt; : > $V/applied.txt
if [ -s $V/stage1.patch ]; then   # an empty patch (the agent changed nothing) leaves the sealed tree as it is
  git -c core.fsmonitor= apply --numstat $V/stage1.patch | cut -f3 | sort > $V/all.txt
  git -c core.fsmonitor= apply --numstat "${EXC[@]}" $V/stage1.patch | cut -f3 | sort > $V/applied.txt
  git -c core.fsmonitor= apply --whitespace=nowarn "${EXC[@]}" $V/stage1.patch
fi
{
  echo "V-hard stage 1 tree: sealed repo at $BASE + agent patch"
  echo "patch touches $(wc -l < $V/all.txt) path(s); applied $(wc -l < $V/applied.txt)"
  echo "--- applied:"; cat $V/applied.txt
  echo "--- NOT applied (excluded by rule):"; comm -23 $V/all.txt $V/applied.txt
} > $V/hard_report.txt
cp /tests/verify/$HIDDEN tests/
git -c core.fsmonitor= ls-tree -r --name-only "$BASE" tests/ | grep -E '/test_[^/]*\.py$' > $V/base_tests.txt
chown -R root:root $V; chmod -R go-w,a+rX $V
