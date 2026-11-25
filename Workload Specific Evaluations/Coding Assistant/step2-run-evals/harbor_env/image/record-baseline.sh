#!/bin/bash
# At image build, as root: record the tree as built as refs/hb/baseline, so an agent that changes nothing has an empty patch.
#   record-baseline.sh REPO GIT_DIR
set -eu
REPO=$1 GD=$2
D=$(mktemp -d)
g() { GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null HOME=/nonexistent GIT_INDEX_FILE="$D/index" \
      git --no-replace-objects --git-dir="$GD" --work-tree="$REPO" -c core.hooksPath=/dev/null -c core.fsmonitor=false "$@"; }
cd "$REPO"
g read-tree HEAD
g add -A --force -- . ':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/*.pyc' ':(exclude,glob)**/.pytest_cache/**'
g update-ref refs/hb/baseline "$(g write-tree)"
rm -rf "$D"
echo "baseline tree $(g rev-parse refs/hb/baseline): $(g ls-tree -r --name-only refs/hb/baseline | wc -l) files"
