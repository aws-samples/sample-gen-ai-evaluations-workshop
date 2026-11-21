#!/bin/bash
# Oracle: apply the merged fix of strands-agents/evals#371 (source only, without its tests).
set -euo pipefail
cd /work/repo
git apply --whitespace=nowarn /solution/oracle.patch
echo "Applied the oracle patch."
