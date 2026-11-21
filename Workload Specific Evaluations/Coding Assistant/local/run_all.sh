#!/bin/bash
# Test ladder for a plain terminal (srt needs sandbox-exec, which fails inside another sandbox, for example an agent tool's).
#   ./run_all.sh                       all stages in order; STOPS at the first failing stage
#   STAGES="L5s L6a L6" ./run_all.sh   only these stages (still in the order below, stops at the first failure)
#   CONTINUE=1 ./run_all.sh            do not stop on failure (not for L6/L7)
# Stage order: L0 L1 L2 L4 L5s L6a L6 L7   (L3, L3rt and L5 are maintainer stages and are not shipped, see MANIFEST.md)
#   L5s  verifier inside its srt profile, oracle patch must score 1 (pr4121 and pr371)          ~4 min
#   L6a  Codex "reply OK" inside the agent's srt profile; tries fixes if it fails                ~1 min
#   L6   two real Codex trials (pr4121 review, pr371 fix); fails if the agent did not run         ~10-12 min
#   L7   4 tasks x 3 Codex trials, then oracle and empty controls                                ~60-70 min
# Logs: logs/run_all/<stage>.log. One run_all.sh at a time per checkout: it holds .bedrock.lock.
cd "$(dirname "$0")"; source ./env.sh
mkdir -p logs/run_all
LOCKD=.bedrock.lock
STAGES=${STAGES:-"L0 L1 L2 L4 L5s L6a L6 L7"}
for s in $STAGES; do case " L0 L1 L2 L4 L5s L6a L6 L7 " in *" $s "*) ;; *) echo "stage $s is not part of this ladder"; exit 2;; esac; done
PYR="uv run --no-project --python 3.12 --quiet python"
RUN="$PYR runner/p1.py"
FED=scripts/agent-policy.json           # agent gets a federated token limited to the three model IDs

# one lock for the whole run (children see P1_LOCK_HELD=1 and do not take it again); say who holds it while waiting
for i in $(seq 1 720); do
  # a lock left by a killed run_all.sh (its pid is gone) is removed
  op=$(sed -n 's/^run_all.sh pid \([0-9]*\) .*/\1/p' "$LOCKD/owner" 2>/dev/null); [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null && rm -rf "$LOCKD"
  if mkdir "$LOCKD" 2>/dev/null; then echo "run_all.sh pid $$ $(date +%T) stages: $STAGES" > "$LOCKD/owner"; break; fi
  [ $((i % 6)) = 1 ] && echo "[$(date +%T)] waiting for $LOCKD held by: $(cat $LOCKD/owner 2>/dev/null || echo 'unknown')"
  sleep 5
done
[ -f "$LOCKD/owner" ] && grep -q "pid $$ " "$LOCKD/owner" || { echo "could not take the lock"; exit 2; }
trap 'rm -rf "$LOCKD"' EXIT
export P1_LOCK_HELD=1

stage() { # name, command...
  local n=$1; shift; local s=$(date +%s); echo "[$(date +%T)] == $n"; "$@" > "logs/run_all/$n.log" 2>&1; local rc=$?
  echo "[$(date +%T)] $n rc=$rc in $(( $(date +%s)-s ))s  (logs/run_all/$n.log)"
  if [ $rc -ne 0 ] && [ -z "$CONTINUE" ]; then echo "STOP at $n. Last lines:"; tail -5 "logs/run_all/$n.log" | cut -c1-240; exit $rc; fi
}
has() { [[ " $STAGES " == *" $1 "* ]]; }
l1() { $RUN prepare pr371-fix pr338-fix pr4121-approve pr4615-reject && $RUN base-suite pr371-fix pr338-fix pr4121-approve pr4615-reject; }
l2() { $RUN sandbox-run pr4121-approve --tag l2 --script scripts/l2_inner.sh | tee logs/L2-gates.txt; ! grep -E '\|(REACHED|NOT reachable)' logs/L2-gates.txt; }
l4() { $PYR scripts/l4_seal.py | tee logs/L4-static.json; }
# L5s: the verifier inside srt with fake agents. An oracle patch must score 1.0 with a non-empty agent.patch; otherwise the sandboxed verifier is broken.
l5s() { $RUN trial pr4121-approve --agent oracle --job L5s --tag oracle && $RUN trial pr371-fix --agent oracle --job L5s --tag oracle && $PYR - <<'PY'
import json, glob, sys
bad = 0
for f in sorted(glob.glob("jobs/L5s/*/verifier/reward.json")):
    r = json.load(open(f)); d = f.rsplit("/verifier/", 1)[0]
    patch = __import__("os").path.getsize(d + "/verifier/agent.patch") if "pr371" in f else -1
    ok = (r.get("reward") == 1.0 and patch > 0) if "pr371" in f else r.get("coverage_all") == 1.0
    print(("OK  " if ok else "FAIL"), d.split("/")[-1], "reward", r.get("reward"), "coverage_all", r.get("coverage_all"), "patch_bytes", patch); bad += not ok
sys.exit(1 if bad else 0)
PY
}
l6a() { scripts/diag_codex.sh; }
l6() { $RUN trial pr4121-approve --agent codex --job L6 --federated $FED && $RUN trial pr371-fix --agent codex --job L6 --federated $FED; }
l7() { $RUN job --tasks pr371-fix,pr338-fix,pr4121-approve,pr4615-reject --n 3 --agent codex --job L7 --federated $FED \
       && for a in oracle nop; do for t in pr371-fix pr338-fix pr4121-approve pr4615-reject; do $RUN trial $t --agent $a --job L7-$a --tag $a || return 1; done; done; }
has L0  && stage L0 bash -c 'srt -- true && srt -c "echo sandbox-ok"'
has L1  && stage L1 l1
has L2  && stage L2 l2
has L4  && stage L4 l4
has L5s && stage L5s l5s
has L6a && stage L6a l6a
has L6  && stage L6 l6
has L7  && stage L7 l7
echo "all requested stages passed"
