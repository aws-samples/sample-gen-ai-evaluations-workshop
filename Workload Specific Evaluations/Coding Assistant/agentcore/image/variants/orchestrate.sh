#!/bin/bash
# Runs as root after the agent is stopped and the verifier phase is set (allowlist = bedrock-runtime). Replaces `bash /tests/test.sh`.
#   P2_PAR=1   stage 1 (base + hidden pytest) runs at the same time as the LLM stages, on its own copy of the tree
#   P2_HARD=1  stage 1 runs on a fresh root-owned checkout of the sealed repo with the agent's patch applied (see prep-s1.sh)
# Both: stage 1 on the fresh checkout, concurrently. Neither: same stages in the same order as tests/test.sh.
# Everything the agent could influence runs as the `agent` uid through /opt/firewall/as-agent, as in the default path.
set -u
P2_PAR=${P2_PAR:-0}; P2_HARD=${P2_HARD:-0}
L=/logs/verifier; AS=/opt/firewall/as-agent
T=/run/p2v; rm -rf $T; install -d -m 0700 -o root -g root $T
tick() { echo "$1 $(date +%s.%N)" >> $T/ticks; }
mkdir -p $L
install -m 0644 -o root -g root /opt/hidden/variants/variant.sh /tests/variant.sh
V="P2_PAR=$P2_PAR P2_HARD=$P2_HARD"
tick start
$AS "$V bash /tests/variant.sh stage0"; tick stage0
env $V bash /opt/hidden/variants/prep-s1.sh > $T/prep.log 2>&1 || echo "PREP FAILED, see p2_prep.log"; tick prep
run_s1() { unshare -m bash -c "mount --bind /opt/verify/repo /work/repo && exec $AS '$V bash /tests/variant.sh s1'" > $T/s1.out 2>&1; tick s1_end; }
if [ "$P2_PAR" = 1 ]; then run_s1 & S1=$!; else run_s1; S1=; fi
$AS "$V bash /tests/variant.sh rest"; tick rest_end
[ -n "$S1" ] && wait $S1
$AS "$V bash /tests/variant.sh aggregate"; tick end
for f in ticks prep.log s1.out; do rm -f $L/p2_$f; cp $T/$f $L/p2_$f 2>/dev/null; done
cp /opt/verify/hard_report.txt $L/p2_hard_report.txt 2>/dev/null
chown agent:agent $L/p2_* 2>/dev/null
exit 0
