#!/bin/bash
# Why does Codex fail inside srt ("Connection failed: error sending request")? Runs a no-tool "reply OK" Codex call inside the agent's
# srt profile in up to 3 variants and reports which one works. ~1 min. Needs the same environment as run_all.sh (source env.sh first).
cd "$(dirname "$0")/.."; source ./env.sh
RUN="uv run --no-project --python 3.12 --quiet python runner/p1.py"
PROMPT="Reply with the single word OK. Do not run any commands."
hello() { # name, then VAR=value pairs for env
  local n=$1; shift
  env "$@" $RUN sandbox-run pr4121-approve --tag diag --keep --creds federated -- codex exec --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check --model openai.gpt-6-sol --json --enable unified_exec -c web_search=disabled -- "$PROMPT" > "logs/diag-$n.log" 2>&1
  if grep -q '"type":"turn.completed"' "logs/diag-$n.log"; then echo "[diag] $n: WORKS"; return 0; else echo "[diag] $n: FAILS (logs/diag-$n.log)"; tail -3 "logs/diag-$n.log" | cut -c1-200; return 1; fi
}
# v0: plain node inside the agent profile (Codex is a node script), with the CA variables unset, set to /etc/ssl/cert.pem (the runner default)
# and set to an empty string. A SIGABRT in node::InitializeOncePerProcessInternal means node cannot start in that state.
node_try() { # name, JSON for P1_AGENT_ENV_EXTRA
  P1_AGENT_ENV_EXTRA="$2" $RUN sandbox-run pr4121-approve --tag diag --keep -- node -e 'console.log("node-ok", process.version)' > "logs/diag-v0-$1.log" 2>&1
  if grep -q node-ok "logs/diag-v0-$1.log"; then echo "[diag] v0 node in srt, CA vars $1: OK"; return 0; else echo "[diag] v0 node in srt, CA vars $1: FAILS (logs/diag-v0-$1.log)"; tail -2 "logs/diag-v0-$1.log" | cut -c1-200; return 1; fi
}
node_try unset '{"CODEX_CA_CERTIFICATE":null,"SSL_CERT_FILE":null}'; r1=$?
node_try set '{}'; r2=$?
node_try empty '{"CODEX_CA_CERTIFICATE":"","SSL_CERT_FILE":""}'; r3=$?
echo "[diag] node start: unset=$r1 set=$r2 empty=$r3 (0 = starts)"
[ $r2 -ne 0 ] && { echo "[diag] node does not start with the default environment; stop here and send logs/diag-v0-*.log"; exit 1; }
hello v1-as-is X=1 && { echo "[diag] result: Codex works inside srt with the default profile (CA bundle from /etc/ssl/cert.pem)."; exit 0; }
echo "[diag] collecting debug output (srt proxy log + Codex RUST_LOG) for the failing default profile"
hello v2-debug X=1 P1_SRT_DEBUG=1 'P1_AGENT_ENV_EXTRA={"RUST_LOG":"debug"}' >/dev/null
grep -iE "CONNECT|bedrock|denied|blocked|allow" logs/diag-v2-debug.log | head -20 | cut -c1-220
hello v3-weaker-network-isolation 'P1_SRT_EXTRA={"enableWeakerNetworkIsolation":true}' && { echo "[diag] result: needs enableWeakerNetworkIsolation (opens com.apple.trustd.agent to the sandbox; srt documents it as a possible exfiltration channel)."; exit 10; }
hello v4-ca-vars-unset 'P1_AGENT_ENV_EXTRA={"CODEX_CA_CERTIFICATE":null,"SSL_CERT_FILE":null}' && { echo "[diag] result: works only without the CA variables"; exit 11; }
echo "[diag] result: no variant worked. Send logs/diag-*.log"; exit 1
