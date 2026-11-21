#!/bin/bash
# P1 one-time setup (macOS arm64). Prerequisite: uv (https://docs.astral.sh/uv/), curl, git, tar. No sudo, no Docker.
# Installs into this directory only: Node 22, srt, Codex 0.162.1, ripgrep. Prints the elapsed time per step.
set -euo pipefail
P1="$(cd "$(dirname "$0")" && pwd)"; cd "$P1"
NODE_V=22.23.5; RG_V=15.2.0; CODEX_V=0.162.1; SRT_V=0.0.83
t() { local s=$(date +%s); "$@"; echo "[$(date +%T)] $(( $(date +%s)-s ))s  $*" >&2; }
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "tested on macOS arm64 only"; exit 1; }
mkdir -p node bin home logs
if [ ! -x bin/timeout ]; then   # macOS has no GNU timeout; the verifier's test.sh calls `timeout 900 python -m pytest`
cat > bin/timeout <<'SHIM'
#!/usr/bin/env python3
"""Minimal GNU `timeout SECONDS CMD...` for macOS (test.sh calls it). Exit 124 on timeout."""
import os, signal, subprocess, sys
secs = float(sys.argv[1].rstrip("smhd")); cmd = sys.argv[2:]
p = subprocess.Popen(cmd, start_new_session=True)
try:
    sys.exit(p.wait(timeout=secs))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try: p.wait(10)
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL)
    sys.exit(124)
SHIM
chmod +x bin/timeout
fi
if [ ! -x node/bin/node ]; then
  t curl -fsSL -o node.tgz "https://nodejs.org/dist/v$NODE_V/node-v$NODE_V-darwin-arm64.tar.gz"
  tar -xzf node.tgz -C node --strip-components=1 && rm node.tgz
fi
source ./env.sh
if [ ! -x bin/rg ]; then
  t curl -fsSL -o rg.tgz "https://github.com/BurntSushi/ripgrep/releases/download/$RG_V/ripgrep-$RG_V-aarch64-apple-darwin.tar.gz"
  tar -xzf rg.tgz --strip-components=1 -C bin "ripgrep-$RG_V-aarch64-apple-darwin/rg" && rm rg.tgz
fi
[ -x npm/bin/srt ]   || t npm install -g "@anthropic-ai/sandbox-runtime@$SRT_V"
[ -x npm/bin/codex ] || t npm install -g "@openai/codex@$CODEX_V"
node --version; srt --version; codex --version; rg --version | head -1; uv --version
echo "next: uv run --no-project runner/p1.py prepare pr371-fix pr338-fix pr4121-approve pr4615-reject"
