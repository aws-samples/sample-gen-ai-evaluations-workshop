"""A Harbor agent that changes nothing and checks the seal from inside the agent phase.

    harbor run -p TASK_DIR -e harbor_env.agentcore_env:AgentCoreEnvironment -a harbor_env.probe_agent:ProbeAgent

Output: /logs/agent/probe.txt and the trial's agent/ folder.
"""

from __future__ import annotations

import os
import shlex
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

JUDGE_MODEL = os.environ.get("PROBE_JUDGE_MODEL", "us.anthropic.claude-haiku-5-5")
AGENT_MODEL = os.environ.get("PROBE_AGENT_MODEL", "us.amazon.nova-2-lite-v1:0")
OTHER_MODEL = os.environ.get("PROBE_OTHER_MODEL", "us.amazon.nova-pro-v1:0")

CONVERSE = """
import boto3, botocore.config, sys
c = boto3.client("bedrock-runtime", region_name="us-east-1", config=botocore.config.Config(retries={"max_attempts": 1}, read_timeout=30))
key = boto3.Session().get_credentials()
print("credentials:", type(key).__name__, (key.access_key[:4] + "...") if key else None)
for label, model in (("judge", sys.argv[1]), ("agent-model", sys.argv[2]), ("other-model", sys.argv[3])):
    try:
        r = c.converse(modelId=model, messages=[{"role": "user", "content": [{"text": "Say OK."}]}], inferenceConfig={"maxTokens": 5})
        print(f"CONVERSE {label} {model}: ALLOWED, answer={r['output']['message']['content'][0]['text']!r}")
    except Exception as e:
        print(f"CONVERSE {label} {model}: REFUSED {type(e).__name__}: {str(e)[:300]}")
"""


def probe_script() -> str:
    def curl(label, url, extra=""):
        return (f'echo "PROBE {label} {url}: $(curl -sS -m 10 {extra} -o /dev/null -w "http=%{{http_code}}" {shlex.quote(url)} 2>&1 | tr "\\n" " ")"')
    return "\n".join([
        'echo "PROBE uid=$(id -u) user=$(id -un) netns_links=$(ls /sys/class/net | tr "\\n" " ")"',
        curl("github", "https://github.com"),
        curl("pypi", "https://pypi.org/simple/"),
        curl("imds", "http://169.254.169.254/latest/api/token", "-X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' --noproxy '*'"),
        curl("imds-via-proxy", "http://169.254.169.254/latest/api/token", "-X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60'"),
        curl("api.openai.com", "https://api.openai.com/v1/models"),
        curl("mantle", "https://bedrock-mantle.us-east-1.api.aws/"),
        curl("bedrock-runtime", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
        curl("sts", "https://sts.amazonaws.com/"),
        'echo "PROBE allow.txt: $(tr "\\n" " " </run/fw/allow.txt 2>&1)"',
        'echo "PROBE sysctl protected_hardlinks=$(cat /proc/sys/fs/protected_hardlinks 2>&1) '
        'protected_symlinks=$(cat /proc/sys/fs/protected_symlinks 2>&1)"',
        'echo "PROBE extra.env readable: $(head -c 20 /run/fw/extra.env >/dev/null 2>&1 && echo YES || echo no)"',
        'echo "PROBE /tests: $(ls -A /tests 2>&1 | head -3 | tr "\\n" " ")"',
        'echo "PROBE /opt/hidden: $(ls -A /opt/hidden 2>&1 | head -3 | tr "\\n" " ")"',
        'echo "PROBE /solution: $(ls -A /solution 2>&1 | head -3 | tr "\\n" " ")"',
        'cd /work/repo 2>/dev/null && echo "PROBE repo commits=$(git rev-list --all --count 2>&1) remotes=[$(git remote -v | tr "\\n" " ")]'
        ' refs=$(git for-each-ref --format="%(refname)" | tr "\\n" " ") reflog=$(git reflog 2>/dev/null | wc -l)"',
        f"python3 -c {shlex.quote(CONVERSE)} {shlex.quote(JUDGE_MODEL)} {shlex.quote(AGENT_MODEL)} {shlex.quote(OTHER_MODEL)} 2>&1",
    ])


class ProbeAgent(BaseAgent):
    @staticmethod
    @override
    def name() -> str:
        return "probe"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        r = await environment.exec(probe_script() + " 2>&1 | tee /logs/agent/probe.txt", timeout_sec=300)
        (self.logs_dir / "probe-host-copy.txt").write_text((r.stdout or "") + (r.stderr or ""))
