"""Test only, NOT A SEAL: AgentCoreEnvironment with a local Docker container instead of an AgentCore session (flow test).

    HB_LOCAL_IMAGES='{"<image_key>": "IMAGE", ...}' harbor run ... -e harbor_env.tests.local_docker:LocalDockerAgentCoreEnvironment
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import override

from harbor_env import agentcore_env as ace


class DockerSession:
    def __init__(self, image: str):
        self.sid = "local-" + uuid.uuid4().hex[:12]
        self.calls = self.throttles = 0
        self.start_attempts = 1
        subprocess.run(["docker", "run", "-d", "--name", self.sid, "--entrypoint", "sleep", image, "infinity"], check=True,
                       capture_output=True)
        self.cmd("mkdir -p /run/fw && touch /run/fw/allow.txt /run/fw/extra.env && echo OK > /run/fw.status && "
                 "sed -i 's#^exec nsenter --net=/run/fw/agent.net -- #exec #' /opt/firewall/as-agent")

    def cmd(self, command, timeout=300, retries=12):
        self.calls += 1
        t0 = time.time()
        r = subprocess.run(["docker", "exec", self.sid, "/bin/bash", "-c", command], capture_output=True, timeout=timeout + 30)
        return (r.stdout + r.stderr).decode(errors="replace"), r.returncode, round(time.time() - t0, 2)

    def start(self, command, attempts=3):
        return self.cmd(command)

    def agent(self, command, timeout=300):
        return self.cmd("/opt/firewall/as-agent " + ace.shlex.quote(command), timeout)

    def put(self, path, data: bytes, mode="0644", owner="root"):
        b = base64.b64encode(data).decode()
        r = subprocess.run(["docker", "exec", "-i", self.sid, "/bin/bash", "-c",
                            f"base64 -d > {path} && chmod {mode} {path} && chown {owner} {path}"], input=b.encode(), capture_output=True)
        assert r.returncode == 0, r.stderr

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.sid], capture_output=True)


class LocalDockerAgentCoreEnvironment(ace.AgentCoreEnvironment):
    @override
    async def start(self, force_build: bool) -> None:
        self._tag = ace.image_key(Path(self.environment_dir), self.environment_name, self._is_verifier_env)
        image = json.loads(os.environ["HB_LOCAL_IMAGES"])[self._tag]
        ace.aws.agent_creds = lambda seconds=3600: {"AWS_ACCESS_KEY_ID": "AGENT_DUMMY", "AWS_SECRET_ACCESS_KEY": "x", "AWS_SESSION_TOKEN": "x"}
        ace.aws.runtime_role_creds = lambda s: {"AWS_ACCESS_KEY_ID": "GRADER_DUMMY", "AWS_SECRET_ACCESS_KEY": "x", "AWS_SESSION_TOKEN": "x"}
        self._session = DockerSession(image)
        # no sysctl: never change the host kernel from a local container
        self._event(event="session_start", session_id=self._session.sid, runtime="local:" + image,
                    kernel_protection="skipped (local Docker stand-in)")
        self.logger.info("local Docker stand-in: fs.protected_hardlinks/protected_symlinks check skipped")
        await self._enter_phase("verifier" if self._is_verifier_env else "agent", self._network_policy)
