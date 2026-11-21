
# ===== P2 layer (appended to the task's own environment/Dockerfile, which is used verbatim above) =====
# AgentCore Runtime needs linux/arm64 and an HTTP server on :8080 (/ping, /invocations). main.py is that shim; it first runs the
# in-session firewall (start.sh, proxy.py, as-agent; design in agentcore/README.md).
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 iproute2 util-linux procps curl netcat-openbsd dnsutils strace kmod less file ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# The agent runs as `agent` (uid 2000) and owns /work (the repo), its HOME and /logs/agent, /logs/artifacts. The proxy runs as the stock `proxy` user.
RUN useradd -u 2000 -d "$HOME" -s /bin/bash agent \
 && mkdir -p "$HOME" /logs/agent /logs/artifacts \
 && chown -R agent:agent "$HOME" /work /logs/agent /logs/artifacts
# PR-review verifier needs boto3==1.43.89 (the task's own verifier image, datasets/_shared/pr-review/tests/Dockerfile).
RUN UV_OFFLINE=0 uv venv --python 3.12 /opt/verifier-venv && UV_OFFLINE=0 uv pip install --python /opt/verifier-venv/bin/python boto3==1.43.89
# Hidden verifier files: root-only, copied to /tests by the runner only after the agent has stopped (a shared-container verifier).
COPY hidden/ /opt/hidden/tests/
RUN chown -R root:root /opt/hidden && chmod -R go-rwx /opt/hidden
# The Dockerfile ENV values, for as-agent, so the agent sees the same PATH/HOME/VIRTUAL_ENV as in the task's own image.
RUN mkdir -p /opt/firewall && printenv | grep -E '^(HOME|PATH|VIRTUAL_ENV|UV_[A-Z_]+|LANG|LC_ALL)=' > /opt/firewall/task.env
COPY main.py /opt/shim/main.py
COPY start.sh proxy.py as-agent /opt/firewall/
# strip every setuid/setgid bit, then lock down the firewall files
RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null; \
    chown -R root:root /opt/firewall /opt/shim \
 && chmod 0755 /opt/firewall && chmod 0700 /opt/firewall/start.sh \
 && chown root:proxy /opt/firewall/proxy.py && chmod 0640 /opt/firewall/proxy.py \
 && chmod 0644 /opt/firewall/task.env && chmod 0755 /opt/firewall/as-agent && chmod 0700 /opt/shim/main.py
EXPOSE 8080
CMD ["python3", "/opt/shim/main.py"]
