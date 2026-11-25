
# ===== Harbor-on-AgentCore layer, appended to a task's environment/ or a verifier's tests/ Dockerfile =====
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 util-linux procps curl ca-certificates \
 && (command -v python3 >/dev/null || apt-get install -y --no-install-recommends python3) \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# The firewall blocks Harbor's installers, so harnesses are preinstalled. No npm cache, to keep the image small.
RUN if command -v npm >/dev/null; then \
      npm install -g opencode-ai@1.19.6 && opencode --version \
   && npm cache clean --force && rm -rf /root/.npm "${HOME:-/root}/.npm" /tmp/*; \
    fi
# /work stays root-owned, so the agent cannot swap /work/repo for a symlink.
RUN H="${HOME:-/home/agent}"; [ "$H" = /root ] && H=/home/agent; \
    useradd -u 2000 -d "$H" -s /bin/bash agent \
 && mkdir -p "$H" /logs/agent /logs/artifacts /logs/verifier \
 && chown -R agent:agent "$H" /logs/agent /logs/artifacts \
 && if [ -d /work ]; then chown root:root /work && chmod 0755 /work \
      && find /work -mindepth 1 -maxdepth 1 -exec chown -R agent:agent {} +; fi
# The grader user and a root-only copy of the repository's git dir, for staged grading.
COPY record-baseline.sh /tmp/record-baseline.sh
RUN useradd -u 2001 -d /home/grader -s /bin/bash grader && install -d -m 0700 -o grader -g grader /home/grader \
 && if [ -d /work/repo/.git ]; then mkdir -p /opt/hidden && cp -a /work/repo/.git /opt/hidden/base.git \
      && chown -R root:root /opt/hidden && bash /tmp/record-baseline.sh /work/repo /opt/hidden/base.git \
      && chown -R root:root /opt/hidden && chmod -R go-rwx /opt/hidden; fi && rm -f /tmp/record-baseline.sh
RUN mkdir -p /opt/firewall && { printenv | grep -E '^(PATH|VIRTUAL_ENV|UV_[A-Z_]+|LANG|LC_ALL)='; echo "HOME=$(getent passwd agent | cut -d: -f6)"; } > /opt/firewall/task.env
COPY main.py /opt/shim/main.py
COPY start.sh proxy.py as-agent as-agent-open as-grader /opt/firewall/
# strip every setuid/setgid bit, then lock down the firewall files
RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} + 2>/dev/null; \
    chown -R root:root /opt/firewall /opt/shim \
 && chmod 0755 /opt/firewall && chmod 0700 /opt/firewall/start.sh \
 && chown root:proxy /opt/firewall/proxy.py && chmod 0640 /opt/firewall/proxy.py \
 && chmod 0644 /opt/firewall/task.env && chmod 0755 /opt/firewall/as-agent /opt/firewall/as-agent-open /opt/firewall/as-grader && chmod 0700 /opt/shim/main.py
EXPOSE 8080
CMD ["python3", "/opt/shim/main.py"]
