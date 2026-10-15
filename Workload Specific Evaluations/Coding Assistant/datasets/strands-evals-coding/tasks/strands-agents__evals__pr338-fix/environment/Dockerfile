# Sealed agent environment for a coding task. The repository is at the parent commit of the fix, with
# no change applied. The agent writes the fix. The hidden tests are in tests/ and are not in this image.
#
# node:22-bookworm is buildpack-deps:bookworm with Node.js added (Node.js is for Codex).
FROM public.ecr.aws/docker/library/node:22-bookworm

WORKDIR /work

# A shallow, single-branch, no-tags clone: only one tree, none of the history around it, and no tag
# that points at a later state.
RUN git clone --depth 1 --single-branch --no-tags "https://github.com/strands-agents/evals.git" repo
WORKDIR /work/repo

# Fetch exactly the parent commit of the fix and reset the working tree to it.
RUN git fetch --depth 1 origin 0bf12520908fcf02ca39acaf11cd302587988787 \
 && git reset --hard 0bf12520908fcf02ca39acaf11cd302587988787

# Seal the repository: remove the remote, delete every remote-tracking ref, expire the reflog, and
# prune everything that is now unreachable. Only the base commit remains, so the agent cannot find the
# fix in the git history.
RUN git remote remove origin \
 && git for-each-ref --format='%(refname)' refs/remotes | xargs -r -n1 git update-ref -d \
 && git reflog expire --expire=now --all \
 && git gc --prune=now --aggressive

# The verifier compares the agent's work with this commit.
RUN git rev-parse HEAD > /var/lib/task_base_ref

# Python 3.12 and the test dependencies of the repository. The repository has no lockfile, so
# requirements.lock pins every package to the newest version on the day that the fix was merged
# (uv pip compile --universal --exclude-newer). It has the extras that the repository's own test
# environment uses (hatch-test in pyproject.toml) and moto. The environment is outside the
# repository, so `git status` stays clean. At run time uv is offline: the agent's allowlist has no
# package index.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
COPY requirements.lock /opt/requirements.lock
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.12 /opt/venv \
 && uv pip install -r /opt/requirements.lock \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --no-deps -e .
ENV PATH="/opt/venv/bin:${PATH}" UV_OFFLINE=1
# Codex runs each command in a login shell (`bash -lc`). Debian's /etc/profile sets PATH again and
# drops /opt/venv/bin, so put it back for login shells too.
RUN echo 'export PATH="/opt/venv/bin:$PATH"' > /etc/profile.d/00-venv.sh

# A HOME that is not /root. Harbor runs the agent as root.
ENV HOME=/home/agent
RUN mkdir -p /home/agent

# The environment must be sound: the full unit test suite passes on the base. If a test fails here,
# the build fails, and the datapoint is not used.
RUN python -m pytest -q -p no:cacheprovider tests

# Codex, installed at build time: the build has open network, the agent has only its Bedrock
# allowlist. Harbor finds `codex` on PATH and skips its own npm install, which needs the network.
RUN npm install -g @openai/codex@0.162.1 \
 && apt-get update && apt-get install --no-install-recommends --yes ripgrep \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && codex --version
