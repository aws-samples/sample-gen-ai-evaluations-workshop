# Source this file: isolated toolchain for P1 (own HOME, caches, npm prefix, Node 22).
P1="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export P1
export P1_REAL_HOME="${P1_REAL_HOME:-$HOME}"
export HOME="$P1/home"
# work dir (sealed clones, trial dirs, verifier copies): no spaces allowed, see runner/p1.py. Delete it to clean up.
export P1_WORK="${P1_WORK:-$P1_REAL_HOME/.coding-assistant-local/work}"
export UV_CACHE_DIR="$P1/uv-cache"
export UV_PYTHON_INSTALL_DIR="$P1/uv-python"
export NPM_CONFIG_PREFIX="$P1/npm"
export NPM_CONFIG_CACHE="$P1/npm-cache"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$P1/npm/bin:$P1/node/bin:$P1/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
unset AWS_PROFILE AWS_DEFAULT_PROFILE
