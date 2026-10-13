"""Make CLI tools and the module settings visible to the Jupyter kernel.

Editors launched from the macOS Dock/Finder (Kiro, VS Code) do not read
~/.zshrc or ~/.bash_profile. The Jupyter kernel then inherits a minimal PATH
(/usr/bin:/bin:/usr/sbin:/sbin), so shutil.which() and subprocess.run() cannot
find docker, gh, codex, or harbor. fix_path() repairs os.environ["PATH"] for
the current kernel. Child processes inherit it.

load_env() reads AWS_REGION and AWS_PROFILE from the module's .env file.
"""
import os
import subprocess
import sys
from pathlib import Path

# Common install locations, used even if the login shell lookup fails.
FALLBACK_DIRS = [
    os.path.dirname(sys.executable),  # pip-installed scripts such as harbor
    "/opt/homebrew/bin",  # Homebrew on Apple silicon
    "/usr/local/bin",  # Homebrew on Intel, Docker Desktop symlinks
    "~/.local/bin",
    "~/.toolbox/bin",
    "~/.npm-global/bin",  # npm global prefix (codex)
    "~/.docker/bin",
    "/Applications/Docker.app/Contents/Resources/bin",
]


def _login_shell_path():
    """Return the PATH that the user's login shell sets, or [] on failure."""
    shell = os.environ.get("SHELL", "/bin/zsh")
    try:
        out = subprocess.run(
            [shell, "-ilc", 'printf %s "$PATH"'],
            capture_output=True, text=True, timeout=10,
            stdin=subprocess.DEVNULL,
        )
        lines = out.stdout.strip().splitlines()
        # The last line is the PATH. Earlier lines are shell banner noise.
        return lines[-1].split(os.pathsep) if lines else []
    except Exception:
        return []


def fix_path():
    """Add missing tool directories to os.environ["PATH"]. Safe to run again."""
    current = os.environ.get("PATH", "").split(os.pathsep)
    if os.name == "nt":
        candidates = [os.path.dirname(sys.executable)]
    else:
        fallbacks = [os.path.expanduser(d) for d in FALLBACK_DIRS]
        candidates = [d for d in fallbacks if os.path.isdir(d)] + _login_shell_path()
    # New dirs go first, existing entries are kept, duplicates are dropped.
    merged = list(dict.fromkeys(d for d in candidates + current if d))
    added = [d for d in merged if d not in current]
    os.environ["PATH"] = os.pathsep.join(merged)
    print(f"fix_path: added {len(added)} dirs to PATH" + (f": {', '.join(added)}" if added else ""))


def load_env(path=".env"):
    """Load the settings in .env into os.environ. Values in .env replace values in the kernel."""
    from dotenv import dotenv_values, load_dotenv

    if not Path(path).is_file():
        print(f"load_env: no {path} file. Using the kernel environment and the default AWS credential chain.")
        return
    load_dotenv(path, override=True)
    keys = [k for k, v in dotenv_values(path).items() if v]
    print(f"load_env: loaded {', '.join(keys) or 'no values'} from {path}")
