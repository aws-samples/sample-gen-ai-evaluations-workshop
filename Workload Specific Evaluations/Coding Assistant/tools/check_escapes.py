#!/usr/bin/env python3
"""Check that each option folder (docker/, local/, agentcore/) stands alone.

Temporary: delete this folder when the module keeps one option (see PRUNING.md).

For every file in an option folder it reports:
  link     a relative markdown/HTML link that leaves the folder or points at nothing
  dotdot   a `..` path that resolves outside the folder
  parent   Path(__file__).parent... that climbs above the folder, or <folder variable>.parent / Path.cwd().parent
  import   an import of a module-root helper (coding_eval, env_setup, model_config, harvest_env) the folder does not have
  name     a name of another option, of the chooser or the comparison doc, or of a module-root file the folder does not have
  words    (README, TESTER-GUIDE, MANIFEST, notebooks of local/ and agentcore/ only) wording that refers to another option
  abspath  a /Users/<name> path

datasets/ is checked for link, dotdot, parent, import and abspath, not for name and words: its files are byte-identical
copies (tools/check_copies.py), and their comments are listed in PRUNING.md.

Usage:
  python3 tools/check_escapes.py                      # the three option folders of this module
  python3 tools/check_escapes.py --dir PATH --as local   # one folder, for example a pruned and promoted module
Exit 1 if anything is found.
"""
from __future__ import annotations

import argparse, json, re, sys, urllib.parse
from pathlib import Path

MODULE = Path(__file__).resolve().parent.parent
OPTIONS = ("docker", "local", "agentcore")
SKIP_DIRS = {".git", ".venv", "node", "npm", "npm-cache", "uv-cache", "uv-python", "home", "bin", "jobs", "logs", "work",
             "__pycache__", ".ipynb_checkpoints", ".bedrock.lock"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".ipynb", ".toml", ".json", ".yml", ".yaml", ".txt", ".patch", ".lock", ".example", ""}
ROOT_HELPERS = ("coding_eval", "env_setup", "model_config", "harvest_env")

# Names that must not appear in an option's own files (outside datasets/), by option.
OTHER = {
    "docker": r"\bagentcore\b|AgentCore|(?<![\w/.~])local/|Local mode|local mode|\bsrt\b|sandbox-runtime|lite-modes|check_copies|check_escapes|PRUNING\.md",
    "local": r"agentcore|AgentCore|(?<![\w/.~])docker/|lite-modes|check_copies|check_escapes|PRUNING\.md|\btests_local\b|\bharvest_env\b|\bcoding_eval\b|\benv_setup\b|recorded-runs",
    "agentcore": r"(?<![\w/.~])local/|Local mode|local mode|\bsrt\b|sandbox-runtime|(?<![\w/.~])docker/|lite-modes|check_copies|check_escapes|PRUNING\.md|\btests_local\b|\bharvest_env\b|\bcoding_eval\b|\benv_setup\b|recorded-runs",
}
# Wording in the docs and notebooks of local/ and agentcore/ that points at another way to run the module.
WORDS = (r"\bbaseline\b|Harbor|\bnotebooks? 0[1-4]\b|\bnotebooks 02 and 03\b|Docker (path|mode|baseline|and Harbor)|"
         r"\b(this|other|the three|three|each|every|another) modes?\b|\bmodes of\b|choose a mode|chooser")
DOC_NAMES = re.compile(r"^(README|TESTER-GUIDE|MANIFEST)\.md$|\.ipynb$")

# Accepted hits: (path relative to the option folder, kind, text that matched, why it is fine).
ALLOW = [
    ("scripts/test_runner_v2.py", "dotdot", "../../outside", "a test fixture: a symlink target that the archive reader must refuse"),
    ("scripts/test_runner_v2.py", "dotdot", "../traversal.txt", "a test fixture: an archive member name that the reader must refuse"),
    ("05 Local mode walkthrough.ipynb", "dotdot", "../tmp/probe", "runs inside the agent's trial folder (cwd = trial repo), not this folder"),
    ("05 Local mode walkthrough.ipynb", "dotdot", "..", "`find ..` runs inside the agent's trial folder (cwd = trial repo)"),
    ("05 Local mode walkthrough.ipynb", "words", "Local mode", "the notebook's own file name"),
    ("05 Local mode walkthrough.ipynb", "name", "Local mode", "the notebook's own file name"),
    ("README.md", "words", "Local mode", "link text and target of the notebook's own file name"),
    ("README.md", "name", "Local mode", "link text and target of the notebook's own file name"),
    ("MANIFEST.md", "words", "Local mode", "the notebook's own file name"),
    ("MANIFEST.md", "name", "Local mode", "the notebook's own file name"),
    ("tests_local/test_coding_eval.py", "abspath", "/Users/<name>", "a test fixture: a leak that the secret scan must find"),
    ("p2_v2.py", "name", "docker/", "'the docker/buildkit line': the build tool's log format, not a folder"),
]


def files(root: Path):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts[:-1]) & SKIP_DIRS) and (p.suffix in TEXT_SUFFIXES or p.name.startswith(".")):
            yield p, rel


def chunks(p: Path):
    """(kind, text) pieces: notebook cells one by one (markdown or code), other files whole."""
    if p.suffix == ".ipynb":
        for c in json.loads(p.read_text())["cells"]:
            yield c["cell_type"], "".join(c["source"])
    else:
        yield ("markdown" if p.suffix == ".md" else "code"), p.read_text(errors="ignore")


def inside(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def scan(root: Path, name: str) -> list[tuple[str, str, str, str]]:
    found = []
    allow = {(a, k, t) for a, k, t, _ in ALLOW}

    def hit(rel, kind, text, line):
        if (str(rel), kind, text) not in allow and (rel.name, kind, text) not in allow:
            found.append((str(rel), kind, text, line.strip()[:160]))

    for p, rel in files(root):
        in_data = rel.parts[0] == "datasets"
        is_doc = bool(DOC_NAMES.search(rel.name)) and not in_data
        for ctype, text in chunks(p):
            for line in text.splitlines():
                # links
                if ctype == "markdown" or p.suffix == ".md":
                    for m in re.finditer(r"\]\(((?:[^()\s]|\([^()\s]*\))+)\)|(?:href|src)=\"([^\"]+)\"", line):
                        u = m.group(1) or m.group(2)
                        if re.match(r"[a-z]+:|#", u):
                            continue
                        t = (p.parent / urllib.parse.unquote(u.split("#")[0])) if u.split("#")[0] else p
                        if not inside(root, t) or not t.exists():
                            hit(rel, "link", u, line)
                # .. paths
                for m in re.finditer(r"[\w./${}()\"-]*(?<![\w.])\.\.(?![\w.])[\w./-]*", line):
                    tok = m.group(0)
                    if "../" not in tok + "/" or re.search(r"\.\.\.|\[[^\]]*\.\.", tok):
                        continue
                    core = re.sub(r"^.*?(?=(\.\.|/\.\.))", "", tok).lstrip("/").strip("\"'")
                    if not core.startswith(".."):
                        continue
                    if not inside(root, p.parent / core):
                        hit(rel, "dotdot", core.rstrip("/") or "..", line)
                # Path(__file__) climbs and option-root variables
                if ctype == "code":
                    for m in re.finditer(r"__file__\)(?:\.resolve\(\))?((?:\.parent)+)(?:\.parents\[(\d+)\])?", line):
                        k = m.group(1).count(".parent") + (int(m.group(2)) + 1 if m.group(2) else 0)
                        if k > len(rel.parts):
                            hit(rel, "parent", m.group(0), line)
                    for m in re.finditer(r"__file__\)(?:\.resolve\(\))?\.parents\[(\d+)\]", line):
                        if int(m.group(1)) + 1 > len(rel.parts):
                            hit(rel, "parent", m.group(0), line)
                    for m in re.finditer(r"\b(HERE|LOCAL|MODULE|MODULE_DIR|P1|ROOT)\.parent\b|Path\.cwd\(\)\.parent", line):
                        hit(rel, "parent", m.group(0), line)
                    for m in re.finditer(r"^\s*(?:import|from)\s+(\w+)", line):
                        mod = m.group(1)
                        if mod in ROOT_HELPERS and not ((root / f"{mod}.py").exists() or (root / mod).is_dir()):
                            hit(rel, "import", mod, line)
                for m in re.finditer(r"/Users/[A-Za-z]", line):
                    hit(rel, "abspath", "/Users/<name>", line)
                if in_data:
                    continue
                for m in re.finditer(OTHER[name], line):
                    if name != "docker" and m.group(0) in ROOT_HELPERS and (root / f"{m.group(0)}.py").exists():
                        continue
                    hit(rel, "name", m.group(0), line)
                if is_doc and name != "docker":
                    for m in re.finditer(WORDS, line, re.I):
                        hit(rel, "words", m.group(0), line)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, help="scan this one folder instead of the three option folders")
    ap.add_argument("--as", dest="name", choices=OPTIONS, help="which option the --dir folder is")
    a = ap.parse_args(argv)
    targets = [(a.dir, a.name)] if a.dir else [(MODULE / o, o) for o in OPTIONS]
    if a.dir and not a.name:
        ap.error("--dir needs --as")
    bad = 0
    for root, name in targets:
        n = sum(1 for _ in files(root))
        found = scan(root, name)
        print(f"== {name}: {root.resolve().name}/ ({n} files): " + ("clean" if not found else f"{len(found)} finding(s)"))
        for f, kind, text, line in found:
            print(f"   {kind:8} {f}: {text!r}   | {line}")
        bad += len(found)
    print("accepted (not reported):")
    for f, kind, text, why in ALLOW:
        print(f"   {kind:8} {f}: {text!r}  ({why})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
