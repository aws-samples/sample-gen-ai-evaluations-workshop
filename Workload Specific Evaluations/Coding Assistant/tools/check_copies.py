#!/usr/bin/env python3
"""Drift check: the files that docker/, local/ and agentcore/ each keep a copy of, and which copies differ.

Temporary: delete this folder when the module keeps one option (see PRUNING.md).

The shared files are model_config.py, codex-bedrock.toml and everything in datasets/ (the task directories, the
PR-review candidates, rubric.md, risk-map.toml, and the verifier sources in datasets/_shared/). docker/ has the full
set. local/ and agentcore/ leave out what they do not read (NOT_COPIED below).

For each shared file the script compares the SHA-256 of every copy. When a file differs, it prints which folders agree,
so that a fix made in one copy during the test week can be mirrored, for example:

    cp "local/datasets/_shared/pr-review/tests/score.py" "docker/datasets/_shared/pr-review/tests/score.py"

Then, in each folder, it runs `datasets/_shared/sync.py --check` (the task copies of the verifier files against
_shared/). After you edit a file in datasets/_shared/, run `python datasets/_shared/sync.py` in that folder.

Usage, from anywhere:
    python3 tools/check_copies.py        # differences only
    python3 tools/check_copies.py -v     # also list every file that is the same everywhere
Exit 1 if a copy differs, is missing, or a sync.py --check fails.
"""
from __future__ import annotations

import argparse, hashlib, subprocess, sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent.parent
OPTIONS = ("docker", "local", "agentcore")
SHARED = ("model_config.py", "codex-bedrock.toml", "datasets")
# Files of docker/ that local/ and agentcore/ do not read, so they have no copy.
NOT_COPIED = ("datasets/recorded-runs/", "datasets/awslabs-mcp-pr-review/metric.py")


def shared_files(root: Path) -> set[str]:
    out = set()
    for name in SHARED:
        p = root / name
        if p.is_file():
            out.add(name)
        elif p.is_dir():
            out |= {str(f.relative_to(root)) for f in p.rglob("*") if f.is_file()
                    and "__pycache__" not in f.parts and f.name != ".DS_Store"}
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    roots = {o: MODULE / o for o in OPTIONS if (MODULE / o).is_dir()}
    have = {o: shared_files(r) for o, r in roots.items()}
    bad = same = 0
    for rel in sorted(set().union(*have.values())):
        expected = [o for o in roots if o == "docker" or not rel.startswith(NOT_COPIED)]
        missing = [o for o in expected if rel not in have[o]]
        digests = {o: sha(roots[o] / rel) for o in expected if rel in have[o]}
        extra = [o for o in roots if o not in expected and rel in have[o]]
        groups: dict[str, list[str]] = {}
        for o, d in digests.items():
            groups.setdefault(d, []).append(o)
        if len(groups) > 1 or missing or extra:
            bad += 1
            parts = [f"{'+'.join(v)}={k}" for k, v in groups.items()]
            parts += [f"{o}=MISSING" for o in missing] + [f"{o}=UNEXPECTED (listed in NOT_COPIED)" for o in extra]
            print(f"DIFFERS  {rel}: " + ", ".join(parts))
        else:
            same += 1
            if a.verbose:
                print(f"same     {rel}  ({'+'.join(digests)})")
    print(f"{same} shared files identical, {bad} differ or are missing"
          + (f"; not copied to local/ and agentcore/ on purpose: {', '.join(NOT_COPIED)}" if "docker" in roots else ""))
    for o, r in roots.items():
        s = r / "datasets" / "_shared" / "sync.py"
        if s.exists():
            p = subprocess.run([sys.executable, "-B", str(s), "--check"], capture_output=True, text=True)
            print(f"{o}: sync.py --check: {(p.stdout.strip().splitlines() or ['(no output)'])[-1]}")
            bad += p.returncode != 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
