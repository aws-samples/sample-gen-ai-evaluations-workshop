#!/usr/bin/env python3
"""L4: seal check on the host clones, hidden-test reachability, leakage scan (the notebook's scan, inlined below)."""
import json, os, shutil, subprocess, sys, tempfile, tomllib
from pathlib import Path
P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1 / "runner"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import p1


class ce:
    """The leakage scan: search the agent-visible files of a task (instruction.md, environment/) for each needle,
    after a self-test that plants a canary in a temporary copy of the task (the task files never change)."""
    CANARY = "CANARY-7f3a91"

    @staticmethod
    def read_toml(path): return tomllib.loads(Path(path).read_text())

    @staticmethod
    def _scan(task_dir, needles):
        task_dir = Path(task_dir)
        files = [task_dir / "instruction.md"] + [p for p in sorted((task_dir / "environment").rglob("*")) if p.is_file()]
        return [(str(f.relative_to(task_dir)), n) for f in files for n in needles
                if n and n.lower() in f.read_text(errors="ignore").lower()]

    @staticmethod
    def leakage_scan(task_dir, needles):
        with tempfile.TemporaryDirectory(prefix="leakage-canary-") as tmp:
            copy = Path(tmp) / Path(task_dir).name
            shutil.copytree(task_dir, copy)
            (copy / "environment" / "canary.txt").write_text(ce.CANARY)
            if not ce._scan(copy, [ce.CANARY]):
                raise AssertionError("The scan did not find the canary. The scan has a defect.")
        return ce._scan(task_dir, needles)


out = {}
def git(repo, *a): return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
for name in ("pr371-fix", "pr338-fix", "pr4121-approve", "pr4615-reject"):
    t = p1.Task(name); p1.prepare(t)
    W = p1.WORK / "trials" / f"{t.full}__basesuite"   # a trial dir as the agent sees it (repo clone + venv)
    repo = W / "repo"
    r = {"commits_all": len(git(repo, "log", "--all", "--oneline").splitlines()), "remotes": len(git(repo, "remote").splitlines()),
         "status_lines": len(git(repo, "status", "--porcelain").splitlines()),
         "unreachable_objects": len(git(repo, "fsck", "--unreachable", "--no-reflogs").splitlines()),
         "reflog_entries": len(git(repo, "reflog", "--all").splitlines()), "head_is_base": git(repo, "rev-parse", "HEAD") == (t.base / "base_ref").read_text().strip()}
    r["seal_ok"] = r["commits_all"] == 1 and r["remotes"] == 0 and (r["status_lines"] > 0 if t.kind == "pr" else r["status_lines"] == 0)
    # files named like hidden tests / oracle inside the agent's trial dir (repo + venv + home)
    r["hidden_in_trial_dir"] = subprocess.run(["find", str(W), "(", "-name", "*_verify.py", "-o", "-name", "oracle.patch", "-o", "-name", "reference.patch", "-o", "-name", "oracle.json", ")"], capture_output=True, text=True).stdout.split()
    # leakage scan with the notebook's needles
    td = t.dir; meta = ce.read_toml(td / "task.toml")["metadata"]
    if t.kind == "coding":
        o = meta["origin"]; needles = [f"#{n}" for n in o["pr_numbers"] + o["issue_numbers"]] + [f"pull/{n}" for n in o["pr_numbers"]] + [o["merge_commit"][:12], Path(meta["verifier"]["hidden_tests"]).stem]
        needles += [l[1:].strip() for l in (td / "solution/oracle.patch").read_text().splitlines() if l.startswith("+") and not l.startswith("+++") and len(l.strip()) >= 40]
        r["leakage_hits"] = [list(h) for h in ce.leakage_scan(td, needles)]
        r["canary_in_repo"] = subprocess.run(["grep", "-rl", ce.CANARY, str(repo), str(W / "home")], capture_output=True, text=True).stdout.split()
    out[t.full] = r
# static reachability: allowed read roots of the agent profile + system roots
roots = [str(p) for p in (p1.P1 / "node", p1.P1 / "npm", p1.P1 / "uv-python", p1.P1 / "bin")] + ["/usr", "/opt", "/Library", "/private/etc"]
f = subprocess.run(["find", *roots, "(", "-name", "*_verify.py", "-o", "-name", "oracle.patch", "-o", "-name", "reference.patch", ")"], capture_output=True, text=True).stdout.split()
out["reachable_outside_trial_dir"] = f
print(json.dumps(out, indent=1))
