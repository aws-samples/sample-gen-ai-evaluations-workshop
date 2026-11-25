"""The last gate of a new coding task: Harbor's nop agent must score 0 and its oracle agent 1, in stages.

    .venv/bin/python step1-build-tasks/check_gate.py <nop job dir> <oracle job dir>

nop changes nothing, so a task it passes asks for nothing. oracle applies the real change, so a task it
fails cannot be solved. Both trials must have been graded in stages (`grading staged` in summary.py):
the score then came from the trusted grading stage, not from a file the agent could write.
Exits 1 when the task fails the gate.
"""

import json
import sys
from pathlib import Path


def trials(job_dir: Path) -> list[dict]:
    """Every trial result Harbor wrote under a job directory."""
    found = []
    for path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(path.read_text())
        events = path.parent / "agentcore.jsonl"
        modes = [json.loads(line).get("mode") for line in events.read_text().splitlines() if '"grader_mode"' in line] if events.exists() else []
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        found.append({"trial": path.parent.name, "reward": reward, "grading": modes[-1] if modes else "unknown"})
    return found


def main(nop_dir: str, oracle_dir: str) -> int:
    ok = True
    for label, job, want in (("nop", Path(nop_dir), 0.0), ("oracle", Path(oracle_dir), 1.0)):
        rows = trials(job)
        if not rows:
            print(f"   {label}: no trial in {job}")
            ok = False
        for row in rows:
            good = row["reward"] == want and row["grading"] == "staged"
            ok &= good
            print(f"   {label:<6} reward {row['reward']}  grading {row['grading']}  {'ok' if good else f'expected reward {want}, staged'}")
    print("   usable task" if ok else "   NOT usable: leave it out of the dataset")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
