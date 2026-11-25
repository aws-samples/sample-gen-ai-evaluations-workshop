#!/usr/bin/env python3
"""One row per trial of Harbor job folders: the reward and, for a PR-review task, the agent's verdict next to the expected one.

    python3 harbor_env/summary.py JOB_DIR... [--datasets DIR]

Prints a Markdown table and writes summary.json in each job folder.
"""
import argparse, collections, json, pathlib, tomllib

DATASETS = pathlib.Path(__file__).resolve().parents[1] / "datasets"
CODING_KEYS = ("reward", "correct", "fail_to_pass", "pass_to_pass")
REVIEW_KEYS = ("coverage_required", "coverage_all", "uncredited")


def _json(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return default


def task_config(task_name, datasets=DATASETS):
    folder = (task_name or "").replace("/", "__")
    for f in pathlib.Path(datasets).glob(f"*/tasks/{folder}/task.toml"):
        return tomllib.loads(f.read_text())
    return {}


MISSING_GRADER_MODE = "MISSING grader_mode"


def grading_mode(trial_dir, result):
    for line in reversed((pathlib.Path(trial_dir) / "agentcore.jsonl").read_text().splitlines()
                         if (pathlib.Path(trial_dir) / "agentcore.jsonl").exists() else []):
        e = json.loads(line)
        if e.get("event") == "grader_mode":
            return e.get("mode", "-")
    mode = result.get("verifier_environment_mode") or "-"
    return mode if mode == "separate" else f"{MISSING_GRADER_MODE} ({mode})"


NOT_SUBMITTED = "not submitted"


def review_verdict(trial_dir, reward, cfg):
    harvest = cfg.get("metadata", {}).get("harvest", {})
    expected = harvest.get("expected_verdict")
    if not expected:
        return {}
    blocking = harvest.get("blocking_severity", "high")
    path = pathlib.Path(trial_dir) / "artifacts/logs/artifacts/findings.json"
    findings = _json(path, [])
    findings = findings if isinstance(findings, list) else []
    severities = collections.Counter(str(f.get("severity", "")).lower() for f in findings if isinstance(f, dict))
    predicted = ("block" if severities.get(blocking) else "approve") if path.is_file() else NOT_SUBMITTED
    passed = None
    if reward is not None and "passed" in reward:
        passed = reward["passed"] == 1.0
    elif reward is not None:
        passed = predicted == expected and (expected != "block" or reward.get("coverage_required") == 1.0)
    return {"expected_verdict": expected, "predicted_verdict": predicted, "n_findings": len(findings),
            "severities": dict(severities), "oracle_all": sum(v for k, v in (reward or {}).items() if k.startswith("oracle_")),
            "passed": passed}


def trial_row(trial_dir, datasets=DATASETS):
    t = pathlib.Path(trial_dir)
    r = _json(t / "result.json", {})
    info = r.get("agent_info") or {}
    reward = _json(t / "verifier/reward.json")
    row = {"trial": t.name, "task": r.get("task_name"), "agent": info.get("name"),
           "model": (info.get("model_info") or {}).get("name"), "grading": grading_mode(t, r),
           "exception": (r.get("exception_info") or {}).get("exception_type"), "reward": reward}
    row.update(review_verdict(t, reward, task_config(r.get("task_name"), datasets)))
    return row


def job_rows(job_dir, datasets=DATASETS):
    return [trial_row(p.parent, datasets) for p in sorted(pathlib.Path(job_dir).glob("*/result.json"))
            if (p.parent / "config.json").exists() or (p.parent / "verifier").exists()]


def markdown(rows):
    out = ["| trial | agent | model | grading | score | verdict (agent / expected) | findings | passed | exception |",
           "|---|---|---|---|---|---|---|---|---|"]
    for w in rows:
        rw = w["reward"] or {}
        keys = REVIEW_KEYS if "expected_verdict" in w else CODING_KEYS
        score = ", ".join(f"{k} {rw[k]}" for k in keys if k in rw) or "no reward"
        if "expected_verdict" in w:
            score += f", oracle_all {w['oracle_all']}"
            verdict = f"{w['predicted_verdict']} / {w['expected_verdict']}"
            findings = f"{w['n_findings']} {w['severities']}" if w["n_findings"] else "0"
        else:
            verdict, findings = "-", "-"
        passed = {True: "yes", False: "no", None: "-"}[w.get("passed")]
        out.append(f"| {w['trial']} | {w['agent']} | {w['model'] or '-'} | {w['grading']} | {score} | {verdict} | {findings} | "
                   f"{passed} | {w['exception'] or ''} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--datasets", default=str(DATASETS))
    a = ap.parse_args()
    for job in a.jobs:
        if not pathlib.Path(job).is_dir():
            continue
        rows = job_rows(job, a.datasets)
        (pathlib.Path(job) / "summary.json").write_text(json.dumps(rows, indent=1))
        print(f"## {pathlib.Path(job).name}\n\n{markdown(rows)}\n")
        missing = [w["trial"] for w in rows if w["grading"].startswith(MISSING_GRADER_MODE)]
        if missing:
            print(f"WARNING: no grader_mode event in {len(missing)} trial(s), how they were graded is unknown: {', '.join(missing)}\n")
        unsent = [w["trial"] for w in rows if w.get("predicted_verdict") == NOT_SUBMITTED]
        if unsent:
            print(f"{len(unsent)} trial(s) did not submit a review (no findings.json): {', '.join(unsent)}. The agent may have failed, "
                  "or written its review in its reply instead of the file. Check its log with `harbor view`, or in agent/ of the trial folder.\n")


if __name__ == "__main__":
    main()
