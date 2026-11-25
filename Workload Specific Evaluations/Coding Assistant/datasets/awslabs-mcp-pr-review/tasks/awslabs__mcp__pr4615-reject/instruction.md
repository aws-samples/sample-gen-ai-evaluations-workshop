# Review this change

You are reviewing one change to this repository. The repository is in `/work/repo`, at its base
state with the change applied as uncommitted edits: `git diff` shows the change, and the same diff is
in `/work/change.patch`. The Python dependencies of the changed package are installed (`python` and
`pytest` are on `PATH`), so you can run its tests and try inputs. Read the change and review it as a
competent reviewer would.

Report the defects a reviewer should catch. For each defect, give:

- the file and line it occurs on,
- a one-sentence statement of the problem,
- a severity — one of `low`, `medium`, or `high` (only `high` defects block a merge).

Then give an overall verdict: **approve** the change, or **request changes**. Request changes
when, and only when, the change contains at least one blocking (high-severity) defect.

Write your findings as a JSON array to `/logs/artifacts/findings.json`, each entry an object with
`path`, `line`, `statement`, and `severity`. Judge the change on its own merits: do not assume a
defect is present, and do not assume the change is correct.
