## Task

`ToolSelectionAccuracyEvaluator` gives us false negatives on some multi-agent traces, with no warning.

our setup: a coordinator agent hands parts of the request to sub-agents. in some exports every agent
invocation span in the trace has no parent (the trace is flat). on those traces the judge rejects
tool calls that are correct, for example the `calculate` call that takes the difference of two
temperatures. the reason the judge gives makes it look like it thinks the user only asked about one
city.

the coordinator always starts first, but it is not always first in the span list. when we move the
coordinator span to the front of the list, the false negatives go away.

the evaluator must judge each tool call against what the user actually asked, whatever the order of
the spans.

## General instructions

- The code repo is at `/work/repo`.
- You are inside of a Docker container. The Python dependencies of the repository are installed, and
  `python` and `pytest` are on `PATH`. The network is not available, so you cannot install packages,
  and you cannot call a model on Amazon Bedrock.
- You are expected to act autonomously as a software engineer to complete tasks you are given.
- Do not stop until you feel you have completed the task and your code changes can be merged.
- You may need to use software engineering skills like analyzing the codebase, running the tests, and
  building small reproductions to complete the task. Not all tasks will be solvable by reading source
  code alone.
