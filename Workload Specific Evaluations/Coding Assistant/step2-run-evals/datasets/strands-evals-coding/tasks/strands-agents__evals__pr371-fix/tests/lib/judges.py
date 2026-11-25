# Shared file. Edit datasets/_shared/coding/tests/lib/judges.py, then run `python datasets/_shared/sync.py`.
"""LLM judges and patch bloat. Usage: python3 judges.py rubric|taste|bloat

- rubric: the task rubric (tests/judge/rubric.toml). Each criterion is fail_to_pass (the change must do
  this) or pass_to_pass (the change must not break this). The judge reads the patch and the code, and
  scores each criterion 0 or 1. Writes rubric.json.
- taste: the global taste rubric. Ten yes/no checks: five compare the change with the reference patch,
  and five check that it fits the practices of the codebase. Each check passes or fails, with a reason.
  relative_taste and practice_alignment are the fractions of checks that pass. Writes taste.json.
- bloat: source lines that the agent's patch changes, divided by the source lines of the reference patch.
  Tests, documentation, and changelogs are not counted. Writes bloat.json.
"""

from __future__ import annotations

import re
import sys
import tomllib

from bedrock import LOGS, REPO, TESTS, Conversation, read_only_tools, write_json

REFERENCE = TESTS / 'judge' / 'reference.patch'
NOT_SOURCE = re.compile(r'(^|/)(tests?|docs?)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(md|rst|txt)$|CHANGELOG', re.I)


def agent_patch() -> str:
    return (LOGS / 'agent.patch').read_text()


def instruction() -> str:
    text = (TESTS / 'judge' / 'instruction.md').read_text()
    return text.split('## General instructions')[0].strip()


# ── Rubric ──────────────────────────────────────────────────────────────────

RUBRIC_SYSTEM = '''You grade a code change against a task rubric. You get the task instruction, the change (a diff),
and criteria. Read the code in /work/repo as you need: the change is applied there. For each criterion,
decide if the change fulfills it. Score 1 (fulfills) or 0 (does not), with a one-sentence reason. Judge the
behavior and the design that the criterion describes, not the names or the layout of the code. Different
correct designs fulfill a criterion. An empty change fulfills no fail_to_pass criterion. Read only the code
that you need to decide; most criteria need the changed functions and their callers only.'''


def rubric() -> None:
    criteria = tomllib.loads((TESTS / 'judge' / 'rubric.toml').read_text())['criteria']
    schema = {'type': 'object', 'properties': {'scores': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'id': {'type': 'string'}, 'score': {'type': 'integer', 'enum': [0, 1]},
                                         'reason': {'type': 'string'}}, 'required': ['id', 'score', 'reason']}}},
        'required': ['scores']}
    listing = '\n'.join(f"- `{c['id']}` ({c['kind']}): {c['text'].strip()}" for c in criteria)
    patch = agent_patch()
    convo = Conversation('rubric_judge', RUBRIC_SYSTEM, read_only_tools([REPO, LOGS]), 'submit_rubric_scores', schema)
    result = convo.ask(f'# Task\n{instruction()}\n\n# Change\n```diff\n{patch or "(empty: no change)"}\n```\n\n'
                       f'# Criteria\n{listing}\n\nScore every criterion.')
    by_id = {s['id']: s for s in result['scores']}
    scored = [{**c, **by_id.get(c['id'], {'score': 0, 'reason': 'not scored'})} for c in criteria]
    mean = lambda kind: (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)([s['score'] for s in scored if s['kind'] == kind])
    write_json('rubric.json', {'rubric_score': round(sum(s['score'] for s in scored) / len(scored), 4),
                               'fail_to_pass_score': mean('fail_to_pass'), 'pass_to_pass_score': mean('pass_to_pass'),
                               'criteria': scored, 'usage': convo.usage})


# ── Taste ───────────────────────────────────────────────────────────────────

TASTE_SYSTEM = '''You are a senior code reviewer. You grade the quality of a coding agent's change, not whether it
works (other checks test that). Before you grade, learn the conventions of the project: read its
contributing or style guides (for example AGENTS.md, CONTRIBUTING.md, STYLE_GUIDE.md), the code around the
change, and how the codebase handles the same concerns elsewhere. The change is applied in /work/repo.

Answer each check yes (passed: true) or no (passed: false), with a one-sentence reason. Each check is one
question. Answer yes when a maintainer would accept the change on that point without asking for
significant changes. A small nit is still yes.

Compared with the reference change, which a maintainer merged:
- minimality: Is the change focused on the task, with no unrelated changes or scope creep?
- approach_quality: Is it the right kind of solution, for example a root-cause fix and not a symptom patch?
- hygiene: Is it free of shortcuts, hard-coded values, dead code, and workarounds?
- fluency: Does it use the language, the libraries, and the domain concepts correctly?
- craftsmanship: Would a senior reviewer approve it as code that is easy to build on?

Fit with the codebase:
- style_consistency: Do the formatting, the naming, and the structure match the surrounding code?
- pattern_adherence: Does it use the established patterns and idioms of the project?
- library_usage: Does it reuse what the project already has, with no needless new dependency or hand-rolled helper?
- abstraction_level: Is the abstraction at the right level for this part of the code?
- documentation_fit: Do the comments and docstrings match the style and the density of the project? Fewer
  comments is better in a codebase that documents little.'''

RELATIVE = ('minimality', 'approach_quality', 'hygiene', 'fluency', 'craftsmanship')
PRACTICE = ('style_consistency', 'pattern_adherence', 'library_usage', 'abstraction_level', 'documentation_fit')


def taste() -> None:
    patch = agent_patch()
    if not patch.strip():
        write_json('taste.json', {'status': 'skipped', 'reason': 'empty patch', 'relative_taste': 0, 'practice_alignment': 0})
        return
    check = {'type': 'object', 'properties': {'passed': {'type': 'boolean'}, 'reason': {'type': 'string'}},
             'required': ['passed', 'reason']}
    schema = {'type': 'object', 'properties': {k: check for k in RELATIVE + PRACTICE}, 'required': list(RELATIVE + PRACTICE)}
    convo = Conversation('taste_judge', TASTE_SYSTEM, read_only_tools([REPO, LOGS]), 'submit_taste_checks', schema)
    result = convo.ask(f'# Task\n{instruction()}\n\n# Agent change\n```diff\n{patch}\n```\n\n'
                       f'# Reference change (merged by a maintainer)\n```diff\n{REFERENCE.read_text()}\n```\n\n'
                       'Explore the codebase, then answer all ten checks.')
    pass_rate = lambda keys: round(sum(bool(result[k]['passed']) for k in keys) / len(keys), 2)
    write_json('taste.json', {'status': 'scored', 'relative_taste': pass_rate(RELATIVE), 'practice_alignment': pass_rate(PRACTICE),
                              'dimensions': result, 'usage': convo.usage})


# ── Patch bloat ─────────────────────────────────────────────────────────────

def source_sloc(patch: str) -> int:
    """Changed source lines: added or removed, not blank, not a comment, not in tests or docs."""
    count, counted = 0, False
    for line in patch.splitlines():
        if line.startswith('diff --git'):
            path = line.split(' b/')[-1]
            counted = not NOT_SOURCE.search(path)
        elif counted and line[:1] in '+-' and not line.startswith(('+++', '---')):
            body = line[1:].strip()
            if body and not body.startswith('#'):
                count += 1
    return count


def bloat() -> None:
    agent, reference = source_sloc(agent_patch()), source_sloc(REFERENCE.read_text())
    write_json('bloat.json', {'agent_sloc': agent, 'reference_sloc': reference,
                              'patch_bloat': round(agent / reference, 3) if reference and agent else None})


if __name__ == '__main__':
    stage = sys.argv[1]
    try:
        {'rubric': rubric, 'taste': taste, 'bloat': bloat}[stage]()
        print(f'{stage}: done')
    except Exception as error:  # an infrastructure failure: the trial is invalid, not a fail for the agent
        write_json(f'{stage}.json', {'status': 'error', 'error': repr(error)})
        print(f'{stage} error: {error!r}')
