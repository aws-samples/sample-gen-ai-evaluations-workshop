"""LLM judges and patch bloat. Usage: python3 judges.py rubric|taste|bloat

- rubric: the task rubric (tests/judge/rubric.toml). Each criterion is fail_to_pass (the change must do
  this) or pass_to_pass (the change must not break this). The judge reads the patch and the code, and
  scores each criterion 0 or 1. Writes rubric.json.
- taste: the global taste rubric. Five dimensions relative to the reference patch, and five dimensions of
  alignment with the practices of the codebase. Each is scored 1-5. Writes taste.json.
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

Score each dimension as an integer 1-5, with a short reason.

Relative taste: compare with the reference change, which a maintainer merged. 4 = on par with the
reference. 5 = clearly better than the reference. 3 = acceptable but weaker. 2 = a reviewer would ask for
significant changes. 1 = a reviewer would reject it.
- minimality: focused on the task, no unrelated changes or scope creep.
- approach_quality: the right kind of solution, for example a root-cause fix, not a symptom patch.
- hygiene: no shortcuts, hard-coded values, dead code, or workarounds.
- fluency: correct use of the language, libraries, and domain concepts.
- craftsmanship: makes later work easier, as a senior reviewer would approve.

Codebase practice alignment (absolute): 5 = reads as if a regular contributor wrote it. 1 = foreign to
the codebase.
- style_consistency: formatting, naming, and structure match the surrounding code.
- pattern_adherence: uses the established patterns and idioms of the project.
- library_usage: reuses what the project already has, no needless new dependency or hand-rolled helper.
- abstraction_level: the right level of abstraction for this part of the code.
- documentation_fit: comments and docstrings match the style and density of the project. Fewer comments is
  better in a codebase that documents little.'''

RELATIVE = ('minimality', 'approach_quality', 'hygiene', 'fluency', 'craftsmanship')
PRACTICE = ('style_consistency', 'pattern_adherence', 'library_usage', 'abstraction_level', 'documentation_fit')


def taste() -> None:
    patch = agent_patch()
    if not patch.strip():
        write_json('taste.json', {'status': 'skipped', 'reason': 'empty patch', 'relative_taste': 0, 'practice_alignment': 0})
        return
    dim = {'type': 'object', 'properties': {'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                                            'reason': {'type': 'string'}}, 'required': ['score', 'reason']}
    schema = {'type': 'object', 'properties': {k: dim for k in RELATIVE + PRACTICE}, 'required': list(RELATIVE + PRACTICE)}
    convo = Conversation('taste_judge', TASTE_SYSTEM, read_only_tools([REPO, LOGS]), 'submit_taste_scores', schema)
    result = convo.ask(f'# Task\n{instruction()}\n\n# Agent change\n```diff\n{patch}\n```\n\n'
                       f'# Reference change (merged by a maintainer)\n```diff\n{REFERENCE.read_text()}\n```\n\n'
                       'Explore the codebase, then submit scores for all ten dimensions.')
    mean = lambda keys: round(sum(result[k]['score'] for k in keys) / len(keys), 2)
    write_json('taste.json', {'status': 'scored', 'relative_taste': mean(RELATIVE), 'practice_alignment': mean(PRACTICE),
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
