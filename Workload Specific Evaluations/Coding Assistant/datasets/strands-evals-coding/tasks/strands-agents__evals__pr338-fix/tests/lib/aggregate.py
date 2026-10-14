"""Combine the stages into reward.json, after the "tasteful solve" of Senior SWE-Bench.

Gates (all must hold for `tasteful_solve`):
- verifiers: every hidden fail_to_pass and pass_to_pass test, and every base unit test, passes.
- validation: the validation judge accepted the tests, and every story passes.
- rubric_score > 0.5, patch_bloat < 2, practice_alignment > 2, relative_taste > 2.

`correct` is the first two gates only: the change works. `reward` is `tasteful_solve`.
`agent_tests_*` are reported, but they do not gate.

If an LLM stage failed (for example, Bedrock throttling) or the validation judge discarded the trial, the
trial is invalid: there is no reward.json, so Harbor reports a reward-file error and the trial is not
counted as a fail of the agent. reward_details.json has the reason.

Usage: python3 aggregate.py <hidden test module name>
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

LOGS = Path('/logs/verifier')
MODELS = tomllib.loads(Path('/tests/judge/models.toml').read_text())
GATES = MODELS['gates']


def read(name: str) -> dict:
    path = LOGS / name
    return json.loads(path.read_text()) if path.exists() else {'status': 'missing'}


def verifiers(hidden: str) -> tuple[dict, list[str]]:
    groups: dict[str, list[bool]] = {'fail_to_pass': [], 'pass_to_pass': [], 'existing_suite': []}
    failed = []
    junit = LOGS / 'junit.xml'
    if junit.exists():
        for case in ET.parse(junit).getroot().iter('testcase'):  # noqa: S314 (pytest wrote it in this run)
            if case.find('skipped') is not None:
                continue
            name, classname = case.get('name', ''), case.get('classname', '')
            ok = case.find('failure') is None and case.find('error') is None
            if hidden in classname.split('.'):
                group = 'fail_to_pass' if name.startswith('test_f2p_') else 'pass_to_pass'
            else:
                group = 'existing_suite'
            groups[group].append(ok)
            if not ok:
                failed.append(f'{group}: {classname}::{name}')
    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    complete = all(groups.values())
    return {**{k: rate(v) for k, v in groups.items()}, 'pass': complete and all(all(v) for v in groups.values()),
            'complete': complete}, failed


def main(hidden: str) -> None:
    verify, failed = verifiers(hidden)
    validation, rubric, taste, bloat, agent_tests = (read(n) for n in (
        'validation.json', 'rubric.json', 'taste.json', 'bloat.json', 'agent_tests.json'))
    (LOGS / 'failed_tests.txt').write_text(''.join(f'{line}\n' for line in failed))

    invalid = []
    if not verify['complete']:
        invalid.append('the verifier tests did not run (see pytest.log)')
    if validation.get('status') != 'accepted':
        invalid.append(f"validation {validation.get('status')}: {validation.get('discard_reason') or validation.get('error', '')}")
    for name, stage in (('rubric', rubric), ('taste', taste)):
        if stage.get('status') in ('error', 'missing'):
            invalid.append(f"{name} judge {stage.get('status')}: {stage.get('error', '')}")

    validation_pass = validation.get('status') == 'accepted' and validation.get('passed_stories') == validation.get('total_stories')
    correct = verify['pass'] and validation_pass
    bloat_ratio = bloat.get('patch_bloat')
    checks = {
        'verifiers': verify['pass'],
        'validation': validation_pass,
        'rubric': (rubric.get('rubric_score') or 0) > GATES['rubric_min'],
        'patch_bloat': bloat_ratio is not None and bloat_ratio < GATES['bloat_max'],
        'practice_alignment': (taste.get('practice_alignment') or 0) > GATES['taste_min'],
        'relative_taste': (taste.get('relative_taste') or 0) > GATES['taste_min'],
    }
    tasteful = all(checks.values())

    flat = {
        'reward': float(tasteful), 'tasteful_solve': float(tasteful), 'correct': float(correct),
        'verifiers_pass': float(verify['pass']), 'fail_to_pass': verify['fail_to_pass'],
        'pass_to_pass': verify['pass_to_pass'], 'existing_suite': verify['existing_suite'],
        'validation_score': float(validation.get('validation_score') or 0),
        'rubric_score': float(rubric.get('rubric_score') or 0),
        'relative_taste': float(taste.get('relative_taste') or 0),
        'practice_alignment': float(taste.get('practice_alignment') or 0),
        'patch_bloat': float(bloat_ratio or 0),
        'agent_tests_added': float(agent_tests.get('tests_added') or 0),
        'agent_tests_reproduce_bug': float(agent_tests.get('reproduces_bug') or 0),
    }
    models = {role: os.environ.get('VERIFIER_MODEL_ID') or MODELS[role]['model']
              for role in ('validation_agent', 'validation_judge', 'rubric_judge', 'taste_judge')}
    details = {'invalid': invalid, 'models': models, 'checks': checks, 'flat': flat, 'verifiers': verify, 'failed_tests': failed,
               'validation': validation, 'rubric': rubric, 'taste': taste, 'bloat': bloat, 'agent_tests': agent_tests}
    (LOGS / 'reward_details.json').write_text(json.dumps(details, indent=2))

    if invalid:
        (LOGS / 'reward.json').unlink(missing_ok=True)
        (LOGS / 'reward.txt').write_text('')
        print('INVALID TRIAL:', '; '.join(invalid))
    else:
        (LOGS / 'reward.json').write_text(json.dumps(flat))
        print(json.dumps(flat))
    print('checks:', json.dumps(checks))


if __name__ == '__main__':
    main(sys.argv[1])
