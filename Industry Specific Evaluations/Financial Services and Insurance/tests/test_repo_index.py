"""The workshop's module lists name this module, so a reader of the front pages can find it.

The repository root README lists each Industry Specific module as one bullet
("  - Name: what it evaluates."), and the Industry Specific Evaluations README
has a table row per module. Both must include Financial Services and Insurance.
"""

import re
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
INDUSTRY_README = MODULE_ROOT.parent / "README.md"
ROOT_README = MODULE_ROOT.parent.parent / "README.md"
MODULE_NAME = "Financial Services and Insurance"


def test_root_readme_lists_the_module_under_industry_specific_evaluations():
    text = ROOT_README.read_text()
    section = text[text.index("[Industry Specific Evaluations]"):]
    section = section[:section.index("\n- ####")]
    assert re.search(rf"^  - {MODULE_NAME}: \S.*\.$", section, re.MULTILINE), section


def test_industry_readme_has_a_row_for_the_module():
    assert f"| [{MODULE_NAME}](Financial%20Services%20and%20Insurance/) |" in INDUSTRY_README.read_text()
