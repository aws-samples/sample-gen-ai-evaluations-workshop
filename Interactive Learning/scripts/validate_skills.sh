#!/usr/bin/env bash
# validate_skills.sh — Validates SKILL.md and CHALLENGE.md structural requirements
#
# SKILL files must have:
#   - YAML frontmatter (starts with ---)
#   - Sections: Prerequisites, Learning Objectives, Setup
#   - At least one fenced code block (python or bash)
#   - Warning if over 500 lines
#
# CHALLENGE files must have:
#   - Assessment Criteria section
#   - Scoring rubric

set -euo pipefail; ERRORS=0

# SKILL structural validation
for f in $(find . -name 'SKILL*.md'); do
  head -1 "$f" | grep -q '^---$'         || { echo "FAIL: $f no frontmatter"; ((ERRORS++)); }
  for s in "Prerequisites" "Learning Objectives" "Setup"; do
    grep -q "^## $s" "$f"                || { echo "FAIL: $f missing '## $s'"; ((ERRORS++)); }
  done
  grep -q '```python\|```bash' "$f"      || { echo "FAIL: $f no code blocks"; ((ERRORS++)); }
  lines=$(wc -l < "$f")
  [ "$lines" -gt 500 ] && echo "WARN: $f has $lines lines (over 500 limit)"
done

# Challenge validation
for f in $(find . -name 'CHALLENGE*.md'); do
  grep -q "Assessment Criteria" "$f"                              || { echo "FAIL: $f missing Assessment Criteria"; ((ERRORS++)); }
  grep -q 'Assessment criteria\|Assessment Criteria\|Criterion' "$f" || { echo "FAIL: $f missing scoring rubric"; ((ERRORS++)); }
done

[ "$ERRORS" -eq 0 ] && echo "All SKILL and CHALLENGE files valid." || { echo "$ERRORS error(s)"; exit 1; }
