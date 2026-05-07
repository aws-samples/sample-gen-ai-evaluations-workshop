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

# --- New checks from oracle review process ---

for file in $(find . -name 'SKILL*.md' -o -name 'CHALLENGE*.md'); do
  # Check for wrong terminology
  if grep -q "Success criteria" "$file"; then
      echo "  FAIL: Uses 'Success criteria' instead of 'Assessment criteria'"
      errors=$((errors + 1))
  fi

  # Check section count (SKILL files only)
  if [[ "$file" == *SKILL* ]]; then
      section_count=$(grep -c '^## Section\|^### Section' "$file" 2>/dev/null || echo 0)
      if [ "$section_count" -gt 5 ]; then
          echo "  FAIL: $section_count lesson sections (max 5)"
          errors=$((errors + 1))
      elif [ "$section_count" -lt 3 ] && [ "$section_count" -gt 0 ]; then
          echo "  WARN: Only $section_count lesson sections (recommend 3-5)"
      fi
  fi

  # Check for non-standard section names
  if grep -q '^## What You Will Build\|^## What You Will Learn' "$file"; then
      echo "  FAIL: Non-standard section name (use '## Learning Objectives')"
      errors=$((errors + 1))
  fi

  # Check Assessment criteria label in SKILL files
  if [[ "$file" == *SKILL* ]]; then
      if ! grep -q '\*\*Assessment criteria' "$file"; then
          echo "  WARN: No '**Assessment criteria:**' label found in challenges"
      fi
  fi

  # Check CHALLENGE cross-references in Wrap-Up
  if [[ "$file" == *workload*SKILL* ]]; then
      if ! grep -qi 'CHALLENGE-capstone' "$file"; then
          echo "  WARN: Workload SKILL doesn't reference CHALLENGE-capstone.md in Wrap-Up"
      fi
  fi
  if [[ "$file" == *framework*SKILL* ]]; then
      if ! grep -qi 'CHALLENGE-deep-dive' "$file"; then
          echo "  WARN: Framework SKILL doesn't reference CHALLENGE-deep-dive.md in Wrap-Up"
      fi
  fi
done

[ "$ERRORS" -eq 0 ] && echo "All SKILL and CHALLENGE files valid." || { echo "$ERRORS error(s)"; exit 1; }
