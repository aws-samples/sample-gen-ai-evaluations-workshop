#!/usr/bin/env bash
# validate_skills.sh — Validates SKILL.md and CHALLENGE.md structural requirements
#
# SKILL files must have:
#   - YAML frontmatter (starts with ---)
#   - Sections: Prerequisites, Learning Objectives, Setup
#   - At least one fenced code block (python or bash)
#   - 3-5 lesson sections
#   - Assessment criteria label
#   - No "Success criteria" terminology
#   - Warning if over 500 lines
#
# CHALLENGE files must have:
#   - Assessment Criteria section
#   - Scoring rubric

ERRORS=0

# SKILL structural validation
while IFS= read -r -d '' f; do
  echo "Checking: $f"
  head -1 "$f" | grep -q '^---$'         || { echo "  FAIL: no frontmatter"; ERRORS=$((ERRORS + 1)); }
  for s in "Prerequisites" "Learning Objectives" "Setup"; do
    grep -q "^## $s" "$f"                || { echo "  FAIL: missing '## $s'"; ERRORS=$((ERRORS + 1)); }
  done
  grep -q '```python\|```bash' "$f"      || { echo "  FAIL: no code blocks"; ERRORS=$((ERRORS + 1)); }

  # Line count
  lines=$(wc -l < "$f")
  [ "$lines" -gt 500 ] && echo "  WARN: $lines lines (over 500 limit)"

  # Section count (3-5)
  section_count=$(grep -c '^## Section\|^### Section' "$f" 2>/dev/null || true)
  if [ "$section_count" -gt 5 ]; then
    echo "  FAIL: $section_count lesson sections (max 5)"; ERRORS=$((ERRORS + 1))
  elif [ "$section_count" -lt 3 ] && [ "$section_count" -gt 0 ]; then
    echo "  WARN: Only $section_count lesson sections (recommend 3-5)"
  fi

  # Assessment criteria label
  if ! grep -q '\*\*Assessment criteria' "$f"; then
    echo "  WARN: No '**Assessment criteria:**' label found"
  fi

  # Wrong terminology
  if grep -q "Success criteria" "$f"; then
    echo "  FAIL: Uses 'Success criteria' instead of 'Assessment criteria'"; ERRORS=$((ERRORS + 1))
  fi

  # Non-standard section names
  if grep -q '^## What You Will Build\|^## What You Will Learn' "$f"; then
    echo "  FAIL: Non-standard section name (use '## Learning Objectives')"; ERRORS=$((ERRORS + 1))
  fi

  # CHALLENGE cross-references
  if [[ "$f" == *workload*SKILL* ]]; then
    grep -qi 'CHALLENGE-capstone' "$f" || echo "  WARN: No reference to CHALLENGE-capstone.md"
  fi
  if [[ "$f" == *framework*SKILL* ]]; then
    grep -qi 'CHALLENGE-deep-dive' "$f" || echo "  WARN: No reference to CHALLENGE-deep-dive.md"
  fi
done < <(find . -name 'SKILL*.md' -print0)

# Challenge validation
while IFS= read -r -d '' f; do
  echo "Checking: $f"
  grep -q 'Assessment criteria\|Assessment Criteria\|Criterion' "$f" || { echo "  FAIL: missing scoring rubric"; ERRORS=$((ERRORS + 1)); }
done < <(find . -name 'CHALLENGE*.md' -print0)

echo ""
[ "$ERRORS" -eq 0 ] && echo "✅ All SKILL and CHALLENGE files valid." || { echo "❌ $ERRORS error(s) found"; exit 1; }
