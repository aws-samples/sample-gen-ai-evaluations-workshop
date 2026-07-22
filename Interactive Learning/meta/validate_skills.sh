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

  # Likert-scale regression guard — the workshop teaches binary pass/fail, not 1-5 rating scales.
  # Matches prescriptive scoring mechanics (rubric instructions, JSON score fields, X/5 output),
  # NOT prose that mentions 1-5 as an anti-pattern (those use an en-dash "1–5").
  # NOTE: SKILL-quality.md is a known pending exception (notebooks 02 vs 03 contradict; conversion
  # awaiting a design decision) — it WARNs instead of failing until converted.
  if grep -qiE 'score each.*1-[0-9]|"score":[[:space:]]*<?[0-9]-[0-9]|[[:space:]]X/[0-9]|scored 1-[0-9]|on a 1-[0-9] scale' "$f"; then
    if [ "$(basename "$f")" = "SKILL-quality.md" ]; then
      echo "  WARN: Likert 1-5 scoring present (SKILL-quality.md — conversion pending decision)"
    else
      echo "  FAIL: Likert 1-5 scoring detected — use binary pass/fail (see AGENTS.md)"; ERRORS=$((ERRORS + 1))
    fi
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
done < <(find . -name 'SKILL*.md' -not -path '*/meta/*' -print0)

# Challenge validation
while IFS= read -r -d '' f; do
  echo "Checking: $f"
  grep -q 'Assessment criteria\|Assessment Criteria\|Criterion' "$f" || { echo "  FAIL: missing scoring rubric"; ERRORS=$((ERRORS + 1)); }
done < <(find . -name 'CHALLENGE*.md' -not -path '*/meta/*' -print0)

echo ""
[ "$ERRORS" -eq 0 ] && echo "✅ All SKILL and CHALLENGE files valid." || { echo "❌ $ERRORS error(s) found"; exit 1; }
