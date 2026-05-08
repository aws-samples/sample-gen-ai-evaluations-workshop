# Contributing a SKILL or CHALLENGE

This directory contains AI-tutored skill files for the AWS GenAI Evaluations Workshop. Each SKILL teaches one module interactively; each CHALLENGE integrates concepts across multiple SKILLs.

## Adding a New SKILL

Use the generation tool: [`meta/SKILL-BUILDER.md`](meta/SKILL-BUILDER.md)

It walks you through the full workflow — from gathering source notebooks to producing a validated SKILL file.

## Quick Reference

### Validate

```bash
bash meta/validate_skills.sh path/to/your-SKILL.md
```

### Review (6 Dimensions, 0–2 each)

| Dimension | What to Check |
|-----------|---------------|
| Technical Accuracy | APIs correct? Code runs without modification? |
| Pedagogical Flow | Motivation before instruction? One concept per section? |
| Completeness | All learning objectives addressed and assessed? |
| Challenge Quality | Beyond teaching? Achievable with taught material only? |
| Cross-Set Consistency | Matching tone, terminology, correct cross-references? |
| Learner Experience | Completable unaided? "Why" before "how"? |

**Pass threshold:** ≥9/12 total, no single 0.

### When to Create a CHALLENGE

If a category (foundational, workload, framework) has 3+ SKILLs, add a CHALLENGE file that integrates concepts across them. Workload SKILLs reference `CHALLENGE-capstone.md`; framework SKILLs reference `CHALLENGE-deep-dive.md`.

### Module Map

See [`curriculum.md`](curriculum.md) for the full dependency graph and module inventory.
