<!-- TOOL: SKILL-BUILDER | VERSION: 1.0 | INPUTS: source notebooks + structured input + this file | OUTPUT: one SKILL or CHALLENGE .md file -->

# SKILL-BUILDER

This file is a generation tool. Feed it — along with source notebooks and a structured input — to an AI agent to produce a SKILL or CHALLENGE file for the workshop.

## Inputs

To generate a SKILL, you need three things:

1. **Source notebooks** — Jupyter notebooks from the workshop modules:
   - `../Foundational Evaluations/` (modules 01–04)
   - `../Workload Specific Evaluations/` (workload-specific topics)
   - `../Framework Specific Evaluations/` (framework-specific topics)
2. **A structured input** — tells the generator how to map notebooks to sections (template below)
3. **A reference SKILL** — an existing SKILL file to match in tone and structure

## Reference SKILL Selection

Pick the closest match to your target:

| Target Type | Reference File |
|-------------|----------------|
| Foundational SKILL | `foundational evaluations/SKILL-quality.md` |
| Workload SKILL | `workload evals/SKILL-guardrails.md` |
| Framework SKILL | `framework evals/SKILL-promptfoo.md` |
| CHALLENGE | `workload evals/CHALLENGE-capstone.md` |

## Generation Rules

Every SKILL file MUST have:

- **YAML frontmatter** with `name`, `description`, and activation phrases
- **Standard sections** in this order:
  1. `## Prerequisites`
  2. `## Learning Objectives`
  3. `## Setup` (with runnable code block)
  4. `## Section 1:` through `## Section N:` (3–5 lesson sections)
  5. `## Challenges` (with `**Assessment criteria:**` labels)
  6. `## Wrap-Up` (must reference the correct CHALLENGE file)
- **3–5 lesson sections** — never fewer, never more
- **"Assessment criteria"** — never "Success criteria"
- **At least one** `python` or `bash` fenced code block
- **≤500 lines** preferred (quality > brevity — exceeding is acceptable if content demands it)
- **Challenges must require novel application** — not just repeating the taught workflow on different data. Include at least one decision the learner wasn't explicitly taught (e.g., handling ambiguity, resolving conflicts, adapting when the approach doesn't fit cleanly)
- **Match sibling SKILLs' framework/API surface** — if other SKILLs in the same category use a specific framework (e.g., boto3 + bedrock.converse()), your SKILL should use the same. If the source notebook uses a different framework, adapt the code examples to match the category's established pattern.

## Three Warnings

These are the failure modes we discovered through testing. Skip them and the output breaks.

### 1. Override Rule

The structured input is **advisory, not binding**. If the source material reveals that two notebooks teach genuinely different concepts (even though the structured input groups them), split them into separate sections and document why.

**What breaks without this:** Agents force-fit unrelated notebooks into one section, producing incoherent lessons that jump between topics.

### 2. Multi-Source Compression

When multiple notebooks map to one SKILL, group sections by **concept taught** — not by notebook number.

**What breaks without this:** Agents follow file order (notebook 01 → section 1, notebook 02 → section 2), producing sections that don't build on each other pedagogically.

### 3. Code Completeness

Every code block must be **copy-paste runnable**. No `...`, no `# your code here`, no missing imports. If setup code is long, put it in the Setup section and reference it.

**What breaks without this:** Learners hit immediate errors on the first code block. Trust in the SKILL collapses.

## Structured Input Template

Fill this out before generating. It tells the agent how to compress source material into sections.

```markdown
## Structured Input: [Module Name]

**Source notebooks:** [list with relative paths]
**Category:** foundational evaluations | workload evals | framework evals
**Output type:** SKILL | CHALLENGE
**Reference SKILL:** [which existing SKILL to match]

**Section mapping (≤5 sections):**
1. [Section title] ← notebooks [X, Y] — [what concept this teaches]
2. [Section title] ← notebook [Z] — [what concept this teaches]
3. ...

**Compression decisions:**
- [Which notebooks merge and why]
- [What gets dropped and why]

**Challenge design:**
- [What the challenge tests]
- [Assessment criteria (measurable, achievable with taught material only)]

**Cross-references:**
- Wrap-Up points to: [CHALLENGE file]
- Prerequisites: [other SKILLs needed first]
```

## Structured Input Example

Here's a filled example for a 4-notebook module:

```markdown
## Structured Input: Red Teaming for GenAI Applications

**Source notebooks:**
- ../Workload Specific Evaluations/Red Teaming/01-LLM-App-Red-Teaming/notebook.ipynb
- ../Workload Specific Evaluations/Red Teaming/02-Bedrock-Guardrails-Red-Teaming/notebook.ipynb
- ../Workload Specific Evaluations/Red Teaming/03-RAG-Red-Teaming/notebook.ipynb
- ../Workload Specific Evaluations/Red Teaming/04-Agent-Red-Teaming/notebook.ipynb

**Category:** workload evals
**Output type:** SKILL
**Reference SKILL:** workload evals/SKILL-guardrails.md

**Section mapping (5 sections):**
1. Red Teaming Concepts and Promptfoo Configuration ← shared setup from all READMEs — teaches attack taxonomy + tooling setup
2. Attacking LLM Applications ← notebook 01 — teaches plugins, strategies, graders
3. Stress-Testing Bedrock Guardrails ← notebook 02 — teaches guardrail-specific adversarial config
4. Red Teaming RAG Pipelines ← notebook 03 — teaches indirect injection, context poisoning
5. Red Teaming Agentic Systems ← notebook 04 — teaches tool misuse, privilege escalation

**Compression decisions:**
- No compression needed — 4 notebooks + 1 intro section = exactly 5
- Each notebook is a self-contained attack surface; natural 1:1 mapping

**Challenge design:**
- Design a red team test suite for a given application type
- Assessment criteria: identifies ≥3 vulnerability categories, produces runnable Promptfoo config, includes at least one custom grader

**Cross-references:**
- Wrap-Up points to: CHALLENGE-capstone.md
- Prerequisites: SKILL-guardrails.md (understands what guardrails do before attacking them)
```

## Invocation Sequence

Follow these steps exactly:

```
1. GATHER
   Identify source notebooks for the module.
   → e.g., Workload Specific Evaluations/Red Teaming/*.ipynb (4 notebooks)

2. WRITE STRUCTURED INPUT
   Fill the template above.
   → Map notebooks to ≤5 sections, document compression decisions.

3. GENERATE
   Feed an AI agent ALL of:
   - This entire SKILL-BUILDER.md file
   - The structured input you wrote
   - The source notebook contents (full text)
   - The reference SKILL file
   → Prompt: "Generate a SKILL following this spec."

4. VALIDATE
   bash meta/validate_skills.sh output.md
   → Fix any structural errors. Re-run until 0 errors.

5. REVIEW
   Score against the 6 dimensions below (self or peer).
   → Pass ≥9/12, no 0s. If fail, revise and re-validate.
```

## Iteration

The first draft will need 1–2 revision passes. This is normal.

Common failures to watch for:
- Code blocks missing imports (add them to Setup or inline)
- Sections exceeding one screen height (~60 lines) — split or trim
- Challenge difficulty miscalibrated (too easy = just repeats teaching; too hard = requires untaught concepts)
- Wrap-Up missing CHALLENGE cross-reference

Re-run `bash meta/validate_skills.sh` after each revision.

## Validation

Run the structural validator:

```bash
bash meta/validate_skills.sh path/to/SKILL-name.md
```

**What it checks:**
- YAML frontmatter present
- Required sections (Prerequisites, Learning Objectives, Setup, Challenges, Wrap-Up)
- 3–5 lesson section headings
- At least one `python` or `bash` code block
- `**Assessment criteria:**` label present
- No "Success criteria" anywhere
- ≤500 lines (warning, not error)
- CHALLENGE cross-references in Wrap-Up

**What it does NOT check:**
- Pedagogical quality (use Review Criteria below)
- Code correctness or API accuracy
- Whether code blocks actually run
- Content depth or learner experience

## Review Criteria

After validation passes, score the SKILL on 6 dimensions:

| # | Dimension | 0 (Blocking) | 1 (Functional) | 2 (Publication-ready) |
|---|-----------|--------------|-----------------|------------------------|
| 1 | Technical Accuracy | Wrong APIs, hallucinated imports, code won't run | Minor issues (deprecated method, missing edge case) | All code correct, imports verified, APIs current |
| 2 | Pedagogical Flow | No motivation, concepts jumbled, no progression | Mostly ordered but some jumps | Why-before-how, one concept/section, smooth ramp |
| 3 | Completeness | Learning objectives unaddressed | Most covered, one gap | Every objective taught and assessed |
| 4 | Challenge Quality | Trivial (just repeats teaching) or impossible | Achievable but predictable | Beyond teaching, novel application, measurable criteria |
| 5 | Cross-Set Consistency | Wrong terminology, broken cross-refs | Minor tone differences | Matches voice, terms, and refs of sibling SKILLs |
| 6 | Learner Experience | Can't complete without external help | Completable with some guessing | Clear path, errors anticipated, "why" explained |

**Pass:** ≥9/12 total, no single 0.

**Who reviews:** The generating agent self-scores first. Then a second agent (or human) validates the scores. Disagreements on any 0 or 2 must be resolved before shipping.
