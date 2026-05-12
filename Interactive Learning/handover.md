# Interactive Learning Module — Handoff Documentation


**Status:** Working on `rename-dirs-hyphens` branch; PR to `main` pending

---

## ⚡ Getting Started (TL;DR)

```bash
git clone git@github.com:lukewma-aws/sample-gen-ai-evaluations-workshop.git
cd sample-gen-ai-evaluations-workshop
git checkout rename-dirs-hyphens
kiro-cli --agent tutor   # then say: "Teach me about operational metrics"
```

> **Use `rename-dirs-hyphens` branch.** The `main` branch is functional but behind — it has SKILL files but lacks root routing files (AGENTS.md, CLAUDE.md, .kiro/agents/tutor.json). The merge PR is pending.

---

## Table of Contents

- [What This Is](#what-this-is)
- [Quick-Start](#quick-start)
- [Repository & Branch Structure](#repository--branch-structure)
- [File Structure](#file-structure)
- [Architecture: Three Tools, One Source of Truth](#architecture-three-tools-one-source-of-truth)
- [The Tutor Prompt (AGENTS.md)](#the-tutor-prompt-agentsmd)
- [How SKILL Files Work](#how-skill-files-work)
- [The SKILL-BUILDER Meta-Tool](#the-skill-builder-meta-tool)
- [Pedagogical Flow](#pedagogical-flow)
- [Validation & QA](#validation--qa)
- [Environment Compatibility](#environment-compatibility)
- [Known Issues](#known-issues)
- [Done vs. Aspirational](#done-vs-aspirational)
- [What Needs Fixing Before Workshop-Ready](#what-needs-fixing-before-workshop-ready)
- [Teaching & Delivery](#teaching--delivery)
- [Relation to Jason's Curriculum-Builder](#relation-to-jasons-curriculum-builder)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Common Questions & Stumbling Points](#common-questions--stumbling-points)
- [Glossary](#glossary)
- [Video Walkthrough Script](#video-walkthrough-script)

---

## What This Is

An interactive learning mode for the **AWS Gen AI Evaluations Workshop**. Instead of reading Jupyter notebooks, learners interact with an AI tutor (via Kiro, Claude Code, or Codex) that uses Socratic questioning to test understanding through challenges — not lectures.

**The pitch:** "You've done the notebooks. Now prove you understand them — without looking."

This is NOT a replacement for the notebooks. It's a complement:

1. Learner completes traditional notebook modules (or a subset)
2. Opens Interactive Learning mode to test understanding
3. AI tutor asks questions, presents scenarios, challenges them to apply concepts
4. If they can't answer, the tutor guides them back to the relevant concept

---

## Quick-Start

```bash
# 1. Clone the fork
git clone git@github.com:lukewma-aws/sample-gen-ai-evaluations-workshop.git
cd sample-gen-ai-evaluations-workshop

# 2. Switch to the working branch
git checkout rename-dirs-hyphens

# 3. Open with your tool of choice:

# Option A: Kiro CLI
kiro-cli --agent tutor
# Then say: "Teach me about operational metrics"

# Option B: Claude Code
# Run `claude` from the repo root — CLAUDE.md auto-loads the tutor
claude
# Then say: "Quiz me on RAG evaluation"

# Option C: Codex (designed for, but UNTESTED)
# Open directory — root AGENTS.md should route to tutor
```

### Recommended Starting Prompts

| Learner Goal | Prompt |
|---|---|
| Explore what's available | "What modules can I learn about?" |
| Start a specific topic | "Teach me about RAG evaluation" |
| Test understanding | "Quiz me on operational metrics" |
| Challenge mode | "Give me the hardest challenge you have" |
| After completing modules | "I've done modules 01-03. What should I try next?" |

---

## Repository & Branch Structure

- **Fork:** https://github.com/lukewma-aws/sample-gen-ai-evaluations-workshop
- **Upstream:** https://github.com/aws-samples/sample-gen-ai-evaluations-workshop

| Branch | What's On It | Status |
|---|---|---|
| `rename-dirs-hyphens` | Latest everything — AGENTS.md v2+, root routing, all SKILLs, SKILL-BUILDER | **Working branch — clone this** |
| `main` | SKILL files, but older AGENTS.md, no root routing files | Behind; functional for SKILL content only |
| `zero-config-agent-setup` | Root-level agent config | Merged into rename-dirs-hyphens |
| `rename-dirs-hyphens-backup` | Backup before a risky operation | Archive |
| `aws-samples/main` | Upstream — no interactive learning content | Upstream |

> **Branch state:** `main` is behind by the root routing files and latest AGENTS.md. The interactive learning feature **does not fully work** on `main` because tool discovery (CLAUDE.md, root AGENTS.md, .kiro/agents/) is missing. Always use `rename-dirs-hyphens` until the PR merges.

---

## File Structure

```
sample-gen-ai-evaluations-workshop/          ← repo root
├── AGENTS.md                                ← thin router for Codex (points to Interactive Learning/AGENTS.md)
├── CLAUDE.md                                ← one line: @Interactive Learning/AGENTS.md
├── .kiro/agents/tutor.json                  ← Kiro agent config (prompt: file://Interactive Learning/AGENTS.md)
├── README.md                                ← repo README with Interactive Learning section
├── Interactive Learning/                    ← THE interactive module
│   ├── AGENTS.md                            ← FULL tutor prompt (110 lines, Socratic, process-loop)
│   ├── README.md                            ← learner-facing instructions
│   ├── CONTRIBUTING.md                      ← how to add new skills
│   ├── curriculum.md                        ← module dependency map (which modules prereq which)
│   ├── foundational evaluations/
│   │   ├── SKILL-operational.md             ← Operational metrics (latency, throughput, cost)
│   │   ├── SKILL-quality.md                 ← Quality metrics (relevance, coherence, faithfulness)
│   │   ├── SKILL-agentic.md                 ← Agentic evaluation (tool use, planning, multi-step)
│   │   └── SKILL-understanding-failures.md  ← Failure analysis (error taxonomy, root cause)
│   ├── workload evals/
│   │   ├── SKILL-structured-data.md
│   │   ├── SKILL-guardrails.md
│   │   ├── SKILL-rag-evaluation.md
│   │   ├── SKILL-speech-reasoning.md
│   │   ├── SKILL-chatbot.md
│   │   ├── SKILL-red-teaming.md
│   │   ├── SKILL-tool-calling.md
│   │   ├── SKILL-multiagent-context.md
│   │   └── CHALLENGE-capstone.md            ← cross-module capstone challenge
│   ├── framework evals/
│   │   ├── SKILL-promptfoo.md
│   │   ├── SKILL-agentcore.md
│   │   ├── SKILL-strands.md
│   │   ├── SKILL-dspy.md
│   │   ├── SKILL-mlflow.md
│   │   └── CHALLENGE-deep-dive.md           ← framework deep-dive challenge
│   └── meta/
│       ├── SKILL-BUILDER.md                 ← meta-tool for generating new SKILLs
│       └── validate_skills.sh               ← structural validation script
├── Foundational Evaluations/                ← source notebooks (01-04)
├── Workload Specific Evaluations/           ← source notebooks
└── Framework Specific Evaluations/          ← source notebooks
```

---

## Architecture: Three Tools, One Source of Truth

All three supported tools point to the **same file**: `Interactive Learning/AGENTS.md`.

| Tool | Entry Point | Mechanism | Status |
|---|---|---|---|
| Kiro | `.kiro/agents/tutor.json` | `"prompt": "file://Interactive Learning/AGENTS.md"` loads it as agent | ✅ Verified |
| Claude Code | `./CLAUDE.md` (root) | `@Interactive Learning/AGENTS.md` imports the full tutor prompt | ✅ Verified |
| Codex | `./AGENTS.md` (root) | Reads root AGENTS.md which says "read Interactive Learning/AGENTS.md" | ⚠️ Designed for, untested |

**Why this matters:** One file to maintain. No drift between tools. Edit `Interactive Learning/AGENTS.md` and all three tools get the update.

**Trade-off:** Single source of truth = single point of failure. If someone edits AGENTS.md to fix a quirk in one tool, it affects all three. Be careful with tool-specific workarounds.

---

## The Tutor Prompt (AGENTS.md)

The tutor prompt is ~110 lines and contains:

- **YAML frontmatter** — name/description for tool discovery
- **`<required>` blocks** — non-negotiable rules that survive context pressure
- **Per-Response Process:** classify → evaluate → formulate question → self-check
- **WITHHOLD rule:** never reveal methods/frameworks before student attempts
- **ONE-THING rule:** one concept, one question per response
- **Few-shot examples** — good and bad, showing correct Socratic behavior
- **Module Table** — maps topics → SKILL file paths
- **Escape hatches** — for skip requests, stuck learners, module jumping
- **Scaffolding protocol** — 3-strike hint rule (see Pedagogical Flow)
- **Session Process:** read SKILL silently → assess learner → pose first question → wait

### Key Structural Rules

| Rule | Purpose |
|---|---|
| `<required>` XML tags | Agent won't drop these even in long conversations |
| WITHHOLD | Prevents the agent from lecturing; forces learner to attempt first |
| ONE-THING | Prevents information overload; one concept per exchange |
| Few-shot examples | Most effective way to shape behavior; shows what NOT to do |
| Per-response process loop | Forces the agent to think before responding |
| 3-strike scaffolding | Prevents infinite loops when learner is stuck |

---

## How SKILL Files Work

Each SKILL file is a **tutor instruction set**, not student-facing content. The tutor reads it silently for context, then uses it to guide questioning.

SKILL file structure:
- YAML frontmatter (name, description)
- 3–5 sections max
- Assessment criteria with measurable thresholds
- Challenges requiring **novel application** (not repetition)
- "Why before how" pacing
- One concept per section

**Important:** Challenges are scored 0 if the learner just repeats taught content. They must apply concepts to a new scenario.

---

## The SKILL-BUILDER Meta-Tool

Located at `Interactive Learning/meta/SKILL-BUILDER.md`. It generates new SKILL files from source notebooks.

**To use:** Point an agent at SKILL-BUILDER.md with a notebook path, and it generates the SKILL.

What it does:
1. Takes a source notebook path as input
2. Reads the notebook content
3. Generates a SKILL file following strict structural rules
4. Has an override rule for multi-source compression (multiple notebooks → 1 SKILL)
5. Includes validation criteria and review dimensions

See `Interactive Learning/CONTRIBUTING.md` for the full workflow.

---

## Pedagogical Flow

```
1. Learner opens repo in AI tool
2. Tool discovers tutor (via root routing files)
3. Learner states what they want to learn
4. Tutor reads SKILL file silently (for its own context)
5. Tutor asks: "What do you already know about [topic]?"
6. Based on response, tutor calibrates:
   - Knows nothing → start from Section 1, basic questions
   - Knows basics → skip to Section 3, harder questions
   - Knows a lot → jump to Challenge
7. Per section: pose question → wait → evaluate → correct/advance
8. After final section: present the CHALLENGE
9. After challenge: suggest next module from curriculum.md
```

**Key principle:** 70% learner activity, 30% tutor guidance. The learner should be DOING more than READING.

### Scaffolding Protocol (Stuck Learners)

When a learner fails the same concept 3 times:

| Strike | Tutor Action |
|---|---|
| 1st wrong answer | Rephrase the question, offer a different angle |
| 2nd wrong answer | Provide a targeted hint (not the answer) |
| 3rd wrong answer | Give the answer, then ask the learner to explain it back in their own words. If they can't, redirect to prerequisite material via `curriculum.md` |

This prevents infinite loops while maintaining the Socratic approach.

---

## Validation & QA

### Structural Validation Script

**Location:** `Interactive Learning/meta/validate_skills.sh`

```bash
# Run from repo root:
bash "Interactive Learning/meta/validate_skills.sh"

# What it checks:
# - YAML frontmatter present in all SKILL files
# - Required sections exist (assessment criteria, challenges)
# - File naming convention (SKILL-*.md, CHALLENGE-*.md)
# - No orphaned SKILL files (all referenced in curriculum.md)

# Passing output looks like:
# ✓ SKILL-operational.md — valid
# ✓ SKILL-quality.md — valid
# ... (one line per file)
# All 17 SKILL files valid. 2 CHALLENGE files valid.
```

### Manual QA Checklist (Testing a SKILL File)

Use this when adding or modifying a SKILL file:

1. **Start a fresh session** — `kiro-cli --agent tutor` in the repo root
2. **Request the module** — "Teach me about [topic]"
3. **Verify assessment** — Does the tutor ask what you already know? (not lecture)
4. **Test wrong answer** — Give a deliberately wrong answer. Does it correct without revealing the full answer?
5. **Test advancement** — Give a correct answer. Does it move to the next section?
6. **Test scaffolding** — Fail 3 times. Does it provide hints then the answer?
7. **Test challenge** — Complete all sections. Is the challenge a novel scenario (not repetition)?
8. **Test escape hatch** — Say "just tell me the answer." Does it comply then ask you to explain back?

**Time:** ~10 min per SKILL file.

---

## Environment Compatibility

| Environment | Works? | Notes |
|---|---|---|
| Local + kiro-cli | ✅ Yes | Full functionality — discovers .kiro/agents/tutor.json |
| Local + Claude Code | ✅ Yes | CLAUDE.md `@import` loads the tutor prompt |
| Local + Codex | ⚠️ Untested | Root AGENTS.md should work but space-in-path unverified |
| SageMaker + Kiro Chat (built-in) | ❌ Partial | Does NOT auto-discover repo agents |
| SageMaker + kiro-cli (terminal) | ✅ Yes | Works via JupyterLab terminal |
| Workshop Studio | ⚠️ Unknown | AWS workshop delivery platform; depends on tool availability |

**SageMaker workaround:** Use kiro-cli through the JupyterLab terminal instead of the built-in Kiro Chat panel:
```bash
# In JupyterLab terminal:
cd ~/SageMaker/sample-gen-ai-evaluations-workshop
kiro-cli --agent tutor
```

---

## Known Issues

| Issue | Severity | Status | Workaround |
|---|---|---|---|
| Space in path (`Interactive Learning/`) | Medium | **Unverified** on Codex | kiro-cli and Claude Code confirmed working |
| SageMaker Kiro Chat doesn't discover agents | Medium | Known limitation | Use kiro-cli via terminal |
| Codex support untested | Low | Designed for, not verified | Root AGENTS.md exists; needs testing |
| `main` branch behind | Low | Intentional | PR pending; work on `rename-dirs-hyphens` |
| Some SKILL code examples may reference deprecated APIs | Low | Partially addressed | Oracle review caught major ones; edge cases may remain |
| SKILL file staleness | Low | No sync process | If source notebooks change, SKILLs must be manually regenerated |

---

## Done vs. Aspirational

| Feature | Status |
|---|---|
| 17 SKILL files covering all workshop modules | ✅ Done |
| 2 CHALLENGE files (capstone + deep-dive) | ✅ Done |
| Socratic tutor prompt with structural rules | ✅ Done |
| Root-level routing for 3 tools | ✅ Done |
| SKILL-BUILDER meta-tool | ✅ Done |
| Validation script | ✅ Done |
| Kiro CLI support | ✅ Done |
| Claude Code support | ✅ Done |
| Codex support | ⚠️ Untested (root AGENTS.md exists) |
| SageMaker Kiro Chat auto-discovery | ❌ Not supported (use terminal) |
| Learner progress tracking | ❌ Dropped (stateless by design) |
| Difficulty tiers per challenge | ❌ Deferred (v2) |
| Dynamic challenge generation | ❌ Aspirational |
| CI/CD validation of SKILLs | ❌ Deferred (validate_skills.sh exists but not in CI) |

---

## What Needs Fixing Before Workshop-Ready

| Item | Priority | Effort | Notes |
|---|---|---|---|
| Merge `rename-dirs-hyphens` → `main` | High | Low | PR needed; all content is ready |
| Test full learner flow end-to-end | High | Medium | Run through 2-3 modules using QA checklist above |
| Verify Codex space-in-path | Medium | Low | Test with Codex CLI; report back |
| SageMaker instructions in README | Medium | Low | Document the kiro-cli terminal workaround |
| Validate all SKILL code examples still work | Medium | High | APIs may have changed since generation |
| Record video walkthrough | High | Medium | See script below |
| Add scaffolding protocol to tutor prompt | Medium | Low | 3-strike rule (see Pedagogical Flow section) |

---

## Teaching & Delivery

### For Workshop Facilitators

- Introduce it **AFTER** at least Module 01-02 are complete
- Frame it as "office hours with an AI expert" — not "another tutorial"
- Emphasize: the tutor will NOT give you answers — it will ask you questions
- Let learners choose their module (`Interactive Learning/curriculum.md` shows dependencies)
- Expect **15-30 min per module** in interactive mode

### For Self-Paced Learners

1. Clone the fork, checkout `rename-dirs-hyphens`
2. Open in Kiro (kiro-cli) or Claude Code
3. Say "I want to learn about [topic]" or "Test me on quality metrics"
4. The tutor handles the rest

### What Happens When a Learner Starts

The tutor reads the relevant SKILL file, assesses what the learner knows, then poses questions. It **never dumps content**.

---

## Relation to Jason's Curriculum-Builder

Jason (jsgs@) built `curriculum-builder` — a two-tier system with an author-module skill (phased workflow) and a learner agent (guided/mentor modes, learner profiles, progress tracking). 

### Jason's 7-Phase Authoring Workflow

The `author-module` skill walks through a collaborative 7-phase process for converting source material into learning modules:

| Phase | What It Produces | Gate |
|---|---|---|
| 1. **Scope** | One-paragraph scope agreement (position, purpose, outcome, boundaries) | User agrees to scope |
| 2. **Narrative Arc** | Section-by-section outline pairing concepts with build tasks, ordered as a story | User approves outline |
| 3. **Draft Sections** | Individual section drafts with concept explanation + build task (exact starter content) | Each section reviewed |
| 4. **Challenge Design** | Capstone exercise with 4-6 assessment criteria | Challenge integrates module content |
| 5. **Assemble & Review** | Complete SKILL.md following template (under 500 lines) | Follows template exactly |
| 6. **Place the Module** | Module installed in correct directory with all supporting files | Checklist complete |
| 7. **Close the Loop** | Surface improvements to templates, skills, or guide pedagogy | Explicit reflection |

Key rules enforced during drafting: **why before how** (motivation precedes mechanics), **one concept at a time** (no bundling), **let it land** (each concept gets its own turn).

### What Was Borrowed

- SKILL-as-lesson-plan pattern (each module = a SKILL.md file)
- Socratic delivery (agent asks, doesn't tell)
- Challenge-at-end-of-module pattern
- Assessment criteria per challenge

### What Was Changed/Dropped

| Jason's System | Luke's System | Why |
|---|---|---|
| Author-module workflow (7 phases) | SKILL-BUILDER (simpler meta-tool) | Less overhead for contributors |
| Guided/mentor modes | Single Socratic mode with adaptive difficulty | Simpler; one mode that adjusts |
| Learner profiles/progress tracking | Stateless (each session starts fresh) | Workshop setting; no persistence needed |
| Feedback files | Inline assessment during conversation | Less file management |

### What Was Added

- **Structural rules** in the tutor prompt (`<required>`, WITHHOLD, ONE-THING) to prevent lecture mode
- **Few-shot examples** showing correct vs incorrect tutor behavior
- **Per-response process loop** (classify → evaluate → question → self-check)
- **Root-level routing** for multi-tool support (Kiro, Claude Code, Codex)

**Key insight:** Jason's system trusts the agent to be Socratic by instruction. This system **structurally forces it** — every response must end with a question, content is withheld until the learner attempts, and bad examples show what NOT to do.

---

## Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Stateless (no learner profile) | Simpler for workshop setting; learners won't persist across sessions |
| Single AGENTS.md as source of truth | Avoids drift between tools; one file to maintain |
| SKILL files are tutor instructions, not student-facing | Prevents the agent from just reading content aloud |
| `<required>` XML tags | Survives context pressure — agent won't drop these rules even in long conversations |
| Few-shot examples in the prompt | Most effective way to shape agent behavior; worth the token cost |
| SKILL-BUILDER as meta-tool | Anyone can generate new SKILLs without understanding the full system |
| Challenges require novel application | Explicitly scored 0 if they just repeat taught content |
| Space in directory name | Matches upstream convention (`Foundational Evaluations/` etc.); root routing files handle tool discovery |

---

## Common Questions & Stumbling Points

| Question/Issue | Answer |
|---|---|
| "The tutor just reads me the content" | The current AGENTS.md has structural rules preventing this. If it still happens, the SKILL file may need restructuring. |
| "How do I know which module to pick?" | Read `Interactive Learning/curriculum.md` or ask the tutor "What's available?" |
| "It won't let me advance" | The tutor gates progression. Demonstrate understanding (explain in own words, solve a mini-problem) to advance. |
| "I want the answer, not more questions" | Say so — the escape hatch gives the answer but then asks you to explain it back. |
| "Does this work in SageMaker?" | Kiro Chat (built-in) won't auto-discover agents. Use kiro-cli via the JupyterLab terminal. |
| "Can I add my own module?" | Yes — use the SKILL-BUILDER. Read `Interactive Learning/CONTRIBUTING.md`. |
| "The paths are broken" | Verify you're on the correct branch. If paths still don't resolve, check whether files were moved — path errors can be valid after restructuring. |

---

## Glossary

| Term | Meaning |
|---|---|
| **kiro-cli** | Command-line AI assistant (like Claude Code). Run with `kiro-cli --agent tutor` in a repo directory. |
| **Claude Code** | Anthropic's coding assistant. Reads `CLAUDE.md` at repo root for project context. Run with `claude` from the repo root. |
| **Codex** | OpenAI's coding assistant. Reads `AGENTS.md` at repo root for agent discovery. |
| **`@import`** | Claude Code syntax — `@path/to/file` in CLAUDE.md includes that file's content in the agent's context. |
| **Workshop Studio** | AWS internal platform for delivering hands-on workshops. Provides pre-configured environments. |
| **SKILL file** | A markdown file containing tutor instructions for one module. NOT student-facing content. |
| **SKILL-BUILDER** | A meta-tool (itself a markdown prompt) that generates new SKILL files from source notebooks. |
| **`<required>` blocks** | XML tags in the tutor prompt that signal "never drop this rule" to the AI model. |
| **curriculum.md** | Located at `Interactive Learning/curriculum.md`. Maps module dependencies (which modules prereq which). |

---

## Video Walkthrough Script

**Duration:** 5-8 minutes  
**Environment:** Local machine with kiro-cli

### Outline

1. **Intro (30s)** — "This is the interactive learning mode for the evals workshop." Show the repo structure briefly.

2. **Architecture (60s)** — Show root AGENTS.md, CLAUDE.md, .kiro/agents/tutor.json. Explain single source of truth. Open `Interactive Learning/AGENTS.md` — highlight `<required>` block and process loop.

3. **Live demo — starting a session (90s)** — Open kiro-cli. Type: "Teach me about quality metrics." Show the tutor asking what you already know (NOT lecturing). Give a partial answer, show it probing deeper.

4. **Live demo — Socratic flow (90s)** — Show predict/explain/debug question. Give a wrong answer — show correction without revealing full answer. Give a right answer — show advancement.

5. **Live demo — challenge mode (60s)** — Complete a module (or skip to challenge). Show CHALLENGE presented as scenario. Show assessment criteria revealed after attempt.

6. **Adding new content (45s)** — Show SKILL-BUILDER briefly. "Point it at a notebook, it generates a SKILL file." Show validate_skills.sh.

7. **Known limitations (45s)** — SageMaker: use terminal. Codex: untested. Space in path: unverified on Codex.

8. **Wrap-up (30s)** — "Clone the fork, open in kiro-cli or Claude Code. The tutor handles the rest."
