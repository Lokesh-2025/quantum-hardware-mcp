# Project: Quantum Hardware MCP Server

## FIRST THING — read memory before answering anything

Before starting ANY task, read these files in order:
1. `/Users/lokeshpullakandam/.claude/projects/-Users-lokeshpullakandam-quantum-hardware-mcp/memory/journal.md` — what happened last session, what's next
2. `/Users/lokeshpullakandam/.claude/projects/-Users-lokeshpullakandam-quantum-hardware-mcp/memory/agent_prompt.md` — full context: who Lokesh is, who Jack is, every result, every decision
3. Then confirm in one line: "Checked CLAUDE.md + memory, ready."

If you skip this, you will repeat things Lokesh has already decided, ask questions already answered, and waste his time. Do not skip it.

---

## What this project is

An open-source MCP server (Python) giving AI assistants live access to real quantum hardware — IBM Quantum, IonQ, AWS Braket. Built by Lokesh Pullakandam, a recent CIS grad learning in public, in collaboration with Jack Woehr (IBM Quantum veteran, Qiskit contributor).

**Current state:** 34 tools, 19 backends, listed on Glama/mcp.so/PulseMCP. Has real hardware results (178.8× amplitude amplification on ibm_fez). Jack runs it on his own machine.

---

## How to work (efficiency rules)

- ONE feature per session. Finish, test, stop. Never start extra work Lokesh didn't ask for.
- Keep answers short. No long explanations unless he asks. Code > talk.
- Before any task: state a 2-3 step plan in one line each. Wait for OK if it touches more than 2 files.
- Don't re-read files you haven't changed. Don't re-run tests that already passed.
- Smallest possible change that works. No refactors, no "improvements" he didn't ask for.
- Minimal dependencies: MCP SDK + qiskit-ibm-runtime. Ask before adding anything else.
- If stuck after 3 attempts, STOP and explain the problem in plain English. Do not loop.

---

## Learning rules (Lokesh is still learning QC)

- Prefer simple over clever. Comment WHY, never WHAT.
- When he asks "explain", use plain English and short analogies. No jargon.
- After finishing a feature: 1-line summary + one interview question about it.
- Translate Jack's technical messages into Telugu movie / kids story analogy first, then explain practically.

---

## Safety rules (non-negotiable)

- NEVER put IBM API token in code. Use .env, keep .env in .gitignore. Check .gitignore before every commit.
- Never run rm -rf or force push without asking.
- git commit after each working feature with a clear message. Never commit broken code.

---

## Definition of done (per feature)

Code works + tested + committed + README updated + 1-line summary to Lokesh.

---

## Auto memory update rule

When the session is ending or context is getting long:
1. Update `journal.md` — what happened, what was decided, what's next
2. Update `agent_prompt.md` if any major facts changed (new results, new Jack context, new career decisions)
3. Tell Lokesh: "I updated the journal. Here's what I saved: [3 bullet points]"

Do this without being asked.
