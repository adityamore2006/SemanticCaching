---
name: code-analysis-interactive
description: Analyze AI-generated code architecture step by step, one logical chunk at a time, with pauses for your questions at each stage. Use this whenever you want to understand how a piece of code is structured, what its components do, and how they fit together. Perfect for reviewing code from Claude Code, breaking down complex functions, or learning unfamiliar patterns. The skill groups code into logical chunks by responsibility or flow, explains what matters in each chunk, shows how chunks connect, then synthesizes the full architecture.
---

# Code Analysis: Interactive Architecture Walkthrough

A skill for understanding AI-generated code by breaking it into logical chunks and walking through each one step by step, with pauses for your questions.

## Overview

You paste code into Claude Code (or describe the code you want analyzed), and this skill:
1. Breaks the code into logical chunks (responsibility-based or flow-based, auto-detected)
2. Walks you through each chunk one at a time
3. Highlights what matters (dependencies, state changes, error handling, role in architecture)
4. Stops after each chunk so you can ask questions
5. After all chunks, synthesizes the full architecture with diagrams and observations

You learn the *architecture*, not just what each line does.

## Module 1: Chunk Identification & Grouping

**What happens before you see anything:**

When code arrives, scan it for structure:
- Functions, classes, hooks, state initialization, entry points
- Determine if the code is **responsibility-based** (functions cluster by what they do: all auth, all fetching) or **flow-based** (procedural steps: request → parse → validate → respond)
- Group accordingly:
  - **Responsibility-based**: Functions that do related things go in one chunk
  - **Flow-based**: Sequential steps get chunked in order; responsibility groups nest within flow steps
- Lump helper functions (`formatDate()`, `sanitize()`, etc.) with the chunk that calls them, not as orphans
- Use best judgment; no micro-chunks

**Result:** You'll see 3-7 chunks (depends on code size), ordered logically.

---

## Module 2: One-Chunk-at-a-Time Loop

**For each chunk:**

1. **Point out what matters** (don't reproduce the code)
   - Reference the chunk by name, line numbers, or function signature
   - Explain what it does in 1-2 sentences
   - Call out the key details (see Module 3)

2. **Explain its role in architecture**
   - Is it an entry point? Utility? Handler? Orchestrator?
   - What does it depend on?
   - What depends on it?

3. **Leave space for questions**
   - Stop after explaining
   - Wait for you to ask, clarify, or say "next"
   - Don't ask "do you have questions?" just... pause

4. **Mention how it connects to the next chunk** (if applicable)
   - Use connector types from Module 4 (direct call, data dependency, execution order, etc.)
   - Flag coupling issues if they exist

---

## Module 3: What to Highlight

For each chunk, highlight what actually matters—don't be rigid about categories. Use best judgment:

- **Function signatures**: inputs, outputs, async/sync, hooks, side effects
- **State mutations**: what the chunk changes (variables, DB, API, DOM, cache)
- **Dependencies**: what it calls (other functions, APIs, libraries, external services)
- **Error handling**: try/catch, validation, error codes, defaults, failure paths
- **Conditionals & branches**: if/else, loops, ternaries, guard clauses—and why they exist
- **Architecture role**: entry point vs utility vs orchestrator vs leaf

**Not a checklist.** Some chunks are all about error handling. Others are about orchestration. Read the code and highlight what teaches understanding.

---

## Module 4: Architectural Connectors

When moving between chunks, name or flag the relationship:

1. **Direct call**: Chunk A calls Chunk B
2. **Data dependency**: Chunk A needs output from Chunk B (but doesn't call it)
3. **Execution order**: Chunk A must run before Chunk B (state coupling)
4. **Side effect coupling**: Chunk A's action triggers Chunk B's behavior
5. **Shared context**: Both chunks read/write the same state or object
6. **Independent**: No coupling, work in isolation

Flag tight coupling, circular dependencies, or shared state issues when you spot them.

---

## Module 5: Synthesis & Full Architecture

After you've walked through all chunks, do this:

**Step 1: Full architecture diagram**
- Visual showing all chunks and connections (boxes + arrows, not code)
- Entry point clear, data flow visible

**Step 2: Data flow narrative**
- "Here's what happens when you call `main()`: input flows through validation → processing → output"
- Connect chunks into a story

**Step 3: Coupling hotspots**
- Flag shared state, circular dependencies, overly tight coupling

**Step 4: Key observations**
- Patterns ("This uses a pipeline"), strengths, weaknesses, oddities
- What stands out?

**Step 5: Answer high-level questions**
- "Why was it structured this way?" "Is this scalable?" "Should we refactor?"
- Only after chunks are understood—never during

---

## Workflow: What You Do

1. **Paste code or describe it** to Claude Code
2. **Invoke this skill** (mention "analyze code architecture" or "walk me through this code")
3. **For each chunk:**
   - Read my explanation (no code reproduced, just key details)
   - Ask questions or say "next"
   - I stop and wait for you
4. **After all chunks**, I show the full diagram and synthesis
5. **Ask architecture questions** ("why?", "should we refactor?", etc.)

---

## Tips for Best Results

- **Bring real code**: Copy/paste from VS Code, GitHub, or Claude Code artifacts
- **State your goal**: "I want to understand the auth flow" or "Is this scalable?" shapes how I chunk and explain
- **Ask mid-way**: If a chunk doesn't make sense, ask—don't wait until the end
- **Shallow vs deep**: You can zoom in on one chunk or zoom out for the full picture

---

## What This Skill Is NOT

- Not a code review or linter
- Not "make this code better" (that's a different skill)
- Not for detailed line-by-line explanation (you read the code yourself)
- Not for teaching a language (assumes you know the syntax)

This skill is for *understanding architecture and how pieces fit together*.