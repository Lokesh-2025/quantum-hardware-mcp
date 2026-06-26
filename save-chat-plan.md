# Plan: Add `/save` Command to agent/chat.js

## Top-Level Overview

Add a `/save` slash command to the CLI chat interface (`agent/chat.js`) that lets the
user write the current in-memory chat history to a Markdown file.

**Scope:**
- Single file change: `agent/chat.js`
- No new modules, no server-side changes

**Approach:**
- Parse `/save` (and optional `@/path` argument) inside `processInput()`
- If the path argument is absent, prompt the user interactively using `rl.question`
- An empty reply to the prompt cancels the save
- If the target file already exists, warn and ask for confirmation before overwriting
- If history is empty, inform the user and skip
- Write the chat as a Markdown file using the already-imported `fs` and `path` modules

---

## Sub-Tasks

---

### Sub-Task 1 — Add `saveChatToFile()` helper function

**Intent:**  
Encapsulate all Markdown serialisation and file-write logic in a single async function,
keeping `processInput()` clean.

**Expected Outcomes:**
- A new `async function saveChatToFile(filePath)` exists in `agent/chat.js`
- It serialises `chatHistory` as Markdown: each entry becomes a `## User` or
  `## Assistant` heading followed by the content, entries separated by `---`
- It uses the already-imported `fs.writeFile` and `path.resolve` (no new imports)
- It prints a success confirmation on completion

**Todo List:**
1. Add `saveChatToFile(filePath)` above `processInput()`
2. Resolve the path with `path.resolve(process.cwd(), filePath)` to support relative paths
3. Build the Markdown string: for each `chatHistory` entry emit
   `## User\n\n{content}` or `## Assistant\n\n{content}`, entries separated by `\n\n---\n\n`
4. Call `fs.writeFile(resolvedPath, markdown, 'utf-8')`
5. Print a success confirmation: `✅ Chat saved to: <resolvedPath>`

**Relevant Context:**
- `chatHistory` — `agent/chat.js` line 9, array of `{ role, content }`
- `fs` — `agent/chat.js` line 4, already `require('fs').promises`
- `path` — `agent/chat.js` line 5, already imported

**Status:** [ ] pending

---

### Sub-Task 2 — Add `/save` branch in `processInput()`

**Intent:**  
Detect `/save` (exact case), extract the optional `@/path` argument, handle empty
history, prompt when the path is missing, confirm before overwriting, and call
`saveChatToFile()`.

**Expected Outcomes:**
- Input matching `/save` (with or without a trailing `@/...` argument) is intercepted
  before the `default` branch sends it to the LLM
- An exact-case check is used (not the lowercased `trimmed` variable)
- If `chatHistory` is empty, print a notice and return without saving
- If no path argument is supplied, `rl.question` prompts for one; an empty reply
  cancels with a brief message
- If the resolved file already exists (`fs.stat` succeeds), print a warning and prompt
  `Overwrite? (y/N)>`; anything other than `y`/`Y` cancels
- On I/O error, print an error message (do not crash the process)
- After handling (save, cancel, or error), return `true` so the main loop continues

**Todo List:**
1. At the top of `processInput()`, before the `switch`, add an exact-case check:
   `if (input.trim().startsWith('/save')) { ... return true; }`
2. Parse the argument: split `input.trim()` on the first whitespace; if a second token
   exists and starts with `@`, strip the `@` prefix — **and `.trim()` the result** —
   to get the file path with no leading or trailing whitespace
3. If path came from `rl.question`, also `.trim()` the raw answer before use; an
   empty string after trimming means cancel
4. If history is empty, print `⚠️  Nothing to save — chat history is empty.` and return
5. Check if the file exists with `fs.stat`; if it does, prompt
   `⚠️  File exists. Overwrite? (y/N)> `; cancel unless the trimmed answer is `y`/`Y`
6. Call `await saveChatToFile(filePath)`, wrapping in try/catch to print errors

**Relevant Context:**
- `processInput()` — `agent/chat.js` lines 192-221
- Existing `rl.question` Promise pattern — `agent/chat.js` lines 107-111
- Exact-case requirement and trim-all-paths requirement confirmed by user

**Status:** [ ] pending

---

### Sub-Task 3 — Update `displayWelcome()` help text

**Intent:**  
Document the new command so users know it exists and understand the syntax.

**Expected Outcomes:**
- `displayWelcome()` includes a line for `/save` alongside the existing command list
- The format matches the existing help lines exactly

**Todo List:**
1. In `displayWelcome()` (lines 167-185), add after the `/help` line:
   `  - Type "/save @/path/to/file" to save chat history to a Markdown file`

**Relevant Context:**
- `displayWelcome()` — `agent/chat.js` lines 167-185

**Status:** [ ] pending
