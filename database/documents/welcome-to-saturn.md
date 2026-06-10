# Welcome to Saturn

Saturn (Saturday.ai) is a **local-first, transparent, general-purpose agent** that runs on your own
machine. You talk to it in a chat loop; it plans its work out loud, calls tools, and shows you exactly
what it did. Nothing leaves your computer unless a tool you can see and approve sends it there.

This document is part of Saturn's knowledge base — it's here so that asking *"what can Saturn do?"* or
*"how do I add my own documents?"* returns a real answer on a fresh install. It's the only file Saturn
ships with. Once you've read it, you can remove it with `/docs remove welcome-to-saturn.md` and replace it
with your own corpus.

---

## What makes Saturn different

- **Local-first.** Models run through Ollama on your hardware by default. Web search falls back to a
  keyless provider, so Saturn works with no API keys at all. Your documents, workspace, and memory live
  in plain files on disk.
- **Transparent.** Saturn drafts a **plan** before it acts and keeps it visible the whole turn — a live
  rail of steps you can watch, pause, and edit. You can see every tool call, its arguments, and its
  result.
- **In your control.** Risky actions (writing files, running shell commands) pause at an **approval
  gate** that shows you the exact change — a diff for file writes, the full command for the shell —
  before anything happens.

## How a turn works

Each time you send a message, Saturn runs a **living-plan loop**:

1. **Ground** — gathers context: your profile, durable memory, what documents and workspace files exist,
   and a recap of the recent conversation.
2. **Plan** — drafts a short, numbered plan of steps to answer you.
3. **Act** — works the plan one step at a time, calling tools as needed. You see each step light up as
   it runs.
4. **Synthesize** — composes the final answer from what it gathered, and streams it back token by token.

If Saturn drifts or the plan needs changing, you can steer it mid-turn (see *Steering*, below). If it
tries to answer without doing a step it planned, it corrects itself. The plan is both the transparency
surface and a real driver of behavior — better plans, better answers.

## What Saturn can do (tools)

Saturn has a built-in toolbox. It decides which to use; you approve anything risky.

| Tool | What it does |
|---|---|
| `calculate` | Precise arithmetic and math. |
| `web_search` | Search the web (Tavily if you set `TAVILY_API_KEY`, otherwise keyless DuckDuckGo). |
| `web_extract` | Pull the readable text out of a web page (local, no key). |
| `deep_research` | Multi-source research over several pages, bounded by a time budget. |
| `read_file` / `write_file` / `edit_file` | Read, create, and surgically edit files in your **workspace** (see below). Writes are snapshotted first, so `/undo` can revert them. |
| `list_directory` / `search_files` / `find_files` | Browse the workspace, grep file contents, find files by name. |
| `search_knowledge_base` | Semantic search over your ingested **documents** (this is RAG). |
| `remember` / `recall` | Save and look up durable facts across conversations. |
| `run_shell` | Run a shell command in the workspace. **Always** asks for approval first. |

## The two folders you work with

Saturn reads and writes inside `database/`, and there are two folders that are *yours*:

- **`database/documents/`** — your **knowledge base**. Drop PDFs, text, or markdown files here (or use
  `/docs add <path>`) and Saturn embeds them so `search_knowledge_base` can retrieve from them. This is
  how you give Saturn things to *know*.
- **`database/workspace/`** — the **sandbox** where the file tools read and write. When you ask Saturn to
  create, edit, or save a file, it lives here. This is how you give Saturn a place to *work*.

Everything else under `database/` (the vector cache, checkpoints, memory, saved sessions) is managed for
you.

## Adding your own knowledge

```
/docs                        # list what's currently ingested
/docs add path/to/file.pdf   # add a document to the knowledge base
/docs remove some-doc.md     # remove one
/docs sync --force           # rebuild the whole knowledge base from disk
```

The first time Saturn embeds documents it can take a moment — it's running an embedding model locally.
After that, only changed files are re-embedded.

## Useful slash commands

Anything you type starting with `/` is a command, not a message to the agent.

- `/help` — list every command; `/help <name>` for details.
- `/plan` — view the plan and execution mode; pause or edit it mid-turn.
- `/trace` — drill into what just happened: tool inputs/outputs, the model's reasoning, and cost.
- `/tools` — list available tools and their risk level.
- `/models`, `/config`, `/system` — see and tune which models run and how.
- `/context`, `/compact` — manage the conversation's context window.
- `/memory` — see, add, or delete the facts Saturn permanently remembers.
- `/undo` — revert the file changes of the last turn that wrote anything.
- `/init` — survey your workspace and draft `SATURDAY.md`, standing instructions Saturn loads
  every turn.
- `/save`, `/load`, `/resume` — keep and restore conversations.
- `/risk`, `/allow`, `/autoapprove` — tune the approval gate (use the last with care).
- `/update` — pull the latest Saturn (your data is never touched).
- `/clear` — start a fresh conversation; `/quit` — exit.

Run `/config setup` (or `/config doctor`) for a first-run health check: it tells you whether Ollama is
up, whether your models are pulled, and whether any keys are set — with the fix for each gap.

## Approvals and safety

Saturn sorts tools by risk. Read-only tools (search, calculate, read a file) just run. Tools that change
something — writing a file, running a shell command — **pause and ask you first**, showing the exact
change. You approve or reject each one. Rejecting cleanly tells Saturn to find another way rather than
re-asking.

## Steering and pausing

While Saturn is working you can:

- **Type a correction and press Esc** — it's injected as guidance and the running turn adjusts course.
- **Press Esc on an empty line** — pause at the next step boundary to inspect or edit the plan.
- **Type ahead** — queue your next question or command while the agent is still working.

## Mention files inline

Type `@` followed by a path (with fuzzy completion) to pull a file's contents straight into your
message — handy for "summarize `@notes.txt`" without ingesting it.

---

**That's the tour.** Ask Saturn anything, watch the plan, and check `/trace` when you're curious how it
got there. When you're ready to make the knowledge base your own, `/docs remove welcome-to-saturn.md` and
start ingesting your own documents.
