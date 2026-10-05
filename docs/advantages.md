# Saturn — what to lean into

_2026-09-28. A note on the advantages that a local-first, terminal-native agent has over a cloud
agent, and the work that turns each one into something a user feels. Companion to `pivot.md`
(the ranked list, this folder); this file says why those items matter, not when to do them._

## 1. Latency that scales with the request (adaptive thinking) — shipped 2026-09-28 (`runtime.think`)

The v2 loop gets the floor right: a chat question is one model call, a lookup is two.
Until 2026-09-28 it did not scale the *depth* of a pass: think was hard-off everywhere, so a
genuinely hard multi-step task got the same zero-reasoning call as "what time is it" and paid
for it in extra passes, hygiene bounces, or a wrong plan.

The lever is **adaptive thinking**:

- A pass thinks, with a bounded budget, only when the tool round just before it had an error:
  a tool failure or a hygiene refusal (unknown tool, bad arguments, a repeat). That is the one
  decision the model has to rethink. Since 2026-09-29 this is the whole rule: the first
  version also thought after a `plan` call, a declined call, or pass three, and the loop
  benchmark caught it swallowing answers.
- Every other pass runs think-off: a chat question, a clean lookup, the answer after a declined
  call, and the wrap-up answer of a multi-step task. On qwen3.5 (4b and 9b) a thinking pass
  whose right move is a short answer writes the answer inside its reasoning and emits nothing,
  so a thinking pass that comes back empty is rerun once think-off.
- The cap pass (from `runtime.max_iterations` on) stays think-off and keeps its tools bound so
  the cached prefix holds: a call it emits is refused and the model answers from what it has;
  only a model that calls again is rerun with tools unbound.

Cost on the chat shape: zero. Cost on a hard shape: one bounded thinking budget where it helps.
This is a bounded change in `nodes/agent.py` plus a config knob, and it is the only item in the
backlog that changes the latency *shape* rather than adding a feature.

Two things to confirm alongside it: the idle prime still covers the bound agent lineage after
the v2 cut (the tool catalog is ~6k tokens of prefix; cached it is free, cold it is ~15 s at
400 tok/s) — confirmed, `test_prime_lineage_is_the_bound_agent_prefix` pins it — and a
multi-call batch in `nodes/tools.py` runs concurrently, not serially. It runs serially, and
deliberately: the per-call egress slice (`_egress_slice`) attributes ledger events to the call
that produced them by sequence, which a concurrent batch would scramble. Making it concurrent
means per-call ledger tagging first.

## 2. Local-first

Nothing leaves the machine unless the user lets it. Ollama is the only backend; a remote
`OLLAMA_HOST` is the one network boundary and it is on the ledger. This gives Saturn three things
a cloud agent cannot promise:

- **No account, no key, no bill, no outage.** It works on a plane. Latency is the machine's, not
  a queue's.
- **Every byte out is a decision.** The egress ledger, the air-gap, and quarantine of untrusted
  content are not compliance features; they are what makes the next two sections safe to want.
- **The whole computer is in scope.** Files, shell, Apple Notes / Calendar / Mail / Contacts /
  Reminders / Messages, Shortcuts, launchd, MCP.
  A cloud agent reaches these through a bridge and a consent screen; Saturn is already on the
  right side of the door. Both gaps that stood here closed 2026-09-29: the file tools work in
  the launch folder plus `/add-dir` folders (pivot #1) and `read_file` reads PDF / .docx / .xlsx
  directly (pivot #2), so "summarize this PDF" is a one-line request.

## 3. Terminal-native

The audience is people who live in a terminal. Lean into what that gives:

- **The work is visible.** The rail shows every call and its result as it lands; Esc pauses into
  continue / steer / abort. A chat window cannot show a plan being executed the way a rail can.
- **It composes.** `saturn -q "…"` on stdout, `--json`, `--export`, `--replay`, pipes in and out.
  Every turn is a record, so every turn can be replayed or diffed.
- **It is fast to start and cheap to keep open.** Import time is under a second; the model stays
  loaded between turns. A launch brief (pivot #6) turns "open the terminal" into "here is your
  day".
- **It no longer looks like an audit console.** From 2026-09-28 bare `/help` listed five
  commands with the auditor's ones behind `/help --all` (pivot #11); dogfooding reversed that
  on 2026-10-05 — the short list hid commands people went looking for, so `/help` lists them all.
- **It is yours to shape the way Claude Code is.** A global `SATURN.md` and hooks shipped (pivot
  #7, #10) and any Shortcut is a tool (`run_shortcut`, the Mac-native half of #9); markdown
  skills (#8) and script-file tools are still open. Same file formats, same vocabulary, pointed at a life instead of a
  codebase.

## 4. Memory — the thing local-first unlocks

This is the advantage to lean on hardest. A cloud agent has to be careful about what it is told;
a local agent does not. **Because nothing leaves the machine, the user can tell Saturn
everything** — their calendar, their mail, the names of the people in their life, what they are
worried about, what they decided about the lease — and the agent can know them the way a good
assistant does. Knowing the user is the feature, not a privacy concession.

What exists: six memory layers in one markdown file, `user`, `commitments` and the last five
`memo` entries loaded every turn, the rest by token match, `sens=` facts withheld whenever inference is not local, every fact
provenance-stamped, learning gated behind a review.

What is missing is the *rate* at which it learns:

- **Facts the user states in their own words should land without a click** (pivot #4). "I'm
  vegetarian", "Petra is my manager", "my lease ends in March" come from the trusted principal
  and go straight to `user` / `entities` / `commitments` with `by=user`; the rail shows a one-line
  `· remembered: …` leaf and `/memory remove` undoes it. Only *inferred* facts and compaction
  summaries keep the review queue. A tool result must never plant a memory; that gate stays.
- **The first run should be an interview, not a model picker** (pivot #5, shipped 2026-10-05).
  Three questions offered after the model pick, five under `/memory setup`, and the second
  turn already knows who it is talking to.
- **Contacts and Reminders as readers** (pivot #3, shipped 2026-10-01) give the memory something
  to resolve against: "Petra" is now an address and a number.

The pitch, in one line: *the agent that can be told everything, because it keeps everything
here.* Every other advantage in this file exists to make that line safe.

## 5. What else local-first unlocks

Beyond memory, the things a local agent can do that a cloud agent structurally cannot, ranked by
daily value against what already exists in the repo.

1. **Index everything, not a curated knowledge base.** A cloud agent gets the documents the user
   chooses to upload. A local one can index the whole home directory: the mail archive, Notes,
   Downloads, the PDFs never filed. "What did I decide about the lease" is answered from an
   email two years old. The RAG store and loaders exist and the embedder pulls lazily; the
   change is scope — point it at `~` with an exclusion list instead of `database/documents`, and
   index in the background at idle.
2. **Free background compute.** Cloud tokens cost money, so a cloud agent does nothing between
   messages. A local GPU is idle most of the night. That pays for overnight reindexing, a
   precomputed launch brief, a weekly memo digest, and memory candidates proposed from the day's
   mail and calendar for the morning review. The launchd seam in `notify/` already schedules
   one-shot jobs.
3. **Drafts in the user's own voice.** It can read the sent folder, so `draft_mail` can be
   few-shot primed with how the user actually writes to that recipient. No cloud agent gets the
   sent folder. `read_mail` and `draft_mail` exist; the change is a prompt-side sample of recent
   sent messages to the same person.
4. **The logged-in apps are the integrations.** No OAuth dance, no token stored on someone's
   server. Notes, Calendar, Mail, Contacts, Reminders, Messages, the front browser tab, the Finder
   selection and Shortcuts already work this way through AppleScript. The same pattern
   reaches Photos metadata, Finder tags, and the
   Keychain via the `security` CLI. Each is a hundred-line reader module.
5. **Ambient awareness.** A local process can watch the filesystem, the clipboard, and the
   frontmost app. "Fix this" can mean what is on the clipboard; a new file in Downloads can
   prompt "file this?". A cloud agent cannot see the machine between messages. The menu bar
   agent already outlives the terminal (off by default since 2026-09-29), so it has somewhere to live.
6. **Shell composition.** The agent lives where the user already works: `git diff | saturn -q
   "summarize"`, `saturn` in a cron job, `!cmd` passthrough in the REPL, `@file` mentions, shell
   completions. `-q`, `--json` and `core/mentions.py` exist; the rest is small ergonomics that
   make it feel native rather than hosted. (`!cmd` passthrough shipped 2026-09-28.)
7. **Any script on disk is a tool** (pivot #9). A cloud agent needs a hosted MCP server for a
   custom tool. Locally a shell script with a frontmatter is enough, and it still faces the gate.
8. **The user's data stays in files they own.** Memory is one markdown file, traces are sqlite,
   config is YAML. The user can grep, edit and git-version what the agent knows; a cloud agent's
   memory is opaque. Saturn should say this out loud and `/memory` should show the path.
   (`/memory` shows the path since 2026-09-28.)
9. **Unlimited retries and self-checks.** With no per-token bill, a hard turn can afford a
   second sample, a verification pass, or the adaptive-thinking budget from §1. Only latency is
   spent, and only on hard shapes.

Pursue 1, 3 and 2 first. They compound with §4: the agent knows the user's files, writes like
them, and does its homework while they sleep. (`research.md`, 2026-10-01, weighed these: 3 is
the next plan to write, 1 is declined in favour of Spotlight search plus plain files, 2 becomes
routines, and 5 is declined.)
