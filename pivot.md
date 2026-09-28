# Saturn v2 — the pivot

_2026-09-27. Written the day the plan/execute engine was replaced by one loop
(`docs/superpowers/specs/2026-09-27-v2-react-loop-design.md`). This file states the new goal
and ranks the work that closes the distance to it. It replaces `PLAN.md`, the v1 strategy
document (deleted 2026-09-27)._

## The goal

**Saturn is Claude Code for daily life: a local companion you can hand your personal world to.**

Three things follow from "local" that a cloud agent cannot promise, and the product leans on all
three:

1. **You can tell it everything.** Your calendar, mail, notes, files, the names of the people in
   your life, what you're worried about — none of it leaves the machine, so the agent can know you
   the way a good assistant does. Knowing the user is the feature, not a privacy concession.
2. **It is yours to shape.** Standing instructions, your own procedures, your own tools, your own
   tone — the same customization surface Claude Code gives developers (CLAUDE.md, skills, hooks,
   MCP), pointed at a life instead of a codebase.
3. **Every action is visible and gated.** The trust stack stays: you watch it work in the rail,
   every risky action asks first, every byte that leaves is on the ledger, every run replays.
   This is what makes (1) and (2) safe to want.

The audience is the Claude Code audience — people who live in a terminal — doing non-code life
admin: "reply to Petra about Thursday", "what did I decide about the lease", "rename these
photos by date", "remind me to call the dentist when I'm home". Latency is a first-class
constraint: a chat question is one model call, a lookup is two.

## What changed today, and what stays

**Changed.** The planner / plan review / rectify / replan / synthesize engine is gone. One ReAct
loop makes one native tool-calling call per pass; deterministic guards replace the judge; the
plan is a checklist tool the model may use; the rail shows every call with its result; Esc pauses
into continue / steer / abort. Measured on the 9b: chat = 1 call, calculator = 2 calls in 3.7 s,
a two-file comparison issues both reads in one pass.

**Stays.** The gate, the egress ledger, the air-gap, quarantine, the trace DB and replay, the
six-layer memory with gated learning, the tool surface (files, shell, web, knowledge base, Apple
Notes / Calendar / Mail, notifications, MCP), the tiers and the prefix cache.

**Parked.** Token steering and confidence coloring (modules kept, loop does not arm them). The
trust benchmark's engine metrics.

## Where Saturn is against the goal (honest)

| Goal leg | Today | Gap |
|---|---|---|
| Know the user | Six memory layers exist; every fact still needs a review click; the agent reads Notes/Calendar/Mail but not Contacts or Reminders; first run asks for a model tier, not for you | The agent starts every session knowing almost nothing and learns slowly |
| Yours to shape | `SATURDAY.md` per workspace; MCP config in YAML with zero servers enabled; `/policy` for the gate | No global instructions file, no user-authored procedures, no hooks, no way to add a tool without Python |
| Visible + gated | Done and true | Fifteen slash-command modules and a trace vocabulary built for auditors, not for someone asking about their Thursday |
| Works where you are | File tools are jailed to `database/workspace`; `read_file` is UTF-8 text only | "summarize the PDF on my desktop" fails before the model is consulted |
| Feels like a product | Install hardcodes the 4b tier; no hardware probe; the 27b experience is a demo, not the default | Most people's first turn is on the weakest model |

## Next improvements, ranked

Ranked by how much of the goal each unlocks per day of work. The first three are the cheapest
large gains in the repo and each is a bounded change; four through six are the "know the user"
sub-project; the rest are "make it yours" and "feel like a product".

### 1. Work where you launched — `saturn` in any folder (1–2 days)
Root the file tools at the launch directory (or `--workspace`), keeping the same sandbox check;
`/undo` snapshots and `SATURDAY.md` discovery follow the root; the fixed `database/workspace`
stays the default only for wheel installs started from `$HOME`. Add `~/Desktop`, `~/Downloads`,
`~/Documents` as always-listed roots so "the file on my desktop" resolves. This is the single most
important daily-task property Claude Code has and Saturn lacks.

### 2. Read the files people actually have (half a day)
Route `read_file` through the loaders already in `stores/rag.py` (pypdf, python-docx) for a
direct read of PDF / .docx / .xlsx-as-CSV; keep the knowledge base for search across many
documents. "Summarize this PDF" stops depending on the embedder.

### 3. Contacts and Reminders as native readers (1 day)
Two more AppleScript readers beside Notes / Calendar / Mail — `search_contacts` (resolves
"Petra" to an email address and a phone number) and `list_reminders` / `create_reminder`
(the natural home for "remind me to…", replacing the launchd notification for anything with a
due date). Readers `untrusted=True` as today.

### 4. Learn from what the user *says*, without the click (2–3 days)
Provenance-gated auto-memory. A fact the user states in their own words ("I'm vegetarian",
"Petra is my manager", "my lease ends in March") is written by the trusted principal and lands
directly in the `user` / `entities` / `commitments` layer with `by=user`; the rail shows a one-line
`· remembered: …` leaf and `/memory forget` undoes it. Only *inferred* facts (the model's
proposals, compaction summaries) keep the review queue. Keep the gate on `remember` when the
source of the fact is a tool result — a web page must never plant a memory. This is the change
that turns "it never remembers" into "it knows me", and it costs nothing at the gate for the
common case.

### 5. A first-run interview, not a model picker (1 day)
After the tier is chosen, ask five questions in the REPL — name, what you do, the people you
mention most, what you want help with, what it should never do — and write the answers to the
`user` / `entities` / `negative` layers. Skippable, re-runnable as `/memory setup`. The agent's
second turn should already know who it is talking to.

### 6. A launch brief (1 day)
On session start, one dim block: today's events, mail threads waiting on a reply (from
`list_mail`), open commitments, the memo digest — read-only calls, auto-approved, cached for the
session. Nothing runs in the background; the brief is the first turn, done for you. `/brief`
re-runs it; `runtime.brief: false` turns it off.

### 7. `SATURN.md` global + per-folder instructions (half a day)
`~/.saturn/SATURN.md` loaded every turn (tone, standing rules, "always metric", "never draft to
my boss without asking"), merged under a folder's `SATURDAY.md`. Rename the per-folder file to
`SATURN.md` too and keep reading the old name. `/init` drafts both.

### 8. Skills: user-authored procedures as markdown (2–3 days)
`~/.saturn/skills/<name>.md` with a one-line description and a body of steps the agent follows
when the request matches (`/weekly-review`, `/expense`, `/travel-checklist`). The runbook idea
from the v1 plan, in Claude Code's vocabulary and file format: no DSL, no parameters beyond the
request text, listed by `/skills`, injected into the prompt only when invoked or matched by
name. This is where "the plan" comes back — authored by the user, once.

### 9. Script tools without Python (1–2 days)
`~/.saturn/tools/*.sh|py` with a frontmatter (`name`, `description`, `risk`, `args`) registered
through `toolspec.register_tool_object` like MCP tools — the cheapest way for a user to teach
Saturn one thing their life needs (a `pay-rent` script, a `home-assistant` toggle). Risk fails
closed to `destructive`; every call faces the gate; stdout is untrusted.

### 10. Hooks (half a day)
`~/.saturn/hooks.yaml`: shell commands on `turn-start`, `turn-end`, `before-write`,
`after-write`. The same seam Claude Code exposes; the memory review and the launch brief could be
built on it.

### 11. The command diet (1 day)
Five commands a person needs — `/memory`, `/skills`, `/policy`, `/trace`, `/help` — listed by
default; `/confidence`, `/privacy`, `/notify`, `/mcp`, `/models`, `/config`, `/docs`, `/undo`
behind `/help --all`. Nothing is removed; the first screen stops looking like an audit console.

### 12. Hardware-probed install (1 day)
`install.sh` / first run reads RAM / VRAM and recommends the tier (27b on a 32 GB Mac, 9b on
16 GB, 4b below), pulls the model on consent, and says plainly what each tier feels like. Until
this exists, "most users run the good model" is hoped for, not true.

### 13. `send_mail` / `send_message`, gated (1 day, after 1–6)
The one new egress chokepoint the daily-task goal needs: a reply that actually sends. Always
gated, never auto-approved, shown whole at the gate, on the ledger. The findings are in
`docs/superpowers/specs/2026-09-06-macos-apps.md`.

## What to stop investing in

- **RAG as a first-class feature.** Keep `search_knowledge_base` working; direct reads (2) cover
  the daily case. Lazy-pull the embedder on first `/docs add`.
- **Confidence coloring and token steering.** Parked. Bring back only if a demo needs them.
- **The trust benchmark's capability suites.** Keep the trust probes (gate, egress, quarantine,
  memory planting) as the regression floor; drop the engine metrics they lost today.
- **Auditor-grade trace surfaces.** `/trace export`, replay and `/trace why` stay; no new views.

## How to decide, from here

Two knives. **"Would a person hand this to Saturn on a Tuesday?"** — if a feature does not
make one concrete daily request possible or faster, it waits. **"Does it cost the chat turn
anything?"** — a chat question is one call; every addition states its cost on that shape, and a
safeguard that cannot fire on it must cost it nothing (`nodes/agent.py` is the template).
