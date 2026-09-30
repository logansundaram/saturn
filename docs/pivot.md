# Saturn v2 — the pivot

_2026-09-27. Written the day the plan/execute engine was replaced by one loop
(`superpowers/specs/2026-09-27-v2-react-loop-design.md`, this folder). This file states the new goal
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

### 1. Work where you launched — `saturn` in any folder (1–2 days) — shipped 2026-09-29 (launch folder + /add-dir, /rm-dir; no always-listed roots)
Root the file tools at the launch directory (or `--workspace`), keeping the same sandbox check;
`/undo` snapshots and `SATURDAY.md` discovery follow the root; the fixed `database/workspace`
stays the default only for wheel installs started from `$HOME`. Add `~/Desktop`, `~/Downloads`,
`~/Documents` as always-listed roots so "the file on my desktop" resolves. This is the single most
important daily-task property Claude Code has and Saturn lacks.

**Known slow case (2026-09-29):** launched from `~`, a content search that matches nothing
reads every text file under home — measured 80 s (17,101 files, 796 MB); a name search takes
0.7 s and startup walks nothing. `search_files` now stops at 10 s and says the scan was partial
(2026-09-29).

### 2. Read the files people actually have (half a day) — shipped 2026-09-29 (`core/doctext.py`; .xlsx via the stdlib, no new dependency; `@file` attachments too)
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

### 7. `SATURN.md` global + per-folder instructions (half a day) — shipped 2026-09-28 (`/init` still writes SATURDAY.md)
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

### 10. Hooks (half a day) — shipped 2026-09-29 (`core/hooks.py`; before-write can block; the file tools never write hooks.yaml)
`~/.saturn/hooks.yaml`: shell commands on `turn-start`, `turn-end`, `before-write`,
`after-write`. The same seam Claude Code exposes; the memory review and the launch brief could be
built on it.

### 11. The command diet (1 day) — shipped 2026-09-28 (`/help --all`)
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

## Cut list and loop improvements (2026-09-28 survey)

_A pass over the tree the day after the v2 cut, ranked by lines removed per risk. Each cut is a
bounded deletion with a test file to drop alongside it; the improvements change the loop's
shape without adding a call to the chat turn. Pivot #1 and #2 above stay the top gains._

### Cut

- ~~**Outbound redaction**~~ — cut 2026-09-29 down to the scanner the gate's secret-argument
  warning uses (`trust/secret_scan.py`); the warn/redact modes and `runtime.redaction` are gone.
  (Was: `trust/redaction.py`, 183 lines, `runtime.redaction`,
  `tests/test_redaction.py`, a receipt branch). It only runs when `OLLAMA_HOST` is remote
  (`core/llms.py::_wrap_ollama`) and the default is `off`. The egress ledger already records
  the remote host.
- ~~**The Sources footer on the answer.**~~ Kept (2026-09-29, the user's call): the receipt of
  what informed an answer stays on the recorded message.
- **The menu bar LaunchAgent** — defaulted off 2026-09-29 (`notify.menubar: false`). (`notify/menubar.py` + `notify/menubar_app.py`, 443 lines, the
  pyobjc dependency). Every interactive launch installs a login item (`app/repl.py`,
  `_menubar.ensure_running()`) for an icon that lists pending notifications. Default it off, or
  cut it until the ambient-awareness work (advantages.md §5.5) gives it a job.
- **The second model role.** All four tiers bind `utility` to the same model as `tool_caller`;
  the `capabilities` block is read once for a startup warning and `max_context_window` is
  display only. Collapse each tier to `model`, `num_ctx`, `embedder`; `/models`, `config.py`
  and every "role" reference simplify with it.
- ~~**The structured-output layer** (`core/structured.py`).~~ Cut 2026-09-29: the options
  builder moved to `core/llms.invoke_kwargs`, the review makes one constrained call. One caller left, the memory review's
  proposals. Move `_invoke_kwargs` / `_model_tag` into `core/llms.py`; the review makes one
  constrained call with a default.
- ~~**`requirements.txt`.**~~ Cut 2026-09-29: install.sh installs editable from pyproject.toml. CI installs from `pyproject.toml`; a second list kept in sync is pure
  upkeep.
- ~~**`recall_runs`** (`tools/knowledge.py`).~~ Cut 2026-09-29; `/trace search` stays. A model-facing search over the trace DB, marked
  untrusted, overlapping memory. "What did I decide" is memory's job.

### Fix (found dogfooding)

- **`saturn -p` hangs when stdin is open but is not a terminal** (2026-09-29, found running
  it from a background job). Headless mode reads piped stdin to append it to the query, and
  waits forever when nothing is written. Read stdin only when something is waiting there
  (`select` on POSIX), and say so in `--help`; anyone scripting Saturn hits this.

### Trim (rot that misleads the next reader)

- ~~`README.md` (the "life of a turn" block and the layout listing) still documents plan,
  rectify and synthesize.~~ Fixed 2026-09-29, with the stale headers below and `/trace answer`.
- Stale headers: `tui/ui/__init__.py` describes plan_gate / update_plan / synthesize rows;
  `nodes/tools.py` opens with "living-plan ReAct loop (Phase 1)"; `nodes/approval.py` says
  "Phase 2"; `config.default.yaml` says "Phase 3" and "Saturday.ai"; `core/context.py`
  references plan_context.
- `/trace` usage still lists `answer`, cut with the Glass Box.
- Two names: done 2026-09-29 (`SATURN_*` with `SATURDAY_*` fallbacks; `/init` writes
  `SATURN.md`); 2026-09-29 also: one home — a new wheel install keeps its data in ~/.saturn
  (`config.wheel_data_home`; an existing ~/.saturday install stays put), the installer's clone
  folder is `SATURN_INSTALL_DIR`. Pick `SATURN_*`, read the old spellings as fallbacks for one release.

### Improve

1. **Put the date in the dynamic grounding** (`nodes/ground.py::grounding_node`) — shipped
   2026-09-29 as the `### Now` section; `current_time` stays for the exact time mid-task. The prompt
   routes every "Thursday" through a `current_time` round trip; one line makes date questions a
   single call, and the tool and its prompt line can go.
2. **Move the ask_user interrupt out of the tool.** A resumed interrupt re-runs the tools node,
   so the agent node forces ask_user to run alone (`ASK_ALONE_TEXT`). If the approval node
   raises the question interrupt and writes the ToolMessage itself, the hack goes and siblings
   run.
3. **Size the observation clamp to the window.** `nodes/tools.py::_MAX_OBSERVATION` is a fixed
   12k characters while windows run 32k–128k tokens; derive it from
   `core/llms.active_context_window`.
4. **Think on evidence, not on pass count** — shipped 2026-09-29. A pass thinks only right
   after a tool round with an error; the plan, declined-call and `think_after` triggers are
   gone, and a thinking pass that returns nothing is rerun think-off. Found by the loop
   benchmark: on the 4b and 9b, a thinking pass whose right move is a short answer writes the
   answer inside its reasoning and emits no content.
5. **One walk-back helper** — shipped 2026-09-29 (`core.state.issuing_message`). "Skip trailing ToolMessages to find the issuing AIMessage" exists
   in `nodes/agent.py::route_after_agent`, `nodes/tools.py::tool_node` and `nodes/approval.py`.
   Put it in `core/state`.
6. **Concurrent tool batches.** Serial today so egress events attribute by sequence
   (`_egress_slice`). Tag ledger events with a call id via a contextvar and the batch can run
   in a pool; a two-file compare then reads both at once.
7. **Fold the `/models` page into first run.** `commands/runtime.py` (697 lines, a 696-line
   test) already probes hardware and recommends a tier — pivot #12. Run the recommendation once
   at first launch and shrink `/models` to list and use.


## Loop improvements (2026-09-28 brainstorm)

_Ideas for the engine itself — the shape of the loop, not features around it — ranked for all
model sizes: the guards matter most on the 4b/9b, the projection and the grounding matter on
every tier. The two knives below apply to each. The "Improve" list above still stands; where
an item here subsumes one of those it says so._

1. **A loop benchmark first** (`python benchmark.py --loop`). Twenty-odd daily requests with the
   tool sequence each should take, graded from the turn record: passes per shape (chat = 1,
   lookup = 2, multi ≤ N), wrong tool, missing tool, phantom action (text that describes an
   action with no call), stub answer, hygiene bounces, capped turns, and a verifiable value in
   the answer where one exists. Every idea below is a guess until this exists; the trust
   benchmark lost its engine metrics in the v2 cut. Shipped 2026-09-28.
   **Baselines (2026-09-29 evening, after the small wins):** 4b 19/24, 9b 20/24, no phantoms,
   hygiene bounces or capped turns on either. Stable misses: `file_long_middle` (the clamp drops
   the middle — item 6 below) and `robust_no_math_in_head` on the 9b (seven `calculate` calls
   to test primality — eight passes). Trust benchmark: 4b gate 3/3 · injection 2/2 · memory
   supersession MISSED (it called `recall` with `remember`'s arguments seven times, hygiene
   bounced each, then claimed the update — the incidents note disclosed it); 9b all pass. A
   4b-only miss is a model limit; confirm on the 9b before changing the engine. Later that
   evening, with the rounds rule in the prompt and the foreign-arguments refusal: 4b loop
   20/25 (the new `multi_dependent` passes: read in pass 1, write in pass 2), 4b trust all
   pass including supersession.
2. **`_llm_input` becomes a budgeted prompt projection.** Today it maps state to the prompt and
   only strips trailers, so ten reads on a 32k window push the system prompt off the front. Give
   the projection a token budget: an observation a later pass has already moved past collapses
   to a one-line stub in the PROMPT only — state, the trace and replay stay whole. Subsumes
   "size the observation clamp to the window". Costs the chat turn nothing.
3. **A question is an answer: delete the `ask_user` interrupt.** The model's last message is
   the answer; when it needs a value it answers with the question and the turn ends, and the
   user's reply is the next turn with the history intact. Deletes the interrupt, the run-alone
   hack (`ASK_ALONE_TEXT`), the headless special case and the tool; `plan` state carries across
   the boundary so a mid-checklist question resumes. Replaces "move the ask_user interrupt out
   of the tool".
4. **An environment snapshot in the dynamic grounding** — the date half shipped 2026-09-29
   (`### Now`); the launch folder was already in the stable half; the workspace listing is open. Date, weekday, time, the launch
   directory and a short workspace listing, the way Claude Code puts cwd and git status in
   front of the model. Subsumes "put the date in the dynamic grounding"; `current_time` and
   its prompt line go. "The file on my desktop" and "Thursday" resolve on pass one.
5. **A phantom-action guard.** The characteristic small-model failure: "I'll read the file
   now." with no tool call, and the turn ends. Deterministic check on an answer — no calls,
   short, ends in an intent verb, no tool used this turn — followed by ONE nudge pass. Fires only
   on the failure shape; the benchmark's phantom count says whether it earns its place.
6. **Compress oversize observations instead of clipping them.** Head-and-tail loses the middle
   of a web page or a long file. Past the clamp, one utility-role call extracts what is relevant
   to the request. Costs a call only on an oversize result. Pulls against collapsing the utility
   role (cut list above) — decide the two together.
7. **A wall-clock budget beside the pass cap.** `runtime.turn_seconds` triggers the same
   capped last pass. Sixteen passes on a 4b can be minutes; a companion should not make someone
   wait that long without a decision.
8. **Record the reasoning** — shipped 2026-09-29, in the `llm_calls` record (not the
   AIMessage: langchain-ollama would send `reasoning_content` back as `thinking`). A thinking pass streams `reasoning_content` and drops it. Stamp it
   on the recorded AIMessage so `/trace why` can show why the pass chose its calls, and the
   loop benchmark can grade it.
9. **The catalog's shape — measure before touching.** Twenty-six schemas is 3.7k tokens and
   twenty-six choices for a 4b. Two candidates to benchmark: domain tools with an action enum
   (`mail(action=…)`), or a small core set plus a deferred group. Both fight the prefix cache,
   so only if the loop benchmark shows tool-choice errors.

Order: 1, then 2 and 3 as one sub-project, then 4 and 5. Items 2 and 3 are the ones that
change the loop's shape; the rest are guards and grounding.

## How to decide, from here

Two knives. **"Would a person hand this to Saturn on a Tuesday?"** — if a feature does not
make one concrete daily request possible or faster, it waits. **"Does it cost the chat turn
anything?"** — a chat question is one call; every addition states its cost on that shape, and a
safeguard that cannot fire on it must cost it nothing (`nodes/agent.py` is the template).
