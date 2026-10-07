# Saturn

> A **private, local-first AI agent** that runs on your own machine — every step it takes is
> visible, auditable, and yours to approve. Built by **Saturday.ai** — [saturdayai.org](https://saturdayai.org).

Saturn is a personal agent you run in your terminal: a local companion you can hand your day to.
"Reply to Petra about Thursday", "what did I decide about the lease", "rename these photos by
date", "remind me to call the dentist tomorrow at 9", "text Sam I'm 15 minutes late". It reads
and writes your files, your notes, calendar, mail, contacts and reminders, searches the web,
runs commands, and remembers things across sessions — showing you **every step it takes** and
pausing for your approval before anything changes or leaves the machine. Nothing is a black
box: you watch each tool call and its result as it happens, and any run can be replayed
afterward.

It runs entirely on **local models** (via [Ollama](https://ollama.com)) — your data stays on
your machine, and no API key is required for anything. That is what makes it safe to tell it
about your life.

---

## Prove it in 60 seconds

Don't take the claims on faith — run one loop and check each one yourself:

```
» what changed in local LLMs this week?
```

1. **Watch it work.** Each `web_search`/`web_extract` call shows in the rail as it runs, with
   its result; the answer streams as it is written.
2. **Read the receipt.** The stats line under the answer carries the trust segment — here it
   shows `⇅ N sends · <bytes> → <host>` in yellow, because something *did* leave your machine,
   and the receipt says so instead of hiding it.
3. **`/policy egress`** — the per-event ledger: exactly what left, channel / host / bytes.
4. **Make it ask.** `» save a two-line summary to notes.md` — the approval gate shows the exact
   file diff and waits; bare Enter rejects (the default is always *no*).
5. **Hold the record.** `/trace export` writes the run's complete record as JSON — every
   model pass, every tool call and observation, every human gate decision. Replay it offline any time:

   ```bash
   saturn --replay logging/exports/run_1.json    # renders the full drill-down, no DB needed
   ```

   The record is a shareable, replayable artifact of exactly what the agent did — a real
   execution log, not a screenshot.

---

## Why Saturn?

**AI agents should show their work.** Most assistants are a black box in front of a remote
model: a chat transcript shows you a summary of what the agent *claims* it did, not what it
actually did, in what order, with what inputs. Saturn is built around one idea — **you can see
it** — backed by two guarantees:

- **Nothing leaves your machine.** Local models by default, zero required API keys, zero
  telemetry, MIT-licensed source. The privacy claim is not a policy promise — it is inspectable
  in the code and observable on the network.
- **Nothing happens without you.** Every side effect stops at an approval gate that shows the
  real artifact of the decision — the full shell command, a colored diff of the proposed file
  write, the number and the exact text of a message. Press Esc at any moment to pause the turn:
  continue, steer it with a correction, or abort. Every run can be replayed after the fact
  (`/trace`), and file changes reversed (`/undo`).

Underneath, everything is configurable (one `config.yaml` for models, safety policy, context,
paths — most of it live-tweakable with slash commands) and yours to shape: standing
instructions in `SATURN.md`, your own Shortcuts and MCP servers as tools, your own shell
commands as hooks. Tools and commands are small registry-based modules; adding a capability is
adding a file.

---

## What it can do

Breadth, but behind one boundary: every capability below is a tool call you can watch in the
rail, faces the same approval gate, runs locally where it can, and lands in a trace you can
replay. Reading runs without asking; anything that changes or sends something asks first.

**Your Mac** (macOS; each goes through the app itself, so what Saturn does shows up where you
would look for it)

- **Notes** — search, read, create, and append to an existing note.
- **Calendar** — list events; create, move, rename or delete one (a repeating event only as a
  whole series, and you are told when attendees may be notified).
- **Mail** — list, search and read; `draft_mail` and `reply_mail` open an **unsent** draft in
  the right thread (you press Send); mark read/unread, flag, move or trash a whole list of
  messages in one approval.
- **Contacts** — a name becomes the addresses and numbers on the card, so a reply or a text
  goes to a real address instead of a guessed one.
- **Reminders** — "remind me to call the dentist tomorrow at 9" creates a reminder that reaches
  your phone; list what is open or overdue, tick one off.
- **Messages** — read your message history (needs Full Disk Access for the terminal), and
  **text someone**: a send is the one action that *always* asks. You see the number, whose it
  is, and the exact text every time; no setting, no "always allow" and no `--yolo` skips it.
- **Shortcuts** — run any shortcut you have built ("lights off", a Focus mode, a HomeKit
  scene). It asks first unless you allow that one shortcut by name (`/policy shortcut`).
- **"This page", "these files"** — the page in your front browser tab, and the files selected
  in Finder. Type `@clipboard` to attach what is on the clipboard; `/copy` puts the last answer
  on it.
- **Notifications** — a one-off desktop alert at a time you name (`/notify` lists and cancels).

**Your files and the web**

- **Your files, where you are** — Saturn works in the folder you launched it from; `/add-dir`
  reaches another one. Read (PDF, Word and Excel files too), write, edit (anchored string
  replace), rename and move, search (content regex + name glob, with Spotlight's index on
  macOS), and delete — a delete moves to the Trash. Every write is snapshotted first, so `/undo`
  reverts any turn's file changes. Type `@file` to attach one to a message.
- **Web search** — **no API key, no account, ever**: keyless DuckDuckGo search + local page
  extraction (`trafilatura`). Your queries never route through a keyed SaaS backend.
- **Your documents (RAG)** — ingest PDFs, text, markdown, HTML, CSV, and Word (.docx) files into
  a local knowledge base it can search across (`/docs add`, or ask: "add notes.md to my
  knowledge base" — adding and removing a document always asks first).
- **Shell commands** — run shell commands (scripts, build tools, git, package managers) in the
  working folder, through `/bin/sh`. Every run is a bounded **foreground** run: the process
  lives and dies inside the turn you approved (no detached-process surface), with a timeout so
  a hung command can't wedge the turn. A command never reads your terminal (a prompt for input
  gets end-of-input at once), and a non-zero exit is a failed call the answer tells you about.
  `!command` at the prompt runs one yourself, without the agent.
- **Math & time** — a precise calculator and the machine's own clock, so arithmetic and
  "today" are computed, never guessed from memory.

**Knowing you, and yours to shape**

- **Memory that grows** — a layered store that persists across sessions: what you told it
  (`remember` / `recall`), open commitments, dated notes, what it learned about this machine, the
  people and projects in your life, and what not to do again. Loaded selectively each turn under
  one cap, every fact carries the run it came from. What you tell it in your own words is kept
  without a prompt (a `remembered #n` line after the answer; `/memory remove n` undoes it); a fact
  whose words came from a web page, a mail or a file still asks, and what it infers waits for a
  review screen (`/memory review`). A password, a card number or a key in a recognisable form is
  refused.
- **Standing instructions** — `~/.saturn/SATURN.md` loads every turn (your tone, your rules);
  a `SATURN.md` in the working folder adds that folder's own. `/init` surveys the folder and
  drafts one.
- **Hooks** — `~/.saturn/hooks.yaml` runs your own shell commands on `turn-start`, `turn-end`,
  `before-write` and `after-write`; a `before-write` hook that exits non-zero blocks the write.
- **MCP servers** — plug in any [Model Context Protocol](https://modelcontextprotocol.io) server
  (stdio or remote HTTP/SSE) by declaring it in `config.yaml`; its tools join the agent behind
  the **same approval gate** as everything else. Remote tools always prompt until *you* lower
  their risk tier — a server's own "read-only" claim is never trusted. `/mcp` shows status.

**The trust stack**

- **The approval gate** — every call that changes or sends something stops for you and shows
  the real artifact: the diff, the command, the recipient. Bare Enter rejects. `/policy` is the
  one place its levers live (risk tiers, always-allow prefixes, shortcuts).
- **It asks instead of guessing** — when a needed value, choice, or confirmation is missing,
  the agent pauses mid-run with one question (`ask_user`), and your typed answer resumes the
  turn. A phone number or address that appears in nothing you typed and nothing a tool returned
  is refused before the call runs.
- **Pause and steer** — Esc pauses a running turn: continue, type a correction to steer it, or
  abort. On a multi-step task the agent keeps a checklist you can watch in the rail.
- **Prompt-injection quarantine** — web pages, files, mail, shared notes and remote tool
  results are untrusted input. Content that tries to steer the agent ("ignore your previous
  instructions", "run this command") is detected, visibly flagged in the trace, fenced off as
  data the model must not obey — and the next action that can change or send something faces
  your approval gate regardless of risk tier, so a malicious page can't quietly redirect the
  agent.
- **The egress ledger and the air-gap** — every network operation is recorded: what left,
  to which host, how many bytes (`/policy egress`). `/policy airgap on` blocks all of it, and a
  shell command or shortcut — a process Saturn cannot see inside — is marked untracked and
  always asks while the seal is on.
- **Trust receipt when it matters** — the stats line under a response says exactly how many
  bytes went to which host and how many actions faced the approval gate — whenever anything
  actually left your machine or was gated. A fully-local turn stays clean: silence means
  nothing left.
- **Sourced answers** — an answer that drew on tools or documents ends with a Sources list of
  the exact calls and documents behind it (completed reads and searches; a failed call is in
  the incidents note instead); `/trace source 3` shows the full material behind any line.

**Outside the chat loop**

- **Headless mode** — `saturn -p "query"` runs one query and prints the answer; piped stdin
  attaches to the turn (`git diff | saturn -p "review this change"`). Gated tools are denied by
  default (no human at the gate) unless you pass `--yolo` — and a send is refused even then.
  Add `--json` for a machine-readable result (answer, plan, tools, tokens, timing, plus a
  `gates` record of which calls were prompted and denied); `--export <file>` writes the run's
  replayable export record after the turn; `saturn --replay <file>` renders an exported record
  offline. The CLI is strict: unknown flags exit 2 instead of silently launching the chat loop.
- **One-shot query mode** — `saturn -q "question"` is the pipe-friendly rendering of the same
  headless turn: stdout carries *only* the final answer, progress lines go to stderr, and the
  run auto-exports so the closing `recorded: saturn --replay <file>` receipt replays exactly
  what happened — gates, tool calls — offline. Same engine, same deny-by-default gate.

---

## How it works (the short version)

Every turn is one loop of small, inspectable steps:

```
ground → agent ─(no tool calls)─→ answer
           ↑          │ tool calls
           └── tools ← approval
```

- **ground** loads your standing instructions (`~/.saturn/SATURN.md` and the folder's
  `SATURN.md`), today's date, the memory facts relevant to this request, and the
  knowledge-base manifest.
- **agent** makes one model call: it either calls tools or answers. A message without tool
  calls *is* the answer, and it streams as it is written.
- **approval** pauses for your OK before anything side-effecting runs.
- **tools** run; results flow back so the agent can decide what's next.

On a multi-step task the agent keeps a checklist with the `plan` tool, shown live in the rail.
Esc pauses the loop: continue, steer, or abort.

---

## Getting started

### Quick install (recommended)

One command. It installs [Ollama](https://ollama.com) if needed, clones Saturn into `~/.saturday`
in an isolated virtualenv, pulls the small local chat model, and puts a `saturn` command on your PATH.

```bash
# macOS / Linux
curl -fsSL saturdayai.org/install.sh | sh
```

Then open a new terminal and run `saturn`. The first run pulls a few GB of models, so it takes a
minute. Prefer to read before you pipe? The script is this repo's [`install.sh`](install.sh),
served from the site — read it there or here first.

The installer defaults to the lightweight **`4b`** size class (`qwen3.5:4b`). On the first launch
`/models` reads your hardware, prices every size against it, and asks which tier and embedder to
run — Enter takes the recommendation, and anything not pulled yet is pulled on consent. Re-run
`/models` anytime, or set `SATURN_TIER=9b` (or `27b`/`35b`) before installing.
Other knobs: `SATURN_INSTALL_DIR` (install dir), `SATURN_MODELS` (models to pull), `SATURN_BRANCH`.

Right after the model pick, Saturn offers three quick questions — what to call you, what you
do, and anything it should never do — and saves your answers to its memory (a markdown file
you can read and edit; `/memory` shows it). Enter starts them, `n` skips, and any question can
be skipped; `/memory setup` asks them again later, with two more about your people and what
you want help with.

Saturn ships one recommended tier per parameter size (the qwen3.5–3.8 ladder); `/models`
shows the ladder priced against your machine. Any Ollama model with native tool-calling binds
with `/models use <id>`.

> Prefer to set it up by hand, or hacking on Saturn itself? Use the **Manual install** below.

### Install with pipx / uv

Already manage Python tools with [pipx](https://pipx.pypa.io) or [uv](https://docs.astral.sh/uv/)?
Saturn ships on PyPI as [`saturn-agent`](https://pypi.org/project/saturn-agent/):

```bash
pipx install saturn-agent
# or
uv tool install saturn-agent
```

(Want the unreleased tip of `main` instead? `pipx install git+https://github.com/logansundaram/saturn`.)

Then run `saturn`. You still need [Ollama](https://ollama.com/download) running and the tier's
chat model pulled — for the `4b` tier that's `ollama pull qwen3.5:4b` (a multi-GB download;
Ollama prints its exact size as the pull starts). The quick installer above does that for you,
and the first launch's `/models` page offers to run any missing pull (y/N, default no); later
launches warn about anything missing. The knowledge-base embedder (`qwen3-embedding:8b`) is
pulled on consent by the first `/docs add`. Installed this way, your data and `config.yaml`
live in `~/.saturn` beside your `SATURN.md` (override with `SATURN_HOME`), and you upgrade with
`pipx upgrade saturn-agent` / `uv tool upgrade saturn-agent` instead of `/update`.

### Manual install (from source)

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running locally.
- The local model pulled. The default (`4b`) tier uses one small qwen3.5 model for everything —
  a multi-GB download (Ollama prints its exact size as the pull starts):

  ```bash
  ollama pull qwen3.5:4b           # the chat model (~4B)
  # the knowledge-base embedder (qwen3-embedding:8b) is pulled on consent by the first /docs add
  ```

  > More hardware to spare? The first launch runs `/models`, which reads your Apple chip, its GPU
  > cores, memory and memory bandwidth, and offers the largest size class that fits *and* runs at a
  > usable speed (pulling it first); re-run `/models` any time,
  > or `/models list` to just see the fit table. Or edit `active_tier` in `config.yaml`
  > yourself — `9b`, `27b`, or `35b` — and pull that class's tag instead (same embedder). Any
  > other Ollama model with native tool-calling works too: `/models use <id>`.
  > (Small models are less reliable at choosing tools — if your hardware fits the `9b` or
  > larger, use it.)

### 2. Clone and install

```bash
git clone https://github.com/logansundaram/saturn
cd saturn

# (recommended) a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

> `prompt_toolkit` (live command highlighting) is optional — Saturn runs fine without it.

There is **no API key step**: web search is keyless and inference is local. (Custom env vars —
e.g. for an MCP server's `${VAR}` expansion — go in a plain `.env` file next to the install,
or in `~/.saturn/.env` for pipx installs.)

### 3. Run it

```bash
python agent.py        # inside the venv
python3 agent.py       # without a venv (if `python` isn't in PATH)
```

You'll get an interactive prompt (`»`). Just type. Anything starting with `/` is a command;
everything else is a turn for the agent.

```
» what's on my calendar tomorrow, and is anything in mail waiting on me?
» summarize report.pdf and list the decisions in it
» remind me to call the dentist tomorrow at 9
» remember that I prefer concise answers
```

> **Shortcut launchers**
>
> The launcher prefers the repo's own `.venv` interpreter when one exists, and does not `cd`
> into the repo — relative paths in your arguments resolve against *your* directory, and nothing
> leaks a directory change into your shell.
>
> Make `saturn.sh` executable once, then run it directly or add the repo to your
> `PATH`:
> ```bash
> chmod +x saturn.sh
> ./saturn.sh
> ```

---

## Configuration

Everything lives in **`config.yaml`**:

- **`active_tier`** — which size-class preset is live (`4b`, `9b`, `27b`, `35b`).
- **`tiers`** — binds each size class to one concrete `model` (and an `embedder`), so
  swapping hardware is a one-line change. Every tag is qwen3.5/3.6/3.8 — `/models` lists the
  ladder with weights, context window, and what each needs on your machine. The shipped windows
  step up the ladder (32k for 4b, 64k for 9b/27b, 128k for 35b), sized so each tier fits
  its hardware; `/models` prices any window you set against your memory.
- **`runtime`** — loop and safety knobs: `max_iterations`, `auto_approve` (the approval policy),
  `num_ctx` (context window), `citations` (inline source citations in answers).
- **`web`** — web-tool knobs (`max_results`, `request_timeout`). The backend is fixed and
  keyless: DuckDuckGo search + local extraction.

Most of it is also adjustable **live** (session-only) with slash commands — handy for
experimenting without restarting.

### macOS / Linux notes

Saturn runs on macOS (the native Notes / Calendar / Mail / Contacts / Reminders / Messages /
Shortcuts tools, the browser-tab and Finder readers, and notifications need it) and Linux (file,
shell, web and knowledge-base tools). The `run_shell` tool hands commands to `/bin/sh`, so write
Unix shell syntax (`ls`, `&&`, `|`, etc.). Windows support was dropped 2026-09-27.

**macOS permissions.** The first time a tool reaches an app, macOS asks whether your terminal
may control it (one dialog per app: Notes, Calendar, Mail, Contacts, Reminders, Messages, your
browser, Finder). The dialog names the terminal app you launched Saturn from, and a grant made
in one terminal does not carry to another. Reading your Messages history needs Full Disk Access
for that terminal (System Settings → Privacy & Security); Saturn tells you where when it is
missing.

---

## Useful commands

Type `/help` for every command, or `/<command> --help` for details on any one. Highlights:

| Command | What it does |
|---|---|
| `/help` | Every command by theme, opening with the trust-stack map (posture · activity · record); `/help <cmd>` details one. |
| `/models` | The model page: your hardware, the qwen ladder (four chat sizes + three embedders) priced against it — fit and estimated speed — pick a row to switch, pulling what's missing on consent. |
| `/config` | View/edit settings (`/config runtime.num_ctx <size|auto>` resizes the context window). |
| `/docs` | The knowledge base: list documents, `add <path>`, `remove <name>`, `rebuild` (launch syncs on its own). |
| `/tools` | The toolkits — the tools in groups (files, web, mail, calendar, messages…): `/tools off messages` turns one off so the model never sees it, `/tools on messages` brings it back, `/tools mail` lists one toolkit's tools and risk tiers, `/tools --all` every tool. |
| `/mcp` | MCP server status + the remote tools they add; `reload` after a config edit. |
| `/memory` | See, add, edit, and review the layered facts the agent permanently remembers; `review` is the gated learning step (also runs at `/quit`), `why <n>` its provenance. |
| `/policy` | Your trust settings in one place: bare = what runs without asking and what can leave the machine; `risk`/`allow`/`shortcut`/`open` are the gate's levers, `egress` the ledger of what left, `airgap` the seal (bare forms report, changing is always explicit). |
| `/trace source` | Show the full material behind a citation `[n]` of the last answer. |
| `/undo` | Revert the file changes of the last turn that wrote anything. |
| `/init` | Survey the working folder and draft `SATURN.md` standing instructions. |
| `/add-dir` · `/rm-dir` | Let Saturn reach another folder for this session / stop reaching it. |
| `/copy` | Copy the last answer to the clipboard. |
| `/notify` | Scheduled desktop notifications: what is pending, cancel one, send a test alert. |
| `/trace` | Inspect past runs, tool I/O, and LLM calls; `/trace why` explains a run's decisions; `/trace export` writes the run's complete record as JSON; `/trace replay` (or `saturn --replay <file>`) re-renders an exported record anywhere — no database needed. |
| `/resume` | Continue your last session (autosaved); `save [name]`/`list`/`<name>` for named sessions (plain `.json` files under `database/sessions/`). |
| `/update` | Self-update: pull the latest Saturn (your data is never touched). |
| `/clear` · `/quit` | Start a fresh conversation / exit. |
| `!command` | Run a shell command yourself, without the agent. |
| `@file` · `@clipboard` | In a message: attach a file, or what is on the clipboard. |

Every command takes `--help` as its first or last argument (mid-position it's ordinary data, so
`/memory add …` can store a fact that mentions it), and removal verbs are interchangeable
everywhere (`remove`/`rm`/`delete`/`del`/`forget`/`drop`).

---

## Web search without an API key

Saturn's web tools are **API-less by design** — a product whose pitch is "your data stays
yours" shouldn't steer your search queries through a keyed SaaS backend:

- **`web_search`** is keyless DuckDuckGo — no key, no account, nothing to sign up for.
- **`web_extract`** reads pages **locally** (via `trafilatura`) — only the page's own host is
  contacted, plus any host its redirects lead to. Saturn follows each redirect itself, so
  every host is checked against the air-gap and written to the egress ledger before it is
  reached.

For deeper research, the agent issues several search and read calls — each one visible in the
rail, every call traced — rather than hiding them inside a monolithic research tool. A page
address the model composed itself after reading outside content is held for your approval, so a
page cannot make Saturn carry your data out in a URL.

---

## Project layout

```
agent.py            # entry point: routes the command line into app/
app/                # the application shell: CLI, graph assembly, turn driver, REPL, headless
config.default.yaml # the tracked template; your config.yaml is seeded from it on first run
core/               # the engine room: state, model factory, prompts, workspace, hooks, mentions
nodes/              # the graph's four nodes: ground, agent, approval, tools
trust/              # the trust stack: gate policy, egress ledger, quarantine, trust receipt
tools/              # the agent's tools (files, shell, web, knowledge, memory, the macOS apps)
                    #   + the registry and MCP client
notify/             # scheduled notifications and the optional menu bar icon (macOS)
commands/           # slash commands (one module per /help theme)
stores/             # persistence: RAG + its manifest, durable memory, snapshots, trace
tui/                # the terminal UI / live trace rail
docs/               # the documents: README.md indexes them (ARCHITECTURE.md is the code map)
database/           # your data: documents/, memory/, sessions/, snapshots/, caches, trace DB
```

See **`docs/ARCHITECTURE.md`** for the guided code map, **`docs/engine.md`** for the loop's
shape, and **`docs/pivot.md`** for the product goal and the ranked work toward it.

---

## Benchmarking

Two benchmarks, both against a running Ollama with the active tier pulled.

The **trust benchmark** measures the trust stack itself: approval-gate coverage (every
non-read-only tool call must have faced the gate), the injection-quarantine flag rate (a planted
instruction-shaped document must be fenced), and the memory tasks (recall across runs,
supersession, and a planted memory that must face the gate).

The **loop benchmark** runs everyday requests and grades each turn from its record: how many
model passes it took, whether it chose the right tools, and whether it described an action it
never performed.

```bash
python benchmark.py                                   # the trust benchmark
python benchmark.py --strict                          # …and exit 1 on any graded FAIL (CI-friendly)
python benchmark.py --loop                            # the loop benchmark
```

Reports are written to `logging/benchmarks/trust_<ts>.json` and `loop_<ts>.json`. A benchmark
run never acts on your real world: its approver declines every gated call into the macOS apps.

---

## Status

Saturn (by Saturday.ai) is an actively developed, terminal-native agent — trust-first, local,
and built for the life admin of people who live in a terminal. The terminal is the product:
there is no GUI chat window on the roadmap, by design.

**v2** (2026-09-27) replaced the plan / execute engine with one loop: a chat question is one
model call, a lookup is two. Since then Saturn works in the folder you launch it from, reads the
documents people actually have, and reaches Notes, Calendar, Mail, Contacts, Reminders, Messages
and Shortcuts.

What is not built yet, said plainly: mail is drafted, never sent; every fact Saturn learns
still needs your accept; there is no morning brief and no
user-authored skills; and the macOS permission dialogs name your terminal, not Saturn.
`docs/pivot.md` ranks that work. Contributions and feedback welcome — file issues at the GitHub
repo. The site is [saturdayai.org](https://saturdayai.org): the install line, real run records
rendered from exports, and the blog.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
