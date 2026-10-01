# Saturn

> A **private, local-first AI agent** that runs on your own machine — every step it takes is
> visible, auditable, and yours to approve. Built by **Saturday.ai**.

Saturn is a personal agent you run in your terminal — and the whole point is that you can see
what it's doing. It plans its work in the open, then calls tools to search the web, read and
write your files, query your own documents, run commands, and remember things across sessions —
showing you **every step it takes** and pausing for your approval before anything touches the
outside world. Nothing is a black box: you watch the plan it draws up and the tools it calls,
and any run can be replayed afterward.

It runs entirely on **local models** (via [Ollama](https://ollama.com)) — your data stays on
your machine, and no API key is required for anything.

---

## Prove it in 60 seconds

Don't take the claims on faith — run one loop and check each one yourself:

```
» what changed in local LLMs this week?
```

1. **Watch it work.** The plan renders live; each `web_search`/`web_extract` call shows in the
   rail as it runs.
2. **Read the receipt.** The stats line under the answer carries the trust segment — here it
   shows `⇅ N sends · <bytes> → <host>` in yellow, because something *did* leave your machine,
   and the receipt says so instead of hiding it.
3. **`/policy egress`** — the per-event ledger: exactly what left, channel / host / bytes.
4. **Make it ask.** `» save a two-line summary to notes.md` — the approval gate shows the exact
   file diff and waits; bare Enter rejects (the default is always *no*).
6. **Hold the record.** `/trace export` writes the run's complete record as JSON — the plan,
   every tool call and observation, every human gate decision. Replay it offline any time:

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
- **Nothing happens without you.** The plan is a live, editable object you can pause, steer,
  and rewrite mid-run. Every side effect stops at an approval gate that shows the real artifact
  of the decision — the full shell command, a colored diff of the proposed file write. Every
  run can be replayed after the fact (`/trace`), and file changes reversed (`/undo`).

Underneath, everything is configurable (one `config.yaml` for models, safety policy, context,
paths — most of it live-tweakable with slash commands) and extensible (tools, commands, and
nodes are small registry-based modules; adding a capability is adding a file).

---

## What it can do

Breadth, but behind one boundary: every capability below surfaces as a step in the plan you can
watch, faces the same approval gate, runs locally where it can, and lands in a trace you can
replay. The point isn't how much Saturn can do — it's that you can see and control all of it.

- **Multi-step reasoning** — a living-plan ReAct loop: it drafts a plan, executes one step at a
  time, sees each tool result, and decides the next action. Multi-source research works this
  way too: search + read steps composed in a plan you can watch and edit, not an opaque
  "research" call.
- **Web search** — **no API key, no account, ever**: keyless DuckDuckGo search + local page
  extraction (`trafilatura`). Your queries never route through a keyed SaaS backend.
- **Your APIs** — connect any service as an MCP server (`config.yaml`): its tools face the same
  approval gate as everything else, never self-declare their risk tier, and their outgoing
  arguments are scanned for secrets. One integration mechanism, fully gated.
- **Your files** — read, write, edit (anchored string replace), search (content regex + name
  glob), and list files in a sandboxed workspace — with pre-write snapshots, so `/undo` can
  revert any turn's file changes.
- **Your documents (RAG)** — ingest PDFs, text, markdown, HTML, CSV, and Word (.docx) files into
  a local knowledge base it can search.
- **Math & time** — a precise calculator and the machine's own clock, so arithmetic and
  "today" are computed, never guessed from memory.
- **Memory that grows** — a layered store that persists across sessions: what you told it
  (`remember` / `recall`), open commitments, dated notes, what it learned about this machine, the
  people and projects in your life, and what not to do again. Loaded selectively each turn under
  one cap, every fact carries the run it came from, and it learns at session end through a review
  screen — nothing is written without your accept (`/memory review`).
- **It asks instead of guessing** — when a needed value, choice, or confirmation is missing,
  the agent pauses mid-run with one question (`ask_user`), and your typed answer resumes the
  turn. The alternative to asking is fabrication; Saturn asks.
- **Shell commands** — run arbitrary shell commands (scripts, build tools, git, package managers)
  in the sandboxed workspace, through `/bin/sh` — write Unix shell syntax. Every run is a
  bounded **foreground** run: the process lives and dies inside the turn you approved (no
  detached-process surface), with a timeout so a hung command can't wedge the turn. A command
  never reads your terminal (a prompt for input gets end-of-input at once), and a non-zero exit
  is a failed step the answer tells you about.
- **Sourced answers** — an answer that drew on tools or documents ends with a Sources list of
  the exact calls and documents behind it (completed reads and searches; a failed call is in
  the incidents note instead); `/trace source 3` shows the full material behind any line.
- **MCP servers** — plug in any [Model Context Protocol](https://modelcontextprotocol.io) server
  (stdio or remote HTTP/SSE) by declaring it in `config.yaml`; its tools join the agent behind
  the **same approval gate** as everything else. Remote tools always prompt until *you* lower
  their risk tier — a server's own "read-only" claim is never trusted. `/mcp` shows status.
- **Human-in-the-loop planning** — pause and edit the agent's plan mid-run if it's heading the
  wrong way, or type a correction and press Esc to steer the running turn.
- **Prompt-injection quarantine** — web pages, API responses, and remote tool results are
  untrusted input. Content that tries to steer the agent ("ignore your previous instructions",
  "run this command") is detected, visibly flagged in the trace, fenced off as data the model
  must not obey — and the next tool action faces your approval gate regardless of risk tier, so
  a malicious page can't quietly redirect the agent.
- **Trust receipt when it matters** — the stats line under a response says exactly how many
  bytes went to which host and how many actions faced the approval gate — whenever anything
  actually left your machine or was gated. A fully-local turn stays clean: silence means
  nothing left. `/policy` carries the full readout on demand.
- **Hooks** — `~/.saturn/hooks.yaml` runs your own shell commands on `turn-start`, `turn-end`,
  `before-write` and `after-write`; a `before-write` hook that exits non-zero blocks the write.
- **Per-workspace instructions** — `/init` surveys your workspace and drafts `SATURN.md`,
  standing instructions loaded every turn (like a per-project system prompt).
- **Headless mode** — `saturn -p "query"` runs one query and prints the answer; piped stdin
  attaches to the turn (`git diff | saturn -p "review this change"`). Gated tools are denied by
  default (no human at the gate) unless you pass `--yolo`. Add `--json` for a machine-readable
  result (answer, plan, tools, tokens, timing, plus a `gates` record of which calls were
  prompted and denied); `--export <file>` writes the run's replayable export record after the
  turn; `saturn --replay <file>` renders an exported record offline. The CLI is strict:
  unknown flags exit 2 instead of silently launching the chat loop.
- **One-shot query mode** — `saturn -q "question"` is the pipe-friendly rendering of the same
  headless turn: stdout carries *only* the final answer, progress lines go to stderr, and the
  run auto-exports so the closing `recorded: saturn --replay <file>` receipt replays exactly
  what happened — plan, gates, tool calls — offline. Same engine, same deny-by-default gate.

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
in an isolated virtualenv, pulls the small local models, and puts a `saturn` command on your PATH.

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/logansundaram/saturn/main/install.sh | sh
```

Then open a new terminal and run `saturn`. The first run pulls a few GB of models, so it takes a
minute. Prefer to read before you pipe? The script is plain text at the URL above — download
and inspect first.

The installer defaults to the lightweight **`4b`** size class (`qwen3.5:4b`). On the first launch
`/models` reads your hardware, prices every size against it, and asks which tier and embedder to
run — Enter takes the recommendation, and anything not pulled yet is pulled on consent. Re-run
`/models` anytime, or set `SATURN_TIER=9b` (or `27b`/`35b`) before installing.
Other knobs: `SATURN_INSTALL_DIR` (install dir), `SATURN_MODELS` (models to pull), `SATURN_BRANCH`.

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

Then run `saturn`. You still need [Ollama](https://ollama.com/download) running and the tier
models pulled — for the `4b` tier that's `ollama pull qwen3.5:4b` and
`ollama pull qwen3-embedding:8b` (multi-GB downloads; Ollama prints each one's exact size as the
pull starts). The quick installer above does both for you, and the first launch's `/models` page offers to
run any missing pull for you (y/N, default no); later launches warn about anything missing. Installed this way, your data and `config.yaml` live in `~/.saturn` beside your `SATURN.md`
(override with `SATURN_HOME`; an earlier install's `~/.saturday` keeps being used), and you upgrade with `pipx upgrade saturn-agent` / `uv tool upgrade
saturn-agent` instead of `/update`.

### Manual install (from source)

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running locally.
- The local models pulled. The default (`4b`) tier uses one small qwen3.5 model for everything
  plus an embedding model — both multi-GB downloads (Ollama prints each one's exact size as the
  pull starts):

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
  > (Small models are still less reliable at tool-calling — see the gotchas in `CLAUDE.md`.)

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
» what's 15% of 2,340, and find me the latest news on local LLMs?
» read the file notes.md in my workspace and summarize it
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

Saturn runs on macOS (the native Notes / Calendar / Mail tools and notifications need it)
and Linux (file, shell, web and knowledge-base tools). The `run_shell` tool hands commands to
`/bin/sh`, so write Unix shell syntax (`ls`, `&&`, `|`, etc.). Windows support was dropped
2026-09-27.

---|---|---|
| Launcher | `saturn.cmd` | `./saturn.sh` (run `chmod +x saturn.sh` once) |
| Shell tool syntax | PowerShell | `/bin/sh` (`bash`, `zsh`, etc.) |
| Python command | `python` | `python3` (or `python` inside a venv) |

The `run_shell` tool hands commands directly to the host shell, so write Unix shell syntax
(`ls`, `&&`, `|`, etc.) on macOS/Linux and PowerShell syntax on Windows.

---

## Useful commands

Type `/help` for the full list, or `/<command> --help` for details on any one. Highlights:

| Command | What it does |
|---|---|
| `/help` | The grouped command list, opening with the trust-stack map (posture · activity · proof); `/help <cmd>` details one. |
| `/models` | The model page: your hardware, the qwen ladder (six chat sizes + three embedders) priced against it, pick a row to switch — pulling what's missing on consent. |
| `/config` | View/edit settings (`/config runtime.num_ctx <size|auto>` resizes the context window). |
| `/plan` | Show the plan; control review mode and the mid-run pause (bare subcommands report status). |
| `/draft` | Write your OWN plan in the step editor — your next message executes YOUR steps instead of the agent's draft (same per-step reflection and approval gates). |
| `/docs` | The knowledge base: list documents, `add <path>`, `remove <name>`, `rebuild` (launch syncs on its own). |
| `/tools` | List the agent's tools and their risk tiers. |
| `/mcp` | MCP server status + the remote tools they add; `reload` after a config edit. |
| `/memory` | See, add, edit, and review the layered facts the agent permanently remembers; `review` is the gated learning step (also runs at `/quit`), `why <n>` its provenance. |
| `/policy` | Your trust settings in one place: bare = what runs without asking and what can leave the machine; `risk`/`allow`/`open` are the gate's levers, `egress` the ledger of what left, `airgap` the seal (bare forms report, changing is always explicit). |
| `/trace source` | Show the full material behind a citation `[n]` of the last answer (folded in from `/source`). |
| `/undo` | Revert the file changes of the last turn that wrote anything. |
| `/init` | Survey the workspace and draft `SATURN.md` standing instructions. |
| `/trace` | Inspect past runs, tool I/O, and LLM calls; `/trace why` explains a run's decisions; `/trace export` writes the run's complete record as JSON; `/trace replay` (or `saturn --replay <file>`) re-renders an exported record anywhere — no database needed. |
| `/resume` | Continue your last session (autosaved); `save [name]`/`list`/`<name>` for named sessions (plain `.json` files under `database/sessions/`). |
| `/update` | Self-update: pull the latest Saturn (your data is never touched). |
| `/clear` · `/quit` | Start a fresh conversation / exit. |

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

For deeper research, the agent plans multiple search + read steps — visible in the plan rail,
every call traced — rather than hiding them inside a monolithic research tool.

---

## Project layout

```
agent.py            # entry point: routes the command line into app/
app/                # the application shell: CLI, graph assembly, turn driver, REPL
config.yaml         # all settings: models, paths, safety, web knobs
core/               # the engine room: state, model factory, prompts, structured output
trust/              # the trust stack: gate policy, egress ledger, quarantine,
                    #   trust receipt
tools/              # the agent's tools (web, files, shell, calculator, knowledge)
                    #   + the registry and MCP client
nodes/              # the graph's nodes: ground, agent, approval, tools
commands/           # slash commands (one module per /help theme)
stores/             # persistence: RAG + its manifest, durable memory, snapshots, trace
tui/                # the terminal UI / live trace rail
docs/               # the documents: README.md indexes them (ARCHITECTURE.md is the code map)
database/           # your data: documents/, workspace/, memory/, caches, trace DB
```

See **`docs/ARCHITECTURE.md`** for the guided code map, and **`CLAUDE.md`** for the deep
architectural reference (including the roadmap).

---

## Benchmarking

The benchmark is the **graded trust benchmark** — it measures the trust stack itself:
approval-gate coverage (every non-read-only tool call must have faced the gate), the
injection-quarantine flag rate (a planted instruction-shaped document must be fenced), and the
memory tasks (recall across runs, supersession, and a planted memory that must face the gate):

```bash
python benchmark.py                                   # the trust benchmark
python benchmark.py --strict                          # …and exit 1 on any graded FAIL (CI-friendly)
```

Reports are written to `logging/benchmarks/trust_<ts>.json`.

---

## Status

Saturn (by Saturday.ai) is an actively developed, terminal-native agent — a trust-first agent
built around privacy, local execution, and auditability, not a general-purpose assistant racing
on breadth. The terminal is the product — there is no GUI on the roadmap, by design, and no plans
for consumer integrations (email/calendar/Drive). Current focus: first-run reliability across platforms, exportable
trace records (the seed of an audit layer), MCP client support behind the existing risk-tier
approval system, and a public trust benchmark. Contributions and feedback welcome — file issues
at the GitHub repo.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
