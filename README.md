# Saturn

> A **private, local-first AI agent** that runs on your own machine — every step it takes is
> visible, auditable, and yours to approve. Built by **Saturday.ai**.

Saturn is a personal agent you run in your terminal — and the whole point is that you can see
what it's doing. It plans its work in the open, then calls tools to search the web, read and
write your files, query your own documents, run commands, and remember things across sessions —
showing you **every step it takes** and pausing for your approval before anything touches the
outside world. Nothing is a black box: you watch the plan it draws up and the tools it calls,
and any run can be replayed afterward.

It runs on **local models** (via [Ollama](https://ollama.com)) by default, so your data stays on
your machine; cloud models are an opt-in upgrade when you want more horsepower, never a requirement.

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
- **Web search** — works **with or without an API key**. With a [Tavily](https://tavily.com)
  key you get premium search; without one it falls back automatically to keyless DuckDuckGo
  search + local page extraction.
- **Your APIs** — `http_request` talks to any REST endpoint or self-hosted service (Home
  Assistant, Gitea, Jellyfin, …), and shows you the exact request — method, URL, headers,
  body — for approval before anything is sent. One auditable tool instead of fifty opaque integrations.
- **Your files** — read, write, edit (anchored string replace), search (content regex + name
  glob), and list files in a sandboxed workspace — with pre-write snapshots, so `/undo` can
  revert any turn's file changes.
- **Your documents (RAG)** — ingest PDFs, text, markdown, HTML, CSV, and Word (.docx) files into
  a local knowledge base it can search.
- **Math & time** — a precise calculator and the machine's own clock, so arithmetic and
  "today" are computed, never guessed from memory.
- **Memory** — durable facts that persist across sessions (`remember` / `recall`), fully
  inspectable and editable with `/memory`.
- **Shell commands** — run arbitrary shell commands (scripts, build tools, git, package managers)
  in the sandboxed workspace. Uses PowerShell on Windows and `/bin/sh` on macOS/Linux — write
  commands in your platform's native syntax. Long-running processes (a dev server, a watcher)
  can run **in the background** with their output captured to a log the agent can check and a
  job it can stop; anything still running when you quit is cleaned up.
- **Cited answers** — answers that drew on tools or documents cite their sources inline (`[1]`)
  and end with a Sources list mapping each number to the exact tool call or document behind it;
  `/source 3` shows the full material behind any citation.
- **MCP servers** — plug in any [Model Context Protocol](https://modelcontextprotocol.io) server
  (stdio or remote HTTP/SSE) by declaring it in `config.yaml`; its tools join the agent behind
  the **same approval gate** as everything else. Remote tools always prompt until *you* lower
  their risk tier — a server's own "read-only" claim is never trusted. `/mcp` shows status.
- **Human-in-the-loop planning** — pause and edit the agent's plan mid-run if it's heading the
  wrong way, or type a correction and press Esc to steer the running turn. A plan that worked
  can be saved as a **recipe** (`/plan save`) and re-run any time with fresh approvals.
- **Prompt-injection quarantine** — web pages, API responses, and remote tool results are
  untrusted input. Content that tries to steer the agent ("ignore your previous instructions",
  "run this command") is detected, visibly flagged in the trace, fenced off as data the model
  must not obey — and the next tool action faces your approval gate regardless of risk tier, so
  a malicious page can't quietly redirect the agent.
- **Trust receipt on every answer** — the stats line under each response also says what left
  your machine that turn: `local-only`, or exactly how many bytes went to which host, plus how
  many actions faced the approval gate. The privacy claim, proven per answer.
- **Policy as a file** — your whole safety posture (approval threshold, per-tool risk levels,
  shell allowlist, air-gap and redaction modes) exports to one shareable YAML profile
  (`/policy export`) and applies anywhere — including headless runs (`saturn --policy ci.yaml`).
- **User-defined commands** — drop a markdown template into `database/commands/` and it becomes
  a slash command (`/brief notes.md`); `$ARGUMENTS` expands, and the template runs as a normal
  gated, traced agent turn.
- **Per-workspace instructions** — `/init` surveys your workspace and drafts `SATURDAY.md`,
  standing instructions loaded every turn (like a per-project system prompt).
- **Headless mode** — `saturn -p "query"` runs one query and prints the answer; gated tools are
  denied by default (no human at the gate) unless you pass `--yolo`. Add `--json` for a
  machine-readable result (answer, plan, tools, tokens, timing) you can pipe into scripts.

---

## How it works (the short version)

Every turn flows through a graph of small, inspectable steps:

```
ground → plan → [review?] → agent → [approval?] → tools → update plan → … → synthesize
```

- **ground** loads your profile, memory, and document/workspace manifests.
- **plan** drafts a step-by-step plan (the transparency surface you can inspect and edit).
- **agent** picks the next tool to call (or finishes).
- **approval** pauses for your OK before anything side-effecting runs.
- **tools** run; results flow back so the agent can decide what's next.
- **synthesize** writes the final answer from what was actually gathered.

The plan is a first-class, editable object — it both *shows* you what's happening and *drives*
execution.

---

## Getting started

### Quick install (recommended)

One command. It installs [Ollama](https://ollama.com) if needed, clones Saturn into `~/.saturday`
in an isolated virtualenv, pulls the small local models, and puts a `saturn` command on your PATH.

```bash
# macOS / Linux / WSL2
curl -fsSL https://raw.githubusercontent.com/logansundaram/saturn/main/install.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/logansundaram/saturn/main/install.ps1 | iex
```

Then open a new terminal and run `saturn`. The first run pulls a few GB of models, so it takes a
minute. Prefer to read before you pipe? Both scripts are plain text at the URLs above — download
and inspect first.

The installer defaults to the lightweight **`laptop`** tier (`gemma4:e4b`); switch to a bigger
tier anytime with `/models`, or set `SATURDAY_TIER=workstation` before installing. Other knobs:
`SATURDAY_HOME` (install dir), `SATURDAY_MODELS` (models to pull), `SATURDAY_BRANCH`.

> Prefer to set it up by hand, or hacking on Saturn itself? Use the **Manual install** below.

### Install with pipx / uv

Already manage Python tools with [pipx](https://pipx.pypa.io) or [uv](https://docs.astral.sh/uv/)?
Saturn is pip-installable:

```bash
pipx install git+https://github.com/logansundaram/saturn
# or
uv tool install git+https://github.com/logansundaram/saturn
```

Then run `saturn`. You still need [Ollama](https://ollama.com/download) running and the tier
models pulled (the quick installer above does both for you; `/config setup` reports what's
missing). Installed this way, your data and `config.yaml` live in `~/.saturday` (override with
`SATURDAY_HOME`), and you upgrade with `pipx upgrade saturn-agent` / `uv tool upgrade
saturn-agent` instead of `/update`.

### Manual install (from source)

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running locally.
- The local models pulled. The default (`workstation`) tier uses one ~9B model for everything
  plus an embedding model:

  ```bash
  ollama pull qwen3.5:9b
  ollama pull qwen3-embedding:8b
  ```

  > Lighter on hardware? Edit `active_tier` in `config.yaml` to `laptop` and pull `gemma4:e4b`
  > instead. (Small models are less reliable at tool-calling — see the gotchas in `CLAUDE.md`.)

### 2. Clone and install

```bash
git clone https://github.com/logansundaram/saturn
cd saturn

# (recommended) a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

> `prompt_toolkit` (live command highlighting) and `tavily-python` (premium web search) are
> optional — Saturn runs fine without either.

### 3. (Optional) Add API keys

Web search works out of the box with **no key** (keyless DuckDuckGo). To upgrade to Tavily, or
to use cloud models, add keys — easiest via the built-in command once you're running:

```
/config key set TAVILY_API_KEY    tvly-...        # premium web search/extract/research
/config key set ANTHROPIC_API_KEY sk-ant-...      # cloud models (cloud-hybrid tier)
/config key                                       # list keys and whether each is set
```

These are saved to a `.env` file in the repo root and applied immediately.

### 4. Run it

```bash
python agent.py        # Windows / venv-activated
python3 agent.py       # macOS/Linux without a venv (if `python` isn't in PATH)
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
> **Windows:** `saturn.cmd` launches from anywhere. Wire a `saturn` function into your PowerShell
> profile to type just `saturn`.
>
> **macOS/Linux:** make `saturn.sh` executable once, then run it directly or add the repo to your
> `PATH`:
> ```bash
> chmod +x saturn.sh
> ./saturn.sh
> ```

---

## Configuration

Everything lives in **`config.yaml`**:

- **`active_tier`** — which hardware preset is live (`laptop`, `workstation`, `cloud-hybrid`).
- **`tiers`** — maps each role (planner / tool_caller / synthesizer / …) to a concrete model, so
  swapping hardware is a one-line change.
- **`runtime`** — loop and safety knobs: `max_iterations`, `auto_approve` (the approval policy),
  `num_ctx` (context window), `lockstep`, `token_budget` (a hard session spend ceiling — useful
  with cloud models), `citations` (inline source citations in answers).
- **`web`** — search backend: `auto` (prefer Tavily if a key exists, else DuckDuckGo), `tavily`,
  or `duckduckgo`.

Most of it is also adjustable **live** (session-only) with slash commands — handy for
experimenting without restarting.

### macOS / Linux notes

No platform-specific config is required — `config.yaml` works as-is on all platforms. The only
differences to know about:

| | Windows | macOS / Linux |
|---|---|---|
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
| `/help` | List all commands (or detail one). |
| `/models` | List installed Ollama models; switch what drives each role. |
| `/config` | View/edit settings and **API keys** (`/config key …`). |
| `/context` | Runtime readout (context window + fill, CPU/RAM/GPU); resize the window. |
| `/plan` | Show the plan; control review mode, mid-run pause, and lockstep; `save`/`run` plan recipes. |
| `/docs` | The knowledge base: list documents, `add <path>`, `remove <name>`, `sync`. |
| `/tools` | List the agent's tools and their risk tiers. |
| `/mcp` | MCP server status + the remote tools they add; `reload` after a config edit. |
| `/memory` | See, add, or delete the facts the agent permanently remembers. |
| `/risk` · `/allow` · `/autoapprove` | Tune the safety gate (persistable overrides + shell allowlist). |
| `/policy` | The whole safety posture as one object: show it, `export` it as a shareable YAML profile, `import` one (also `saturn --policy <file>`). |
| `/source` | Show the full material behind a citation `[n]` of the last answer. |
| `/commands` | List user-defined slash commands (markdown templates in `database/commands/`); `reload` after editing. |
| `/privacy` | The privacy surface: what CAN leave (`/privacy`), what DID (`/privacy egress`), seal the boundary (`/privacy airgap`), strip secrets from cloud sends (`/privacy redact`). |
| `/undo` | Revert the file changes of the last turn that wrote anything. |
| `/rewind` | Drop the last exchange from the conversation (files untouched — that's `/undo`). |
| `/retry` | Regenerate the last answer; `/retry full` re-runs the whole turn from scratch. |
| `/init` | Survey the workspace and draft `SATURDAY.md` standing instructions. |
| `/trace` | Inspect past runs, tool I/O, LLM calls, cost; `/trace why` explains a run's decisions; `/trace export` writes a tamper-evident run record; `/trace replay` re-renders an exported record anywhere — no database needed. |
| `/resume` | Continue your last session (autosaved); `save`/`list`/`delete`/`rename`/`<name>` for named sessions. |
| `/update` | Self-update: pull the latest Saturn (your data is never touched). |
| `/clear` · `/quit` | Start a fresh conversation / exit. |

---

## Web search without an API key

Saturn treats Tavily as an **upgrade, not a requirement**:

- **`web_search`** prefers Tavily when a healthy key exists and **automatically falls back to
  keyless DuckDuckGo** on a missing key or quota error.
- **`web_extract`** reads pages **locally** (via `trafilatura`) — no key, no API call.

So you can use every web feature with zero keys. Control the backend with `web.provider` in
`config.yaml`. For deeper research, the agent plans multiple search + read steps — visible in
the plan rail, every call traced — rather than hiding them inside a monolithic research tool.

---

## Project layout

```
agent.py            # entry point: builds the graph + runs the interactive loop
config.yaml         # all settings: models, paths, safety, web provider
commands/           # slash commands (one module per command)
node_registry/      # the graph's nodes (ground, plan, agent, tools, synthesize, …)
tool_registry/      # the agent's tools (web, files, shell, calculator, knowledge, memory)
stores/             # persistence: RAG, document manifests, durable memory, trace
tui/                # the terminal UI / live trace rail
database/           # your data: documents/, workspace/, memory/, caches, trace DB
```

See **`CLAUDE.md`** for a deep architectural tour (including the roadmap).

---

## Benchmarking

The headline is the **graded trust benchmark** — it measures the trust stack itself: the
grounding judge's catch rate (queries that bait a confabulated answer, graded on whether the
agent looked the fact up or the judge caught the ungrounded draft) and approval-gate coverage
(every non-read-only tool call must have faced the gate). Ungraded capability suites exist too,
as regression checks on the loop's mechanics:

```bash
python benchmark.py                                   # the trust benchmark (the headline)
python benchmark.py --capability                      # capability suites + conversations (regression)
python benchmark.py --capability --suites rag web_search   # just some suites
python benchmark.py --all                             # everything in one combined report
```

Reports are written to `logging/benchmarks/` (`trust_<ts>.json` / `benchmark_<ts>.json`).

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
