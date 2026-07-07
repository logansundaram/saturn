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
3. **`/trace answer`** — answer-level provenance: each cited source's origin (local vs network)
   and trust, and what left the machine this turn.
4. **`/privacy egress`** — the per-event ledger: exactly what left, channel / host / bytes.
5. **Make it ask.** `» save a two-line summary to notes.md` — the approval gate shows the exact
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
- **It asks instead of guessing** — when a needed value, choice, or confirmation is missing,
  the agent pauses mid-run with one question (`ask_user`), and your typed answer resumes the
  turn. The alternative to asking is fabrication; Saturn asks.
- **Shell commands** — run arbitrary shell commands (scripts, build tools, git, package managers)
  in the sandboxed workspace. Uses PowerShell on Windows and `/bin/sh` on macOS/Linux — write
  commands in your platform's native syntax. Every run is a bounded **foreground** run: the
  process lives and dies inside the turn you approved (no detached-process surface), with a
  timeout so a hung command can't wedge the turn.
- **Cited answers** — answers that drew on tools or documents cite their sources inline (`[1]`)
  and end with a Sources list mapping each number to the exact tool call or document behind it;
  `/trace source 3` shows the full material behind any citation.
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
  nothing left. `/privacy` and `/trace answer` carry the full readout on demand.
- **Per-workspace instructions** — `/init` surveys your workspace and drafts `SATURDAY.md`,
  standing instructions loaded every turn (like a per-project system prompt).
- **Headless mode** — `saturn -p "query"` runs one query and prints the answer; piped stdin
  attaches to the turn (`git diff | saturn -p "review this change"`). Gated tools are denied by
  default (no human at the gate) unless you pass `--yolo`. Add `--json` for a machine-readable
  result (answer, plan, tools, tokens, timing, plus a `gates` record of which calls were
  prompted and denied); `--export <file>` writes the run's replayable export record after the
  turn; `saturn --replay <file>` renders an exported record offline. The CLI is strict:
  unknown flags exit 2 instead of silently launching the chat loop.

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
models pulled — for the `laptop` tier that's `ollama pull gemma4:e4b` and
`ollama pull qwen3-embedding:8b` (multi-GB downloads; Ollama prints each one's exact size as the
pull starts). The quick installer above does both for you, and `/config setup` reports what's
missing — when models are missing it offers to run the pulls for you (y/N, default no). Installed this way, your data and `config.yaml` live in `~/.saturday` (override with
`SATURDAY_HOME`), and you upgrade with `pipx upgrade saturn-agent` / `uv tool upgrade
saturn-agent` instead of `/update`.

### Manual install (from source)

### 1. Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/download)** installed and running locally.
- The local models pulled. The default (`workstation`) tier uses one ~9B model for everything
  plus an embedding model — both multi-GB downloads (Ollama prints each one's exact size as the
  pull starts):

  ```bash
  ollama pull qwen3.5:9b           # all five roles (~9B)
  ollama pull qwen3-embedding:8b   # the embedder (RAG)
  ```

  > Lighter on hardware? Edit `active_tier` in `config.yaml` to `laptop` and pull the smaller
  > `gemma4:e4b` instead (same embedder). (Small models are less reliable at tool-calling — see
  > the gotchas in `CLAUDE.md`; `/config setup` will say so too.)

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

> `prompt_toolkit` (live command highlighting) is optional — Saturn runs fine without it.

There is **no API key step**: web search is keyless and inference is local. (Custom env vars —
e.g. for an MCP server's `${VAR}` expansion — can still be stored with `/config key set MY_VAR
<value>`; they live in a `.env` file and apply immediately.)

### 3. Run it

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
> Both launchers prefer the repo's own `.venv` interpreter when one exists, and no longer `cd`
> into the repo — relative paths in your arguments resolve against *your* directory, and nothing
> leaks a directory change into your shell.
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

- **`active_tier`** — which hardware preset is live (`laptop`, `workstation`).
- **`tiers`** — maps each role (planner / tool_caller / synthesizer / …) to a concrete model, so
  swapping hardware is a one-line change.
- **`runtime`** — loop and safety knobs: `max_iterations`, `auto_approve` (the approval policy),
  `num_ctx` (context window), `citations` (inline source citations in answers).
- **`web`** — web-tool knobs (`max_results`, `request_timeout`). The backend is fixed and
  keyless: DuckDuckGo search + local extraction.

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
| `/help` | The grouped command list, opening with the trust-stack map (posture · activity · proof); `/help <cmd>` details one. |
| `/models` | List installed Ollama models; switch what drives each role (`--save` persists a binding to `config.yaml`). |
| `/config` | View/edit settings and **API keys** (`/config key …`); `/config setup` is the health check; `/config context` is the runtime readout (context window + fill, CPU/RAM/GPU) + window resize (`--save` persists). |
| `/plan` | Show the plan; control review mode and the mid-run pause (bare subcommands report status). |
| `/docs` | The knowledge base: list documents, `add <path>`, `remove <name>`, `sync`. |
| `/tools` | List the agent's tools and their risk tiers. |
| `/mcp` | MCP server status + the remote tools they add; `reload` after a config edit. |
| `/memory` | See, add, or delete the facts the agent permanently remembers. |
| `/policy` | The whole safety posture as one object: bare = status; `risk`/`allow`/`open` are its levers (bare forms report, changing is always explicit). The old `/risk`/`/allow`/`/autoapprove` spellings print a pointer here. |
| `/trace source` | Show the full material behind a citation `[n]` of the last answer (folded in from `/source`). |
| `/trace answer` | Answer-level provenance — each cited source's origin + trust, what left the machine, and the human gate decisions (folded in from `/glass`; `#id` for past runs). |
| `/privacy` | The privacy surface: what CAN leave (`/privacy`), what DID (`/privacy egress`), seal the boundary (`/privacy airgap`), and strip secrets from off-machine sends (`/privacy redact`). |
| `/undo` | Revert the file changes of the last turn that wrote anything. |
| `/rewind` | Drop the last exchange from the conversation (files untouched — that's `/undo`). |
| `/retry` | Regenerate the last answer; `/retry full` re-runs the whole turn from scratch. |
| `/init` | Survey the workspace and draft `SATURDAY.md` standing instructions. |
| `/trace` | Inspect past runs, tool I/O, LLM calls, cost; `/trace why` explains a run's decisions; `/trace export` writes the run's complete record as JSON; `/trace replay` (or `saturn --replay <file>`) re-renders an exported record anywhere — no database needed. |
| `/resume` | Continue your last session (autosaved); `save`/`list`/`delete`/`rename`/`<name>` for named sessions. |
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
  contacted.

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
                    #   trust receipt, answer provenance
tools/              # the agent's tools (web, files, shell, calculator, knowledge)
                    #   + the registry and MCP client
nodes/              # the graph's nodes (ground, plan, execute, tools, rectify, synthesize, …)
commands/           # slash commands (one module per /help theme)
stores/             # persistence: RAG, document manifests, durable memory, trace
tui/                # the terminal UI / live trace rail
docs/               # ARCHITECTURE.md — the code map for reading the source by hand
database/           # your data: documents/, workspace/, memory/, caches, trace DB
```

See **`docs/ARCHITECTURE.md`** for the guided code map, and **`CLAUDE.md`** for the deep
architectural reference (including the roadmap).

---

## Benchmarking

The headline is the **graded trust benchmark** — it measures the trust stack itself: the
grounding judge's catch rate (queries that bait a confabulated answer, graded on whether the
agent looked the fact up or the judge caught the ungrounded draft) and approval-gate coverage
(every non-read-only tool call must have faced the gate). Ungraded capability suites exist too,
as regression checks on the loop's mechanics:

```bash
python benchmark.py                                   # the trust benchmark (the headline)
python benchmark.py --strict                          # …and exit 1 on any graded FAIL (CI-friendly)
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
