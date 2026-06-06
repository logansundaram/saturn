# Saturn

> A **local-first, transparent, general-purpose** AI agent that runs on your own machine.

Saturn is a personal copilot you run in your terminal. It can search the web, read and
write your files, retrieve from your own document library, do math, and remember things across
sessions — and it shows you **every step it takes** while doing it. Nothing is a black box: you
watch the plan it draws up, the tools it calls, and you approve anything that touches the outside
world before it happens.

It's built to run on **local models** (via [Ollama](https://ollama.com)) so your data stays on
your machine by default, with optional cloud models when you want more horsepower.

---

## Why Saturn?

Most AI assistants are a chat box in front of a remote model. Saturday.ai is built around four
priorities, in order:

- **Transparency** — A live execution trace shows each node, the evolving plan, every tool
  call and its result. You're never guessing what it did.
- **Configurability** — One `config.yaml` controls which models run, the safety policy, the
  context window, and paths. Most of it is also tweakable live with slash commands.
- **Safety** — An approval gate stops before any side-effecting or destructive action (writing
  a file, hitting the web) and asks you first. You decide the policy.
- **Extensibility** — Tools, commands, nodes, and models are small, registry-based modules.
  Adding a capability is adding a file, not rewiring the app.

---

## What it can do

- **Multi-step reasoning** — a living-plan ReAct loop: it drafts a plan, executes one step at a
  time, sees each tool result, and decides the next action.
- **Web search & research** — works **with or without an API key**. With a
  [Tavily](https://tavily.com) key you get premium search/extract/research; without one it falls
  back automatically to keyless DuckDuckGo search + local page extraction.
- **Your files** — read, write, and list files in a sandboxed workspace.
- **Your documents (RAG)** — ingest PDFs/text/markdown into a local knowledge base it can search.
- **Math** — a precise calculator tool.
- **Memory** — durable facts that persist across sessions (`remember` / `recall`).
- **Shell commands** — run arbitrary shell commands (scripts, build tools, git, package managers)
  in the sandboxed workspace. Uses PowerShell on Windows and `/bin/sh` on macOS/Linux — write
  commands in your platform's native syntax.
- **Human-in-the-loop planning** — pause and edit the agent's plan mid-run if it's heading the
  wrong way.

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
git clone <your-repo-url> saturday
cd saturday

# (recommended) a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

> `prompt_toolkit` (live command highlighting) and `tavily-python` (premium web search) are
> optional — Saturday runs fine without either.

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
  `num_ctx` (context window), `lockstep`.
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
| `/context` | Show / resize the model context window. |
| `/plan` | Show the plan; control review mode, mid-run pause, and lockstep. |
| `/ingest <path>` | Add a document to the knowledge base. |
| `/docs` | List ingested documents and workspace files. |
| `/tools` | List the agent's tools and their risk tiers. |
| `/risk` · `/autoapprove` | Tune the safety gate. |
| `/trace` · `/calls` | Inspect past runs and tool I/O. |
| `/save` · `/load` | Persist and restore a conversation. |
| `/reset` · `/quit` | Clear the conversation / exit. |

---

## Web search without an API key

Saturday treats Tavily as an **upgrade, not a requirement**:

- **`web_search`** prefers Tavily when a healthy key exists and **automatically falls back to
  keyless DuckDuckGo** on a missing key or quota error.
- **`web_extract`** reads pages **locally** (via `trafilatura`) — no key, no API call.
- **`deep_research`** uses Tavily's research job when available, otherwise runs a local loop:
  search → read top results → synthesize.

So you can use every web feature with zero keys. Control the backend with `web.provider` in
`config.yaml`.

---

## Project layout

```
agent.py            # entry point: builds the graph + runs the interactive loop
config.yaml         # all settings: models, paths, safety, web provider
commands.py         # slash-command layer
node_registry/      # the graph's nodes (ground, plan, agent, tools, synthesize, …)
tool_registry/      # the agent's tools (web, files, calculator, knowledge, memory)
stores/             # persistence: RAG, document manifests, durable memory, trace
tui/                # the terminal UI / live trace rail
database/           # your data: documents/, workspace/, memory/, caches, trace DB
```

See **`CLAUDE.md`** for a deep architectural tour and **`SATURDAY_MVP_PLAN.md`** for the vision
and roadmap.

---

## Benchmarking

A query-suite harness exercises each capability:

```bash
python benchmark.py                                   # all suites
python benchmark.py --suites calculator web_search    # just some
```

Reports are written to `logging/benchmarks/`.

---

## Status

Saturday.ai is an actively developed MVP: a CLI-first personal copilot today, with an
Electron/React frontend and integrations (email, calendar, Drive) on the post-MVP roadmap. It's
built so adding those is straightforward. Contributions and feedback welcome.

---

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
