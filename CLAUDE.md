# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Saturn is a local-first, terminal-native AI agent (Python 3.11+, LangGraph, Ollama only). The product
thesis is the **trust stack**: every action is traced, every risky action faces a human gate, every byte
that leaves the machine is recorded in an egress ledger, and every run is replayable.
`docs/ARCHITECTURE.md` is the guided code map and reading order; read it before touching `nodes/` or `trust/`.

Note: `CLAUDE.md` is tracked (its `.gitignore` entry was removed 2026-09-02).

## Commands

```bash
# setup (the CI way — a dep missing from pyproject.toml fails here, not in pipx installs)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]                # or: pip install -r requirements.txt  (keep both lists in sync)

# run
python agent.py                      # interactive TUI (./saturn.sh prefers the repo .venv)
saturn -p "query"                    # headless: one answer to stdout; gated tools DENIED unless --yolo
saturn -q "query"                    # pipe-friendly one-shot (answer only on stdout, auto-exports)
saturn -p "query" --json --export run.json
saturn --replay logging/exports/run_1.json   # render an exported record, no DB/models needed

# tests — fully offline (no Ollama, no network, no embedder); ~4s for ~1300 tests
python -m pytest tests/ -q
python -m pytest tests/test_engine.py -q                 # one file
python -m pytest tests/test_policy.py -q -k prefix       # one test by name

# trust benchmark — needs a running Ollama with the active tier pulled
python benchmark.py [--strict]       # reports to logging/benchmarks/trust_<ts>.json

# dev utilities (live daemon, not shipped in the wheel)
python utilities/confidence_calibrate.py --models qwen3.8:27b   # regenerates core/confidence_calibration.py
python utilities/continuation_contract.py                       # the "supported model" contract for token steering

SATURDAY_DEBUG=1 python agent.py     # echo logging/diag.log lines to stderr
```

There is no linter or formatter configured. CI (`.github/workflows/tests.yml`) runs the suite on
ubuntu + windows × Python 3.11–3.13 and smoke-tests the built wheel in a clean venv.

**Releases:** push a `v*` tag. `pyproject.toml` `version`, `app/__init__.py` `__version__`, and a
`## [x.y.z]` section in `CHANGELOG.md` must all agree (`tests/test_version.py` pins the first pair on
every push). User-visible changes go under `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog format).

Commit messages follow `area: what changed` in lowercase (`gate: …`, `trace: …`, `review: …`).

## Architecture

### Life of a turn

`agent.py` is a thin router into `app/`; `app/graph.py::build_agent` compiles the graph from `nodes/`
(one file per node) with a SqliteSaver checkpointer, and `app/turn.py::run_turn` streams it:

```
ground → plan → plan_gate → execute → [approval] → tools → update_plan → rectify → (replan | plan_gate | synthesize)
```

- `ground` assembles `state["context"]` (SATURDAY.md, the memory selection for this request,
  manifests, attachments).
- `plan` drafts the step list through `core/structured.py`. `/draft` pre-seeds a user-authored plan and
  the planner call is skipped.
- `execute` runs ONE step per pass with a curated context (`core/plan_context.py`), never raw history.
  Three shapes: reasoning step (no tool), write step (semantic write gate first), tool step (a single
  call bound to exactly the planned tool; arg recovery in `core/tool_args.py`).
- `approval` asks `trust/policy.approves(name, risk, args)` — the ONE gate question — and interrupts the
  graph for the human when it says no.
- `tools` (the node, not the package) executes, clamps the observation, records egress, fences
  injection-suspicious content (`trust/quarantine.py`).
- `rectify` reflects after EVERY step. Its branch order is load-bearing and deterministic-first:
  guarded outcome → no-call guard → resolution check → concrete pending → dead-end retry → LLM verdict. Do not reorder
  without reading the module docstring; `tests/test_engine.py` pins each branch.
- `synthesize` streams the answer from recorded step results, disclosing incidents and citing `[n]`.

### The plan is the data bus (`core/state.py`)

Steps are **plain dicts** `{step_id, label, status, intended_tool, result, needs_resolution}` — never a
custom class (the checkpointer serializer won't round-trip it). `current_step()` = the first step whose
`result is None`; that is THE execution pointer. Any status other than `done` (`skipped`, `blocked`,
`error`, `cancelled`) is an incident synthesize must disclose. `gate_events` is the only
non-recomputable record (human decisions).

When slicing conversation history, use `core.state.is_turn_start` — a mid-turn steer note is a
`HumanMessage` with `STEER_PREFIX` and is NOT a turn boundary; a hand-rolled isinstance check mis-slices.

### Models and config

Code references model **roles** (`planner`, `tool_caller`, `synthesizer`, `utility`, `judge`) via
`core/llms.get_model(role)`; never name a model in graph code. Roles resolve through `active_tier` →
`tiers` in `config.yaml`. Ollama is the only provider; cloud bindings refuse to build (shelved
2026-07-03, reintroduction seam documented in `core/llms.py`). The qwen3.5/3.6/3.8 ladder is closed
(per-model confidence calibration in `core/confidence_calibration.py`, a generated module).

`config.yaml` is **gitignored user data**, seeded on first run from the tracked template
`config.default.yaml` (or `$SATURDAY_HOME/config.yaml` for wheel installs). Change defaults in the
template. `config.persist()` does a surgical single-line YAML edit to preserve comments — don't replace
it with a full dump. `config.py`, `diag.py`, `textutil.py` import nothing project-side and are safe
leaves; `diag.log()` replaces `print()` in nodes/tools (stdout collides with the rich Live TUI).

### Memory (`stores/memory_registry.py`, `core/memory_review.py`)

One markdown file (`paths.memory`), six `## layer` sections: `user` and `commitments` load every
turn, the last five `memo` entries too, `agent` / `entities` / `negative` (and any other
heading a hand edit or `layer=` introduced) only by token match against the request — all under
`memory.context_cap`, with a trailer naming what didn't load. `sens=` facts are withheld from
both the block and `recall` when inference is not local.
Every bullet ends in a `{#id by=user|inferred run=N used=DATE n=K sens=… due=…}` metadata token;
ids come from the `<!-- next-id -->` high-water mark and are never reused (`replaces=#id`
supersedes; `/memory why <n>` points at `/trace why #run`). Learning is gated: `core/memory_review`
queues candidates from each turn's state and from compaction summaries into
`database/memory/pending_review.json`; `/memory review` (also `/quit`) accepts them one at a
time. Never write a fact without a user action (a gated `remember`, `/memory add`, or a review
accept). The benchmark's memory tasks and `tests/test_memory_*.py` pin this.

### Trust stack (`trust/`)

- `policy.py` — one object behind `/policy risk|allow|open`, `runtime.auto_approve`, and `--yolo`.
  Shell prefix matching is token-based and refuses metacharacters. Persisted in `database/permissions.json`.
- `egress.py` — every outbound network op calls `check()` (air-gap) then `record()`. The complete list
  of egress chokepoints is `core/llms.py`, `core/continuation.py`, `tools/web.py`, `tools/mcp_client.py`;
  `tests/test_no_new_egress.py` fails on a network-client import anywhere else. A new chokepoint is a
  deliberate edit to that test plus check/record wiring. A remote `OLLAMA_HOST` counts as egress.
- `quarantine.py` — untrusted output (web, MCP, corpus) is scanned, fenced as data, and the next tool
  batch is escalated to the gate.

### Tools

Define a tool in its own module under `tools/` with `@register_tool(risk=...)` from `tools/toolspec.py`;
`tools/registry.py` imports the modules to trigger registration — nothing else to edit. Unknown risk fails
closed to `destructive`; `run_shell` is always `destructive`. MCP tools register as `mcp_<server>_<tool>`
and never trust a server's self-declared tier. `tools/toolspec.py` is separate from `registry.py`
precisely to avoid the import cycle — keep it that way.

`notify/` is the scheduled-notification seam behind `schedule_notification` and `/notify`: `backend()` picks
by `sys.platform` (macOS = one launchd LaunchAgent per one-shot, shown by `osascript`; anything else is the
honest `Unsupported`). A new platform is one module plus one branch in `backend()`. It is not egress.
`notify/menubar.py` (tested, Cocoa-free) + `notify/menubar_app.py` (AppKit, pyobjc, macOS-only dep) are the
menu bar icon: a login LaunchAgent the REPL starts, which outlives the terminal; its Quit is `quit_all()`.

Native macOS app tools (`tools/notes.py`, `tools/calendar.py`, `tools/mail.py`) go through `tools/applescript.py` — `run(script,
app=)` opens the app hidden then runs `osascript` (osascript alone gets -600 on a closed Calendar); output is
RS/US-delimited via `records()`. AppleScript, not EventKit: EventKit access from a terminal Python depends on the
terminal app's Info.plist. Readers are `untrusted=True` (shared notes, invitations, email); tests capture `applescript._run`.
`draft_mail` opens an unsent draft and is NOT egress; a `send_mail`/`send_message` would be a new egress chokepoint
(deferred, with the Messages-history findings, in `docs/superpowers/specs/2026-09-06-macos-apps.md`).

### Slash commands

`commands/_framework.py` provides `@command(name, summary, aliases=, usage=, details=)`; one module owns
every view of a feature (`commands/trace.py` = `/trace` + export/replay engine, etc.). Every command
must accept `--help`; cut command spellings live in `_RENAMED` and print pointers rather than vanishing.
Shared verb grammar (remove/rm/delete/…, `--save`) is in `commands/_utils.py`.

### Same name, different file

`tools/` (implementations) vs `nodes/tools.py` (execution node) · `trace`: `stores/trace.py` records,
`tui/ui/trace.py` renders the rail, `commands/trace.py` is `/trace` · `plan`: `nodes/`, `tui/ui/`,
`commands/` · `config.py` (loader) vs `commands/config.py` (`/config`) · `trust/policy.py` (mechanism)
vs `commands/policy.py` (front door).

### Tests

`tests/conftest.py` adds the repo root to `sys.path`; use the `isolated_paths` fixture whenever a test
touches configured paths so it can never write to the real `database/`. Grant lifecycle in
`trust/policy` is reset around every test automatically. LLM seams are monkeypatched at each node's
namespace — no test may reach a model, the network, or the embedder. `tests/` and `benchmark.py` import
compatibility names from `agent` (`from agent import build_agent, run_turn, …`); new code should import
from `app/` directly.

### Docs

`docs/ARCHITECTURE.md` — code map. `docs/OPTIMIZATIONS.md` — latency techniques: shipped, next, and
to-measure, with the numbers behind them. `PLAN.md` — product focus and verified gap list (moat vs. trim).
`CHANGELOG.md` — user-visible history. `docs/FEATURE_INVENTORY.md` — historical, pre-transplant
snapshot only. `docs/superpowers/` — planning specs from past feature work.
