# Saturn — Architecture & Code Map

A reading guide for going through the codebase by hand. `CLAUDE.md` is the dense dev
reference (every mechanism, every historical decision); this file is the *map*: what lives
where, why it's grouped that way, and the order that makes the code easiest to absorb.

Saturn (Saturday.ai) is a local-first, transparent terminal agent. The product thesis is the
**trust stack**: every action is visible (trace), every risky action asks a human (gate),
every byte that leaves the machine is accounted for (egress ledger), and every answer can
show its provenance (/trace answer). The engine is a **plan/execute loop**: an LLM drafts a step
plan, each step executes one at a time against a curated context, and a judge reflects after
every step.

## The 30-second map

```
agent.py            entry point — parses the CLI, routes into app/ (thin; re-exports for tests)
benchmark.py        the graded trust benchmark + capability regression suites (dev-only)
config.py/.yaml     the single source of truth for model bindings, paths, and runtime knobs
diag.py             diagnostic logging to logging/diag.log (never print() — TUI-safe)
textutil.py         leaf text helpers (truncation, head+tail clamping, byte formatting)
env_keys.py         .env-backed secret management (the /config key front end)

app/        the application shell: CLI, graph assembly, turn driver, headless + REPL loops
core/       the engine room: state, model factory, prompts, structured output, plan plumbing
nodes/      the graph nodes, one per file (ground → plan → … → synthesize)
tools/      the tool implementations + registry + MCP client (risk tiers declared at definition)
notify/     scheduled desktop notifications: the platform seam, the macOS launchd/osascript backend, the menu bar item
trust/      the trust stack: gate policy, egress ledger, redaction, quarantine, receipt, answer provenance
commands/   the slash-command layer (/help themes, one module each)
stores/     data + persistence: RAG corpus, manifests, memory, snapshots, trace DB
tui/        presentation: the rich-based terminal UI, type-ahead reader, system metrics

tests/      offline pytest suite (no LLM, no network — conftest redirects all paths to tmp)
utilities/  dev-only helpers (graph rendering); not shipped in the wheel
docs/       this file + historical planning artifacts
database/   user data at runtime (corpus, workspace, memory, sessions, permissions…)
logging/    diagnostics, benchmarks, exports, MCP server logs (gitignored)
```

## Life of a turn

The whole product is one loop. Reading it end to end explains 80% of the repo:

1. **You type a line** — `app/repl.py` (or `app/headless.py` for `saturn -p`). Slash commands
   short-circuit into `commands/`; everything else becomes a turn. `app/session.py` compacts
   old history and resets per-turn state; `core/mentions.py` expands `@file` attachments.
2. **The graph runs** — `app/turn.py::run_turn` streams the compiled graph that
   `app/graph.py::build_agent` assembled from `nodes/`:
   - `nodes/ground.py` builds `state["context"]` in two halves: `context_stable` (workspace
     instructions from SATURDAY.md, document/workspace manifests, the always-loaded memory
     layers — `stores/memory_registry`: user facts + open commitments + the memo digest) and
     `context_dynamic` (agent/entities/negative facts matched to this request, a recap of
     recent Q&A, attachments). Every node sends the stable half as its own message right after
     its system prompt, and `core/prime.py` re-sends exactly that prefix between turns so the
     daemon's prompt cache resumes there (see `core/serving.py`, "the prefix cache").
   - `nodes/quick.py` is where most turns go next (2026-09-08): `route_after_ground` runs the
     request-side complexity check (`core/complexity.py` — a regex, zero tokens) and a chat
     question or a single lookup skips the planner and the judge. One grammar-bound router call
     picks "answer" (straight to synthesize) or ONE read-only tool (`core/plan_context.QUICK_TOOLS`);
     the call rides the same approval → tools → update_plan nodes below and its step lands on
     the same plan, then update_plan returns to quick (`state["route"]`). A tool outside the set,
     an error, or the three-call budget hands the turn to the engine with the observations kept.
   - `nodes/plan.py` drafts the step list via the hardened structured-output path
     (`core/structured.py`). **The plan is the data bus**: each step is a plain dict that will
     carry its own `result`; the first step with `result: None` is the execution pointer.
   - `nodes/plan_gate.py` is the plan-review checkpoint at every step boundary — a
     pass-through unless the user pressed Esc (pause/steer) or `/plan review` is on.
   - `nodes/execute.py` runs ONE step per pass against a curated context
     (`core/plan_context.py`) — a reasoning step answers in text; a write step first faces the
     semantic write gate; a tool step generates its arguments under the planned tool's own
     JSON schema as a response grammar — never a native tool bind, which would render the
     schema into the system prompt (argument recovery in `core/tool_args.py`).
   - `nodes/approval.py` is the human gate: `trust/policy.py` decides whether the call is
     auto-approved (risk tier, /policy allow prefixes) or must interrupt and ask you.
   - `nodes/tools.py` executes the call, clamps the observation, attributes egress
     (`trust/egress.py`), and fences injection-suspicious content (`trust/quarantine.py`).
   - `nodes/update_plan.py` mechanically records the observation onto the current step and
     derives its status (done / skipped / blocked / error).
   - `nodes/rectify.py` reflects after EVERY step — deterministic short-circuits first
     (guarded outcome → cancel the rest; no-call guard → a tool that fails to generate a call
     twice ends the run; unresolved reference; dead-end retry), an LLM verdict
     last. If the plan must change, `nodes/replan.py` redrafts the remaining steps.
   - `nodes/synthesize.py` streams the final answer from the plan's recorded outcomes +
     numbered tool results, disclosing incidents and citing sources.
3. **The answer renders** — `tui/ui/response.py` streams tokens, then closes with the trust
   receipt (`trust/receipt.py`) and trust-colored sources (`trust/glassbox.py`). The trace of
   every node/tool landed in `stores/trace.py`'s SQLite as it happened (`/trace` replays it).

## Package by package

### `app/` — the application shell
| File | What it does |
|---|---|
| `cli.py` | The strict argparse surface (`-p`, `--json`, `--export`, `--replay`, `--yolo`) + piped-stdin capture. Unknown flags exit 2, never fall through to the TUI. |
| `graph.py` | `build_agent()`: wires `nodes/` into the compiled LangGraph with the SqliteSaver checkpointer. The only place graph assembly happens. |
| `turn.py` | `run_turn()`: streams one turn (node updates + answer tokens), resolves interrupts through the caller's approver, surfaces trace degradation. |
| `session.py` | Per-turn state shape + fresh-turn reset + the two history compactions (mechanical every turn; LLM summary past the threshold). |
| `startup.py` | Shared startup: knowledge-base sync + graph build, one-line ingest warnings, attachment admission warnings. |
| `headless.py` | The `-p` path: one query → stdout; gated calls denied by default; `--json` / `--export` contracts. |
| `repl.py` | The interactive session: splash, banner, posture line, health checks, first-run setup, the prompt loop, autosave. |

### `core/` — the engine room
| File | What it does |
|---|---|
| `state.py` | `AgentState` + the plan-step vocabulary. `current_step` (first step with `result is None`) is THE execution pointer; `gate_events` is the one non-recomputable record (human decisions). |
| `llms.py` | `get_model(role)` — the five-role model factory (planner / tool_caller / synthesizer / utility / judge) over Ollama; locality boundary wrapping for a remote `OLLAMA_HOST`; startup health check. Cloud providers are shelved (refuse actionably). |
| `messages.py` | Every system prompt, in one place: the planner prompt (built per call from the live registry), execute/rectify/write-gate/synthesize prompts. |
| `structured.py` | The hardened structured-output layer: flat JSON schemas, salvage parsing, temp-escalating retries, tool-name normalization (`TOOL_SYNONYMS`), `to_steps` minting the plan dicts. Exists because small local models emit near-miss output. |
| `plan_context.py` | Curated context builders — the engine's LLM calls see request + grounding + earlier step results, never raw message history. Prefix-stable by construction: results are capped in landing order and never re-truncated; `exec_parts` lays the step context out as message-boundary parts. |
| `prime.py` | The idle prefix primes: between turns (and once after the weights load) each node lineage's `[system][stable grounding]` prefix is re-sent with one predicted token so the next turn's calls resume from that checkpoint. Off under tests and `runtime.prime: false`. |
| `tool_args.py` | Tool-argument recovery: alias coercion onto real schemas, text-format call parsing (small-model tolerance). |
| `plan_ops.py` | The plan-review seam: pure plan-editor functions + the `PauseController` that `plan_gate` consults at each boundary. |
| `compaction.py` | The heavier LLM compaction (automatic past threshold) folding old turns into a summary message. |
| `memory_review.py` | Session-end learning, gated: collects memory candidates from each turn (steer notes, vetoes, gate denials, unfinished steps) and from compaction summaries into a pending queue, optionally asks the utility model for proposals, and runs the accept-each review screen (`/memory review`, `/quit`). Never writes without a y. |
| `mentions.py` | `@file` expansion into clamped attachment blocks; drag-and-drop path detection. |

### `nodes/` — the graph, one file per node
`ground` → `plan` → `plan_gate` → `execute` → `approval` → `tools` → `update_plan` →
`rectify` → (`replan` | back to `plan_gate` | `synthesize`). Routing helpers live beside
their node (`route_after_execute` in `execute.py`, etc.). See "Life of a turn" above for what
each does; `CLAUDE.md`'s Architecture section documents every branch. Note: `nodes/tools.py`
is the *tool-execution node*, not the `tools/` package (see the name-collision table below).

### `tools/` — capabilities behind the gate
| File | What it does |
|---|---|
| `toolspec.py` | `@register_tool(risk[, retrieval])` — risk tier declared at definition, timing wrapper. Unknown risk fails closed to `destructive`. |
| `registry.py` | Imports the tool modules (which registers them), exposes the live registry + risk views, connects MCP, applies persisted `/policy risk` overrides. |
| `mcp_client.py` | MCP client: stdio/HTTP/SSE servers from config.yaml, remote tools registered as `mcp_<server>_<tool>` (never trusting self-declared tiers), redaction parity, one background asyncio bridge. |
| `calculator.py` | `calculate` (whitelisted AST evaluator — never `eval`) + `current_time` (clock grounding). |
| `web.py` | `web_search` (keyless DuckDuckGo — API-less by design since 2026-07-06), `web_extract` (local trafilatura), `http_request` (the universal REST integration — always gated, request shown in full). |
| `files.py` | Workspace-sandboxed file tools: read/write/edit/list/search/find. Mutating tools snapshot first for `/undo`. |
| `knowledge.py` | `search_knowledge_base` (RAG) + `remember`/`recall` (the layered memory; `remember` takes a layer and a `replaces=#id`) + `recall_runs` (FTS5 search over past runs). |
| `shell.py` | `run_shell` — always `destructive` (the human approving the exact command is the boundary), bounded foreground runs only. |
| `interaction.py` | `ask_user` — pauses the running graph via `interrupt()` to ask the human ONE question; the typed answer resumes as the observation. `read_only` (asking never gates); degrades honestly headless. |
| `notify.py` | `schedule_notification` — a one-shot desktop reminder handed to the OS scheduler via `notify/` (launchd + osascript on macOS; other platforms refuse honestly). `side_effecting`; not egress. Human side: `/notify`. |
| `applescript.py` | The one seam for native macOS app tools: `run(script, app=)` opens the target app hidden then runs `osascript`, translating "not macOS" / Automation denied / not running / timeout into model-readable errors; `quote` + RS/US `records` so user text never splits a field. Not egress; imports nothing project-side. |
| `notes.py` | Apple Notes: `search_notes` / `read_note` (`read_only`, **untrusted** — shared or pasted content is scanned like a web page) + `create_note` (`side_effecting`). |
| `mail.py` | Apple Mail: `list_mail` / `search_mail` / `read_mail` (`read_only`, **untrusted** — email is the canonical injection vector) + `draft_mail` (`side_effecting`, opens a visible UNSENT draft; the human is the send button, so it is not egress). No `send_mail` by decision — see `docs/superpowers/specs/2026-09-06-macos-apps.md`. |
| `calendar.py` | Apple Calendar: `list_calendar_events` (`read_only`, **untrusted** — invitations are someone else's text) + `create_calendar_event` (`side_effecting`). AppleScript, not EventKit: a terminal-launched Python only gets EventKit access when the terminal app carries Apple's usage key. Slow (6–15s for a window across all calendars, under a second narrowed to one), so it takes calendar names. |

### `notify/` — the OS-scheduled side
| File | What it does |
|---|---|
| `__init__.py` | `Notification`, the `Backend` protocol, `backend()` by platform (macOS or the honest `Unsupported`), `parse_when` (the tool's time grammar). |
| `macos.py` | One LaunchAgent per notification, shown by `osascript`; the job deletes itself after firing. `_run` / `agents_dir` / `_uid` are the test seams. |
| `menubar.py` | The menu bar item's Cocoa-free half: its login LaunchAgent (`com.saturn.menubar`), the agent pidfile, `menu_model()` (plain rows), `quit_all()` (stop agent · cancel all · unregister). Fully tested offline. |
| `menubar_app.py` | The AppKit half (`python -m notify.menubar_app`): draws the ringed-planet template icon and renders the model. Needs pyobjc (macOS-only dep). |

### `trust/` — the product's namesake
| File | What it does |
|---|---|
| `policy.py` | THE gate policy object. `approves(name, risk, args)` is the single question the approval node asks; `/policy risk`·`allow`·`open` and `--yolo` are all views of it. Durable state in `database/permissions.json`. |
| `egress.py` | The network chokepoint: in-memory egress ledger (every exit calls `check` then `record`), the air-gap gate, and the inference-locality classifier (`ollama_is_local`). |
| `redaction.py` | Secret stripping/warning at the cloud boundary (key patterns, JWTs, private keys); `scan_args` backs the gate's secret warning. |
| `quarantine.py` | Prompt-injection quarantine: scan untrusted observations, fence instruction-shaped content as data, escalate the next tool batch to the gate. Also screens corpus/attachment admission. |
| `receipt.py` | The ambient surfaces: per-answer trust receipt spans, the session posture line, one-time discovery hints. |
| `glassbox.py` | Answer-level provenance (surfaced as `/trace answer`): per cited source — origin, trust, injection flag — live after each answer and reconstructed from recorded runs. |

### `commands/` — the slash-command layer
`_framework.py` (dispatcher + `@command` registry), `_session.py` (autosave/session store),
`_utils.py` (shared grammar: removal/list verbs, `--save` parsing, toggle status). Themed
modules: `conversation.py` (/clear /resume), `knowledge.py` (/docs
/memory /init /undo), `runtime.py` (/tools /models /mcp), `system.py` (/help /quit
/update), `config.py` (/config, incl. the `context` subview — the folded-in /context),
`plan.py` (/plan), `policy.py` (/policy — the legacy /risk /allow /autoapprove spellings were
cut 2026-07-06 and print pointers), `privacy.py` (/privacy), `trace.py` (/trace — incl. the
`answer` + `source` provenance subviews, the folded-in /glass and /source — + export/replay
engine; the three folds landed 2026-07-07 and print _RENAMED pointers). Convention: one file
owns every view of a feature.

### `stores/` — data + persistence
`rag.py` (corpus sync + vector store), `document_registry.py` (workspace/doc manifests),
`memory_registry.py` (the layered memory file: six layers, per-fact metadata token, selection
under a cap), `snapshots.py` (pre-write snapshots for /undo), `trace.py` (the run/event/LLM-call
trace DB behind /trace and exports, plus the `runs_fts` index behind `recall_runs` / `/trace
search` and the current-run seam `remember` stamps provenance from).

### `tui/` — presentation only
`typeahead.py` (the in-turn console reader: type-ahead queue, Esc steer/pause),
`system_monitor.py` (CPU/RAM/GPU for the status bar), and `ui/` split by screen concern,
re-exported flat (`from tui import ui`): `_base` (console plumbing), `statusbar`, `art`
(the frozen Saturn splash), `prompt` (prompt_toolkit line editor), `trace` (the live rail),
`plan` (plan panel + review editor), `approval` (the gate UI), `response` (streamed answer +
receipt), `glass` (the /trace answer provenance renderer), `readouts`, `listing` (the shared table/section
vocabulary every listing command renders through).

## Same name, different file

The convention is "each package names its module after the feature", which produces
deliberate name reuse. When you're jumping by filename, disambiguate here:

| Name | Which one? |
|---|---|
| `tools/` vs `nodes/tools.py` | The package holds tool *implementations*; the node *executes* the calls the model makes. |
| `trace` ×3 | `stores/trace.py` records runs to SQLite · `tui/ui/trace.py` renders the live rail · `commands/trace.py` is the `/trace` drill-down + export/replay. |
| `plan` ×3 | `nodes/plan.py` drafts the plan · `tui/ui/plan.py` renders it · `commands/plan.py` is `/plan`. |
| `approval` ×2 | `nodes/approval.py` decides + interrupts · `tui/ui/approval.py` renders the gate prompt. |
| `config` ×2 | root `config.py` loads/persists config.yaml · `commands/config.py` is `/config`. |
| `policy`/`privacy` | `trust/policy.py`/`trust/egress.py` are the mechanisms · `commands/policy.py`/`commands/privacy.py` are their front doors. |
| `glass` | `trust/glassbox.py` assembles provenance · `tui/ui/glass.py` renders it. |
| `knowledge` ×2 | `tools/knowledge.py` = the RAG/memory tools · `commands/knowledge.py` = /docs /memory /init /undo. |

## Suggested reading order

1. **`config.yaml` + `config.py`** — the knob surface; everything else reads it.
2. **`agent.py` → `app/graph.py` → `app/turn.py`** — the skeleton: what runs, in what order.
3. **`core/state.py`** — the state shape; the plan-as-data-bus idea lives here.
4. **`nodes/` in graph order** — ground, plan, plan_gate, execute, approval, tools,
   update_plan, rectify, replan, synthesize. This is the heart; take it slowly at
   `execute.py` and `rectify.py` (branch order is load-bearing — see CLAUDE.md gotcha #8).
5. **`core/structured.py` + `core/plan_context.py` + `core/tool_args.py`** — the small-model
   hardening the nodes lean on.
6. **`trust/policy.py` → `nodes/approval.py` → `tui/ui/approval.py`** — the gate, end to end.
7. **`trust/egress.py`, `quarantine.py`, `receipt.py`, `glassbox.py`** — the rest of the
   trust stack.
8. **`app/repl.py` + `commands/_framework.py`** — the interactive shell around it all.
9. Everything else (`tools/`, `stores/`, `tui/`) as reference when a node touches it.

## Where the rules live

- **Design rules & gotchas:** `CLAUDE.md` (bottom sections) — invariants like "the plan step's
  `result is None` is the execution pointer" and rectify's branch ordering.
- **Feature log / changelog:** `documentation.md` (untracked, repo root).
- **Tests as documentation:** `tests/test_engine.py` (the whole plan/execute engine),
  `tests/test_policy.py` (the gate), `tests/test_quarantine.py` (injection defense) — each
  test file names the surface it pins.
