# Saturn — Architecture & Code Map

A reading guide for going through the codebase by hand. `CLAUDE.md` is the dense dev
reference (every mechanism, every historical decision); this file is the *map*: what lives
where, why it's grouped that way, and the order that makes the code easiest to absorb.

Saturn (Saturday.ai) is a local-first, transparent terminal agent. The product thesis is the
**trust stack**: every action is visible (trace), every risky action asks a human (gate),
every byte that leaves the machine is accounted for (egress ledger), and every answer can
show its provenance (/trace answer). The engine is **one ReAct loop** (v2, 2026-09-27): the
model makes one native tool-calling call per pass, every call faces the gate, and its first
message without tool calls is the answer.

## The 30-second map

```
agent.py            entry point — parses the CLI, routes into app/ (thin; re-exports for tests)
benchmark.py        the graded trust benchmark + capability regression suites (dev-only)
config.py/.yaml     the single source of truth for model bindings, paths, and runtime knobs
diag.py             diagnostic logging to logging/diag.log (never print() — TUI-safe)
textutil.py         leaf text helpers (truncation, head+tail clamping, byte formatting)
env_keys.py         .env-backed secret management (the /config key front end)

app/        the application shell: CLI, graph assembly, turn driver, headless + REPL loops
core/       the engine room: state, model factory, prompts, invoke options, the pause latch
nodes/      the graph nodes, one per file (ground → agent → approval → tools → agent …)
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
     `context_dynamic` (agent/entities/negative facts matched to this request, attachments).
     The agent sends the stable half as its own message right after its system prompt, and
     `core/prime.py` re-sends exactly that prefix — through the same bound model, so the tool
     schemas the chat template renders are inside it — between turns so the daemon's prompt
     cache resumes there (see `docs/OPTIMIZATIONS.md`, "the prefix cache").
   - `nodes/agent.py` makes ONE native tool-calling call per pass (`bind_tools` over the
     registry, think off, streamed) over `[system][stable][history…][dynamic + request][turn…]`.
     Around the call, deterministic checks in a fixed order: a steer (Esc + text) lands as a
     `STEER_PREFIX` message; a pause (Esc) `interrupt()`s for the pause prompt (continue /
     steer / abort); past `runtime.max_iterations` the last pass runs with tools unbound and a
     budget note; each emitted call passes hygiene (unknown tool, missing or malformed
     arguments via `core/tool_args`, a repeat of a call the user declined this turn, a third
     identical call — each answered with an error ToolMessage back to the model). A message
     without tool calls is the answer: the Sources receipt and the incidents note (declined /
     blocked / failed rounds, read off the ToolMessages' `saturn_status` stamp) are appended
     to the recorded message.
   - `nodes/approval.py` is the human gate: `trust/policy.py` decides whether each call the
     agent did not answer itself is auto-approved (risk tier, /policy allow prefixes) or must
     interrupt and ask you. A fully-rejected batch routes back to the agent.
   - `nodes/tools.py` executes the calls, clamps the observation, attributes egress
     (`trust/egress.py`), fences injection-suspicious content (`trust/quarantine.py`), and
     maps a `plan` call (`tools/planning.py` — the model's checklist) onto `state["plan"]`.
3. **The answer renders** — `tui/ui/response.py` streamed the agent's answer tokens as they
   generated (a preamble before a tool call is discarded and shown as the rail's agent leaf
   instead); the loop then closes with the trust
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
| `state.py` | `AgentState` + the step-dict vocabulary of the model's checklist. `current_step` (first item with `result is None`) is the gate's step context; `gate_events` is the one non-recomputable record (human decisions); `is_turn_start` is THE turn-boundary predicate. |
| `llms.py` | `get_model(role)` — the five-role model factory (planner / tool_caller / synthesizer / utility / judge) over Ollama; locality boundary wrapping for a remote `OLLAMA_HOST`; startup health check. Cloud providers are shelved (refuse actionably). |
| `messages.py` | Every system prompt, in one place: `agent_sys_msg()` (the loop's one prompt — no tool catalog, the tools ride the native bind) plus the compaction, memory-review and /init prompts. |
| `structured.py` | `_invoke_kwargs` — THE builder of the per-task decoding options every model call sends (num_ctx, num_predict, think) — plus the hardened structured-output call the memory review uses. |
| `context.py` | `grounding_parts` (the stable / per-turn halves of the grounding block) and `clean` (workspace paths collapse in observations). |
| `sources.py` | `build_sources` — the answer's source numbering, shared by the Sources footer, `/trace source` and the Glass Box. |
| `pause.py` | The `PauseController`: the Esc pause / steer latch the agent node consults at the top of every pass. |
| `prime.py` | The idle prefix prime: between turns (and once after the weights load) the agent's `[system][stable grounding]` prefix is re-sent through the bound model with one predicted token so the next turn's call resumes from that checkpoint. Off under tests and `runtime.prime: false`. |
| `tool_args.py` | Tool-argument recovery: alias coercion onto real schemas + the schema hint the agent sends back on a rejected call (small-model tolerance). |
| `compaction.py` | The heavier LLM compaction (automatic past threshold) folding old turns into a summary message. |
| `memory_review.py` | Session-end learning, gated: collects memory candidates from each turn (steer notes, gate denials, failed tool calls) and from compaction summaries into a pending queue, optionally asks the utility model for proposals, and runs the accept-each review screen (`/memory review`, `/quit`). Never writes without a y. |
| `mentions.py` | `@file` expansion into clamped attachment blocks; drag-and-drop path detection. |

### `nodes/` — the graph, one file per node
`ground` → `agent` → `approval` → `tools` → `agent` … → END. Routing helpers live beside
their node (`route_after_agent` in `agent.py`; `approval` routes through `Command(goto=…)`).
See "Life of a turn" above for what each does; `CLAUDE.md`'s Architecture section documents
every check in the agent node. Note: `nodes/tools.py` is the *tool-execution node*, not the
`tools/` package (see the name-collision table below).

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
`policy.py` (/policy — the legacy /risk /allow /autoapprove spellings were
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
`plan` (the checklist panel), `approval` (the gate UI + the Esc pause prompt lives in `prompt`), `response` (streamed answer +
receipt), `glass` (the /trace answer provenance renderer), `readouts`, `listing` (the shared table/section
vocabulary every listing command renders through).

## Same name, different file

The convention is "each package names its module after the feature", which produces
deliberate name reuse. When you're jumping by filename, disambiguate here:

| Name | Which one? |
|---|---|
| `tools/` vs `nodes/tools.py` | The package holds tool *implementations*; the node *executes* the calls the model makes. |
| `trace` ×3 | `stores/trace.py` records runs to SQLite · `tui/ui/trace.py` renders the live rail · `commands/trace.py` is the `/trace` drill-down + export/replay. |
| `plan` ×2 | `tools/planning.py` is the tool the model calls · `tui/ui/plan.py` renders the checklist. |
| `approval` ×2 | `nodes/approval.py` decides + interrupts · `tui/ui/approval.py` renders the gate prompt. |
| `config` ×2 | root `config.py` loads/persists config.yaml · `commands/config.py` is `/config`. |
| `policy`/`privacy` | `trust/policy.py`/`trust/egress.py` are the mechanisms · `commands/policy.py`/`commands/privacy.py` are their front doors. |
| `glass` | `trust/glassbox.py` assembles provenance · `tui/ui/glass.py` renders it. |
| `knowledge` ×2 | `tools/knowledge.py` = the RAG/memory tools · `commands/knowledge.py` = /docs /memory /init /undo. |

## Suggested reading order

1. **`config.yaml` + `config.py`** — the knob surface; everything else reads it.
2. **`agent.py` → `app/graph.py` → `app/turn.py`** — the skeleton: what runs, in what order.
3. **`core/state.py`** — the state shape.
4. **`nodes/` in graph order** — ground, agent, approval, tools. This is the heart; take it
   slowly at `agent.py` (the check order around the call is load-bearing — the module
   docstring and `tests/test_agent_loop.py` pin it).
5. **`core/structured.py` + `core/tool_args.py`** — the small-model hardening and the
   per-task decoding options the agent leans on.
6. **`trust/policy.py` → `nodes/approval.py` → `tui/ui/approval.py`** — the gate, end to end.
7. **`trust/egress.py`, `quarantine.py`, `receipt.py`, `glassbox.py`** — the rest of the
   trust stack.
8. **`app/repl.py` + `commands/_framework.py`** — the interactive shell around it all.
9. Everything else (`tools/`, `stores/`, `tui/`) as reference when a node touches it.

## Where the rules live

- **Design rules & gotchas:** `CLAUDE.md` (Architecture section) — invariants like the agent
  node's check order and "steps are plain dicts".
- **Feature log / changelog:** `CHANGELOG.md`.
- **Tests as documentation:** `tests/test_agent_loop.py` (the whole loop),
  `tests/test_policy.py` (the gate), `tests/test_quarantine.py` (injection defense) — each
  test file names the surface it pins.
