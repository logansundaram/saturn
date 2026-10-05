# Saturn — Architecture & Code Map

A reading guide for going through the codebase by hand. `CLAUDE.md` is the dense dev
reference (every mechanism, every historical decision); this file is the *map*: what lives
where, why it's grouped that way, and the order that makes the code easiest to absorb.

Saturn (Saturday.ai) is a local-first, transparent terminal agent. The product thesis is the
**trust stack**: every action is visible (trace), every risky action asks a human (gate),
every byte that leaves the machine is accounted for (egress ledger), and every run replays
(/trace export). The engine is **one ReAct loop** (v2, 2026-09-27): the
model makes one native tool-calling call per pass, every call faces the gate, and its first
message without tool calls is the answer.

## The 30-second map

```
agent.py            entry point — parses the CLI, routes into app/ (thin; re-exports for tests)
benchmark.py        the graded trust benchmark; --loop is the loop benchmark (dev-only)
config.py/.yaml     the single source of truth for model bindings, paths, and runtime knobs
diag.py             diagnostic logging to logging/diag.log (never print() — TUI-safe)
textutil.py         leaf text helpers (truncation, head+tail clamping, byte formatting)
env_keys.py         the .env reader behind MCP's ${VAR} expansion

app/        the application shell: CLI, graph assembly, turn driver, headless + REPL loops
core/       the engine room: state, model factory, prompts, invoke options, the pause latch
nodes/      the graph nodes, one per file (ground → agent → approval → tools → agent …)
tools/      the tool implementations + registry + MCP client (risk tiers declared at definition)
notify/     scheduled desktop notifications: the platform seam, the macOS launchd/osascript backend, the menu bar item
trust/      the trust stack: gate policy, egress ledger, secret scan, quarantine, receipt
commands/   the slash-command layer (/help themes, one module each)
stores/     data + persistence: RAG corpus + its manifest, memory, snapshots, trace DB
tui/        presentation: the rich-based terminal UI and the type-ahead reader

tests/      offline pytest suite (no LLM, no network — conftest gives every test a throwaway HOME and SATURN_HOME; `isolated_paths` redirects the configured paths)
docs/       every document but the root three (README.md here is the index); superpowers/ holds the specs and plans
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
   - `nodes/ground.py` builds `state["context"]` in two halves: `context_stable` (the
     working folder — the launch folder plus `/add-dir` folders —, the standing instructions from
     ~/.saturn/SATURN.md and the workspace SATURN.md, the knowledge-base manifest, the always-loaded memory
     layers — `stores/memory_registry`: user facts + open commitments + the memo digest) and
     `context_dynamic` (the `### Now` date line, agent/entities/negative facts matched to this
     request, a skill typed as `/<name>`, attachments).
     The agent sends the stable half as its own message right after its system prompt, and
     `core/prime.py` re-sends exactly that prefix — through the same bound model, so the tool
     schemas the chat template renders are inside it — between turns so the daemon's prompt
     cache resumes there (see `docs/OPTIMIZATIONS.md`, "the prefix cache").
   - `nodes/agent.py` makes ONE native tool-calling call per pass (`bind_tools` over the
     registry, streamed; whether the pass thinks is `core/think.py`'s decision from the kind of step —
     `runtime.think` fast | auto | deep, at most two calls per pass because of thinking) over `[system][stable][history…][dynamic + request][turn…]`.
     Around the call, deterministic checks in a fixed order: a pause (Esc) `interrupt()`s for
     the pause prompt (continue / steer / abort); a steer (Esc + text) lands as a
     `STEER_PREFIX` message, drained only past the pause so a resumed interrupt cannot lose it;
     from pass `runtime.max_iterations` on no call runs (the pass stays bound so the cached
     prefix holds; an emitted call is answered with the budget refusal, and a model that calls
     again is rerun once with tools unbound); each emitted call passes hygiene (unknown tool, missing or malformed
     arguments via `core/tool_args`, arguments that belong to another tool, a number or address that appears in nothing the user
     typed and no tool result (`quarantine.handle_hold` — `send_message.to`, `read_messages.contact`),
     a group chat ref no tool returned (`quarantine.chat_hold` — `chat=`), a messaging call naming both a
     person and a group or neither (`tools/messages.route_target`, which moves a target in the wrong slot),
     a repeat of a call the user declined this turn, a third
     identical call with nothing changed in between — each answered with an error ToolMessage back to the model;
     `ask_user` runs alone — its siblings in the batch are answered with "ask first"). A message
     without tool calls is the answer: the Sources receipt and the incidents note (declined /
     blocked / failed calls, read off the ToolMessages' `saturn_status` stamp; a call's last
     outcome decides, so one that failed and then ran is not listed) are appended to the
     recorded message. A malformed reply is retried once, and what it streamed is retracted
     first (`nodes.agent.RETRACT` on LangGraph's custom stream → `run_turn`'s `on_retract`).
   - `nodes/approval.py` is the human gate: `trust/policy.py` decides whether each call the
     agent did not answer itself is auto-approved (risk tier, /policy allow prefixes) or must
     interrupt and ask you. A fully-rejected batch routes back to the agent.
   - `nodes/tools.py` executes the calls, stamps each `done` / `error` / `blocked` (a tool
     reports failure by raising `ToolError`, so an error is never read as done), makes terminal controls visible (`textutil.visible_controls`), clamps the observation, attributes egress
     (`trust/egress.py`), fences injection-suspicious content (`trust/quarantine.py`), and
     maps a `plan` call (`tools/planning.py` — the model's checklist) onto `state["plan"]`.
3. **The answer renders** — `tui/ui/response.py` streamed the agent's answer tokens as they
   generated (a preamble before a tool call is discarded and shown as the rail's agent leaf
   instead); the loop then closes with the trust
   receipt (`trust/receipt.py`). The trace of
   every node/tool landed in `stores/trace.py`'s SQLite as it happened (`/trace` replays it).

## Package by package

### `app/` — the application shell
| File | What it does |
|---|---|
| `cli.py` | The strict argparse surface (`-p`, `-q`, `--json`, `--export`, `--replay`, `--yolo`, `--version`) + piped-stdin capture. Unknown flags exit 2, never fall through to the TUI. |
| `graph.py` | `build_agent()`: wires `nodes/` into the compiled LangGraph with the SqliteSaver checkpointer. The only place graph assembly happens. |
| `turn.py` | `run_turn()`: streams one turn (node updates + answer tokens + the agent's retract signal), resolves interrupts through the caller's approver, surfaces trace degradation; `open_run` / `close_run` are the run lifecycle the REPL and headless share. |
| `session.py` | Per-turn state shape + fresh-turn reset + the two history compactions (mechanical every turn; LLM summary past the threshold). |
| `startup.py` | Shared startup: knowledge-base sync + graph build, one-line ingest warnings, attachment admission warnings. |
| `headless.py` | The `-p` / `-q` path: one query → stdout; gated calls denied by default; `--json` / `--export` contracts; `-q` is the same turn rendered pipe-friendly and always exported. |
| `bang.py` | `!<command>` at the prompt: the user's own shell command (no gate, no trace row), its output attached to their next message under the attachment clamp and admission warning. |
| `repl.py` | The interactive session: splash, banner, posture line, health checks, first-run setup, the prompt loop, autosave. |

### `core/` — the engine room
| File | What it does |
|---|---|
| `state.py` | `AgentState`, the turn-boundary helpers (`is_turn_start`, `turn_start`, `this_turn`) + the step-dict vocabulary of the model's checklist. `current_step` (first item with `result is None`) is the gate's step context; `gate_events` is the one non-recomputable record (human decisions); `is_turn_start` is THE turn-boundary predicate; `grounding_parts` splits the grounding into its stable / per-turn halves. |
| `llms.py` | `get_model()` — the model factory (one chat model per tier, shared by the agent and the background calls) over Ollama; `invoke_kwargs`, THE builder of the per-task decoding options every call sends (num_ctx, num_predict, think) — the agent, compaction, the memory review and `/init` each pass a task, so thinking is explicitly off and output bounded; locality boundary wrapping for a remote `OLLAMA_HOST`; startup health check; `list_local_models` / `model_capabilities`, what `/models` asks the daemon (the capability lookup only when the daemon is local). Ollama is the only backend (cloud providers were cut 2026-09-27; a leftover `{provider, model}` tier is refused by `config.chat_model` with what to write instead). |
| `messages.py` | Every system prompt, in one place: `agent_sys_msg()` (the loop's one prompt — no tool catalog, the tools ride the native bind) plus the compaction, memory-review and /init prompts. |
| `doctext.py` | Text out of PDF / .docx / .xlsx (`extract`) for `read_file` and `@file` attachments, and the PDF / Word loaders the knowledge base shares. A leaf; the format libraries load lazily. |
| `sources.py` | `build_sources` — the answer's source numbering, shared by the Sources footer and `/trace source`. |
| `pause.py` | The `PauseController`: the Esc pause / steer latch the agent node consults at the top of every pass. |
| `think.py` | Which passes think: `step_kind` (what the pass is reacting to) and `decide(level, kind)` — `runtime.think` fast / auto / deep, where `auto` is one rule: think before a pass acts (the pass is drafted think-off and rethought only if it calls a tool), and after an error or a steer. Plus the per-pass record (`entry`, `describe`) the rail, `/think`, `/trace why` and the loop benchmark read. Pure: no model call, never the request text. |
| `prime.py` | The idle prefix prime: between turns (and once after the weights load) the agent's `[system][stable grounding]` prefix is re-sent through the bound model with one predicted token so the next turn's call resumes from that checkpoint. Off under tests and `runtime.prime: false`. |
| `tool_args.py` | Tool-argument recovery: alias coercion onto real schemas (required and optional arguments), the foreign-arguments check (`tool_for_args`: `recall(fact=…)` is `remember`'s call), and the schema hint the agent sends back on a rejected call (small-model tolerance). |
| `compaction.py` | The heavier LLM compaction (automatic past threshold) folding old turns into a summary message. |
| `memory_review.py` | Session-end learning, gated: collects memory candidates from each turn (steer notes, gate denials) and from compaction summaries into a pending queue, optionally asks the model for proposals, and runs the accept-each review screen (`/memory review`, `/quit`). Never writes without a y (facts the user typed in conversation take the auto-learn path instead — core/auto_memory). Its model pass reads `own_words`: what the user typed and what Saturn answered, never a tool result or a summary. |
| `auto_memory.py` | Auto-learn: `why_not(call, state)` — the deterministic check that lets a `remember` skip the gate (its words typed by hand — not pasted, recalled from history or typed ahead — and stated by the user as whole clauses of one sentence, nothing left out, its negation and its tense where they put them, no clause left behind that qualifies, takes back or reports it, a list item only with the clause it hangs off unless that clause is the user's own, nothing external ever in the conversation, no `remember` declined earlier in the turn, a `replaces=` that drops only words the user named); the approval node decides and hands the ids to the tools node in `state["user_stated"]`; `rule_layer` sends a standing rule to the always-loaded `user` layer; `similar` names the stored facts a new one may contradict. |
| `provenance.py` | `of(state)` — what the user typed, what else entered the conversation, and whether anything came from outside the trust boundary — ever, as far as anyone knows (`untrusted`, with `state["outside_seen"]` — a restored session's `OUTSIDE_UNKNOWN` counts: auto-learn and the memory review) or where this session saw it enter, now or in an earlier turn (`entered`: the URL hold and the skill note); `by_hand` is what was typed on a line known to be typed by hand (no paste, not recalled from history, not typed ahead). The one reading the gate's holds, auto-learn and the memory review share. |
| `hooks.py` | The user's `~/.saturn/hooks.yaml`: shell commands on turn-start / turn-end (`app/turn.py`) and before- / after-write (`tools/files.py`); a before-write non-zero exit blocks the write; `problems()` feeds the startup warning. |
| `skills.py` | The user's skills (`~/.saturn/skills`, `<workspace>/.saturn/skills`): discovery, parsing, `problems()` for the startup warning and `/skills`, `invocation()` for a typed `/<name>`, `block()` for the grounding. Its folders (and `linked_targets()`, where a symlinked skill really lives) are control folders for the file tools. The pure half of a drafted skill is here too: `render`, `draft_problem`, `steps_text`, `draft_name`. |
| `mentions.py` | `@file` expansion into clamped attachment blocks; drag-and-drop path detection; `@clipboard` attaches the clipboard the same way (read only when the user types it — there is no clipboard tool). |
| `model_family.py` | The qwen size ladder (`SIZE_LADDER`, `EMBEDDER_LADDER`): the recommended tag per size class — a recommendation, not a gate. A stdlib-only leaf `config.py` imports. |
| `hardware.py` | The hardware probe and the fit / speed arithmetic behind `/models` and the first launch: `probe()` reads the machine, `recommend()` is a pure function of the profile; `live()` is the GPU / memory sample (ioreg + vm_stat, no sudo) the status bar's sampler thread takes every 2 s. Leaf (stdlib + `model_family`). |
| `workspace.py` | Where Saturn works: the launch folder, `/add-dir` folders, the one containment check (`resolve`), `same` (file identity — a case-only spelling on macOS is the same file), and the pruned walk. |

### `nodes/` — the graph, one file per node
`ground` → `agent` → `approval` → `tools` → `agent` … → END. Routing helpers live beside
their node (`route_after_agent` in `agent.py`; `approval` routes through `Command(goto=…)`).
See "Life of a turn" above for what each does; `CLAUDE.md`'s Architecture section documents
every check in the agent node. Note: `nodes/tools.py` is the *tool-execution node*, not the
`tools/` package (see the name-collision table below).

### `tools/` — capabilities behind the gate
| File | What it does |
|---|---|
| `toolspec.py` | `@register_tool(risk[, retrieval, untrusted])` — risk tier declared at definition, timing wrapper. Unknown risk fails closed to `destructive`. `ToolError`: what a tool RAISES when the call did not do its job (the tools node stamps it `error`; never return a failure as a string). |
| `registry.py` | Imports the tool modules (which registers them), exposes the live registry + risk views, connects MCP, applies persisted `/policy risk` overrides. Owns the toolkit switch behind `/tools`: `tool` / `tools_by_name` are the BOUND tools (toolkits that are on, recomputed in place by `apply_toolkits`), `all_tools` / `all_by_name` every registered one; the toolkit table itself is `toolspec.TOOLKITS`. |
| `mcp_client.py` | MCP client: stdio/HTTP/SSE servers from config.yaml, remote tools registered as `mcp_<server>_<tool>` (never trusting self-declared tiers), remote calls on the egress ledger, one background asyncio bridge. |
| `calculator.py` | `calculate` (whitelisted AST evaluator — never `eval`) + `current_time` (clock grounding). |
| `web.py` | `web_search` (keyless DuckDuckGo — API-less by design since 2026-07-06), `web_extract` (httpx fetch + local trafilatura extraction; redirects followed one hop at a time, each new host air-gap checked and recorded before it is contacted). `http_request` was cut 2026-07-16 — MCP is the integration surface. |
| `files.py` | Workspace-sandboxed file tools: read/write/edit/move/delete/list/search/find. `search_files` asks macOS's Spotlight index (`mdfind`) for a plain phrase — candidates only: each is re-matched by the regex and passes the walk's containment and pruning rules — which covers what the direct walk cannot reach in 2 s and finds text inside PDF / Word / Excel files. Mutating tools snapshot first for `/undo` (a move records itself, no byte copy; a delete is the same move into the user's Trash, never an unlink); Saturn's control files (config.yaml, permissions.json, hooks.yaml, the memory file and its pending-review queue, the two SATURN.md files — `_control_files`) and everything inside the skills folders (`_control_dirs`) are refused by file identity, so a case-only spelling is refused too, as is moving a folder that holds one. |
| `knowledge.py` | `search_knowledge_base` (RAG) + `remember`/`recall` (the layered memory; `remember` takes a layer, a `replaces=#id` and a `sensitivity`) |
| `shell.py` | `run_shell` — always `destructive` (the human approving the exact command is the boundary), bounded foreground runs only; stdin is `/dev/null` (the Esc watcher owns the terminal); a non-zero exit or timeout raises `ToolError`. |
| `interaction.py` | `ask_user` — pauses the running graph via `interrupt()` to ask the human ONE question; the typed answer resumes as the observation. `read_only` (asking never gates); degrades honestly headless. |
| `notify.py` | `schedule_notification` — a one-shot desktop reminder handed to the OS scheduler via `notify/` (launchd + osascript on macOS; other platforms refuse honestly). `side_effecting`; not egress. Human side: `/notify`. |
| `applescript.py` | The one seam for native macOS app tools: `run(script, app=)` opens the target app hidden then runs `osascript`, translating "not macOS" / Automation denied / not running / timeout into model-readable errors; `quote` + RS/US `records` so user text never splits a field. Not egress; imports nothing project-side. |
| `notes.py` | Apple Notes: `search_notes` / `read_note` (`read_only`, **untrusted** — shared or pasted content is scanned like a web page) + `create_note` / `append_note` (`side_effecting`; append rewrites the HTML body, so a locked note or one with attachments is refused, and it writes only to a note id or the ONE note with exactly that title — a read may take the first title containing the text, a write never does). |
| `mail.py` | Apple Mail: `list_mail` / `search_mail` / `read_mail` (`read_only`, **untrusted** — email is the canonical injection vector) + `draft_mail` / `reply_mail` (`side_effecting`, open a visible UNSENT draft or threaded reply; the human is the send button, so neither is egress) + `update_mail` (`side_effecting`: read/unread/flag/unflag/move/trash over a list of ids — one gate prompt per cleanup). No `send_mail` by decision — see `docs/superpowers/specs/2026-09-06-macos-apps.md`. |
| `calendar.py` | Apple Calendar: `list_calendar_events` (`read_only`, **untrusted** — invitations are someone else's text) + `create_calendar_event` / `update_calendar_event` (`side_effecting`) + `delete_calendar_event` (`destructive`; returns the whole event). Update and delete address an event by uid + calendar, refuse a recurring event unless `whole_series`, and say when attendees may be notified; a bare clock time on an update is applied to the event's own day inside the script (`atclock`), never parsed as "the next 15:00 from now". AppleScript, not EventKit: a terminal-launched Python only gets EventKit access when the terminal app carries Apple's usage key. Slow (6–15s for a window across all calendars, under a second narrowed to one), so it takes calendar names. |
| `contacts.py` | Apple Contacts: `search_contacts` (`read_only`, **untrusted**) — a name to the addresses, numbers and birthday on the card; an exact name ranks before a substring match, and a miss offers the closest names. Resolves ids once, then fetches at most `limit` people by id (a loop over the `whose` result is ~0.6s a person). |
| `reminders.py` | Apple Reminders: `list_reminders` (`read_only`, **untrusted** — shared lists) + `create_reminder` / `complete_reminder` (`side_effecting`). One `properties of` fetch per list (each Apple event costs ~1s). The dictionary has no recurrence and no location; the tool says so. |
| `shortcuts.py` | The user's Shortcuts through the `shortcuts` CLI: `list_shortcuts` (`read_only`) + `run_shortcut` (`destructive`, **untrusted** output). A shortcut is a process the ledger cannot see inside: each run is recorded `UNTRACKED`, air-gap holds it, and the gate is relaxed per shortcut by name (`/policy shortcut`), never by a blanket always-allow. Two tools, not one per shortcut — the bound schemas are the cached prefix. |
| `messages.py` | `send_message` — an iMessage through Messages; an **egress chokepoint** (`egress.check` → `record`, recipient as the host label), `destructive`, and in `policy.ALWAYS_ASKS`: no tier, open gate, override or always-allow skips the human, headless refuses it even with `--yolo`. `to` must be a number or address, never a name; `chat=` sends to ONE existing group chat instead (a `g…` ref from `find_group_chats`, which lists the app's group chats with their people over AppleScript — no Full Disk Access; one egress event per recipient; the gate lists every member via `describe_group`). `read_messages` (`read_only`, **untrusted**) reads `~/Library/Messages/chat.db` read-only — needs Full Disk Access — and decodes `attributedBody` typedstreams. A contact is matched against the handle and chat tables and selected in SQL; a text filter runs in Python over the newest `_SCAN` rows and the result says when that was not the whole history. |
| `skills.py` | `create_skill` — saves one of the user's skills to `~/.saturn/skills` (never a workspace's). In `policy.ALWAYS_ASKS`, and refuses unless `human_approved()`. `draft(args)` is the one builder of (path, text) the gate renders and the tool writes; `/undo` takes a save back. |
| `desktop.py` | What the user is pointing at: `read_browser_tab` (`read_only`, **untrusted**; Safari's page text is a plain property, read locally with no fetch; Chromium needs *Allow JavaScript from Apple Events* and degrades to URL + title; a closed browser is never launched; with several browsers running, the frontmost by `lsappinfo` window order is read and one without a window passes to the next) and `finder_selection` (`read_only`; out-of-reach paths come back with the `/add-dir` that would allow them). |

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
| `policy.py` | THE gate policy object. `approves(name, risk, args)` is the single question the approval node asks; `/policy risk`·`allow`·`shortcut`·`open` and `--yolo` are all views of it. Two holds sit above the tier: `airgap_holds` (a shell command, a shortcut, an MCP server under air-gap) and `always_asks` (`send_message` — never auto-approved by anything). `run_shortcut` has a per-name allowlist beside the shell prefixes. A shell always-allow prefix covers a command only past the metacharacter screen and the argument-tail screen (interpreters, capability flags, globs and brace expansion, paths outside the workspace — bare or as a flag's value). Durable state in `database/permissions.json`. |
| `egress.py` | The network chokepoint: in-memory egress ledger (every exit calls `check` then `record`), the air-gap gate, and the inference-locality classifier (`ollama_is_local`). `UNTRACKED` marks a run Saturn cannot see inside (`run_shell`, `run_shortcut`, a stdio MCP server); `is_private_host` keeps a model-chosen fetch off local services. |
| `secret_scan.py` | Credential-shaped values (key patterns, JWTs, private keys) as display-safe findings; `scan_args` backs the gate's secret-argument warning. |
| `quarantine.py` | Prompt-injection quarantine: scan untrusted observations (a failed call's text included), fence instruction-shaped content as data, escalate the next batch that can act to the gate. `url_hold` gates a `web_extract` URL the model composed after external content, or a private address the user did not type. `handle_hold` refuses a recipient the model invented — a number or address found in nothing the user typed and no tool result — from the agent's hygiene, before any gate; at the gate the prompt names the contact card the handle came from (`nodes/approval._handle_note`). Also screens corpus/attachment admission. |
| `receipt.py` | The ambient surfaces: per-answer trust receipt spans, the session posture line, one-time discovery hints. |

### `commands/` — the slash-command layer
`_framework.py` (dispatcher + `@command` registry), `_session.py` (autosave/session store),
`_utils.py` (shared grammar: removal/list verbs, `--save` parsing, toggle status). Themed
modules: `conversation.py` (/clear /resume /copy), `knowledge.py` (/docs
/memory /init /undo), `runtime.py` (/tools /models /mcp), `system.py` (/help /quit
/update), `config.py` (/config), `notify.py` (/notify — pending notifications and the menu bar
icon), `workspace_dirs.py` (/add-dir /rm-dir), `think.py` (/think — the think level, what
each kind of pass does, the last turn pass by pass; `/think <request>` is one turn at deep), `policy.py` (/policy — the gate's levers plus the egress
ledger and air-gap, /privacy merged in 2026-09-30), `trace.py` (/trace — incl. the `source`
subview; `context` is an alias of `invoke --full` — + export/replay engine). A cut spelling
prints a `_RENAMED` pointer for one release. Convention: one file owns every view of a
feature.

### `stores/` — data + persistence
`rag.py` (corpus sync + vector store), `document_registry.py` (the knowledge-base manifest),
`memory_registry.py` (the layered memory file: six layers, per-fact metadata token, selection
under a cap), `snapshots.py` (pre-write snapshots for /undo), `trace.py` (the run/event/LLM-call
trace DB behind /trace and exports, plus the current-run seam `remember` stamps provenance
from).

### `tui/` — presentation only
`typeahead.py` (the in-turn console reader: type-ahead queue, Esc steer/pause),
and `ui/` split by screen concern,
re-exported flat (`from tui import ui`): `_base` (console plumbing), `statusbar`, `art`
(the frozen Saturn splash), `prompt` (prompt_toolkit line editor), `trace` (the live rail),
`plan` (the checklist panel), `approval` (the gate UI + the Esc pause prompt lives in `prompt`), `response` (streamed answer +
receipt), `readouts`, `listing` (the shared table/section
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
| `policy` | `trust/policy.py` (the gate) and `trust/egress.py` (ledger, air-gap) are the mechanisms · `commands/policy.py` is their one front door. |
| `knowledge` ×2 | `tools/knowledge.py` = the RAG/memory tools · `commands/knowledge.py` = /docs /memory /init /undo. |
| `notify` ×3 | `notify/` is the OS-scheduler seam · `tools/notify.py` is `schedule_notification` · `commands/notify.py` is `/notify`. |
| `messages` ×2 | `core/messages.py` holds every system prompt · `tools/messages.py` is iMessage (`send_message` / `read_messages` / `find_group_chats`). |

## Suggested reading order

1. **`config.yaml` + `config.py`** — the knob surface; everything else reads it.
2. **`agent.py` → `app/graph.py` → `app/turn.py`** — the skeleton: what runs, in what order.
3. **`core/state.py`** — the state shape.
4. **`nodes/` in graph order** — ground, agent, approval, tools. This is the heart; take it
   slowly at `agent.py` (the check order around the call is load-bearing — the module
   docstring and `tests/test_agent_loop.py` pin it).
5. **`core/tool_args.py` + `core/llms.invoke_kwargs`** — the small-model hardening and the
   per-task decoding options the agent leans on.
6. **`trust/policy.py` → `nodes/approval.py` → `tui/ui/approval.py`** — the gate, end to end.
7. **`trust/egress.py`, `quarantine.py`, `receipt.py`** — the rest of the trust stack.
8. **`app/repl.py` + `commands/_framework.py`** — the interactive shell around it all.
9. Everything else (`tools/`, `stores/`, `tui/`) as reference when a node touches it.

## Where the rules live

- **Design rules & gotchas:** `CLAUDE.md` (Architecture section) — invariants like the agent
  node's check order and "steps are plain dicts".
- **Feature log / changelog:** `CHANGELOG.md`.
- **Tests as documentation:** `tests/test_agent_loop.py` (the whole loop),
  `tests/test_policy.py` (the gate), `tests/test_quarantine.py` (injection defense) — each
  test file names the surface it pins.
