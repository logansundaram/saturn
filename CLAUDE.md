# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to work (since 2026-09-27)

**All new work happens in the `v2` worktree at `/Users/Logan/Documents/saturn-v2` (branch `v2`).**
v2 is a complete overhaul into a simpler agent. The main checkout at `/Users/Logan/Documents/saturn`
(branch `main`) is the running v1 and a read-only reference (`git show main:<path>`); do not edit or
commit there unless the user says so explicitly. The worktree has its own `.venv`.

Saturn is a local-first, terminal-native AI agent (Python 3.11+, LangGraph, Ollama only). The product
thesis is the **trust stack**: every action is traced, every risky action faces a human gate, every byte
that leaves the machine is recorded in an egress ledger, and every run is replayable.
`docs/ARCHITECTURE.md` is the guided code map and reading order; read it before touching `nodes/` or `trust/`.

Note: `CLAUDE.md` is tracked (its `.gitignore` entry was removed 2026-09-02).

## Commands

```bash
# setup (the CI way — a dep missing from pyproject.toml fails here, not in pipx installs)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]                # pyproject.toml is the one dependency list

# run
python agent.py                      # interactive TUI (./saturn.sh prefers the repo .venv)
saturn -p "query"                    # headless: one answer to stdout; gated tools DENIED unless --yolo
saturn -q "query"                    # pipe-friendly one-shot (answer only on stdout, auto-exports)
saturn -p "query" --json --export run.json
saturn --replay logging/exports/run_1.json   # render an exported record, no DB/models needed

# tests — fully offline (no Ollama, no network, no embedder); ~5s for ~1350 tests
python -m pytest tests/ -q
python -m pytest tests/test_agent_loop.py -q             # one file
python -m pytest tests/test_policy.py -q -k prefix       # one test by name

# trust benchmark — needs a running Ollama with the active tier pulled
python benchmark.py [--strict]       # reports to logging/benchmarks/trust_<ts>.json
python benchmark.py --loop           # the loop benchmark: daily requests graded on passes/tool choice/phantoms → loop_<ts>.json

SATURN_DEBUG=1 python agent.py       # echo logging/diag.log lines to stderr
```

There is no linter or formatter configured. CI (`.github/workflows/tests.yml`) runs the suite on
ubuntu + macos × Python 3.11–3.13 and smoke-tests the built wheel in a clean venv.

**Releases:** push a `v*` tag. `pyproject.toml` `version`, `app/__init__.py` `__version__`, and a
`## [x.y.z]` section in `CHANGELOG.md` must all agree (`tests/test_version.py` pins the first pair on
every push). User-visible changes go under `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog format).

Commit messages follow `area: what changed` in lowercase (`gate: …`, `trace: …`, `review: …`).

## Architecture

### Life of a turn

`agent.py` is a thin router into `app/`; `app/graph.py::build_agent` compiles the graph from `nodes/`
(one file per node) with a SqliteSaver checkpointer, and `app/turn.py::run_turn` streams it. Since
2026-09-27 (v2) the engine is ONE ReAct loop — spec:
`docs/superpowers/specs/2026-09-27-v2-react-loop-design.md` (the loop's shape today: `docs/engine.md`):

```
ground → agent ─(no tool calls)─→ END
           ↑          │ tool calls
           └── tools ← approval      (a fully-rejected batch → agent)
```

- `ground` assembles `state["context"]` in two halves (the working folder, `~/.saturn/SATURN.md` then the workspace `SATURN.md`, the knowledge-base manifest, the always-loaded
  memory layers = stable; the `### Now` date line + memory matches + attachments = dynamic). No model call.
- `agent` (`nodes/agent.py`) makes ONE native tool-calling call per pass (`get_model()`,
  `bind_tools(registry)`, streamed; whether the pass THINKS is `core/think.py`'s decision — see
  "Thinking" below). Prompt order is prefix-cache order:
  `[system][stable grounding][history…][dynamic + request][turn messages…]`; the bound tool
  schemas render into the chat template's system section, so the catalog is part of the prefix
  `core/prime.py` caches. The checks around the call are deterministic, in this order, and each
  is pinned by `tests/test_agent_loop.py`: **pause** (Esc → `interrupt({"type": "pause"})` →
  `ui.pause_prompt`: continue / steer / abort) → **steer** (Esc + text → a `STEER_PREFIX`
  HumanMessage; drained only past the pause, since a resumed interrupt re-runs the node) → **cap** (from pass `runtime.max_iterations` on no tool call runs: the pass stays an ordinary
  bound call so the cached prefix holds, a call it emits is answered with `BUDGET_TEXT` and routed
  back for the answer; a model that calls again is rerun once with tools UNBOUND and a budget
  note — a real answer, never a stub) → generate → **hygiene** on each emitted call (unknown tool,
  arguments that belong to another tool via `core/tool_args.tool_for_args`,
  missing arguments via `core/tool_args.coerce_args`, malformed JSON, a messaging call with both
  a person and a group chat, neither, or a name for a recipient (`tools/messages.route_target` — a
  target in the wrong slot is moved, not refused), a recipient or group chat ref the model
  invented (`quarantine.handle_hold` / `chat_hold`), a repeat of a call the user
  DECLINED this turn, a third identical call with nothing changed in between — each answered with
  an error ToolMessage that routes straight back to `agent`, no gate, no model call; `ask_user`
  runs alone, its siblings answered the same way) → **answer** (a message without tool calls IS
  the answer; the Sources receipt and the incidents note are appended to the RECORDED message,
  never the stream). `nodes.agent._generate` is the one model seam tests replace.
- `approval` asks `trust/policy.approves(name, risk, args)` — the ONE gate question — on the
  issuing message's calls MINUS those the agent already answered, and interrupts the graph for the
  human when it says no. Two holds sit beside it, both from `trust/quarantine.py`: the escalation
  after flagged output (the next batch that can ACT — send or change something) and the URL hold
  (`url_hold`: a `web_extract` address the model composed after external content entered the
  conversation, or a private address the user did not type). The prompt says why (`notes`). A
  fully-rejected batch routes back to `agent`; the declined-repeat guard keeps that "no" for the
  rest of the turn.
- `tools` (the node, not the package) executes, clamps the observation, records egress, fences
  injection-suspicious content (`trust/quarantine.py`), and maps a `plan` call onto `state["plan"]`.

There is no synthesize node, no judge, no planner: the model's last message is the answer and it
streams under `── response` as it generates (`app/turn.py` filters LangGraph messages mode to the
`agent` node; a text preamble before a tool call is discarded from the response region by
`_make_on_update` and shown as the rail's agent leaf instead).

### Thinking (`core/think.py`, `commands/think.py`)

Whether a pass reasons before it answers is decided by the harness (Qwen3.5 has no in-model
switch), from the KIND OF STEP the pass is — a pure function of the turn's messages, never the
request text, no model call. `step_kind` (first match wins): `capped` · `recovery` (an `error`
stamp in the latest round; `ASK_ALONE_TEXT` is not evidence) · `steered` · `first` · `wrap-up`
(every completed call was an action — `tools/registry.is_action`, the Sources rule — or nothing
completed) · `information`. `decide(level, kind, policy)`: `runtime.think` is `fast | auto |
deep` (the old `off | adaptive | on` and YAML booleans read through `think.normalise`; anything
else runs as `auto` and warns at startup). `auto` is ONE rule, not a setting: **think before a
pass acts, never before a text answer** (`act`) — every deciding pass (`first`, `information`,
`wrap-up`) is DRAFTED think-off; a text answer stands at no cost, a tool call is retracted and
the pass rethought; `recovery` and `steered` think outright. It was picked by measurement
(2026-10-04: 29.5 of 34 loop tasks on the 4b against 23.5 for the rule before it, 31 against 30
on the 9b, no empty thoughts; `docs/engine.md` item 11). The rule before it (`recover`: think
only after an error) survives as the benchmark's baseline (`benchmark.py --think recover`,
`think.set_policy`) — there is no `runtime.think_policy`.
`nodes/agent._run_pass` holds the one bound: **at most two model calls per pass because of
thinking** — a thought that is empty, past `runtime.think_budget` (`_generate` counts reasoning
chunks and closes the stream) or stopped by Esc (a pending pause; the pause itself stays
pending) is dropped and the pass answers without it; a drafted call stands when its rethink
came to nothing. A model that rejects the think flag never thinks and gets no widened
`num_predict`. One `think.entry` per pass lands in `state["think"]` (never the prompt): the
rail row and leaf, the status bar's `thinking 3s · esc stops thinking` (`thinking` events on
the custom stream → `run_turn(on_thinking=)`), the receipt, `/think`, `/trace why` and the
loop benchmark (`--think`, `--tier`, `--runs`) all read it; `llm_calls.output.think` is what
each call was sent with. `/think <request>` is one turn at `deep`
(`app/session.think_for_line` → `state["think_level"]`). Spec:
`docs/superpowers/specs/2026-10-04-adaptive-thinking-design.md`. `tests/test_think.py` and
`tests/test_think_command.py` pin all of it.

### The plan is the model's checklist (`tools/planning.py`, `core/state.py`)

`plan(steps=[{label, status}])` is a read-only tool the prompt asks for only on multi-step tasks.
`nodes/tools.py` maps a successful call onto `state["plan"]` as the same **plain dicts**
`{step_id, label, status, result}` every reader already renders
(the rail's `show_plan`, the gate's `step` context via `current_step()` = first item with
`result is None`, `/trace why`, replay, the headless `plan` field). It is intent, not record:
the answer's incidents note reads the tool rounds that actually ran (the ToolMessages'
`saturn_status` stamp), never the checklist. `gate_events` is the only non-recomputable record
(human decisions).

When slicing conversation history, use `core.state.is_turn_start` — a mid-turn steer note is a
`HumanMessage` with `STEER_PREFIX` and is NOT a turn boundary; a hand-rolled isinstance check mis-slices.

### Models and config

Each tier binds ONE chat model (`tiers.<t>.model`) — the agent's call and the background calls
(compaction, the memory review, `/init`) share it — plus an `embedder`. Code gets it from
`core/llms.get_model()`; never name a model in graph code. Every call — background ones too —
sends `core/llms.invoke_kwargs(..., task=…)` (a `NUM_PREDICT` entry), so thinking is explicitly
off and output bounded; a bare `.invoke()` gets the model's defaults. It resolves through `active_tier` →
`tiers` in `config.yaml`; a leftover `roles:` block is refused with the one `model:` line to
write instead (the old `SATURDAY_*` / `SATURDAY.md` / `~/.saturday` spellings are no longer read
either, since 2026-09-30). Ollama is the only backend (cloud providers were cut 2026-09-27; a remote
`OLLAMA_HOST` is the one network boundary, wrapped in `core/llms.py`). The qwen3.5/3.6/3.8 size ladder in
`core/model_family.py` is the recommended default per size, not a gate: any Ollama tool-calling
model binds.

`runtime.max_iterations` bounds the passes that may run tools; from that pass on calls are refused
and the turn ends in an answer (one pass later at most).

`config.yaml` is **gitignored user data**, seeded on first run from the tracked template
`config.default.yaml` (or `~/.saturn/config.yaml` for wheel installs — `config.saturn_home`). Change defaults in the
template. `config.persist()` does a surgical single-line YAML edit to preserve comments — don't replace
it with a full dump. `textutil.py` imports nothing project-side, `diag.py` only `textutil`, and
`config.py` only the leaves `diag` and `core/model_family`; all three are safe
leaves; `diag.log()` replaces `print()` in nodes/tools (stdout collides with the rich Live TUI).

### Memory (`stores/memory_registry.py`, `core/memory_review.py`)

One markdown file (`paths.memory`), six `## layer` sections: `user` and `commitments` load every
turn, the last five `memo` entries too, `agent` / `entities` / `negative` (and any other
heading a hand edit or `layer=` introduced) only by token match against the request — all under
`memory.context_cap`, with a trailer naming what didn't load. `sens=` facts are withheld from
both the block and `recall` when inference is not local.
Every bullet ends in a `{#id by=user|inferred run=N used=DATE n=K sens=… due=… src=…}` metadata token;
ids come from the `<!-- next-id -->` high-water mark and are never reused (`replaces=#id`
supersedes; `/memory why <n>` points at `/trace why #run`). Learning is gated: `core/memory_review`
queues candidates from each turn's state and from compaction summaries into
`database/memory/pending_review.json`; `/memory review` (also `/quit`) accepts them one at a
time. Never write a fact without a user action: a gated `remember`; a `remember` whose every
content word the user TYPED in a conversation no external content entered (auto-learn,
`core/auto_memory.why_not` — deterministic, never a model's judgement; stamped `src=said`, noted
after the answer, off headless); `/memory add`; or a review accept. A standing rule ("never…")
lands in `user`, which loads every turn (`auto_memory.rule_layer`). Never write a credential:
`memory_registry.secret_problem` is asked by the two writers of fact text (`add_memory`,
`edit_memory` raise `SecretRefused`), so a new write path inherits it. A write that lands beside a
related fact names it (`auto_memory.similar` — the after-answer note, the gate, `/memory add`);
nothing is retired without `replaces=`. The model reads each fact with its day and `[inferred]`
(`_context_line`), under a header that says what outranks it. The review's model pass reads
`memory_review.own_words` — what the user typed and what Saturn answered, never a tool result or
a summary. Spec: `docs/superpowers/specs/2026-10-04-know-the-user-design.md`. The benchmark's
memory tasks, `tests/test_memory_*.py` and `tests/test_auto_memory.py` pin this.

### Trust stack (`trust/`)

- `policy.py` — one object behind `/policy risk|allow|shortcut|open`, `runtime.auto_approve`, and `--yolo`.
  `/policy` (`commands/policy.py`) is the ONE trust front door: its bare readout, the gate's levers,
  and `egress` / `airgap` over `egress.py` (`/privacy` merged in 2026-09-30).
  Shell prefix matching is token-based and refuses metacharacters; the tail past a granted prefix
  is screened too (`arg_tail_rejects`: interpreters, capability flags, globs and `{}`, paths
  outside the workspace — bare or as a flag's value). `run_shortcut` has its own allowlist, one
  shortcut by exact name (`/policy shortcut`). `ALWAYS_ASKS` (`send_message`, `create_skill`;
  `name -> (what, why)`) sits above every lever: no tier, open gate, risk override or
  always-allow lets a send or a skill save through, and headless refuses them even with `--yolo`; the gate's `a` never drops the tier of a tool in
  `NO_BLANKET_GRANT`. Persisted in `database/permissions.json`.
- `egress.py` — every outbound network op calls `check()` (air-gap) then `record()`. The complete list
  of egress chokepoints is `core/llms.py`, `tools/web.py`, `tools/mcp_client.py`, `tools/messages.py`
  (`send_message`); `tests/test_no_new_egress.py` fails on a network-client import anywhere else (`stores/rag.py` may import `trafilatura` for local extraction; the same test pins that it never fetches), and
  `tests/test_messages.py` pins the one chokepoint that imports none (it sends through `osascript`).
  A new chokepoint is a deliberate edit to that test plus check/record wiring. A remote `OLLAMA_HOST` counts as egress
  (`is_loopback_host` parses the address; never match it as a string). `run_shell`, `run_shortcut` and
  stdio MCP servers are processes the ledger cannot see inside: each run is recorded `UNTRACKED`
  (never a send, never absent), and under air-gap `policy.airgap_holds` keeps them from being
  auto-approved — headless refuses them even with `--yolo`.
- `quarantine.py` — untrusted output (web, MCP, files, shell, corpus — a FAILED call's text too) is
  scanned, fenced as data, and the next batch that can act is escalated to the gate; `url_hold`
  is the exfiltration hold on `web_extract` (above). `handle_hold` is the invented-recipient check
  the agent's hygiene asks for `send_message.to` and `read_messages.contact` (`HANDLE_ARGS`): a
  number or address found in nothing the user typed and no tool result never reaches the gate; for
  one that does, the gate prompt names the contact it belongs to (`nodes/approval._handle_note`).
  `chat_hold` (`CHAT_ARGS`) is the same check for a group chat ref (`chat=`), and the gate names
  every member of the group, resolved from Messages at approval. `core/provenance.of` is the one
  typed / seen / untrusted reading the holds, auto-learn and the memory review share.
- Terminal safety — text from outside Saturn never reaches the terminal as a live escape
  sequence. `textutil.visible_controls` is the one rule (SGR and `ESC[K` removed, CR → LF, every other
  control → its picture `␛`); `nodes/tools.py` applies it to every observation before the clamp
  and quarantine (also `@file`/`@clipboard` and `!cmd` attachments), and the sink applies it
  again: `tui/ui/_base.SafeConsole` (all TUI output), `commands/_framework._print`, headless
  stdout/stderr, `diag.log`. The gate shows bidi and every unprintable character as `⟨U+202E⟩`
  and ESC as `␛` — nothing is removed there (`visible_format_chars`); an answer's Markdown
  links print as `text (address)`, never as terminal hyperlinks. Never print model or tool text with a bare `print()`.

### Tools

Define a tool in its own module under `tools/` with `@register_tool(risk=...)` from `tools/toolspec.py`;
`tools/registry.py` imports the modules to trigger registration — nothing else to edit. A call that
did not do its job RAISES `toolspec.ToolError` (never returns an error string): the tools node
stamps it `error`, which makes the next pass a `recovery` step (core/think) and puts it in the answer's incidents note.
Only a call that completed and gathered something is a source (`tool_results` /
`documents_retrieved`): failed or blocked calls and `side_effecting` tools are not cited.
`toolspec.human_approved()` tells a tool whether a person approved THIS call at the gate
(`remember` stamps `by=user` only then, `by=inferred` otherwise).
Unknown risk fails closed to `destructive`; `run_shell` is always `destructive`. MCP tools register as `mcp_<server>_<tool>`
and never trust a server's self-declared tier. `tools/toolspec.py` is separate from `registry.py`
precisely to avoid the import cycle — keep it that way.

File tools, `run_shell`'s working directory, the workspace `SATURN.md` and `/init` follow
`core/workspace.py`: the launch folder (`agent.main` sets it from the cwd) plus folders added
with `/add-dir`. `workspace.resolve` is the ONE containment check (`tools/files._resolve` wraps
it); unset, the root falls back to `paths.workspace`, which is what tests and the benchmark use.
Snapshots record absolute paths, so `/undo` restores the right file from any folder; `move_file`
records the move itself (no byte copy) and `/undo` moves the file back; `delete_file` is the
same move into the user's Trash (`files._trash_dir`), never an unlink. `search_files` asks
Spotlight (`mdfind`) for a plain phrase on macOS — candidates only, re-matched by the regex and
filtered by the same containment and pruning as the walk; `tests/conftest.py` turns that seam
off for every test.

`notify/` is the scheduled-notification seam behind `schedule_notification` and `/notify`: `backend()` picks
by `sys.platform` (macOS = one launchd LaunchAgent per one-shot, shown by `osascript`; anything else is the
honest `Unsupported`). A new platform is one module plus one branch in `backend()`. It is not egress.
`notify/menubar.py` (tested, Cocoa-free) + `notify/menubar_app.py` (AppKit, pyobjc, macOS-only dep) are the
menu bar icon: a login LaunchAgent the REPL starts when `notify.menubar` is on (default off since
2026-09-29), which outlives the terminal; its Quit is `quit_all()`.

`core/hooks.py` runs the user's `~/.saturn/hooks.yaml` (`$SATURN_HOME`) commands on turn-start /
turn-end (`app/turn.run_turn`) and before- / after-write (`tools/files.py`). They are the user's
commands: no gate, not egress — which is why the file tools refuse to write the hooks file, and
likewise the live `config.yaml`, `permissions.json`, the memory file and its pending-review queue,
and the two `SATURN.md` instruction files (`tools/files._control_files`) — or to move a folder
that holds one of them.
`tests/conftest.py` gives every test an empty `SATURN_HOME` and a throwaway `HOME`.

Native macOS app tools (`tools/notes.py`, `tools/calendar.py`, `tools/mail.py`, `tools/contacts.py`,
`tools/reminders.py`, `tools/messages.py`, `tools/desktop.py`) go through `tools/applescript.py` — `run(script,
app=)` opens the app hidden then runs `osascript` (osascript alone gets -600 on a closed Calendar); output is
RS/US-delimited via `records()`. Each Apple event costs real time (Reminders ~1s, a Contacts person ~0.2s):
fetch in bulk, never per item, and never issue two `messages of <mailbox>` references in one Mail script. AppleScript, not EventKit: EventKit access from a terminal Python depends on the
terminal app's Info.plist. Readers are `untrusted=True` (shared notes, invitations, email); tests capture `applescript._run`.
`draft_mail` and `reply_mail` open an unsent draft and are NOT egress. `send_message` (iMessage) IS — the
chokepoint wiring above — to ONE person (`to=`, a handle from `search_contacts`) or ONE existing group
chat (`chat=`, a short `g…` ref only `find_group_chats` hands out; groups are never created; one ledger
event per recipient); `send_mail` is still deferred. `read_messages` reads `chat.db` and needs Full Disk
Access. `tools/shortcuts.py` runs the user's Shortcuts through the `shortcuts` CLI (two tools, not one per
shortcut: the bound schemas are the cached prefix). There is no clipboard tool on purpose — the user types
`@clipboard` (`core/mentions.py`) or `/copy`. Probe findings and what is still unverified:
`docs/superpowers/specs/2026-09-06-macos-apps.md`. `benchmark.py`'s approver (`bench_approver`) declines
every gated call into these modules: a benchmark run must never act on the user's real world.

### Slash commands

`commands/_framework.py` provides `@command(name, summary, aliases=, usage=, details=)`; one module owns
every view of a feature (`commands/trace.py` = `/trace` + export/replay engine, etc.). Every command
must accept `--help`; cut command spellings live in `_RENAMED` and print pointers for one release, then go.
Shared verb grammar (remove/rm/delete/…, `--save`) is in `commands/_utils.py`.

### Skills (`core/skills.py`, `commands/skills.py`)

The user's procedures as markdown: `$SATURN_HOME/skills/<name>/SKILL.md` or `<name>.md`, plus
`<workspace>/.saturn/skills/…` (wins on a shared name). The name is the file name; frontmatter
`name` / `description` / `disable-model-invocation`, every other key ignored. Typing
`/<name> [request]` (REPL `app/repl.py`, headless `-p`, both through
`app/session.skill_for_line`) sets `state["skill"]`, which `nodes/ground.py` folds into the
DYNAMIC half — no extra call, no prefix change, reset every turn. Built-in commands always win
(`commands._framework.resolves`). `/skills` lists, shows, creates (never over a file the loader
skipped) and deletes (Trash, the user's alone). The skills folders are control folders: the file
tools refuse to write, move or delete anything inside them, and a skill symlinked in is refused
where it really lives too (`tools/files._control_dirs`, `core.skills.linked_targets`). A
frontmatter value that is not one line of text is a problem, never expanded.

The ONE agent writer is `create_skill` (`tools/skills.py`, global folder only, in
`policy.ALWAYS_ASKS`, and it refuses unless `human_approved()`): `tools.skills.draft` builds
the path and text that both the gate (`tui/ui/approval._render_skill_draft` — no fold, no cut
line) and the tool use; `core.skills.draft_problem` is asked by the agent's hygiene
(`nodes/agent._skill_hygiene`, before the gate) and again at the write, and refuses characters
the gate would not show (`textutil.unseen_chars`); a draft over an existing skill without
`replace` is answered with the current text, stamped `done`. Saving is not running: the tool's
description says so, and a declined save carries `nodes/approval.SKILL_DECLINE_NOTE`. A skill
never changes the gate. Spec: `docs/superpowers/specs/2026-10-03-user-skills-design.md`.

### Same name, different file

`tools/` (implementations) vs `nodes/tools.py` (execution node) · `trace`: `stores/trace.py` records,
`tui/ui/trace.py` renders the rail, `commands/trace.py` is `/trace` · `plan`: `tools/planning.py` (the tool),
`tui/ui/plan.py` (the panel) · `config.py` (loader) vs `commands/config.py` (`/config`) · `trust/policy.py` (mechanism)
vs `commands/policy.py` (front door) · `think`: `core/think.py` (the decision), `commands/think.py` (`/think`).

### Tests

`tests/conftest.py` adds the repo root to `sys.path`; use the `isolated_paths` fixture whenever a test
touches configured paths so it can never write to the real `database/`. Grant lifecycle in
`trust/policy` is reset around every test automatically. LLM seams are monkeypatched at each node's
namespace — no test may reach a model, the network, or the embedder; `conftest.py` also turns
off the status bar's GPU / memory reader (`tui.ui.statusbar._read_usage` → `core/hardware.live`,
ioreg + vm_stat). `tests/` and `benchmark.py` import
compatibility names from `agent` (`from agent import build_agent, run_turn, …`); new code should import
from `app/` directly.

### Docs

Every document but the three the root needs (`README.md`, `CHANGELOG.md`, this file) lives under
`docs/`; `docs/README.md` is the index. `docs/ARCHITECTURE.md` — code map. `docs/pivot.md` — the
product goal since 2026-09-27 and the ranked work that closes the distance to it. `docs/engine.md`
— the loop's shape today and the ranked engine improvements. `docs/dogfood.md` — the prompts a real
user would try. `docs/advantages.md` — why the pivot items matter. `docs/OPTIMIZATIONS.md` — latency
techniques with the numbers behind them. `docs/research.md` — everything still open in pivot /
engine / advantages, ranked, with plans for the top items and an outside survey.
`docs/superpowers/` — specs and plans (of the plans dated 2026-10-01 `terminal-escape-sanitising`, Phase 1 of `skills` and `auto-memory-from-user-statements` — with the know-the-user spec's amendments — are built; `2026-10-03-create-skill` is built too, and `2026-10-04-adaptive-thinking` except its sampling experiment). `CHANGELOG.md` — user-visible history.
