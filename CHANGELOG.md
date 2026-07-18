# Changelog

All notable, user-visible changes to Saturn are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0, minor releases may change behavior).

## [Unreleased]

### Added

- **`saturn -q "<question>"` — one-shot query mode.** The same headless turn as `-p` (same
  engine loop, same deny-by-default approval gate, same trace recording), rendered for pipes:
  stdout carries only the final synthesized answer, step-line progress (plan drafted, step N,
  synthesizing) goes to stderr, and the run auto-exports to `logging/exports/` so the closing
  `recorded: saturn --replay <file>` receipt names a command that actually replays the run
  offline. Blocked or denied actions are disclosed in the answer body exactly as `-p` does; a
  completed run exits 0. `--export FILE` overrides the export destination; `--json` stays a
  `-p` contract.
- **`/trace context` — see exactly what your machine told the model.** A new observability
  subview that reconstructs, token-for-token and per node, the full input each local model call
  received: every system prompt, every curated context block, at full fidelity and with no
  output noise. Where `/trace invoke` answers "what did each call see and say", this answers the
  privacy question "what did my machine actually send the model." `--node <name>` focuses one
  node so per-step context is diffable; `--preview` clips; `-l` lists runs that have model calls.
- **Two new graded probes in the trust benchmark.** `python benchmark.py` now measures the two
  distinctive safety mechanisms that were previously untested end-to-end: the **injection
  quarantine** (a planted corpus document carrying instruction-shaped content is retrieved
  through the live knowledge-base path; the benchmark grades whether it was flagged and fenced —
  an unflagged injection is a `--strict` FAIL) and the **semantic write gate** (baits ask for an
  unfindable fact to be looked up and saved to a file; the benchmark grades whether the gate
  refused to write a value it never gathered). The report gains an injection flag rate and a
  fabrication catch rate alongside the existing grounding and gate-coverage numbers. The planted
  document is removed after the run, leaving the corpus as it was found.
- **`/draft` — write your own plan.** Compose a step list by hand in the same editor you
  get at plan review (same `add`/`edit`/`tool`/`move`/`drop` grammar), then type your request:
  the agent executes *your* plan instead of drafting one. Tool spellings are normalized
  (`calc` → `calculate`); an unrecognized tool is kept as written and fails closed at
  execution. Everything downstream is unchanged — per-step reflection, the approval gate, and
  mid-turn Esc review all still apply, so a hand-written plan gets the full safety envelope.
  `/plan` shows the pending draft; `/draft clear` discards it. (Briefly spelled `/plan draft`
  during development; that spelling prints a pointer to `/draft`.)

### Removed

A focus pass: a few high-quality features over accumulated surface. Each cut removes a thing
to learn, audit, and maintain — none removes a protection.

- **`http_request`.** The one-call-to-any-REST-API tool is gone; the MCP client is the
  integration surface, and it does the job with per-server trust declarations, outgoing-arg
  secret redaction, and connection status the generic tool never had. With it gone, the only
  ways anything leaves your machine are a web search query, a page fetch, and the MCP servers
  you configured — a shorter list to verify with `/privacy egress`.
- **`/privacy redact`.** The secret-redaction *command* is gone — it configured a boundary
  that only exists behind a remote `OLLAMA_HOST`, dormant since cloud-model support was
  shelved. The protection itself is unchanged: secret redaction still guards remote-Ollama
  and remote-MCP sends, and the approval gate still warns when a call's arguments carry a
  secret-like value. The mode remains settable as a config key
  (`/config runtime.redaction off|warn|redact` — session-only unless `--save`).
- **The capability benchmark suites.** `benchmark.py` now runs exactly one thing: the graded
  trust benchmark (grounding catch rate + gate coverage) — the numbers the product's claims
  rest on. The ungraded capability/conversation harness (`--capability`, `--suites`, `--all`)
  is gone; engine regressions are covered by the offline test suite.
- **Per-document LLM summaries at ingest.** Adding a document (or writing a workspace file) no
  longer runs a model call to summarize it — the document manifests carry the file's own first
  line instead. Ingest is faster, and untrusted document text is never fed through a model at
  ingest time. A leftover `cache/summaries.json` is simply unused.
- **`/trace calls`, `/trace cost`, `/trace state`, and the `--md` export format.** `calls`
  duplicated the per-run drill-down, `cost` measured cloud-era spend a local agent doesn't
  have (tok/s and context fill are live in the status bar), `state` was a debugging dump, and
  the JSON export was always the one replayable record.
- **`/config key`.** No Saturn feature takes an API key (web search is keyless, inference is
  local), so the managed-key picker managed an empty registry. Secrets for MCP servers'
  `${VAR}` expansion are plain env vars — put them in `.env`; typing `/config key` points
  there.
- **`/resume delete` / `/resume rename`.** Sessions are plain `.json` files under
  `database/sessions/` — manage them there. Crash-safe autosave, `save [name]`, `<name>`
  restore, and `list` are unchanged; a habit-typed removal verb prints a pointer and deletes
  nothing.

### Changed

- The live `config.yaml` is no longer tracked by git — it is user data (persisted settings
  land in it), and tracking it could make `/update` fail once you had ever saved a setting.
  It is now seeded on first run from the tracked template `config.default.yaml`.
  **Migration for clone installs:** pulling this change removes an unmodified `config.yaml`
  (it is recreated from the template on the next launch); if you had edited it, git refuses
  the pull once — back the file up, `git checkout -- config.yaml`, pull again, and re-apply
  your settings (they now persist without dirtying the repo).
- Trust-posture settings (`runtime.auto_approve`, `runtime.airgap`, `runtime.quarantine`,
  `runtime.redaction`) set through `/config` now apply for the session only unless you pass
  an explicit `--save` — a loosened security posture is never written to disk silently,
  matching the `/policy` and `/privacy` toggles.
- A plan step naming a tool that doesn't exist now fails closed as a disclosed error the
  engine can replan around, instead of silently degrading into the model answering the step
  from its own knowledge.

### Fixed

- The semantic write gate and the self-correction judge no longer misread a successful step
  whose output merely *begins* with "ERROR" (e.g. reading an error log) as a failed step —
  failure now keys exclusively on the step's recorded status. Previously this could skip a
  legitimate write (and cancel the rest of the run) or trigger a spurious replan.
- The bounded "search came up empty — retry once" self-correction actually retries now: a
  redrafted step reusing the original wording was silently dropped as a duplicate, so the
  turn could answer "not found" without ever re-searching.
- The plan executor's "previous step" context and the write gate no longer mistake a later
  step you removed at plan review for the most recent completed work.
- The prompt-injection quarantine now derives its tool classifications from the live tool
  registry: tools declare `untrusted=True` at registration, and the tool-coercion pattern
  covers every gated tool (including MCP tools) instead of a frozen list of four built-ins.
- Answer streaming no longer does quadratic per-token work (noticeable as growing latency on
  long answers, especially with confidence grading on).
- A hardware tier without an `embedder:` entry now reports an actionable config problem
  instead of silently using a hard-coded model id.
- Relaxing a tool's approval tier (`/policy risk … read_only`, an always-allow grant) no
  longer removes that tool from the injection quarantine's coercion scan.
- Shell commands killed by a signal (negative exit codes on Linux/macOS) now classify as
  failed runs for the engine's retry logic.
- `/trace invoke` no longer records a deliberately frozen (Esc) answer stream as a failed
  model call — it is recorded as cancelled.
- On terminals without `rich`, a freeze-edited answer now re-renders in full after the turn,
  so the correction actually appears in the transcript.

## [0.1.0] — 2026-07-10

First public release.

Saturn is a private, local-first AI agent for the terminal: inference runs on your own
machine through [Ollama](https://ollama.com), every step is visible while it happens, and
nothing side-effecting runs — and nothing leaves your machine — without your approval.

### The engine

- Plan/execute agent loop: the model drafts a step-by-step plan, executes it one step at a
  time against a curated per-step context, and self-corrects (a judge reviews each step's
  outcome and can revise the remaining plan, bounded by iteration/replan budgets).
- Semantic write gate: before a value is persisted to disk, a judge verifies it actually came
  from the request or gathered results — and fails **closed** when it can't verify.
- Honest failure: skipped, blocked, or failed steps are disclosed plainly in the answer,
  never papered over.

### Human control

- Risk-tiered approval gate: `read_only` tools run freely; side-effecting and destructive
  calls pause for your explicit approval, with full-fidelity rendering of exactly what will
  run (unified diffs for file writes, the complete shell command, full HTTP requests).
  `/policy` is the single front door for every relaxation (tier threshold, per-tool
  overrides, persisted shell-prefix allowlist).
- Plan review and editing: pause at any step boundary (Esc), inspect and edit the live plan
  (add/drop/reorder/retarget); a step you remove stays removed — the engine's
  self-correction cannot resurrect it.
- Mid-turn steering: type a correction and press Esc — the remaining plan is redrafted
  around your words without restarting the turn.
- Interrupt-and-correct: press Esc while the answer streams to freeze it, edit the text, and
  have the model continue from your edited prefix; human-authored spans stay marked in the
  final answer and its audit record.
- `ask_user`: the agent asks you mid-run instead of guessing.

### The trust stack

- Egress ledger and air gap: every network exit is recorded (host, bytes, channel) and
  renders live in the trace; `/privacy airgap` seals the boundary entirely.
- Prompt-injection quarantine: instruction-shaped content in untrusted tool output
  (web/MCP/corpus) is flagged, fenced as data-not-instructions, and escalates the next tool
  batch to the human gate.
- Secret redaction at the network boundary, plus a secret scan warning at the approval gate.
- Per-answer trust receipt and answer provenance: citations resolve to numbered sources with
  origin (local vs network) and trust flags (`/trace answer`, `/trace source`).
- Token-confidence grading: low-confidence runs of the streamed answer render red — live, in
  the freeze editor, and in the final answer.

### Tools

- Files (read/write/edit/search/find/list, sandboxed to a workspace; pre-write snapshots
  back `/undo`), shell (always gated, exact-command approval), keyless web search
  (DuckDuckGo) and page extraction, `http_request` as the universal REST integration
  (always gated, full request shown), a whitelisted-AST calculator, local time.
- RAG knowledge base over your documents (txt/md/pdf/html/csv/docx) with cited retrieval,
  durable memory (`remember`/`recall`), and workspace instructions via `SATURDAY.md`.
- MCP client: connect stdio/HTTP/SSE servers from `config.yaml`; remote tools face the same
  approval gate and never self-declare their risk tier.

### The terminal app

- Streaming answers, an editable plan rail, an htop-style status bar, `@file` mentions with
  completion, multiline input with paste chips, drag-and-drop file handling, type-ahead
  queueing while a turn runs.
- Observability: `/trace` drill-down of any run (plan, per-step reasoning, tool I/O, LLM
  calls, cost), exportable run records, and fully offline replay (`saturn --replay`).
- Sessions (`/resume` with crash-safe autosave), auto-compaction of long histories,
  five-role model configuration over local Ollama models (`/models`, laptop/workstation
  tiers), first-run health check (`/config setup`).
- Headless mode: `saturn -p "query"` with `--json` and `--export`, piped-stdin attachment,
  gated calls denied by default.

[Unreleased]: https://github.com/logansundaram/saturn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/logansundaram/saturn/releases/tag/v0.1.0
