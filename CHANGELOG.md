# Changelog

All notable, user-visible changes to Saturn are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0, minor releases may change behavior).

## [Unreleased]

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
