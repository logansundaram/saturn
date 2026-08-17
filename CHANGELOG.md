# Changelog

All notable, user-visible changes to Saturn are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0, minor releases may change behavior).

## [Unreleased]

### Changed

- **One model family.** Saturday.ai now binds qwen3.5 / qwen3.6 / qwen3.8 only, as six tiers
  keyed by parameter size (`800m`, `2b`, `4b`, `9b`, `27b`, `35b`), each on the most advanced
  tag the family offers at that size. Confidence coloring is calibrated per model, and a red run
  only means "worse than 95 % of this model's clean output" for a model that was measured — five
  of the six tags were, with the 27b tier's thresholds estimated from its measured 27.8B sibling
  pending daemon support for qwen3.8 logprobs (`/confidence` names which you are on, and
  `/confidence tune` re-measures against your own daemon). `/models tier` lists parameters,
  context window and calibration state instead of a nickname.
- A binding left over from an older config (gemma4, qwen3-coder) is substituted in memory with
  the nearest size class and reported at startup. **config.yaml is never rewritten** — rebind
  with `/models tier <size>` to make it permanent.
- The fresh-install pull drops from `gemma4:e4b` (9.6 GB) to `qwen3.5:4b` (3.4 GB).

### Added
- **`/confidence`** — the front door for confidence coloring: `on`/`off` (on by default),
  `tune` to re-measure the active model against your daemon, `set <enter> [exit]` to type your
  own thresholds, `reset` to go back to the shipped calibration. Your values live in
  `database/confidence_calibration.json` and survive `/update`.

- Interrupt-and-correct (Esc to freeze and edit the streaming answer) now supports the
  qwen3.8 family (`qwen3.8:27b` verified with the splice-and-continue contract), alongside
  qwen3.5/3.6 — the whole supported family (gemma4 support is gone with the family lock above).
- **Always-allow grants now have a lifetime.** Answering `a` at the approval gate grants for
  the rest of the current turn by default (`runtime.grant_scope: task`): the tool's tier drop
  and any shell-prefix grant expire at the turn boundary, and the turn's closing note says what
  expired. `session` keeps the old behavior (grants live until Saturn exits); `persist` writes
  both halves — the tier drop and the prefix — to `permissions.json`. Previously one keypress
  relaxed a tool for the whole session and persisted a shell prefix forever, with nothing to see
  or revoke it. `/policy allow <prefix>` (the explicit command) still persists; the allowlist
  readout names each prefix's lifetime. The scope is a trust setting (session-only unless
  `--save`).
- **Shell environment scrubbing.** `run_shell` children no longer inherit secret-shaped
  environment variables (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*CREDENTIAL*`,
  `ANTHROPIC*`, `OPENAI*`, `AWS_*`, `GITHUB_*` by default) — a command can read a secret straight
  out of its own environment, and the workspace sandbox does nothing about that. The fragment
  list is `shell.env_scrub` in config.yaml; emptying it is a trust setting (session-only unless
  `--save`).

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

- Every model call now states explicitly whether the model may "think" (emit a hidden
  rationale) instead of inheriting the model's default: only the planner keeps its rationale
  (measured to matter for plan quality on small models); the judge, tool-argument generation,
  reasoning steps and the streamed answer run without it (measured faster and more accurate on
  small models, and a rationale can no longer eat the output budget and return an empty answer).
  Every call also carries an output-token bound so a looping generation ends as a truncated
  result instead of filling the context window; a model that rejects the think flag is
  detected once and the call retried without it; a degenerate, repeating draw is retried with a
  repeat penalty on that retry only.
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

- **A long shell command at the approval gate no longer reads as several commands.** When a
  command wrapped, the `$ ` marker was repeated on every wrapped fragment — a 200-character
  one-liner and a three-line script looked identical, and the destructive tail of a wrapped
  command read as its own separate, innocuous command. The marker now appears exactly once per
  logical line; continuation rows carry a dim `↳` at the same width, so the count of `$ ` is the
  count of commands.

- **A write the sandbox will refuse, or one over a binary file, no longer reads as "(no textual
  change)" at the approval gate.** Every non-diffable verdict — refused, binary, unreadable,
  no-op — printed the same dim caption as a genuinely empty diff, over the top of its own
  warning. For a refused or binary write that caption is false: a change *is* pending, it just
  can't be rendered. It now prints only when the diff really is empty. The same applies to an
  edit that cannot run (missing file, no match, ambiguous match).

- **Deciding per call (`s` at the approval gate) can no longer confuse two similar calls.** Each
  prompt now names its position in the batch (`call 2/3`) and shows a much longer argument
  summary, clamped from both ends instead of only the head — a path or a shell command carries
  its distinguishing token in the *tail*, so two calls sharing a long prefix used to render
  identically at the very prompt that exists to tell them apart.

- **An approval prompt raised by the injection quarantine now shows every argument in full.** A
  read-only call reaches the gate only when earlier tool output was flagged for embedded
  instructions — precisely when its arguments may have been steered and are the thing you are
  being asked to check. They were being cut to an 80-character summary under a banner telling you
  to check them.
- **The quarantine banner counts the flagged sources it doesn't have room to name** (`+6 more`),
  matching the gate's secret-scan warning. Previously it listed three and silently dropped the
  rest, understating the exposure.

- **The streaming answer no longer re-wraps the moment it finishes.** The live tail wrapped at
  the full terminal width while the finished answer renders at a readable ~100-column measure, so
  on any terminal wider than about 102 columns every line break in the answer moved at the
  hand-off. Both now use the same measure: the text stays put and only picks up its formatting.

- **The answer no longer jumps down mid-stream.** The `synthesize` trace row was printed when
  the node finished — i.e. after the answer had already started streaming — landing inside the
  response block and pushing the text down. The row now waits for `/trace full`; its metrics are
  unchanged in the status bar and the receipt, and a freeze or correction still gets its row.

- **The status bar no longer claims the wrong node is running.** It was showing `▸ plan` in
  active styling while `execute` was already working — the update it reads arrives when a node
  *finishes* — directly contradicting the `✓ plan` trace line above it. It now reports the last
  node that finished, in past tense, and says `starting` until the first one does.

- **A corrected answer gets its `── response` heading back.** After you froze and edited an
  answer, the resumed text and the final answer landed bare underneath the editor's own block —
  the original heading had scrolled away. The heading reopens and says what happened (`resumed
  after your edit`, or that you kept the text unchanged).

- **The screen no longer goes dead after you resume an edited answer.** Leaving the freeze editor
  left no status bar running, so the seconds while the model re-primes its context showed a
  frozen screen — right after the most interactive moment in the product. The bar is re-pinned on
  the way out, exactly as the approval gate and plan-review editor already did.

- **Answers are checked against what was actually gathered.** After a turn that observed
  something, every figure the answer states (three or more digits, or any decimal) is traced
  back to your words or the turn's tool results; a figure that traces to nothing gets ONE
  corrective regeneration, and anything still untraceable is disclosed under the answer
  ("these figures could not be traced to any gathered result") rather than passed off as
  gathered. The inverse check makes sure the value the plan's own calculate step produced
  actually appears in the answer. On a resumed (Esc-edited) answer the checks only mark — your
  edit is never regenerated over.
- Two more gaps between your request and the plan are closed deterministically once every step
  has run cleanly: a request for a total/average/difference/comparison that no step computed now
  gets its calculate step(s) instead of arithmetic done in the answer's prose, and a request that
  defers a target to an earlier result ("read the file it names") that the plan never followed
  gets the second hop. Both read your words only, stay quiet on an absence or after you edited
  the plan at review, and are bounded.
- **Removing a step at plan review now revokes its EFFECT, not just its wording.** The target
  of a state-changing step you drop or retire at the review editor (the file it names, or every
  write for a step that names none) is refused for the rest of the turn — checked on the step's
  description before anything is generated and again on the generated arguments right before
  the call is emitted, so a redraft cannot re-do the work under a different sentence or a
  hidden path. Removing a read revokes nothing; a step you merely reworded still runs; the
  refusal reads as your single-step veto (the rest of the plan continues). Removed steps whose
  redraft keeps coming back end the turn honestly instead of spending the replan budget.
- **A step redrafted after results came back may only act on what you asked for.** A
  state-changing step the engine adds mid-turn (after files or pages have been read) is dropped
  unless your own words asked for a workspace change and named that target — text inside a
  file or web page can no longer add a write or a shell command to the plan (checked on the
  generated arguments; a mid-turn steering correction counts as your words). Steps drafted up
  front, before anything was read, are exempt.
- `ask_user` is gated by three deterministic rules before it interrupts you: one question per
  turn; if your request names something the agent can search itself ("search my notes…") it
  searches before asking; and a question whose answer no later step could use is reported in the
  answer instead of stopping the run. A question you asked for in your own words ("ask me
  which…") always runs. When a question is refused, the plan is redrafted around it.
- Pressing Esc to review the plan and then typing a steering correction (Esc with text) before
  the next step boundary no longer loses the review: the pause is honored first and the
  correction is applied at the following boundary (several corrections land together, oldest
  first). Corrections that arrive after the turn's last boundary each run as their own next
  message.
- When a request names a workspace file that no plan step ever acted on and every step has
  already run, the engine now adds the missing steps deterministically (bounded by the replan
  budget) instead of answering with the work half done. Only paths YOU named count — text inside
  a file or web page can never make the engine demand work of itself — and a step you removed at
  plan review is honored, not re-added.
- `calculate` can no longer be used to launder a made-up number into a "computed" result: an
  expression that is a bare value (`551`) is refused with a hint to write the actual arithmetic
  over gathered values, and only lands as an incident when every retry does the same.
- A turn that keeps issuing the exact same tool call with the same arguments is stopped on the
  third repeat as a disclosed "step is looping" incident (the engine reads its own record of
  executed calls) instead of burning the iteration budget; a legitimate second read still runs.
- `/policy allow` prefix grants (and the gate's always-allow `a`) now screen the arguments
  AFTER the granted prefix at every use: capability-introducing flags (`--output`, `-c`,
  `--exec`, …), globs, and paths outside the workspace disqualify the command, a
  general-purpose interpreter (`python`, `npm`, `powershell`, …) is only ever exempt as the exact
  granted command, and non-ASCII text (a lookalike `；`) never passes the automation path.
  Previously `git log --output=<path>` rode in on a `git log` grant.
- A hand-edited `permissions.json` whose fields have the wrong shape (a string where the
  allowlist should be, a list where the overrides mapping should be) now fails closed like a
  garbled file — strict defaults, recorded at startup, the file kept aside as `.corrupt` —
  instead of being iterated as-is.
- Confidence marking is now calibrated per model: `runtime.confidence_threshold` defaults to
  `auto`, which uses the synthesizer model's own measured threshold ("worse than 95 % of this
  model's clean output" — the shipped table covers the tier synthesizers and qwen3.8:27b;
  regenerate with `utilities/confidence_calibrate.py`) and falls back to the old fixed 0.20 for
  an uncalibrated model. Set a number to pin it as before. Note: Ollama 0.32 reports per-token
  logprobs for qwen3.8 on the first chunk only, so its marks are unmeasured for now — the table
  carries a provisional entry inherited from qwen3.6:27b until the daemon reports them.
- Confidence marking is steadier: an uncertain run no longer flickers off on one merely-unlikely
  token (two-threshold hysteresis — `runtime.confidence_exit_threshold`, derived by default),
  and function words (the, of, is, …) never count toward or break a run — they draw low
  probability from many valid continuations, not from uncertainty about content.
- Interrupt-and-correct: pressing Esc mid-word now lets the streaming answer finish the word
  before freezing (a few more tokens at most; a chunk that starts the next word is not kept), so
  the editor opens on a clean boundary and the continuation picks up naturally — press Esc a
  second time to cut immediately.
- Interrupt-and-correct: the edited answer prefix is trimmed of trailing spaces/tabs before
  generation resumes (a trailing space is a token boundary the model never produces, so the
  continuation could start awkwardly); newlines are kept and a resume without changes is not
  recorded as an edit.
- The approval gate's file-write preview now says what the write will actually do: a
  byte-identical rewrite (including a Windows CRLF-vs-LF no-op) reads "no change" instead of a
  full-file diff, an existing binary file is named as binary instead of rendering as garbage,
  and a path the workspace sandbox will refuse is flagged REFUSED at the prompt. The preview
  resolves paths through the same sandbox check the file tools use.
- A plan step with an unrecognized status (a garbled or legacy record) now renders as
  `? ⟨unknown status: …⟩` instead of being shown as pending.
- The approval prompt always renders: if a preview (the file diff, the shell command view)
  fails to draw, a plain view names the call and the same reject-by-default prompt runs —
  previously the turn died with the human never asked.
- A display bug while rendering the live trace rail or plan can no longer fail the turn: the
  render error prints as one line, the run stays recorded, and the answer still arrives.
- An oversized node delta no longer vanishes from the trace record: instead of slicing the
  stored JSON (an undecodable blob — the whole update gone from `/trace`, `data: null` in
  exports), the tracer clips long values, then keeps the fields that fit and records an explicit
  `truncated` marker naming what was dropped; `/trace` replay discloses it under the node row.
- Choosing an approval tier explicitly (Shift+Tab, `/config runtime.auto_approve`) while the
  gate is open now supersedes the pre-open snapshot, so `/policy open off` lands on the tier
  you set last instead of restoring a looser one.
- The approval gate now approves a batch only on an explicit approval; any unrecognized
  resume value rejects (previously any truthy value approved).
- An answer that came back empty no longer swallows the engine's own disclosures: the
  "could not be completed" incidents note and the Sources footer are appended regardless, and
  the recorded answer states that no answer text was produced.
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
