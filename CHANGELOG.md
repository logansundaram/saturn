# Changelog

All notable, user-visible changes to Saturn are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0, minor releases may change behavior).

## [Unreleased]

### Changed

- **Prompts are laid out for the daemon's prompt cache, and an idle prime keeps it warm.**
  Measured on the 9b tier: every plan call re-read ~2,000 tokens of unchanged grounding (5 s),
  and every execute, rectify and synthesize call re-read its whole prompt (a 16-call turn spent
  ~96 s of 141 s in prefill). llama-server can only resume from a checkpoint it saved 1024 or 4
  tokens before an earlier prompt's end, so three things now hold: the grounding is split into
  a stable half (instructions, manifests, the always-loaded memory layers) and a per-turn half
  (matched memory, the recap, attachments), and every node sends the stable half as its own
  message right after its system prompt; the results block caps each result when it lands
  (landing-order budget) instead of re-truncating every earlier result as the plan grows, and
  the previous-step callout is bounded at 2,000 chars so a step's changing tail fits one
  prefill batch; and between turns (plus once after the weights load) the agent re-sends each
  node's stable prefix with one predicted token so the next turn's calls resume at that
  boundary (`runtime.prime`, default on — measured: the plan call's prefill 5 s → 0.2 s).
- **Tool arguments are generated under the tool's JSON schema as a grammar, not a native
  tool bind.** Ollama's chat template renders bound tools into the system message, so every
  tool step's prompt began differently and re-prefilled whole (8k tokens, 20 s per step). The
  model now answers `{"arguments": …}` under the schema — or `{"refusal": "…"}` when the step's
  tool must not be called, which lands as the same text-fallback incident as before.
- **The rectify judge is asked for a one-or-two-sentence rationale** (it averaged 113 output
  tokens, ~3 s per verdict on the 9b).

- **The planner thinks only as much as the request needs.** Every model call except the
  planner's now runs with the model's "thinking" off. The planner keeps it: without its
  rationale the 9b turned an open request like "write me a story" into a single "ask the user
  what kind" step, which the ask gate refuses and the turn then ends with "action guarded"
  instead of a story. Its prompt now tells it to decide simple requests in a sentence or two
  and think longer only for multi-part or ambiguous ones. Measured on the 9b tier: a one-tool
  request's warm plan call 5.9 s → 3.1 s, the story request plans correctly again, and a
  five-step request keeps its full rationale.
- **A step that names its target no longer pays a model call for the arguments.** A `read_file`
  or `list_directory` step whose label spells exactly one existing workspace path ("Read
  notes.md", "List data/"), a `find_files` step with exactly one glob ("Find every *.csv"),
  and `current_time`, get their call copied from the plan step instead of generated — about 3 s
  less per such step on the 9b; a "read all files" plan drops one call per file. Anything less
  certain (two paths, a placeholder, a folder named in words, a name that isn't there) is
  generated as before, and every
  later check — plan-review revocation, effect authorization, the stall detector, the approval
  gate — still reads the arguments. The gate's explain shows "arguments copied from the plan
  step itself".
- **Earlier results take half the room in per-step prompts.** The shared results block that
  rides every execute step and the plan text rectify/replan read is now budgeted at 8k
  characters (was 16k, the per-result floor 400, was 800) — one or two results still ride
  whole, the immediately preceding step keeps its own fuller callout, and the answer still reads
  the full observations; a long plan's per-step prompt prefill roughly halves.
- **The model stays loaded between turns.** Every request now carries `runtime.keep_alive`
  (default 30 minutes; `-1` never unloads, `null` keeps Ollama's 5-minute default), so a pause
  longer than five minutes no longer costs the whole model load on the next turn.
- **The planner reuses the daemon's prompt cache between turns.** The grounding block now
  lists what is stable first (workspace instructions, the knowledge-base and workspace
  manifests) and what changes per turn last (memory's by-match facts, the recent-conversation
  recap, attachments), so the next turn's plan call only re-reads its tail instead of the whole
  catalog (measured: 9 s → 1.4 s on the 9b when the turn-to-turn change fits in the last 512
  tokens).

### Added

- **Memory grows, gated.** The flat `remember` notepad became a layered store: `user`
  (preferences, identity), `commitments` (open items with a due date), `memo` (dated notes),
  `agent` (what it learned about this machine and its tools), `entities` (people, projects, your
  shorthand), `negative` (what not to do again). User facts, open commitments and the recent
  memo digest load every turn; the rest load only when they match the request, all under one cap
  (`memory.context_cap`) with a trailer naming what didn't load — `/trace context` shows the exact
  block. Every fact carries provenance (the run it came from, who said it, last use,
  confirmations, a sensitivity mark that withholds it from remote inference); `remember` takes
  a `layer` and a `replaces=#id` so a correction retires the old fact instead of contradicting
  it. Learning happens at `/memory review` (and `/quit`): your mid-task corrections, plan-review
  vetoes, gate denials, unfinished steps and each compaction summary queue as candidates, the
  model can add proposals from the transcript, and each is shown as a diff line and kept only
  on your y — never a silent write. `/memory` gained `list <layer>`, `add --layer/--replaces`,
  `edit`, `why`, `review`, `pending`, `stale`, `done`. The old file format migrates on its
  first write. The gate frame for `remember` names the session grant (`/policy risk remember
  read_only`) instead of inviting a lower tier.
- **Past runs are searchable.** An FTS5 index over every recorded run's request and answer
  (no embedder): the `recall_runs` tool for the agent ("what did we do last week"), `/trace
  search <words>` for you. The last compaction summary is persisted as
  `database/memory/last_summary.md`.
- **Memory knobs and escape hatches.** `memory.stale_days` (default 90) sets when an unmatched
  by-match fact is flagged stale in `/memory`; `memory.review_llm` (default true) decides
  whether the review also asks the utility model for proposals; `/quit --no-review` exits
  without the review; `/memory add --sens <mark>` (and `remember(sensitivity=…)`) marks a fact
  sensitive so it is withheld — by the grounding block AND by `recall` — whenever inference is
  not local. `/trace search` and `recall_runs` drop stopwords and fall back to any-term matches,
  so "the report we made Monday" finds the report.
- **Benchmark memory tasks.** `benchmark.py` runs three tasks over an isolated memory file:
  recall in a fresh run, supersession of a corrected fact, and a planted corpus document that
  tries to store a memory (must face the gate).

- **`/models` is now the model page — the hardware scan folded in.** It probes the chip, cores,
  RAM and NVIDIA VRAM once at launch, works out the memory the model runner can address (Apple
  unified memory at ~75% of RAM; a card's full VRAM; half of RAM and capped at 9b with no
  accelerator), and prices the two qwen ladders against it: the six chat tiers (the most advanced
  tag per size) and the three qwen3-embedding sizes. Each row shows weights, the context window
  this config gives it, the memory it needs at that window (weights + KV cache + headroom — only
  1 in 4 layers of these hybrid models keeps a cache), whether it is pulled, and whether it fits;
  the recommendation is marked, and the embedder recommended is the largest that fits BESIDE the
  chat model, and each chat row says whether its confidence coloring is calibrated. Rows are
  numbered: Enter takes the recommended tier (and asks, y/N, before also moving the embedder —
  that re-embeds the corpus), a number picks one tier or embedder, n keeps things as they are. A
  pick whose model isn't pulled asks (y/N) and only switches after the pull lands; an embedder
  pick is set on every tier. Runs on the very first
  launch right before `/config setup`; `/models list` renders without prompting, `/models rescan`
  re-reads the hardware, `/scan` is an alias. The verbatim `ollama list` view and the two-step
  role picker are gone — the direct binds (`/models all|<role>|embedder|tier …`) remain.

### Fixed

- **Long writes actually happen.** `write_file` / `edit_file` calls now generate under a
  file-sized output bound; before, a story or a list longer than ~500 tokens was cut off inside
  the tool call, the step reported "no tool call emitted", and the engine retried the identical
  truncation three times. A call that still exceeds the bound is refused once with the limit
  named instead of looping.
- **No more redraft loops on an un-generatable call.** A tool the engine fails to generate a
  valid call for twice in one turn (without it ever executing) ends the run with the incidents
  disclosed, instead of the judge redrafting the same step until the replan budget runs out.
- **A failed write is not "fabrication".** The semantic write gate arms only on a failed
  search/read upstream, not on a prior write attempt that errored — the redraft after a
  transient write failure is no longer skipped with a fabricated-value disclosure.
- **`search_knowledge_base` works from a clean install.** `numpy` is declared; without it every
  knowledge-base search failed at query time.
- **One dead-end retry per turn.** An empty search or listing gets one retry with a different
  approach; when that also comes up empty the turn reports it instead of replanning again
  (before: an empty workspace was listed three times over 90 s).
- **The first turn no longer pays the model load.** The REPL warms the tier's model on a
  background thread right after the startup health check (a cold "hello" took 50 s; warm, 15 s).
- **`read_file` says where the file is not.** A missing path reports "File not found in the
  workspace" and, when the name is an ingested knowledge-base document, points at
  `search_knowledge_base` instead of surfacing a raw `[Errno 2]`.
- **The workspace list is true every turn.** The grounding context's "Workspace files" block
  is reconciled with the directory on disk before each turn: files deleted or renamed outside
  the agent no longer linger as phantom entries the planner tries to read, and files dropped in
  by hand are listed with a first-line summary.
- **Named knowledge-base documents are searched, not read.** A `read_file` step whose
  description names an ingested document with no workspace file of that name is retargeted to
  `search_knowledge_base` at plan and replan time (the planner rule says so too, but the engine
  knows both manifests and does not need the model to get it right).
- **The judge sees the evidence it judges.** Earlier results in the rectify/replan/execute
  prompts were clipped to 800 characters each, so the judge saw a third of a web search and
  asked for another one (its own verdict said the answer was not "in the truncated" results).
  Results now ride whole up to 3,000 characters, sharing a 16k-character block budget that
  shrinks the per-result share on long plans instead of exceeding the window.
- **Hidden entries are not workspace content.** `list_directory`, `find_files` and
  `search_files` skip dot-entries (`.manifest.md`, `.git`, `.DS_Store`), so an empty workspace
  reads as empty instead of the agent listing and relaying its own bookkeeping file.
- **Empty reasoning steps are bounded too.** A reasoning step that produces nothing twice in a
  turn ends the run with the incident disclosed, under the same guard as an un-generatable
  tool call.
- **Write-only turns are not figure-checked.** The groundedness note fires only when the turn
  gathered something (a read, a search, a shell run); a turn whose tools only wrote a file the
  agent authored no longer marks the file's own numbers as untraceable.
- The rectify judge's output bound is 1024 (a verbose verdict was being cut mid-JSON), and a
  structured draw the daemon cuts at the bound is named in `logging/diag.log`.

### Removed

- The never-written `user_profile.md` / `agent_profile.md` workspace files are no longer read
  by the grounding node — the `user` and `agent` memory layers replace them.

### Changed

- **Context windows now step up the ladder.** `config.default.yaml` ships 32k for 800m–4b,
  64k for 9b and 27b, and 128k for 35b (it was 32k everywhere), priced so each tier fits the
  hardware it lands on with headroom (a 27b at 64k is ~22.5 GB; the 35b MoE's cache is so cheap
  that 128k costs less than 27b's 64k). Existing config.yaml files keep their values —
  `/config context` changes the window live, and `/scan` shows what each choice costs.

- **Launch and sync got faster.** Startup no longer re-reads every corpus document to hash it —
  an unchanged file is recognized from its recorded size+mtime (content hashing remains the
  authority: any mismatch, and any file modified too close to its last verification, is
  re-hashed — a stale document can never be skipped on a timestamp alone). Heavy libraries
  (PDF/HTML extraction, the text chunker) now load only when an ingest or page fetch actually
  needs them, and the trace database gained indexes so `/trace` stays fast on long-lived
  installs.

- **`/policy risk` renders as an aligned, color-coded table** (the same green/yellow/red the
  approval gate uses), and the bare `/policy` readout reports the EFFECTIVE airgap/redaction/
  quarantine modes from their owning modules rather than echoing raw config strings.

- **`/models` lists one model per size, not your whole `ollama list`.** The table (and the
  numbered picker behind it) was a verbatim daemon inventory — every tag you had ever pulled,
  most of which the family gate refuses, so the picker offered numbers that led straight to a
  refusal. It now shows one tag per size class — the most recent of the supported qwen3.5-3.8
  family at that size, so a version bump like `qwen3.8:27b` supersedes `qwen3.6:27b` instead of
  sitting beside it — plus embedding models, which are exempt from that gate (this is the only
  listing that surfaces them; `/models embedder <id>` still binds one by name), plus anything
  currently bound or literally named in your config.yaml (a pulled legacy binding stays visible
  so it can be understood). Everything else is counted and disclosed under the
  table (`(16 other installed models hidden …)`), never silently dropped, and a superseded tag is
  still bindable by name (`/models all qwen3.6:27b`). If nothing at all is offerable, the table
  says so honestly and points at the by-name binds instead of asking whether the daemon is down.

### Fixed

- **Confidence coloring now measures qwen3.8:27b for real.** Ollama 0.33 runs qwen3.8 under
  multi-token-prediction speculative decoding and reports a logprob only for the first token of
  each draft batch, which is what the earlier "logprobs on the first chunk only" reading was: the
  answer overlay for that model was all but empty, and the calibration utility could not measure
  it (its shipped row was an estimate inherited from qwen3.6:27b). While confidence coloring is
  on, every request Saturn sends the daemon — planner, execute, judge, the streamed answer, the
  interrupt-and-correct continuation, the background summaries — now turns drafting off
  (`draft_num_predict: 0`; models without a drafter and older daemons are unaffected), and the
  shipped table carries a measured qwen3.8:27b row. It has to be every request, not only the ones
  that ask for logprobs: the daemon treats the option as a runner setting and reloads the weights
  whenever two requests disagree on it, which made every qwen3.8 turn reload the model twice. No
  speed cost: in the daemon's own timings drafting was slower for this model with or without
  logprobs (about 8 versus 12 tokens per second on an M-series Mac).
- **`/trace why` no longer mis-renders an answer whose prose contains the word "Sources:".** The
  cited-sources footer is now recognized only in its exact produced shape, through the one shared
  parser every footer reader uses (the trust-colored answer footer and `/trace answer` included);
  an answer record truncated mid-footer still has the partial footer stripped before the
  per-source analysis rather than being analyzed as prose.

- **The plan editor understands the same tool spellings as `/draft`.** A synonym (`calc`) maps to
  the registered tool, and a no-tool marker (`none`, `reasoning`, `answer`, …) makes a genuine
  reasoning step instead of minting a step that fails closed at execution.

- **Always-allow grants leave an accurate audit trail.** Every tier drop that actually changed a
  tool's live tier is logged with the lifetime it really got — a grant that could not be
  persisted (read-only install) is recorded as session-scoped instead of claiming durability.

- **Live confidence coloring can no longer lose the tail of a long uncertain stretch.** The
  visible-window grading now walks back to a real run boundary instead of a fixed margin, so the
  streaming tail and the final render mark the same spans.

- **A second `a` at the gate no longer makes an always-allow grant permanent.** Granting the same
  tool twice in one turn — reachable without an adversary, since an injection-flagged observation
  re-arms the quarantine escalation and gates the tool again — registered a second undo that
  captured the already-granted tier. It then re-dropped the tier the first undo had just restored,
  so the grant stood for the rest of the process while the closing note said "always-allow grants
  expired with this turn". The grant now expires as promised, and the note counts it once.

- **A step the engine adds after reading results is no longer silently dropped.** Effect
  authorization matched only *path* tokens, so a state-changing step naming none — a `remember`,
  an MCP call, or a write whose description says "into the report" rather than a filename — was
  refused whenever the request happened to mention any path at all. The refusal then removed the
  step with no record, and the answer described a finished turn whose file was never written.
  Effects that name no target now fall through to the same residual that already applied when the
  request named no path, and a step that IS refused is recorded as a disclosed incident instead of
  vanishing. The guarantee is unchanged where it matters: a request asking for no change still
  authorizes nothing, and a real file write always names its path in the call it generates.

- **"Remember that I prefer terse answers" is no longer refused as an unauthorized effect.** The
  vocabulary that decides whether you asked for something to change had no word for memory, while
  `remember` is a state-changing tool — so a remember step added mid-turn was blocked, and the
  record claimed you had never asked for it. Asking what Saturn remembers is still a read.

- **Removing a read step at plan review no longer cancels the write you kept.** A dropped step
  whose wording merely contained a write word ("Read the current notes.md before appending")
  revoked that path for the rest of the turn, and the write step you deliberately left in the plan
  was refused with a message saying you had removed it. Only an actual destination ("…and save it
  to notes.md") counts now.

- **A figure the model made up can no longer vouch for itself by being written to a file.** The
  groundedness check treated the arguments of a tool call as gathered evidence, so an invented
  number passed to `write_file` became "traceable" the moment the call was echoed back.

- **The two answer-correction passes no longer undo each other.** When an answer both stated an
  untraceable figure and omitted one the plan computed, the second rewrite regenerated from the
  original draft — throwing away the first correction — and its result was never re-checked, so a
  fabricated figure could ship with no disclosure at all. Both passes now revise the current
  answer and the check follows the text being shipped.

- **`/trace` and exports keep the answer's provenance history again.** A record larger than the
  write-time cap was shrunk by clipping text, which cannot shrink the per-token confidence
  ledger — so past roughly 320 words the whole answer buffer was dropped, taking the edit history
  and provenance spans with it. Long lists are now trimmed with the loss named, and a record that
  lost anything is reported as INCOMPLETE in `/trace answer` instead of rendering as complete.

- **`/models tier` shows the model your config actually binds.** For a size-named tier the table
  printed the newest tag at that size regardless of what you had bound, so binding
  `qwen3.6:27b` on tier `27b` ran `qwen3.6:27b` while the table said `qwen3.8:27b` — directly above
  a migration note contradicting it.

- **An errored search no longer cancels the rest of the plan.** A failed search still counted as
  evidence gathered, so the engine asked whether the error text contained the file a later step
  referred to, concluded the item was missing, and cancelled everything remaining. With no search
  at all the same plan ran fine.

- **A malformed exported record no longer crashes `saturn --replay`.** A hand-edited or corrupt
  export raised a traceback instead of reporting that the file could not be rendered.

- **Hostnames and email addresses are no longer mistaken for files in your workspace.** Asking
  Saturn to email a summary to `jo.smith@corp.com`, or to look at `anthropic.com`, made it decide
  the request named a file no step had touched — and spend a re-plan trying to read it, failing,
  and disclosing the failure on a request that was already complete.

- **`/docs add` blames the right file.** A loader error from an unrelated document whose name
  merely ended with the same text ("my-notes.md" for "notes.md") was reported against the file you
  just added, hiding the fact that it was ingested fine.

- **Running out of re-plans no longer abandons steps that are ready to run.** The re-plan budget
  bounds *redrafting*; steps already drafted and concrete now execute instead of being reported as
  never run.

- **An always-allow grant no longer switches off the plan-review revocation lock.** Answering `a`
  at the approval gate drops a tool to the auto-approved tier — which is what auto-approval means
  — but the revocation lock and effect authorization were reading that same live tier to decide
  whether a tool can change state at all. One keypress at an unrelated prompt therefore made a
  step you had deleted at plan review runnable again, unprompted, for the rest of the turn (and,
  under `grant_scope: persist`, for every session after it). Both now read the tier a tool
  *declares*, which no grant or `/policy risk` edit can lower.
- **Removing one vague step at plan review no longer cancels every other action.** A dropped write
  whose wording named no file revokes writes for the turn — that part is deliberate, since a
  redraft can just omit the filename — but it was also cancelling unrelated effects the plan
  kept, including memory writes and MCP calls. It now covers exactly the effect class that
  produced it (file writes and shell), and a step it stops is disclosed as collateral of the
  removed write rather than as an action you removed yourself. Rewording such a step (or
  sharpening it into a concrete path) is again recognized as a relabel, not a removal.
- **Confidence coloring now grades at your model's calibrated exit threshold.** The measured value
  from the shipped table, `/confidence tune`, and `/confidence set <enter> <exit>` was resolved,
  displayed by `/confidence`, and then discarded by the renderer, which fell back to a derived
  1.5× of the enter threshold. Marked spans on a calibrated model may now end slightly later,
  which is the measured behavior the table always described.
- **File sizes and digest fragments are no longer treated as figures the answer must state.** A
  `run_shell` step counted as the turn's calculation, so numbers incidental to its output (an
  `ls -l` size, a hash) became values the answer was required to report — spending a corrective
  regeneration steering toward them and then disclosing them as "the plan's own calculation
  step". Only `calculate` creates that obligation now; `run_shell` still counts as a computation
  for deciding whether the plan needs one at all.
- **The streaming answer no longer re-flows when it lands.** The live tail indented each physical
  line while the finished answer pads every visual row, so soft-wrapped continuation lines were
  two columns wider and every break moved at the handoff on wide terminals. Both now render
  through one measure and one indent. (Markdown still formats at finish — that is formatting, not
  re-flow.)

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

- **Freezing an answer no longer makes it jump around before you can type.** Pressing Esc used
  to re-render the same answer three more times in three different layouts — a gutter-indented
  tail, then the editor's pre-filled copy at column 0, then the final markdown — so the text
  moved under the cursor twice on the way to the edit. The freeze now prints one line saying what
  it captured (`✂ frozen — 412 chars, 3 low-confidence runs`) and hands straight to the editor,
  which is indented to match everything else on screen. Whenever the wizard fallback runs instead
  — no prompt_toolkit, or an editor that failed to open — the body still prints, at the same
  indent and width as the streamed answer, since there it is the only way to see the text you are
  being asked to cut from.

- Assorted readability fixes: a hot CPU/RAM gauge no longer shouts in the same bold red as
  `⚠ GATE OFF` (load is not risk); the air-gap "blocked" marker uses a glyph that actually takes
  its colour and fits the trace rail, in both the trace and the receipt; a long model tag in
  `/models` no longer pushes every later column out of line; diff hunk headers line up with the
  diff body; the always-allow prompt says the command it echoes is whitespace-normalized rather
  than calling it "this exact command"; and the startup splash no longer prints into pipes.

- **Low-confidence phrases in an answer are marked with dim underline instead of red.** Red
  already means "this failed" everywhere else in the interface, and being a pure colour it
  disappeared entirely under `NO_COLOR=1` and on monochrome terminals — leaving the receipt
  reporting `◌ 3 uncertain spans` with nothing marked to look at. The new marking carries no
  colour, so it survives, and stays distinct from the cyan used for your own edits.

- **The approval frame stays intact when you press `e`, `a` or `s`.** The closing `┗━` was part
  of the prompt line, so the explanation, the always-allow disclosures and the per-call prompts
  all printed *below* the corner and visibly broke the box open. The frame now closes once, after
  the decision, and names it (`┗━ 1 of 3 approved`).
- **The gate's key legend is always visible**, not just after you mistype — `a` widens the gate
  and `s` splits a batch, and neither should be discovered by accident. The header names the
  batch size and each call carries its position (`2/3`), matching the per-call prompts.

- **`/trace invoke` and `/trace context` sit in the trace rail** like every other view, instead
  of starting at a bare indent of their own.
- **A message the trace had to cap now says so accurately, and says it where you can see it.**
  The `(+N chars)` marker was computed against the wrong constant (reporting a number that was
  simply wrong), was appended to the message body and then clipped off along with it — for
  exactly the long messages it described — and was suppressed under `--full`, where a silently
  capped message matters most. It is now its own line, always shown, with the right number — on
  the model's reply as well as its inputs, so a long answer is never presented as the whole of
  what the model said.

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
