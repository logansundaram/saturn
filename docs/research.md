# Saturn — what to build next

_2026-10-01. Two things in one file. Part 1 ranks every improvement still open in `pivot.md`,
`engine.md` and `advantages.md` and points at the implementation plan written for each of the
most promising. Part 2 is an outside survey — the agent literature and what the large vendors
and the open-source personal agents ship — turned into further ideas, ranked, with a sketch for
each. Companion to `pivot.md` (the goal and the two knives), which this file does not replace._

How to read the evidence. Claims about Saturn's own code were checked in the working tree on
2026-10-01 and say so. Outside sources are marked: **[F]** the page was fetched and read,
**[A]** only the abstract was read, **[S]** seen in search results only. Numbers quoted from
vendors are the vendors' own. Nothing here has been run against a model: every idea is a guess
until `benchmark.py --loop` and the trust benchmark say otherwise.

---

## Part 1 — what is left, ranked

Ranked by the pivot's two knives (would a person hand this to Saturn on a Tuesday; what does it
cost the chat turn) and by how much of the goal each closes per day of work. The eighth item
came out of Part 2. All eight plans are written (2026-10-02) and none is built. Each opens with
its own Design section and "Assumptions made without asking", and carries merge notes for the
siblings that touch the same files. Except for plan 7, each plan's code was dry-run in a
scratch copy of the tree, and the full offline suite passed there. The benchmark tasks were not
run: they need Ollama.

| # | Improvement | From | Effort | Why it ranks here | Plan |
|---|---|---|---|---|---|
| 1 | **Learn from what the user says, without the click** | pivot #4 | 2–3 d | The largest gap against the goal: "know the user" is leg one and today the agent learns only through a review click. The research agrees and narrows the design: auto-accept only what is provably the user's own typed words; everything else keeps the queue (Part 2, memory). | `2026-10-01-auto-memory-from-user-statements.md` |
| 2 | **Skills: user-authored procedures as markdown** | pivot #8 | 2–3 d | The largest gap in "yours to shape", and the unit every other agent has converged on (a markdown file with a description; body loaded on use). Routines, triage and the weekly review in Part 2 all become skills once this exists. | `2026-10-01-skills.md` |
| 3 | **A launch brief** | pivot #6 | 1 d | Turns "open the terminal" into "here is your day" with the readers that already exist. Deterministic: no model call, nothing the model can be steered by. | `2026-10-01-launch-brief.md` |
| 4 | **A first-run interview** | pivot #5 | 1 d | The second turn already knows who it is talking to. Cheap, no model call, and it compounds with #1. | `2026-10-01-first-run-interview.md` |
| 5 | **Loop guards**: a hygiene budget, a failure-aware final pass, replay of a repeated read | engine 1–3 | ½–1 d | Robustness first: the 4b's 14-pass, 7-bounce spiral that ended in "I've updated my memory" is the worst behaviour on record, and each guard costs the clean path nothing. | `2026-10-01-loop-guards.md` |
| 6 | **The observation path**: a relevance-aware clamp, then a budgeted prompt projection | engine 4–5 | 2–3 d | `file_long_middle` fails on every tier and on both of today's 4b runs; long errands on a 32k window push the system prompt off the front. The literature is unusually clear that deterministic masking is the right tool (Part 2, context). | `2026-10-01-observation-budget.md` |
| 7 | **A question is an answer**: delete the `ask_user` interrupt | engine 6 | 1–2 d | Removes a tool, an interrupt, a hack (`ASK_ALONE_TEXT`), a headless special case and a grader ambiguity. Lowest user-visible value of the seven, highest deletion. | `2026-10-01-question-is-an-answer.md` |
| 8 | **Terminal escape sanitising** | Part 2, trust | ½ d | Built 2026-10-03. Was not on the backlog: a hole found while checking the research against the code. | `2026-10-01-terminal-escape-sanitising.md` |

**Order to build.** 8 (done) and 5 first (a day together, both make the product more honest), then 1
and 4 as the "know the user" pair, then 6 (clamp before projection), 2, 3, and 7 last. Plans 5,
6 and 7 all edit `nodes/agent.py` and are written to land in that order; each carries merge
notes for the other two.

**Before any of the engine plans is measured:** make "three runs per change" (engine 13) the
benchmark's default and add a pass^k column (all k runs pass) beside the pass rate. Every plan's
last task is a before/after benchmark, and a delta of one task is noise today. tau-bench's
finding that a model at ~61% single-run success drops to ~25% across eight runs
([S] arxiv.org/abs/2406.12045) is the reason consistency is the number to watch on a 4b.

### The rest of the backlog, with a verdict

| Item | From | Verdict |
|---|---|---|
| Drafts in the user's own voice | advantages §5.3 | **Next plan to write.** Few-shot `draft_mail` / `reply_mail` with the last few messages the user sent to that recipient. Every email product leads with it (Part 2). Needs one bulk Sent-mailbox reader; costs the chat turn nothing. |
| The catalog's shape | pivot loop 9 | **Now has evidence.** The catalog is 42 tools and ~6k tokens of schema (checked 2026-10-01; 43 since `delete_file`, 2026-10-02), and two of the 4b's five stable misses today are wrong-tool picks (`list_mail` for "what can you do with my email?", `draft_mail` for "send a text"). See Part 2, engine 1. |
| Sharper tool descriptions; `calculate` takes the whole expression | engine 7–8 | An afternoon each. Do them with schema examples (Part 2, engine 3). `robust_no_tool` also needs its expectation revisited: Saturn can send a text now. |
| Concurrent tool batches | engine 9 | Worth a day once ledger events carry a call id. Run only the read-only calls of a batch concurrently and keep Apple events serial. LLMCompiler reports up to 3.7× latency speedup from parallel calls ([F] arxiv.org/abs/2312.04511). |
| `send_mail` | pivot #13 | Wait. `reply_mail` already leaves the user one keystroke from sent, and a second always-asks chokepoint is real work. Revisit now that `send_message` has run for real (first send 2026-10-02). |
| Script tools (`~/.saturn/tools/*.sh`) | pivot #9 | Wait. `run_shortcut` covers the Mac-native half and skills cover procedures; build when a user asks for it by name. |
| Index everything | advantages §5.1 | No. `pivot.md` already says stop investing in RAG, Spotlight search shipped today, and the memory benchmarks in Part 2 favour plain files plus search over a store. |
| Free background compute | advantages §5.2 | Becomes **routines** (Part 2, product 1), not an indexer. |
| Ambient awareness | advantages §5.5 | No. Every shipped version of it drew a backlash (Part 2, what not to copy). `@clipboard`, `read_browser_tab` and `finder_selection` are the user-initiated form and they exist. |
| A wall-clock budget | engine 10 | Stays declined; the hygiene budget in plan 5 removes its main cause. |
| A phantom-action guard | pivot loop 5 | No. The benchmark shows zero phantoms on both tiers. |
| Compress oversize observations with a model call | pivot loop 6 | No — replaced by plan 6. A compressor is a second reader of injected text, and summaries did not beat masking where it was measured. |
| Pick `auto`'s think policy; grade the recorded reasoning | engine 11–12 | Done 2026-10-04: `auto` thinks before acting (`act`), picked on the loop benchmark over six other modes. Open: item 12, the two tasks first-move thinking breaks, and the avenues in engine 11a (a small trained classifier of simple vs complex requests, a learned per-step router, a think tool). |

---

## Part 2 — what the outside world suggests

Three surveys were run on 2026-10-01: the research literature on agent memory, context and
small-model tool calling; the security literature and vendor practice on tool-using agents;
and what the shipping products do. What follows is what survives contact with Saturn's
constraints — local models only, one loop, one model call for a chat turn, a stable cached
prefix, every action gated and logged.

### What the research changes in the seven plans

- **Auto-memory (plan 1).** Persistent-memory injection is a demonstrated attack, not a
  hypothetical: SpAIware had a web page drive ChatGPT's memory tool to store instructions that
  then exfiltrated every later conversation ([F]
  embracethered.com/blog/posts/2024/chatgpt-macos-app-persistent-data-exfiltration/); MINJA-style
  attacks report >95% injection success in idealised conditions ([F] arxiv.org/abs/2601.05504);
  "provenance laundering" shows a summarisation step rewriting external content as user history
  ([A] arxiv.org/abs/2607.29167). Saturn's compaction-summary candidates are exactly that path.
  So the plan's rule is deterministic — a fact lands without a click only when its words are
  covered by what the user typed this turn — and candidates from compaction summaries or from
  turns that read untrusted content keep the queue.
- **Standing rules must load every turn (plans 1 and 4).** Only `user`, `commitments` and the
  last five `memo` entries load every turn; `negative` loads by token match. "Never schedule
  anything before 10am" would not load for "book a dentist appointment". The most-reported
  agent incident of the year was this shape: an instruction to confirm before acting was lost
  when the context was compacted, and the agent deleted an inbox ([F]
  sfstandard.com/2026/02/25/openclaw-goes-rogue/). Both plans were told to put rules where they
  are in front of the model on every turn.
- **The observation path (plan 6).** On SWE-bench Verified, masking old tool outputs matched
  LLM summarisation at about half the cost, on Qwen3-32B among others (raw 17.0%, masking 15.0%,
  summary 16.0%), and summaries made runs ~15% longer ([F] arxiv.org/html/2508.21433).
  Anthropic's context editing keeps the three most recent tool results and clears in large
  steps *because* clearing breaks the cached prefix ([F]
  platform.claude.com/docs/en/build-with-claude/context-editing). Manus's rule is that
  compression must be restorable: drop a page body only when a URL or path remains ([F]
  manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus). And "context
  rot" was measured on Qwen3-8B and 32B: accuracy falls with input length even on trivial
  tasks ([F] trychroma.com/research/context-rot) — a reason not to simply raise the clamp with
  the window.
- **Skills (plan 2).** Keep the Claude Code file shape and its one useful safety key,
  manual-only skills ([F] code.claude.com/docs/en/skills). No registry and no install-from-URL:
  one public skill registry was found to hold hundreds of malicious skills, macOS infostealers
  among them ([F] unit42.paloaltonetworks.com/openclaw-ai-supply-chain-risk/).
- **The brief (plan 3).** The shipped briefs are finite and bucketed (missed / needs attention /
  can wait), say nothing when there is nothing to say, and the complaint users make about
  proactive assistants is wrong dates and time zones ([F] saner.ai/blogs/poke-reviews). The
  plan carries a line cap, silence on empty, and date-bucket tests.

### Trust — where the thesis can be made stronger

Saturn reads private data, ingests untrusted content and can act: all three legs of the "lethal
trifecta" ([F] simonwillison.net/2025/Jun/16/the-lethal-trifecta/). Meta's "Rule of Two" says
a session with all three must not run unsupervised ([F]
ai.meta.com/blog/practical-ai-agent-security/); the gate is that supervision, and it is the
right architecture. The useful finding is *which* of Saturn's defenses the literature says
hold. "The Attacker Moves Second" broke twelve published detection-style defenses with adaptive
attacks, most at above 90% success ([F] arxiv.org/abs/2510.09023). Detectors fail; deterministic
mechanisms hold. Saturn's URL hold, `ALWAYS_ASKS` and the gated `remember` are the second kind.
The regex scanner in `trust/quarantine.py` is the first kind, and one control still hangs on it.

**T1. Strip terminal escape sequences (½ day) — built 2026-10-03** (`textutil.visible_controls`,
applied at the source in `nodes/tools.py` and at the sink in `SafeConsole`). What it closed,
checked 2026-10-01: Rich's `Text` and `Markdown` pass `ESC ] 52` (clipboard write) and `ESC [ 2J`
through to the terminal; `textutil.clip` and the rail's `_leaf` keep them; `fmt_args` does not.
A web page, email or file that carries escape bytes could therefore reach the screen through the
rail's result preview or through the model echoing them, and cursor and erase sequences could
rewrite what the user sees — in a product whose promise is that what you see is what happened.
The published attack is "Terminal DiLLMa" ([F]
embracethered.com/blog/posts/2024/terminal-dillmas-prompt-injection-ansi-sequences/). Not
verified end to end in a live terminal; `tests/test_terminal_safe.py` pins it offline.

**T2. Escalate on provenance, not on a regex match (1–2 days).**
Today the quarantine escalation arms only when `scan()` matches one of eight phrasings
(`quarantine.flag` → `_GATE_PENDING`), for one batch, and resets each turn. An injection that
is paraphrased arms nothing. With the default `auto_approve: read_only` every acting call asks
anyway, so the exposure is exactly the users who loosened the gate: an always-allow grant, a
raised tier, `--yolo`. `nodes/approval.provenance` already computes, deterministically and
over the whole conversation, whether untrusted content has entered it (the URL hold uses it).
The change: once content from an *external* source (web, mail, messages, an MCP server, a
browser tab, a shared note) is in the conversation, grants and raised tiers stop applying to
calls that send or cannot be undone, and the gate says why. The regex stays as the note that
explains a prompt, not as the trigger. The open design question is local files: `read_file` is
untrusted too, and treating every file read as taint would make always-allow useless for the
edit-test loop — so workspace files should taint only outbound calls, not snapshotted local
writes. Decide that with the trust benchmark, and add paraphrased-injection probes to it:
today's probes plant a document written in the canonical phrasings and grade whether the
scanner *flagged* it (`benchmark.py`, `INJECTION_DOC_BODY`, `grade_injection`) — they measure
the detector on the inputs it was written for, not whether an attack that avoids those
phrasings reaches an action. The probe that matters asserts the gate was reached.

**T3. Show argument provenance at the gate (1 day) — the Messages half shipped 2026-10-02.**
`send_message` and `read_messages` handles are checked now (`quarantine.handle_hold`: a handle in
nothing the user typed and nothing a tool returned is refused before the gate, and the gate
names the contact card it came from); `draft_mail`, `reply_mail` and `create_calendar_event`
attendees are open.
CaMeL ([F] arxiv.org/abs/2503.18813) and FIDES ([F] arxiv.org/html/2505.23643) both track
where each value came from and check it at the tool call; FIDES reports injections on its
benchmark going from 156 to 0 with flow policies. The full interpreter is too heavy for a 9b,
but the URL hold is already the one-argument version. Extend it to recipients: for
`send_message`, `draft_mail`, `reply_mail` and `create_calendar_event` attendees, compute
whether each address or number appears in text the user typed or on a Contacts card the user
named, and say so at the gate ("this number is not in anything you typed and not on Sam's
card"). A human who approves 93% of prompts ([F]
anthropic.com/engineering/how-we-contain-claude) needs the one unusual prompt to look unusual.

**T4. Pin MCP tools by hash (½ day).**
A server can change a tool's description after the user approved it (the "rug pull"), and
descriptions are instructions the model reads ([F]
invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks). Saturn ignores a
server's self-declared risk tier and prefixes names; no hashing was found in
`tools/mcp_client.py`, and MCP descriptions land in the cached system prefix. Hash each tool's
name, description and schema at first connect into `permissions.json`; on a change, hold the
server's tools out of the catalog until `/mcp` shows the diff and the user accepts it.

**T5. Sandbox what the ledger cannot see (2–4 days).**
`run_shell`, `run_shortcut` and stdio MCP servers are recorded `UNTRACKED` because Saturn cannot
see inside them. Claude Code and Codex CLI both run shell commands under macOS Seatbelt
(`sandbox-exec`): writes limited to the workspace, network off or through an allowlisting
proxy; Anthropic reports 84% fewer permission prompts from it ([F]
anthropic.com/engineering/claude-code-sandboxing; [F] learn.chatgpt.com/docs/agent-approvals-security).
For Saturn the gain is different and bigger than fewer prompts: `UNTRACKED` becomes *contained*,
air-gap becomes enforced for the shell rather than asked about, and a shell command can no
longer write `hooks.yaml`, `config.yaml` or `permissions.json` (the file tools refuse them
today; the shell does not). `sandbox-exec` is deprecated-but-working on macOS; treat this as a
spike first.

**T6. A fetch budget after taint, and a tamper-evident ledger (½ day each).**
OpenAI's URL defense — fetch only addresses the user typed or the index already knows — was
bypassed by spelling data out across many pre-existing URLs ([F]
embracethered.com/blog/posts/2026/data-exfiltration-mitigation-paper-by-openai/). The URL hold
has the same shape: verbatim presence passes. A per-turn cap on distinct hosts fetched after
untrusted content entered the conversation closes the slow leak. `web_search` queries are the
same channel and have no hold at all. Separately, chaining each egress and gate record to the
hash of the one before it, with `/policy verify`, makes "every byte that leaves is on the
ledger" checkable rather than asserted ([S] arxiv.org/abs/2609.01931).

### Product — what the shipping agents do that Saturn does not

**P1. Routines: scheduled runs with a silent inbox (3–4 days, after skills).**
The one feature every vendor has copied from every other: ChatGPT Tasks and Pulse, Gemini
scheduled actions, Claude's scheduled tasks ([F] code.claude.com/docs/en/desktop-scheduled-tasks),
OpenClaw's heartbeat ([F] github.com/openclaw/openclaw/blob/main/docs/gateway/heartbeat.md),
Hermes cron ([F] hermes-agent.nousresearch.com/docs/user-guide/features/cron). "Every weekday
at 8, tell me what needs a reply." Saturn has the pieces: the launchd seam in `notify/`,
headless `-p`, and skills as the prompt. The designs worth copying: a fresh session per run;
*silence when there is nothing to say* (a `[SILENT]` reply is dropped); exactly one catch-up
run after the laptop slept, with skipped runs listed and why; a routine cannot create a
routine. The trust design is where Saturn can be better than all of them. An unattended run
has no human at the gate, so by the Rule of Two it may hold at most two legs: headless already
denies every gated tool (no acting), and a routine that reads mail must also run air-gapped,
because `web_search` and `web_extract` are read-only tools that still send. Results land in an
inbox the next launch shows in the brief — nothing is pushed, nothing acts, every run is in the
trace and replayable. This is also the honest form of "free background compute".

**P2. Triage, then drafts, never send (2 days, mostly a skill).**
Fyxer, Superhuman and Shortwave converge on one loop: label incoming mail (to respond / FYI /
notification / marketing), pre-write drafts for the first bucket, and let the human send ([S]
fyxer.com/blog/how-fyxer-organizes-your-inbox-by-priority). No mainstream product auto-sends,
which is Saturn's gate-first design reached from the other side. With skills shipped this is a
bundled `inbox-triage` skill over `list_mail` / `reply_mail` / `update_mail`, plus one
deterministic reader the model should not be trusted to compute: **awaiting reply** — mail the
user sent that got no answer in N days. It feeds the brief and the `commitments` layer.

**P3. Drafts in the user's voice (1 day).** See the backlog table. The mechanism the products
describe is per-recipient tone learned from Sent mail; the cheap local version is three recent
sent messages to the same person in the draft pass's prompt, fenced as the user's own text.

**P4. An incognito session (½ day).**
Claude and ChatGPT both added sessions that neither read nor write memory, after complaints
about unrelated context bleeding together ([F] support.claude.com/en/articles/11817273; [S]
simonwillison.net/2025/May/21/chatgpt-new-memory/); Google's own launch post warns of
"over-personalization". `saturn --incognito` (and `/incognito`): no memory block in the
grounding, no `recall`, no review candidates queued, no auto-learn. One flag, and it is the
control that makes plan 1 comfortable to leave on.

**P5. A memory receipt (½–1 day).**
Gemini names which personal sources an answer used and offers "retry without personalisation"
([F] blog.google/innovation-and-ai/products/gemini-app/personal-intelligence/). Saturn already
appends a Sources receipt; add the ids of the memory facts that were loaded for the turn, so
`/memory why <n>` is one step away and a wrong answer caused by a stale fact is traceable.

**P6. An "Ask Saturn" entry point outside the terminal (½ day, mostly docs).**
macOS Tahoe runs Shortcuts from Spotlight and on events — a time, a file saved to a folder
([F] apple.com/newsroom/2025/06/macos-tahoe-26-makes-the-mac-more-capable-productive-and-intelligent-than-ever/).
A shipped Shortcut that wraps `saturn -q` gives a global hotkey and event triggers with no
daemon and no new code path: headless already denies gated tools.

**P7. Skills, second version (after plan 2 has been used).**
Typed parameters and a success check (Goose recipes, [F]
goose-docs.ai/docs/guides/recipes/recipe-reference/); "save what we just did as a skill",
drafted into a pending folder the user accepts (Hermes, [F]
hermes-agent.nousresearch.com/docs/user-guide/features/skills) — the request is already in
`dogfood.md` §13, and it fits the review-queue pattern Saturn has for memory.

**P8. Search over past sessions — a decision to revisit, not a recommendation.**
The products keep two memories: a small curated profile and full-text search over history
(Claude's chat search, Hermes's `session_search` over SQLite FTS5). Saturn cut exactly this on
2026-09-29 and 2026-09-30 (`recall_runs`, then `/trace search`) on the argument that "what did I decide" is
memory's job. The research is on the other side of that call, mildly: LongMemEval found that
replacing raw history with extracted facts alone loses information ([F]
arxiv.org/html/2410.10813), and Letta's file-and-grep agent beat dedicated memory systems on
LoCoMo ([F] letta.com/blog/benchmarking-ai-agent-memory/, vendor-reported). The test is in
`dogfood.md` §16: "what did we decide about the lease yesterday?" in a new session. If it
fails once auto-memory has shipped, bring search back as a read-only tool whose output is
untrusted; if it passes, the cut was right.

### Engine — what the literature says about small models

**E1. The catalog is the next engine problem.** 42 bound tools (43 since 2026-10-02), ~6k tokens of schema, and two
of today's five stable 4b misses are wrong-tool picks. Tool-selection accuracy falls steeply
with catalog size (RAG-MCP: 13.6% with everything bound, 43.1% with a retrieved subset, [F]
arxiv.org/abs/2505.03275), but swapping tools per turn breaks the cached prefix — Manus's
answer is to keep every schema bound and constrain the *choice* instead, with consistent name
prefixes ([F] manus.im, above). Three steps, cheapest first: (a) `tools.disabled` in config —
a user without Messages or Shortcuts drops those schemas for good, and the benchmark can
measure a trimmed catalog today; (b) name tools by domain prefix (`mail_list`, `mail_read`,
`calendar_create`) so the model's first tokens pick the domain; (c) only if (a) shows the
gain is real, the pivot's "domain tools with an action enum". Whether Ollama exposes
per-request tool-choice or grammar constraints alongside tool calling was not verified; check
before designing on it.

**E2. Memory retrieval that survives paraphrase (1 day).** Matching is token overlap, so
"my landlord" does not find a fact that says "lease". LongMemEval measured +9.4% recall from
adding extracted keys to the index entry at write time ([F] arxiv.org/html/2410.10813). At
accept time, store a few alias tokens in the bullet's metadata token (no model call if they
are typed by the user at review; one background call otherwise) and rank matches by overlap
plus the `used=` and `n=` counts already stored — the recency and frequency terms of the
Generative Agents scoring ([F] ar5iv.labs.arxiv.org/html/2304.03442).

**E3. Schema examples and errors that say what to do next (an afternoon per tool).**
Anthropic reports parameter accuracy going from 72% to 90% when tool definitions carry
examples ([F] anthropic.com/engineering/advanced-tool-use, vendor's internal number). Add one
example call to the description of each tool the loop benchmark shows being misused, and make
each `ToolError` end with the next move ("…not found. Call `find_files` with part of the
name."). This is engine items 7–8 with a method.

**E4. Idle-time memory upkeep, as proposals (1–2 days, after P1).** "Sleep-time compute" moves
consolidation off the latency path ([F] arxiv.org/abs/2504.13171;
[F] docs.letta.com/guides/agents/architectures/sleeptime). For Saturn: a routine that proposes
duplicates to merge and `replaces=` supersessions into the review queue. Proposals only — it
is a summarising step, and summarising steps are where provenance gets laundered.

### What not to copy

- **A skill or tool marketplace.** The supply-chain record is already bad (plan 2, above).
- **Ambient capture.** Recall shipped, drew an outcry and became opt-in; Rewind's capture was
  switched off after its acquisition ([S]). The value is real and the trust cost is the product.
- **A messaging front end.** Being reachable in iMessage is the feature users of Poke praise
  most, and an inbound message channel that can issue commands is an injection surface with
  the user's phone number as the address. `read_messages` stays a reader.
- **Code mode.** Letting the model write a script that calls many tools helps strong models
  and hides many actions behind one gate prompt and one ledger line ([F] arxiv.org/abs/2402.01030
  for the gain; the cost is Saturn-specific).
- **A classifier that approves actions.** Claude Code's auto mode reports 17% false negatives
  on real overeager actions with a frontier model reviewing ([F]
  anthropic.com/engineering/claude-code-auto-mode). A 9b reviewer would be worse, and a
  detector is the kind of defense that breaks.
- **A knowledge graph or vector store for memory.** Full-context and file-plus-search
  baselines match or beat the dedicated systems on the public benchmarks (above); Saturn's one
  markdown file is the right store. Spend on retrieval (E2), not on storage.
- **Summarising tool output with a model.** Plan 6, above.

### Ranked, across Part 2

By value per day, with the trust items first because they protect the thesis everything else
rests on.

1. T1 — escape sanitising (built 2026-10-03)
2. T2 — escalate on provenance, with paraphrased-injection probes in the trust benchmark
3. P3 — drafts in the user's voice
4. P4 — incognito
5. E1(a) — `tools.disabled`, and measure the trimmed catalog (the switch shipped 2026-10-05 as toolkits: `/tools off <toolkit>`, `benchmark.py --off`; the measurement is open)
6. P1 — routines (after skills)
7. P2 — triage skill and the awaiting-reply reader (after skills)
8. T3 — argument provenance at the gate (mail and calendar; Messages shipped 2026-10-02)
9. T4 — MCP hash pinning
10. E3, E2 — schema examples; alias tokens for memory matching
11. P5, P6 — the memory receipt; the Shortcut entry point
12. T5 — the Seatbelt spike
13. T6, E4, P7 — fetch budget and hash chain; idle-time proposals; skills v2

### What was not verified

- Sources marked [S] and [A] above; the OpenAI primary pages for Pulse, Tasks and link safety
  returned 403 and are cited through secondary write-ups.
- T1 end to end in a live terminal; whether `arg_tail_rejects` screens network-capable
  programs such as `nslookup` under a prefix grant (DNS exfiltration through allowlisted
  commands is a published attack, [S]
  embracethered.com/blog/posts/2025/claude-code-exfiltration-via-dns-requests/ — the
  interpreter table in `trust/policy.py` lists `ssh`, `scp` and `rsync` but no DNS or HTTP
  client, and `curl` under a granted prefix is screened only by its arguments).
- Answered since from the code (`tools/web._fetch`): `web_extract` follows redirects one hop at
  a time — each new host is air-gap checked and recorded, and a hop from a public page to a
  private address is refused; the composed-URL half of the hold is checked on the argument
  only, not per hop.
- Any effect on Saturn's benchmarks. Nothing in this file has been run.
