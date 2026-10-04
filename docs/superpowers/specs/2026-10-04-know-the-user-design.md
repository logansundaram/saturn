# Knowing the user — how facts about the user arrive, are stored and are read

Date: 2026-10-04. Status: **draft for review — nothing built.** This is the one design above
three plans that were written before it (2026-10-01): `../plans/2026-10-01-auto-memory-from-user-statements.md`
(pivot #4), `../plans/2026-10-01-first-run-interview.md` (pivot #5) and
`../plans/2026-10-01-launch-brief.md` (pivot #6). Each plan keeps its own design section; this
spec says what the three are together, what a survey run on 2026-10-04 changes in each, and
the order to build them in. Where this spec and a plan disagree, the plan is the one to amend
("Changes to the plans" lists every amendment).

## The goal

By its second turn Saturn knows who it is working for, and it keeps learning from what the
user says without a click per fact — and nothing the user did not say can plant a memory.

Fixed: one markdown memory file, local models (4b–9b), one model call for a chat turn, the
gate in front of every write the user did not make.

**It is working when:**

1. On a fresh install, a user who accepts the offer has their name, what they do and their
   standing rules in the always-loaded half of the grounding on the next turn.
2. "I'm vegetarian" typed in a conversation that has read nothing from outside is saved with
   no prompt, and one dim line says so and how to undo it.
3. The same sentence, after a web page, a mail or a file has entered the conversation, faces
   the gate.
4. A password, a card number or a government ID is never written to the memory file by any
   Saturn path.
5. The session review's model pass is never shown a tool result or a compaction summary.
6. A plain chat question is still one model call, and its prefix is re-primed only when a
   fact in an always-loaded layer is written.

## What the survey found

Three searches on 2026-10-04: the research literature, what the large vendors ship, and what
comparable products do. Tags as in `docs/research.md`: [F] the primary source was fetched and
read, [A] abstract only, [S] secondary. Every fetch went through a summarising tool; four
claims this design rests on most were fetched a second time and matched (marked ✓).

| Finding | Source | What it changes here |
|---|---|---|
| Agents solve 15–24% of preference-bound tasks, 83–97% when the preference is supplied; six of eight frontier models almost never ask, and forced asking has 6–10% precision | ATRBench, [F] arxiv.org/abs/2605.28108 | The interview is not handed to a 4b–9b model. It stays a fixed, deterministic set of questions. |
| Every measured elicitation gain comes from adaptive questions; nothing tests a fixed questionnaire, and nothing measures how many questions people tolerate | GATE, [F] arxiv.org/abs/2310.11589; Pep, [A] arxiv.org/abs/2602.15012 | No evidence for five questions over three. The count is a product call (below). |
| Zero-shot preference following is under 10% after 10 turns, 7–8B models included; a plain reminder beat reasoning tricks | PrefEval, [F] arxiv.org/abs/2502.09597 | Standing rules live in an always-loaded layer (the auto-memory plan already decides this). |
| One human-readable 2,048-token profile beat full-context frontier models (the writer was trained) | PersonaMem-v2, [F] arxiv.org/abs/2512.06688 | One markdown file is the right store. No graph, no vector store. |
| Qwen3 0.6B–8B answer with the stale stored value 92–100% of the time when memory disagrees with current evidence; with dates and provenance on the facts accuracy is 0.90 (4B) and 0.95 (8B); with the conflict resolved before the read, 1.00 | The Memory Trust Gap, [A ✓] arxiv.org/abs/2609.01852 (numbers past the abstract: [F], not rechecked) | Two conflicting facts must not sit side by side unseen, and the model is shown each fact's date. |
| A deterministic supersession rule with a 7B model reaches 0.95–1.00 on evolving facts; embedding similarity cannot tell a contradiction from a duplicate (AUROC 0.59) | MemStrata, [A] arxiv.org/abs/2606.26511 | Conflicts are surfaced by a deterministic rule and resolved by the user, not judged by a model. |
| When a claim the user made was saved and readable, later tasks failed 71.9% of the time against 45.0% unsaved; agents rewrote the claim as a stable preference, fact or procedure in 51.4% of runs | PASB, [A ✓] arxiv.org/abs/2607.10526 | The provenance gate passes exactly this case. Facts keep the user's words and date, and the block says what outranks it. |
| Every model over-infers 35–49% of its personalised claims; Qwen3-8B is the worst at 48.7% | MirageBench, [F] arxiv.org/abs/2608.04570 | No background consolidation, no inferred fact without a review accept. |
| Payloads in mail and calendar invites reach memory at about 98% | GhostWriter, [F] arxiv.org/html/2607.06595v1 | No profile fact is ever derived from Mail or Calendar; the brief never enters the prompt. |
| 97.8% of 10,134 auto-extracted entries were junk; over half restated system-prompt content; one hallucinated "User prefers Vim" was re-extracted 808 times from recalled memory | mem0 issue #4573, [F ✓] github.com/mem0ai/mem0/issues/4573 | The review's model pass reads only the conversation's own words, never what Saturn injected or fetched. |
| No vendor ships a first-run interview; explicit input is an optional box | OpenAI [S], Anthropic [F], Google [F], Microsoft [F] | The interview is offered, never started on its own. |
| OpenClaw removed the user questions from its first-run ritual in mid-2026: "The user's request always comes first", "Do not turn them into a questionnaire or a long biography" | [F ✓] raw.githubusercontent.com/openclaw/openclaw/main/docs/reference/templates/BOOTSTRAP.md | Same. |
| Vendors save silently or notify after, never confirm before; a per-save confirmation drew friction complaints, an inbox of proposed patches did not | Claude [F] support.claude.com/en/articles/11817273; Gemini CLI [F] github.com/google-gemini/gemini-cli/issues/19967, geminicli.com/docs/cli/auto-memory/ | The split is right: no click for what the user typed, a queue for what was inferred. |
| Anthropic never saves government IDs, financial account numbers or immigration status, "even if you ask" | [F] support.claude.com/en/articles/11817273 | A never-save screen for secrets. |
| Codex skips memory generation for a chat that used the web or MCP; OpenClaw excludes injected content from memory candidates | [F] learn.chatgpt.com/docs/customization/memories; [F] github.com/openclaw/openclaw/pull/114819 | Provenance gating is now common practice; Saturn's "provably typed" rule is the strictest of the three. |
| Memory import by a pasted self-export | Anthropic [F] support.claude.com/en/articles/12123587; Google [S]; Mistral [F] mistral.ai/news/memory/ | `/memory import`, through the review screen. |

Not found: a controlled study of an up-front questionnaire against gradual profiling; any
measurement of a word-coverage gate like the auto-memory plan's; any study of launch briefs.

## The design

### How a fact arrives decides everything else

| How it arrives | Approval | Stored as |
|---|---|---|
| The user types it in a turn, in a conversation that holds nothing from outside | none — an after-answer line and `/memory forget n` | `by=user src=said` |
| The user answers the interview, or types `/memory add` | none | `by=user` (`src=setup:<key>` for the interview) |
| The model proposes it (session review, compaction summary), or the user imports it from another assistant | the review screen, one fact at a time | `by=inferred` (`src=import` for an import) |
| A `remember` call while web, mail, file or shell content is in the conversation, or one whose words the user did not type | the gate, with the reason | `by=user` on a yes |

No row writes a fact unattended from text the user did not type. A stored fact never
authorises an action: the gate asks for every acting call whatever memory says.

### 1. Learning in the flow (the auto-memory plan, with five amendments)

The plan's mechanism stands: the model calls `remember`, `core/auto_memory.why_not` is the
deterministic word-coverage check, the approval node exempts a passing call, the after-answer
line names the fact and its undo. The amendments:

**A1. The memory block says what it is and what outranks it.** Today the stable block is headed
"what the user asked me to remember and what I learned", and `_context_line` prints a date
only for `memo` and `commitments`. Change both: every fact carries the day it was written, a
`by=inferred` fact is tagged `[inferred]`, and the header reads, in substance, *what the user
told me and the day they said it; a later date outranks an earlier one; what the user says in
this conversation, and what a tool returns now, outrank all of it*. This is the Memory Trust
Gap's "dates and provenance" condition and the cheapest answer to PASB. Cost: about 13
characters per fact against `memory.context_cap`, and one re-prime at upgrade.

**A2. A new fact names its neighbour.** A replacement happens only when the model passes
`replaces=#id`. When it does not, "I live in Berlin" lands beside "I live in Paris" and a
small model follows whichever it reads. So `add_memory` reports the stored facts that are
*similar* to a new one — same layer, sharing at least half of the smaller fact's match tokens
(`memory_registry._tokens`), most-shared first, at most two — and every surface that reports
a write shows them: the after-answer line (`remembered #12: I live in Berlin · similar: #3 "I
live in Paris" — /memory forget 3 if that is no longer true`), the gate prompt for a
`remember`, and `/memory add`. Nothing is retired automatically: token overlap cannot tell a
correction from an elaboration, and a wrong deletion is worse than one dim line.

**A3. Secrets are never saved.** One deterministic predicate, asked by every Saturn path that
writes a fact (`remember`, the review accept, the interview, `/memory add`, the import): a
13–19 digit number that passes the Luhn check, a US Social Security number shape, a private
key header, the common API-key shapes (`sk-…`, `ghp_…`, `AKIA…`), and "password / passcode /
PIN" followed by a value. `remember` raises `ToolError` naming what was refused; the commands
say the same sentence. The memory file is plaintext, rides every prompt, and leaves the
machine under a remote `OLLAMA_HOST`. Editing the file by hand stays the user's business.
Health and money are not screened: `sens=` already withholds them from remote inference, and
on a local model they are the user's own facts on their own disk.

**A4. `remember` says what it is for.** One sentence in the tool description, beside the
plan's own edit: a fact about the user or a standing instruction from them — not a claim
about the world, and not something a page, a file or a message said.

**A5. The probe decides two follow-ups.** The plan's `statement` probe (Task 9) measures how
often the model calls `remember` for a plainly stated fact. Add a `correction` probe ("I moved
to Berlin" with a Paris fact in memory): does the call carry `replaces=`? Decision rules, on
the 9b over three runs:
- `statement` under 80% → build the deterministic post-turn catch the plan defers, feeding
  the **review queue**, never a direct write.
- `correction` under 80% → A2's similar-fact line stays as the only defence and the miss rate
  is recorded in `docs/engine.md`; no model-side fix is attempted.

### 2. The interview (the first-run-interview plan, with three amendments)

The plan's mechanism stands: `core/memory_setup.py`, fixed questions, deterministic answer →
fact rules, no model call, once-only marker, `/memory setup` to re-run.

**I1. It is offered, not started.** On a fresh install with an empty memory the launch asks
one line — *Three quick questions so I know who I'm working for? Enter starts · n skips* —
instead of opening on the first question. `n`, `q`, Esc, Ctrl-C or Ctrl-D skip it; the marker
is written either way and the offer is never made again. The first screen a new user answers
is about consent, not about themselves.

**I2. The first run asks three; `/memory setup` asks all five.** First run: `name`, `work`,
`never`. The `people` and `help` questions stay in the table and are asked only by
`/memory setup`. People are better learned in the flow ("Petra is my manager" is exactly what
§1 saves without a click), a list of people is the privacy-heavy answer, and the first launch
should cost under a minute.

**I3. Every interview answer goes through A2 and A3** like any other write.

### 3. The review queue reads only the conversation's own words

`core/memory_review.llm_candidates` builds its transcript with `core/compaction._transcript`,
which includes every `ToolMessage` (clipped to 1,000 characters) and every compaction summary.
That is the path the mem0 audit and the laundering paper describe. Every proposal still faces
the review screen, so nothing is written silently, but a person accepting candidates one key
at a time is a weak filter for a fact a web page planted.

**R1.** The transcript for the review's model pass is the user's typed turns and steer notes
plus the assistant's answers (an `AIMessage` with text and no tool calls). No `ToolMessage`,
no summary, no attachment text. It uses `core/provenance.of(state)` from the auto-memory
plan's Task 1 for "typed", so there is one reading.

**R2.** A candidate from a session that read outside content is labelled on the review screen
(`· this session read outside content`), as are compaction-summary candidates, which are the
model's words by construction.

A candidate whose text is already stored or already pending is dropped today
(`memory_review.add_pending`); that stays.

### 4. Controls that make default-on learning comfortable

Both are already ranked in `docs/research.md` (P4, P5); this spec makes them part of the
system and each gets its own plan.

- **An incognito session** (`saturn --incognito`, `/incognito`): no memory block in the
  grounding, no `recall`, no auto-learn, nothing queued for review.
- **A memory receipt**: the ids of the facts loaded for a turn, beside the Sources receipt, so
  a wrong answer caused by a stale fact is one `/memory why <n>` away.

### 5. `/memory import`

`/memory import <file>` (and `--clipboard`) reads another assistant's export — the text the
user gets by asking it "write down everything you know about me" — and `/memory import
--prompt` prints that request to paste. One bullet or line is one candidate, queued with source
`import` and shown on the existing review screen. A line over the interview's 300-character
limit is skipped and counted, never cut; A3 applies; `add_pending` already drops what is
stored or pending. Accepted facts are `by=inferred src=import`: the user
approved them, but another model wrote them. No model call.

### 6. The launch brief — unchanged

The plan already matches the evidence: no model reads the brief, it never enters the prompt or
the state (so it cannot switch off §1's "nothing external" rule), its lines are made inert, it
is capped and silent when empty, and it derives no fact about the user.

## What is not built, and why

- **Background consolidation ("dreaming").** The 2026 direction at OpenAI, Anthropic's managed
  agents and Perplexity. It needs a model that infers well about people (a 9b over-infers
  nearly half its claims) and a summarising step is where outside content becomes "user
  history". If it is ever built it proposes into the review queue.
- **A model-led interview.** Models do not know what to ask, and every `remember` would face
  the gate or the gate would have to be loosened for it.
- **A profile from Mail, Calendar or files.** The best measured system profiles a user from
  their files at 48.3%, and mail is the demonstrated injection channel.
- **A graph or vector store.** As `docs/research.md` already concludes.
- **Automatic retirement of a conflicting fact.** A2 above.

## Build order

1. **Auto-memory** with A1–A5 and R1–R2. R1 depends on its Task 1 (`core/provenance.py`), so
   the review guard ships in the same plan.
2. **Measure** (the plan's Task 11, on mains power): `statement` and `correction` on the 4b
   and the 9b. The numbers decide the deterministic catch.
3. **Incognito and the memory receipt** — two small plans, to be written.
4. **The interview** with I1–I3.
5. **`/memory import`** — a small plan, to be written; it reuses the interview's fact limit
   and the review screen.
6. **The launch brief** — independent of all of the above; any time.

## Changes to the plans

All three plans were dry-run against commit `7055b57` plus the 2026-10-01 working tree. The
tree has moved (skills, `create_skill`, adaptive thinking), so each plan's dry run is repeated
at the current head when it is amended.

- **`2026-10-01-auto-memory-from-user-statements.md`**
  - Design: add A1–A5; move "the model is the extractor" from an assumption to a measured
    question with A5's decision rules.
  - New tasks: the block header and dated `_context_line` (A1, with the prefix re-prime noted);
    `similar` in the registry and on the three reporting surfaces (A2); the secret predicate
    and its five callers (A3); the review transcript and the label (R1–R2).
  - Task 4: the description sentence (A4). Task 9: the `correction` probe (A5).
  - Review Focus gains: a secret typed by the user; a correction without `replaces=`; a tool
    result in the review transcript.
- **`2026-10-01-first-run-interview.md`**
  - Design and Task 4: `launch_action` returns an offer, not the interview (I1); the first run
    asks `name`, `work`, `never` (I2). Task 3: answers go through `similar` and the secret
    predicate (I3).
  - `docs/pivot.md` #5 is reworded when this ships: three questions offered at first run, five
    under `/memory setup`.
- **`2026-10-01-launch-brief.md`** — no change.

## Decisions made without asking

Each is a product call this draft made so that it has no blanks. Any of them can be reversed
without touching the rest.

1. **Three questions at first run, five under `/memory setup`** (I2). `docs/pivot.md` says
   five. The evidence does not settle the count; the precedent (OpenClaw, Dot's "first date
   that felt more like an interview") leans short.
2. **The interview is offered with Enter as yes** (I1), not printed as a hint. A hint is what
   nobody acts on; a started interview is what OpenClaw removed.
3. **Secrets are refused on `/memory add` too** (A3), not only on the model's path.
4. **Every fact shows its date to the model** (A1), at about 13 characters each.
5. **80% on the 9b** as the line for both probes (A5).
6. **`/memory import` is in scope** (§5), after the interview.

## Not verified

- Every number tagged [A] comes from an abstract; those tagged [F] came back through a
  summarising fetch and, except the four marked ✓, were not fetched twice.
- OpenAI's own pages returned 403; what is said about ChatGPT is from secondary sources.
- Whether a 4b or 9b calls `remember` for a stated fact, or passes `replaces=` for a
  correction. Nothing in this spec has been run.
- How the launch offer reads on a first launch that is still pulling a model.
