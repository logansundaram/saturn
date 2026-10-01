# The engine — its shape today, and what to improve

_2026-09-29. A description of the v2 loop as it runs now, then the engine improvements ranked
by the evidence from the loop and trust benchmarks. "Engine" means the loop itself — the
nodes, the prompt, the guards, the observation path — not the features around it; those are in
`pivot.md` (this folder). Code map: `ARCHITECTURE.md`. Spec: `superpowers/specs/2026-09-27-v2-react-loop-design.md`._

## The shape

One ReAct loop, four nodes, compiled by `app/graph.py::build_agent` with a SqliteSaver
checkpointer and driven by `app/turn.py::run_turn`:

```
ground → agent ─(no tool calls)─→ END
           ↑          │ tool calls
           └── tools ← approval      (a fully-rejected batch → agent)
```

There is no planner, no judge, no synthesizer. The model's last message is the answer, and it
streams as it is written. Measured on the 4b: a chat question is one model call in ~2–3 s, a
lookup is two calls in ~3 s, a two-file comparison issues both reads in one pass.

### ground (`nodes/ground.py`) — no model call

Assembles `state["context"]` in two halves so the prefix cache holds:

- **stable** — byte-identical across turns while nothing on disk changed: the working folder
  (launch folder + `/add-dir` folders), `~/.saturn/SATURN.md`, the folder's `SATURN.md`
  (`SATURDAY.md` still read), the knowledge-base manifest, and the always-loaded memory layers
  (`user`, `commitments`, the last five `memo` entries).
- **dynamic** — per turn: the `### Now` line (weekday, date, time, UTC offset), memory facts
  matched to the request by token overlap, and `@file` attachments (a PDF / .docx / .xlsx
  attaches as its text).

### agent (`nodes/agent.py`) — ONE native tool-calling call per pass

The prompt is built in prefix-cache order:

```
[system][user: stable grounding][history…][user: dynamic grounding + request][this turn…]
```

The bound tool schemas render into the chat template's system section, so the catalog is part
of the cached prefix; `core/prime.py` re-sends `[system][stable]` between turns so the daemon
holds a checkpoint there and the next turn prefills only what is new. Prior answers go into
the history with their trailers stripped (`strip_trailers`).

Around the call, deterministic checks in this order (each pinned by `tests/test_agent_loop.py`,
each costing a chat turn nothing):

1. **pause** — Esc alone `interrupt()`s into continue / steer / abort.
2. **steer** — Esc + text mid-turn lands as a `STEER_PREFIX` HumanMessage (not a turn boundary;
   `core.state.is_turn_start` is the one predicate). Drained only PAST the pause: a resumed
   interrupt re-runs the node from the top, so steers taken before it would be lost.
3. **cap** — from pass `runtime.max_iterations` (16) on, no tool call runs. The pass is the same
   bound call (nothing appended, so the prompt extends the cached prefix); a call it emits is
   answered with `BUDGET_TEXT` and routes back, and the next pass answers. A model that calls
   again is rerun once with tools UNBOUND and a budget note: a real answer, never a stub.
   Measured 2026-10-01 at 7.5k prompt tokens: the unbound pass re-prefilled everything (4b
   10.1 s, 9b 9.2 s) where the bound one paid 0.3–0.5 s, and both models answered in text once
   their calls were refused; the 9b ignored a budget NOTE while tools were bound, which is why
   the refusal, not a note, is the mechanism.
4. **generate** — `_generate`, the one seam tests replace: `bind_tools(registry)`, streamed,
   chunks folded into one AIMessage. Options ride `core/llms.invoke_kwargs` (num_ctx,
   `num_predict` 4096 for the agent task, `reasoning` explicitly off). Thinking is adaptive
   (`runtime.think: adaptive`): a pass thinks, under `think_budget` (4096), only when the tool
   round just before it had an error; a thinking pass that returns nothing is rerun think-off
   (the 4b/9b write the answer inside the reasoning and emit no content). A thinking pass's
   reasoning is recorded in `llm_calls` (shown by `/trace why`), never on the message. A model
   reply whose tool arguments were not valid JSON is retried once with a corrective note;
   what the failed attempt streamed is retracted first (`RETRACT` on LangGraph's custom
   stream, which `app/turn.run_turn` hands to `on_retract` — the REPL's `answer.discard`).
5. **hygiene** — on each emitted call, answered with an error ToolMessage that routes straight
   back to `agent` (no gate, no execution): an unknown tool; arguments that belong to another
   tool (`recall(fact=…)` → "those belong to remember"); missing required arguments after
   alias coercion (`core/tool_args`); a repeat of a call the user DECLINED this turn; a third
   identical call with nothing changed since the first (`STALL_REPEATS` — a completed write,
   edit or command in between resets the count, so edit → test → edit → test is not a stall). `ask_user` runs alone — a resumed interrupt re-executes
   the tools node, so siblings in its batch are answered with `ASK_ALONE_TEXT`.
6. **answer** — a message without tool calls IS the answer. The incidents note (calls that
   were declined, blocked or failed, read off the ToolMessages' `saturn_status` stamp; a
   call's LAST outcome decides, so one that failed and then ran is not listed; a stall refusal is
   not an outcome, and a call refused at the cap reads "not run") and the Sources receipt (the
   completed calls and documents the turn gathered — not failures, not writes — numbered as
   `/trace source` numbers them) are appended to the RECORDED message, never the stream.

A text preamble before a tool call is not part of the response: `app/turn.py` shows it as the
rail's agent leaf and discards it from the response region.

### approval (`nodes/approval.py`) — the one gate question

Takes the issuing message's calls minus those already answered
(`core.state.issuing_message`, shared with `route_after_agent` and the tools node) and asks
`trust/policy.approves(name, risk, args)` for each. Read-only calls pass; anything past
`runtime.auto_approve` (`read_only` by default) `interrupt()`s for the human, per batch or per
call, with always-allow grants scoped by `runtime.grant_scope`. The next batch
that can act (send or change something) after quarantine-flagged output is escalated to the gate
regardless of tier; a `web_extract` URL the model composed after external content entered the
conversation, or a private address the user did not type, is held too (`quarantine.url_hold`);
under air-gap `run_shell` and MCP calls always ask (`policy.airgap_holds`). Rejected calls get a
decline ToolMessage (the declined-repeat guard keeps that "no" for the rest of the turn); a
fully-rejected batch routes back to `agent`. Every human decision lands in `gate_events`, the
one record nothing can recompute.

### tools (`nodes/tools.py`) — execute, clamp, record

Runs the pending calls one after another (serial, so egress events attribute to their call by
sequence). Per call: the observation is clamped to `_MAX_OBSERVATION` (12,000 characters,
head and tail); the egress slice is attached; untrusted output (web, MCP, files, the shell,
notes, mail — a failed call's text too) is scanned by `trust/quarantine.py` and fenced as data; a structural `saturn_status` stamp
(`done` / `error` / `blocked` / `skipped`) rides the ToolMessage so no reader has to sniff
outcome from text — a tool reports failure by RAISING `tools.toolspec.ToolError` (an edit
whose text was not found, a non-zero shell exit, a refused path, an MCP error), which the node
stamps `error`; before 2026-09-30 most tools returned such failures as strings, stamped `done`; a successful `plan` call maps onto `state["plan"]` (the rail's checklist —
intent, not record). Each tool event records the agent pass that issued it.

### Around the loop

- `runtime.keep_alive` keeps the model loaded between turns; the idle prime warms the prefix.
- Auto-compaction (`runtime.auto_compact`, threshold 0.85 of the window) summarizes older turns
  with the same model after a turn (thinking off, `num_predict` bounded — `invoke_kwargs`
  task `compaction`, as are `memory_review` and `init`); the memory review queues candidates from each turn.
- Headless `-p` / `-q` run the same loop with gated tools DENIED unless `--yolo`.
- `hooks.yaml` fires on turn-start / turn-end (from `run_turn`) and before- / after-write
  (from the file tools); a before-write non-zero exit refuses the write as a failed step.

### What the benchmarks say (2026-09-29, after the small wins)

These numbers predate the 2026-09-30 review fixes: failed tool calls now stamp `error`, so the
adaptive think fires after them and the incidents note lists them. Re-run both benchmarks
before comparing.

Loop benchmark (`python benchmark.py --loop`, 25 daily requests): 4b 20/25, 9b 20/24 (before
the dependent-calls task); no phantom actions, no hygiene bounces, no capped turns on either.
Trust benchmark: gate 3/3, injection 2/2, memory recall / supersession / planting all pass on
both tiers. The stable misses:

- `file_long_middle` (every tier): the fact sits in the middle of a 30k-character file and the
  head+tail clamp drops it.
- `robust_no_math_in_head` (9b): seven `calculate` calls to test whether 391 is prime — eight
  passes for a right answer.
- `multi_read_then_calc` (intermittent): 4471 × 3 done in the model's head, correctly.
- `robust_underspecified` (9b): the question came as a numbered list, which the grader does
  not count as asking.
- `chat_about_tools`, `robust_no_tool` (4b): "what can you do with my email?" calls
  `list_mail`; "send a text to Petra" reaches for `draft_mail` / `recall`.

An A/B of the "rounds" prompt rule on two dependent-call tasks (read a value, then write it;
search for a file, then read it) was 8/8 in order with the old prompt and 8/8 with the new
on the 4b at temperature 0: the model already waited for the result. The rule stays as a
correct description of the loop, not as a measured gain. The 4b's supersession miss was
`recall(fact=…, replaces=…)` — `remember`'s call under the wrong name, which a
no-required-args tool used to swallow and run — fixed by the foreign-arguments refusal.

## Improvements, ranked

Ranked by the evidence above and the pivot's two knives: would a person hand this to Saturn on
a Tuesday, and what does it cost the chat turn. Every item below costs a plain chat question
nothing — each guard fires only on its failure shape.

### Guards (robustness first)

1. **A hygiene budget per turn.** The 4b's supersession spiral was 14 passes with 7 hygiene
   bounces before it answered; today the only bounds are the pass cap (16) and the third
   identical call. After N bounced calls in one turn (3), go straight to the cap's refusal and
   answer. Deterministic; `nodes/agent.py`.
2. **A failure-aware final pass.** That same run ended with "I've updated my memory" over an
   incidents note saying it had not. When the last tool round was all errors or declines, the
   final pass gets a one-line note — "these calls did not happen: …; say so" — before it
   answers. The incidents note already tells the user; this puts the same fact in front of
   the model. Fires only after a failed round.
3. **Replay a repeated read-only call instead of refusing it.** The third-identical-call guard
   answers with an error the model then has to reason about. For a read-only tool the
   observation is already in context: hand it back as the result (no gate, no execution).
   Side-effecting tools keep the refusal — a second `write_file` may be meant.

### The prompt the model sees

4. **A relevance-aware clamp.** `file_long_middle` fails on every tier. Keep the head plus the
   lines that share rare words with the request, then the tail. No model call; replaces the
   fixed head+tail in `nodes/tools.py`. The one stable miss in the loop benchmark.
5. **A token budget for the prompt** (pivot loop item 2). Ten reads on a 32k window push the
   system prompt off the front. `_llm_input` becomes a budgeted projection: an observation a
   later pass has already moved past collapses to a one-line stub in the PROMPT only — state,
   the trace and replay stay whole. Subsumes the clamp long-term. (Between turns this is
   handled since 2026-10-01: auto-compaction trims the finished turn's tool results when it
   filled the window. Within one turn it is still open.)
6. **A question is an answer** (pivot loop item 3). Delete the `ask_user` interrupt: when the
   model needs a value it answers with the question and the turn ends; the reply is the next
   turn with the history intact. Removes the run-alone hack (`ASK_ALONE_TEXT`), the headless
   special case, the tool, and the `no_question` grader ambiguity; `plan` state carries across
   the boundary so a mid-checklist question resumes.
7. **Sharper tool descriptions where the 4b reaches wrong.** A prompt line ("a question about
   what you can do is answered from your tools' descriptions, without calling them") and a
   tightened `list_mail` description are cheap. The catalog experiment (pivot loop item 9 —
   domain tools with an action enum, or a small core set plus a deferred group) is the bigger
   version, only if wrong-tool picks keep showing up; both fight the prefix cache.
8. **Give `calculate` the whole expression.** One prompt line; the primality case is partly
   task design (the calculator cannot answer "is it prime"), so expect a smaller gain.

### Execution

9. **Concurrent tool batches** (pivot "Improve" 6). The model already emits several calls per
   pass; the tools node runs them serially so egress attributes by sequence. Tag ledger events
   with a call id (a contextvar) and run the batch in a pool. Matters for two web fetches or
   two AppleScript readers in one pass, not for file reads.
10. **A wall-clock budget** (pivot loop item 7). Declined for now; stays the cheap safety net
    if dogfooding finds a slow turn.

### Measurement

11. **Thinking on/off as a benchmark flag.** The adaptive think fires after a tool error and
    the 4b's thinking passes often produce nothing. A `--think off` run of the loop benchmark
    says whether thinking earns its latency per tier.
12. **Grade the recorded reasoning.** It is in `llm_calls` now; the loop benchmark can check
    that a thinking pass's thought names the failed tool — what the pass exists for.
13. **Three runs per change.** The 4b at temperature 0 is close to deterministic per task, but
    the run-to-run flips (`chat_greeting` one day, `chat_about_tools` the next) mean a pass-rate
    delta of 1 is noise. Three runs per change make the number mean something.

Order: 1 and 2 together (a half-day, both in `nodes/agent.py`), then 4, then 5 and 6 as one
sub-project, then 9. Items 7, 8 and 11–13 are an afternoon each and can go anywhere.
