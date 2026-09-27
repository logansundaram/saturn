# v2: one ReAct loop replaces the plan/execute engine

_2026-09-27. The first v2 sub-project. Decided against main @ cbd1bd7 after the pivot to a
"Claude Code for daily tasks" companion: terminal stays the surface (the Claude Code audience,
technical people doing non-code life admin); the trust stack stays; the plan-as-control-surface
half of the old moat goes._

## Why

PLAN.md's common-case contract measured the traffic: 41 of 46 turns were single-step, and each
paid the plan call, the execute pass, the rectify judge and synthesize. The quick path routed
around the engine for the simplest shapes but left two engines to maintain and a regex deciding
between them. A daily companion needs one loop that is cheap on a chat question and still
finishes a five-tool task — Claude Code's shape.

## The loop

```
START → ground → agent ─(no tool calls)─→ END
                   ↑          │ tool calls
                   │          ▼
                   └── tools ← approval   (a fully-rejected batch routes back to agent)
```

Four nodes. `ground` (unchanged, minus the recap section — the loop sees real history) builds
the two-half grounding block. `agent` makes ONE native tool-calling call, think off, streamed.
`approval` is the human gate (`trust/policy.approves`), unchanged except its rejected route.
`tools` executes, clamps, records egress, fences quarantine — unchanged except it maps the `plan`
tool onto state (below).

Deleted: `nodes/plan, quick, plan_gate, execute, update_plan, rectify, replan, synthesize,
answer_gate`; `core/complexity.py`, `core/request_intent.py`; the plan-editor half of
`core/plan_ops.py` (the `PauseController` moves to `core/pause.py`); most of `core/plan_context.py`
(`grounding_parts`, `clean`, `WRITE_TOOLS` survive in `core/context.py`) and `core/messages.py`
(the engine prompts go; `agent_sys_msg()` arrives; compaction/memory-review/init prompts stay).

## The agent node (`nodes/agent.py`)

**Prompt order** (prefix-cache order — changing material last, append-only within a turn):

```
[system: agent_sys_msg()]            identity, rules, when to use `plan`; tools ride the native bind
[user: stable grounding]             the primed boundary (core/prime.py)
[prior history …]                    the compacted conversation (app/session._compact_history)
[user: dynamic grounding + request]  memory matches + attachments prepended to the turn's query
[this turn's AI / tool messages …]
```

**The call.** `get_model("tool_caller").bind_tools(registry.tool)`, streamed through `llms.stream`
with `_invoke_kwargs("tool_caller", None, 0.0, task="agent")`. New serving task `agent`:
`strict=False, num_predict=4096, think=False` (a `write_file` payload must fit). Content chunks
stream to the UI via LangGraph messages mode filtered to `langgraph_node == "agent"`.

**Per pass, in order (deterministic first, the model last):**

1. **Steer.** When no pause is pending, drain `controller.take_steers()`; each becomes a
   HumanMessage `STEER_PREFIX <note>` appended to the conversation (standalone — after a
   ToolMessage there is no HumanMessage to merge onto; `is_turn_start` already skips it).
2. **Pause.** `controller.pending()` → `interrupt({"type": "pause", "reason", "iteration",
   "plan"})`. Resume value `{"action": "continue" | "steer" | "abort", "text"}`; the controller
   is cleared after the interrupt returns (the determinism rule). `steer` appends the note as in
   1; `abort` ends the turn with a final AIMessage stating it stopped at the user's request
   (orphaned calls cancelled; no model call).
3. **Iteration cap.** `iteration >= runtime.max_iterations` (16, now agent passes): one final
   call with tools UNBOUND and a trailing note "the action budget is spent — answer from what you
   have and state plainly what is undone". A real answer, never a stub.
4. **Generate.** Stream the call; fold `tok_per_sec` / `context_tokens` off the final chunk.
5. **Tool-call hygiene** on the emitted calls, each producing an error ToolMessage that routes
   straight back to `agent` (no approval, no model call for the check itself):
   - unknown tool name → "unknown tool";
   - `coerce_args(name, args)` returns None → "missing argument" + `schema_hint`;
   - **declined-repeat**: identical (name + args) to a call the user declined at the gate this
     turn → "already declined this turn — do not retry; tell the user";
   - **stall**: the third identical call this turn → "you already have this result; answer from
     it or do something different".
   Calls that pass go to `approval` as a tool-calling AIMessage (the batch shape `tools` already
   handles; several calls per pass are allowed).
6. **Answer.** No tool calls → the message IS the answer. Mechanical trailers appended to the
   recorded AIMessage (never to the streamed text — the REPL re-renders the recorded message):
   - the Sources footer (`runtime.citations`): every tool call / retrieved document this turn, a
     receipt of what informed the answer — inline `[n]` markers are no longer requested;
   - the incidents note: tool rounds that ended `skipped` / `blocked` / `error` (read off the
     ToolMessages' `saturn_status` stamp) and `plan` items still pending.
   The groundedness and computed-figure regeneration ladders are removed: they cost a call each
   and compensated for a synthesizer that never saw raw observations; the loop does.

## The plan is a tool

`plan(steps=[{label, status}])` in `tools/planning.py`, `read_only`, never gated. Statuses
`pending | done`. The `tools` node maps a successful `plan` call onto `state["plan"]` as the
existing step dicts (`step_id, label, status, intended_tool=None, result=None|"done",
needs_resolution=False`) so every reader works unchanged: the rail's `show_plan`, the gate's
`step` context (`current_step` = first pending item), `/trace why`'s "what it set out to do",
replay, the headless `plan` field, `memory_review`'s unfinished-item candidates. The prompt asks
for it only on tasks needing several tool calls, and to re-call it as items complete.

## The trust stack, unchanged

`approval` still asks `policy.approves` per call, interrupts with the same payload (`step` is
the current plan item), records `gate_events`, applies always-allow grants, and escalates a
quarantine flag. `tools` still clamps, records egress per call, fences untrusted output. The
declined-repeat guard is additive: the gate's "no" is honored for the rest of the turn.

## Surface

- **Rail.** `agent` rows carry the iteration/ctx/tok-s metrics and a dim leaf with the model's
  pre-call text. Tool result previews render by default (one clipped line per call; `/trace
  calls` stays the full dump). `ground` folds at normal verbosity; `approval`/`tools` as today.
- **Pause prompt** replaces the plan editor: `ui.pause_prompt(value)` shows the plan if any, then
  `[Enter] continue · type a correction · q abort`.
- **Commands.** `/plan`, `/draft`, `/quick` → `_RENAMED` pointers ("the plan engine was removed
  in v2 — Esc pauses a running turn; Esc with text steers it"). `/plan review` mode and
  `pending_plan` / `pending_turn` / `review_plan` context fields go. CLI `--plan` / `--quick` go.
- **`/trace why`** renders agent passes (thought + chosen calls) and drops the quick/rectify/replan
  branches. `/trace full|calls` unchanged.
- **Headless `-q`** progress announces each tool call and plan update.
- **Config.** `runtime.quick_path` removed from the template; `runtime.max_iterations` re-documented
  as agent passes. Roles/tiers untouched (`tool_caller` serves the loop; `planner`/`judge`
  bindings stay valid and unused — no migration). `check_models` drops the planner/judge
  capability advisories.
- **Prime.** One lineage: `("agent", "tool_caller", bound model, [agent_sys_msg(), stable])`.
  The bound tools render into the chat template's system section, so the ~3.7k-token catalog is
  part of the primed prefix (measured: 26 tools, 14.6k chars of schema). Startup primes it.

## State (`core/state.py`)

Removed: `route`, `rectify`, `reasoning`, `replans`, `aborted`, `plan_vetoes`, `revoked_writes`.
Kept: `messages, current_query, context*, attachments, plan, iteration, tools_called,
tool_results, documents_retrieved, tool_events, gate_events, answer_buffer (always None — the
parked token-steering seam), tok_per_sec, context_tokens`. `TERMINAL_STATUSES`,
`INCIDENT_STATUSES`, `current_step`, `unfinished_steps`, `incident_steps`, `summarize_gates`,
`STEER_PREFIX`, `is_steer_message`, `is_turn_start` stay.

## Parked, not deleted

`core/confidence*`, `core/provenance`, `core/continuation`, `tui/ui/correction.py`
(`ui.edit_answer`) and their module tests stay. The loop never produces an `answer_buffer`, so
the TUI's buffer/confidence rendering is dormant. Re-adding token steering is one seam: the
agent node's stream.

## Tests

Deleted (263 tests): `test_engine, test_revocation, test_quick, test_ask_gate, test_completeness,
test_target_coverage, test_plan_vetoes, test_plan_draft, test_effect_vocabulary,
test_plan_retarget, test_synthesize_disclosure`. Edited: the ~20 files that construct plan state,
import `PauseController` from `plan_ops`, assert `goto == "update_plan"`, render `replans`, or
list `/plan` in `/help`. New `tests/test_agent_loop.py` pins: graph wiring; each guard in the
agent node (steer, pause continue/steer/abort, cap, unknown tool, missing arg, declined-repeat,
stall); answer trailers (sources, incidents); the `plan` tool → state mapping; the stream filter;
the prime lineage; `/trace why` on agent calls. Fully offline as before.

## Verification

Offline suite green. Then four live turns on the running tier (kept short — battery): a chat
question (one call), a single read (two calls), a gated write declined at the gate (disclosed,
never retried), a three-step task. `llm_calls.prompt_tokens` on the second turn's first call
confirms the catalog prefix is cached (prefill ≈ the request, not ≈ 5k).

## Out of scope (later sub-projects)

Knowing the user (onboarding interview, provenance-gated auto-memory, home-directory workspace,
PDF/docx reads, more AppleScript readers, launch brief); making it yours (`SATURN.md`, skills,
hooks, script tools); feeling like a product (hardware-probed install, command diet, gate
"always allow" polish). Token steering's return. The "Trim / demote" deletion pass.
