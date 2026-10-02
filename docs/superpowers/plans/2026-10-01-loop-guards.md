# Loop Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three deterministic guards in the agent loop — a per-turn hygiene budget, a failure-aware note before the pass that follows a round where nothing happened, and a replay of a repeated read instead of a refusal — so a small model that slips ends its turn quickly and honestly.

**Architecture:** Every ToolMessage the agent node writes itself (no tool ran) gets a structural kind marker (`textutil.HYGIENE_KEY`). The hygiene budget counts passes with a refused call off those markers and, past three, reuses the existing cap mechanics with its own refusal text. The failure note rides the prompt tail through `_llm_input`'s existing `extra` parameter (never state). The replay answers a third identical `read_only` call from the observation already in state, once. All three live in `nodes/agent.py`; the rail and the loop benchmark learn to read the marker.

**Tech Stack:** Python 3.11+, LangGraph, langchain-core messages, pytest (offline: `nodes.agent._generate` replaced in every test).

**Spec:** `docs/engine.md` "Improvements, ranked" items 1–3 (Guards), and the Design section below.

---

## Design

### The problem

The loop and trust benchmarks (2026-09-29, `docs/engine.md` "What the benchmarks say") show the worst behaviour on record: on the 4b, the memory-supersession probe called `recall` with `remember`'s arguments, was refused by hygiene, re-issued the same wrong call, and so on — **14 passes with 7 hygiene bounces** before it answered. The answer it finally gave was **"I've updated my memory"**, over an incidents note saying it had not. (The foreign-arguments refusal added that evening gives the model a better hint, but nothing bounds the spiral except the 16-pass cap, and nothing tells the model, at the moment it writes its answer, that its last round did nothing.)

Three distinct failures:

1. **No bound on repeated slips.** Today the only bounds are `runtime.max_iterations` (16) and the third-identical-call stall (`STALL_REPEATS`). A model that varies its broken call slightly (different arguments, a different wrong tool) never stalls and burns all sixteen passes — on a 4b that can be a minute or more.
2. **The answer claims what did not happen.** The incidents note tells the *user*; nothing tells the *model*. The ToolMessages that say "declined" or "error" are in context, but a small model writing a wrap-up after a long turn summarises intent, not outcome.
3. **The stall refusal on a re-read costs a pass for nothing.** A third identical `read_file` with nothing changed is refused with `STALL_TEXT` — an `error`-stamped refusal that wakes adaptive thinking, appears to the model as a failure it must reason about, and adds a pass. For a read, the right answer is already in context.

### Guard 1 — the hygiene budget

**Approaches considered.**

- *(a) Count refused calls; stop after N.* Simple, but a single pass that emits three malformed calls in one batch (a small model's parallel slip) would stop the turn at once.
- *(b) Count passes that had at least one refused call; stop after N.* **Chosen.** Robust to parallel batches; matches the spiral shape (one bad call per pass, again and again).
- *(c) A lower pass cap on small tiers.* Rejected: it punishes long clean errands, which are exactly what a daily-life agent is for, and does nothing on the 9b/27b where a spiral can also happen.

**What counts.** A pass counts when the agent node refused at least one of its calls as the model's slip: an unknown tool (`unknown_tool`), malformed JSON (`malformed`), arguments that belong to another tool (`foreign_args`), required arguments missing (`missing_args`), a repeat of a call the user declined this turn (`declined_repeat`), a third identical call with nothing changed (`stall`). It does **not** count: a real tool error (the tools node ran the call — the model may be right to try something else), a gate decline (the approval node's reply is a human decision, not a slip; a user who says no three times must not trigger "malformed calls" wording), a budget refusal (`budget` — the turn's limit), a replayed read (`replay` — Guard 3, a legitimate re-read). The `ASK_ALONE_TEXT` reply carries no marker either (it is the engine's own constraint, and the sibling plan deletes it).

**Mechanism.** After `HYGIENE_BUDGET` (3, a module constant beside `STALL_REPEATS`) bad passes, the next pass is treated exactly like the cap pass: tools stay bound (the cached prefix holds), any call it emits is answered with a refusal — here `BOUNCE_TEXT`, worded so the model and the incidents note stay truthful ("too many of this turn's tool calls were malformed or refused") — and routes back; the pass after that ends the turn, with the existing tools-unbound rerun as the last resort for a model that still calls. No second stop path: the cap's code is generalised.

**Exact condition changes in `agent_node`.** Today: `capped = iteration >= cap`; the unbound hard stop and the answer path fire on `iteration > cap`. After:

```
stopped = a BUDGET_TEXT or BOUNCE_TEXT refusal already answered a call this turn
bounced = _bounced_passes(this_turn) >= HYGIENE_BUDGET
capped  = iteration >= cap or stopped or bounced     # this pass's calls are refused
ending  = iteration > cap  or stopped                # this pass must be the answer
```

`stopped` is what makes the hygiene stop end the turn one pass later, exactly as `iteration > cap` does for the cap: at the cap the two coincide (a pass past the cap always follows a budget refusal), so the cap's behaviour is unchanged. The refusal text is `BUDGET_TEXT` when `iteration >= cap`, else `BOUNCE_TEXT`.

**Cost on the clean path:** two linear scans of this turn's messages (no model call, no prompt change). A chat question has an empty turn record: both are no-ops.

### Guard 2 — the failure-aware final pass

**Approaches considered.**

- *(a) A prompt-only note at the very tail, via `_llm_input(..., extra=[...])`.* **Chosen.** Exactly how `BUDGET_NOTE` already reaches the model: a HumanMessage after the ToolMessages, in the prompt only. Nothing is inserted before the turn's last message, so the cached prefix holds; state, the trace and replay never see it, and `is_turn_start` cannot mistake it for a new request.
- *(b) Rewrite the failed ToolMessages' content to shout "THIS DID NOT HAPPEN".* Rejected: changes state and the trace, and the wording would leak into later turns' history.
- *(c) A post-hoc check on the answer ("claims success while every action failed").* **Non-goal.** It needs a claim parser — a model-like judgement on free text — and the incidents note already tells the user the truth under every answer. The note in (a) fixes the cause; the incidents note covers the rest.
- *(d) A static system-prompt line ("never claim a call that failed").* Rejected: already implied by the prompt, measured not to work on the 4b, and it would change the cached prefix for every user.

**When it fires.** The round the coming pass reacts to (`_latest_round` — every ToolMessage after the last tool-calling message, skipping a steer note) has *nothing* that completed: every call declined (`skipped`), air-gapped (`blocked`), failed or refused (`error`). It stays silent when anything in the round completed (the incidents note covers a partial round), after a round that is only stall refusals (the call already ran twice; its results are above — `incidents()` already treats a stall as not-an-outcome, and the note reuses `incidents()`), and after a round that is only budget refusals (`BUDGET_TEXT` / `BOUNCE_TEXT` already say "answer now and state what was not done").

**Wording.** It reuses `incidents()` on the latest round, so the model and the user read the same account: "None of the calls in your last round happened:\n- write_file(…) — declined at the approval gate — not done\nIf you answer now, say plainly that these were not done; do not write as if they were." It deliberately does **not** say "answer now": after a failed read, calling a different tool may be the right move, and the note must not suppress that retry.

**Template and thinking.** A trailing user message after tool messages is already sent today (the hard stop's `BUDGET_NOTE`); the Ollama chat templates render it as an ordinary user turn. Adaptive thinking reads `this_turn` from state, not the prompt, so the note neither triggers nor suppresses a thinking pass.

**Cost:** zero on the clean path (no round, or a round with a completed call → `extra` stays empty and the prompt bytes are identical). After a failed round, a few dozen tokens at the tail, which the next pass drops again (the cached prefix up to the last ToolMessage holds).

### Guard 3 — replay a repeated read

**Approaches considered.**

- *(a) Refuse (today).* An error the model must reason about; wakes thinking; costs a pass.
- *(b) Replay the earlier observation verbatim, always.* Duplicates up to 12k characters of context per replay.
- *(c) Replay small observations inline, point at large ones.* **Chosen.** `REPLAY_INLINE_CAP` = 2,000 characters: under it the replay carries the result (cheap, and a small model reads the nearest copy best); over it the replay says the result is above, in the earlier identical call. A large observation is already in the prompt unchanged (sibling plan `2026-10-01-observation-budget.md` may later stub it — see Merge notes).

**When it replays.** All of: the tool's declared risk is `read_only` (`tools.registry.DECLARED_RISK`; an MCP tool never is — its tier fails closed); the identical call has been made `STALL_REPEATS` (2) times since anything changed (the existing `_repeats_since_change` rule, refactored to return the rounds); the most recent of those completed (`done` — replaying a failure would hand back an error the model needs a new approach for, so it keeps `STALL_TEXT`); none of them was already a replay (a fourth identical ask after a replay is a loop → `STALL_TEXT`, which counts toward the hygiene budget); and the result is not fenced by the quarantine. That last rule is a security property: a replay skips the tools node, which is where a real re-read would be re-scanned and would re-arm the gate escalation (`quarantine.flag`). Rather than duplicate that logic, a fenced result is never replayed. Side-effecting tools keep the refusal — a third identical `write_file` whose first two completed is still a stall (same-key rounds never reset the count), and a second write may be meant, so the model must say so.

**The stamp.** `saturn_status: "done"` plus `HYGIENE_KEY: "replay"` — not a new status value. Audit of every reader:

| Reader | Sees | Result |
|---|---|---|
| `nodes/agent._rounds` → `incidents()` | status `done` | the call's last outcome is `done` (true: it completed, and nothing changed since) → not an incident |
| `nodes/agent._wants_think` | not `error` | not evidence → no thinking pass |
| `nodes/agent._same_since_change` | a round of the same key | counts as a repeat; `read_only`, so never resets anything; the next identical call stalls |
| `nodes/agent._round_notes` (Guard 2) | `done` | the round completed → no note |
| `nodes/agent._bounced_passes` (Guard 1) | kind `replay` ∈ `UNCOUNTED_KINDS` | not a slip |
| `nodes/approval.py`, `nodes/tools.py` | — | never see it: `core.state.issuing_message` lists it as answered |
| `tool_results` / `documents_retrieved` / Sources receipt / `/trace source` | — | written only by the tools node → not a new source (the original call already is one) |
| `tools_called` / `tool_events` / the rail's tools sub-tree / the trace's tool events | — | written only by the tools node → not an execution |
| the rail's agent leaf | kind `replay` | `↺ read_file(…) — same result as before, not run again` (Task 5) |
| `stores/trace` (agent delta) | the message | recorded as what the agent node answered — honest; the pointer keeps a large result from being recorded twice |
| `benchmark.py` | kind `replay` | counted as `replays`, not as a hygiene bounce, not as a tool call |
| `core/compaction.py` | a ToolMessage | trimmed like any other between turns; a pointer is short |

### Assumptions made without asking

1. `HYGIENE_BUDGET = 3` bad passes, a module constant (like `STALL_REPEATS`), no `runtime.*` knob. The user's standing robustness priority is "no silent loops"; a knob nobody tunes is upkeep.
2. The budget counts passes, not calls (parallel batches); it counts the declined-repeat refusal (the model re-issuing a call the user refused) but never a gate decline.
3. The replay is a `done` result with a kind marker, not a new `saturn_status` value — every reader's behaviour stays right without touching it (table above).
4. `REPLAY_INLINE_CAP = 2000` characters.
5. A `current_time` call made a third time in one turn with nothing changed gets the earlier time back. Accepted: the turn is seconds long, and the date is in the `### Now` grounding anyway.
6. The rail change (Task 5) is in scope: the brief requires a truthful rail label for a replay, and refused calls had no label of their own (today the rail shows the last message of the agent's delta as if it were the agent's thought).
7. The loop benchmark's `hygiene` count switches from "ToolMessages minus tool events" to the marker. The old subtraction also counted gate declines; the new count is what the tag's comment always claimed.

### Non-goals

- A post-hoc success-claim checker (Guard 2, option c).
- A wall-clock budget (`docs/engine.md` item 10 stays declined; this budget removes its main cause).
- Any change to the cap's arithmetic for a turn without refusals.

---

## Global Constraints

- Every guard is deterministic, fires only on its failure shape, and costs a plain chat question nothing: no extra model call, and no change to the prompt on the clean path — the cached prefix must hold, so nothing may be inserted before the turn's last message.
- Each new check is pinned by a test in `tests/test_agent_loop.py` (the benchmark helpers' tests live beside the existing ones in `tests/test_cli.py`).
- A note for the model rides the PROMPT only (`_llm_input(..., extra=[...])`), never state: a HumanMessage in state is a new request to `core.state.is_turn_start`.
- Tests are offline: `nodes.agent._generate` is replaced in every test that runs `agent_node`; no model, network or embedder.
- `diag.log()`, never `print()`, in `nodes/`.
- The user's robustness priority: no silent loops, no excessive retries, honest answers about what did not happen.
- User-visible changes go under `## [Unreleased]` in `CHANGELOG.md`.
- Commit messages: `area: what changed`, lowercase.
- Run everything with `.venv/bin/python` from `/Users/Logan/Documents/saturn-v2`. Baseline before Task 1: `.venv/bin/python -m pytest tests/test_agent_loop.py -q` → `55 passed`.

## Review Focus

The inputs no task's main tests exercise but a person will meet, most likely first. Each has its test in the owning task.

1. **A long errand with two unrelated slips early** (an unknown tool once, a missing argument once, then ten clean calls) must not be cut off by the budget. Test: Task 2, `test_two_slips_do_not_stop_a_long_errand`.
2. **A pass that emits three malformed calls in one batch** counts as one bad pass, not three. Test: Task 2, `test_a_bad_batch_is_one_bad_pass`.
3. **A user who declines three batches at the gate** must not trip the hygiene budget (its wording would call the model's calls "malformed"); replays never count either. Test: Task 2, `test_gate_declines_and_replays_are_not_slips`.
4. **A steer typed after a round where nothing happened**: the note still fires, after the steer. Test: Task 3, `test_the_note_survives_a_steer`.
5. **Injected content repeated by a re-read**: a quarantine-fenced result is never replayed past the tools node's re-scan. Test: Task 4, `test_a_fenced_result_is_never_replayed`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `textutil.py` | add `HYGIENE_KEY` | the marker key — a leaf both `nodes/agent.py` and `tui/ui/trace.py` import |
| `nodes/agent.py` | markers, `answered_kind`, `UNCOUNTED_KINDS`; `HYGIENE_BUDGET`, `BOUNCE_TEXT`, `_bounced_passes`, `_stopped`, `_NOT_RUN`; `FAILED_ROUND_NOTE`, `_status`, `_round_notes`; `REPLAY_*`, `_same_since_change`, `_replay` | all three guards |
| `trust/quarantine.py` | `FENCE_BEGIN`, `FENCE_END`, `is_fenced` | one spelling of the fence, read back |
| `tui/ui/trace.py` | `_render_agent_thought`, `_answered_label` | a leaf per agent-answered call |
| `benchmark.py` | `agent_answered`; `run_query`, `grade_loop_task`, `summarize_loop`, `run_loop_benchmark`, `run_loop` | counts by marker; `replays`, `bounced_out` |
| `tests/test_agent_loop.py` | new "loop guards" section; one existing test re-pointed | pins |
| `tests/test_cli.py` | two benchmark tests | pins |
| `tests/test_quarantine.py` | one test | pins |
| `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/engine.md`, `CHANGELOG.md` | text | docs |

## Merge notes for sibling plans

This plan lands **first**. `2026-10-01-observation-budget.md` lands second and `2026-10-01-question-is-an-answer.md` third; both edit `nodes/agent.py`. Every edit below is anchored on quoted text.

**For `question-is-an-answer` (its Task 3):**

- `_hygiene`'s unknown-tool lines now read `return refuse(UNKNOWN_TOOL_TEXT.format(name=name), kind="unknown_tool")`. Insert its `REMOVED_TOOLS` check directly above them as written. Its `refuse(REMOVED_TOOLS[name])` takes `refuse`'s default kind `"refused"`, which counts toward the hygiene budget like any slip — what its merge note asks for, with no extra wiring.
- The run-alone block (`# ask_user runs ALONE: …` through the `ASK_ALONE_TEXT` ToolMessage) is untouched here and still sits directly between the hygiene loop and `answered = [r for r in replies if r is not None]`; delete its lines as quoted there. This plan inserted nothing between them. Its replies carry no `HYGIENE_KEY` marker, so deleting them changes no count.
- `incidents()`: the line `rounds = [r for r in _rounds(this_turn) if r[4] != STALL_TEXT]` is untouched (its `_NOT_OUTCOMES` edit applies as written). This plan changed only the two `BUDGET_TEXT` comparisons to `_NOT_RUN`.
- `test_incidents_note_omits_a_call_that_later_succeeded` is untouched here.
- Docs (its Task 6): this plan's Task 6 keeps every line that plan anchors on. In `CLAUDE.md` the line `an error ToolMessage that routes straight back to `agent`, no gate, no model call) → **answer** (a message without tool calls IS` is unchanged (the loop-guards text goes in the lines above it). In `docs/ARCHITECTURE.md` only the `again is rerun once with tools unbound); …` line changes; the `identical call with nothing changed in between — …` / `without tool calls is the answer: …` pair is unchanged. In `docs/engine.md` step 5 the two `ask_user runs alone … ASK_ALONE_TEXT.` lines are unchanged and the loop-guards paragraph is appended after them, so its replacement of those two lines leaves the paragraph in place.

**For `observation-budget` (its Phase B projection):**

- `_llm_input(state, messages, extra)` now carries the failed-round note in `extra` on ordinary passes, not only the hard stop's `BUDGET_NOTE`. The projection must keep `extra` last and unprojected, and must never stub the latest round (the note's facts and the next decision come from it).
- `_replay` reads the earlier observation from STATE (`_rounds` over `state["messages"]`), so its inline copy is always whole. But for an observation over `REPLAY_INLINE_CAP` it sends `REPLAY_POINTER` — "its full result is the output of your earlier identical … call, above" — which is false if the projection stubbed that observation. When Phase B lands, either `_replay` inlines the (clamped) earlier result whenever its target is stubbed in the prompt, or the projection never stubs an observation a later replay pointer names. Pick one there and pin it.
- Re-reads after stubbing: a re-read is the model's only way back to a stubbed observation. With this plan the second identical read runs, and the third is a replay (inline or pointer, as above) — never a refusal while the last run completed. If the projection wants a re-read of a stubbed observation to count as "something changed", reset `_same_since_change` there; this plan does not.

**For every later change:** every ToolMessage the agent node writes itself carries `HYGIENE_KEY`. A new refusal goes through `refuse(...)` inside `_hygiene` (kind defaults to `"refused"`, which counts) or sets the marker explicitly.

---

### Task 1: Mark every call the agent node answers itself

**Files:**
- Modify: `textutil.py` (after `SOURCE_ENTRY_RE`)
- Modify: `nodes/agent.py` (the `textutil` import; constants after `STALL_REPEATS`; `_hygiene`'s `refuse` and its six call sites; the cap refusal in `agent_node`)
- Modify: `benchmark.py` (`agent_answered`; `run_query`)
- Test: `tests/test_agent_loop.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `textutil.HYGIENE_KEY: str` (`"saturn_hygiene"`); `nodes.agent.answered_kind(m) -> str | None`; `nodes.agent.UNCOUNTED_KINDS: tuple[str, ...]` (`("budget", "replay")`); `benchmark.agent_answered(messages: list) -> dict` with keys `"hygiene": int`, `"replays": int`. Kinds written: `unknown_tool`, `malformed`, `foreign_args`, `missing_args`, `declined_repeat`, `stall`, `budget` (and `refused`, `refuse`'s default).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_loop.py`:

```python
# ── loop guards (2026-10-01, docs/superpowers/plans/2026-10-01-loop-guards.md) ────────────────


def _refused(name, args, cid, kind, content="Error: refused", status="error"):
    """A pass whose one call the agent node answered itself, as hygiene leaves it in state."""
    from textutil import HYGIENE_KEY

    return [AIMessage(content="", tool_calls=[_call(name, args, cid)]),
            ToolMessage(content=content, tool_call_id=cid, name=name,
                        additional_kwargs={"saturn_status": status, HYGIENE_KEY: kind})]


def test_every_refusal_carries_its_kind(monkeypatch):
    """Readers key on the structural marker, never on the refusal's wording: the hygiene
    budget, the rail's label and the loop benchmark's counts all read it."""
    from nodes import agent

    bad = {"name": "read_file", "args": "a.txt", "id": "m", "error": "not json"}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("nope", {}, "u"), _call("read_file", {}, "x"),
                                _call("recall", {"fact": "f", "replaces": "#2"}, "f")],
        invalid_tool_calls=[bad]))
    out = agent.agent_node(_state([HumanMessage(content="q")]))
    kinds = {m.tool_call_id: agent.answered_kind(m) for m in out["messages"] if isinstance(m, ToolMessage)}
    assert kinds == {"u": "unknown_tool", "x": "missing_args", "f": "foreign_args", "m": "malformed"}


def test_declined_repeat_stall_and_cap_refusals_carry_their_kind(monkeypatch):
    from nodes import agent
    from nodes.approval import DECLINE_TEXT

    args = {"file_path": "x", "content": "y"}
    declined = [HumanMessage(content="q")] + _round("write_file", args, "c1", DECLINE_TEXT, "skipped")
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("write_file", args, "c2")]))
    assert agent.answered_kind(agent.agent_node(_state(declined))["messages"][-1]) == "declined_repeat"

    test = {"command": "pytest -q"}
    stalled = ([HumanMessage(content="q")] + _round("run_shell", test, "c1", "1 failed", "error")
               + _round("run_shell", test, "c2", "1 failed", "error"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("run_shell", test, "c3")]))
    assert agent.answered_kind(agent.agent_node(_state(stalled))["messages"][-1]) == "stall"

    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", {"file_path": "b"}, "c9")]))
    capped = agent.agent_node(_cap_state())["messages"][-1]
    assert capped.content == agent.BUDGET_TEXT and agent.answered_kind(capped) == "budget"


def test_a_call_a_tool_ran_has_no_kind():
    from nodes import agent

    ran = _round("read_file", {"file_path": "a"}, "c1")[1]
    assert agent.answered_kind(ran) is None
```

Append to `tests/test_cli.py`, after `test_loop_summary_by_shape`:

```python
def test_agent_answered_counts_by_marker_not_by_subtraction():
    """The loop benchmark's hygiene count reads the agent node's marker: refusals that were the
    model's slip count, a budget refusal and a replayed read do not, and a gate decline (no
    marker — a human decision) never does. The old count subtracted tool events from
    ToolMessages, which also counted gate declines."""
    import benchmark
    from langchain.messages import ToolMessage
    from nodes import agent
    from textutil import HYGIENE_KEY

    def tm(cid, kind=None, content="x", status="error"):
        kw = {"saturn_status": status}
        if kind:
            kw[HYGIENE_KEY] = kind
        return ToolMessage(content=content, tool_call_id=cid, name="read_file", additional_kwargs=kw)

    msgs = [tm("a", "unknown_tool"), tm("b", "malformed"), tm("c", "budget", agent.BUDGET_TEXT),
            tm("d", "replay", status="done"), tm("e", None, "Declined by the user.", "skipped"),
            tm("f", None, "file text", "done")]
    out = benchmark.agent_answered(msgs)
    assert out["hygiene"] == 2 and out["replays"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_cli.py -q -k "kind or agent_answered"`
Expected: 4 failed — the three agent-loop tests with `AttributeError: module 'nodes.agent' has no attribute 'answered_kind'`, and the benchmark test with `ImportError: cannot import name 'HYGIENE_KEY' from 'textutil'`.

- [ ] **Step 3: The marker key**

In `textutil.py`, directly after

```python
SOURCE_ENTRY_RE = re.compile(r"^\s*\[(\d+)\]\s")
```

add

```python

# The marker on a ToolMessage the agent node wrote ITSELF — no tool ran — naming its kind (a
# hygiene refusal, a budget refusal, a replayed read). `saturn_status` says how a call ended;
# this says it never reached the tools node. Readers key on it, never on a refusal's wording:
# nodes/agent.py writes it; the rail's label and the loop benchmark's counts read it.
HYGIENE_KEY = "saturn_hygiene"
```

- [ ] **Step 4: Write the marker in the agent node**

In `nodes/agent.py`, replace

```python
from textutil import SOURCES_HEADER, clip, fmt_args, split_sources_footer
```

with

```python
from textutil import HYGIENE_KEY, SOURCES_HEADER, clip, fmt_args, split_sources_footer
```

Directly after

```python
STALL_REPEATS = 2
```

add

```python
# Kinds of agent-answered call that are not the model's slip: a budget refusal (the turn's
# limit, not a mistake) and a replayed read (a legitimate re-read answered from context).
UNCOUNTED_KINDS = ("budget", "replay")
```

Directly after the `_rounds` function (before the `# ── the prompt ──` banner), add

```python


def answered_kind(m) -> "str | None":
    """The kind of a ToolMessage the agent node wrote itself (textutil.HYGIENE_KEY), or None for
    a call a tool ran — and for a gate decline, which is the approval node's reply to a human
    decision, not hygiene."""
    return (getattr(m, "additional_kwargs", None) or {}).get(HYGIENE_KEY)
```

In `_hygiene`, replace

```python
    def refuse(text, status="error"):
        return call, ToolMessage(content=text, tool_call_id=call["id"], name=name,
                                 additional_kwargs={"saturn_status": status})

    if name not in tools_by_name:
        return refuse(UNKNOWN_TOOL_TEXT.format(name=name))
    if malformed:
        return refuse("Error: " + schema_hint(name, "the arguments were not valid JSON"))
    raw = call.get("args")
    other = tool_for_args(name, raw)
    if other:
        return refuse("Error: " + schema_hint(
            other, f"those arguments belong to {other}, not {name}; the call was not run"))
    args = coerce_args(name, raw)
    if args is None:
        problem = ("the arguments were not an object" if not isinstance(raw, dict)
                   else f"required arguments missing from {raw!r}")
        return refuse("Error: " + schema_hint(name, problem))
    key = _call_key(name, args)
    if any(r[3] == "skipped" for r in rounds if r[0] == key):
        return refuse(ALREADY_DECLINED_TEXT, "skipped")
    if _repeats_since_change(key, rounds) >= STALL_REPEATS:
        return refuse(STALL_TEXT)
    return {**call, "args": args}, None
```

with

```python
    def refuse(text, status="error", kind="refused"):
        return call, ToolMessage(content=text, tool_call_id=call["id"], name=name,
                                 additional_kwargs={"saturn_status": status, HYGIENE_KEY: kind})

    if name not in tools_by_name:
        return refuse(UNKNOWN_TOOL_TEXT.format(name=name), kind="unknown_tool")
    if malformed:
        return refuse("Error: " + schema_hint(name, "the arguments were not valid JSON"),
                      kind="malformed")
    raw = call.get("args")
    other = tool_for_args(name, raw)
    if other:
        return refuse("Error: " + schema_hint(
            other, f"those arguments belong to {other}, not {name}; the call was not run"),
            kind="foreign_args")
    args = coerce_args(name, raw)
    if args is None:
        problem = ("the arguments were not an object" if not isinstance(raw, dict)
                   else f"required arguments missing from {raw!r}")
        return refuse("Error: " + schema_hint(name, problem), kind="missing_args")
    key = _call_key(name, args)
    if any(r[3] == "skipped" for r in rounds if r[0] == key):
        return refuse(ALREADY_DECLINED_TEXT, "skipped", kind="declined_repeat")
    if _repeats_since_change(key, rounds) >= STALL_REPEATS:
        return refuse(STALL_TEXT, kind="stall")
    return {**call, "args": args}, None
```

In `agent_node`, replace

```python
            fixed, reply = call, ToolMessage(content=BUDGET_TEXT, tool_call_id=call["id"],
                                             name=str(call.get("name") or ""),
                                             additional_kwargs={"saturn_status": "error"})
```

with

```python
            fixed, reply = call, ToolMessage(content=BUDGET_TEXT, tool_call_id=call["id"],
                                             name=str(call.get("name") or ""),
                                             additional_kwargs={"saturn_status": "error",
                                                                HYGIENE_KEY: "budget"})
```

- [ ] **Step 5: The benchmark counts by marker**

In `benchmark.py`, directly after the `_stub_texts` function, add

```python


def agent_answered(messages: list) -> dict:
    """What the agent node answered itself this turn, read off its structural marker
    (textutil.HYGIENE_KEY): `hygiene` = refused calls that were the model's slip (unknown tool,
    bad arguments, a repeat); `replays` = repeated reads answered from context. A budget
    refusal is neither, and a gate decline carries no marker (it is a human decision)."""
    from nodes.agent import UNCOUNTED_KINDS, answered_kind

    kinds = [answered_kind(m) for m in messages if isinstance(m, ToolMessage)]
    return {"hygiene": sum(1 for k in kinds if k and k not in UNCOUNTED_KINDS),
            "replays": kinds.count("replay")}
```

In `run_query`, replace

```python
        # Loop-shape facts (the loop benchmark grades these): calls the agent node answered
        # itself — a ToolMessage with no tool_events record (hygiene: unknown tool, bad
        # arguments, a repeat; the harness auto-approves, so no gate declines land here) —
        # and whether the turn ran into the pass cap. A call refused AT the cap is the cap, not
        # a hygiene bounce; a capped turn is one that needed the pass after max_iterations (its
        # calls at the cap were refused, or it lost its tools) — a text answer on the last
        # allowed pass is an ordinary finish.
        from nodes.agent import BUDGET_TEXT

        n_tool_msgs = sum(1 for m in result["messages"]
                          if isinstance(m, ToolMessage) and m.content != BUDGET_TEXT)
        hygiene = max(0, n_tool_msgs - len(result.get("tool_events") or []))
        iterations = int(result.get("iteration") or 0)
```

with

```python
        # Loop-shape facts (the loop benchmark grades these): the calls the agent node answered
        # itself, read off their structural marker (agent_answered) — hygiene refusals and
        # replayed reads; a budget refusal is neither, and a gate decline carries no marker —
        # and whether the turn ran into the pass cap. A capped turn is one that needed the pass
        # after max_iterations (its calls at the cap were refused, or it lost its tools) — a
        # text answer on the last allowed pass is an ordinary finish.
        answered = agent_answered(result["messages"])
        iterations = int(result.get("iteration") or 0)
```

and in the same function's returned dict replace

```python
            "hygiene": hygiene,
```

with

```python
            "hygiene": answered["hygiene"],
            "replays": answered["replays"],
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_cli.py -q`
Expected: all pass (the agent loop file: `58 passed`).

- [ ] **Step 7: Commit**

```bash
git add textutil.py nodes/agent.py benchmark.py tests/test_agent_loop.py tests/test_cli.py
git commit -m "agent: mark every call the agent node answers itself with its kind"
```

---

### Task 2: The hygiene budget

**Files:**
- Modify: `nodes/agent.py` (module docstring item 3; constants; `_bounced_passes`, `_stopped`; the trailer wordings and `incidents`; the cap block, the hard stop, the answer condition and the hygiene loop in `agent_node`)
- Modify: `benchmark.py` (`agent_answered`, `run_query`, the tag comment, `grade_loop_task`, `summarize_loop`, `run_loop_benchmark`, `run_loop`)
- Test: `tests/test_agent_loop.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `answered_kind`, `UNCOUNTED_KINDS`, `HYGIENE_KEY` (Task 1).
- Produces: `nodes.agent.HYGIENE_BUDGET: int` (3); `nodes.agent.BOUNCE_TEXT: str`; `nodes.agent._bounced_passes(this_turn: list) -> int`; `nodes.agent._stopped(this_turn: list) -> bool`; `nodes.agent._NOT_RUN: dict[str, str]` (refusal text → incidents wording). `benchmark.agent_answered` gains `"bounced_out": bool`; run_query entries and loop results gain `"bounced_out"`; `summarize_loop` gains `"replays"` and `"bounced_out"`; a new tag `bounced_out`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_loop.py`:

```python
def _drive(state, max_passes=40):
    """Run agent passes the way the graph does while the agent node answers every call itself
    (route → agent). Returns (final state, passes run)."""
    from nodes import agent

    for n in range(1, max_passes + 1):
        out = agent.agent_node(state)
        state = {**state, "messages": state["messages"] + out["messages"], "iteration": out["iteration"]}
        if agent.route_after_agent({"messages": state["messages"]}) == "end":
            return state, n
    raise AssertionError("the turn never ended")


def test_a_spiral_of_refused_calls_ends_within_the_budget(monkeypatch):
    """The 4b's supersession spiral (2026-09-29): `recall` called with `remember`'s arguments,
    refused, re-issued — fourteen passes, seven refusals. Now: three bad passes, one pass whose
    call is refused by the budget, one answer — five passes, and the answer's note names the
    call that did not happen."""
    from nodes import agent

    def fake(llm_input, *, tools, think=False):
        if not tools:
            return AIMessage(content="I could not update that memory.")
        return AIMessage(content="", tool_calls=[
            _call("recall", {"fact": "I live in Lisbon", "replaces": "#2"}, f"c{len(llm_input)}")])

    monkeypatch.setattr(agent, "_generate", fake)
    state, passes = _drive(_state([HumanMessage(content="I moved to Lisbon, update that")]))
    assert passes == agent.HYGIENE_BUDGET + 2
    replies = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [agent.answered_kind(m) for m in replies] == ["foreign_args"] * agent.HYGIENE_BUDGET + ["budget"]
    assert replies[-1].content == agent.BOUNCE_TEXT
    final = state["messages"][-1]
    assert final.content.startswith("I could not update that memory.")
    assert agent.INCIDENTS_NOTE_HEADER in final.content and "recall(" in final.content


def test_after_the_budget_of_bad_passes_calls_are_refused(monkeypatch):
    from nodes import agent

    prior = ([HumanMessage(content="q")] + _refused("nope", {}, "r1", "unknown_tool")
             + _refused("read_file", {}, "r2", "missing_args")
             + _refused("read_file", {}, "r3", "malformed"))
    seen = {}

    def fake(llm_input, *, tools, think=False):
        seen.update(tools=tools, think=think)
        return AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "a"}, "c9")])

    monkeypatch.setattr(agent, "_generate", fake)
    out = agent.agent_node(_state(prior, iteration=3))
    assert seen == {"tools": True, "think": False}       # still the cached prefix; never thinks
    reply = out["messages"][-1]
    assert reply.content == agent.BOUNCE_TEXT and agent.answered_kind(reply) == "budget"
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"


def test_two_slips_do_not_stop_a_long_errand(monkeypatch):
    """Review focus 1: two unrelated slips early in a long errand are not a spiral."""
    from nodes import agent

    prior = ([HumanMessage(content="q")] + _refused("nope", {}, "r1", "unknown_tool")
             + _round("read_file", {"file_path": "a"}, "c1")
             + _refused("read_file", {}, "r2", "missing_args")
             + _round("read_file", {"file_path": "b"}, "c2"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", {"file_path": "c"}, "c3")]))
    out = agent.agent_node(_state(prior, iteration=8))
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_a_bad_batch_is_one_bad_pass(monkeypatch):
    """Review focus 2: three malformed calls in one pass are one bad pass, not three."""
    from nodes import agent
    from textutil import HYGIENE_KEY

    batch = [AIMessage(content="", tool_calls=[_call("nope", {}, "a"), _call("nada", {}, "b"),
                                               _call("zip", {}, "c")])]
    batch += [ToolMessage(content="Error: unknown tool", tool_call_id=cid, name=n,
                          additional_kwargs={"saturn_status": "error", HYGIENE_KEY: "unknown_tool"})
              for cid, n in (("a", "nope"), ("b", "nada"), ("c", "zip"))]
    prior = [HumanMessage(content="q")] + batch + _refused("nope", {}, "d", "unknown_tool")
    assert agent._bounced_passes(prior) == 2
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", {"file_path": "a"}, "e")]))
    out = agent.agent_node(_state(prior, iteration=2))
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_gate_declines_and_replays_are_not_slips(monkeypatch):
    """Review focus 3: a user who says no three times has not watched the model make three
    malformed calls; a replayed read is not a slip either."""
    from nodes import agent
    from nodes.approval import DECLINE_TEXT

    declines = [HumanMessage(content="q")]
    for i in range(3):
        declines += _round("write_file", {"file_path": f"f{i}", "content": "x"}, f"d{i}",
                           DECLINE_TEXT, "skipped")
    assert agent._bounced_passes(declines) == 0
    replays = [HumanMessage(content="q")]
    for i in range(3):
        replays += _refused("read_file", {"file_path": f"f{i}"}, f"p{i}", "replay", "same", "done")
    assert agent._bounced_passes(replays) == 0
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("write_file", {"file_path": "g", "content": "y"}, "w")]))
    out = agent.agent_node(_state(declines, iteration=3))
    assert agent.route_after_agent({"messages": out["messages"]}) == "approval"


def test_a_call_refused_by_the_hygiene_budget_reads_not_run():
    from nodes import agent

    turn = [HumanMessage(content="q")] + _refused(
        "write_file", {"file_path": "b", "content": "x"}, "c9", "budget", agent.BOUNCE_TEXT)
    assert agent.incidents(turn) == [
        "write_file(file_path='b', content='x') — not run: too many malformed tool calls this turn"]
```

Append to `tests/test_cli.py`, after `test_agent_answered_counts_by_marker_not_by_subtraction`:

```python
def test_loop_grade_and_summary_count_a_bounced_out_turn():
    import benchmark
    from langchain.messages import ToolMessage
    from nodes import agent
    from textutil import HYGIENE_KEY

    stop = ToolMessage(content=agent.BOUNCE_TEXT, tool_call_id="b", name="read_file",
                       additional_kwargs={"saturn_status": "error", HYGIENE_KEY: "budget"})
    assert benchmark.agent_answered([stop])["bounced_out"] is True
    assert benchmark.agent_answered([])["bounced_out"] is False
    task = {"id": "t", "shape": "chat", "query": "q", "tools": set(), "required": [], "max_passes": 6}
    assert "bounced_out" in benchmark.grade_loop_task(task, _loop_entry(bounced_out=True, iterations=5))
    s = benchmark.summarize_loop([
        {"id": "a", "shape": "chat", "tags": ["bounced_out"], "iterations": 5, "latency_s": 1.0,
         "hygiene": 3, "capped": False, "bounced_out": True, "replays": 1},
        {"id": "b", "shape": "chat", "tags": [], "iterations": 1, "latency_s": 1.0,
         "hygiene": 0, "capped": False},
    ])
    assert s["bounced_out"] == 1 and s["replays"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_cli.py -q -k "spiral or budget_of_bad or two_slips or bad_batch or not_slips or reads_not_run or bounced_out"`
Expected: 6 failed, 1 passed — `AttributeError: module 'nodes.agent' has no attribute 'HYGIENE_BUDGET'` / `'_bounced_passes'` / `'BOUNCE_TEXT'`, and `KeyError: 'bounced_out'` for the benchmark test. `test_two_slips_do_not_stop_a_long_errand` passes already — it pins what must stay true.

- [ ] **Step 3: The constants and the two predicates**

In `nodes/agent.py`, directly after

```python
BUDGET_NOTE = ("The action budget for this turn is spent. Answer now from what you have, and "
               "state plainly what was not done.")
```

add

```python
BOUNCE_TEXT = ("Not executed: too many of this turn's tool calls were malformed or refused, so no "
               "further tool call will run. Answer now from what you have, and state plainly what "
               "was not done.")
```

Directly after the `UNCOUNTED_KINDS = ("budget", "replay")` line (Task 1), add

```python
# Passes with at least one refused call (a counted kind) before the turn stops taking calls:
# three slips is a spiral, not a typo (the 4b's supersession miss, 2026-09-29: seven refusals
# over fourteen passes — and its answer claimed the update).
HYGIENE_BUDGET = 3
```

Directly after the `answered_kind` function (Task 1), add

```python


def _bounced_passes(this_turn: list) -> int:
    """How many passes this turn had at least one call the agent node refused as the model's
    slip. Counted per PASS, not per call: three malformed calls in one batch are one bad pass."""
    owner: dict = {}
    for i, m in enumerate(this_turn):
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                owner[tc.get("id")] = i
    return len({owner.get(m.tool_call_id) for m in this_turn
                if isinstance(m, ToolMessage) and answered_kind(m)
                and answered_kind(m) not in UNCOUNTED_KINDS})


def _stopped(this_turn: list) -> bool:
    """Whether a budget refusal (the cap's or the hygiene budget's) already answered a call
    this turn — from then on the turn ends in an answer."""
    return any(isinstance(m, ToolMessage) and str(m.content) in _NOT_RUN for m in this_turn)
```

- [ ] **Step 4: The incidents note words a budget refusal as "not run"**

In `nodes/agent.py`, replace

```python
_BUDGET_WORDING = "not run: the turn's action budget was spent"
```

with

```python
_BUDGET_WORDING = "not run: the turn's action budget was spent"
_BOUNCE_WORDING = "not run: too many malformed tool calls this turn"
# The budget refusals — the cap's and the hygiene budget's — each with how the incidents note
# words a call it refused. A call one of them refused never ran.
_NOT_RUN = {BUDGET_TEXT: _BUDGET_WORDING, BOUNCE_TEXT: _BOUNCE_WORDING}
```

In `incidents`, replace

```python
    guard's and the cap's refusals are not outcomes: a stalled call already ran twice, and those
    runs are what happened to it; a call refused at the cap is reported as not run, unless the
    same call did run earlier in the turn."""
```

with

```python
    guard's and the budgets' refusals are not outcomes: a stalled call already ran twice, and
    those runs are what happened to it; a call refused by a budget (the cap's or the hygiene
    budget's) is reported as not run, unless the same call did run earlier in the turn."""
```

then replace

```python
    last = {key: status for key, _n, _a, status, obs in rounds if obs != BUDGET_TEXT}
```

with

```python
    last = {key: status for key, _n, _a, status, obs in rounds if obs not in _NOT_RUN}
```

and replace

```python
        if obs == BUDGET_TEXT:
            if key in last:
                continue
            why = _BUDGET_WORDING
```

with

```python
        if obs in _NOT_RUN:
            if key in last:
                continue
            why = _NOT_RUN[obs]
```

- [ ] **Step 5: The budget in `agent_node`**

Replace

```python
    # 3. the cap — from pass max_iterations on, no tool call runs. The pass itself is unchanged
    # (tools stay bound, nothing is appended), so its prompt extends the cached prefix; a call
    # it emits is refused in step 5 and the refusal routes back here for the answer.
    cap = get_config().max_iterations
    capped = iteration >= cap

    # 4. generate — a malformed model output is retried once, then answered honestly; any
    # other failure propagates (the REPL reports "Turn failed").
    this_turn = _this_turn(messages + new)
    think = _wants_think(this_turn, capped)
```

with

```python
    # 3. the cap — from pass max_iterations on, no tool call runs. The pass itself is unchanged
    # (tools stay bound, nothing is appended), so its prompt extends the cached prefix; a call
    # it emits is refused in step 5 and the refusal routes back here for the answer. The
    # hygiene budget is the same stop, early: after HYGIENE_BUDGET passes with a refused call,
    # this pass's calls are refused with BOUNCE_TEXT. Once either refusal has answered a call
    # (`stopped`), this pass is the answer — a model that calls again loses its tools below.
    cap = get_config().max_iterations
    this_turn = _this_turn(messages + new)
    stopped = _stopped(this_turn)
    capped = iteration >= cap or stopped or _bounced_passes(this_turn) >= HYGIENE_BUDGET
    ending = iteration > cap or stopped

    # 4. generate — a malformed model output is retried once, then answered honestly; any
    # other failure propagates (the REPL reports "Turn failed").
    think = _wants_think(this_turn, capped)
```

Replace

```python
    if ai is not None and iteration > cap and _calls_of(ai)[0]:
```

with

```python
    if ai is not None and ending and _calls_of(ai)[0]:
```

Replace

```python
    if not calls or iteration > cap:
```

with

```python
    if not calls or ending:
```

Replace

```python
    for call in calls:
        if capped:  # the budget is spent: nothing runs, whatever the call is
            fixed, reply = call, ToolMessage(content=BUDGET_TEXT, tool_call_id=call["id"],
```

with

```python
    stop_text = BUDGET_TEXT if iteration >= cap else BOUNCE_TEXT
    for call in calls:
        if capped:  # a budget is spent: nothing runs, whatever the call is
            fixed, reply = call, ToolMessage(content=stop_text, tool_call_id=call["id"],
```

In the module docstring, replace

```python
                is rerun once with tools UNBOUND and a budget note — a real answer, never a
                stub (unbinding re-prefills the whole prompt, so it is the fallback only);
```

with

```python
                is rerun once with tools UNBOUND and a budget note — a real answer, never a
                stub (unbinding re-prefills the whole prompt, so it is the fallback only).
                The hygiene budget is the same stop, early: after HYGIENE_BUDGET passes with
                a refused call, the next pass's calls get BOUNCE_TEXT, then the turn answers;
```

- [ ] **Step 6: The benchmark reports a bounced-out turn**

In `benchmark.py`, in `agent_answered`, replace

```python
    from nodes.agent import UNCOUNTED_KINDS, answered_kind

    kinds = [answered_kind(m) for m in messages if isinstance(m, ToolMessage)]
    return {"hygiene": sum(1 for k in kinds if k and k not in UNCOUNTED_KINDS),
            "replays": kinds.count("replay")}
```

with

```python
    from nodes.agent import BOUNCE_TEXT, UNCOUNTED_KINDS, answered_kind

    kinds = [answered_kind(m) for m in messages if isinstance(m, ToolMessage)]
    return {"hygiene": sum(1 for k in kinds if k and k not in UNCOUNTED_KINDS),
            "replays": kinds.count("replay"),
            "bounced_out": any(isinstance(m, ToolMessage) and m.content == BOUNCE_TEXT
                               for m in messages)}
```

In `run_query`'s returned dict, replace

```python
            "replays": answered["replays"],
```

with

```python
            "replays": answered["replays"],
            "bounced_out": answered["bounced_out"],
```

In the tag comment block above `LOOP_SHAPES`, replace

```python
#   capped             the turn ran past runtime.max_iterations (the budget answer)
```

with

```python
#   capped             the turn ran past runtime.max_iterations (the budget answer)
#   bounced_out        the hygiene budget stopped the turn (three passes with a refused call)
```

In `grade_loop_task`, replace

```python
    if entry.get("capped"):
        tags.append("capped")
```

with

```python
    if entry.get("capped"):
        tags.append("capped")
    if entry.get("bounced_out"):
        tags.append("bounced_out")
```

In `summarize_loop`'s returned dict, replace

```python
        "capped": sum(1 for r in results if r.get("capped")),
    }
```

with

```python
        "capped": sum(1 for r in results if r.get("capped")),
        "bounced_out": sum(1 for r in results if r.get("bounced_out")),
        "replays": sum(int(r.get("replays") or 0) for r in results),
    }
```

In `run_loop_benchmark`, replace

```python
                "capped": entry.get("capped", False),
```

with

```python
                "capped": entry.get("capped", False),
                "bounced_out": entry.get("bounced_out", False),
                "replays": entry.get("replays", 0),
```

In `run_loop`, replace

```python
          f"hygiene bounces {s['hygiene']} · capped {s['capped']}")
```

with

```python
          f"hygiene bounces {s['hygiene']} · bounced out {s['bounced_out']} · "
          f"replays {s['replays']} · capped {s['capped']}")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_cli.py -q`
Expected: all pass. In particular the four existing cap tests (`test_at_the_cap_calls_are_refused_and_the_prompt_is_unchanged`, `test_after_the_refusal_the_answer_names_what_was_not_run`, `test_a_model_that_keeps_calling_gets_the_tools_taken_away`, `test_cap_lands_on_the_max_iterations_pass`) pass unchanged — the cap's behaviour is the same.

- [ ] **Step 8: Commit**

```bash
git add nodes/agent.py benchmark.py tests/test_agent_loop.py tests/test_cli.py
git commit -m "agent: a hygiene budget — three passes with a refused call end the turn"
```

---

### Task 3: The failure-aware final pass

**Files:**
- Modify: `nodes/agent.py` (module docstring item 4; `FAILED_ROUND_NOTE`; `_status`, `_round_notes`; the prompt and hard-stop lines in `agent_node`)
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `_latest_round(this_turn)`, `incidents(this_turn)`, `_INCIDENT_STATUSES`, `_NOT_RUN` (Task 2), `_llm_input(state, messages, extra=None)`.
- Produces: `nodes.agent.FAILED_ROUND_NOTE: str` (a format string with `{lines}`); `nodes.agent._status(m) -> str`; `nodes.agent._round_notes(this_turn: list) -> list[HumanMessage]` (empty or one note).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_loop.py`:

```python
def _prompt_of(monkeypatch, state):
    """The prompt the next pass sends, and the node's output (the model answers "ok")."""
    from nodes import agent

    seen = {}
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: seen.setdefault("input", i) and AIMessage(content="ok"))
    out = agent.agent_node(state)
    return seen["input"], out


def test_after_a_round_where_nothing_happened_the_model_is_told(monkeypatch):
    """That spiral's answer was "I've updated my memory" over an incidents note saying it had
    not. The model now reads the same facts as the user, at the prompt's tail — never in state
    (a HumanMessage there would read as a new turn), never before the turn's last message (the
    cached prefix holds)."""
    from nodes import agent
    from nodes.approval import DECLINE_TEXT

    prior = [HumanMessage(content="save it")] + _round(
        "write_file", {"file_path": "a", "content": "x"}, "c1", DECLINE_TEXT, "skipped")
    llm_input, out = _prompt_of(monkeypatch, _state(prior))
    note = llm_input[-1]
    assert isinstance(note, HumanMessage)
    assert note.content.startswith("None of the calls in your last round happened")
    assert "write_file(file_path='a', content='x') — declined at the approval gate — not done" in note.content
    assert "do not write as if they were" in note.content
    assert [m.content for m in llm_input[:-1]] == [m.content for m in agent._llm_input(_state(prior), prior)]
    assert not any(isinstance(m, HumanMessage) for m in out["messages"])


def test_a_refused_round_is_named_in_the_note(monkeypatch):
    prior = [HumanMessage(content="q")] + _refused(
        "recall", {"fact": "I live in Lisbon", "replaces": "#2"}, "c1", "foreign_args",
        "Error: those arguments belong to remember")
    llm_input, _ = _prompt_of(monkeypatch, _state(prior))
    assert "recall(fact='I live in Lisbon', replaces='#2') — failed: Error: those arguments" in llm_input[-1].content


def test_no_note_when_anything_in_the_round_happened(monkeypatch):
    clean = [HumanMessage(content="q")] + _round("read_file", {"file_path": "a"}, "c1")
    mixed = [HumanMessage(content="q"),
             AIMessage(content="", tool_calls=[_call("read_file", {"file_path": "a"}, "c1"),
                                               _call("read_file", {"file_path": "b"}, "c2")]),
             ToolMessage(content="ok", tool_call_id="c1", name="read_file",
                         additional_kwargs={"saturn_status": "done"}),
             ToolMessage(content="Error: gone", tool_call_id="c2", name="read_file",
                         additional_kwargs={"saturn_status": "error"})]
    for prior in (clean, mixed):
        llm_input, _ = _prompt_of(monkeypatch, _state(prior))
        assert isinstance(llm_input[-1], ToolMessage)


def test_no_note_after_a_stall_or_a_budget_refusal(monkeypatch):
    """A stalled call already ran twice (its results are above); a budget refusal already says
    to answer and name what was not done."""
    from nodes import agent

    args = {"file_path": "a"}
    stalled = ([HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
               + _round("read_file", args, "c3", agent.STALL_TEXT, "error"))
    budget = [HumanMessage(content="q")] + _round("read_file", {"file_path": "b"}, "c1", agent.BUDGET_TEXT, "error")
    for prior in (stalled, budget):
        llm_input, _ = _prompt_of(monkeypatch, _state(prior))
        assert isinstance(llm_input[-1], ToolMessage)


def test_the_note_survives_a_steer(monkeypatch):
    """Review focus 4: a steer typed after a round where nothing happened does not hide that
    round; the note comes after the steer."""
    from core.state import STEER_PREFIX

    prior = ([HumanMessage(content="q")]
             + _round("web_search", {"query": "z"}, "c1", "air-gap refused", "blocked")
             + [HumanMessage(content=f"{STEER_PREFIX} try the file instead")])
    llm_input, _ = _prompt_of(monkeypatch, _state(prior))
    assert llm_input[-2].content.startswith(STEER_PREFIX)
    assert "web_search(query='z') — blocked by the air-gap — nothing was sent" in llm_input[-1].content
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py -q -k "nothing_happened or refused_round or no_note or survives_a_steer"`
Expected: 3 failed — `test_after_a_round_where_nothing_happened_the_model_is_told`, `test_a_refused_round_is_named_in_the_note` and `test_the_note_survives_a_steer` (the last prompt message is the ToolMessage / the steer, not a note). The two `no_note` tests pass already — they pin what must stay true.

- [ ] **Step 3: The note**

In `nodes/agent.py`, directly after the `BOUNCE_TEXT = …` assignment (Task 2), add

```python
FAILED_ROUND_NOTE = ("None of the calls in your last round happened:\n{lines}\nIf you answer now, "
                     "say plainly that these were not done; do not write as if they were.")
```

Directly after the `answered_kind` function (Task 1), add

```python


def _status(m) -> str:
    """A ToolMessage's structural outcome stamp (nodes/tools.py, nodes/approval.py, hygiene)."""
    return (getattr(m, "additional_kwargs", None) or {}).get("saturn_status") or "done"
```

Directly after the `_is_empty` function (before the `# ── hygiene ──` banner), add

```python


# ── the failed-round note ─────────────────────────────────────────────────────────────────────


def _round_notes(this_turn: list) -> list:
    """[the failed-round note] when NOTHING happened in the round this pass reacts to — every
    call declined, air-gapped, failed or refused — else []. Prompt-only (`_llm_input`'s
    `extra`): never state, where a HumanMessage would read as a new turn. It names the calls in
    the incidents note's own words, so the model and the user read one account, and it does
    not say "answer now": a different call may still be the right move. Silent after a round
    in which anything completed, after a stall (the call already ran; its results are above —
    `incidents` drops a stall) and after a budget refusal (which already says what to do)."""
    rnd = _latest_round(this_turn)
    if not rnd or any(_status(m) not in _INCIDENT_STATUSES for m in rnd):
        return []
    if all(str(m.content) in _NOT_RUN for m in rnd):
        return []
    ids = {m.tool_call_id for m in rnd}
    issuing = next((m for m in reversed(this_turn) if isinstance(m, AIMessage)
                    and any(tc.get("id") in ids for tc in getattr(m, "tool_calls", None) or [])),
                   None)
    lines = incidents([issuing, *reversed(rnd)]) if issuing is not None else []
    if not lines:
        return []
    return [HumanMessage(content=FAILED_ROUND_NOTE.format(lines="\n".join(f"- {line}" for line in lines)))]
```

- [ ] **Step 4: Send it**

In `agent_node`, replace

```python
    think = _wants_think(this_turn, capped)
    llm_input = _llm_input(state, messages + new)
```

with

```python
    think = _wants_think(this_turn, capped)
    notes = _round_notes(this_turn)  # [] on the clean path: the prompt bytes are unchanged
    llm_input = _llm_input(state, messages + new, notes)
```

and replace

```python
        ai = _generate_or_retry(_llm_input(state, messages + new, [HumanMessage(content=BUDGET_NOTE)]),
                                tools=False, think=False)
```

with

```python
        ai = _generate_or_retry(_llm_input(state, messages + new,
                                           notes + [HumanMessage(content=BUDGET_NOTE)]),
                                tools=False, think=False)
```

In the module docstring, replace

```python
                declined or blocked call and the capped passes stay think-off. A thinking pass
                that returns neither text nor a call is rerun once think-off;
```

with

```python
                declined or blocked call and the capped passes stay think-off. A thinking pass
                that returns neither text nor a call is rerun once think-off. A pass after a
                round in which nothing happened gets the failed-round note at the prompt's
                tail (never state): those calls, in the incidents note's words;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_prefix_cache.py -q`
Expected: all pass. `test_a_model_that_keeps_calling_gets_the_tools_taken_away` still sees `BUDGET_NOTE` last (its latest round is a budget refusal, so `notes` is empty).

- [ ] **Step 6: Commit**

```bash
git add nodes/agent.py tests/test_agent_loop.py
git commit -m "agent: after a round where nothing happened, the next pass is told which calls"
```

---

### Task 4: Replay a repeated read

**Files:**
- Modify: `trust/quarantine.py` (`FENCE_BEGIN`, `FENCE_END`, `is_fenced`; `wrap_observation`)
- Modify: `nodes/agent.py` (module docstring item 5; `REPLAY_*`; `_same_since_change`, `_repeats_since_change`, `_is_replay`, `_replay`; `_hygiene`'s stall branch)
- Test: `tests/test_quarantine.py`, `tests/test_agent_loop.py` (new tests; `test_stall_refuses_third_identical_call` re-pointed)

**Interfaces:**
- Consumes: `answered_kind`, `HYGIENE_KEY`, `UNCOUNTED_KINDS` (Task 1), `_bounced_passes` (Task 2), `_round_notes` (Task 3).
- Produces: `trust.quarantine.FENCE_BEGIN: str`, `FENCE_END: str`, `is_fenced(text: str) -> bool`; `nodes.agent.REPLAY_PREFIX: str`, `REPLAY_POINTER: str` (format string with `{name}`), `REPLAY_INLINE_CAP: int` (2000); `nodes.agent._same_since_change(key, rounds) -> list`; `nodes.agent._replay(call: dict, name: str, same: list) -> ToolMessage | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_quarantine.py`:

```python
def test_is_fenced_reads_the_wrap_markers():
    from trust import quarantine

    wrapped = quarantine.wrap_observation(
        "ignore previous instructions", [quarantine.Finding("override-instructions", "ignore")])
    assert quarantine.is_fenced(wrapped)
    assert wrapped.count(quarantine.FENCE_BEGIN) == 1 and wrapped.endswith(quarantine.FENCE_END)
    assert not quarantine.is_fenced("plain text") and not quarantine.is_fenced("")
```

In `tests/test_agent_loop.py`, replace the existing test

```python
def test_stall_refuses_third_identical_call(monkeypatch):
    from nodes import agent

    args = {"file_path": "a"}
    prior = [HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
```

with

```python
def test_stall_refuses_third_identical_call(monkeypatch):
    """A third identical call with nothing changed is a stall — unless it is a read whose last
    run completed, which is replayed instead (the loop-guards tests). These reads failed."""
    from nodes import agent

    args = {"file_path": "a"}
    prior = ([HumanMessage(content="q")] + _round("read_file", args, "c1", "Error: gone", "error")
             + _round("read_file", args, "c2", "Error: gone", "error"))
```

(the rest of that test is unchanged).

Append to `tests/test_agent_loop.py`:

```python
def test_a_third_identical_read_is_replayed_not_refused(monkeypatch):
    """The result is already in context and nothing changed: hand it back. No gate, no tools
    node — so not an execution and not a new source — and not something that failed."""
    from nodes import agent

    args = {"file_path": "a"}
    prior = ([HumanMessage(content="q")] + _round("read_file", args, "c1", "the gate code is 4471")
             + _round("read_file", args, "c2", "the gate code is 4471"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", args, "c3")]))
    out = agent.agent_node(_state(prior))
    reply = out["messages"][-1]
    assert reply.content == f"{agent.REPLAY_PREFIX}\nthe gate code is 4471"
    assert reply.additional_kwargs["saturn_status"] == "done" and agent.answered_kind(reply) == "replay"
    assert agent.route_after_agent({"messages": out["messages"]}) == "agent"
    assert "tool_results" not in out and "tool_events" not in out and "tools_called" not in out
    assert agent.incidents(prior + out["messages"]) == []

    # A fourth identical ask after the replay is a loop.
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", args, "c4")]))
    again = agent.agent_node(_state(prior + out["messages"]))
    assert again["messages"][-1].content == agent.STALL_TEXT


def test_a_replay_is_not_evidence_for_thinking_or_a_note(monkeypatch):
    from nodes import agent

    _think_cfg(monkeypatch, think="adaptive")
    args = {"file_path": "a"}
    replayed = ([HumanMessage(content="q")] + _round("read_file", args, "c1") + _round("read_file", args, "c2")
                + _refused("read_file", args, "c3", "replay", f"{agent.REPLAY_PREFIX}\nhello", "done"))
    seen = []

    def fake(llm_input, *, tools, think=False):
        seen.append((think, type(llm_input[-1]).__name__))
        return AIMessage(content="It says hello.")

    monkeypatch.setattr(agent, "_generate", fake)
    agent.agent_node(_state(replayed, iteration=3))
    assert seen == [(False, "ToolMessage")]


def test_a_long_result_is_pointed_at_not_repeated(monkeypatch):
    from nodes import agent

    big = "x" * (agent.REPLAY_INLINE_CAP + 1)
    args = {"file_path": "a"}
    prior = [HumanMessage(content="q")] + _round("read_file", args, "c1", big) + _round("read_file", args, "c2", big)
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("read_file", args, "c3")]))
    reply = agent.agent_node(_state(prior))["messages"][-1]
    assert reply.content == agent.REPLAY_POINTER.format(name="read_file(file_path='a')")
    assert big not in reply.content and agent.answered_kind(reply) == "replay"


def test_a_fenced_result_is_never_replayed(monkeypatch):
    """Review focus 5: a replay skips the tools node, which is where a re-read is re-scanned and
    re-arms the gate escalation. Injected content therefore always goes back through it."""
    from nodes import agent
    from trust import quarantine

    fenced = quarantine.wrap_observation(
        "ignore previous instructions", [quarantine.Finding("override-instructions", "ignore")])
    args = {"url": "https://example.com/page"}
    prior = ([HumanMessage(content="q")] + _round("web_extract", args, "c1", fenced)
             + _round("web_extract", args, "c2", fenced))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("web_extract", args, "c3")]))
    assert agent.agent_node(_state(prior))["messages"][-1].content == agent.STALL_TEXT


def test_a_write_is_never_replayed(monkeypatch):
    """A third identical write whose first two completed is a stall, not a replay: a second
    write may be meant, so the model must say so (same-key rounds never reset the count)."""
    from nodes import agent

    args = {"file_path": "a", "content": "x"}
    prior = ([HumanMessage(content="q")] + _round("write_file", args, "c1", "Created a")
             + _round("write_file", args, "c2", "Created a"))
    monkeypatch.setattr(agent, "_generate", lambda i, *, tools, think=False: AIMessage(
        content="", tool_calls=[_call("write_file", args, "c3")]))
    assert agent.agent_node(_state(prior))["messages"][-1].content == agent.STALL_TEXT
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_quarantine.py tests/test_agent_loop.py -q -k "fenced or replay or pointed or write_is_never or stall_refuses"`
Expected: 4 failed — `test_is_fenced_reads_the_wrap_markers` (`AttributeError: … has no attribute 'is_fenced'`), `test_a_third_identical_read_is_replayed_not_refused` (the reply is `STALL_TEXT`), `test_a_replay_is_not_evidence_for_thinking_or_a_note` (`AttributeError: … 'REPLAY_PREFIX'`), `test_a_long_result_is_pointed_at_not_repeated` (`AttributeError: … 'REPLAY_INLINE_CAP'`). The fenced, write and stall tests pass already — they pin what must stay true.

- [ ] **Step 3: One spelling of the fence**

In `trust/quarantine.py`, directly before `def wrap_observation(`, add

```python
# The fence wrap_observation writes around a flagged observation — one spelling, read back by
# is_fenced (nodes/agent.py never replays a fenced result: a re-read must be re-scanned).
FENCE_BEGIN = "<<<UNTRUSTED CONTENT BEGIN>>>"
FENCE_END = "<<<UNTRUSTED CONTENT END>>>"


```

In `wrap_observation`, replace

```python
        "<<<UNTRUSTED CONTENT BEGIN>>>\n"
        + observation
        + "\n<<<UNTRUSTED CONTENT END>>>"
```

with

```python
        + FENCE_BEGIN + "\n"
        + observation
        + "\n" + FENCE_END
```

Directly after the `wrap_observation` function, add

```python


def is_fenced(text: str) -> bool:
    """Whether an observation carries the quarantine fence (it was flagged when it arrived)."""
    return FENCE_BEGIN in str(text or "")
```

- [ ] **Step 4: The replay**

In `nodes/agent.py`, directly after the `FAILED_ROUND_NOTE = …` assignment (Task 3), add

```python
REPLAY_PREFIX = ("(Same result as before: nothing has changed since this exact call last ran, so "
                 "it was not run again.)")
REPLAY_POINTER = ("(Same result as before: nothing has changed since this exact call last ran, so "
                  "it was not run again. Its full result is the output of your earlier identical "
                  "{name} call, above.)")
```

Directly after `HYGIENE_BUDGET = 3` (Task 2), add

```python
# A replayed result longer than this is pointed at, not repeated: it is already in the prompt,
# and a second copy would only spend the window.
REPLAY_INLINE_CAP = 2000
```

Replace the whole `_repeats_since_change` function

```python
def _repeats_since_change(key: str, rounds: list) -> int:
    """How many times this exact call has been made since something last changed. A completed
    call to a tool that is not read_only (an edit, a write, a command) resets the count: the
    same `pytest -q` after an edit is a new question, not a repeat."""
    from tools.registry import DECLARED_RISK

    n = 0
    for k, name, _args, status, _obs in rounds:
        if k == key:
            n += 1
        elif status == "done" and DECLARED_RISK.get(name, "destructive") != "read_only":
            n = 0
    return n
```

with

```python
def _same_since_change(key: str, rounds: list) -> list:
    """The earlier rounds of this exact call since something last changed. A completed call to
    a DIFFERENT tool call that is not read_only (an edit, a write, a command) resets them: the
    same `pytest -q` after an edit is a new question, not a repeat."""
    from tools.registry import DECLARED_RISK

    same: list = []
    for r in rounds:
        k, name, _args, status, _obs = r
        if k == key:
            same.append(r)
        elif status == "done" and DECLARED_RISK.get(name, "destructive") != "read_only":
            same = []
    return same


def _repeats_since_change(key: str, rounds: list) -> int:
    """How many times this exact call has been made since something last changed."""
    return len(_same_since_change(key, rounds))


def _is_replay(observation: str) -> bool:
    return str(observation).startswith(REPLAY_PREFIX[:24])  # both replay texts open the same way


def _replay(call: dict, name: str, same: list) -> "ToolMessage | None":
    """The earlier result handed back for a repeated read, or None when the repeat must be
    refused instead: the tool is not read_only (a second write may be meant), its last run did
    not complete (a failure needs a new approach, not its error again), it was already replayed
    since anything changed (a fourth ask is a loop), or the result was fenced by the quarantine
    (a re-read must go back through the tools node, which re-scans it and re-arms the gate)."""
    from tools.registry import DECLARED_RISK
    from trust import quarantine

    if not same or DECLARED_RISK.get(name, "destructive") != "read_only":
        return None
    _key, _name, args, status, obs = same[-1]
    if status != "done" or quarantine.is_fenced(obs) or any(_is_replay(r[4]) for r in same):
        return None
    if len(obs) > REPLAY_INLINE_CAP:
        content = REPLAY_POINTER.format(name=f"{name}({fmt_args(args or {}, 60)})")
    else:
        content = f"{REPLAY_PREFIX}\n{obs}"
    return ToolMessage(content=content, tool_call_id=call["id"], name=name,
                       additional_kwargs={"saturn_status": "done", HYGIENE_KEY: "replay"})
```

In `_hygiene`, replace

```python
    if _repeats_since_change(key, rounds) >= STALL_REPEATS:
        return refuse(STALL_TEXT, kind="stall")
    return {**call, "args": args}, None
```

with

```python
    same = _same_since_change(key, rounds)
    if len(same) >= STALL_REPEATS:
        replay = _replay(call, name, same)
        if replay is not None:
            return {**call, "args": args}, replay
        return refuse(STALL_TEXT, kind="stall")
    return {**call, "args": args}, None
```

In the module docstring, replace

```python
  5. hygiene    on each emitted call: unknown tool, missing arguments (core/tool_args),
                a repeat of a call the user DECLINED this turn, or a third identical call
                with nothing changed since the first — each answered with an error ToolMessage
                that routes straight back here;
```

with

```python
  5. hygiene    on each emitted call: unknown tool, missing arguments (core/tool_args),
                a repeat of a call the user DECLINED this turn, or a third identical call
                with nothing changed since the first — each answered with an error ToolMessage
                that routes straight back here. A third identical READ whose last run
                completed is answered once with that result instead (a replay). Every reply
                this node writes carries its kind (textutil.HYGIENE_KEY);
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_quarantine.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add trust/quarantine.py nodes/agent.py tests/test_agent_loop.py tests/test_quarantine.py
git commit -m "agent: a third identical read is answered from what it already returned"
```

---

### Task 5: The rail labels calls the agent node answered itself

**Files:**
- Modify: `tui/ui/trace.py` (the `textutil` import; `_render_agent_thought`; new `_answered_label`)
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `textutil.HYGIENE_KEY` (Task 1); `nodes.agent.REPLAY_PREFIX`, `UNKNOWN_TOOL_TEXT` (tests only).
- Produces: `tui.ui.trace._answered_label(m, call: dict) -> tuple[str, str]` (text, style).

Today `_render_agent_thought` renders `messages[-1]` — for a pass whose calls the agent node answered, that is the last refusal, shown as if it were the agent's thought, and every other refusal in the pass is invisible (no tool ran, so the tools sub-tree never shows them).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_loop.py`:

```python
def test_rail_labels_calls_the_agent_answered_itself(capsys, monkeypatch):
    from nodes import agent
    from textutil import HYGIENE_KEY

    trace = _plain_rail(monkeypatch)
    ai = AIMessage(content="checking again", tool_calls=[_call("read_file", {"file_path": "a"}, "r"),
                                                         _call("nope", {}, "n")])
    replay = ToolMessage(content=f"{agent.REPLAY_PREFIX}\nhello", tool_call_id="r", name="read_file",
                         additional_kwargs={"saturn_status": "done", HYGIENE_KEY: "replay"})
    refusal = ToolMessage(content=agent.UNKNOWN_TOOL_TEXT.format(name="nope"), tool_call_id="n",
                          name="nope", additional_kwargs={"saturn_status": "error", HYGIENE_KEY: "unknown_tool"})
    trace.show_node("agent", {"messages": [ai, replay, refusal], "iteration": 3})
    out = capsys.readouterr().out
    assert "checking again" in out
    assert "↺ read_file(file_path='a')" in out and "same result as before" in out
    assert "✗ nope() not run" in out and "unknown tool" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py -q -k rail_labels`
Expected: FAIL — `checking again` is missing (the rail rendered only the last message, the refusal).

- [ ] **Step 3: Render the thought and one leaf per answered call**

In `tui/ui/trace.py`, replace

```python
from textutil import clip, fmt_args, fmt_call, human_bytes, split_call_result
```

with

```python
from textutil import HYGIENE_KEY, clip, fmt_args, fmt_call, human_bytes, split_call_result
```

Replace the whole `_render_agent_thought` function

```python
def _render_agent_thought(messages: list) -> None:
    """Render the agent's message text as dim, wrapped leaf lines under its trace row: the
    pre-call thought alongside its tool calls (the same words the gate's `e(xplain)` shows).
    Quietly does nothing when the message has no text."""
    msg = messages[-1] if messages else None
    if msg is None:
        return
    text = msg.content if isinstance(getattr(msg, "content", ""), str) else str(getattr(msg, "content", ""))
    text = clip(text, _REASONING_CAP)
    if not text:
        return
    _node_leaf(text, _DIM)
```

with

```python
def _render_agent_thought(messages: list) -> None:
    """Render the agent's message text as dim, wrapped leaf lines under its trace row: the
    pre-call thought alongside its tool calls (the same words the gate's `e(xplain)` shows) —
    then one leaf per call the agent node answered itself, so a refused call and a replayed
    read show where they happened (no tool ran, so the tools sub-tree never shows them).
    Quietly does nothing for a pass with neither."""
    ai = next((m for m in reversed(messages) if getattr(m, "type", "") == "ai"), None)
    if ai is not None:
        content = getattr(ai, "content", "")
        text = clip(content if isinstance(content, str) else str(content), _REASONING_CAP)
        if text:
            _node_leaf(text, _DIM)
    calls = {tc.get("id"): tc for tc in (getattr(ai, "tool_calls", None) or [])}
    for m in messages:
        if getattr(m, "type", "") == "tool":
            text, style = _answered_label(m, calls.get(getattr(m, "tool_call_id", None)) or {})
            _node_leaf(text, style)


def _answered_label(m, call: dict) -> "tuple[str, str]":
    """The leaf for a call the agent node answered without running it (textutil.HYGIENE_KEY):
    a replayed read says so; a refusal says it was not run, with the first line of why."""
    name = str(getattr(m, "name", "") or call.get("name") or "?")
    head = f"{name}({fmt_args(call.get('args') or {}, 40)})"
    if (getattr(m, "additional_kwargs", None) or {}).get(HYGIENE_KEY) == "replay":
        return f"↺ {head} — same result as before, not run again", _DIM
    why = (str(getattr(m, "content", "") or "").strip().splitlines() or [""])[0]
    return f"✗ {head} not run — {clip(why, 120)}", "yellow"
```

- [ ] **Step 4: Run the rail and turn tests**

Run: `.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_tui_polish.py tests/test_turn_display_guard.py -q`
Expected: all pass (`test_rail_agent_row_hidden_for_the_answer_shown_for_a_call` still sees "reading it").

- [ ] **Step 5: Commit**

```bash
git add tui/ui/trace.py tests/test_agent_loop.py
git commit -m "rail: a leaf for each call the agent node refused or replayed"
```

---

### Task 6: The docs say what is true

**Files:**
- Modify: `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/engine.md` (the shape, step 5), `CHANGELOG.md`

- [ ] **Step 1: CLAUDE.md**

Replace

```
  note — a real answer, never a stub) → generate → **hygiene** on each emitted call (unknown tool,
  missing arguments via `core/tool_args.coerce_args`, malformed JSON, a repeat of a call the user
  DECLINED this turn, a third identical call with nothing changed in between — each answered with
  an error ToolMessage that routes straight back to `agent`, no gate, no model call) → **answer** (a message without tool calls IS
```

with

```
  note — a real answer, never a stub; after `HYGIENE_BUDGET` (3) passes with a refused call the
  same stop comes early, as `BOUNCE_TEXT`) → generate (a pass after a round in which nothing
  happened gets a prompt-only note naming those calls, so the answer cannot claim them) →
  **hygiene** on each emitted call (unknown tool,
  missing arguments via `core/tool_args.coerce_args`, malformed JSON, a repeat of a call the user
  DECLINED this turn, a third identical call with nothing changed in between — except a third
  identical READ whose last run completed, answered once with that result; every reply carries
  its kind under `textutil.HYGIENE_KEY` — each refusal answered with
  an error ToolMessage that routes straight back to `agent`, no gate, no model call) → **answer** (a message without tool calls IS
```

The last line is kept byte-for-byte: `2026-10-01-question-is-an-answer.md` Task 6 anchors on it.

- [ ] **Step 2: docs/ARCHITECTURE.md**

Replace the one line

```
     again is rerun once with tools unbound); each emitted call passes hygiene (unknown tool, missing or malformed
```

with

```
     again is rerun once with tools unbound; three passes with a refused call stop the turn the
     same way, early; a pass after a round in which nothing happened gets a prompt-only note
     naming those calls; a third identical read whose last run completed is answered once with
     that result instead of being refused); each emitted call passes hygiene (unknown tool, missing or malformed
```

Only that line changes: the two lines after it are `2026-10-01-question-is-an-answer.md` Task 6's anchor.

- [ ] **Step 3: docs/engine.md — the shape**

In "The shape" → step 5, replace

```
   edit or command in between resets the count, so edit → test → edit → test is not a stall). `ask_user` runs alone — a resumed interrupt re-executes
   the tools node, so siblings in its batch are answered with `ASK_ALONE_TEXT`.
```

with

```
   edit or command in between resets the count, so edit → test → edit → test is not a stall). `ask_user` runs alone — a resumed interrupt re-executes
   the tools node, so siblings in its batch are answered with `ASK_ALONE_TEXT`.
   Three guards sit on this step (2026-10-01, `superpowers/plans/2026-10-01-loop-guards.md`).
   Every ToolMessage the agent node writes carries its kind under `textutil.HYGIENE_KEY`.
   **The hygiene budget:** after `HYGIENE_BUDGET` (3) passes with a refused call (counted per
   pass; gate declines, budget refusals and replays never count), the next pass's calls get
   `BOUNCE_TEXT` and the turn ends the way the cap ends it. **The replay:** a third identical
   call to a `read_only` tool whose last run completed is answered with that result
   (`REPLAY_PREFIX`; over `REPLAY_INLINE_CAP` characters, a pointer to it), once — the next
   identical call is the stall; a quarantine-fenced result is never replayed. **The
   failed-round note:** a pass after a round in which nothing happened (every call declined,
   blocked, failed or refused) gets those calls, in the incidents note's words, as a
   prompt-only message at the tail (`_round_notes` → `_llm_input`'s `extra`).
```

- [ ] **Step 4: CHANGELOG.md**

Under `## [Unreleased]` → `### Changed`, replace

```
### Changed

- **`/privacy` is part of `/policy`.**
```

with

```
### Changed

- **A turn that keeps getting its tool calls wrong now stops and answers.** After three passes
  in one turn with a malformed or refused call (an unknown tool, arguments that belong to
  another tool, the same call a third time), Saturn takes no more calls and answers with what it
  has — about five passes, where a small model used to spiral to the sixteen-pass limit. Before
  it answers it is told which calls did not happen, so it says so instead of claiming them; the
  note under the answer lists them as before.
- **Asking for the same thing a third time gets the same answer, not an error.** A third
  identical read with nothing changed in between is answered from the result Saturn already has
  (the rail shows `↺ … same result as before, not run again`); a fourth is refused. Writes,
  commands and anything the injection quarantine flagged are never answered this way.
- **The rail shows calls Saturn refused before running them** — one `✗ … not run — why` line
  each, under the pass that made them.
- **`/privacy` is part of `/policy`.**
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `0 failed` (the baseline count plus the 23 tests this plan adds).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/ARCHITECTURE.md docs/engine.md CHANGELOG.md
git commit -m "docs: the loop guards — hygiene budget, replayed reads, the failed-round note"
```

---

### Task 7: Measure (manual — needs a running Ollama with the tier pulled)

Nothing here is offline or automated. It decides whether the guards earned their place, and it is where `docs/engine.md` items 1–3 get marked shipped — with numbers, not before.

**Files:**
- Modify: `docs/engine.md` ("What the benchmarks say"; "Improvements, ranked" items 1–3) — only after the runs.

- [ ] **Step 1: The "before" numbers**

Check out the commit before Task 1 into a scratch worktree (never `git stash` in this repo — the stash is shared with other worktrees) and run three times per tier (a delta of one task is noise — engine.md Measurement item 13):

```bash
git worktree add /private/tmp/saturn-before-guards \
  "$(git log --format=%H -1 --grep='^agent: mark every call the agent node answers itself')~1"
cp config.yaml /private/tmp/saturn-before-guards/
cd /private/tmp/saturn-before-guards && python -m venv .venv && .venv/bin/pip install -q -e '.[dev]'
for i in 1 2 3; do .venv/bin/python benchmark.py --loop; done   # active tier: 4b
for i in 1 2 3; do .venv/bin/python benchmark.py; done          # the trust benchmark
# set active_tier to the 9b in /private/tmp/saturn-before-guards/config.yaml and repeat both loops
```

Record per run, from `logging/benchmarks/loop_<ts>.json` → `loop_summary` and `trust_<ts>.json` → `trust`:
passed/total; `hygiene`; `capped`; mean passes per shape; and for the trust benchmark's
`memory_results` entry with `"task": "supersession"`, the verdict and each run's `iterations` and `hygiene`.

- [ ] **Step 2: The "after" numbers**

In `/Users/Logan/Documents/saturn-v2`, the same six runs per tier:

```bash
for i in 1 2 3; do .venv/bin/python benchmark.py --loop; done
for i in 1 2 3; do .venv/bin/python benchmark.py; done
```

| Watch | Expected after | If not |
|---|---|---|
| supersession (trust, 4b) | verdict `superseded`; when it does spiral, ≤ 5 passes and `bounced_out` | more than 5 passes: a refusal is missing its marker — find it with `answered_kind(m) is None` on the run's agent-answered ToolMessages |
| a spiral's answer | names the call that did not happen; never "I've updated" over an incidents note | read the run's `llm_calls` (`/trace invoke --full`) to confirm the failed-round note was in the prompt; if it was and the 4b ignored it, confirm on the 9b before changing the wording (a 4b-only miss is a model limit) |
| `bounced_out` (loop) | 0 on clean tasks | a clean task bouncing out means two unrelated slips plus one more — read its `tools_called` |
| `replays` (loop) | small; `robust_no_math_in_head` on the 9b may replay `calculate` | — |
| passed/total, mean passes | not worse | any task newly failing with `over_passes` or a changed answer: diff its response against the before run |
| `hygiene` (loop) | not higher (the count no longer includes gate declines) | — |

- [ ] **Step 3: Record and mark shipped**

Add a dated paragraph to `docs/engine.md` "What the benchmarks say" with the before/after table (three runs per tier, both suites). Then in "Improvements, ranked" → "Guards (robustness first)", append to each of items 1, 2 and 3 a line in this form, with the measured values:

```
   — shipped 2026-10-0N (`superpowers/plans/2026-10-01-loop-guards.md`): supersession on the
   4b <before passes> → <after passes> passes; loop <before passed>/<total> → <after passed>/<total>.
```

Then clean up and commit:

```bash
git worktree remove /private/tmp/saturn-before-guards
git add docs/engine.md
git commit -m "docs: engine.md — the loop guards, measured before and after"
```

---

## Self-Review (done while writing)

**1. Spec coverage.** engine.md item 1 (hygiene budget: count from the record, precisely what counts, N = 3 as a constant, reuse the cap mechanics, truthful wording, the exact condition changes) → Design + Task 2. Item 2 (failure-aware final pass: when, where — prompt tail via `extra`, template and thinking checked, no "answer now", reuses `incidents()` wording, the claim checker stated as a non-goal) → Design + Task 3. Item 3 (replay: `read_only` only, earlier observation with a short prefix, a stamp audited across every reader, bounded — a replay counts toward the stall, side-effecting tools keep the refusal, a pointer above a threshold, the quarantine fence kept by never replaying a fenced result) → Design + Task 4. The rail label → Task 5. The benchmark's grader → Tasks 1–2. Merge notes for both sibling plans, consistent with `2026-10-01-question-is-an-answer.md`'s own notes → its own section. Measurement with three runs per tier on both suites, engine.md / CLAUDE.md / CHANGELOG → Tasks 6–7. No gap found.

**2. Placeholder scan.** Every code step shows the code and every edit quotes the text it replaces. The only values left to fill are the measured numbers in Task 7 Step 3, which cannot exist before the runs; the step gives the exact line format.

**3. Type consistency.** `HYGIENE_KEY` (textutil; Tasks 1, 4, 5). `answered_kind(m) -> str | None`, `UNCOUNTED_KINDS` (Task 1; read in Tasks 2 and `benchmark.agent_answered`). `HYGIENE_BUDGET`, `BOUNCE_TEXT`, `_bounced_passes(this_turn) -> int`, `_stopped(this_turn) -> bool`, `_NOT_RUN: dict` (Task 2; `_NOT_RUN` read by Task 3's `_round_notes`). `FAILED_ROUND_NOTE`, `_status(m) -> str`, `_round_notes(this_turn) -> list` (Task 3). `REPLAY_PREFIX`, `REPLAY_POINTER`, `REPLAY_INLINE_CAP`, `_same_since_change(key, rounds) -> list`, `_replay(call, name, same) -> ToolMessage | None` (Task 4). `quarantine.FENCE_BEGIN`, `FENCE_END`, `is_fenced(text) -> bool` (Task 4). `benchmark.agent_answered(messages) -> {"hygiene", "replays", "bounced_out"}` (Tasks 1–2). Test helpers `_refused` (Task 1), `_drive` (Task 2), `_prompt_of` (Task 3) are each defined once in `tests/test_agent_loop.py` before their first use. `_stopped` reads `_NOT_RUN`, defined later in the module — resolved at call time, as every module-level name in Python.

**4. Review Focus.** Five lines, each with its test written out in the owning task (Tasks 2, 2, 2, 3, 4).

**5. Dry run (2026-10-02).** Tasks 1–6 were applied verbatim, from this file, to a scratch copy of the working tree (`git status` as of 2026-10-01, uncommitted macOS-tools work included): every quoted anchor matched exactly once, and `python -m pytest tests/ -q` went from 1301 passed to **1324 passed** (the 23 new tests; `tests/test_agent_loop.py` 55 → 75). Not run: the "verify it fails" steps one by one, and Task 7 (needs Ollama).
