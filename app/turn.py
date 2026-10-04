"""Drive one turn of the compiled graph.

`run_turn` streams the graph (node updates for the trace/plan panel, per-token answer chunks
for the live response) and resolves each interrupt — the approval gate, the Esc pause, ask_user —
through the caller-supplied `approver`. `open_run` / `close_run` bracket a turn with its
checkpoint thread and trace run (shared by the REPL and headless). `_make_on_update` fans a node
delta out to the tracer and the TUI; `_trace_warning` surfaces the trace circuit breaker's silent
degradation.
"""

import uuid

from langgraph.types import Command
from langchain.messages import AIMessageChunk

import diag
from core import hooks
from tui import ui


def run_turn(graph, payload, config, approver, on_update=None, pause=None, on_token=None,
             on_retract=None, on_thinking=None):
    """Drive one turn to completion, streaming node updates and pausing at an interrupt.

    `approver(interrupt_value) -> decision` resolves each interrupt (the approval gate, the pause
    prompt, an ask_user question) and the result is fed back as the `Command(resume=...)` value.
    `on_update(node, delta)` is called for every node update (the trace + live plan panel). `on_token(text)`, if given, receives the *agent* node's answer tokens as
    they generate (LangGraph `stream_mode="messages"`, filtered to that node) so the UI can render
    the final answer live. `pause`, if given, is a `typeahead.InputQueue` (any start()/stop() console
    reader): it's started only while the graph is executing and stopped before any blocking input(),
    so it can capture type-ahead + the Esc pause without ever stealing the prompt's keystrokes (the
    queued lines themselves are drained by the REPL loop, not here). `on_retract()`, if given, is
    called when the agent node takes back what it streamed (a malformed attempt about to be
    retried — `nodes.agent.RETRACT` on the custom stream). `on_thinking(bool)`, if given, is told
    when a thought begins and ends (the status bar's `thinking 3s · esc stops`). Returns the
    final state.

    Streams three modes at once: "updates" drives the trace/plan and carries the interrupt marker
    (pause/resume is decided by get_state below); "messages" carries the per-token answer stream;
    "custom" carries the retract signal. Each streamed item is a `(mode, data)` pair."""
    # The loop visits three nodes per tool round; LangGraph's default recursion_limit (25) would
    # kill a healthy multi-step turn mid-flight. Generous but finite — the REAL bound is
    # runtime.max_iterations (agent passes), which lands at an honest answer long before this.
    config.setdefault("recursion_limit", 200)
    query = payload.get("current_query") if isinstance(payload, dict) else None
    hooks.run("turn-start", query=query)  # the user's hooks.yaml (core/hooks); none → no cost
    pending = payload
    while True:
        if pause is not None:
            pause.start()
        # Tokens flow to on_token the moment they generate — NEVER buffered until the node's
        # updates event: LangGraph emits a node's update only after the node COMPLETES, so
        # holding chunks for it delivers the whole answer in one burst and silently kills the
        # token-by-token streaming the ResponseStream exists for. The agent rail line
        # (metrics from the update) consequently prints after the stream opens; rich inserts
        # it above the live tail, and the final render follows it.
        try:
            for mode, data in graph.stream(
                pending, config, stream_mode=["updates", "messages", "custom"]
            ):
                if mode == "custom":
                    if on_retract and isinstance(data, dict) and data.get("type") == "retract":
                        on_retract()
                    if on_thinking and isinstance(data, dict) and data.get("type") == "thinking":
                        on_thinking(data.get("phase") == "start")
                    continue
                if mode == "messages":
                    # (message_chunk, metadata) — stream only the agent node's tokens. Filters:
                    # skip other nodes' model calls, and require an AIMessageChunk (a streaming
                    # delta) — messages mode ALSO emits the node's returned, complete AIMessage
                    # (the full answer written to state), which would otherwise re-deliver the
                    # whole text once more and double the display.
                    message_chunk, metadata = data
                    if (
                        on_token
                        and isinstance(message_chunk, AIMessageChunk)
                        and metadata.get("langgraph_node") == "agent"
                    ):
                        text = getattr(message_chunk, "content", "")
                        if text:
                            on_token(text if isinstance(text, str) else str(text))
                    continue
                # mode == "updates"
                if "__interrupt__" in data:
                    continue  # detected via get_state below
                for node, delta in data.items():
                    if on_update:
                        on_update(node, delta or {})
        finally:
            if pause is not None:
                pause.stop()  # never leave the watcher live across the input() below

        snapshot = graph.get_state(config)
        if not snapshot.next:
            _turn_end_hooks(query, snapshot.values)
            return snapshot.values  # turn complete

        # Paused on an interrupt — pull its payload, ask the approver/reviewer, resume.
        interrupt_value = None
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
        decision = approver(interrupt_value)
        pending = Command(resume=decision)


def _turn_end_hooks(query, values: dict) -> None:
    """turn-end with the recorded answer (the last message, trailers included)."""
    messages = (values or {}).get("messages") or []
    answer = str(getattr(messages[-1], "content", "") or "") if messages else ""
    hooks.run("turn-end", query=query, answer=answer)


def open_run(tracer, query: str) -> "tuple[str, int, dict]":
    """Open one turn: a fresh checkpoint thread (the interrupts pause/resume on it; cross-turn
    memory rides on the carried `messages`, not the checkpointer) and its trace run. Returns
    `(thread_id, run_id, config)`; the config carries the LLM-call tracer as a run-scoped callback,
    which LangChain's contextvars propagate into every nested model call (`/trace invoke`)."""
    thread_id = str(uuid.uuid4())
    run_id = tracer.start_run(thread_id, query)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer.llm_handler(run_id)],
    }
    return thread_id, run_id, config


def close_run(graph, thread_id: str) -> dict:
    """Close one turn at its boundary, whatever the outcome. Prunes the thread's checkpoints —
    each turn runs on a fresh thread, so once it returns they are dead weight that would
    accumulate in db.sqlite forever (delete_thread touches only the checkpointer's own tables) —
    and ends the grant-lifecycle task, so a task-scoped always-allow grant never outlives the turn
    that motivated it. Returns the expired grants (policy.end_task). Never raises."""
    try:
        graph.checkpointer.delete_thread(thread_id)
    except Exception as exc:
        diag.log(f"checkpoint prune failed for thread {thread_id}: {exc}")
    try:
        from trust import policy

        return policy.end_task()
    except Exception as exc:
        diag.log(f"grant lifecycle end_task failed: {exc}")
        return {"prefixes": [], "tools": []}


def _make_on_update(tracer, run_id, show_ui=True, answer=None):
    """The per-delta subscriber: record first (the tracer self-guards — the watcher never takes
    down the watched), then render. DISPLAY is fail-soft: a render bug — a hostile step dict, a width edge case — must never raise out of
    run_turn, land a healthy turn as `error`, and lose the answer. Each ui call is guarded on
    its own; a failure prints one line and the loop continues.

    `answer` is the REPL's ResponseStream: an agent pass whose message ends in tool calls may
    have streamed a text preamble into the response region as if it were the answer — that
    region is discarded here (the rail's agent leaf shows the preamble where it belongs)."""
    def _render(what, fn, *args):
        try:
            fn(*args)
        except Exception as exc:  # display only — the record already landed above
            diag.log(f"turn: display error rendering {what}: {type(exc).__name__}: {exc}")
            try:
                ui.warn(f"display error ({what}): {type(exc).__name__}: {exc} — "
                        "the run is recorded; see /trace")
            except Exception:
                pass

    def on_update(node, delta):
        tracer.log_event(run_id, node, delta)
        if node == "agent" and answer is not None and getattr(answer, "started", False):
            # The pass's AIMessage may be followed by hygiene ToolMessages — look at every
            # message, not the last one.
            if any(getattr(m, "tool_calls", None) for m in delta.get("messages") or []):
                _render("discard", answer.discard)
        if show_ui:
            _render(f"node {node}", ui.show_node, node, delta)
            if delta.get("plan"):
                _render("plan", ui.show_plan, delta["plan"])

    return on_update


def _trace_warning(tracer) -> "str | None":
    """The user-facing notice when this turn's trace circuit breaker tripped (stores/trace._trip),
    or None. The tracer degrades to silence by design (the watcher must never stall the watched),
    but the DEGRADATION itself must be loud — the user believes /trace and /trace
    export are accumulating a record, and this turn's may be partial or missing entirely. The
    breaker re-arms on the next start_run, so the warning is per-affected-turn, not permanent."""
    if not getattr(tracer, "broken", False):
        return None
    return ("trace recording degraded this turn (a db.sqlite write failed — locked by another "
            "process?) — this run may be missing from /trace; details in logging/diag.log")
