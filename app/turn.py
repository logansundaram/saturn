"""Drive one turn of the compiled graph.

`run_turn` streams the graph (node updates for the trace/plan panel, per-token answer chunks
for the live response) and resolves each interrupt — the approval gate, the plan-review gate —
through the caller-supplied `approver`. `_make_on_update` fans a node delta out to the tracer
and the TUI; `_trace_warning` surfaces the trace circuit breaker's silent degradation.
"""

from langgraph.types import Command
from langchain.messages import AIMessageChunk

from tui import ui


def run_turn(graph, payload, config, approver, on_update=None, pause=None, on_token=None):
    """Drive one turn to completion, streaming node updates and pausing at an interrupt.

    `approver(interrupt_value) -> decision` resolves each interrupt — for the approval gate a bool,
    for the plan-review gate the editor's `{action, plan}` dict — and the result is fed back as the
    `Command(resume=...)` value. `on_update(node, delta)` is called for every node update (the trace
    + live plan panel). `on_token(text)`, if given, receives the *synthesize* node's answer tokens as
    they generate (LangGraph `stream_mode="messages"`, filtered to that node), so the UI can render
    the final answer live. `pause`, if given, is a `typeahead.InputQueue` (any start()/stop() console
    reader): it's started only while the graph is executing and stopped before any blocking input(),
    so it can capture type-ahead + the Esc pause without ever stealing the prompt's keystrokes (the
    queued lines themselves are drained by the REPL loop, not here). Returns the final state.

    Streams two modes at once: "updates" drives the trace/plan and carries the interrupt marker
    (unchanged routing — pause/resume is still decided by get_state below); "messages" carries the
    per-token answer stream. Each streamed item is a `(mode, data)` pair."""
    # The plan/execute engine visits ~5 nodes per step; LangGraph's default recursion_limit (25)
    # would kill a healthy multi-step plan mid-flight. Generous but finite — the REAL bounds are
    # runtime.max_iterations (execute passes) and MAX_REPLANS, which land at an honest synthesize
    # long before this.
    config.setdefault("recursion_limit", 200)
    pending = payload
    while True:
        if pause is not None:
            pause.start()
        # Tokens flow to on_token the moment they generate — NEVER buffered until the node's
        # updates event: LangGraph emits a node's update only after the node COMPLETES, so
        # holding chunks for it delivers the whole answer in one burst and silently kills the
        # token-by-token streaming the ResponseStream exists for. The synthesize rail line
        # (metrics from the update) consequently prints after the stream opens; rich inserts
        # it above the live tail, and the final render follows it.
        try:
            for mode, data in graph.stream(pending, config, stream_mode=["updates", "messages"]):
                if mode == "messages":
                    # (message_chunk, metadata) — stream only the synthesize node's answer tokens.
                    # Filters: skip other LLM nodes (agent draft, planner/judge structured output);
                    # and require an AIMessageChunk (a streaming delta) — messages mode ALSO emits the
                    # node's returned, complete AIMessage (the full answer written to state), which
                    # would otherwise re-deliver the whole text once more and double the display.
                    message_chunk, metadata = data
                    if (
                        on_token
                        and isinstance(message_chunk, AIMessageChunk)
                        and metadata.get("langgraph_node") == "synthesize"
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
            return snapshot.values  # turn complete

        # Paused on an interrupt — pull its payload, ask the approver/reviewer, resume.
        interrupt_value = None
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_value = task.interrupts[0].value
                break
        decision = approver(interrupt_value)
        pending = Command(resume=decision)


def _make_on_update(tracer, run_id, show_ui=True):
    def on_update(node, delta):
        tracer.log_event(run_id, node, delta)
        if show_ui:
            ui.show_node(node, delta)
            if delta.get("plan"):
                ui.show_plan(delta["plan"])

    return on_update


def _trace_warning(tracer) -> "str | None":
    """The user-facing notice when this turn's trace circuit breaker tripped (stores/trace._trip),
    or None. The tracer degrades to silence by design (the watcher must never stall the watched),
    but the DEGRADATION itself must be loud — the user believes /trace, /glass #id, and signed
    exports are accumulating a record, and this turn's may be partial or missing entirely. The
    breaker re-arms on the next start_run, so the warning is per-affected-turn, not permanent."""
    if not getattr(tracer, "broken", False):
        return None
    return ("trace recording degraded this turn (a db.sqlite write failed — locked by another "
            "process?) — this run may be missing from /trace; details in logging/diag.log")
