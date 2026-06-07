import sys

# Force UTF-8 console output. Node prints (plan glyphs, tool results, model output) routinely
# contain non-cp1252 characters that crash print() on the default Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import sqlite3
import uuid
from datetime import datetime

__version__ = "0.1.0"

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk

import diag
from config import get_config
from state import AgentState

# loop nodes
from node_registry.ground import grounding_node
from node_registry.plan import plan_node
from node_registry.synthesize import synthesize_node
from node_registry.update_plan import update_plan_node
from node_registry.agent import agent_node, route_after_agent
from node_registry.tools import tool_node
from node_registry.approval import approval_node
from node_registry.replan import replan_node
from node_registry.plan_gate import plan_gate_node, route_after_gate

# RAG ingest (reconciles the disk-cached vector store the search_knowledge_base tool reads)
from stores.rag import sync

# transparency + safety UI
from stores.trace import Tracer
from tui import ui

# REPL meta-commands (lines starting with `/`)
import commands

# `@file` mention expansion (pull a referenced file's contents into the turn's context)
import mentions

# Human-in-the-loop plan-review pause: the shared controller.
from interrupts import get_pause_controller

# Type-ahead: the single console reader during a turn — queues follow-up queries/commands the user
# types while the agent works, and carries the Esc plan-review pause trigger.
from typeahead import InputQueue

DB_PATH = str(get_config().path("db_sqlite"))

from utilities.print_graph import print_graph


def build_agent():
    """Assemble the living-plan ReAct loop with a human-in-the-loop approval gate AND a
    plan-review gate:

        START -> ground -> plan -> plan_gate -> agent -> approval -> (tools -> update_plan -> plan_gate -> agent)* -> synthesize -> END
                                       │ │         │          │
                          (pause? edit │ │  (no tool calls)  (reject -> back to agent)
                           the plan) ──┘ │         ▼
                          (abort) ───────┘      replan ──(grounded)──> synthesize
                          → synthesize             └──(ungrounded: insert web_search)──> agent

    When the agent finishes with no tool calls and no planned gathering step is left to nudge
    toward, `replan` (the `judge` role) verifies the draft answer is grounded; if it leans on
    facts that were never looked up it inserts a web_search step and loops back to `agent`
    (bounded by REPLAN_BUDGET). See node_registry/replan.py.

    `plan_gate` runs at every step boundary: a pass-through unless a pause has been requested, in
    which case it `interrupt()`s so the user can inspect/edit the plan and resume (see
    node_registry/plan_gate.py). Compiled with a SqliteSaver checkpointer, which both persists
    sessions and is what lets the approval / plan-review `interrupt`s pause and resume.
    """
    builder = StateGraph(AgentState)

    builder.add_node("ground", grounding_node)
    builder.add_node("plan", plan_node)
    builder.add_node("plan_gate", plan_gate_node)
    builder.add_node("agent", agent_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", tool_node)
    builder.add_node("update_plan", update_plan_node)
    builder.add_node("replan", replan_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "ground")
    builder.add_edge("ground", "plan")
    # Every step boundary flows through plan_gate (the plan-review checkpoint) before the agent acts.
    builder.add_edge("plan", "plan_gate")
    builder.add_conditional_edges(
        "plan_gate",
        route_after_gate,
        # Normally -> agent; if the user aborted at the review prompt -> wrap up at synthesize.
        {"agent": "agent", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        # "agent" self-loop: the plan-aware nudge — when the model finishes with an un-run
        # planned tool still pending, route back to act on it (bounded; see route_after_agent).
        # "replan": an apparently-complete finish goes to the judge, which may insert a web_search
        # step and loop back if the draft answer leans on facts that were never looked up.
        {"approval": "approval", "synthesize": "synthesize", "agent": "agent", "replan": "replan"},
    )
    # approval + replan route dynamically via Command(goto=...): approval -> "tools"/"agent",
    # replan -> "agent" (escalate with an inserted web_search) / "synthesize" (answer is grounded).
    builder.add_edge("tools", "update_plan")
    builder.add_edge("update_plan", "plan_gate")
    builder.add_edge("synthesize", END)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)


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
    pending = payload
    while True:
        if pause is not None:
            pause.start()
        # Buffer synthesize tokens so the rail line (with full metrics from the node delta) can
        # print before the response section opens. The update event is guaranteed to arrive after
        # all message chunks for the same node, so flushing on the update preserves ordering.
        _synth_buf: list[str] = []
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
                            _synth_buf.append(text if isinstance(text, str) else str(text))
                    continue
                # mode == "updates"
                if "__interrupt__" in data:
                    continue  # detected via get_state below
                for node, delta in data.items():
                    if on_update:
                        on_update(node, delta or {})
                    if node == "synthesize" and on_token and _synth_buf:
                        for tok in _synth_buf:
                            on_token(tok)
                        _synth_buf.clear()
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


def _compact_history(messages: list, keep_recent_turns: int = 1) -> list:
    """Collapse OLDER completed turns to their conversational essence (user questions + final
    answers), but keep the ReAct scratchpad — tool-call AIMessages and their ToolMessages — of
    the most recent `keep_recent_turns` turns verbatim.

    Why a window instead of stripping everything: the scratchpad of the turn that just finished
    is exactly what the user's *next* message refers back to — "open the second result", "what
    did that file say", "multiply that by two". Dropping it on every boundary (the old
    behaviour) is what made real multi-turn use brittle: the follow-up's referent had silently
    vanished, so the model re-ran a search (getting different results) or fabricated. One turn
    of live scratchpad covers the overwhelming majority of those references.

    The original concerns still hold for OLD turns, which is why they're still compacted:
    carrying many turns of scratchpad makes the model treat a long-finished tool call as "already
    done" (reusing stale results instead of re-running a planned gather), bloats context with
    heavy tool outputs, and desyncs the model's view (`messages`) from the plan machinery's
    (per-turn `tools_called`, reset each turn — so the nudge still correctly sees this turn's
    planned tools as un-run regardless of what's in the retained window).

    A turn starts at a HumanMessage. Everything from the boundary onward is kept as-is (the
    scratchpad is intact, so no orphaned tool calls); everything before it is reduced to
    Human + non-empty final-AI messages (also orphan-free). Run only at the turn boundary.

    `keep_recent_turns=0` reproduces the old strip-everything behaviour."""
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if keep_recent_turns > 0 and human_idxs:
        # Boundary = start of the Nth-from-last turn (clamped to the first turn).
        boundary = human_idxs[-min(keep_recent_turns, len(human_idxs))]
    else:
        boundary = len(messages)

    kept = []
    for m in messages[:boundary]:
        if isinstance(m, HumanMessage):
            kept.append(m)
        elif (
            isinstance(m, AIMessage)
            and not getattr(m, "tool_calls", None)
            and str(m.content).strip()
        ):
            kept.append(m)
        # else: ToolMessage or tool-call/empty AIMessage from an OLD turn — drop it.
    return kept + messages[boundary:]


def _maybe_autocompact(state: AgentState) -> AgentState:
    """If the turn that just finished left the context filled past `runtime.compact_threshold`, fold
    the older turns into an LLM summary (compaction.summarize_messages) so the NEXT turn doesn't
    re-send — and overflow — the window. This is the heavier LLM compaction; the mechanical
    `_compact_history` still runs every turn regardless.

    Best-effort and non-fatal: disabled via `runtime.auto_compact`, skipped when the fill is unknown,
    and any summary failure leaves the history untouched (summarize_messages swallows it). Mutates +
    returns `state` so the caller can keep its handle current."""
    cfg = get_config()
    if not cfg.get("runtime.auto_compact", True):
        return state
    used = int(state.get("context_tokens", 0) or 0)
    from llms import active_context_window

    window = active_context_window()
    if not window or used <= 0:
        return state
    threshold = float(cfg.get("runtime.compact_threshold", 0.85) or 0.85)
    if used / window < threshold:
        return state

    from compaction import summarize_messages

    new_msgs, stats = summarize_messages(state["messages"])
    if stats["summarized_turns"] > 0 and stats["after"] < stats["before"]:
        state["messages"] = new_msgs
        ui.note(
            f"auto-compacted {stats['summarized_turns']} earlier turn(s) "
            f"({stats['before']}→{stats['after']} messages) — context was "
            f"{used / window * 100:.0f}% full ({_human_int(used)}/{_human_int(window)} tok)."
        )
    return state


def _human_int(n: int) -> str:
    """Compact integer for the auto-compaction notice (1800 -> 1.8k)."""
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def _fresh_turn(state: AgentState, user_input: str) -> AgentState:
    """Append the new query and reset per-turn fields (accumulators + loop counter).
    `messages` persists across turns to keep in-process conversation memory, but is first
    compacted (see _compact_history): older turns collapse to a clean Q&A transcript while the
    most recent turn's tool scratchpad is retained so a follow-up can refer back to it."""
    state["messages"] = _compact_history(state["messages"])
    state["messages"].append(HumanMessage(content=user_input))
    state["current_query"] = user_input
    state["current_response"] = ""
    state["context"] = ""
    state["attachments"] = ""  # set by the loop after expanding @file mentions (mentions.expand)
    state["plan"] = []
    state["iteration"] = 0
    state["agent_nudges"] = 0
    state["replans"] = 0
    state["pause_requested"] = False
    state["pause_reason"] = ""
    state["aborted"] = False
    state["tools_called"] = []
    state["tool_results"] = []
    state["documents_retrieved"] = []
    state["tool_events"] = []
    state["tok_per_sec"] = 0.0
    # context_tokens persists across turns (the context only grows; overwritten on next LLM call).
    return state


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "current_query": "",
        "current_response": "",
        "context": "",
        "attachments": "",
        "plan": [],
        "iteration": 0,
        "agent_nudges": 0,
        "replans": 0,
        "pause_requested": False,
        "pause_reason": "",
        "aborted": False,
        "tools_called": [],
        "tool_results": [],
        "documents_retrieved": [],
        "tool_events": [],
        "tok_per_sec": 0.0,
        "context_tokens": 0,
    }


def main():
    """CLI entry point: ingest the knowledge base, build the graph, run the REPL loop.

    Flags:
      -p / --prompt QUERY   Run one query headlessly, print the answer to stdout, and exit.
                            No TUI, no interactive prompts; all tool calls auto-approved.
                            Compatible with `saturn -p "..."` and shell pipelines.
      --version             Print the version and exit.
    """
    import argparse

    _parser = argparse.ArgumentParser(prog="saturn", add_help=False)
    _parser.add_argument("-p", "--prompt", metavar="QUERY", default=None,
                         help="Run a single query headlessly and print the answer to stdout.")
    _parser.add_argument("--version", action="version", version=f"saturn {__version__}")
    _args, _ = _parser.parse_known_args()

    # The slow startup loading (knowledge-base ingest + graph build) runs while the ring art
    # animates in interactive mode, or directly (no TUI) in headless mode.
    def _startup_load():
        warn = None
        # Reconcile the knowledge base against the disk cache at startup: only new/changed
        # documents are embedded, the rest load from the persisted store. Non-fatal if it fails
        # (e.g. embedding model not pulled) — search_knowledge_base just returns "no documents".
        try:
            sync(verbose=False)
        except Exception as exc:
            warn = f"knowledge-base ingest failed, continuing without RAG: {exc}"
        return build_agent(), warn

    # --- headless path: one query, print answer, exit ---------------------------------
    if _args.prompt:
        graph, ingest_warning = _startup_load()
        if ingest_warning:
            print(ingest_warning, file=sys.stderr)
        tracer = Tracer(DB_PATH)
        state = _initial_state()
        state = _fresh_turn(state, _args.prompt)
        thread_id = str(uuid.uuid4())
        run_id = tracer.start_run(thread_id, _args.prompt)
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [tracer.llm_handler(run_id)],
        }
        try:
            state = run_turn(
                graph,
                state,
                config,
                approver=lambda _v: True,  # auto-approve: no interactive gates in headless mode
                on_update=_make_on_update(tracer, run_id, show_ui=False),
            )
            answer = state["messages"][-1].content
            tracer.end_run(run_id, "ok", answer)
            print(answer)
        except Exception as exc:
            tracer.end_run(run_id, "error", str(exc))
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            try:
                graph.checkpointer.delete_thread(thread_id)
            except Exception:
                pass
        return

    # --- interactive path -------------------------------------------------------------
    graph, ingest_warning = ui.splash(
        _startup_load
    )  # ring-and-planet art over the load
    if ingest_warning:
        ui.warn(ingest_warning)
    tracer = Tracer(DB_PATH)
    state = _initial_state()

    # Startup header — tier/model / tool count / corpus size, like a tool's first line.
    from llms import model_id, check_models
    from registry import tool as _tools
    from stores.rag import iter_documents

    cfg = get_config()
    n_docs = sum(1 for _ in iter_documents())  # same definition RAG ingests by
    ui.banner(
        f"{cfg.active_tier}:{model_id('tool_caller')}", len(_tools), n_docs, DB_PATH
    )

    # Health-check the active tier up front: a down daemon / un-pulled model / missing cloud key is
    # surfaced now with an actionable fix, rather than as a generic turn failure on the first query.
    # Non-fatal — the REPL still starts (commands work; an affected turn fails cleanly).
    for problem in check_models():
        ui.warn(problem)

    # Carries the live session into slash-command handlers. `make_initial_state` lets
    # /reset rebuild state without commands.py importing back into agent.py.
    cmd_ctx = commands.CommandContext(
        state=state,
        make_initial_state=_initial_state,
        db_path=DB_PATH,
        session_started_at=datetime.now().isoformat(),  # session boundary for /cost
    )

    # First-run setup check: if the sentinel hasn't been written yet, auto-run /config setup so a
    # fresh install surfaces any gaps (Ollama down, models not pulled, keys missing) before the
    # user's first query, rather than as a confusing turn failure. Non-fatal: a dispatch error
    # mustn't prevent the REPL from starting. The sentinel lives in the database directory so
    # deleting the database also resets first-run (a full reinstall should re-check).
    _setup_sentinel = get_config().path("database") / ".setup_done"
    if not _setup_sentinel.exists():
        ui.note("First launch — running /config setup (won't repeat; re-run any time with /config setup).")
        try:
            commands.dispatch("/config setup", cmd_ctx)
        except Exception as exc:
            ui.warn(f"/config setup failed: {exc}")
        try:
            _setup_sentinel.parent.mkdir(parents=True, exist_ok=True)
            _setup_sentinel.touch()
        except Exception as exc:
            diag.log(f"first-run sentinel write failed: {exc}")

    # One input reader for the session. While a turn runs it captures type-ahead so the user can
    # queue follow-up queries / slash commands without waiting (drained between turns below). The Esc
    # key acts on whatever is typed: with text, it's a mid-turn steering correction (injected into
    # the running turn at the next step boundary via plan_gate, acknowledged by ui.steer_note); with
    # an empty line, it asks the plan_gate to pause for plan review. The in-progress line + queue
    # depth render live in the status bar (on_change -> ui). No-ops cleanly off-TTY (see
    # typeahead.InputQueue).
    input_queue = InputQueue(on_change=ui.set_input_preview, on_steer=ui.steer_note)
    pause_controller = get_pause_controller()

    # print_graph(graph=graph)

    def _next_input() -> str:
        """The next line to process: anything the user typed-ahead while the last turn ran is
        drained first (FIFO, echoed so it reads like it was entered live), and only once the queue
        is empty do we block on the `»` prompt. A queued line can be a query or a slash command —
        both flow through the same handling below — so follow-ups and commands alike can be lined
        up mid-turn and run the moment the agent is free."""
        queued = input_queue.pop()
        if queued is not None:
            ui.echo_queued(queued)
            return queued
        return ui.prompt(commands.command_completions())

    while True:
        user_input = _next_input()

        # `/`-prefixed lines are REPL meta-commands, not agent turns — intercept them here.
        if commands.is_command(user_input):
            commands.dispatch(user_input, cmd_ctx)
            if cmd_ctx.should_quit:
                break
            state = cmd_ctx.state  # a command (e.g. /reset) may have swapped state out
            continue

        if not user_input.strip():
            continue

        state = _fresh_turn(state, user_input)
        # Expand @file mentions: read any files the user referenced as `@path` and stash their
        # contents on state for the grounding node to fold into context (so every node sees the
        # file inline). The message text itself is left untouched — the @mention stays visible.
        attach_block, attached = mentions.expand(user_input)
        if attached:
            state["attachments"] = attach_block
            ui.note("attached " + ", ".join(mentions.display(p) for p in attached))
        # Persistent review mode (/plan review on) arms a pause at the FIRST gate every turn, so the
        # plan is vetted before any execution. A one-shot /plan pause set the controller directly.
        if cmd_ctx.review_plan:
            pause_controller.request(
                "review", "review mode: vet the plan before executing"
            )
        # Fresh thread per turn: gives the interrupts a stable thread to pause/resume on,
        # while cross-turn memory rides on the manually-carried `messages`.
        thread_id = str(uuid.uuid4())
        run_id = tracer.start_run(thread_id, user_input)
        # Attach the LLM-call tracer as a run-scoped callback: it captures every model call's input
        # messages + output (across all nodes) into the trace DB, surfaced by `/trace invoke`.
        # Callbacks in the config propagate into the nested model.invoke()/stream() calls via
        # LangChain's contextvars — the same propagation the token stream above already relies on.
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [tracer.llm_handler(run_id)],
        }
        ui.reset_turn()  # reset node-timing + plan-diff state for this turn's trace
        # Renders the synthesize node's answer token-by-token as it streams (on_token below). It
        # opens the response section on the first token and is finished (or aborted) after the turn.
        answer = ui.ResponseStream()

        # Resolve each interrupt by type: the plan-review gate -> the plan editor; the approval
        # gate -> /autoapprove (always yes) or the approval prompt. Keeping this dispatch here lets
        # run_turn stay interrupt-type-agnostic (it just feeds the result back as the resume value).
        base_approver = (lambda _v: True) if cmd_ctx.auto_approve else ui.ask_approval

        def on_interrupt(value, _approve=base_approver):
            if isinstance(value, dict) and value.get("type") == "plan_review":
                return ui.review_plan(value)
            return _approve(value)

        try:
            state = run_turn(
                graph,
                state,
                config,
                approver=on_interrupt,
                on_update=_make_on_update(tracer, run_id, show_ui=cmd_ctx.show_ui),
                pause=input_queue,
                on_token=answer.feed,
            )
            tracer.end_run(run_id, "ok", state["messages"][-1].content)
        except KeyboardInterrupt:
            # Ctrl-C abandons the in-flight turn but not the session — record it and return to
            # the prompt. (KeyboardInterrupt is not an Exception, so it bypasses the catch below.)
            answer.abort()  # tear down the live answer region (never leak it across the prompt)
            tracer.end_run(run_id, "interrupted", "turn cancelled by user (Ctrl-C)")
            ui.warn("Turn cancelled.")
            cmd_ctx.state = state
            continue
        except Exception as exc:
            # A node/tool failure must never kill the REPL — the whole point of an in-memory
            # conversation is that one bad turn (Ollama timeout, decode error, tool bug) doesn't
            # lose the session. Record it, tell the user, and drop back to the prompt with the
            # conversation intact (the unanswered query stays in `messages`; the next turn's
            # _compact_history tolerates it).
            answer.abort()  # tear down the live answer region before the warning prints
            tracer.end_run(run_id, "error", str(exc))
            ui.warn(f"Turn failed: {exc}")
            cmd_ctx.state = state
            continue
        finally:
            # Discard any pause request still pending at turn end. A keypress that lands AFTER the
            # turn's last plan_gate (e.g. during the final agent message or synthesize) is never
            # consumed by the gate's clear(), and would otherwise leak into the next, unrelated turn
            # and pause it. A `/plan pause` issued at the prompt is set after this point, so it
            # survives; review mode re-arms each turn — neither is affected.
            pause_controller.clear()
            # Prune this turn's checkpoints. Each turn runs on a fresh thread_id and cross-turn
            # memory rides on the manually-carried `messages` (not the checkpointer), so once the
            # turn returns its checkpoints/writes are dead weight — without this they accumulate in
            # db.sqlite forever (one thread per turn). delete_thread touches only the checkpointer's
            # own tables; the trace (runs/events) and the in-memory state we carry forward are
            # untouched. Best-effort: a prune failure must never end the turn or the session.
            try:
                graph.checkpointer.delete_thread(thread_id)
            except Exception as exc:
                diag.log(f"checkpoint prune failed for thread {thread_id}: {exc}")
            # Autosave the conversation to the reserved /resume slot. The checkpoints we just pruned
            # can't restore a session, so this slot is what survives a quit/crash/Ctrl-C. Runs for
            # every outcome (ok/error/interrupt) since `state` always carries the latest messages;
            # write_autosave is itself best-effort, so a failure can't end the turn or the session.
            commands.write_autosave(state)

        cmd_ctx.state = state  # keep the command context pointed at the latest state
        # The answer streamed live during synthesize — close it out (final markdown render + receipt).
        # If nothing streamed (e.g. the model yielded no content, or the turn aborted at the plan
        # gate before synthesize produced text), fall back to rendering the recorded final message.
        if answer.started:
            answer.finish()
        else:
            ui.response(state["messages"][-1].content)

        # If this turn pushed the context past the compaction threshold, summarize older turns now so
        # the next turn starts with a smaller window (best-effort; see _maybe_autocompact). Runs after
        # the answer is rendered so its LLM call never delays the response the user is waiting on.
        state = _maybe_autocompact(state)
        cmd_ctx.state = state


if __name__ == "__main__":
    main()
