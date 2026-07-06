"""The interactive loop: prompt → turn → answer, with everything that makes it a session.

Startup (splash, banner, posture line, health checks, first-run setup), the one input reader
(type-ahead + Esc steering/pause), drag-and-drop file offers, slash-command dispatch, the
turn lifecycle (trace run, interrupts, streaming answer, provenance), checkpoint pruning,
autosave, and auto-compaction. One call — `run_repl()` — owns the whole session.
"""

import uuid
from datetime import datetime
from pathlib import Path

import commands
import diag
from app.graph import DB_PATH
from app.session import _fresh_turn, _initial_state, _maybe_autocompact
from app.startup import startup_load, _warn_flagged_attachments
from app.turn import run_turn, _make_on_update, _trace_warning
from config import get_config
from core import mentions
from core.plan_ops import get_pause_controller
from stores.rag import SUPPORTED_EXTENSIONS
from stores.trace import Tracer
from tui import ui
from tui.typeahead import InputQueue


def run_repl() -> None:
    """The interactive session: load under the splash, print the startup readouts, then loop —
    prompt (or drained type-ahead) → slash command or agent turn → streamed answer — until
    /quit or EOF."""
    graph, ingest_warning = ui.splash(
        startup_load
    )  # ring-and-planet art over the load
    if ingest_warning:
        ui.warn(ingest_warning)
    tracer = Tracer(DB_PATH)
    state = _initial_state()

    # Startup header — tier/model / tool count / corpus size, like a tool's first line.
    from core.llms import model_id, check_models
    from tools.registry import tool as _tools
    from stores.rag import iter_documents

    cfg = get_config()
    n_docs = sum(1 for _ in iter_documents())  # same definition RAG ingests by
    ui.banner(
        f"{cfg.active_tier}:{model_id('tool_caller')}", len(_tools), n_docs, DB_PATH
    )
    # The session's trust posture, ambient from the first prompt: gate tier, inference locality,
    # boundary modes — the no-command twin of /privacy + /policy (receipt.posture_spans).
    ui.posture_line()

    # First-run sentinel, checked early: when it is absent, /config setup auto-runs just below
    # (once the command context exists) and reports every gap once, formatted — so the standalone
    # warning pass here is SKIPPED on first launch (the same problems would otherwise print twice
    # on the install's very first screen). The sentinel lives in the database directory so
    # deleting the database also resets first-run (a full reinstall should re-check).
    _setup_sentinel = get_config().path("database") / ".setup_done"
    _first_run = not _setup_sentinel.exists()

    # Health-check the active tier up front: a down daemon / un-pulled model / missing cloud key is
    # surfaced now with an actionable fix, rather than as a generic turn failure on the first query.
    # Non-fatal — the REPL still starts (commands work; an affected turn fails cleanly).
    if not _first_run:
        for problem in check_models():
            ui.warn(problem)
    # MCP servers connected (or failed) while registry imported — surface any problems with the
    # rest of the startup health report. /mcp shows the full status any time.
    from tools import mcp_client

    for problem in mcp_client.problems():
        ui.warn(problem)
    # A permissions.json that failed to load degraded the gate to defaults inside policy._load —
    # silently, since trust/ never imports tui. Surface it with the rest of the startup health
    # report (the mcp_client.problems() pattern); registry already triggered the load at import,
    # so the report is final by now.
    from trust import policy as _policy

    _policy_problem = _policy.load_problem()
    if _policy_problem:
        ui.warn(_policy_problem)

    # Carries the live session into slash-command handlers. `make_initial_state` lets
    # /reset rebuild state without commands.py importing back into agent.py.
    cmd_ctx = commands.CommandContext(
        state=state,
        make_initial_state=_initial_state,
        db_path=DB_PATH,
        session_started_at=datetime.now().isoformat(),  # session boundary for /cost
    )

    # First-run setup check: if the sentinel hasn't been written yet (checked above, where it
    # also suppresses the duplicate warning pass), auto-run /config setup so a fresh install
    # surfaces any gaps (Ollama down, models not pulled, keys missing) before the user's first
    # query, rather than as a confusing turn failure. Non-fatal: a dispatch error mustn't
    # prevent the REPL from starting.
    if _first_run:
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
    # an empty line, it asks the plan_gate to pause for plan review (acknowledged immediately by
    # ui.pause_note — the gate itself may be many seconds away on a local model). The in-progress
    # line + queue depth render live in the status bar (on_change -> ui). No-ops cleanly off-TTY
    # (see typeahead.InputQueue).
    input_queue = InputQueue(
        on_change=ui.set_input_preview, on_steer=ui.steer_note, on_pause=ui.pause_note
    )
    pause_controller = get_pause_controller()

    # Dev-only graph render. Import lazily here when enabling: utilities/ is deliberately not
    # part of the installed wheel (pyproject.toml), so a module-level import would crash every
    # pipx/uv-installed launch.
    # from utilities.print_graph import print_graph; print_graph(graph=graph)

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

    # Files dropped on the prompt and queued for the next turn (the drag-and-drop "[a]ttach"
    # choice below); consumed and cleared when that turn starts.
    pending_attachments: list[str] = []

    while True:
        # The idle prompt's exit semantics mirror the gate/ask: Ctrl-C is a soft no (drop the
        # half-typed line, keep the session), Ctrl-D / exhausted stdin EXITS through the same
        # /quit path as typing it (autosave included; never `continue` — a closed stdin would
        # spin forever). run_turn owns in-turn Ctrl-C separately.
        try:
            user_input = _next_input()
        except KeyboardInterrupt:
            ui.note("cancelled — /quit (or Ctrl-D) exits")
            continue
        except EOFError:
            commands.dispatch("/quit", cmd_ctx)  # the one quit path (autosave + farewell)
            break

        # A line that is nothing but an existing file path is a drag-and-drop onto the terminal
        # (the terminal pastes the path, quoted when it has spaces) — offer the two things a file
        # gets dropped for instead of sending a bare path to the agent as a query. Checked before
        # the slash-command intercept so an absolute POSIX path (which starts with `/`) isn't
        # mistaken for a command. Enter falls through and the path runs as an ordinary message.
        dropped = mentions.dropped_path(user_input)
        if dropped:
            label = mentions.display(dropped)
            ingestable = Path(dropped).suffix.lower() in SUPPORTED_EXTENSIONS
            choices = ("[i]ngest into knowledge base · " if ingestable else "") + \
                "[a]ttach to next message · [Enter] send as-is"
            choice = ui.ask(f"file dropped: {label} — {choices} » ").lower()
            if choice.startswith("i") and ingestable:
                commands.dispatch(f"/docs add {dropped}", cmd_ctx)
                continue
            if choice.startswith("a"):
                pending_attachments.append(dropped)
                ui.note(f"{label} will be attached to your next message.")
                continue

        # `/`-prefixed lines are REPL meta-commands, not agent turns — intercept them here.
        # `not dropped` keeps the drag-and-drop promise: a POSIX absolute path ("/home/…")
        # whose owner chose "[Enter] send as-is" must run as a message, not fall through to
        # dispatch as an unknown slash command.
        if not dropped and commands.is_command(user_input):
            commands.dispatch(user_input, cmd_ctx)
            if cmd_ctx.should_quit:
                break
            state = cmd_ctx.state  # a command (e.g. /reset) may have swapped state out
            if cmd_ctx.requeue:
                # The command queued a query to run NOW (today only /retry full, which rewinds
                # the last turn and re-runs its question) — fall through and run it as an
                # ordinary agent turn instead of returning to the prompt.
                user_input = cmd_ctx.requeue
                cmd_ctx.requeue = None
            else:
                continue

        if not user_input.strip():
            continue

        state = _fresh_turn(state, user_input)
        # Expand @file mentions: read any files the user referenced as `@path` and stash their
        # contents on state for the grounding node to fold into context (so every node sees the
        # file inline; dropped files queued via "[a]ttach" ride along as extra_paths). The message
        # text itself is left untouched — the @mention stays visible.
        attach_block, attached = mentions.expand(user_input, extra_paths=pending_attachments)
        pending_attachments = []
        if attached:
            state["attachments"] = attach_block
            ui.note("attached " + ", ".join(mentions.display(p) for p in attached))
            _warn_flagged_attachments(attach_block, ui.warn)
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
        # gate -> the approval prompt. (/autoapprove no longer needs a branch here: it opens the
        # gate policy itself, so the approval node stops interrupting at all.) Keeping this
        # dispatch here lets run_turn stay interrupt-type-agnostic (it just feeds the result back
        # as the resume value).
        def on_interrupt(value):
            if isinstance(value, dict) and value.get("type") == "plan_review":
                return ui.review_plan(value)
            return ui.ask_approval(value)

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
            if (trace_note := _trace_warning(tracer)):
                ui.warn(trace_note)
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
            if (trace_note := _trace_warning(tracer)):
                ui.warn(trace_note)
            cmd_ctx.state = state
            continue
        finally:
            # Discard any pause request still pending at turn end. A keypress that lands AFTER the
            # turn's last plan_gate (e.g. during the final agent message or synthesize) is never
            # consumed by the gate's clear(), and would otherwise leak into the next, unrelated turn
            # and pause it. A `/plan pause` issued at the prompt is set after this point, so it
            # survives; review mode re-arms each turn — neither is affected.
            # One exception: a STEER carries the user's typed correction — don't silently drop
            # their words. Salvage it into the type-ahead queue so it runs as the next message
            # (echoed when drained); the note explaining why prints after the answer renders,
            # never here (printing inside finally could interleave with the live answer region).
            late_req = pause_controller.peek()
            late_steer = (
                late_req.reason
                if late_req is not None and late_req.source == "steer" and late_req.reason
                else None
            )
            if late_steer:
                input_queue.push(late_steer)
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
        # Hand the finished turn to the answer renderer as provenance (the live Glass Box slice):
        # the Sources footer renders trust-colored and a tainted answer is called out in red under
        # the receipt — the /glass headline facts, native on every answer. Best-effort inside ui.
        ui.set_turn_provenance(state)
        # The answer streamed live during synthesize — close it out (final markdown render + receipt).
        # If nothing streamed (e.g. the model yielded no content, or the turn aborted at the plan
        # gate before synthesize produced text), fall back to rendering the recorded final message.
        if answer.started:
            # Pass the RECORDED final message: synthesize may append the citations Sources footer
            # after the token stream ended, so the streamed chars alone would silently drop it.
            final = state["messages"][-1].content if state.get("messages") else None
            answer.finish(final if isinstance(final, str) else None)
        else:
            ui.response(state["messages"][-1].content)

        # A steering correction that landed after the turn's last step boundary was salvaged into
        # the type-ahead queue (see the finally block above) — tell the user what happened to it.
        # On the error/Ctrl-C paths this note is skipped, but the queued line still echoes when
        # drained, so the correction is never invisible.
        if late_steer:
            ui.note(
                "your steering correction arrived after the turn had finished — it could not be "
                "applied mid-turn, so it will run as your next message instead."
            )

        # A tripped trace breaker (stores/trace._trip) degrades recording silently by design —
        # the degradation itself must not be silent. After the answer renders, never inside the
        # live region.
        if (trace_note := _trace_warning(tracer)):
            ui.warn(trace_note)

        # If this turn pushed the context past the compaction threshold, summarize older turns now so
        # the next turn starts with a smaller window (best-effort; see _maybe_autocompact). Runs after
        # the answer is rendered so its LLM call never delays the response the user is waiting on.
        state = _maybe_autocompact(state)
        cmd_ctx.state = state
