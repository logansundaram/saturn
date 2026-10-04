"""The interactive loop: prompt → turn → answer, with everything that makes it a session.

Startup (splash, banner, posture line, health checks, first-run setup), the one input reader
(type-ahead + Esc steering/pause), drag-and-drop file offers, `!command` passthrough, slash-command dispatch, the
turn lifecycle (trace run, interrupts, streaming answer), checkpoint pruning,
autosave, and auto-compaction. One call — `run_repl()` — owns the whole session.
"""

import os
import sys

from datetime import datetime
from pathlib import Path

import commands
import diag
from app import bang
from app.graph import DB_PATH
from app.session import (_fresh_turn, _initial_state, _maybe_autocompact, skill_completions,
                         skill_for_line, think_for_line)
from app.startup import startup_load, start_warm_up, _warn_flagged_attachments
from core import prime
from app.turn import close_run, open_run, run_turn, _make_on_update, _trace_warning
from config import get_config
from core import mentions
from core import skills
from core.pause import get_pause_controller
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

    # The menu bar icon (notify/menubar.py): record this agent's pid so the icon can show
    # "agent running" and its Quit can stop us, then make sure the icon itself is up (a login
    # LaunchAgent that outlives this terminal). Best-effort — a launch never depends on it.
    import atexit
    import signal

    from notify import menubar as _menubar

    _menubar.write_pid()
    atexit.register(_menubar.clear_pid, os.getpid())  # ours only — a second REPL may own it now

    # The icon's Quit sends SIGTERM; turn it into a normal exit so the prompt's raw mode and
    # the atexit hooks unwind instead of leaving the terminal wedged.
    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except (ValueError, OSError):
        pass  # not the main thread / unsupported here
    if _menubar.enabled():
        _menubar_status = _menubar.ensure_running()
        diag.log(f"menubar: {_menubar_status}")
        if _menubar_status.startswith("failed"):
            ui.warn(f"menu bar icon {_menubar_status} (/notify icon start to retry)")

    # The user's hooks.yaml (core/hooks): say once what it got wrong, so a hook that silently
    # never runs can't happen. No file, no cost.
    from core import hooks as _hooks

    for problem in _hooks.problems():
        ui.warn(f"{_hooks.hooks_path()}: {problem}")

    # The user's skills (core/skills): a broken file, or one a built-in command shadows, is
    # named once here — a skill that silently never runs can't happen.
    from core import skills as _skills

    for problem in _skills.problems():
        ui.warn(f"skill {problem}")
    for name in sorted(n for n in _skills.discover() if commands.resolves(n)):
        ui.warn(f"skill /{name}: the built-in /{name} wins, so it never runs — rename its file")
    # A think setting that is not one of its values runs as its default — and says so, rather
    # than silently meaning something else (a bare YAML `on` used to run as adaptive).
    from core import think as _think

    for problem in _think.problems():
        ui.warn(problem)

    # Startup header — tier/model / tool count / corpus size, like a tool's first line.
    from core.llms import model_id, check_models
    from tools.registry import tool as _tools
    from stores.rag import iter_documents

    cfg = get_config()
    n_docs = sum(1 for _ in iter_documents())  # same definition RAG ingests by
    ui.banner(f"{cfg.active_tier}:{model_id()}", len(_tools), n_docs)
    # Launched from "/" or an unreadable folder, core/workspace fell back to home — say so once.
    from core import workspace as _ws
    try:
        _started = Path.cwd().resolve()
    except OSError:
        _started = None
    if _started != _ws.root():
        ui.warn(f"working in {_ws.display(_ws.root())}, not the folder Saturn was started in — "
                "cd to a specific folder and restart to work there")
    # The session's trust posture, deviation-only (receipt.posture_spans): silent on a
    # default-safe install; speaks (with the /policy pointer) when the gate is
    # loosened, the air-gap holds, inference leaves the machine, or a guard is weakened.
    ui.posture_line()

    # First-run sentinel: when it is absent, /models runs just below (once the command context
    # exists) and the health check waits until after it — the pick may change the tier. The
    # sentinel lives in the database directory so deleting the database also resets first-run
    # (a full reinstall should re-check).
    _setup_sentinel = get_config().path("database") / ".setup_done"
    _first_run = not _setup_sentinel.exists()

    def _health_check() -> None:
        """Surface a down daemon / un-pulled model now, with the fix, rather than as a generic
        turn failure on the first query. Non-fatal — the REPL still starts. A healthy tier gets
        its weights loaded on a background thread, so the first query does not pay the model
        load inside its first agent call (app.startup.warm_model)."""
        problems = check_models()
        for problem in problems:
            ui.warn(problem)
        if not problems:
            start_warm_up()

    if not _first_run:
        _health_check()
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
    # /clear rebuild state without commands.py importing back into agent.py.
    cmd_ctx = commands.CommandContext(
        state=state,
        make_initial_state=_initial_state,
        db_path=DB_PATH,
    )

    # First launch: /models prices the ladder against this machine and asks which tier and
    # embedder to run (Enter = the recommendation; missing models are pulled on consent), then
    # the health check examines the tier the session will actually use. Non-fatal: a dispatch
    # error mustn't prevent the REPL from starting.
    if _first_run:
        ui.note("First launch — choose your models (won't repeat; re-run any time with /models).")
        try:
            commands.dispatch("/models", cmd_ctx)
        except Exception as exc:
            ui.warn(f"/models failed: {exc}")
        _health_check()
        try:
            _setup_sentinel.parent.mkdir(parents=True, exist_ok=True)
            _setup_sentinel.touch()
        except Exception as exc:
            diag.log(f"first-run sentinel write failed: {exc}")

    # Memory candidates left over from an earlier session (a /quit that skipped the review, a
    # crash, a bare Ctrl-D) — say so once; the review itself is never forced on launch.
    try:
        from core.memory_review import load_pending

        _n_pending = len(load_pending())
        if _n_pending:
            ui.note(f"{_n_pending} memory candidate(s) waiting from an earlier session — "
                    "/memory review to keep or drop them.")
    except Exception as exc:
        diag.log(f"memory review: pending check failed: {exc}")

    # One input reader for the session. While a turn runs it captures type-ahead so the user can
    # queue follow-up queries / slash commands without waiting (drained between turns below). The Esc
    # key acts on whatever is typed: with text, it's a mid-turn steering correction (injected into
    # the running turn at the agent's next pass, acknowledged by ui.steer_note); with an empty
    # line, it asks the agent to pause at its next pass (acknowledged immediately by
    # ui.pause_note — the pass itself may be many seconds away on a local model). The in-progress
    # line + queue depth render live in the status bar (on_change -> ui). No-ops cleanly off-TTY
    # (see typeahead.InputQueue).
    input_queue = InputQueue(
        on_change=ui.set_input_preview, on_steer=ui.steer_note, on_pause=ui.pause_note,
    )
    pause_controller = get_pause_controller()

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
        return ui.prompt(commands.command_completions() + skill_completions())

    # Files dropped on the prompt and queued for the next turn (the drag-and-drop "[a]ttach"
    # choice below); consumed and cleared when that turn starts. `pending_blocks` are ready-made
    # context blocks queued the same way — the output of a `!command` the user ran.
    pending_attachments: list[str] = []
    pending_blocks: list[str] = []

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

        # `!command` runs the command in the user's own shell — their action, not the agent's —
        # prints the output, and attaches it to the next message (app/bang.py).
        if bang.is_bang(user_input):
            cmd = bang.command_of(user_input)
            output, code = bang.run(cmd)
            if output:
                print(output)
            pending_blocks.append(bang.attachment(cmd, output, code))
            ui.note(f"exit {code} · the output is attached to your next message")
            continue

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

        # `/think <request>` is that request, run for one turn at `deep` (core/think): the
        # prefix comes off here and the rest goes down the ordinary path. Bare `/think` and
        # `/think <level>` are the command's own and fall through to dispatch.
        deep_request = think_for_line(user_input) if not dropped else None
        if deep_request is not None:
            user_input = deep_request

        # `/`-prefixed lines are REPL meta-commands, not agent turns — intercept them here.
        # `not dropped` keeps the drag-and-drop promise: a POSIX absolute path ("/home/…")
        # whose owner chose "[Enter] send as-is" must run as a message, not fall through to
        # dispatch as an unknown slash command.
        # A `/name` that is not Saturn's own command but names one of the user's skills
        # (core/skills) runs as an ordinary turn with the skill in its grounding; `/name --help`
        # shows the skill instead, like every other slash spelling.
        invoked = skill_for_line(user_input) if not dropped else None
        if invoked is not None and invoked[1].lower() in ("--help", "-h"):
            commands.dispatch(f"/skills show {invoked[0].name}", cmd_ctx)
            continue
        if invoked is None and not dropped and commands.is_command(user_input):
            commands.dispatch(user_input, cmd_ctx)
            if cmd_ctx.should_quit:
                break
            state = cmd_ctx.state  # a command (e.g. /clear) may have swapped state out
            continue

        if not user_input.strip():
            continue

        state = _fresh_turn(state, user_input)
        if deep_request is not None:
            state["think_level"] = "deep"
            ui.note("thinking deep for this turn")
        if invoked is not None:
            state["skill"] = skills.block(invoked[0])
            ui.note(f"running your skill /{invoked[0].name} · {mentions.display(invoked[0].path)}")
        # Expand @file mentions: read any files the user referenced as `@path` and stash their
        # contents on state for the grounding node to fold into context (so every node sees the
        # file inline; dropped files queued via "[a]ttach" ride along as extra_paths). The message
        # text itself is left untouched — the @mention stays visible.
        attach_block, attached = mentions.expand(user_input, extra_paths=pending_attachments)
        pending_attachments = []
        if pending_blocks:
            attach_block = "\n\n".join(b for b in [attach_block, *pending_blocks] if b)
            pending_blocks = []
        if attach_block:
            state["attachments"] = attach_block
            if attached:
                ui.note("attached " + ", ".join(mentions.display(p) for p in attached))
            _warn_flagged_attachments(attach_block, ui.warn)
        thread_id, run_id, config = open_run(tracer, user_input)
        ui.reset_turn()  # reset node-timing + plan-diff state for this turn's trace
        # Renders the agent's answer token-by-token as it streams (on_token below). It
        # opens the response section on the first token and is finished (or aborted) after the turn.
        answer = ui.ResponseStream()

        # Resolve each interrupt by type: the Esc pause -> the pause prompt; ask_user -> the
        # question prompt; the approval gate -> the approval prompt. (/policy open needs no
        # branch: it opens the gate policy itself, so the approval node stops interrupting.)
        # Keeping this dispatch here lets run_turn stay interrupt-type-agnostic (it just feeds
        # the result back as the resume value).
        def on_interrupt(value):
            if isinstance(value, dict) and value.get("type") == "pause":
                # The agent node paused at the top of a pass: Enter continues, typed text steers
                # the running turn, q aborts it (nodes/agent.py reads the decision).
                return ui.pause_prompt(value)
            if isinstance(value, dict) and value.get("type") == "ask_user":
                # The ask_user tool: the agent's question renders at the prompt and the typed
                # line resumes the turn as the tool's observation ("" = no answer, reported
                # honestly by the tool).
                return ui.answer_question(value)
            return ui.ask_approval(value)

        prime.set_busy(True)  # an idle prime sequence must not queue this turn's calls
        try:
            state = run_turn(
                graph,
                state,
                config,
                approver=on_interrupt,
                on_update=_make_on_update(tracer, run_id, show_ui=cmd_ctx.show_ui,
                                          answer=answer),
                pause=input_queue,
                on_token=answer.feed,
                on_retract=answer.discard,
                on_thinking=ui.set_thinking,
            )
            tracer.end_run(run_id, "ok", state["messages"][-1].content)
            # The daemon is idle now: re-plant every lineage's prefix checkpoint for the next
            # turn (core/prime.py — rebuilt from disk, so this turn's writes are in it).
            prime.set_busy(False)
            prime.start_priming()
        except (KeyboardInterrupt, Exception) as exc:
            # Ctrl-C abandons the in-flight turn but not the session, and a node/tool failure
            # (Ollama timeout, decode error, tool bug) must never kill the REPL: record it, tell
            # the user, and drop back to the prompt with the conversation intact (the unanswered
            # query stays in `messages`; the next turn's _compact_history tolerates it).
            prime.set_busy(False)
            answer.abort()  # tear down the live answer region before the warning prints
            if isinstance(exc, KeyboardInterrupt):
                tracer.end_run(run_id, "interrupted", "turn cancelled by user (Ctrl-C)")
                ui.warn("Turn cancelled.")
            else:
                tracer.end_run(run_id, "error", str(exc))
                ui.warn(f"Turn failed: {exc}")
            if (trace_note := _trace_warning(tracer)):
                ui.warn(trace_note)
            cmd_ctx.state = state
            continue
        finally:
            # The turn boundary: prune the checkpoints and expire every task-scoped grant
            # (disclosed after the answer renders, below).
            expired_grants = close_run(graph, thread_id)
            # Discard any pause request still pending at turn end: a keypress that lands AFTER
            # the agent's last pass is never consumed by the node's clear(), and would leak into
            # the next, unrelated turn. A STEER carries the user's typed words, so it is salvaged
            # into the type-ahead queue to run as the next message; the note explaining why
            # prints after the answer renders, never here (inside finally it could interleave
            # with the live answer region).
            late_steers = [
                r.reason for r in pause_controller.take_steers() if r is not None and r.reason
            ]
            late_steer = "; ".join(late_steers) if late_steers else None
            late_req = pause_controller.peek()
            # A late PAUSE can't be salvaged (the turn is over), but the user saw the ⏸
            # acknowledgement — dropping it silently reads as "pause is broken". Noted after
            # the answer renders, like the late-steer note below.
            late_pause = late_req is not None and late_req.source != "steer"
            for text in late_steers:
                input_queue.push(text)  # each salvaged correction runs as its own next message
            pause_controller.reset()  # the turn boundary: nothing outstanding survives it
            # Autosave the conversation to the reserved /resume slot. The checkpoints we just pruned
            # can't restore a session, so this slot is what survives a quit/crash/Ctrl-C. Runs for
            # every outcome (ok/error/interrupt) since `state` always carries the latest messages;
            # write_autosave is itself best-effort, so a failure can't end the turn or the session.
            commands.write_autosave(state)

        cmd_ctx.state = state  # keep the command context pointed at the latest state
        # Learnable signal from this turn (steer notes, vetoes, gate denials, unfinished steps)
        # becomes memory CANDIDATES in the pending queue — reviewed at /memory review or /quit,
        # never written on their own (core/memory_review). Best-effort, off the answer path.
        try:
            from core.memory_review import add_pending, collect_turn

            add_pending(collect_turn(state, run_id))
        except Exception as exc:
            diag.log(f"memory review: candidate collection failed: {exc}")
        # The answer streamed live during the agent's last pass — close it out (final markdown
        # render + receipt). If nothing streamed (the model yielded no content, an abort at the
        # pause prompt), fall back to rendering the recorded final message.
        if answer.started:
            # Pass the RECORDED final message: the agent node appends the Sources footer and the
            # incidents note after the token stream ended, so the streamed chars alone would
            # silently drop them.
            final = state["messages"][-1].content if state.get("messages") else None
            answer.finish(final if isinstance(final, str) else None)
        else:
            ui.response(state["messages"][-1].content)

        # A steering correction that landed after the turn's last step boundary was salvaged into
        # the type-ahead queue (see the finally block above) — tell the user what happened to it.
        # On the error/Ctrl-C paths this note is skipped, but the queued line still echoes when
        # drained, so the correction is never invisible.
        if expired_grants.get("prefixes") or expired_grants.get("tools"):
            what = ", ".join(
                [f'run_shell "{p}"' for p in expired_grants.get("prefixes") or []]
                + list(expired_grants.get("tools") or [])
            )
            ui.note(f"always-allow grants expired with this turn: {what}  "
                    "(lifetime: /config runtime.grant_scope · durable: /policy allow / "
                    "/policy risk --save)")
        if expired_grants.get("failed"):
            ui.warn("a task-scoped tier grant could NOT be restored at the turn boundary — the "
                    "tool may still be auto-approved; check /policy and reset with "
                    "/policy risk <tool> reset ("
                    + "; ".join(expired_grants["failed"]) + ")")
        if late_steer:
            ui.note(
                "your steering correction arrived after the turn had finished — it could not be "
                "applied mid-turn, so it will run as your next message instead."
            )
        elif late_pause and any(e.get("outcome") == "cut-esc" for e in state.get("think") or []):
            # The Esc did its job: it stopped a thought, and the pass that answered without it
            # was the turn's last — there was no later pass to pause before.
            ui.note("Esc stopped the thought; the turn finished on the next answer, so there "
                    "was nothing left to pause.")
        elif late_pause:
            ui.note("your pause request arrived after the turn had finished — nothing left to "
                    "pause this turn.")

        # A tripped trace breaker (stores/trace._trip) degrades recording silently by design —
        # the degradation itself must not be silent. After the answer renders, never inside the
        # live region.
        if (trace_note := _trace_warning(tracer)):
            ui.warn(trace_note)

        # If this turn pushed the context past the compaction threshold, summarize older turns now so
        # the next turn starts with a smaller window (best-effort; see _maybe_autocompact). Runs after
        # the answer is rendered so its LLM call never delays the response the user is waiting on.
        state = _maybe_autocompact(state, run_id)
        cmd_ctx.state = state
