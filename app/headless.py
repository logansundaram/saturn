"""The headless path: `saturn -p "query"` / `saturn -q "question"` — one query, print, exit.

No TUI and no human at the approval gate, so gated (side-effecting/destructive) tool calls are
DENIED by default; `--yolo` opens the gate policy up front (handled in agent.main). `--json`
emits one machine-readable result object; `--export` writes the run record after the answer.

`-q` is a RENDERING of the same turn, not a second execution mode: the engine loop, the
deny-by-default approver, and the trace recording are byte-identical to `-p`. What differs is
presentation only — stdout carries ONLY the final answer (pipe-clean), step-line progress goes
to stderr, and the run is auto-exported so the closing `recorded: saturn --replay <file>`
receipt names a command that actually works.
"""

import sys
import uuid
from pathlib import Path

from app import __version__
from app.cli import _read_piped_stdin
from app.graph import DB_PATH
from app.session import _fresh_turn, _initial_state
from app.startup import startup_load, _warn_flagged_attachments
from app.turn import run_turn, _make_on_update, _trace_warning
from core import mentions
from core.state import current_step
from stores.trace import Tracer


def _q_progress(emit=None):
    """The -q stderr progress renderer — a pure observer over the same node deltas the tracer
    already receives (no engine coupling; a rendering seam only). Announces the plan draft /
    revision, then each move of the execution pointer (the first step whose `result` is None —
    THE pointer, core.state.current_step). Deltas without a plan pass through silently; nothing
    here ever touches stdout (the pipe-clean contract)."""
    if emit is None:
        emit = lambda line: print(line, file=sys.stderr)  # noqa: E731
    seen = {"plan": [], "announced": None}

    def on_progress(node, delta):
        plan = delta.get("plan") if isinstance(delta, dict) else None
        if isinstance(plan, list) and plan:
            seen["plan"] = plan
            if node == "plan":
                emit(f"plan drafted — {len(plan)} step(s)")
            elif node == "replan":
                emit(f"plan revised — {len(plan)} step(s)")
        cur = current_step(seen["plan"])
        if cur is None:
            return
        key = (cur.get("step_id"), cur.get("label"))
        if key == seen["announced"]:
            return  # a status flip (pending -> active) is not a pointer move
        seen["announced"] = key
        pos = next((i + 1 for i, s in enumerate(seen["plan"]) if s is cur), 0)
        label = cur.get("label") or cur.get("intended_tool") or "(unlabeled)"
        emit(f"step {pos}/{len(seen['plan'])}: {label}")

    return on_progress


def _replay_receipt(dest) -> str:
    """The -q closing receipt: the exact `saturn --replay <file>` invocation that renders the
    just-written export offline. Quoted only when the path carries whitespace, so the printed
    command pastes back into a shell verbatim."""
    path = str(dest)
    if any(ch.isspace() for ch in path):
        path = f'"{path}"'
    return f"recorded: saturn --replay {path}"


def run_headless(args) -> None:
    """Run one query headlessly (the -p and -q paths): load, run the turn, print the answer
    (or the -p --json object) to stdout, write the export record (-p: on --export; -q:
    always — the receipt's replay command must name a real file), and return. Exits 1 through
    sys.exit on a failed turn or a failed export write. -q differs from -p in RENDERING only:
    stderr progress + the `recorded:` receipt; engine, approver, and trace are shared."""
    query = args.prompt if args.prompt is not None else args.query
    q_mode = args.prompt is None
    graph, ingest_warning = startup_load(interactive=False)
    if ingest_warning:
        print(ingest_warning, file=sys.stderr)
    # The gate posture warning the interactive startup block prints — same fact, stderr:
    # a permissions.json that failed to load means persisted /policy risk overrides and
    # /policy allow prefixes are NOT in force for this run.
    from trust import policy as _policy

    if _policy.load_problem():
        print(f"warning: {_policy.load_problem()}", file=sys.stderr)
    tracer = Tracer(DB_PATH)
    state = _initial_state()
    state = _fresh_turn(state, query)
    # @file mentions work headlessly too: `saturn -p "summarize @notes.md"` attaches the
    # file exactly as the interactive loop does (the grounding node folds it into context).
    attach_block, attached = mentions.expand(query)
    if attached:
        state["attachments"] = attach_block
    # Piped stdin is the other headless attachment channel (`git diff | saturn -p
    # "review this change"`): clamped like an @file and appended as its own clearly-labeled
    # block. A TTY / closed / empty stdin attaches nothing (see _read_piped_stdin).
    piped = _read_piped_stdin()
    if piped:
        stdin_block = "### Piped input (stdin)\n```\n" + piped + "\n```"
        state["attachments"] = (
            state["attachments"] + "\n" + stdin_block
            if state["attachments"]
            else stdin_block
        )
    _warn_flagged_attachments(
        state["attachments"], lambda m: print(f"! {m}", file=sys.stderr)
    )
    thread_id = str(uuid.uuid4())
    run_id = tracer.start_run(thread_id, query)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer.llm_handler(run_id)],
    }

    def _headless_approver(value):
        """Resolve interrupts with no human present. Gated tool calls are DENIED: the
        approval gate (the user seeing and approving the exact action) is the product's
        safety boundary, and headless mode silently approving a run_shell or write_file
        would delete it. --yolo opens the gate policy itself above, so under it the only
        interrupts that still fire are the quarantine ESCALATIONS (they gate
        independently of the policy threshold) — and those are approved here, because
        --yolo is exactly the user pre-approving everything; denying them would make
        '--yolo to allow them' a lie. The decline path is already honest — the agent tells
        the user the action was not performed. Any other interrupt type (the plan-review
        gate never arms headless) resumes unchanged via a bare True, which plan_gate
        tolerates by design."""
        if isinstance(value, dict) and value.get("type") == "approval_request":
            from trust import policy

            if policy.gate_off():
                return True
            names = ", ".join(
                tc.get("name", "?") for tc in value.get("tool_calls", [])
            )
            why = (
                " (escalated by the injection quarantine — a prior result looked "
                "instruction-shaped)"
                if value.get("quarantine")
                else ""
            )
            print(
                f"denied gated tool call(s): {names}{why} — headless mode does not "
                "approve gated actions; re-run with --yolo to allow them.",
                file=sys.stderr,
            )
            return False
        if isinstance(value, dict) and value.get("type") == "ask_user":
            # No human to ask headless: note the unanswered question on stderr; the bare True
            # resume makes the tool report "no answer" honestly (never a fabricated one).
            print(
                f"ask_user went unanswered (headless mode): {value.get('question')}",
                file=sys.stderr,
            )
            return True
        return True

    # --json: one machine-readable result object on stdout (the scripting/pipe contract).
    # Built from the same state/trace the interactive receipts render; `default=str` so an
    # odd arg value in tool_events can never crash the dump after the run itself succeeded.
    import json as _json
    import time as _time

    from core.state import summarize_gates

    def _emit_json(payload: dict) -> None:
        print(_json.dumps(payload, ensure_ascii=False, default=str))

    # -q rendering seams: the progress observer chained after the tracer's on_update, and a
    # first-token hook that announces "synthesizing…" the moment the answer starts generating.
    # Neither exists under -p — the execution path itself is identical either way.
    trace_update = _make_on_update(tracer, run_id, show_ui=False)
    on_update = trace_update
    on_token = None
    if q_mode:
        progress = _q_progress()

        def on_update(node, delta):
            trace_update(node, delta)
            progress(node, delta)

        _synth_seen = {"done": False}

        def on_token(_text, _logprobs=None):
            if not _synth_seen["done"]:
                _synth_seen["done"] = True
                print("synthesizing…", file=sys.stderr)

    _started = _time.perf_counter()
    try:
        state = run_turn(
            graph,
            state,
            config,
            approver=_headless_approver,
            on_update=on_update,
            on_token=on_token,
        )
        answer = state["messages"][-1].content
        tracer.end_run(run_id, "ok", answer)
        if (trace_note := _trace_warning(tracer)):
            print(f"warning: {trace_note}", file=sys.stderr)
        if args.json:
            _emit_json(
                {
                    "status": "ok",
                    "query": query,
                    "answer": answer,
                    "plan": state.get("plan", []),
                    "tools_called": state.get("tools_called", []),
                    "tool_events": state.get("tool_events", []),
                    # Human-gate record: how many calls were prompted + which tools were
                    # denied (headless denies gated calls by default, so this is the record
                    # of what the run was NOT allowed to do). Derived from the structured
                    # gate_events accumulator — the same record /glass and exports read.
                    "gates": summarize_gates(state.get("gate_events", [])),
                    "documents_retrieved": len(state.get("documents_retrieved", [])),
                    "iterations": state.get("iteration", 0),
                    "context_tokens": state.get("context_tokens", 0),
                    "tok_per_sec": round(float(state.get("tok_per_sec", 0.0) or 0.0), 1),
                    "duration_s": round(_time.perf_counter() - _started, 3),
                    "run_id": run_id,
                    "version": __version__,
                }
            )
        else:
            print(answer)
    except Exception as exc:
        tracer.end_run(run_id, "error", str(exc))
        if (trace_note := _trace_warning(tracer)):
            print(f"warning: {trace_note}", file=sys.stderr)
        if args.json:
            _emit_json(
                {
                    "status": "error",
                    "error": str(exc),
                    "query": query,
                    "duration_s": round(_time.perf_counter() - _started, 3),
                    "run_id": run_id,
                    "version": __version__,
                }
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            graph.checkpointer.delete_thread(thread_id)
        except Exception:
            pass
    # Export: write the run's record only AFTER the answer is out — a failed write must never
    # cost the user the answer the turn already produced (error to stderr, exit 1; stdout stays
    # the answer/JSON contract). -p exports on --export; -q always exports (default dest
    # logging/exports/run_<id>.json, or --export FILE to choose) — the receipt's
    # `saturn --replay <file>` command must name a file that exists.
    if args.export or q_mode:
        from commands.trace import export_run

        try:
            dest, _payload = export_run(
                DB_PATH,
                run_id,
                dest=Path(args.export).expanduser() if args.export else None,
            )
        except Exception as exc:
            target = args.export or "the run record"
            print(f"error: could not write export {target}: {exc}", file=sys.stderr)
            sys.exit(1)
        if q_mode:
            print(_replay_receipt(dest), file=sys.stderr)
        else:
            print(f"run #{run_id} exported -> {dest}", file=sys.stderr)
