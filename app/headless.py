"""The headless path: `saturn -p "query"` — one query, print the answer, exit.

No TUI and no human at the approval gate, so gated (side-effecting/destructive) tool calls are
DENIED by default; `--yolo` opens the gate policy up front (handled in agent.main). `--json`
emits one machine-readable result object; `--export` writes the run record after the answer.
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
from stores.trace import Tracer


def run_headless(args) -> None:
    """Run one query headlessly (the -p path): load, run the turn, print the answer (or the
    --json object) to stdout, optionally --export the run record, and return. Exits 1 through
    sys.exit on a failed turn or a failed export write."""
    graph, ingest_warning = startup_load(interactive=False)
    if ingest_warning:
        print(ingest_warning, file=sys.stderr)
    # The gate posture warning the interactive startup block prints — same fact, stderr:
    # a permissions.json that failed to load means persisted /risk overrides and /allow
    # prefixes are NOT in force for this run.
    from trust import policy as _policy

    if _policy.load_problem():
        print(f"warning: {_policy.load_problem()}", file=sys.stderr)
    tracer = Tracer(DB_PATH)
    state = _initial_state()
    state = _fresh_turn(state, args.prompt)
    # @file mentions work headlessly too: `saturn -p "summarize @notes.md"` attaches the
    # file exactly as the interactive loop does (the grounding node folds it into context).
    attach_block, attached = mentions.expand(args.prompt)
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
    run_id = tracer.start_run(thread_id, args.prompt)
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [tracer.llm_handler(run_id)],
    }

    def _headless_approver(value):
        """Resolve interrupts with no human present. Gated tool calls are DENIED: the
        approval gate (the user seeing and approving the exact action) is the product's
        safety boundary, and headless mode silently approving a run_shell or write_file
        would delete it. --yolo opens the gate policy itself above, so under it the only
        interrupts that still fire are the quarantine/taint ESCALATIONS (they gate
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
        return True

    # --json: one machine-readable result object on stdout (the scripting/pipe contract).
    # Built from the same state/trace the interactive receipts render; `default=str` so an
    # odd arg value in tool_events can never crash the dump after the run itself succeeded.
    import json as _json
    import time as _time

    from core.state import summarize_gates

    def _emit_json(payload: dict) -> None:
        print(_json.dumps(payload, ensure_ascii=False, default=str))

    _started = _time.perf_counter()
    try:
        state = run_turn(
            graph,
            state,
            config,
            approver=_headless_approver,
            on_update=_make_on_update(tracer, run_id, show_ui=False),
        )
        answer = state["messages"][-1].content
        tracer.end_run(run_id, "ok", answer)
        if (trace_note := _trace_warning(tracer)):
            print(f"warning: {trace_note}", file=sys.stderr)
        if args.json:
            _emit_json(
                {
                    "status": "ok",
                    "query": args.prompt,
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
                    "query": args.prompt,
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
    # --export: write the run's export record only AFTER the
    # answer is out — a failed write must never cost the user the answer the turn already
    # produced (error to stderr, exit 1; stdout stays the answer/JSON contract).
    if args.export:
        from commands.trace import export_run

        try:
            dest, _payload = export_run(
                DB_PATH, run_id, dest=Path(args.export).expanduser()
            )
            print(f"run #{run_id} exported -> {dest}", file=sys.stderr)
        except Exception as exc:
            print(f"error: could not write export {args.export}: {exc}", file=sys.stderr)
            sys.exit(1)
