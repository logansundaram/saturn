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
from pathlib import Path

__version__ = "0.1.0"

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk

from core import budget
import diag
from config import get_config
from core.state import AgentState

# loop nodes
from nodes.ground import grounding_node
from nodes.plan import plan_node
from nodes.synthesize import synthesize_node
from nodes.update_plan import update_plan_node
from nodes.agent import agent_node, route_after_agent
from nodes.tools import tool_node
from nodes.approval import approval_node
from nodes.replan import replan_node
from nodes.plan_gate import plan_gate_node, route_after_gate

# RAG ingest (reconciles the disk-cached vector store the search_knowledge_base tool reads);
# SUPPORTED_EXTENSIONS gates the drag-and-drop ingest offer to file types the corpus accepts
from stores.rag import sync, SUPPORTED_EXTENSIONS

# transparency + safety UI
from stores.trace import Tracer
from tui import ui

# REPL meta-commands (lines starting with `/`)
import commands

# `@file` mention expansion (pull a referenced file's contents into the turn's context)
from core import mentions

# Human-in-the-loop plan-review pause: the shared controller.
from core.plan_ops import get_pause_controller

# Type-ahead: the single console reader during a turn — queues follow-up queries/commands the user
# types while the agent works, and carries the Esc plan-review pause trigger.
from tui.typeahead import InputQueue

DB_PATH = str(get_config().path("db_sqlite"))


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
    (bounded by REPLAN_BUDGET). See nodes/replan.py.

    `plan_gate` runs at every step boundary: a pass-through unless a pause has been requested, in
    which case it `interrupt()`s so the user can inspect/edit the plan and resume (see
    nodes/plan_gate.py). Compiled with a SqliteSaver checkpointer, which both persists
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

    A turn starts at a REAL user HumanMessage — not a standalone mid-turn steer note (that
    belongs to the turn it corrected; treating it as a boundary would compact away the very
    scratchpad this function promises to keep) and not a compaction summary (carried history).
    Everything from the boundary onward is kept as-is (the scratchpad is intact, so no orphaned
    tool calls); everything before it is reduced to Human + non-empty final-AI messages (also
    orphan-free). Run only at the turn boundary.

    `keep_recent_turns=0` reproduces the old strip-everything behaviour."""
    from core.state import is_turn_start

    human_idxs = [i for i, m in enumerate(messages) if is_turn_start(m)]
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
    from core.llms import active_context_window

    window = active_context_window()
    if not window or used <= 0:
        return state
    threshold = float(cfg.get("runtime.compact_threshold", 0.85) or 0.85)
    if used / window < threshold:
        return state

    from core.compaction import summarize_messages

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


def _fresh_turn(state: AgentState, user_input: str) -> AgentState:
    """Append the new query and reset per-turn fields (accumulators + loop counter).
    `messages` persists across turns to keep in-process conversation memory, but is first
    compacted (see _compact_history): older turns collapse to a clean Q&A transcript while the
    most recent turn's tool scratchpad is retained so a follow-up can refer back to it."""
    state["messages"] = _compact_history(state["messages"])
    state["messages"].append(HumanMessage(content=user_input))
    # Arm a fresh snapshot batch for this turn (lazy — created only if a file tool mutates
    # something), so /undo can reverse exactly the writes the turn that just ran made.
    from stores.snapshots import begin_turn

    begin_turn(user_input)
    # Clear the prompt-injection quarantine's per-turn flags (a flag raised last turn must not
    # escalate this turn's first tool batch).
    from trust import quarantine

    quarantine.reset_turn()
    state["current_query"] = user_input
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
    state["gate_events"] = []
    state["tok_per_sec"] = 0.0
    # context_tokens persists across turns (the context only grows; overwritten on next LLM call).
    return state


def _initial_state() -> AgentState:
    return {
        "messages": [],
        "current_query": "",
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
        "gate_events": [],
        "tok_per_sec": 0.0,
        "context_tokens": 0,
    }


def _build_parser():
    """The saturn CLI parser — strict (an unknown flag exits 2 instead of silently launching the
    TUI). The flag reference lives here, in --help, not in main()'s docstring."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="saturn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Saturday.ai — local-first, transparent agent.\n"
            "\n"
            "Run with no arguments for the interactive chat loop (/help lists commands,\n"
            "/quit exits). The flags below are the headless/automation surface."
        ),
        epilog=(
            "verbs:\n"
            "  saturn verify <file>  verify a /trace export offline — recomputes the sha256\n"
            "                        integrity digest and reports whether the content is\n"
            "                        unchanged since export. exit 0 = intact, 1 = digest\n"
            "                        mismatch (tampered), 2 = unreadable / not a Saturn export.\n"
            "\n"
            "headless mode (-p):\n"
            "  Read-only tools run freely; gated (side-effecting/destructive) tool calls are\n"
            "  DENIED by default — there is no human at the approval gate, and safe-by-default\n"
            "  must hold in every mode. Pass --yolo to auto-approve them. Piped stdin attaches\n"
            "  to the turn:\n"
            "    git diff | saturn -p \"review this change\"\n"
        ),
    )
    parser.add_argument("-p", "--prompt", metavar="QUERY", default=None,
                               help="Run a single query headlessly and print the answer to "
                                    "stdout (no TUI, no interactive prompts).")
    parser.add_argument("--yolo", action="store_true",
                               help="Open the approval gate for the whole run — auto-approve "
                                    "side-effecting/destructive tool calls, headless or "
                                    "interactive. The same view of the gate policy as "
                                    "/autoapprove (policy.set_gate_off — threshold: destructive).")
    parser.add_argument("--json", action="store_true",
                               help="With -p: print a structured JSON result (answer, plan, "
                                    "tools, tokens, timing) instead of the bare answer. Errors "
                                    "also emit JSON (status: \"error\") and still exit 1.")
    parser.add_argument("--export", metavar="FILE", default=None,
                               help="With -p: after the turn completes, write the run's complete "
                                    "export record (with a sha256 integrity digest) to FILE "
                                    "(the same artifact /trace export writes).")
    parser.add_argument("--replay", metavar="FILE", default=None,
                               help="Replay an exported run record (/trace export) offline — "
                                    "integrity-checked, no database needed — then exit.")
    parser.add_argument("--version", action="version", version=f"saturn {__version__}")
    return parser


def _parse_cli(argv=None):
    """Parse + validate the CLI line. Strict by design: argparse exits 2 on an unknown flag, and
    the cross-flag rules below exit 2 through parser.error — a typo'd invocation must never
    silently fall through to the interactive TUI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    # -p "" (present but blank) is an invocation mistake, distinct from -p absent (interactive).
    if args.prompt is not None and not args.prompt.strip():
        parser.error("empty prompt")
    if args.prompt is not None and args.replay:
        parser.error("--replay renders an export offline and cannot be combined with -p/--prompt")
    if args.export and args.prompt is None:
        parser.error("--export only applies to a headless turn — use it with -p/--prompt")
    if args.json and args.prompt is None:
        parser.error("--json only applies to a headless turn — use it with -p/--prompt")
    return args


def _verify_artifact(path_str) -> int:
    """`saturn verify <file>` — offline verification of a /trace export's integrity digest.
    Mirrors /trace verify's semantics with shell-friendly exit codes: 0 = content unchanged since
    export, 1 = digest mismatch (tampered) or no digest to check, 2 = usage/read errors. Results
    print to stdout, errors to stderr. Tamper-evidence, not provenance (the ed25519 signing layer
    was shelved)."""
    import json

    from trust import digest

    if not path_str:
        print("usage: saturn verify <file>", file=sys.stderr)
        return 2
    path = Path(path_str.strip('"')).expanduser()
    try:
        # utf-8-sig: a BOM (PowerShell 5.1 redirection writes one) must not fail the read.
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: could not read {path}: {e}", file=sys.stderr)
        return 2
    if not (isinstance(payload, dict) and payload.get("saturn_trace_export") == 1):
        print(f"error: {path.name} is not a Saturn trace export.", file=sys.stderr)
        return 2

    v = digest.verify_payload(payload)
    if not v["has_integrity"]:
        print(f"error: {path.name} carries no integrity digest — nothing to verify "
              "(only JSON exports carry one).", file=sys.stderr)
        return 1
    if v["digest_ok"]:
        print(f"{path.name}: digest ok — sha256 {v['stored_digest']}")
        return 0
    print(f"{path.name}: digest MISMATCH — the record was modified after export.")
    print(f"  stored   {v['stored_digest']}")
    print(f"  computed {v['computed_digest']}")
    return 1


def _ingest_warning(exc: Exception, *, reachable: "bool | None" = None,
                    interactive: bool = True) -> str:
    """One readable line for a failed startup knowledge-base ingest (non-fatal: the agent runs on
    without RAG). The common first-launch cause is the Ollama daemon being down — the embedder
    can't run — and the model health check that prints moments later already explains exactly
    that, so this line says it plainly and defers to it instead of dumping a multi-line httpx
    ConnectError repr right above the clean explanation of the same root cause. Headless (-p)
    prints no health check, so the deferral clause is dropped there. Any other failure keeps its
    exception, collapsed to one line. `reachable` overrides the live llms.ollama_reachable()
    probe (offline tests)."""
    if reachable is None:
        from core.llms import ollama_reachable

        reachable = ollama_reachable()
    if not reachable:
        return "knowledge-base ingest skipped (Ollama not reachable" + (
            " — the model check below explains)" if interactive else ")"
        )
    from textutil import clip

    detail = clip(exc, 300) or exc.__class__.__name__
    return f"knowledge-base ingest failed, continuing without RAG: {detail}"


def _warn_flagged_attachments(block: str, emit) -> None:
    """Attachment admission warning — @file mentions and piped stdin attach the user's OWN files,
    but their CONTENT often isn't the user's words (a downloaded PDF, a vendored README, a piped
    log). Instruction-shaped content gets one warning naming the patterns, never a block: the
    human chose to attach it; the point is that they KNOW what rode in with it. `emit` is the
    output channel (ui.warn interactively, stderr headless)."""
    try:
        from trust import quarantine

        if not block or not quarantine.active():
            return
        kinds = sorted({f.kind for f in quarantine.scan(block)})
        if kinds:
            emit(f"attachment contains instruction-shaped content ({', '.join(kinds)}) — "
                 f"the model sees it as data; watch the plan and gate for actions you didn't ask for")
    except Exception:
        pass  # a warning helper must never cost the turn


def _read_piped_stdin() -> str:
    """Piped stdin content for a headless turn, or "" when stdin is a TTY / closed / empty.
    Read as BYTES (sys.stdin.buffer) and decoded as UTF-8 with errors='replace': Windows opens a
    piped text-mode stdin as STRICT cp1252, so `git diff | saturn -p ...` would either mojibake
    the diff or raise UnicodeDecodeError on the first non-cp1252 byte — and a blanket except
    would then silently drop the whole pipe. A genuine OS read failure may still return "", but
    a decode can never empty the input. Clamped to the same per-attachment budget as an @file
    mention (mentions._MAX_FILE_CHARS — the one cap an attachment block honors), with the same
    head-only truncation marker."""
    try:
        stdin = sys.stdin
        if stdin is None or stdin.closed or stdin.isatty():
            return ""
        buffer = getattr(stdin, "buffer", None)
        if buffer is not None:
            # +1 past the budget detects truncation; ×4 because the budget is CHARS and UTF-8
            # spends up to 4 bytes per char — reading only budget+1 BYTES could under-read a
            # multi-byte stream and drop its tail without the truncation marker.
            raw = buffer.read((mentions._MAX_FILE_CHARS + 1) * 4)
            data = raw.decode("utf-8", errors="replace")
        else:
            # A replaced stdin with no byte layer (embedders, tests): already-decoded text,
            # so there is no strict-decode hazard left to guard.
            data = stdin.read(mentions._MAX_FILE_CHARS + 1)
    except (OSError, ValueError, AttributeError):
        return ""
    if not data.strip():
        return ""
    if len(data) > mentions._MAX_FILE_CHARS:
        data = data[: mentions._MAX_FILE_CHARS] + (
            f"\n… [truncated — piped stdin exceeds {mentions._MAX_FILE_CHARS} chars]"
        )
    return data


def main():
    """CLI entry point: parse the command line, then route — the `verify` verb / --replay /
    headless -p / the interactive TUI loop. The flag reference lives in `saturn --help`
    (see _build_parser)."""
    # `saturn verify <file>` is a VERB, not a flag — intercepted before argparse so the bare
    # no-args invocation can stay "launch the TUI" without a positional colliding with it.
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        sys.exit(_verify_artifact(" ".join(sys.argv[2:])))

    _args = _parse_cli()

    # --- replay path: render an exported run record and exit (no graph, no models) -----
    if _args.replay:
        from commands.trace import render_export

        sys.exit(0 if render_export(_args.replay) else 1)

    # --yolo: the CLI view of the gate policy — open the gate up front (threshold ->
    # destructive) so gated calls never interrupt; same mechanism as /autoapprove. Honored in
    # BOTH modes: interactively the status bar derives ⚠ GATE OFF straight from the live
    # threshold, so no extra UI wiring is needed.
    if _args.yolo:
        from trust import policy

        policy.set_gate_off(True)

    # The slow startup loading (knowledge-base ingest + graph build) runs while the ring art
    # animates in interactive mode, or directly (no TUI) in headless mode.
    def _startup_load(interactive: bool = True):
        warn = None
        # Reconcile the knowledge base against the disk cache at startup: only new/changed
        # documents are embedded, the rest load from the persisted store. Non-fatal if it fails
        # (e.g. embedding model not pulled) — search_knowledge_base just returns "no documents";
        # the warning is shaped by _ingest_warning (one line, daemon-down stated plainly).
        try:
            sync(verbose=False)
        except Exception as exc:
            warn = _ingest_warning(exc, interactive=interactive)
        return build_agent(), warn

    # --- headless path: one query, print answer, exit ---------------------------------
    if _args.prompt is not None:
        graph, ingest_warning = _startup_load(interactive=False)
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
        state = _fresh_turn(state, _args.prompt)
        # @file mentions work headlessly too: `saturn -p "summarize @notes.md"` attaches the
        # file exactly as the interactive loop does (the grounding node folds it into context).
        attach_block, attached = mentions.expand(_args.prompt)
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
        run_id = tracer.start_run(thread_id, _args.prompt)
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
            if _args.json:
                _emit_json(
                    {
                        "status": "ok",
                        "query": _args.prompt,
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
                        "session_tokens": budget.spent(),
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
            if _args.json:
                _emit_json(
                    {
                        "status": "error",
                        "error": str(exc),
                        "query": _args.prompt,
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
        # --export: write the run's export record (with a sha256 integrity digest) only AFTER the
        # answer is out — a failed write must never cost the user the answer the turn already
        # produced (error to stderr, exit 1; stdout stays the answer/JSON contract).
        if _args.export:
            from commands.trace import export_run

            try:
                dest, _payload = export_run(
                    DB_PATH, run_id, dest=Path(_args.export).expanduser()
                )
                print(f"run #{run_id} exported -> {dest}", file=sys.stderr)
            except Exception as exc:
                print(f"error: could not write export {_args.export}: {exc}", file=sys.stderr)
                sys.exit(1)
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
        if commands.is_command(user_input):
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

        # Surface the session token budget (runtime.token_budget) when it bites. Once spent, every
        # turn force-lands at synthesize without new tool rounds (route_after_agent) — which would
        # otherwise look like the agent silently refusing to work. Warn-each-turn is deliberate:
        # the condition persists until the user raises or clears the budget.
        if budget.exceeded():
            ui.warn(
                f"session token budget spent ({_human_int(budget.spent())} of "
                f"{_human_int(budget.limit())} tok) — turns now answer from what's already "
                "gathered, with no new tool calls. Raise or clear it with "
                "`/config runtime.token_budget <n|0>`."
            )
        elif budget.near():
            ui.note(
                f"session token budget {budget.spent() / budget.limit() * 100:.0f}% used "
                f"({_human_int(budget.spent())}/{_human_int(budget.limit())} tok)."
            )

        # If this turn pushed the context past the compaction threshold, summarize older turns now so
        # the next turn starts with a smaller window (best-effort; see _maybe_autocompact). Runs after
        # the answer is rendered so its LLM call never delays the response the user is waiting on.
        state = _maybe_autocompact(state)
        cmd_ctx.state = state


if __name__ == "__main__":
    main()
