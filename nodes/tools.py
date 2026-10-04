"""
Tool-execution node for the loop (agent → approval → tools → agent).

tool_node executes the tool calls on the last AI message, appends the results as ToolMessages
back into `messages` (so the model sees them next pass), and mirrors each completed gathering
call as `name(args) -> result` into the source accumulators — paired so the record can't divorce
a value from the call that produced it. Every call, whatever its outcome, is in `tool_events`.
"""

import time

from langchain.messages import ToolMessage
from langgraph.errors import GraphInterrupt

from trust import egress
from trust import quarantine
from tools.registry import RETRIEVAL_TOOLS, is_action, tools_by_name
from tools.planning import PLAN_TOOL, to_plan
from tools.toolspec import _HUMAN_APPROVED, ToolError
from core.state import AgentState, issuing_message
from textutil import CALL_RESULT_SEP, clip, fmt_call, head_tail, visible_controls_n

# Cap the one-line result preview carried in tool_events (UI tree); the full observation still
# rides messages/tool_results untouched.
_MAX_RESULT_PREVIEW = 160

# Hard cap on the observation length we feed BACK INTO the model (the ToolMessage + the paired
# tool_results record). Unbounded tool output — a big read_file, a full web_extract page, a fat
# web_search payload — silently overflows the Ollama context window: it truncates from the front,
# dropping the system prompt, and the agent starts misbehaving with no error. We keep the
# head and tail (the start usually has the answer; the tail often has a summary/conclusion) and
# mark the elision so the model knows it isn't seeing everything. ~12k chars ≈ 3-4k tokens, which
# leaves room for the system prompt and conversation inside an 8k+ window.
_MAX_OBSERVATION = 12000

# Appended when the source layer turned control characters into symbols, so the model does not
# read `␛` as text the page or file literally contains (and edit_file can say why a match fails).
CONTROL_NOTE = ("\n[{n} terminal control character(s) in this output are shown as symbols such "
                "as ␛ (escape); they are not literal text]")


def _clamp_observation(observation: str) -> str:
    """Bound an observation fed back to the model so one large tool result can't blow the context
    window. Keeps a head + tail with a marker noting how much was dropped (textutil.head_tail —
    the one home for the head+tail idiom; the marker text here is part of the contract the model
    and the clamp tests read, so it rides the marker parameter, not a re-rolled copy)."""
    return head_tail(
        observation,
        _MAX_OBSERVATION,
        marker="\n\n... [truncated {dropped} characters of tool output] ...\n\n",
    )


def _preview(observation: str) -> str:
    """Collapse a tool observation to a single capped line for the UI's tool-I/O tree."""
    return clip(observation, _MAX_RESULT_PREVIEW)


# Cap the per-call egress annotation carried in tool_events: a call rarely produces more than a
# couple of boundary events, but a runaway one must not bloat every delta / trace row.
_MAX_EGRESS_EVENTS = 4


def _egress_slice(mark: int) -> list[dict]:
    """The egress events THIS call produced (the ledger slice since `mark`, captured just before
    the call ran), flattened to small JSON-safe dicts for the tool_events record. This is the
    per-call attribution that lets the live rail — and every /trace replay, since tool_events
    ride the deltas into the trace DB — show what left the machine at the moment it left, instead
    of only in the turn-end receipt. Tools execute sequentially in tool_node's loop and nothing
    else records egress while one runs, so the slice belongs to exactly this call. Best-effort:
    an unreadable ledger yields no annotation, never an error."""
    try:
        # An UNTRACKED run (a shell command, a stdio MCP call) crossed no boundary Saturn saw:
        # the receipt counts it, but it is neither a send nor a block to annotate here.
        events = [e for e in egress.events_since(mark) if e.status != egress.UNTRACKED]
    except Exception:
        return []
    out = [
        {
            "channel": e.channel,
            "host": e.host,
            "n_bytes": e.n_bytes,
            "status": e.status,
        }
        for e in events
    ]
    if len(out) > _MAX_EGRESS_EVENTS:
        out = out[:_MAX_EGRESS_EVENTS] + [{"more": len(out) - _MAX_EGRESS_EVENTS}]
    return out


def tool_node(state: AgentState):
    """Execute the pending tool calls and feed results back as ToolMessages.

    The batch is the most recent tool-calling AIMessage's calls MINUS any call that already has
    a ToolMessage: the approval gate answers rejected calls itself (decline ToolMessages) and
    still routes here so the approved/ungated remainder runs (core.state.issuing_message)."""
    last, answered = issuing_message(state["messages"])
    pending_calls = [
        tc for tc in (getattr(last, "tool_calls", None) or []) if tc["id"] not in answered
    ]

    # The calls a human said yes to at the gate (auto-approved calls have no gate event).
    approved_ids = {
        c.get("id")
        for ev in state.get("gate_events") or [] if isinstance(ev, dict)
        for c in ev.get("calls") or [] if isinstance(c, dict) and c.get("approved")
    }

    tool_messages = []
    tools_called = []
    tool_results = []
    documents_retrieved = []
    tool_events = []
    plan_update = None

    for tool_call in pending_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        ok = True
        egress_mark = egress.next_seq()  # anything recorded past this seq belongs to THIS call
        start = time.perf_counter()
        selected = tools_by_name.get(name)
        if selected is None:
            observation = f"Error: unknown tool '{name}'."
            ok = False
        else:
            approved_token = _HUMAN_APPROVED.set(tool_call["id"] in approved_ids)
            try:
                observation = selected.invoke(args)
            except GraphInterrupt:
                # An interrupting tool (ask_user) pausing the graph is CONTROL FLOW, not a tool
                # error — swallowing it here would answer the question with the exception's repr
                # and never reach the human. LangGraph re-runs this node from the top on resume,
                # which is why nodes/agent.py lets ask_user run only ALONE in its batch: a
                # sibling call would execute twice.
                raise
            except ToolError as exc:  # the tool's own "this did not happen", worded for the model
                observation = f"Error: {exc}"
                ok = False
            except Exception as exc:  # surface tool errors to the model instead of crashing
                observation = f"Error calling {name}: {exc}"
                ok = False
            finally:
                _HUMAN_APPROVED.reset(approved_token)
        dur = time.perf_counter() - start
        if name == PLAN_TOOL and ok:
            # The checklist is state, not an observation: the rail, the gate's step context and
            # /trace why read state["plan"]. The observation still lands as a ToolMessage below.
            plan_update = to_plan(args.get("steps") if isinstance(args, dict) else None)

        # Terminal controls become visible symbols BEFORE anything else reads the text (the
        # clamp, quarantine, state, the trace, the model, the rail preview): an untrusted page
        # or file must not reach the terminal as a live escape sequence, and the record stays
        # safe to replay (textutil.visible_controls).
        observation, n_controls = visible_controls_n(str(observation))
        # Clamp what flows back into the model (ToolMessage + paired tool_results) so one large
        # result can't overflow the context window; the UI preview is derived from the same
        # clamped text. The _preview cap above is just for the one-line tool-I/O tree.
        clamped = _clamp_observation(observation)
        if n_controls:
            clamped += CONTROL_NOTE.format(n=n_controls)
        # Prompt-injection quarantine: an UNTRUSTED observation (web, MCP, ingested docs)
        # that carries instruction-shaped content is flagged (rail warning + gate context — and,
        # in `gate` mode, one fresh approval prompt for the next batch) and fenced between
        # explicit data-not-instructions markers before the model sees it. Clean content passes
        # through byte-identical. A FAILED call is scanned too: a remote server writes its own
        # error text, and a failed command's output is as external as a successful one's. See
        # quarantine.py.
        q_kinds: list[str] = []
        if quarantine.active() and quarantine.is_untrusted(name):
            findings = quarantine.scan(clamped)
            if findings:
                quarantine.flag(name, findings)
                clamped = quarantine.wrap_observation(clamped, findings)
                q_kinds = sorted({f.kind for f in findings})
        # Per-call boundary record (computed here so the outcome stamp below can see it): what
        # this call sent over the network, or what air-gap blocked.
        sent = _egress_slice(egress_mark)
        # Structural outcome stamp (nodes/agent.py's incidents note and guards read it off the message):
        # derived HERE, where the call actually ran, so a call's status never has to be sniffed
        # back out of observation text — a successful read of a file whose content happens to
        # start with "ERROR:" or "Blocked …" must not count as a failure. "blocked" = every boundary
        # event this call produced was an air-gap refusal (the observation is the refusal).
        boundary = [e for e in sent if isinstance(e, dict) and "status" in e]
        if not ok:
            call_status = "error"
        elif boundary and all(e.get("status") == egress.BLOCKED for e in boundary):
            call_status = "blocked"
        else:
            call_status = "done"
        tool_messages.append(
            ToolMessage(
                content=clamped,
                tool_call_id=tool_call["id"],
                name=name,
                additional_kwargs={"saturn_status": call_status},
            )
        )
        tools_called.append(name)
        # What the answer could draw on — the Sources receipt, /trace source and /trace why
        # number these (core/sources.py). Only a call that COMPLETED and returns material: a
        # failed or air-gap-blocked call informed nothing (it is in the incidents note), and a
        # tool declared side_effecting (write_file, remember, schedule_notification) returns a
        # confirmation of what it changed, not something to cite — as does a destructive one
        # (send_message, delete_calendar_event) unless what it returns is external output
        # (run_shell, run_shortcut, an MCP tool: the tools declared untrusted). Retrieval
        # results go to documents_retrieved, every other tool's to tool_results paired with
        # its call — keeping retrieval OUT of tool_results keeps a passage from being cited twice.
        action = is_action(name)
        if call_status != "done" or action:
            pass
        elif name in RETRIEVAL_TOOLS:
            documents_retrieved.append(clamped)
        elif name == PLAN_TOOL:
            pass  # the checklist is state, not a source the answer drew on
        else:
            # The call paired with its observation (CALL_RESULT_SEP); the Sources receipt's
            # labels split on it to recover the call half from the observation.
            tool_results.append(f"{fmt_call(name, args)}{CALL_RESULT_SEP}{clamped}")
        # Structured per-call record for the UI's tool-I/O tree (args + result preview + timing).
        event = {
            "name": name,
            "args": args,
            "result": _preview(observation),
            "dur": dur,
            "ok": ok,
            "pass": state.get("iteration"),  # the agent pass that issued it (benchmark ordering)
        }
        if q_kinds:
            event["quarantine"] = q_kinds
        # The per-call egress slice (computed above), rendered live as a rail leaf and persisted
        # with the event for /trace replays.
        if sent:
            event["egress"] = sent
        tool_events.append(event)

    result = {
        "messages": tool_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "documents_retrieved": documents_retrieved,
        "tool_events": tool_events,
    }
    if plan_update is not None:
        result["plan"] = plan_update
    return result
