"""
Tool-execution node for the living-plan ReAct loop (Phase 1).

tool_node executes the tool calls on the last AI message, appends the results as ToolMessages
back into `messages` (so the model sees them next iteration), and mirrors each
`name(args) -> result` into the trace accumulators — paired so synthesis can't divorce a value
from the call that produced it.
"""

import time

from langchain.messages import ToolMessage

from trust import egress
from trust import quarantine
from config import get_config
from tools.registry import tools_by_name, RETRIEVAL_TOOLS
from core.state import AgentState
from textutil import CALL_RESULT_SEP, clip, fmt_args, head_tail

# Cap each argument's length so a big write_file payload doesn't bloat the trace/synthesis input.
_MAX_ARG_REPR = 200

# Cap the one-line result preview carried in tool_events (UI tree); the full observation still
# rides messages/tool_results untouched.
_MAX_RESULT_PREVIEW = 160

# Hard cap on the observation length we feed BACK INTO the model (the ToolMessage + the paired
# tool_results record). Unbounded tool output — a big read_file, a full web_extract page, a fat
# web_search payload — silently overflows the Ollama context window: it truncates from the front,
# dropping the system prompt and plan, and the agent starts misbehaving with no error. We keep the
# head and tail (the start usually has the answer; the tail often has a summary/conclusion) and
# mark the elision so the model knows it isn't seeing everything. ~12k chars ≈ 3-4k tokens, which
# leaves room for the system prompts, plan, and conversation inside an 8k+ window.
_MAX_OBSERVATION = 12000


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


def _fmt_call(name: str, args: dict) -> str:
    """Render a tool call like  calculate(expression='847 * 293 + 12450')  for the trace and
    for synthesis, so results stay linked to the call that produced them."""
    return f"{name}({fmt_args(args, _MAX_ARG_REPR)})"


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
        events = egress.events_since(mark)
    except Exception:
        return []
    out = [
        {
            "channel": e.channel,
            "host": e.host,
            "n_bytes": e.n_bytes,
            "redactions": e.redactions,
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
    still routes here so the approved/ungated remainder runs. Walk back over those trailing
    ToolMessages to find the issuing AIMessage."""
    answered = set()
    last = None
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            answered.add(m.tool_call_id)
            continue
        last = m
        break
    pending_calls = [
        tc for tc in (getattr(last, "tool_calls", None) or []) if tc["id"] not in answered
    ]

    # Dry-run: decide everything, execute nothing. Each pending call is stubbed with an explicit
    # "would execute" observation (no side effects, no network, no file writes) so the agent can lay
    # out its whole intended arc and synthesize can report it. The plan still advances mechanically
    # (tools_called is recorded) so the loop progresses and terminates exactly as a real turn would.
    dry_run = bool(get_config().get("runtime.dry_run", False))

    tool_messages = []
    tools_called = []
    tool_results = []
    documents_retrieved = []
    tool_events = []

    for tool_call in pending_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        ok = True
        egress_mark = egress.next_seq()  # anything recorded past this seq belongs to THIS call
        start = time.perf_counter()
        if dry_run:
            observation = (
                f"[DRY RUN] would execute {_fmt_call(name, args)} — not run "
                "(no side effects, no network, no files touched)."
            )
        else:
            selected = tools_by_name.get(name)
            if selected is None:
                observation = f"Error: unknown tool '{name}'."
                ok = False
            else:
                try:
                    observation = selected.invoke(args)
                except Exception as exc:  # surface tool errors to the model instead of crashing
                    observation = f"Error calling {name}: {exc}"
                    ok = False
        dur = time.perf_counter() - start

        observation = str(observation)
        # Clamp what flows back into the model (ToolMessage + paired tool_results) so one large
        # result can't overflow the context window; the UI preview is derived from the same
        # clamped text. The _preview cap above is just for the one-line tool-I/O tree.
        clamped = _clamp_observation(observation)
        # Prompt-injection quarantine: an UNTRUSTED observation (web, http, MCP, ingested docs)
        # that carries instruction-shaped content is flagged (rail warning + gate context — and,
        # in `gate` mode, one fresh approval prompt for the next batch) and fenced between
        # explicit data-not-instructions markers before the model sees it. Clean content passes
        # through byte-identical. See quarantine.py.
        q_kinds: list[str] = []
        if ok and not dry_run and quarantine.active() and quarantine.is_untrusted(name):
            # Register the untrusted result as a taint source (the data->action check the gate runs
            # over later tool calls) BEFORE fencing — record the content the model will see, not our
            # markers — then scan it for injection phrasing (the data-as-instructions check).
            quarantine.record_untrusted(name, clamped)
            findings = quarantine.scan(clamped)
            if findings:
                quarantine.flag(name, findings)
                clamped = quarantine.wrap_observation(clamped, findings)
                q_kinds = sorted({f.kind for f in findings})
        tool_messages.append(
            ToolMessage(content=clamped, tool_call_id=tool_call["id"], name=name)
        )
        tools_called.append(name)
        # Retrieval results go to documents_retrieved (synthesize's "Retrieved documents"); every
        # other tool's result is paired with its call in tool_results ("Tool results") so synthesis
        # can't divorce the value from what it answers. Keeping retrieval OUT of tool_results avoids
        # feeding the same passage to the synthesizer twice.
        if name in RETRIEVAL_TOOLS:
            documents_retrieved.append(clamped)
        else:
            # The one serialization both downstream parsers split (textutil.split_call_result):
            # synthesize's Sources labels take the call half, the Glass Box's taint corpus the
            # observation half.
            tool_results.append(f"{_fmt_call(name, args)}{CALL_RESULT_SEP}{clamped}")
        # Structured per-call record for the UI's tool-I/O tree (args + result preview + timing).
        event = {
            "name": name,
            "args": args,
            "result": _preview(observation),
            "dur": dur,
            "ok": ok,
        }
        if q_kinds:
            event["quarantine"] = q_kinds
        # Per-call boundary record: what this call sent over the network (or what air-gap blocked),
        # rendered live as a rail leaf and persisted with the event for /trace replays.
        sent = _egress_slice(egress_mark)
        if sent:
            event["egress"] = sent
        tool_events.append(event)

    return {
        "messages": tool_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "documents_retrieved": documents_retrieved,
        "tool_events": tool_events,
    }
