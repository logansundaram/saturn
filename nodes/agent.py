"""
The agent node — the v2 loop (2026-09-27; spec: docs/superpowers/specs/2026-09-27-v2-react-loop-design.md).

One native tool-calling call per pass, streamed. Its message either carries tool calls
(→ approval → tools → back here) or is the answer (→ END). Everything that used to be a
judge is a deterministic check here, in this order, and each costs the common case nothing:

  1. steer      a mid-turn correction (Esc + text) lands as a STEER_PREFIX HumanMessage;
  2. pause      an Esc pause interrupt()s for the pause prompt: continue / steer / abort;
  3. cap        past runtime.max_iterations the pass runs with tools UNBOUND and a budget note
                — a real answer, never a stub;
  4. generate   the call (the `_generate` seam the tests replace). Adaptive thinking
                (`runtime.think`, on evidence since 2026-09-29): a pass thinks, under
                `runtime.think_budget`, only when the tool round just before it had an error —
                the one place the model needs a new approach. Pass one, a clean round, a
                declined or blocked call and the capped pass stay think-off. A thinking pass
                that returns neither text nor a call is rerun once think-off;
  5. hygiene    on each emitted call: unknown tool, missing arguments (core/tool_args),
                a repeat of a call the user DECLINED this turn, or a third identical call —
                each answered with an error ToolMessage that routes straight back here;
  6. answer     a message without tool calls gets the mechanical trailers (the Sources
                receipt, the incidents note) on the RECORDED message — never on the stream.

Prompt order is prefix-cache order (docs/OPTIMIZATIONS.md, "the prefix cache"):
    [system][user: stable grounding][history…][user: dynamic grounding + request][turn…]
The first two are the primed lineage (core/prime.py); the bound tools render into the chat
template's system section, so the catalog is part of that cached prefix.
"""

from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

import diag
from config import get_config
from core.context import grounding_parts
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from core.llms import (extract_prompt_tokens, extract_tok_per_sec, generate, get_model,
                       invoke_kwargs, model_tag)
from core.llms import stream as llm_stream
from core.messages import agent_sys_msg
from core.pause import get_pause_controller
from core.state import STEER_PREFIX, AgentState, is_steer_message, is_turn_start, issuing_message
from core.tool_args import coerce_args, schema_hint, tool_for_args
from core.sources import build_sources
from textutil import SOURCES_HEADER, clip, fmt_args, split_sources_footer

ROLE = "tool_caller"

# The hygiene observations — one producer each; the rail and the tests key on them.
ALREADY_DECLINED_TEXT = ("Not executed: the user already declined this exact call this turn. "
                         "Do not retry it — tell the user it was not done.")
STALL_TEXT = ("Not executed: this exact call was already made twice this turn and its outcome "
              "is above. Do not repeat it — answer from what you have, or do something different.")
ASK_ALONE_TEXT = ("Not executed: ask_user must be called on its own. Ask the question first; act "
                  "on the answer in your next turn.")
MALFORMED_NOTE = ("Your previous reply was not a valid tool call (its arguments were not valid "
                  "JSON). Either call a tool with well-formed JSON arguments, or answer in text.")
MALFORMED_TEXT = ("I could not complete this: the model produced a malformed tool call twice. "
                  "Please rephrase the request.")
UNKNOWN_TOOL_TEXT = "Error: unknown tool {name!r}. Use only the tools you were given."
BUDGET_NOTE = ("The action budget for this turn is spent. Answer now from what you have, and "
               "state plainly what was not done.")
ABORT_TEXT = "Stopped at your request."
NO_ANSWER_TEXT = "No answer text was produced for this turn."
INCIDENTS_NOTE_HEADER = "Note — the following could not be completed:"

# A call may repeat once (a re-read after an edit is legitimate); the third identical call this
# turn is a loop.
STALL_REPEATS = 2
_INCIDENT_STATUSES = ("skipped", "blocked", "error")
_INCIDENT_CAP = 160


# ── this turn's record ────────────────────────────────────────────────────────────────────────


def _turn_start(messages: list) -> int:
    """Index of the current turn's request (a steer note is not a boundary); len() when none."""
    for i in range(len(messages) - 1, -1, -1):
        if is_turn_start(messages[i]):
            return i
    return len(messages)


def _this_turn(messages: list) -> list:
    return list(messages[_turn_start(messages):])


def _call_key(name, args) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
    except Exception:
        return f"{name}:{args!r}"


def _rounds(this_turn: list) -> list:
    """Every tool call this turn that has an observation, as (key, name, args, status,
    observation) — the record the hygiene checks and the incidents note read."""
    calls: dict = {}
    for m in this_turn:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                calls[tc.get("id")] = (tc.get("name"), tc.get("args"))
    out = []
    for m in this_turn:
        if isinstance(m, ToolMessage) and m.tool_call_id in calls:
            name, args = calls[m.tool_call_id]
            status = (getattr(m, "additional_kwargs", None) or {}).get("saturn_status") or "done"
            out.append((_call_key(name, args), name, args, status, str(m.content)))
    return out


# ── the prompt ────────────────────────────────────────────────────────────────────────────────


def strip_trailers(text: str) -> str:
    """A recorded answer without its mechanical trailers (the Sources receipt, the incidents
    note) — what prior answers look like in the model's history. The trailers are for the
    user; sent back every turn they cost tokens and invite the model to imitate the footer."""
    prose, _entries = split_sources_footer(str(text or ""))
    i = prose.rfind("\n\n" + INCIDENTS_NOTE_HEADER)
    if i != -1:
        prose = prose[:i]
    return prose.rstrip()


def _history(messages: list) -> list:
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and isinstance(m.content, str):
            stripped = strip_trailers(m.content)
            if stripped != m.content:
                m = AIMessage(content=stripped, id=getattr(m, "id", None))
        out.append(m)
    return out


def _llm_input(state: AgentState, messages: list, extra: "list | None" = None) -> list:
    """`extra` rides at the very end and is never a turn boundary (the capped pass's budget
    note is a HumanMessage, which is_turn_start would otherwise read as a new request)."""
    stable, dynamic = grounding_parts(state)
    out = [agent_sys_msg()]
    if stable:
        out.append(HumanMessage(content=stable))
    start = _turn_start(messages)
    out.extend(_history(messages[:start]))
    if start < len(messages):
        text = str(messages[start].content)
        if dynamic:
            text = dynamic + "\n\nUser request:\n" + text
        out.append(HumanMessage(content=text))
        out.extend(messages[start + 1:])
    out.extend(extra or [])
    return out


def _calls_of(ai) -> "tuple[list, set]":
    """The message's tool calls plus its MALFORMED ones (arguments that were not valid JSON —
    LangChain parks those on `invalid_tool_calls`), as ordinary call dicts with empty args and
    the set of ids that were malformed, so hygiene refuses them with the schema hint instead
    of the turn dying on a small model's broken JSON."""
    calls = list(getattr(ai, "tool_calls", None) or [])
    malformed: set = set()
    for tc in getattr(ai, "invalid_tool_calls", None) or []:
        cid = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        calls.append({"name": tc.get("name"), "args": {}, "id": cid, "type": "tool_call"})
        malformed.add(cid)
    return calls, malformed


# ── the call ──────────────────────────────────────────────────────────────────────────────────


def _generate(llm_input: list, *, tools: bool, think: bool = False) -> AIMessage:
    """ONE streamed call (the test seam). Tokens reach the UI through LangGraph's messages
    mode (app/turn.py filters this node); the chunks are folded into one AIMessage here so
    state/trace/autosave see exactly what the model produced. `tools=False` is the capped last
    pass: no bind, so the model can only answer. `think=True` is an adaptive thinking pass:
    the reasoning rides the chunks' `reasoning_content`, never `content`, so the response
    stream stays the answer."""
    from tools.registry import tool as registered

    model = get_model(ROLE)
    runnable = model.bind_tools(list(registered)) if tools else model
    kwargs = invoke_kwargs(ROLE, None, 0.0, task="agent", think=think)
    full = None
    for chunk in llm_stream(runnable, llm_input, tag=model_tag(ROLE), **kwargs):
        full = chunk if full is None else full + chunk
    if full is None:  # a model that streamed nothing — blocking fallback
        full = generate(runnable, llm_input, tag=model_tag(ROLE), **kwargs)
    content = full.content if isinstance(full.content, str) else str(full.content)
    calls = []
    for tc in getattr(full, "tool_calls", None) or []:
        calls.append({"name": tc.get("name"), "args": tc.get("args"),
                      "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                      "type": "tool_call"})
    kw = {"response_metadata": dict(getattr(full, "response_metadata", None) or {})}
    if getattr(full, "usage_metadata", None):
        kw["usage_metadata"] = full.usage_metadata
    return AIMessage(content=content, tool_calls=calls,
                     invalid_tool_calls=list(getattr(full, "invalid_tool_calls", None) or []),
                     **kw)


def _is_parse_failure(exc: Exception) -> bool:
    """A model call that failed because the MODEL's output was malformed — not because the
    daemon is down. langchain-ollama raises OutputParserException for non-JSON tool arguments,
    pydantic refuses a tool call whose arguments parsed to a non-object, and the daemon itself
    answers "error parsing tool call" for a broken <tool_call> block (a known small-model
    failure). Anything else propagates: a turn must fail loudly on a real fault."""
    if isinstance(exc, (OutputParserException, ValidationError)):
        return True
    text = str(exc).lower()
    return "tool call" in text and ("pars" in text or "invalid" in text or "malformed" in text)


def _generate_or_retry(llm_input: list, *, tools: bool, think: bool = False) -> "AIMessage | None":
    """`_generate`, retried ONCE with the malformed-call note when the model's output could not
    be parsed; None when the retry failed the same way (the caller answers honestly)."""
    try:
        return _generate(llm_input, tools=tools, think=think)
    except Exception as exc:
        if not _is_parse_failure(exc):
            raise
        diag.log(f"agent_node : malformed model output ({type(exc).__name__}: {exc}) — retrying once")
    try:
        return _generate(llm_input + [HumanMessage(content=MALFORMED_NOTE)], tools=tools, think=think)
    except Exception as exc:
        if not _is_parse_failure(exc):
            raise
        diag.log(f"agent_node : malformed model output twice ({type(exc).__name__}) — answering honestly")
        return None


# ── adaptive thinking ─────────────────────────────────────────────────────────────────────────

# Evidence that the next move needs a new approach: a call in the latest round failed — a tool
# error or a hygiene refusal. A declined ("skipped") or air-gapped ("blocked") call is NOT
# evidence: its next move is already known (say it was not done), and a thinking pass whose
# right move is a short answer is exactly the shape that comes back empty (below).
_EVIDENCE_STATUS = "error"


def _latest_round(this_turn: list) -> list:
    """The ToolMessages answering the last tool-calling message this turn — the round the
    coming pass reacts to. A steer note the user typed after it is skipped, not a boundary."""
    out = []
    for m in reversed(this_turn):
        if isinstance(m, ToolMessage):
            out.append(m)
        elif isinstance(m, HumanMessage) and is_steer_message(m):
            continue
        else:
            break
    return out


def _wants_think(this_turn: list, capped: bool) -> bool:
    """Whether THIS pass thinks (`runtime.think`). `adaptive` thinks only on the pass right
    after a round with an error, so a chat question, a clean lookup and every wrap-up answer
    cost nothing, and the thinking lands on the one decision that proved hard. Not sticky: a
    later clean round turns it off again. `on` thinks on every uncapped pass; `off` never
    does; the capped pass never does."""
    mode = str(get_config().get("runtime.think", "adaptive") or "off").strip().lower()
    if capped or mode == "off":
        return False
    if mode == "on":
        return True
    return any((getattr(m, "additional_kwargs", None) or {}).get("saturn_status") == _EVIDENCE_STATUS
               for m in _latest_round(this_turn))


def _is_empty(ai: AIMessage) -> bool:
    """A pass that produced neither answer text nor a call (valid or malformed)."""
    return (not str(ai.content or "").strip() and not getattr(ai, "tool_calls", None)
            and not getattr(ai, "invalid_tool_calls", None))


# ── hygiene ───────────────────────────────────────────────────────────────────────────────────


def _hygiene(call: dict, rounds: list, malformed: bool = False) -> "tuple[dict, ToolMessage | None]":
    """The corrected call, or the ToolMessage that answers it instead of running it."""
    from tools.registry import tools_by_name

    name = str(call.get("name") or "")

    def refuse(text, status="error"):
        return call, ToolMessage(content=text, tool_call_id=call["id"], name=name,
                                 additional_kwargs={"saturn_status": status})

    if name not in tools_by_name:
        return refuse(UNKNOWN_TOOL_TEXT.format(name=name))
    if malformed:
        return refuse("Error: " + schema_hint(name, "the arguments were not valid JSON"))
    raw = call.get("args")
    other = tool_for_args(name, raw)
    if other:
        return refuse("Error: " + schema_hint(
            other, f"those arguments belong to {other}, not {name}; the call was not run"))
    args = coerce_args(name, raw)
    if args is None:
        problem = ("the arguments were not an object" if not isinstance(raw, dict)
                   else f"required arguments missing from {raw!r}")
        return refuse("Error: " + schema_hint(name, problem))
    same = [r for r in rounds if r[0] == _call_key(name, args)]
    if any(r[3] == "skipped" for r in same):
        return refuse(ALREADY_DECLINED_TEXT, "skipped")
    if len(same) >= STALL_REPEATS:
        return refuse(STALL_TEXT)
    return {**call, "args": args}, None


# ── the answer trailers ───────────────────────────────────────────────────────────────────────


def sources_footer(tool_results, documents_retrieved) -> str:
    """The receipt of what informed the answer: one line per tool call / document, in the order
    they were gathered (core.sources — the same numbering /trace source uses).
    '' when nothing was."""
    _tools, _docs, sources = build_sources(tool_results, documents_retrieved)
    if not sources:
        return ""
    return SOURCES_HEADER + "\n" + "\n".join(f"  [{n}] {label}" for n, label in sources)


_INCIDENT_WORDING = {
    "skipped": "declined at the approval gate — not done",
    "blocked": "blocked by the air-gap — nothing was sent",
}


def incidents(this_turn: list) -> list:
    """One line per tool CALL that did NOT complete — declined at the gate (skipped), refused
    by the air-gap (blocked), or failed (error) — read off the ToolMessages' structural stamp
    and worded for the user (the observations are written for the model). One line per
    distinct call: a declined call the model re-issued is one incident, not two."""
    out = []
    seen: set = set()
    for key, name, args, status, obs in _rounds(this_turn):
        if status not in _INCIDENT_STATUSES or key in seen:
            continue
        seen.add(key)
        why = _INCIDENT_WORDING.get(status) or f"failed: {clip(' '.join(obs.split()), _INCIDENT_CAP)}"
        out.append(f"{name}({fmt_args(args or {}, 60)}) — {why}")
    return out


def _with_trailers(text: str, state: AgentState, this_turn: list) -> str:
    content = str(text or "").rstrip() or NO_ANSWER_TEXT
    inc = incidents(this_turn)
    if inc:
        content += f"\n\n{INCIDENTS_NOTE_HEADER}\n" + "\n".join(f"- {i}" for i in inc)
    if get_config().get("runtime.citations", True):
        footer = sources_footer(state.get("tool_results"), state.get("documents_retrieved"))
        if footer:
            content += "\n\n" + footer
    return content


# ── the node ──────────────────────────────────────────────────────────────────────────────────


def _steer_message(text: str) -> HumanMessage:
    return HumanMessage(content=f"{STEER_PREFIX} {text.strip()}")


def agent_node(state: AgentState):
    start = time.perf_counter()
    controller = get_pause_controller()
    messages = list(state.get("messages") or [])
    new: list = []
    iteration = int(state.get("iteration", 0) or 0) + 1
    updates: dict = {"iteration": iteration}

    # 1. steer — only when no pause is outstanding (a pause outranks a steer, and this branch
    # must not consume the node on a post-interrupt re-run: see core/pause.py).
    if not controller.pending():
        for r in controller.take_steers():
            if r is not None and str(r.reason).strip():
                new.append(_steer_message(str(r.reason)))

    # 2. pause — read non-destructively, cleared only past the interrupt.
    if controller.pending():
        req = controller.peek()
        decision = interrupt({
            "type": "pause",
            "reason": (req.reason if req and req.reason else "pause requested"),
            "iteration": iteration,
            "plan": state.get("plan") or [],
        })
        controller.clear()
        action = decision.get("action") if isinstance(decision, dict) else "continue"
        if action == "abort":
            updates["messages"] = new + [
                AIMessage(content=_with_trailers(ABORT_TEXT, state, _this_turn(messages + new)))
            ]
            diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (aborted at the pause prompt)")
            return updates
        if action == "steer" and str(decision.get("text") or "").strip():
            new.append(_steer_message(str(decision["text"])))

    # 3. the cap — the last pass answers without tools. The budget note rides the prompt only
    # (never state: a HumanMessage there would read as a new turn boundary).
    capped = iteration >= get_config().max_iterations
    extra = [HumanMessage(content=BUDGET_NOTE)] if capped else []

    # 4. generate — a malformed model output is retried once, then answered honestly; any
    # other failure propagates (the REPL reports "Turn failed").
    this_turn = _this_turn(messages + new)
    think = _wants_think(this_turn, capped)
    llm_input = _llm_input(state, messages + new, extra)
    ai = _generate_or_retry(llm_input, tools=not capped, think=think)
    if think and ai is not None and _is_empty(ai):
        # On a pass whose right move is a short answer, a thinking model can write the answer
        # inside its reasoning and emit nothing (qwen3.5 4b and 9b, 2026-09-29). The same
        # pass think-off answers; its prefix is already cached, so the rerun is cheap.
        diag.log("agent_node : thinking pass returned nothing — rerunning think-off")
        think = False
        ai = _generate_or_retry(llm_input, tools=not capped, think=False)
    if ai is None:
        updates["messages"] = new + [AIMessage(content=_with_trailers(MALFORMED_TEXT, state, this_turn))]
        diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (malformed output twice)")
        return updates
    stats = SimpleNamespace(response_metadata=getattr(ai, "response_metadata", None) or {},
                            usage_metadata=getattr(ai, "usage_metadata", None))
    updates["tok_per_sec"] = extract_tok_per_sec(stats)
    updates["context_tokens"] = extract_prompt_tokens(stats)

    calls, malformed = _calls_of(ai)
    if not calls or capped:
        # 6. the answer
        final = AIMessage(content=_with_trailers(ai.content, state, this_turn),
                          response_metadata=ai.response_metadata,
                          usage_metadata=ai.usage_metadata)
        updates["messages"] = new + [final]
        diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (answer, iter {iteration}"
                 f"{', think' if think else ''})")
        return updates

    # 5. hygiene
    rounds = _rounds(this_turn)
    kept, replies = [], []
    for call in calls:
        fixed, reply = _hygiene(call, rounds, malformed=call["id"] in malformed)
        kept.append(fixed)
        replies.append(reply)
    # ask_user runs ALONE: its interrupt re-executes the tools node from the top on resume, so
    # any sibling would run twice (an approved write, a second draft). The first question is
    # kept; every other call in the pass is answered here with "ask first".
    live = [c for c, r in zip(kept, replies) if r is None]
    first_ask = next((c for c in live if c.get("name") == "ask_user"), None)
    if first_ask is not None:
        for i, (c, r) in enumerate(zip(kept, replies)):
            if r is None and c is not first_ask:
                replies[i] = ToolMessage(content=ASK_ALONE_TEXT, tool_call_id=c["id"],
                                         name=str(c.get("name") or ""),
                                         additional_kwargs={"saturn_status": "error"})
    answered = [r for r in replies if r is not None]
    ai = AIMessage(content=ai.content, tool_calls=kept, response_metadata=ai.response_metadata,
                   usage_metadata=ai.usage_metadata)
    updates["messages"] = new + [ai] + answered
    diag.log(f"agent_node : {time.perf_counter() - start:.4f}s -> "
             f"{', '.join(str(c.get('name')) for c in kept)} ({len(answered)} answered here"
             f"{', think' if think else ''})")
    return updates


def route_after_agent(state: AgentState) -> str:
    """A tool-calling message with unanswered calls → approval; one whose every call the node
    answered itself → straight back to agent; anything else → end."""
    msgs = state.get("messages") or []
    last, answered = issuing_message(msgs)
    calls = (getattr(last, "tool_calls", None) or []) if isinstance(last, AIMessage) else []
    if not calls:
        return "end"
    if all(tc.get("id") in answered for tc in calls):
        return "agent"
    return "approval"
