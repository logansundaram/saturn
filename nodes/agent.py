"""
The agent node — the v2 loop (2026-09-27; spec: docs/superpowers/specs/2026-09-27-v2-react-loop-design.md).

One native tool-calling call per pass, think off, streamed. Its message either carries tool
calls (→ approval → tools → back here) or is the answer (→ END). Everything that used to be a
judge is a deterministic check here, in this order, and each costs the common case nothing:

  1. steer      a mid-turn correction (Esc + text) lands as a STEER_PREFIX HumanMessage;
  2. pause      an Esc pause interrupt()s for the pause prompt: continue / steer / abort;
  3. cap        past runtime.max_iterations the pass runs with tools UNBOUND and a budget note
                — a real answer, never a stub;
  4. generate   the call (the `_generate` seam the tests replace);
  5. hygiene    on each emitted call: unknown tool, missing arguments (core/tool_args),
                a repeat of a call the user DECLINED this turn, or a third identical call —
                each answered with an error ToolMessage that routes straight back here;
  6. answer     a message without tool calls gets the mechanical trailers (the Sources
                receipt, the incidents note) on the RECORDED message — never on the stream.

Prompt order is prefix-cache order (core/serving.py "the prefix cache"):
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
from core.llms import extract_prompt_tokens, extract_tok_per_sec, generate, get_model
from core.llms import stream as llm_stream
from core.messages import agent_sys_msg
from core.pause import get_pause_controller
from core.state import STEER_PREFIX, AgentState, is_turn_start
from core.structured import _invoke_kwargs, _model_tag
from core.tool_args import coerce_args, schema_hint
from textutil import SOURCES_HEADER, clip, fmt_args, parse_doc_sources, split_call_result

ROLE = "tool_caller"

# The hygiene observations — one producer each; the rail and the tests key on them.
ALREADY_DECLINED_TEXT = ("Not executed: the user already declined this exact call this turn. "
                         "Do not retry it — tell the user it was not done.")
STALL_TEXT = ("Not executed: this exact call already ran this turn and its result is above. "
              "Answer from it, or do something different.")
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
_MAX_SOURCE_LABEL = 100


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


def _llm_input(state: AgentState, messages: list, extra: "list | None" = None) -> list:
    """`extra` rides at the very end and is never a turn boundary (the capped pass's budget
    note is a HumanMessage, which is_turn_start would otherwise read as a new request)."""
    stable, dynamic = grounding_parts(state)
    out = [agent_sys_msg()]
    if stable:
        out.append(HumanMessage(content=stable))
    start = _turn_start(messages)
    out.extend(messages[:start])
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


def _generate(llm_input: list, *, tools: bool) -> AIMessage:
    """ONE streamed call (the test seam). Tokens reach the UI through LangGraph's messages
    mode (app/turn.py filters this node); the chunks are folded into one AIMessage here so
    state/trace/autosave see exactly what the model produced. `tools=False` is the capped last
    pass: no bind, so the model can only answer."""
    from tools.registry import tool as registered

    model = get_model(ROLE)
    runnable = model.bind_tools(list(registered)) if tools else model
    kwargs = _invoke_kwargs(ROLE, None, 0.0, task="agent")
    full = None
    for chunk in llm_stream(runnable, llm_input, tag=_model_tag(ROLE), **kwargs):
        full = chunk if full is None else full + chunk
    if full is None:  # a model that streamed nothing — blocking fallback
        full = generate(runnable, llm_input, tag=_model_tag(ROLE), **kwargs)
    content = full.content if isinstance(full.content, str) else str(full.content)
    calls = []
    for tc in getattr(full, "tool_calls", None) or []:
        calls.append({"name": tc.get("name"), "args": tc.get("args"),
                      "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                      "type": "tool_call"})
    meta = {k: v for k, v in (getattr(full, "response_metadata", None) or {}).items()
            if k != "logprobs"}
    kw = {"response_metadata": meta}
    if getattr(full, "usage_metadata", None):
        kw["usage_metadata"] = full.usage_metadata
    return AIMessage(content=content, tool_calls=calls,
                     invalid_tool_calls=list(getattr(full, "invalid_tool_calls", None) or []),
                     **kw)


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
    they were gathered. '' when nothing was."""
    labels = []
    for r in tool_results or []:
        labels.append(clip(split_call_result(r)[0], _MAX_SOURCE_LABEL))
    for d in documents_retrieved or []:
        names = parse_doc_sources(d)
        labels.append(clip("knowledge base: " + ", ".join(names), _MAX_SOURCE_LABEL)
                      if names else "knowledge base passage")
    if not labels:
        return ""
    return SOURCES_HEADER + "\n" + "\n".join(f"  [{i}] {lbl}" for i, lbl in enumerate(labels, 1))


def incidents(this_turn: list) -> list:
    """One line per tool round that did NOT complete — declined at the gate (skipped), refused
    by the air-gap (blocked), or failed (error). Read off the ToolMessages' structural stamp."""
    out = []
    for _key, name, args, status, obs in _rounds(this_turn):
        if status in _INCIDENT_STATUSES:
            out.append(f"{name}({fmt_args(args or {}, 60)}) — {status}: "
                       f"{clip(' '.join(obs.split()), _INCIDENT_CAP)}")
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
    capped = iteration > get_config().max_iterations
    extra = [HumanMessage(content=BUDGET_NOTE)] if capped else []

    # 4. generate
    ai = _generate(_llm_input(state, messages + new, extra), tools=not capped)
    stats = SimpleNamespace(response_metadata=getattr(ai, "response_metadata", None) or {},
                            usage_metadata=getattr(ai, "usage_metadata", None))
    updates["tok_per_sec"] = extract_tok_per_sec(stats)
    updates["context_tokens"] = extract_prompt_tokens(stats)

    this_turn = _this_turn(messages + new)
    calls, malformed = _calls_of(ai)
    if not calls or capped:
        # 6. the answer
        final = AIMessage(content=_with_trailers(ai.content, state, this_turn),
                          response_metadata=ai.response_metadata,
                          usage_metadata=ai.usage_metadata)
        updates["messages"] = new + [final]
        diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (answer, iter {iteration})")
        return updates

    # 5. hygiene
    rounds = _rounds(this_turn)
    kept, answered = [], []
    for call in calls:
        fixed, reply = _hygiene(call, rounds, malformed=call["id"] in malformed)
        kept.append(fixed)
        if reply is not None:
            answered.append(reply)
    ai = AIMessage(content=ai.content, tool_calls=kept, response_metadata=ai.response_metadata,
                   usage_metadata=ai.usage_metadata)
    updates["messages"] = new + [ai] + answered
    diag.log(f"agent_node : {time.perf_counter() - start:.4f}s -> "
             f"{', '.join(str(c.get('name')) for c in kept)} ({len(answered)} answered here)")
    return updates


def route_after_agent(state: AgentState) -> str:
    """A tool-calling message with unanswered calls → approval; one whose every call the node
    answered itself → straight back to agent; anything else → end."""
    msgs = state.get("messages") or []
    answered = set()
    last = None
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            answered.add(m.tool_call_id)
            continue
        last = m
        break
    calls = (getattr(last, "tool_calls", None) or []) if isinstance(last, AIMessage) else []
    if not calls:
        return "end"
    if all(tc.get("id") in answered for tc in calls):
        return "agent"
    return "approval"
