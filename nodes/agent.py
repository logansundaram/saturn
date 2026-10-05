"""
The agent node — the ReAct loop (spec: docs/superpowers/specs/2026-09-27-v2-react-loop-design.md).

One native tool-calling call per pass, streamed. Its message either carries tool calls
(→ approval → tools → back here) or is the answer (→ END). The checks around the call are
deterministic, in this order, and each costs the common case nothing:

  1. steer      a mid-turn correction (Esc + text) lands as a STEER_PREFIX HumanMessage;
  2. pause      an Esc pause interrupt()s for the pause prompt: continue / steer / abort;
  3. cap        from pass runtime.max_iterations on, no tool call runs: the pass is the same
                bound call (same prompt, so the daemon's prompt cache holds on the turn's
                largest prompt), and a call it emits is answered with the budget refusal and
                routed back for the answer. A model that answers the refusal with more calls
                is rerun once with tools UNBOUND and a budget note — a real answer, never a
                stub (unbinding re-prefills the whole prompt, so it is the fallback only);
  4. generate   the call (the `_generate` seam the tests replace). Whether it THINKS is
                core/think.py's decision, from the kind of step this pass is (first move,
                new information, after an error, steered, wrap-up, capped) and the user's
                level (`runtime.think`: fast | auto | deep). `auto` thinks before a pass
                ACTS: the pass is drafted think-off, a text answer stands, a tool call is
                retracted and rethought. A thought is bounded (`runtime.think_budget`) and
                Esc stops it; one that comes back empty, cut or stopped is dropped and the
                pass answers without it. At most two model calls per pass because of thinking;
  5. hygiene    on each emitted call: unknown tool, missing arguments (core/tool_args),
                a repeat of a call the user DECLINED this turn, or a third identical call
                with nothing changed since the first — each answered with an error ToolMessage
                that routes straight back here;
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
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

import diag
from config import get_config
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from core.llms import call_failure
from core.llms import (extract_prompt_tokens, extract_tok_per_sec, generate, get_model,
                       invoke_kwargs, model_tag)
from core.llms import stream as llm_stream
from core import think as _think
from core.messages import agent_sys_msg
from core.pause import get_pause_controller
from core.state import (STEER_PREFIX, AgentState, grounding_parts, is_steer_message,
                        issuing_message)
from core.state import this_turn as _this_turn, turn_start as _turn_start
from core.tool_args import coerce_args, schema_hint, tool_for_args
from core.sources import build_sources
from nodes.approval import provenance
from trust import quarantine
from textutil import SOURCES_HEADER, clip, fmt_args, split_sources_footer

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
# A call to a tool whose toolkit the user turned off (/tools). The prompt already says what is
# off; this answers the call a tool description or an earlier turn still led the model to.
TOOLKIT_OFF_TEXT = ("Not executed: {name} is part of the {key} toolkit, which the user has "
                    "turned off. Do not look for another way to do it. Tell the user it is "
                    "turned off and that `/tools on {key}` turns it on.")
UNKNOWN_HANDLE_TEXT = ("Not executed: {handle} appears nowhere in this conversation — not in what "
                       "the user wrote and not in any tool result — so it cannot be used. Look the "
                       "person up with search_contacts and use the number or address from their "
                       "card, or ask the user for it.")
UNKNOWN_CHAT_TEXT = ("Not executed: group chat {ref} did not come from find_group_chats in this "
                     "conversation, so it cannot be used. Find the group with find_group_chats and "
                     "use the chat ref it returns.")
MALFORMED_CALL_TEXT = ("Error: that tool call was not valid JSON (no tool name could be read). "
                       "Emit one call with a tool name you were given and its arguments as a "
                       "JSON object; otherwise answer in plain text.")
BUDGET_TEXT = ("Not executed: the action budget for this turn is spent, so no further tool call "
               "will run. Answer now from what you have, and state plainly what was not done.")
BUDGET_NOTE = ("The action budget for this turn is spent. Answer now from what you have, and "
               "state plainly what was not done.")
ABORT_TEXT = "Stopped at your request."
NO_ANSWER_TEXT = "No answer text was produced for this turn."
INCIDENTS_NOTE_HEADER = "Note — the following could not be completed:"

# A call may repeat once (a re-read is legitimate); the third identical call with nothing
# changed in between is a loop.
STALL_REPEATS = 2
_INCIDENT_STATUSES = ("skipped", "blocked", "error")
# Long enough for a whole remedy: an error that says what to do (grant Full Disk Access …, 186
# characters) is the user's next step, and 160 cut it at "Full Disk…" (2026-10-02, run 45).
_INCIDENT_CAP = 300


# ── this turn's record ────────────────────────────────────────────────────────────────────────


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
    """`extra` rides at the very end and is never a turn boundary (the hard stop's budget
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
        # `or ""`: broken JSON can leave the name unreadable, and the AIMessage the node
        # rebuilds from these dicts refuses a None name — hygiene then says "malformed".
        calls.append({"name": tc.get("name") or "", "args": {}, "id": cid, "type": "tool_call"})
        malformed.add(cid)
    return calls, malformed


# ── the call ──────────────────────────────────────────────────────────────────────────────────


# The custom-stream event that tells the UI a thought began or ended (app/turn.run_turn →
# on_thinking → the status bar's `thinking 3s · esc stops`).
def _emit_thinking(phase: str) -> None:
    try:
        get_stream_writer()({"type": "thinking", "phase": phase})
    except Exception:  # outside a graph run (a test calling the node directly)
        pass


# Where `_generate` reports the pass's thought on the message it returns; `_take_thought` pops
# it again, so it never rides the conversation's AIMessage.
THOUGHT_KEY = "saturn_thought"


def _generate(llm_input: list, *, tools: bool, think: bool = False) -> AIMessage:
    """ONE streamed call (the test seam). Tokens reach the UI through LangGraph's messages
    mode (app/turn.py filters this node); the chunks are folded into one AIMessage here so
    state/trace/autosave see exactly what the model produced. `tools=False` is the cap's hard
    stop: no bind, so the model can only answer. `think=True` is a thinking pass: the
    reasoning rides the chunks' `reasoning_content`, never `content`, so the response stream
    stays the answer.

    The thought is BOUNDED here, while it streams: past `runtime.think_budget` reasoning
    tokens (one per chunk from Ollama — close enough for a circuit breaker), or when the user
    presses Esc (a pause is pending on core.pause), the stream is closed and the message comes
    back with nothing but the report. The pause itself is left pending: the node handles it at
    the next pass boundary, as ever. The report — `response_metadata[THOUGHT_KEY]` =
    {seconds, tokens, text, cut: None | "budget" | "esc"} — is present when the pass
    reasoned (langchain-ollama surfaces `reasoning_content` only on a call sent with the
    think flag on)."""
    from tools.registry import tool as registered

    model = get_model()
    runnable = model.bind_tools(list(registered)) if tools else model
    kwargs = invoke_kwargs(None, 0.0, task="agent", think=think)
    budget = _think.budget() if think else 0
    controller = get_pause_controller()
    thought = {"seconds": 0.0, "tokens": 0, "text": "", "cut": None}
    began = None    # when the first reasoning chunk arrived
    over = False    # the thought ended: the answer or a call began
    full = None
    chunks = llm_stream(runnable, llm_input, tag=model_tag(), **kwargs)
    try:
        for chunk in chunks:
            full = chunk if full is None else full + chunk
            reasoning = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
            answering = bool(chunk.content) or bool(getattr(chunk, "tool_call_chunks", None))
            if reasoning and not answering and not over:
                if began is None:
                    began = time.perf_counter()
                    _emit_thinking("start")
                thought["tokens"] += 1
                if think and budget and thought["tokens"] >= budget:
                    thought["cut"] = "budget"
                elif think and controller.pending():
                    thought["cut"] = "esc"
                if thought["cut"]:
                    break
            elif began is not None and not over:
                over = True
                thought["seconds"] = time.perf_counter() - began
                _emit_thinking("end")
    finally:
        if began is not None and not over:
            thought["seconds"] = time.perf_counter() - began
            _emit_thinking("end")
        close = getattr(chunks, "close", None)
        if close:
            close()  # a cut thought must stop generating, not run on unread
    if full is None:  # a model that streamed nothing — blocking fallback
        full = generate(runnable, llm_input, tag=model_tag(), **kwargs)
    content = full.content if isinstance(full.content, str) else str(full.content)
    calls = []
    for tc in getattr(full, "tool_calls", None) or []:
        calls.append({"name": tc.get("name"), "args": tc.get("args"),
                      "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                      "type": "tool_call"})
    kw = {"response_metadata": dict(getattr(full, "response_metadata", None) or {})}
    if began is not None:
        thought["text"] = str((getattr(full, "additional_kwargs", None) or {})
                              .get("reasoning_content") or "")
        kw["response_metadata"][THOUGHT_KEY] = thought
    if getattr(full, "usage_metadata", None):
        kw["usage_metadata"] = full.usage_metadata
    if thought["cut"]:  # nothing but the report: the pass is rerun without thinking
        return AIMessage(content="", **kw)
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


# The custom-stream event that tells the UI to drop what this pass streamed so far
# (app/turn.run_turn → on_retract): a malformed attempt's tokens are not the answer.
RETRACT = {"type": "retract"}


def _retract_stream() -> None:
    try:
        get_stream_writer()(RETRACT)
    except Exception:  # outside a graph run (a test calling the node directly): nothing streamed
        pass


def _generate_or_retry(llm_input: list, *, tools: bool, think: bool = False) -> "AIMessage | None":
    """`_generate`, retried ONCE with the malformed-call note when the model's output could not
    be parsed; None when the retry failed the same way (the caller answers honestly)."""
    try:
        return _generate(llm_input, tools=tools, think=think)
    except Exception as exc:
        if not _is_parse_failure(exc):
            raise
        diag.log(f"agent_node : malformed model output ({type(exc).__name__}: {exc}) — retrying once")
    _retract_stream()  # the failed attempt may have streamed text; the retry streams its own
    try:
        return _generate(llm_input + [HumanMessage(content=MALFORMED_NOTE)], tools=tools, think=think)
    except Exception as exc:
        if not _is_parse_failure(exc):
            raise
        diag.log(f"agent_node : malformed model output twice ({type(exc).__name__}) — answering honestly")
        return None


# ── thinking: one pass, at most two calls ─────────────────────────────────────────────────────


def _take_thought(ai) -> dict:
    """Pop the thought report off a generated message (it must not ride the conversation's
    AIMessage); {} when the pass did not reason."""
    meta = getattr(ai, "response_metadata", None)
    return (meta.pop(THOUGHT_KEY, None) or {}) if isinstance(meta, dict) else {}


def _thought_outcome(ai, thought: dict) -> str:
    """What came of a thinking call (core.think.OUTCOMES): a model the daemon turned out not
    to let think (`unsupported` — the call ran think-off, so it stands), a thought cut at the
    budget or by Esc, a thinking call whose output was malformed twice, one that returned
    neither text nor a call (`empty` — on a pass whose right move is a short answer, qwen3.5
    writes the answer inside its reasoning; seen on the 4b and 9b), or a thought the pass
    used."""
    if not _think.supported():
        return "unsupported"
    if thought.get("cut"):
        return f"cut-{thought['cut']}"
    if ai is None:
        return "malformed"  # its output could not be parsed, twice (_generate_or_retry)
    if _is_empty(ai):
        return "empty"
    return "thought"


def _run_pass(llm_input: list, decision) -> "tuple[AIMessage | None, dict]":
    """The pass's model call(s) under `decision` (core.think.Decision), and what the think
    record needs: {outcome, draft, thought}. THE bound: at most two calls because of thinking.

      no thought    one think-off call.
      think         the thinking call; empty, cut or stopped → rerun once think-off.
      draft         the think-off call first. A text answer stands (a chat turn pays nothing).
                    A tool call is retracted and the pass rethought; if that thought is
                    empty, cut or stopped, or the daemon fails the call, the DRAFT stands —
                    never a third call. A daemon that cannot be reached at all fails the
                    turn instead (the draft must not act in a turn that cannot go on), and
                    so does an error that is not the daemon's.

    None means the model's output was malformed twice (the caller answers honestly)."""
    if decision.draft:
        draft = _generate_or_retry(llm_input, tools=True, think=False)
        if draft is None or not _calls_of(draft)[0]:
            return draft, {"outcome": "none", "thought": _take_thought(draft)}
        _take_thought(draft)
        _retract_stream()  # the draft's preamble is not the answer
        try:
            ai = _generate_or_retry(llm_input, tools=True, think=True)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            # A daemon error or a timeout on the rethink is a thought that came to nothing:
            # the draft is a valid call already in hand, so the turn does not fail on the
            # optional second call. Not retried — the bound is two calls — and not asked
            # again this turn (core.think.gave_up). Anything else propagates: a bug here is
            # not "the model was busy", and with nothing answering at the address the draft
            # would act — a send, a write — in a turn whose next pass cannot report it.
            if call_failure(exc) != "daemon":
                raise
            diag.log(f"agent_node : the rethink failed ({type(exc).__name__}: {exc}) — "
                     "the drafted call stands, and this turn does not think again")
            return draft, {"outcome": "failed", "thought": {}}
        thought = _take_thought(ai)
        outcome = _thought_outcome(ai, thought)
        if ai is not None and outcome in ("thought", "unsupported"):
            return ai, {"outcome": outcome, "draft": outcome == "thought", "thought": thought}
        diag.log(f"agent_node : the rethink came back {outcome} — the drafted call stands")
        return draft, {"outcome": outcome, "thought": thought}
    ai = _generate_or_retry(llm_input, tools=True, think=decision.think)
    thought = _take_thought(ai)
    if not decision.think:
        return ai, {"outcome": "none", "thought": thought}
    outcome = _thought_outcome(ai, thought)
    if ai is None or outcome in ("thought", "unsupported"):
        return ai, {"outcome": outcome, "thought": thought}
    # The same pass think-off answers; its prefix is already cached, so the rerun is cheap.
    diag.log(f"agent_node : thinking pass came back {outcome} — rerunning think-off")
    ai = _generate_or_retry(llm_input, tools=True, think=False)
    _take_thought(ai)
    return ai, {"outcome": outcome, "thought": thought}


def _is_empty(ai: AIMessage) -> bool:
    """A pass that produced neither answer text nor a call (valid or malformed)."""
    return (not str(ai.content or "").strip() and not getattr(ai, "tool_calls", None)
            and not getattr(ai, "invalid_tool_calls", None))


# ── hygiene ───────────────────────────────────────────────────────────────────────────────────


def _repeats_since_change(key: str, rounds: list) -> int:
    """How many times this exact call has been made since something last changed. A completed
    call to a tool that is not read_only (an edit, a write, a command) resets the count: the
    same `pytest -q` after an edit is a new question, not a repeat."""
    from tools.registry import DECLARED_RISK

    n = 0
    for k, name, _args, status, _obs in rounds:
        if k == key:
            n += 1
        elif status == "done" and DECLARED_RISK.get(name, "destructive") != "read_only":
            n = 0
    return n


def _skill_hygiene(args: dict) -> "tuple[dict, tuple[str, str] | None]":
    """create_skill before any gate: the corrected arguments, and the (text, status) that
    answers the call instead of running it. create_skill always faces the human
    (policy.ALWAYS_ASKS), so a draft that cannot be saved is refused HERE — the person never
    reads a prompt for a save that would fail. The slips a small model makes are corrected
    first (a list of steps, `/Weekly Review` for a name, "true" for a flag), so the gate shows
    the call that will run. A draft over a skill that exists, without `replace`, is answered
    with that skill's current text and stamped `done`: the model had no other way to read it,
    and a read is not an incident."""
    from commands._framework import resolves
    from core import skills

    replace = args.get("replace")
    args = {**args,
            "name": skills.draft_name(args.get("name")),
            "steps": skills.steps_text(args.get("steps")),
            "replace": replace is True or str(replace).strip().lower() == "true"}
    problem = skills.draft_problem(args["name"], args.get("description"), args["steps"],
                                   replace=True, builtin=resolves)
    if problem:
        return args, ("Error: " + problem, "error")
    current = None if args["replace"] else skills.in_scope("global").get(args["name"])
    if current is not None:
        return args, (skills.existing_text(current), "done")
    return args, None


def _hygiene(call: dict, rounds: list, malformed: bool = False,
             provenance: "tuple | None" = None) -> "tuple[dict, ToolMessage | None]":
    """The corrected call, or the ToolMessage that answers it instead of running it.
    `provenance` is (what the user typed, what else entered the conversation) for the handle
    check — a recipient the model composed is refused here, before any gate sees it."""
    from tools.messages import route_target
    from tools.registry import is_off, tools_by_name
    from tools.toolspec import toolkit_of

    name = str(call.get("name") or "")

    def refuse(text, status="error"):
        return call, ToolMessage(content=text, tool_call_id=call["id"], name=name,
                                 additional_kwargs={"saturn_status": status})

    # Before everything else: a tool whose toolkit is off is not "unknown", and nothing about
    # its arguments matters.
    if is_off(name):
        return refuse(TOOLKIT_OFF_TEXT.format(name=name, key=toolkit_of(name)))
    if malformed:
        # Before the unknown-tool check: broken JSON can leave the name unreadable too, and
        # "unknown tool ''" is a corrective the model cannot act on.
        if name not in tools_by_name:
            return refuse(MALFORMED_CALL_TEXT)
        return refuse("Error: " + schema_hint(name, "the arguments were not valid JSON"))
    if name not in tools_by_name:
        return refuse(UNKNOWN_TOOL_TEXT.format(name=name))
    raw = call.get("args")
    other = tool_for_args(name, raw)
    if other and not is_off(other):  # never point the model at a tool that is turned off
        return refuse("Error: " + schema_hint(
            other, f"those arguments belong to {other}, not {name}; the call was not run"))
    args = coerce_args(name, raw)
    if args is None:
        problem = ("the arguments were not an object" if not isinstance(raw, dict)
                   else f"required arguments missing from {raw!r}")
        return refuse("Error: " + schema_hint(name, problem))
    # One person or one group chat, never both — and a recipient in the wrong slot moved to
    # the right one when its form says which it is (tools.messages.route_target).
    args, problem = route_target(name, args)
    if problem:
        return refuse("Error: " + problem)
    if name == "create_skill":
        args, answer = _skill_hygiene(args)
        if answer is not None:
            return refuse(*answer)
    handle_arg = quarantine.HANDLE_ARGS.get(name)
    if handle_arg and provenance is not None:
        handle = str(args.get(handle_arg) or "").strip()
        if quarantine.handle_hold(handle, *provenance[:2]):
            return refuse(UNKNOWN_HANDLE_TEXT.format(handle=handle))
    chat_arg = quarantine.CHAT_ARGS.get(name)
    if chat_arg and provenance is not None:
        ref = str(args.get(chat_arg) or "").strip()
        if quarantine.chat_hold(ref, *provenance[:2]):
            return refuse(UNKNOWN_CHAT_TEXT.format(ref=ref))
    key = _call_key(name, args)
    if any(r[3] == "skipped" for r in rounds if r[0] == key):
        return refuse(ALREADY_DECLINED_TEXT, "skipped")
    if _repeats_since_change(key, rounds) >= STALL_REPEATS:
        return refuse(STALL_TEXT)
    return {**call, "args": args}, None


# ── the answer trailers ───────────────────────────────────────────────────────────────────────


def sources_footer(tool_results, documents_retrieved) -> str:
    """The receipt of what informed the answer: one line per tool call / document, in the order
    they were gathered (core.sources — the same numbering /trace source uses).
    '' when nothing was."""
    sources = build_sources(tool_results, documents_retrieved)
    if not sources:
        return ""
    return SOURCES_HEADER + "\n" + "\n".join(f"  [{n}] {label}" for n, label, _text in sources)


_INCIDENT_WORDING = {
    "skipped": "declined at the approval gate — not done",
    "blocked": "blocked by the air-gap — nothing was sent",
}
_BUDGET_WORDING = "not run: the turn's action budget was spent"
_TOOLKIT_OFF_WORDING = "not run: the {key} toolkit is turned off (/tools on {key})"


def _off_toolkit(name: str, obs: str) -> "str | None":
    """The toolkit when `obs` is hygiene's answer to a call into one that is off, else None."""
    from tools.toolspec import toolkit_of

    key = toolkit_of(name)
    return key if key and obs == TOOLKIT_OFF_TEXT.format(name=name, key=key) else None


def incidents(this_turn: list) -> list:
    """One line per tool CALL that did NOT complete — declined at the gate (skipped), refused
    by the air-gap (blocked), or failed (error) — read off the ToolMessages' structural stamp
    and worded for the user (the observations are written for the model). One line per
    distinct call: a declined call the model re-issued is one incident, not two. A call's LAST
    outcome decides — one that failed and then ran when re-issued is not an incident. The stall
    guard's and the cap's refusals are not outcomes: a stalled call already ran twice, and those
    runs are what happened to it; a call refused at the cap is reported as not run, unless the
    same call did run earlier in the turn. A call into a toolkit that is off is worded as
    that, with the switch."""
    rounds = [r for r in _rounds(this_turn) if r[4] != STALL_TEXT]
    last = {key: status for key, _n, _a, status, obs in rounds if obs != BUDGET_TEXT}
    out = []
    seen: set = set()
    for key, name, args, status, obs in rounds:
        if key in seen:
            continue
        if obs == BUDGET_TEXT:
            if key in last:
                continue
            why = _BUDGET_WORDING
        elif status not in _INCIDENT_STATUSES or last[key] not in _INCIDENT_STATUSES:
            continue
        elif _off_toolkit(name, obs):
            why = _TOOLKIT_OFF_WORDING.format(key=_off_toolkit(name, obs))
        else:
            why = _INCIDENT_WORDING.get(status) or f"failed: {clip(' '.join(obs.split()), _INCIDENT_CAP)}"
        seen.add(key)
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

    # 1. steer and 2. pause. The queued steers are drained only PAST any pause interrupt: a
    # resumed interrupt re-runs this node from the top, so steers taken before it would be lost
    # to an Esc landing in between (see core/pause.py). The pause is read non-destructively and
    # cleared only past the interrupt.
    pause_steer = None
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
            updates["messages"] = [
                AIMessage(content=_with_trailers(ABORT_TEXT, state, _this_turn(messages)))
            ]
            diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (aborted at the pause prompt)")
            return updates
        if action == "steer" and str(decision.get("text") or "").strip():
            pause_steer = str(decision["text"])
    for r in controller.take_steers():
        if r is not None and str(r.reason).strip():
            new.append(_steer_message(str(r.reason)))
    if pause_steer is not None:  # typed at the pause prompt: after anything queued before it
        new.append(_steer_message(pause_steer))

    # 3. the cap — from pass max_iterations on, no tool call runs. The pass itself is unchanged
    # (tools stay bound, nothing is appended), so its prompt extends the cached prefix; a call
    # it emits is refused in step 5 and the refusal routes back here for the answer.
    cap = get_config().max_iterations
    capped = iteration >= cap

    # 4. generate — a malformed model output is retried once, then answered honestly; any
    # other failure propagates (the REPL reports "Turn failed").
    this_turn = _this_turn(messages + new)
    kind = _think.step_kind(this_turn, capped, mechanical=(ASK_ALONE_TEXT,))
    decision = _think.decide(_think.level(state), kind, _think.policy(), _think.supported())
    if _think.gave_up(state.get("think")):  # a thinking call failed this turn: not asked again
        decision = _think.Decision(False, False, decision.why)
    llm_input = _llm_input(state, messages + new)
    ai, thinking = _run_pass(llm_input, decision)
    if ai is not None and iteration > cap and _calls_of(ai)[0]:
        # The hard stop: the model was told its calls will not run and called again. Take the
        # tools away — the one pass that pays a full re-prefill (the system section changes),
        # which is why it is the last resort and not the cap itself. The budget note rides the
        # prompt only (never state: a HumanMessage there would read as a new turn boundary).
        diag.log("agent_node : calls again past the budget refusal — rerunning with tools unbound")
        _retract_stream()
        ai = _generate_or_retry(_llm_input(state, messages + new, [HumanMessage(content=BUDGET_NOTE)]),
                                tools=False, think=False)
        _take_thought(ai)
    # One think record per pass (core.think.entry): what kind of step it was, whether it
    # thought and what came of it — the rail, /think, /trace why and the benchmark read it.
    meta = (getattr(ai, "response_metadata", None) or {}) if ai is not None else {}
    updates["think"] = [_think.entry(
        n=iteration, kind=kind, decision=decision, outcome=thinking["outcome"],
        draft=thinking.get("draft", False), thought=thinking.get("thought"),
        prompt_s=(meta.get("prompt_eval_duration") or 0) / 1e9)]
    thought_note = f", thought: {thinking['outcome']}" if thinking["outcome"] != "none" else ""
    if ai is None:
        updates["messages"] = new + [AIMessage(content=_with_trailers(MALFORMED_TEXT, state, this_turn))]
        diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (malformed output twice)")
        return updates
    stats = SimpleNamespace(response_metadata=getattr(ai, "response_metadata", None) or {},
                            usage_metadata=getattr(ai, "usage_metadata", None))
    updates["tok_per_sec"] = extract_tok_per_sec(stats)
    updates["context_tokens"] = extract_prompt_tokens(stats)

    calls, malformed = _calls_of(ai)
    if not calls or iteration > cap:
        # 6. the answer
        final = AIMessage(content=_with_trailers(ai.content, state, this_turn),
                          response_metadata=ai.response_metadata,
                          usage_metadata=ai.usage_metadata)
        updates["messages"] = new + [final]
        diag.log(f"agent_node : {time.perf_counter() - start:.4f}s (answer, iter {iteration}"
                 f"{thought_note})")
        return updates

    # 5. hygiene
    rounds = _rounds(this_turn)
    prov = provenance({**state, "messages": messages + new})
    kept, replies = [], []
    for call in calls:
        if capped:  # the budget is spent: nothing runs, whatever the call is
            fixed, reply = call, ToolMessage(content=BUDGET_TEXT, tool_call_id=call["id"],
                                             name=str(call.get("name") or ""),
                                             additional_kwargs={"saturn_status": "error"})
        else:
            fixed, reply = _hygiene(call, rounds, malformed=call["id"] in malformed, provenance=prov)
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
             f"{thought_note})")
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
