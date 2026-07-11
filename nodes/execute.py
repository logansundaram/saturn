"""
Execute node — one plan step per pass (the 2026-07-03 engine transplant; replaces the ReAct
`agent` node).

The plan is the data bus: each pass picks the first step whose `result` is None and executes
exactly it, with a CURATED context (plan_context.exec_context — request + earlier step results +
"the previous step" callout) instead of the raw message history.

Three step shapes:
  - a pure reasoning step (`intended_tool` None — the planner said no tool is needed): one text
    call; the result lands on the step directly (and rides `messages` as a plain AIMessage for
    the trace/history). A step naming an UNKNOWN tool is NOT a reasoning step — it records an
    error incident (fail-closed, 2026-07-10) so rectify can redraft it instead of the model
    answering it from its own priors.
  - a write/edit step: the SEMANTIC write gate runs first (`_write_gate`, the fabricated-value
    guard — an LLM check that the value being persisted actually exists in the request or the
    gathered results). A blocked write marks the step `skipped` without generating a call. The
    human approval gate still fronts the actual filesystem action downstream.
  - a tool step: ONE constrained tool call is generated against exactly that tool
    (`_generate_tool_call`: single-tool bind, alias arg coercion, text-format call recovery,
    temperature-escalating retries with a schema hint) and emitted as a tool-calling AIMessage —
    the approval gate and tool node then handle it exactly as before, so the trust envelope
    (policy gate, quarantine escalation, egress attribution, tool_events) is unchanged.

`route_after_execute` sends a generated call to `approval`; everything else (reasoning result,
write-gate skip, argument failure, no step left) falls through to `rectify`.
"""

import time
import uuid

import diag
from langchain.messages import AIMessage, HumanMessage

from config import get_config
from core.llms import get_model, extract_tok_per_sec, extract_prompt_tokens
from core.messages import EXECUTE_TOOL_SYS, EXECUTE_REASONING_SYS, WRITE_GATE_SYS
from core.plan_context import (
    SEARCH_TOOLS,
    WRITE_TOOLS,
    clean,
    exec_context,
    original_request,
    results_block,
    steps_before,
)
from core.state import AgentState
from core.structured import (
    WriteGate,
    WRITE_GATE_FORMAT,
    WRITE_GATE_SHAPE,
    _invoke_kwargs,
    structured,
)
from core.tool_args import coerce_args, parse_text_call, schema_hint

# The steps the semantic write gate fronts (WRITE_TOOLS), and the gathering tools whose
# presence arms it (SEARCH_TOOLS) — both from core/plan_context, THE one home for the engine's
# tool classifications (rectify and synthesize key off the same sets).

# Temperature escalation for the constrained tool-call generation: deterministic first, then
# sampled variety — a failed parse at 0.0 usually reproduces byte-identically.
_ATTEMPT_TEMPS = (0.0, 0.5, 0.7)

# NOTE: no numeric zeros here — "0"/"0.0" from an upstream calculate is a COMPUTED VALUE, not a
# missing one (write "the count" when the count is 0 is a legitimate write, not a fabrication).
_EMPTY_MARKERS = {"", "[]", "()", "{}", "none"}

# The evidence sentinel the write gate's fail-closed default carries, so the skip message can
# distinguish "the judge examined the results and the value is absent" from "the judge was
# unavailable so we fail closed" — both block the write, but the disclosure should be honest.
_GATE_UNAVAILABLE = "gate-unavailable (fail-closed)"


def _is_empty_result(res) -> bool:
    if res is None:
        return True
    return str(res).strip().lower() in _EMPTY_MARKERS


def _write_gate(state: AgentState, step: dict) -> "str | None":
    """The semantic write gate: None = proceed, else the skip text recorded as the step's result.

    Guards ONE hazard — writing an item a SEARCH was meant to find (or a value bridged over a
    failed/empty step). With nothing gathered yet the payload can only come from the request
    itself, and a purely-mechanical plan (read files the user named, compute from them) has no
    presence question to judge — gating those over-blocks legitimate request-literal writes.
    The gate judges the RAW gathered results; a value appearing only in a step description is
    not evidence (steps are drafted by a planner and can carry a substituted value)."""
    plan = state.get("plan") or []
    # Positional prior-work only (steps_before): a LATER step the user retired at plan review
    # carries a result too, and must neither arm the gate nor pose as the "latest" upstream.
    done = [s for s in steps_before(plan, step) if s.get("result") is not None]
    if not done:
        return None
    # A search ARMS the gate only if it actually ran (status done): a retired/declined search
    # step gathered nothing, so there is no search evidence for a value to be bridging from.
    searched = any(
        s.get("intended_tool") in SEARCH_TOOLS and s.get("status") == "done" for s in done
    )
    # Failure is the STRUCTURAL stamp only (status == "error", gotcha #6) — never sniffed from
    # observation text: a successful read of a log that begins "ERROR:" is a done step with an
    # error-looking result, and text-sniffing it armed the gate on purely mechanical plans (the
    # exact false positive the saturn_status contract removed from update_plan, 2026-07-04).
    failed = any(s.get("status") == "error" for s in done)
    if not (searched or failed):
        # A purely mechanical plan (read files the user named, compute from them) never pays
        # for the gate — including its empty-looking results: a computed 0 or an empty diff is
        # a real value, not a missing one. Arming requires a search or a failure upstream.
        return None
    # The mechanical empty-check keys on the last step that PRODUCED a result (status done) —
    # an incident step's stamp text (a decline, a review retirement) is not the upstream value.
    producers = [s for s in done if s.get("status") == "done"]
    if producers and _is_empty_result(producers[-1].get("result")):
        return (
            "skipped write: the upstream result was empty, so nothing was written "
            "(a file must not be created from a missing value)."
        )
    ctx = (
        f"Request: {original_request(state)}\n\n{results_block(done)}\n\n"
        f"The write step: {step.get('label')}\n\n"
        "Is the specific value this step writes available per the rule above?"
    )
    gate = structured(
        "judge",
        [WRITE_GATE_SYS, HumanMessage(content=ctx)],
        WriteGate,
        WRITE_GATE_FORMAT,
        WRITE_GATE_SHAPE,
        # Fail-CLOSED: when the judge is unavailable (every attempt errored, or nothing parsed)
        # structured() returns this default. The gate is armed precisely because a value could be
        # bridging in unverified, so an unverifiable verdict must BLOCK the write — not wave it
        # through. The human approval gate still fronts the filesystem action, but a value the
        # gate could not vouch for should never reach it.
        default=WriteGate(present=False, evidence=_GATE_UNAVAILABLE),
    )
    if not gate.present:
        if gate.evidence == _GATE_UNAVAILABLE:
            return (
                "skipped write: the write gate could not verify the value to write "
                "(the judge was unavailable), so nothing was written — fail-closed."
            )
        return (
            "skipped write: the value to write is not present in the gathered "
            "results, so nothing was written (a file must not be created with a "
            "missing/fabricated value)."
        )
    return None


def _metrics(resp) -> dict:
    if resp is None:
        return {}
    out = {}
    tps = extract_tok_per_sec(resp)
    if tps:
        out["tok_per_sec"] = tps
    used = extract_prompt_tokens(resp)
    if used:
        out["context_tokens"] = used
    return out


def _reasoning_call(context: str):
    """One text generation for a pure reasoning step. Returns (content, last_response)."""
    model = get_model("tool_caller")
    resp = None
    for i, temp in enumerate((0.0, 0.4)):
        try:
            resp = model.invoke(
                [EXECUTE_REASONING_SYS, HumanMessage(content=context)],
                **_invoke_kwargs("tool_caller", None, temp),
            )
        except Exception as exc:
            diag.log(f"execute_node : reasoning attempt {i + 1} failed ({exc})")
            continue
        content = str(getattr(resp, "content", "") or "").strip()
        if content:
            return content, resp
    return "", resp


def _generate_tool_call(tool, context: str):
    """Generate ONE call against exactly `tool` (bound alone, so the model can't wander to a
    different tool than the step planned). Recovers text-format calls, coerces alias args onto
    the real schema, and retries with a schema hint at escalating temperature.

    Returns (args, failure_text, last_response): `args` set on success; otherwise
    `failure_text` is what lands on the step (the model's plain-text fallback answer, or an
    error line)."""
    model = get_model("tool_caller")
    try:
        bound = model.bind_tools([tool])
    except Exception as exc:
        return None, f"error: cannot bind tool {tool.name}: {exc}", None
    block = context
    text_fallback = ""
    problem = "no tool call emitted"
    resp = None
    for temp in _ATTEMPT_TEMPS:
        try:
            resp = bound.invoke(
                [EXECUTE_TOOL_SYS, HumanMessage(content=block)],
                **_invoke_kwargs("tool_caller", None, temp),
            )
        except Exception as exc:
            # A transient provider error (an Ollama timeout) must not spend the whole step —
            # keep escalating through the remaining attempts like _reasoning_call does.
            diag.log(f"execute_node : tool-call attempt at temp {temp} failed ({exc})")
            problem = f"{type(exc).__name__}: {exc}"
            continue
        content = getattr(resp, "content", "")
        content = content if isinstance(content, str) else str(content)
        calls = [{"args": tc.get("args")} for tc in (getattr(resp, "tool_calls", None) or [])]
        if not calls:
            parsed = parse_text_call(content)
            if parsed:
                calls = [{"args": parsed}]
        if calls:
            args = coerce_args(tool.name, calls[0].get("args"))
            if args is not None:
                return args, None, resp
            problem = f"arguments {calls[0].get('args')} do not fit the tool"
        else:
            text_fallback = content.strip() or text_fallback
            problem = "no tool call emitted"
        block = context + "\n\n" + schema_hint(tool.name, problem)
    if text_fallback:
        # The step's tool was never called — the prose is NOT a tool observation, and recording
        # it as a plain "done" result would feed unverified text into later steps' contexts as
        # ground data (and present e.g. a write step as completed when no file was touched).
        # The "error:" prefix makes the recorder mark the step an incident the answer discloses.
        return None, (
            "error: the step's tool was never called — the model answered in text instead: "
            + text_fallback
        ), resp
    return None, f"error: {problem}", resp


def execute_node(state: AgentState):
    """Execute the current step (the first with `result` None). See the module docstring."""
    start = time.perf_counter()
    state_plan = state.get("plan") or []
    idx = next((i for i, s in enumerate(state_plan) if s.get("result") is None), None)
    if idx is None:
        return {}  # nothing left — route_after_execute falls through to rectify -> synthesize

    context = exec_context(state, state_plan[idx])
    plan = [dict(s) for s in state_plan]  # never mutate state's plan in place
    step = plan[idx]
    step["status"] = "active"
    updates: dict = {"plan": plan, "iteration": state.get("iteration", 0) + 1}

    from tools.registry import tools_by_name

    tool_name = step.get("intended_tool")
    tool = tools_by_name.get(tool_name) if tool_name else None

    # A planned tool that doesn't exist fails CLOSED (2026-07-10 — the third fabrication path):
    # silently degrading to a reasoning step would answer the step from the model's own priors
    # and record the invented output as a done result. An error incident routes to rectify,
    # whose judge/replan can redraft the step with a real tool — or the answer discloses it.
    # (Planner output normally can't reach here — structured.to_steps preserves an unresolvable
    # tool spelling precisely so this guard sees it.)
    if tool is None and tool_name:
        step["result"] = (
            f"error: the plan named a tool that is not available: {tool_name!r} — "
            "the step was not executed"
        )
        step["status"] = "error"
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s "
                 f"(unknown tool {tool_name!r} — fail-closed error incident)")
        return updates

    # Pure reasoning step (the planner said no tool is needed).
    if tool is None:
        content, resp = _reasoning_call(context)
        step["result"] = content or "(no result produced)"
        step["status"] = "done" if content else "error"
        if content:
            updates["messages"] = [AIMessage(content=content)]
        updates.update(_metrics(resp))
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (reasoning step)")
        return updates

    # Write/edit steps face the semantic write gate BEFORE a call is generated: a write whose
    # value the gathered results don't actually contain is skipped, not laundered through.
    if tool_name in WRITE_TOOLS:
        blocked = _write_gate(state, state_plan[idx])
        if blocked is not None:
            step["result"] = blocked
            step["status"] = "skipped"
            diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (write gate skipped)")
            return updates

    args, failure, resp = _generate_tool_call(tool, context)
    if args is None:
        step["result"] = clean(failure or "error: no tool call emitted")
        # Structural stamp, unconditional: the step's tool was never called, so this is an
        # error incident by definition. (The old `startswith("error:")` sniff had a dead
        # else-"done" arm that would have presented a future non-prefixed failure as a
        # completed step — status is the producer's stamp, never derived from result text.)
        step["status"] = "error"
        updates.update(_metrics(resp))
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (no call: {failure!r:.80})")
        return updates

    # Emit the corrected call as a tool-calling AIMessage: the approval gate + tool node take it
    # from here exactly as before (policy, quarantine escalation, egress attribution, /undo
    # snapshots all unchanged). The model's own text rides along as the pre-action reasoning the
    # gate's `e(xplain)` answer shows.
    reasoning = getattr(resp, "content", "") if resp is not None else ""
    reasoning = reasoning if isinstance(reasoning, str) else str(reasoning)
    call = {
        "name": tool_name,
        "args": args,
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }
    updates["messages"] = [AIMessage(content=reasoning, tool_calls=[call])]
    updates.update(_metrics(resp))
    diag.log(f"execute_node : {time.perf_counter() - start:.4f}s -> {tool_name}")
    return updates


def route_after_execute(state: AgentState) -> str:
    """A generated tool call -> approval (then tools -> update_plan -> rectify); anything else
    (reasoning result recorded, write-gate skip, argument failure, empty plan) -> rectify."""
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "approval"
    return "rectify"
