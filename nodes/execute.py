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
    (`_generate_tool_call`: the tool's schema as the response grammar, alias arg coercion,
    text-format call recovery,
    temperature-escalating retries with a schema hint) and emitted as a tool-calling AIMessage —
    the approval gate and tool node then handle it exactly as before, so the trust envelope
    (policy gate, quarantine escalation, egress attribution, tool_events) is unchanged.

`route_after_execute` sends a generated call to `approval`; everything else (reasoning result,
write-gate skip, argument failure, no step left) falls through to `rectify`.
"""

import json
import time
import uuid

import diag
from langchain.messages import AIMessage, HumanMessage

from config import get_config
from core import serving
from core.llms import (
    get_model, generate, extract_tok_per_sec, extract_prompt_tokens, was_truncated,
)
from core.messages import EXECUTE_TOOL_SYS, EXECUTE_REASONING_SYS, WRITE_GATE_SYS
from core.plan_context import (
    SEARCH_TOOLS,
    WRITE_TOOLS,
    authorization_basis,
    clean,
    exec_parts,
    is_revoked,
    revocation_kind,
    original_request,
    request_authorized,
    results_block,
    steps_before,
)
from core.plan_ops import retirement_text
from core.state import AgentState
from textutil import looks_repetitive
from core.structured import (
    WriteGate,
    WRITE_GATE_FORMAT,
    WRITE_GATE_SHAPE,
    _invoke_kwargs,
    structured,
    _model_tag,
)
from core.tool_args import (
    coerce_args, concrete_args, launders_a_value, parse_text_call, schema_hint,
)
from core.structured import _extract_json
from langchain_core.utils.function_calling import convert_to_openai_tool

# The steps the semantic write gate fronts (WRITE_TOOLS), and the gathering tools whose
# presence arms it (SEARCH_TOOLS) — both from core/plan_context, THE one home for the engine's
# tool classifications (rectify and synthesize key off the same sets).

# Temperature escalation for the constrained tool-call generation: deterministic first, then
# sampled variety — a failed parse at 0.0 usually reproduces byte-identically.
_ATTEMPT_TEMPS = (0.0, 0.5, 0.7)

# The tools whose ARGUMENTS carry the whole payload (a file's content). Their tool-call
# generation runs under the serving layer's `tool_payload` bound instead of `tool_args`; every
# other tool's arguments are a path, a pattern, a query, an expression — small by construction.
PAYLOAD_TOOLS = frozenset(WRITE_TOOLS)


def _task_for_tool(tool_name: str) -> str:
    """The serving task that bounds this tool's call generation."""
    return "tool_payload" if tool_name in PAYLOAD_TOOLS else "tool_args"


# NOTE: no numeric zeros here — "0"/"0.0" from an upstream calculate is a COMPUTED VALUE, not a
# missing one (write "the count" when the count is 0 is a legitimate write, not a fabrication).
_EMPTY_MARKERS = {"", "[]", "()", "{}", "none"}

# The evidence sentinel the write gate's fail-closed default carries, so the skip message can
# distinguish "the judge examined the results and the value is absent" from "the judge was
# unavailable so we fail closed" — both block the write, but the disclosure should be honest.
_GATE_UNAVAILABLE = "gate-unavailable (fail-closed)"

# THE stable prefix every write-gate skip result starts with — the one producer, so the one
# reader that detects a gate skip (the trust benchmark's fabrication grader) keys off a constant
# instead of a hand-copied string. Change this and every skip return below together.
WRITE_GATE_SKIP_PREFIX = "skipped write:"

# The truncation refusal: the daemon hit the task's num_predict before the call closed. One
# producer (here); the synthesizer discloses any `error:` result, so no parser needs the text.
TRUNCATED_TEXT = "error: the tool call was cut off at the output limit"


# --- the ask gate (transplanted from the engine isolate, 2026-08-15) ---------------------------
#
# Three DETERMINISTIC rules an `ask_user` step faces before a call is generated — none asks
# whether a question was "necessary" (that is judgment, and the judge is the thing that failed):
#   1. a budget: the second `ask_user` of a turn does not run;
#   2. search-first: the REQUEST names a source the engine can look in, nothing has been searched
#      yet, and the plan's answer is to ask the user (read from the human's words only);
#   3. no dangling question: no step follows the ask, so its answer feeds nothing — a question
#      the plan cannot consume belongs in the ANSWER, not in an interrupt.
# A question the USER asked for in their own words is exempt from 2 and 3 (the interrupting-tool
# seam). Measured: dev.absence.kb_miss 3/5 -> 5/5.
#
# The refusal's STATUS is the mechanism's second half: every rule has something to redraft
# TOWARD, so all three stamp `error` and carry ASK_GATE_PREFIX for rectify's 4a redraft. Rules 1
# and 2 redraft to "finish from what is known" / "search the named source". Rule 3 redrafts to
# "keep the question, add the step that uses the answer by reference" — it took the guarded
# posture (`skipped`, run over) from 2026-08-16 to 2026-09-06, on the reading that a plan with
# no consumer had nothing to do with the answer. Measured wrong once tools existed for the
# consumer (runs 51-54): the 9b drafts "ask the time" for a calendar request WITHOUT the
# create_calendar_event step 3/3 replays, the skip ended the run with "I cannot", and the
# user's next message hit the same wall. The substitution risk the skip guarded against ("Send
# an email to Petra" came back writing email_to_petra.txt and claiming it sent) is carried by
# the redraft text instead (no tool for the action -> a single cannot-do step, never a
# substitute), and a second lone ask ends the run through rectify's no-call guard (1b), so the
# redraft is bounded to one. One producer (these two texts), one parser (rectify 4a).
# The plan-review revocation lock's refusal (from the engine isolate): stamped through
# core.plan_ops.retirement_text so the result ends with the review stamp — which is what tells
# rectify this is the user's SINGLE-STEP veto ("skip this one, continue the rest") rather than a
# guard rejection that ends the run.
# One reason per `plan_context.revocation_kind`, because the disclosure must not claim more than
# the state records: only a "target" hit is the step the user actually deleted. A "blanket" hit is
# COLLATERAL of a vague write they removed, and saying "the user removed this action" about a step
# they never touched puts a false statement about their intent into the run record.
_REVOKED_REASONS = {
    "target": "the user removed this action at the plan-review prompt",
    "tool": "the user removed this tool's only action at the plan-review prompt",
    "blanket": "the user removed a write that named no file at the plan-review prompt, so no file "
               "is written this turn",
}

# The effect-authorization refusal: a step redrafted after results existed acts on something the
# user's own words never named. `blocked` — a guarded outcome, so rectify cancels the rest.
UNAUTHORIZED_PREFIX = "blocked: unauthorized effect —"

ASK_GATE_PREFIX = "error: ask_user was not executed:"
# Rule 3's exact text: rectify 4a keys its consumer-step redraft on it (startswith), so the
# dangling case gets its own instruction rather than the search/budget one.
DANGLING_ASK_TEXT = (
    f"{ASK_GATE_PREFIX} no step follows this question, so nothing in the plan can use the "
    "answer — the plan needs the step that acts on the answer, or a plain statement of what "
    "cannot be done."
)
MAX_ASKS_PER_TURN = 1


def _ask_gate(state: AgentState, plan: list, step: dict) -> "tuple[str, str] | None":
    """The ask gate: None = proceed, else `(result_text, status)` for the refused step."""
    from core.request_intent import invites_a_question, names_searchable_source

    asks = sum(
        1
        for ev in state.get("tool_events") or []
        if isinstance(ev, dict) and ev.get("name") == "ask_user"
    )
    if asks >= MAX_ASKS_PER_TURN:
        return (
            f"{ASK_GATE_PREFIX} this turn has already put {asks} question(s) to the user, which "
            "is the limit — the remaining work must be done from what is already known.",
            "error",
        )
    request = authorization_basis(state)
    invited = invites_a_question(request)
    searched = any(
        s.get("intended_tool") in SEARCH_TOOLS and s.get("result") is not None for s in plan or []
    )
    # Rule 2 before rule 3 — load-bearing: when the request names a source, "search it" is a
    # concrete replacement the turn can act on, and a redraft beats ending the run.
    if names_searchable_source(request) and not searched and not invited:
        return (
            f"{ASK_GATE_PREFIX} the request names a source this engine can search for itself, "
            "and nothing has been searched yet — search it first and ask only if the answer is "
            "genuinely not there.",
            "error",
        )
    steps = list(plan or [])
    idx = next((i for i, s in enumerate(steps) if s is step), None)  # identity, not equality
    follows = steps[idx + 1:] if idx is not None else []
    if not follows and not invited:
        return (DANGLING_ASK_TEXT, "error")
    return None


# The stall detector (transplanted from the engine isolate). A second identical call this turn
# (once to inspect, once to verify a write) is ordinary; by the THIRD there is no reading under
# which the turn is making progress — the call is refused as a disclosed error incident instead of
# burning the iteration budget. Counted off `tool_events`, what ACTUALLY RAN, never what was
# planned or proposed: a model cannot notice its own loop; the engine can, for free.
STALL_REPEATS = 2

STALL_TEXT = (
    "error: this exact call ({name} with the same arguments) has already run {n} times this "
    "turn without changing anything — the step is looping, so it was not executed again"
)


def _args_key(args) -> str:
    """A stable, order-independent identity for a call's arguments."""
    if not isinstance(args, dict):
        return str(args)
    return repr(sorted((str(k), str(v)) for k, v in args.items()))


def _identical_call_count(state, name, args) -> int:
    """How many times this exact (tool, arguments) pair has already EXECUTED this turn."""
    target = (str(name), _args_key(args))
    return sum(
        1
        for ev in state.get("tool_events") or []
        if isinstance(ev, dict) and (str(ev.get("name")), _args_key(ev.get("args"))) == target
    )


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
    # And only a failed PRODUCER arms it: the hazard is a value bridged over a search/read that
    # never returned one. A prior WRITE that errored (a truncated call, a daemon timeout) left
    # no value to bridge from, and arming on it turned a transient write failure into a
    # permanent 'fabrication' skip on the redraft (2026-09-02).
    failed = any(
        s.get("status") == "error" and s.get("intended_tool") not in WRITE_TOOLS for s in done
    )
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
            f"{WRITE_GATE_SKIP_PREFIX} the upstream result was empty, so nothing was written "
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
                f"{WRITE_GATE_SKIP_PREFIX} the write gate could not verify the value to write "
                "(the judge was unavailable), so nothing was written — fail-closed."
            )
        return (
            f"{WRITE_GATE_SKIP_PREFIX} the value to write is not present in the gathered "
            "results, so nothing was written (a file must not be created with a "
            "missing/fabricated value)."
        )
    return None


# The pre-action "reasoning" a concrete-filled call carries (the gate's e(xplain) shows it):
# there was no model turn to quote, and the audit trail should say so rather than show blank.
CONCRETE_ARGS_NOTE = "arguments copied from the plan step itself (no model call)"


def _workspace_kind(rel_path: str) -> "str | None":
    """"file" / "dir" / None for a path INSIDE the workspace — the concrete-args predicate. Same
    resolution rule as tools/files._resolve (per call, is_relative_to), so a label can never fill
    a path the tool itself would refuse."""
    try:
        workspace = get_config().path("workspace").resolve()
        target = (workspace / rel_path).resolve()
        if not target.is_relative_to(workspace):
            return None
        if target.is_file():
            return "file"
        if target.is_dir():
            return "dir"
        return None
    except Exception:
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


def _context_messages(context) -> list:
    """The curated context as consecutive user messages: `exec_parts` (stable grounding, request
    + per-turn grounding, results, the per-step tail) or one string (an older caller, a test)."""
    parts = [context] if isinstance(context, str) else list(context)
    return [HumanMessage(content=str(p)) for p in parts]


def _reasoning_call(context):
    """One text generation for a pure reasoning step. Returns (content, last_response)."""
    model = get_model("tool_caller")
    resp = None
    repetition = False  # a degenerate draw arms the retry-only repeat penalty for the next rung
    for i, temp in enumerate((0.0, 0.4)):
        try:
            resp = generate(
                model,
                [EXECUTE_REASONING_SYS] + _context_messages(context),
                tag=_model_tag("tool_caller"),
                **_invoke_kwargs("tool_caller", None, temp, task="reasoning",
                                 repetition=repetition),
            )
        except Exception as exc:
            diag.log(f"execute_node : reasoning attempt {i + 1} failed ({exc})")
            continue
        content = str(getattr(resp, "content", "") or "").strip()
        if content and not looks_repetitive(content):
            return content, resp
        repetition = repetition or looks_repetitive(content)
    content = str(getattr(resp, "content", "") or "").strip() if resp is not None else ""
    return content, resp


# The tool-call generation runs under a JSON GRAMMAR for the tool's own argument schema instead
# of a native single-tool bind (2026-09-04). Ollama's chat template renders bound tools INTO the
# system message, so every step's prompt began with a different schema block and re-prefilled
# whole (measured: 8k tokens, 20 s per step on the 9b; a `format` grammar leaves the prompt
# untouched, so consecutive steps extend one cached lineage). The wrapper keeps the one escape
# the bind gave the model — answering in text instead of calling (a write whose value is
# unavailable) — as an explicit `refusal`, which lands as the text fallback exactly as before.
_TOOL_DESC_CAP = 1500  # every shipped tool rides whole (the longest, remember, is ~1.2k)


def tool_call_format(tool) -> dict:
    """The response grammar for one tool call: `{"arguments": <the tool's parameters>}` or
    `{"refusal": "<why the tool must not be called>"}`. The parameter schema is the tool's
    own (an MCP tool's remote schema included), so the grammar enforces its types."""
    try:
        params = convert_to_openai_tool(tool)["function"].get("parameters")
    except Exception:
        params = None
    if not isinstance(params, dict) or params.get("type") != "object":
        params = {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {"arguments": params, "refusal": {"type": "string"}},
    }


def tool_brief(tool) -> str:
    """The per-step tail that names the tool and its arguments (what the native bind used to put
    in the system prompt) plus the response shape."""
    desc = str(getattr(tool, "description", "") or "").strip()
    desc = desc.split("\n\n", 1)[0].replace("\n", " ").strip().rstrip(".")
    if len(desc) > _TOOL_DESC_CAP:
        desc = desc[:_TOOL_DESC_CAP] + "…"
    props = tool_call_format(tool)["properties"]["arguments"].get("properties") or {}
    shape = ", ".join(
        f"{name}: {spec.get('type', 'any') if isinstance(spec, dict) else 'any'}"
        for name, spec in props.items()
    ) or "(no arguments)"
    return (
        f"The tool for this step is `{tool.name}`" + (f" — {desc}" if desc else "") + ".\n"
        f"Its arguments: {shape}.\n"
        'Respond with ONLY this JSON: {"arguments": {<the arguments for this step>}} — or, if '
        "the tool must NOT be called for this step (the value it needs is unavailable, empty, "
        'or an error), {"refusal": "<why, in one sentence>"}.'
    )


def _parse_arguments(content: str):
    """(args_dict | None, refusal_text): the wrapper, a bare argument object (a model that
    skipped the wrapper), a text-format call (core/tool_args.parse_text_call), or nothing."""
    try:
        obj = json.loads(_extract_json(content))
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        if isinstance(obj.get("arguments"), dict):
            return obj["arguments"], ""  # {} is a call to a tool that takes no arguments
        if str(obj.get("refusal") or "").strip():
            return None, str(obj["refusal"]).strip()
        if obj and "arguments" not in obj and "refusal" not in obj:
            return obj, ""
        return None, ""
    parsed = parse_text_call(content)
    return (parsed, "") if parsed else (None, "")


def _generate_tool_call(tool, context):
    """Generate ONE call against exactly `tool` — its argument schema as the response grammar
    (`tool_call_format`), so the model can't wander to a different tool than the step planned.
    Recovers text-format calls, coerces alias args onto the real schema, and retries with a
    schema hint at escalating temperature.

    Returns (args, failure_text, last_response): `args` set on success; otherwise
    `failure_text` is what lands on the step (the model's refusal / plain-text fallback answer,
    or an error line)."""
    model = get_model("tool_caller")
    try:
        fmt = tool_call_format(tool)
        brief = tool_brief(tool)
    except Exception as exc:
        return None, f"error: cannot describe tool {tool.name}: {exc}", None
    base = _context_messages(context)
    tail = brief
    text_fallback = ""
    problem = "no tool call emitted"
    resp = None
    repetition = False  # armed by a degenerate text answer; never a global setting
    task = _task_for_tool(tool.name)
    for temp in _ATTEMPT_TEMPS:
        try:
            resp = generate(
                model,
                [EXECUTE_TOOL_SYS] + base + [HumanMessage(content=tail)],
                tag=_model_tag("tool_caller"),
                **_invoke_kwargs("tool_caller", fmt, temp, task=task,
                                 repetition=repetition),
            )
        except Exception as exc:
            # A transient provider error (an Ollama timeout) must not spend the whole step —
            # keep escalating through the remaining attempts like _reasoning_call does.
            diag.log(f"execute_node : tool-call attempt at temp {temp} failed ({exc})")
            problem = f"{type(exc).__name__}: {exc}"
            continue
        content = getattr(resp, "content", "")
        content = content if isinstance(content, str) else str(content)
        raw_args, refusal = _parse_arguments(content)
        if raw_args is None and not refusal and was_truncated(resp):
            # The call was cut mid-JSON at num_predict. No temperature fixes that — the
            # remaining rungs would reproduce the cut (measured: three identical 512-token
            # truncations per step). Refuse now with the limit named, so the answer can say
            # WHY and a redraft can split the write instead of retrying it whole.
            cap = serving.num_predict(task)
            diag.log(f"execute_node : {tool.name} call cut off at num_predict={cap}")
            return None, (
                f"{TRUNCATED_TEXT} ({cap} tokens) — the {tool.name} call's arguments are too "
                "long to generate in one call; write a shorter version, or write the first part "
                "with write_file and append the rest with edit_file"
            ), resp
        if raw_args is not None:
            args = coerce_args(tool.name, raw_args)
            if args is None:
                problem = f"arguments {raw_args} do not fit the tool"
            elif launders_a_value(tool.name, args):
                # A `calculate` whose expression is a bare literal computes nothing and mints
                # tool provenance for a number that was never gathered. Refused here, on the
                # ladder, so the retry gets a hint rather than the step an incident outright.
                problem = (
                    f"the expression {args.get('expression')!r} is a bare value, not a "
                    "calculation — write the actual arithmetic over the values from the "
                    "results above (for example '120 + 340 + 55'), never a number you have "
                    "already worked out"
                )
            else:
                return args, None, resp
        else:
            text_fallback = refusal or text_fallback
            problem = "no tool call emitted"
            repetition = repetition or looks_repetitive(content)
        tail = brief + "\n\n" + schema_hint(tool.name, problem)
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

    context = exec_parts(state, state_plan[idx])
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

    # The plan-review revocation lock, first pass: the step's own description already names a
    # target the user removed at review, so refuse before spending a generation on it.
    revoked = state.get("revoked_writes") or []
    kind = revocation_kind(revoked, tool_name, step.get("label"))
    if kind:
        step["result"] = retirement_text("skipped", _REVOKED_REASONS[kind])
        step["status"] = "skipped"
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s "
                 f"(revoked at review: label, {kind})")
        return updates

    # An `ask_user` step faces the ask gate BEFORE a call is generated: a question the engine
    # could have answered for itself, one past this turn's budget, or one nothing can consume.
    # The gate returns the status too — see `_ask_gate` for why that choice is the mechanism.
    if tool_name == "ask_user":
        refused = _ask_gate(state, state_plan, state_plan[idx])
        if refused is not None:
            step["result"], step["status"] = refused
            diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (ask gate: {step['status']})")
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

    # Concrete-step fill (core/tool_args.concrete_args): a step whose only argument is already
    # spelled in its label ("Read notes.md", current_time) gets its call COPIED, not generated —
    # one fewer model call per such step, and the label is the plan the human reviewed. Every
    # check below reads the arguments and runs regardless of who wrote them.
    args = concrete_args(tool_name, step.get("label"), _workspace_kind)
    if args is not None:
        failure, resp = None, None
        diag.log(f"execute_node : concrete args for {tool_name} copied from the step (no model call)")
    else:
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

    # The revocation lock, second pass — and THE guarantee. The label check above is only an
    # optimization: a redraft can drop the filename from its description and still generate a
    # call whose arguments target the revoked path. This reads the ARGUMENTS, i.e. the actual
    # effect, at the last point before it is emitted for approval and execution.
    arg_texts = [str(v) for v in (args or {}).values()]
    kind = revocation_kind(revoked, tool_name, step.get("label"), *arg_texts)
    if kind:
        step["result"] = retirement_text("skipped", _REVOKED_REASONS[kind])
        step["status"] = "skipped"
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s "
                 f"(revoked at review: args, {kind})")
        return updates

    # EFFECT AUTHORIZATION, on the arguments, at the same last-possible point: a step drafted by
    # replan after results existed may have been written BY those results (a file's contents
    # turned into `write_file breach_marker.txt`). replan pre-filters on the description; the
    # arguments are the effect. Results may inform HOW, never WHAT.
    if not request_authorized(state, step, *arg_texts):
        step["result"] = (
            f"{UNAUTHORIZED_PREFIX} this step was added after the results came back and it acts "
            "on something the request never named. Results may inform HOW the request is "
            "carried out, never WHAT is done to the workspace."
        )
        step["status"] = "blocked"
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (unauthorized effect)")
        return updates

    # The stall detector: the same call with the same arguments, already executed STALL_REPEATS
    # times this turn, is a loop — refused as an error incident (structural stamp) with no call
    # emitted; rectify then treats it like any other incident. Deterministic, no model in the loop.
    repeats = _identical_call_count(state, tool_name, args)
    if repeats >= STALL_REPEATS:
        step["result"] = STALL_TEXT.format(name=tool_name, n=repeats)
        step["status"] = "error"
        updates.update(_metrics(resp))
        diag.log(f"execute_node : {time.perf_counter() - start:.4f}s (stall: {tool_name} x{repeats})")
        return updates

    # Emit the corrected call as a tool-calling AIMessage: the approval gate + tool node take it
    # from here exactly as before (policy, quarantine escalation, egress attribution, /undo
    # snapshots all unchanged). The model's own text rides along as the pre-action reasoning the
    # gate's `e(xplain)` answer shows.
    reasoning = getattr(resp, "content", "") if resp is not None else CONCRETE_ARGS_NOTE
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
