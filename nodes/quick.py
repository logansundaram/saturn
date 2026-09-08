"""
Quick node — the simple path (2026-09-08; PLAN.md "The common-case contract",
docs/OPTIMIZATIONS.md §8).

Runs on the turn shape that dominates the traffic (41 of 46 traced plans single-step, 17 a lone
reasoning step): a chat question, or one lookup. Its cost there is ONE grammar-bound router call
(think off, ~0.5 s warm on the 9b) for a chat turn and two for a single read — against the plan
call's thinking (5–19 s), the execute pass and the rectify judge that every turn paid before.

The router decides the ONE next action (core/structured.QuickDecision under quick_format):
  - "answer"                → synthesize streams the answer from what is known;
  - a QUICK_TOOLS tool      → a pending step is appended to the plan (THE data bus, unchanged)
                              and the call is emitted as a tool-calling AIMessage, so approval,
                              tools, egress, quarantine and update_plan run exactly as for the
                              execute node; update_plan routes back here (state["route"]);
  - any other tool, or nothing parseable → the request needs the engine: the turn is HANDED
                              OVER with its observations (route "plan"; an empty plan drafts
                              fresh at plan_node, gathered steps go through replan).

Escalation, not a judge: a hard bound of QUICK_MAX_CALLS tool calls (the model is told when it
has spent them and may still answer), a tool outside the read-only set, or an error observation
hands the turn to the plan engine. A GUARDED outcome (a gate decline, an air-gap refusal) ends
the turn at synthesize — never retried, never substituted, disclosed by the incidents block,
the same rule as rectify's first branch.

Per the contract, what the quick path gives up, by design and only on the turns the check judged
simple: plan-level review (no plan_gate visit), the rectify judge, and the semantic write gate —
which has nothing to gate, since no writing tool is reachable from here.
"""

import json
import time
import uuid

import diag
from langchain.messages import AIMessage, HumanMessage

from config import get_config
from core.complexity import plan_reason
from core.messages import quick_sys_msg, quick_tool_names
from core.plan_context import grounding_parts, original_request
from core.state import AgentState, current_step
from core.structured import QUICK_SHAPE, QuickDecision, quick_format, structured
from core.tool_args import coerce_args
from textutil import fmt_args, head_tail

# The tool-call budget: past it the model may still answer, but not call (a further call hands
# the turn over). Three is the contract's number — a simple turn is one lookup, occasionally a
# search then a read.
QUICK_MAX_CALLS = 3

# What each observation is trimmed to for the ROUTER's eyes (the full clamped observation still
# rides the step result and tool_results into synthesize). A fixed cap per result, so the block
# only ever grows at its end as calls land (prefix-cache order, like plan_context's results).
_OBS_CAP = 3000

ANSWER = "answer"

# The step-label prefix every quick step carries — one producer, so the rail, /trace why, and
# the tests can tell a quick lookup from a planned step.
QUICK_LABEL = "quick lookup: "

# The hand-over reasons (state["reasoning"], replan's revision instruction on the gathered
# steps). Stable prefixes: the rail leaf and the tests key off them.
ESCALATE_TOOL = "the request needs a tool the quick path does not run"
ESCALATE_BUDGET = "the quick path spent its tool-call budget"
ESCALATE_ERROR = "a quick-path tool call failed"
ESCALATE_UNPARSED = "the quick path produced no decision"


def _observations(plan: list) -> list:
    """The gathered results as consecutive user messages, one per step that ran, in plan order."""
    out = []
    for s in plan:
        if s.get("result") is None:
            continue
        out.append(HumanMessage(content=(
            f"Result of {s.get('label')}:\n"
            + head_tail(str(s.get("result") or "").strip(), _OBS_CAP)
        )))
    return out


def _decide(state: AgentState, plan: list, spent: bool) -> QuickDecision:
    """The router call. Prompt-cache order: [system][stable grounding][request + per-turn
    grounding][observations…] — the first two are the primed lineage (core/prime.py), so a turn's
    first call prefills only the request, and each later call extends the previous one."""
    stable, dynamic = grounding_parts(state)
    msgs = [quick_sys_msg()]
    if stable:  # the primed boundary; an empty grounding (a bare test state) sends no message
        msgs.append(HumanMessage(content=stable))
    msgs.append(HumanMessage(content=(
        (dynamic + "\n\n" if dynamic else "") + "User request:\n" + original_request(state)
    )))
    msgs.extend(_observations(plan))
    if spent:
        msgs.append(HumanMessage(content=(
            "The tool-call budget is spent: answer from the results above, or name the tool "
            "the planner should continue with."
        )))
    from tools.registry import tool as registered

    return structured(
        "tool_caller",
        msgs,
        QuickDecision,
        quick_format([t.name for t in registered]),
        QUICK_SHAPE,
        default=QuickDecision(tool="", arguments={}),
    )


def _label(name: str, args: dict) -> str:
    return QUICK_LABEL + f"{name}({fmt_args(args, 60)})"


def quick_node(state: AgentState):
    start = time.perf_counter()
    plan = [dict(s) for s in state.get("plan") or []]  # never mutate state's plan in place
    updates: dict = {"route": "quick", "iteration": state.get("iteration", 0) + 1}

    last = plan[-1] if plan else None
    if last is not None:
        status = last.get("status")
        if status in ("skipped", "blocked"):
            # A guarded outcome ends the run (rectify's first branch, same rule): the incidents
            # block discloses it; nothing is retried or substituted.
            diag.log(f"quick_node : {time.perf_counter() - start:.4f}s (guarded: {status} — landing)")
            return updates
        if status == "error":
            updates.update({"route": "plan", "reasoning": f"{ESCALATE_ERROR}: {last.get('label')}"})
            diag.log(f"quick_node : {time.perf_counter() - start:.4f}s (error — handing over)")
            return updates

    spent = len(plan) >= QUICK_MAX_CALLS
    decision = _decide(state, plan, spent)
    name = str(decision.tool or "").strip()
    raw_args = decision.arguments if isinstance(decision.arguments, dict) else {}

    if name == ANSWER:
        diag.log(f"quick_node : {time.perf_counter() - start:.4f}s (answer)")
        return updates

    from tools.registry import tools_by_name

    allowed = quick_tool_names()
    if not name:
        reason = ESCALATE_UNPARSED
    elif name not in allowed or spent:
        reason = (ESCALATE_BUDGET if spent and name in allowed else ESCALATE_TOOL) + f": {name}"
    else:
        args = coerce_args(name, raw_args)
        if args is None or name not in tools_by_name:
            reason = f"{ESCALATE_TOOL}: {name} with arguments {raw_args!r} did not fit its schema"
        else:
            step = {
                "step_id": len(plan) + 1,
                "label": _label(name, args),
                "status": "active",
                "intended_tool": name,
                "result": None,
                "needs_resolution": False,
            }
            plan.append(step)
            call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call"}
            updates["plan"] = plan
            updates["messages"] = [AIMessage(
                content=f"Quick path: one read-only lookup — {name}.", tool_calls=[call]
            )]
            diag.log(f"quick_node : {time.perf_counter() - start:.4f}s -> {name}")
            return updates

    updates.update({"route": "plan", "reasoning": reason})
    diag.log(f"quick_node : {time.perf_counter() - start:.4f}s (handing over: {reason[:80]})")
    return updates


# --- routing ---------------------------------------------------------------------------------


def route_after_ground(state: AgentState) -> str:
    """quick or plan, for this turn: an explicit `route` (/quick, /plan <request>, --quick,
    --plan) wins; a seeded plan (/draft) is the engine's by definition; `runtime.quick_path`
    off means every turn plans; otherwise the request-side check decides."""
    forced = str(state.get("route") or "")
    if forced in ("quick", "plan"):
        diag.log(f"route_after_ground : {forced} (forced)")
        return forced
    if state.get("plan"):
        return "plan"
    if not get_config().get("runtime.quick_path", True):
        return "plan"
    reason = plan_reason(state.get("current_query", ""))
    if reason:
        diag.log(f"route_after_ground : plan ({reason})")
        return "plan"
    diag.log("route_after_ground : quick")
    return "quick"


def route_after_quick(state: AgentState) -> str:
    """A generated call -> approval; a hand-over -> plan (nothing gathered) or replan (the
    gathered steps are kept, the rest drafted); anything else -> synthesize."""
    if state.get("iteration", 0) >= get_config().max_iterations:
        return "synthesize"  # the engine's cap, honored here too — an honest landing
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) \
            and current_step(state.get("plan") or []) is not None:
        return "approval"
    if state.get("route") == "plan":
        return "replan" if any(s.get("result") is not None for s in state.get("plan") or []) \
            else "plan"
    return "synthesize"


def route_after_update_plan(state: AgentState) -> str:
    """A tool round returns to the engine that issued it: the quick path's rounds come back
    here for the next decision; the plan engine's reflect at rectify."""
    return "quick" if state.get("route") == "quick" else "rectify"
