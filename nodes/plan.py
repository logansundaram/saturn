import time

import diag
from langchain.messages import HumanMessage

from core.state import AgentState
from core.messages import planner_sys_msg
from core.structured import (
    _PlanOut,
    PLAN_SHAPE,
    plan_format,
    registered_tools,
    structured,
    to_steps,
)


# The result recorded when the planner produces nothing parseable. The "error:" prefix makes
# rectify see a failed step (and update_plan/synthesize treat it as an incident to disclose).
PLAN_PARSE_ERROR = (
    "error: could not form a plan for this request — the planner returned no valid steps. "
    "Any answer must be grounded in information actually gathered and must not invent files, "
    "data, or facts."
)


def _fallback_plan() -> list[dict]:
    """The planner emitted nothing parseable after the hardened layer's temp-escalating retries.

    The OLD fallback was a single tool-less "reasoning" step — but the execute node runs that by
    asking the model to answer from its own priors with NO grounding, so a planner failure
    silently became a confident, potentially fabricated answer with no signal that anything went
    wrong (a legitimate no-tool request already parses to its OWN `tool:"none"` step, so reaching
    this path always means the planner FAILED, never "no tool was needed").

    Instead, record an explicit PARSE_ERROR incident: rectify attempts a bounded replan (a
    transient parse failure often succeeds on a fresh draft), and if planning keeps failing the
    turn lands at an honest synthesize that DISCLOSES it could not form a plan — rather than
    presenting an ungrounded answer as authoritative. `result` is set (status `error`) so the
    step is a recorded incident, not the execution pointer."""
    return [
        {
            "step_id": 1,
            "label": "Plan the request",
            "status": "error",
            "intended_tool": None,
            "result": PLAN_PARSE_ERROR,
            "needs_resolution": False,
        }
    ]


def plan_node(state: AgentState):
    """Draft the plan: an ordered list of one-action steps, each naming the ONE tool it calls
    (or none for pure reasoning). The plan is the engine's data bus — each step's result is
    recorded on it as it executes — so plan quality directly drives execution.

    Structured output goes through the hardened path (core/structured.py: flat schema, shape
    hint, JSON salvage, temp-escalating retries); a total parse failure records an explicit
    parse-error incident (see _fallback_plan) rather than aborting the turn OR silently answering
    from the model's priors."""
    start = time.perf_counter()

    prompt = [
        planner_sys_msg(),  # built per call — the tool catalog tracks /mcp reload
        HumanMessage(
            content="Grounding context:\n"
            + state.get("context", "")
            + "\n\nUser request:\n"
            + state["current_query"]
        ),
    ]

    draft = structured(
        "planner",
        prompt,
        _PlanOut,
        plan_format(sorted(registered_tools())),
        PLAN_SHAPE,
        default=_PlanOut(),
    )
    plan = to_steps(draft)

    if not plan:
        diag.log("plan_node : planner returned nothing parseable — recording a parse-error incident")
        plan = _fallback_plan()

    diag.log(f"plan_node : {time.perf_counter() - start:.4f}s ({len(plan)} steps)")
    return {"plan": plan}
