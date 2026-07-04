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


def _fallback_plan() -> list[dict]:
    """One generic step so the engine can still resolve the request when the planner emits
    nothing parseable — the execute node treats a tool-less step as pure reasoning."""
    return [
        {
            "step_id": 1,
            "label": "Resolve the user's request",
            "status": "pending",
            "intended_tool": None,
            "result": None,
            "needs_resolution": False,
        }
    ]


def plan_node(state: AgentState):
    """Draft the plan: an ordered list of one-action steps, each naming the ONE tool it calls
    (or none for pure reasoning). The plan is the engine's data bus — each step's result is
    recorded on it as it executes — so plan quality directly drives execution.

    Structured output goes through the hardened path (core/structured.py: flat schema, shape
    hint, JSON salvage, temp-escalating retries); a total parse failure degrades to a single
    generic step rather than aborting the turn."""
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
        diag.log("plan_node : falling back to a single generic step")
        plan = _fallback_plan()

    diag.log(f"plan_node : {time.perf_counter() - start:.4f}s ({len(plan)} steps)")
    return {"plan": plan}
