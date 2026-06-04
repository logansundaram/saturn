from langchain.messages import HumanMessage
from state import AgentState, steps_to_dicts
from llms import get_plan_model
from messages import planner_sys_msg


def plan_node(state: AgentState):
    """Draft the initial living plan: a short, ordered list of steps with human-readable
    labels. The plan is advisory and will be revised in-loop by update_plan.

    If the local model fails to emit valid structured output, fall back to a single generic
    step rather than aborting the turn — the agent loop can still resolve the request."""
    try:
        result = get_plan_model().invoke(
            [
                planner_sys_msg,
                HumanMessage(
                    content="Grounding context:\n"
                    + state.get("context", "")
                    + "\n\nUser request:\n"
                    + state["current_query"]
                ),
            ]
        )
        plan = steps_to_dicts(result.steps)
    except Exception as exc:
        print(f"plan_node : structured-output failed ({exc}); using fallback plan")
        plan = []

    if not plan:
        plan = [
            {
                "step_id": 1,
                "label": "Resolve the user's request",
                "status": "pending",
                "intended_tool": None,
            }
        ]

    return {"plan": plan}
