import time
from langchain.messages import HumanMessage
from state import AgentState, PlanStep
from llms import llm_with_plan
from messages import planner_system_msg


def plan_node(state: AgentState):
    """Draft the initial living plan: a short, ordered list of PlanStep with human-readable
    labels. The plan is advisory and will be revised in-loop by update_plan.

    If the local model fails to emit valid structured output, fall back to a single generic
    step rather than aborting the turn — the agent loop can still resolve the request."""
    start = time.perf_counter()

    try:
        result = llm_with_plan.invoke(
            [
                planner_system_msg,
                HumanMessage(
                    content="Grounding context:\n"
                    + state.get("context", "")
                    + "\n\nUser request:\n"
                    + state["current_query"]
                ),
            ]
        )
        steps = result.steps
    except Exception as exc:
        print(f"plan_node : structured-output failed ({exc}); using fallback plan")
        steps = []

    if not steps:
        steps = [PlanStep(step_id=1, label="Resolve the user's request", status="pending")]

    # Normalize: ensure sequential 1-based ids and a pending start state.
    for i, step in enumerate(steps, start=1):
        step.step_id = i
        step.status = "pending"

    print(f"plan_node : {time.perf_counter() - start:.4f}s ({len(steps)} steps)")
    return {"plan": steps}
