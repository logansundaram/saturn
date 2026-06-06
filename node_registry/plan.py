import time
import diag
from langchain.messages import HumanMessage
from state import AgentState, steps_to_dicts
from llms import get_plan_model
from messages import planner_sys_msg


def plan_node(state: AgentState):
    """Draft the initial living plan: a short, ordered list of steps with human-readable
    labels. The plan is advisory and will be revised in-loop by update_plan.

    If the local model fails to emit valid structured output, fall back to a single generic
    step rather than aborting the turn — the agent loop can still resolve the request."""
    start = time.perf_counter()

    prompt = [
        planner_sys_msg,
        HumanMessage(
            content="Grounding context:\n"
            + state.get("context", "")
            + "\n\nUser request:\n"
            + state["current_query"]
        ),
    ]

    # Small local models (gemma4:e4b, the laptop tier) intermittently emit invalid JSON for the
    # Plan schema. Sampling differs run to run, so retry once — a second pass frequently parses —
    # before falling back to a single generic step so the loop can still resolve the request.
    plan = []
    for attempt in range(2):
        try:
            result = get_plan_model().invoke(prompt)
            plan = steps_to_dicts(result.steps)
            if plan:
                break
        except Exception as exc:
            diag.log(f"plan_node : structured-output attempt {attempt + 1} failed ({exc})")

    if not plan:
        diag.log("plan_node : falling back to a single generic step")
        plan = [
            {
                "step_id": 1,
                "label": "Resolve the user's request",
                "status": "pending",
                "intended_tool": None,
            }
        ]

    diag.log(f"plan_node : {time.perf_counter() - start:.4f}s ({len(plan)} steps)")
    return {"plan": plan}
