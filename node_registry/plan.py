import time
import diag
from langchain.messages import HumanMessage
from state import AgentState, steps_to_dicts
from llms import get_plan_model
from messages import planner_sys_msg


# One-shot plan seed (/plan run <recipe>): the next plan_node pass uses these steps instead of
# drafting. Consumed on use, so only the recipe's own turn is affected — everything downstream
# (lockstep, the gate, update_plan, the trace) treats the seeded plan exactly like a drafted one.
# The seed is TIED TO ITS QUERY: if the armed turn never reaches plan_node (Ctrl-C or an exception
# during grounding), the leftover seed must not hijack the user's next, unrelated question — a
# mismatched query discards the seed and drafts normally.
_SEEDED_STEPS: "list[dict] | None" = None
_SEEDED_QUERY: "str | None" = None


def seed_next_plan(steps: list[dict], query: str = "") -> int:
    """Arm the next turn's plan with recipe steps ({label, intended_tool} dicts). Returns how
    many steps were armed (empty labels are dropped). `query` is the exact text the recipe turn
    will run — plan_node consumes the seed only for that query (an empty query matches any, for
    callers that requeue immediately)."""
    global _SEEDED_STEPS, _SEEDED_QUERY
    _SEEDED_STEPS = [
        {
            "step_id": i,
            "label": str(s.get("label") or ""),
            "status": "pending",
            "intended_tool": s.get("intended_tool"),
        }
        for i, s in enumerate((s for s in steps if str(s.get("label") or "").strip()), start=1)
    ]
    if not _SEEDED_STEPS:
        _SEEDED_STEPS = None
        _SEEDED_QUERY = None
        return 0
    _SEEDED_QUERY = str(query or "").strip()
    return len(_SEEDED_STEPS)


def plan_node(state: AgentState):
    """Draft the initial living plan: a short, ordered list of steps with human-readable
    labels. The plan is advisory and will be revised in-loop by update_plan.

    If the local model fails to emit valid structured output, fall back to a single generic
    step rather than aborting the turn — the agent loop can still resolve the request."""
    global _SEEDED_STEPS, _SEEDED_QUERY
    if _SEEDED_STEPS:
        plan, query_key = _SEEDED_STEPS, _SEEDED_QUERY
        _SEEDED_STEPS = _SEEDED_QUERY = None
        if not query_key or query_key == str(state.get("current_query") or "").strip():
            diag.log(f"plan_node : using seeded recipe plan ({len(plan)} steps)")
            return {"plan": plan}
        # The armed turn never ran (cancelled/crashed before reaching here) and this is a
        # different question — drop the stale seed instead of hijacking it.
        diag.log("plan_node : discarding stale recipe seed (armed for a different query)")

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
