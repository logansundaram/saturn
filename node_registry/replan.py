"""
replan node — the in-loop verifier/repair step (the `judge` role).

This is the LLM-driven plan reviser the mechanical `update_plan` deliberately is NOT (see the
note in update_plan.py): it can INSERT a step mid-loop, not just advance statuses. It runs only
when the agent has finished with no tool calls and the mechanical nudge had nothing planned left
to escalate to (`route_after_agent`), so the common path never pays for it twice.

What it fixes: the agent answers a question that depends on current/external/specific facts
(rankings, "best X", prices, news, a person/product) straight from its own knowledge — either
because the planner never planned a gathering step, or because a knowledge-base lookup came back
irrelevant. The judge inspects that draft answer; if it is ungrounded it inserts a `web_search`
step into the plan as the new active step and routes back to `agent`, which then actually runs
the search and re-answers with real results. A grounded answer passes straight through to
synthesize untouched.

Gated by REPLAN_BUDGET (in node_registry/agent.py's route_after_agent) so a stubborn model can't
loop, and skipped when a web_search already ran this turn (re-searching the same turn won't help —
synthesize's honesty note covers that case instead). LLM-driven plan revision is only reliable on
a capable model; on the workstation tier (qwen3.5:9b) structured output holds, and a failure here
degrades safely to synthesize rather than aborting the turn.
"""

import time
import diag
from typing import Literal

from langchain.messages import AIMessage, HumanMessage
from langgraph.types import Command

from state import AgentState, active_step
from llms import get_judge_model
from messages import judge_sys_msg
from plan_ops import add_step
from registry import RETRIEVAL_TOOLS

# External-gathering tools: if ANY of these already ran this turn, escalating to a fresh web_search
# won't rescue the answer — re-searching in circles. The web tools aren't retrieval-flagged (their
# results aren't recorded as documents), so union them with the retrieval set explicitly.
_GATHERING_TOOLS = {"web_search", "web_extract"} | set(RETRIEVAL_TOOLS)


def _draft_answer(state: AgentState) -> str:
    """The agent's just-produced no-tool answer (the last AIMessage) — what we're judging."""
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage):
        return str(last.content or "")
    return ""


def replan_node(state: AgentState) -> Command[Literal["agent", "synthesize"]]:
    """Judge the draft answer; if it's ungrounded, insert a web_search step and loop back."""
    start = time.perf_counter()

    plan = state.get("plan", [])
    draft = _draft_answer(state)

    # Re-searching within the same turn won't rescue an answer the model already deemed thin after
    # a web round — let synthesize handle it honestly instead of escalating in circles.
    if _GATHERING_TOOLS & set(state.get("tools_called", [])) or not draft.strip():
        diag.log(f"replan_node : {time.perf_counter() - start:.4f}s (skipped)")
        return Command(goto="synthesize")

    gathered = state.get("tool_results", []) + state.get("documents_retrieved", [])
    judge_input = [judge_sys_msg]
    if state.get("context"):
        judge_input.append(HumanMessage(content=f"Grounding context:\n{state['context']}"))
    if gathered:
        judge_input.append(
            HumanMessage(content="Gathered this turn:\n" + "\n\n".join(map(str, gathered)))
        )
    else:
        judge_input.append(HumanMessage(content="Gathered this turn: (nothing — no tools ran)"))
    judge_input.append(HumanMessage(content=f"User request:\n{state['current_query']}"))
    judge_input.append(HumanMessage(content=f"Draft answer:\n{draft}"))

    try:
        verdict = get_judge_model().invoke(judge_input)
    except Exception as exc:
        # A structured-output failure must never strand the turn — accept the draft.
        diag.log(f"replan_node : judge failed ({exc}); accepting draft -> synthesize")
        return Command(goto="synthesize")

    if verdict.grounded or not (verdict.search_query or "").strip():
        diag.log(f"replan_node : {time.perf_counter() - start:.4f}s (grounded)")
        return Command(goto="synthesize")

    # Ungrounded: insert the web_search at the current step's position so it becomes the active
    # step the agent works next (after it runs, update_plan advances past it to the remaining work).
    # NOTE: web_search is the ONLY repair this node knows. That's deliberate for the MVP (the
    # ungrounded case is almost always "asserted a current/external fact without looking it up"),
    # but it's a coupling to revisit if you add tools: an answer that really needs the knowledge
    # base or a URL extract can't be repaired here — extend ReplanVerdict to carry a tool choice.
    query = verdict.search_query.strip()
    cur = active_step(plan)
    at = cur.get("step_id") if cur else None
    new_plan = add_step(plan, f"Search the web: {query}", "web_search", at=at)

    diag.log(f"replan_node : {time.perf_counter() - start:.4f}s (escalate -> web_search)")
    # Reset agent_nudges so the freshly inserted step gets its own full NUDGE_BUDGET. Otherwise a
    # nudge budget already spent earlier this turn leaves zero budget to make the model honor the
    # judge's inserted search, and route_after_agent silently falls through to synthesize.
    return Command(
        goto="agent",
        update={"plan": new_plan, "replans": state.get("replans", 0) + 1, "agent_nudges": 0},
    )
