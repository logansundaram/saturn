"""Graph assembly: wire the nodes/ package into the compiled plan/execute engine.

This is the ONLY place the LangGraph is assembled (see CLAUDE.md design rules) — the node
functions stay atomic in nodes/ (one file per node, routing helpers beside their node), and
everything runtime-shaped (turn driving, CLI, REPL) lives in the sibling app/ modules.
"""

import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_config
from core.state import AgentState

# loop nodes
from nodes.ground import grounding_node
from nodes.plan import plan_node
from nodes.synthesize import synthesize_node, route_after_synthesize
from nodes.answer_gate import answer_gate_node
from nodes.update_plan import update_plan_node
from nodes.execute import execute_node, route_after_execute
from nodes.rectify import rectify_node, route_after_rectify
from nodes.tools import tool_node
from nodes.approval import approval_node
from nodes.replan import replan_node
from nodes.plan_gate import plan_gate_node, route_after_gate

DB_PATH = str(get_config().path("db_sqlite"))


def build_agent():
    """Assemble the plan/execute engine (the 2026-07-03 agentic-harness transplant) with the
    human-in-the-loop approval gate AND the plan-review gate:

        START -> ground -> plan -> plan_gate -> execute -> approval -> tools -> update_plan ┐
                              ┌──────────────↑   │  │         │                             │
                              │     (no call:    │  │   (fully rejected -> update_plan      │
                              │      reasoning/  │  │    records the decline as `skipped`)  │
                              │      gate skip) ─┘  │                                       │
                              │                     ▼                                       │
                              │  replan <──(rectify=true)── rectify <───────────────────────┘
                              │     │                          │ (steps left)      (done/capped)
                              └─────┴──────> plan_gate ────────┘                        ▼
                                     (abort -> synthesize · steer -> replan)        synthesize -> END

    The plan is the DATA BUS: each step carries its own `result`, written by `update_plan` (tool
    steps) or `execute` itself (reasoning steps, write-gate skips). `execute` runs exactly one
    step per pass with a curated context and a single-tool constrained call; `rectify` reflects
    after every step (deterministic short-circuits first, LLM judgment last); `replan` redrafts
    the remaining steps when rectify — or a mid-turn steer — says they must change.

    `plan_gate` runs at every step boundary: a pass-through unless a pause has been requested, in
    which case it `interrupt()`s so the user can inspect/edit the plan and resume (see
    nodes/plan_gate.py). Compiled with a SqliteSaver checkpointer, which both persists
    sessions and is what lets the approval / plan-review `interrupt`s pause and resume.

    Interrupt-and-correct (token steering) adds one loop at the tail: `synthesize` streams the
    answer into a provenance-tagged buffer; a mid-stream freeze (Esc) routes to `answer_gate`,
    which `interrupt()`s so the user can edit the frozen text, then hands the corrected buffer
    back to `synthesize` — which CONTINUES the edited prefix (raw-mode continuation,
    core/continuation.py) or finalizes it:

        synthesize ── frozen ──> answer_gate ──> synthesize ── … ──> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("ground", grounding_node)
    builder.add_node("plan", plan_node)
    builder.add_node("plan_gate", plan_gate_node)
    builder.add_node("execute", execute_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", tool_node)
    builder.add_node("update_plan", update_plan_node)
    builder.add_node("rectify", rectify_node)
    builder.add_node("replan", replan_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("answer_gate", answer_gate_node)

    builder.add_edge(START, "ground")
    builder.add_edge("ground", "plan")
    # Every step boundary flows through plan_gate (the plan-review checkpoint) before execution.
    builder.add_edge("plan", "plan_gate")
    builder.add_conditional_edges(
        "plan_gate",
        route_after_gate,
        # Normally -> execute; a mid-turn steer arms a replan; an abort wraps up at synthesize.
        {"execute": "execute", "replan": "replan", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        # A generated tool call faces the approval gate; anything else (reasoning result
        # recorded, write-gate skip, argument failure, nothing left to run) reflects at rectify.
        {"approval": "approval", "rectify": "rectify"},
    )
    # approval routes dynamically via Command(goto=...): "tools" (approved) or "update_plan"
    # (fully rejected — the decline is recorded onto the current step as a skipped incident).
    builder.add_edge("tools", "update_plan")
    builder.add_edge("update_plan", "rectify")
    builder.add_conditional_edges(
        "rectify",
        route_after_rectify,
        {"replan": "replan", "plan_gate": "plan_gate", "synthesize": "synthesize"},
    )
    builder.add_edge("replan", "plan_gate")
    # The interrupt-and-correct tail: a frozen answer buffer detours through the answer_gate
    # edit interrupt and re-enters synthesize (which continues the edited prefix); an
    # unfrozen pass ends the turn.
    builder.add_conditional_edges(
        "synthesize",
        route_after_synthesize,
        {"answer_gate": "answer_gate", "end": END},
    )
    builder.add_edge("answer_gate", "synthesize")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
