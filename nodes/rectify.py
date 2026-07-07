"""
Rectify node — the engine's reflective checkpoint, run after every executed step (the 2026-07-03
engine transplant; replaces the nudge/judge routing of the ReAct loop).

Deliberately DETERMINISTIC-FIRST: a chain of mechanical short-circuits answers the common cases
for free, and only the leftover ambiguity pays for an LLM judgment (the `judge` role). In order:

  1. guarded outcome     — the step just recorded was skipped/blocked (user rejection at the
                           approval gate, the semantic write gate, a BLOCKED refusal): CANCEL the
                           remaining steps and report. A guarded action must never be retried or
                           substituted around.
  2. resolution check    — the NEXT step is a needs_resolution placeholder ("the file the
                           listing names"): when a search produced the evidence, an LLM presence
                           check first verifies the referenced item actually exists in the
                           gathered results (a semantic retriever always returns nearest-neighbor
                           hits, so "the search returned something" never implies "found").
                           Found -> rectify=true (replan makes the reference concrete); absent ->
                           cancel the remaining steps and report it missing — never hunt for a
                           substitute. Read-only chains (a pointer path, a manifest list) resolve
                           mechanically without the LLM check. Deferred write/edit steps are
                           EXEMPT from forced resolution: the write gate judges the raw results
                           and is the authority on whether the value exists — replanning a write
                           invites a fabricated provenance hop that launders an unrelated value
                           past it.
  3. concrete pending    — pending steps remain and nothing failed: no LLM call, keep executing.
  4. dead-end retry      — the LAST step came back empty/zero from a retryable search/list tool
                           (a wrong pattern or scope may hide real data): one bounded rectify to
                           try a different concrete approach. A read_file miss or an empty
                           knowledge-base search is a genuine absence, not retryable.
  5. LLM verdict         — everything else: the judge decides whether the plan must change or
                           extend (RECTIFY_SYS), including the groundedness rule — a
                           current/external-facts request that never ran a web_search is sent to
                           replan so a search step gets added.

`route_after_rectify`: rectify -> replan; steps left -> plan_gate (the step boundary, so review
pauses + steering keep their seam) -> execute; else synthesize. The iteration cap
(config runtime.max_iterations, counted in execute passes) and MAX_REPLANS bound every loop edge.
"""

import re
import time

import diag
from langchain.messages import HumanMessage

from config import get_config
from core.messages import RECTIFY_SYS, RESOLVE_CHECK_SYS
from core.plan_context import (
    SEARCH_TOOLS,
    WRITE_TOOLS,
    original_request,
    plan_txt,
    results_block,
    vetoes_block,
)
from core.plan_ops import is_review_retirement
from core.state import AgentState
from core.structured import (
    RectifyBool,
    RECTIFY_FORMAT,
    RECTIFY_SHAPE,
    ResolutionCheck,
    RESOLUTION_FORMAT,
    RESOLUTION_SHAPE,
    structured,
)

# How many times one turn may rewrite the remaining plan. Generous — rectify's short-circuits
# make most cycles free — but hard: a model that keeps finding "one more fix" lands at an honest
# synthesize instead of spinning.
MAX_REPLANS = 5

# SEARCH_TOOLS (arms the LLM presence check) and WRITE_TOOLS (exempt from forced resolution)
# come from core/plan_context — THE one home for the engine's tool classifications, shared with
# execute's write gate and synthesize's write verification.
# Dead ends from these tools are retryable ONCE — a wrong pattern/scope may hide real data.
_RETRYABLE = ("run_shell", "search_files", "find_files", "list_directory")

# run_shell's observation header (tools/shell.py _format). Anchored at the start so a transcript
# that merely MENTIONS "exit code 128" mid-output never reads as a failed run.
_EXIT_CODE_RE = re.compile(r"\[exit code (\d+)\]")


def _cancel_remaining(plan: list[dict], text: str) -> list[dict]:
    """Retire every un-run step with `text` as its result (status `cancelled`). Fresh copies —
    never mutates state's plan in place."""
    plan = [dict(s) for s in plan]
    for s in plan:
        if s.get("result") is None:
            s["result"] = text
            s["status"] = "cancelled"
    return plan


def rectify_node(state: AgentState):
    start = time.perf_counter()
    plan = state.get("plan") or []
    if not plan:
        return {"rectify": False, "reasoning": "no plan"}

    pending = any(s.get("result") is None for s in plan)
    last_done = None
    for s in plan:
        if s.get("result") is None:
            break
        last_done = s
    res = str(last_done.get("result") or "") if last_done else ""
    failed = bool(last_done) and (
        not res.strip()
        or last_done.get("status") == "error"
        or res.lower().startswith("error")
    )

    # 1. A guarded action (gate rejection, write-gate skip, BLOCKED refusal) ends the run:
    #    report it, do not retry or substitute. Checked BEFORE the replan budget (branch order
    #    is load-bearing, gotcha #8): a rejection recorded after the budget is spent must still
    #    cancel the remaining steps, not leave them mislabeled "never ran".
    #    EXEMPT: a step the USER retired at the plan-review editor (plan_ops' review stamp,
    #    2026-07-06) — that is a single-step veto ("don't do this one, continue the rest"),
    #    not a mid-execution rejection ending the run; the veto itself rides
    #    state["plan_vetoes"] into the judge/replan prompts so it is never reinstated.
    if (
        last_done is not None
        and last_done.get("status") in ("skipped", "blocked")
        and not is_review_retirement(last_done)
    ):
        diag.log(f"rectify_node : {time.perf_counter() - start:.4f}s (guarded -> cancel)")
        return {
            "rectify": False,
            "plan": _cancel_remaining(plan, "cancelled: a prior guarded action ended the run"),
            "reasoning": "action guarded; report it, do not retry or substitute",
        }

    # Replan budget spent: no further revision — route_after_rectify lands at synthesize.
    if state.get("replans", 0) >= MAX_REPLANS:
        return {"rectify": False, "reasoning": "replan budget spent"}

    # 2. The next step is a deferred reference — resolve it (or report the item missing).
    nxt = next((s for s in plan if s.get("result") is None), None)
    if nxt is not None and nxt.get("intended_tool") not in WRITE_TOOLS:
        done_before = any(s.get("result") is not None for s in plan[: plan.index(nxt)])
        if done_before and nxt.get("needs_resolution"):
            searched = any(
                s.get("intended_tool") in SEARCH_TOOLS and s.get("result") is not None
                for s in plan
            )
            check = (
                structured(
                    "judge",
                    [
                        RESOLVE_CHECK_SYS,
                        HumanMessage(
                            content=f"Request: {original_request(state)}\n\n"
                            f"{results_block(plan)}\n\n"
                            f"The step that needs resolving: {nxt.get('label')}\n\n"
                            "Do the results actually contain the item it refers to?"
                        ),
                    ],
                    ResolutionCheck,
                    RESOLUTION_FORMAT,
                    RESOLUTION_SHAPE,
                    default=ResolutionCheck(found=True, evidence="check empty; assume found"),
                )
                if searched
                else ResolutionCheck(found=True, evidence="no search step; mechanical resolution")
            )
            if not check.found:
                diag.log(f"rectify_node : {time.perf_counter() - start:.4f}s (item absent -> cancel)")
                return {
                    "rectify": False,
                    "plan": _cancel_remaining(
                        plan,
                        "cancelled: the item this step needs was not found in "
                        "the gathered results — report it missing, do not "
                        "substitute another value",
                    ),
                    "reasoning": "referenced item absent; report it missing",
                }
            diag.log(f"rectify_node : {time.perf_counter() - start:.4f}s (resolve reference)")
            return {
                "rectify": True,
                "reasoning": "Resolve the next step's reference to an exact file/value (or expand "
                "it into one concrete step per item) using the results gathered so "
                "far, then continue. But ONLY if the results actually contain the "
                "referenced item; if they came back with only unrelated content "
                "(nothing that IS the requested item), replace the remaining steps "
                "with one 'none' step reporting the item was not found — never read "
                "other files hunting for a substitute.",
            }

    # 3. Concrete pending steps and nothing failed: keep executing, no LLM call.
    if pending and not failed:
        diag.log(f"rectify_node : {time.perf_counter() - start:.4f}s (pending, pass)")
        return {"rectify": False, "reasoning": "concrete pending steps; nothing to react to"}

    # 4. Search/list/count dead ends are retryable once (a wrong pattern or scope may be hiding
    #    real data); a read_file miss or an empty knowledge-base search is a genuine absence.
    rl = res.strip().lower()
    first = rl.splitlines()[0] if rl else ""
    exit_m = _EXIT_CODE_RE.match(rl)  # the run_shell header, anchored at the observation start
    dead_end = (
        first in ("", "0", "0.0", "[]", "none", "no results", "no matches")
        or "not found" in rl
        or "no such file" in rl
        or "not a directory" in rl
        or rl.startswith("no matches for")
        or rl.startswith("no files matching")
        or bool(exit_m and int(exit_m.group(1)) != 0)
    )
    if (
        not pending
        and dead_end
        and last_done is not None
        and last_done.get("intended_tool") in _RETRYABLE
        and state.get("replans", 0) < 2
    ):
        diag.log(f"rectify_node : {time.perf_counter() - start:.4f}s (dead end -> retry)")
        return {
            "rectify": True,
            "reasoning": "The search/list step came up empty / zero. Before concluding, try "
            "ONE different concrete approach: list the workspace (list_directory) to see "
            "the real filenames, read the specific file directly, or rerun the "
            "search across EVERY file including ones in subfolders (search_files on "
            "the workspace root) with a corrected keyword/pattern (the data may exist under "
            "a different name or token). If that still finds nothing, report it "
            "plainly and do NOT invent.",
        }

    # 5. The leftover ambiguity: the judge decides (incl. the groundedness rule — a
    #    current/external-facts request that never searched the web gets rectify=true). The
    #    user's plan-review vetoes ride along: work the human removed at review is deliberately
    #    out of scope, never a gap for the judge to repair.
    msgs = [RECTIFY_SYS, HumanMessage(content=original_request(state))]
    veto_note = vetoes_block(state)
    if veto_note:
        msgs.append(HumanMessage(content=veto_note))
    msgs.append(HumanMessage(content=f"Current plan:\n{plan_txt(plan)}"))
    decision = structured(
        "judge",
        msgs,
        RectifyBool,
        RECTIFY_FORMAT,
        RECTIFY_SHAPE,
        default=RectifyBool(rectify=False, reasoning="rectify empty; proceeding"),
    )
    diag.log(
        f"rectify_node : {time.perf_counter() - start:.4f}s "
        f"(judge: rectify={decision.rectify})"
    )
    return {"rectify": decision.rectify, "reasoning": decision.reasoning}


def route_after_rectify(state: AgentState) -> str:
    """rectify -> replan; steps left -> plan_gate (the review/steer boundary) -> execute; done ->
    synthesize. The iteration cap and replan budget both force an honest landing."""
    if state.get("iteration", 0) >= get_config().max_iterations:
        return "synthesize"
    if state.get("replans", 0) >= MAX_REPLANS:
        return "synthesize"
    if state.get("rectify"):
        return "replan"
    if any(s.get("result") is None for s in state.get("plan") or []):
        return "plan_gate"
    return "synthesize"
