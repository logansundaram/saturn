"""
Replan node — rewrites the REMAINING plan when rectify (or a mid-turn steer) says it must change
(the 2026-07-03 engine transplant; replaces the single-purpose groundedness judge, whose job now
lives in RECTIFY_SYS's current/external-facts rule).

Completed steps are kept verbatim (their results are the data bus — never re-run, never
redescribed); the planner redrafts everything still pending, with `state["reasoning"]` (rectify's
verdict, or the user's steering correction) as the revision instruction. The instruction block
teaches the hard-won lessons: replace references with EXACT names from the results, never invent
or substitute, expand fan-outs one concrete step per item, each calculation its own step, writes
last and value-free.

Bounded by rectify.MAX_REPLANS; a failed redraft degrades to the untouched plan (the engine
keeps executing the pending steps as they were)."""

import time

import diag
from langchain.messages import HumanMessage

from core.messages import planner_sys_msg
from core.plan_context import (
    ORIGIN_REPLAN,
    is_revoked,
    original_request,
    plan_txt,
    request_authorized,
    vetoes_block,
)
from core.plan_ops import retarget_knowledge_base_reads
from core.state import AgentState
# One vocabulary for the refusal: execute stamps this same prefix when it re-checks the
# generated ARGUMENTS (one producer, as with update_plan's DECLINE_TEXT import from approval).
from nodes.execute import UNAUTHORIZED_PREFIX
from nodes.rectify import MAX_REPLANS, retryable_dead_end
from core.structured import (
    _PlanOut,
    PLAN_SHAPE,
    plan_format,
    registered_tools,
    structured,
    to_steps,
)

# Hard cap on freshly drafted steps per replan — a runaway fan-out must not mint a 40-step plan.
_MAX_NEW_STEPS = 10


def _revision_instruction(state: AgentState) -> str:
    # The user's plan-review vetoes lead the instruction when present: the redraft must never
    # reinstate work the human explicitly removed (plan_context.vetoes_block — the same block
    # rectify's judge sees, so the two ends of the replan seam can't disagree about scope).
    veto = vetoes_block(state)
    return ((veto + "\n\n") if veto else "") + (
        f"The plan so far:\n{plan_txt(state.get('plan') or [])}\n\n"
        f"It needs fixing: {state.get('reasoning') or '(no reason recorded)'}\n\n"
        "Produce ALL the remaining steps needed to FULLY complete the request, "
        "replacing the pending ones. Do not repeat completed steps. Include every "
        "read, calculation, and the final write the request requires, in order.\n"
        "Replace EVERY reference with the EXACT name/value from the results above — "
        "e.g. 'the last CSV the manifest lists' becomes the real filename like "
        "revenue_q3.csv. Never restate a reference and never invent a "
        "name that is not in the results.\n"
        "If the results show the referenced item does NOT exist — the search/read "
        "returned only unrelated content (a value labeled as something else, a file "
        "that merely looks related) — do NOT substitute it and do NOT add steps to "
        "hunt in other files: emit a single 'none' step stating the item was not "
        "found, and drop any write step that depended on it.\n"
        "If a step says to act 'for each' item in a list you already have, expand it "
        "into ONE concrete read step plus ONE compute step PER item (using the exact "
        "filenames), then any comparison/write step. The items are already known — do "
        "NOT add a step to list or re-discover them.\n"
        "A read_file step only READS a file — never describe a read step as computing "
        "a total; each total/sum is its own calculate step after that file's read. E.g. for "
        "a.csv and b.csv: 'Read a.csv' (read_file), 'Sum the revenue column "
        "of a.csv' (calculate), 'Read b.csv' (read_file), 'Sum the revenue column "
        "of b.csv' (calculate), then the comparison as a 'none' step. If the comparison "
        "itself needs arithmetic (a difference, 'by how much'), that arithmetic is "
        "its own calculate step BEFORE the final none step — never done in prose.\n"
        "If a step failed because the shell lacks a tool (bc, python), do NOT retry it "
        "in the shell — arithmetic is a calculate step over the numbers already read.\n"
        "Order matters: compute a grand total or comparison ONLY after every per-item "
        "value is computed, and put any write step LAST. In a write/report step, do "
        "NOT embed specific numbers in the description — say 'write the totals from the "
        "previous steps' so the real computed values are used, not guessed ones."
    )


def _renumber(steps: list) -> list:
    for i, s in enumerate(steps, 1):
        s["step_id"] = i
    return steps


def replan_node(state: AgentState):
    start = time.perf_counter()
    plan = state.get("plan") or []
    done = [dict(s) for s in plan if s.get("result") is not None]

    draft = structured(
        "planner",
        [
            planner_sys_msg(),
            HumanMessage(content=original_request(state)),
            HumanMessage(content=_revision_instruction(state)),
        ],
        _PlanOut,
        plan_format(sorted(registered_tools())),
        PLAN_SHAPE,
        default=_PlanOut(),
    )

    # A completed step blocks a same-label redraft ONLY when it actually produced something:
    # a RETRYABLE step whose result was a dead end (empty search, not-found, nonzero exit) or an
    # incident (status != done) may legitimately be RETRIED under the same wording — rectify's
    # dead-end branch asks for exactly that, and the planner routinely echoes the label it was
    # shown. Filtering those as "duplicates" emptied the redraft and silently defeated the
    # bounded retry (the turn concluded "not found" without ever re-searching). The guard is
    # rectify's retryable_dead_end — tool-gated, so a calculate step whose honest result is "0"
    # (a computed VALUE, the _EMPTY_MARKERS rule) still blocks its duplicate. Still bounded:
    # the retry budget and MAX_REPLANS cap the loop either way.
    done_descs = {
        str(s.get("label") or "").strip().lower()
        for s in done
        if s.get("status") == "done" and not retryable_dead_end(s)
    }
    # Mechanical backstop for the veto instruction above: a redraft that resurrects a
    # user-removed step VERBATIM drops here (exact label match, case-insensitive); a reworded
    # resurrection is the prompt's job to prevent.
    vetoed = {str(v).strip().lower() for v in state.get("plan_vetoes") or []}
    fresh = [
        s for s in retarget_knowledge_base_reads(to_steps(draft))
        if str(s.get("label") or "").strip().lower() not in done_descs
        and str(s.get("label") or "").strip().lower() not in vetoed
    ]
    # The EFFECT-typed backstop for the reworded resurrection the label match misses (2026-08-15,
    # from the engine isolate). A cheap pre-filter, not the guarantee: `execute` refuses a
    # revoked action on its generated ARGUMENTS; dropping it here spares a replan cycle.
    revoked = state.get("revoked_writes") or []
    new_steps = [
        s for s in fresh if not is_revoked(revoked, s.get("intended_tool"), s.get("label"))
    ]
    # EFFECT AUTHORIZATION: this redraft's prompt contains plan_txt, and plan_txt contains the
    # RESULTS — anything a file or a web page says can appear here as a proposed step. Every
    # step drafted while results exist is stamped origin=replan, and a state-changing one is
    # dropped unless the USER'S words name its target (execute re-checks on the arguments).
    refused: list = []
    if done:
        for s in new_steps:
            s["origin"] = ORIGIN_REPLAN
        authorized = []
        for s in new_steps:
            (authorized if request_authorized(state, s) else refused).append(s)
        new_steps = authorized
    # A refused step is RECORDED, never silently dropped. Filtering it out of the redraft with no
    # trace left synthesize composing a completed turn whose effect never happened: incidents_block
    # was empty, so the answer had nothing to disclose and said the work was done. (A REVOKED drop
    # above needs no record of its own — the plan-review veto already rides state["plan_vetoes"]
    # into the answer as work removed at the user's request.) Stamped terminal here, so the step
    # is never the execution pointer; appended LAST so it cannot cancel work in front of it —
    # rectify's guarded branch reaches it only once everything else has run, with nothing left to
    # cancel, and lands the turn at an honest synthesize.
    for s in refused:
        s["status"] = "blocked"
        s["result"] = (
            f"{UNAUTHORIZED_PREFIX} this step was added after the results came back and it acts "
            "on something the request never named. Results may inform HOW the request is "
            "carried out, never WHAT is done to the workspace."
        )

    if fresh and not new_steps:
        # The redraft was ENTIRELY revoked/unauthorized work: another cycle can only produce it
        # again (rectify sees the same plan and asks for the same missing write). Consuming the
        # budget lands the turn at an honest synthesize NOW. A REVOKED redraft needs no record
        # here — the plan-review note already tells the answer to describe it as removed at the
        # user's request — but an UNAUTHORIZED one is a refusal the user never asked for, so the
        # blocked steps ride onto the plan as disclosed incidents rather than vanishing.
        diag.log(f"replan_node : {time.perf_counter() - start:.4f}s (redraft wholly revoked)")
        out = {"replans": MAX_REPLANS, "rectify": False,
               "reasoning": "remaining work revoked or unauthorized"}
        if refused:
            # `done` is already the completed prefix, so the record is exactly
            # `what ran` + `what was refused`; nothing pretends to be still pending.
            out["plan"] = _renumber(done + refused)
        return out

    if not new_steps:
        # A failed/empty redraft keeps the pending steps as they were — degrading to the old
        # plan beats stranding the turn (and beats silently dropping the remaining work).
        diag.log(f"replan_node : {time.perf_counter() - start:.4f}s (empty redraft — plan kept)")
        return {
            "replans": state.get("replans", 0) + 1,
            "rectify": False,
            "reasoning": "",
        }

    # Refused steps land AFTER the surviving work: terminal already, so never the execution
    # pointer, and reached by rectify's guarded branch only once everything in front of them has
    # run — with nothing left to cancel.
    merged = _renumber(done + new_steps[:_MAX_NEW_STEPS] + refused)
    diag.log(
        f"replan_node : {time.perf_counter() - start:.4f}s "
        f"({len(done)} kept + {len(merged) - len(done) - len(refused)} redrafted"
        + (f" + {len(refused)} refused)" if refused else ")")
    )
    return {
        "plan": merged,
        "replans": state.get("replans", 0) + 1,
        "rectify": False,
        "reasoning": "",
    }
