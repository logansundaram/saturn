"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Whether a call skips the human is ONE question asked of ONE object: `policy.approves(name,
risk, args)` (the tier threshold + the /allow shell allowlist — see policy.py). Anything it
doesn't approve pauses via a LangGraph `interrupt` so the user can decide per batch or per
call. The policy is read live each call, so /config, /risk, /allow, /autoapprove and Shift+Tab
all apply to the very next gate. Resuming with the user's decision is handled in agent.run_turn.
"""

from collections import Counter
from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

import diag
from trust import policy
from trust import quarantine
from tools.registry import risk_of
from core.state import AgentState, TERMINAL_STATUSES, active_step


def _skip_rejected_steps(
    plan: list[dict],
    rejected_tools: list[str],
    executing_tools: "list[str] | None" = None,
    called: "list[str] | None" = None,
) -> list[dict]:
    """Mark the planned step(s) a rejected tool call was fulfilling as `skipped`.

    Without this, a rejected call leaves its planned step non-terminal: `active_step` keeps the
    lockstep directive pinned to it and `unrun_planned_tools` keeps `route_after_agent` nudging,
    so the agent re-issues the very call the user just declined and the approval prompt fires again
    and again (bounded only by max_iterations — the reject -> infinite re-approve loop). Skipping
    the step retires that planned work so the plan advances past it.

    Matching mirrors `update_plan`/`unrun_planned_tools`: each rejected tool name is consumed,
    positionally as a multiset, against the first non-terminal step that expects it (so two
    same-tool steps don't both get skipped off one rejection) — BUT only after reserving the steps
    that the batch's surviving calls (`executing_tools`, the approved + ungated calls about to run)
    and prior rounds' calls (`called`) will credit. On a mixed approve/reject decision over a
    same-tool batch this is what keeps skip and credit aligned with update_plan's walk: without the
    reserve, the rejection would skip the FIRST matching step, the executed call's credit would
    flow past it to the next one, and the plan would record the opposite of what happened.

    A rejection that matched NO planned step falls back to skipping the first non-reserved step
    with an intended_tool — the planner's `intended_tool` guess didn't match what the agent
    actually called, but that step is still what lockstep was driving, so it must advance too.
    The fallback fires only when the main walk skipped NOTHING (mirroring update_plan_node's
    `if called and not newly_marked` — gotcha #6: the walkers keep identical accounting): a
    leftover rejected count alone must not trigger it, because rejected calls can OUTNUMBER the
    matching steps (two parallel run_shell calls serving one run_shell step) and the surplus
    would otherwise retire an UNRELATED later step the user was never asked about — silently
    dropping planned work with no skip disclosure. Returns a fresh list; never mutates the
    input."""
    if not plan:
        return plan
    plan = [dict(s) for s in plan]
    # Calls that have already credited steps (prior rounds) or are about to (this batch's
    # survivors). Done steps consume from this first, exactly like update_plan's walk.
    reserve = Counter(called or []) + Counter(executing_tools or [])
    remaining = Counter(rejected_tools)
    reserved_ids: set = set()
    skipped_any = False
    for step in plan:
        tool = step.get("intended_tool")
        status = step.get("status")
        if status in TERMINAL_STATUSES:
            if status == "done" and tool and reserve.get(tool, 0) > 0:
                reserve[tool] -= 1
            continue
        if not tool:
            continue
        if reserve.get(tool, 0) > 0:
            # An executed (or about-to-execute) call covers this step — leave it for update_plan
            # to credit as done.
            reserve[tool] -= 1
            reserved_ids.add(id(step))
            continue
        if remaining.get(tool, 0) > 0:
            remaining[tool] -= 1
            step["status"] = "skipped"
            skipped_any = True
    # Fallback: NO rejection lined up with any planned tool (`rejected_tools and not skipped_any`,
    # update_plan_node's `called and not newly_marked` in this walker's vocabulary). Skip the first
    # un-reserved step so the lockstep directive stops re-pointing the agent at the work it was
    # just told not to do. Guarded on skipped_any rather than a leftover count: once the main walk
    # skipped the step this batch was serving, a surplus same-tool rejection has no step left to
    # retire and must not spill onto an unrelated one.
    if rejected_tools and not skipped_any:
        for step in plan:
            if step.get("status") in TERMINAL_STATUSES or id(step) in reserved_ids:
                continue
            # Only skip a step that expected a tool. A no-intended_tool step (e.g. the generic
            # "answer the request" fallback plan) shouldn't be retired just because an unplanned
            # gated call was declined — the user declined one action, not the whole task.
            if step.get("intended_tool"):
                step["status"] = "skipped"
            break
    return plan


def gate_event(
    gated_calls: list,
    approved_ids,
    *,
    quarantine: bool = False,
    step: "str | None" = None,
) -> dict:
    """The structured record of ONE human gate decision, appended to state["gate_events"] only
    when the gate actually PROMPTED (auto-approved batches record nothing — there was no human
    decision to record). A human's yes/no is the one fact about a run that cannot be recomputed
    later, so this is the single justified persisted exception to the Glass Box's
    recompute-everything design. ONE minimal, JSON-serializable shape: the same record feeds the
    headless --json "gates" field, the Glass Box's gate_summary, and the signed export's answer
    attestation — resist letting it grow.

    `decision` summarizes the per-call verdicts: "approved" (everything let through),
    "rejected" (nothing), "partial" (a per-call select split the batch)."""
    calls = [
        {"id": tc["id"], "name": tc["name"], "approved": tc["id"] in approved_ids}
        for tc in gated_calls
    ]
    n_approved = sum(1 for c in calls if c["approved"])
    decision = (
        "approved" if n_approved == len(calls)
        else "rejected" if n_approved == 0
        else "partial"
    )
    return {
        "calls": calls,
        "decision": decision,
        "quarantine": bool(quarantine),
        "step": step,
    }


def _apply_always_grants(decision: dict) -> None:
    """Apply the gate's `a(lways)` grants: drop each listed tool to the auto-approved tier for
    the session (live registry.TOOL_RISK — exactly what /risk <tool> read_only does) and persist
    any scoped run_shell prefix grants through the one policy store (policy.grant_shell_prefix:
    screen -> coverage check with the one matcher -> add).

    The UI COLLECTS these at decision time without mutating anything (it validates with
    grant_shell_prefix(dry_run=True)); they are applied HERE, past the interrupt, because
    LangGraph re-executes this node from the top on resume and `gated` recomputes against the
    live policy — a grant applied while the interrupt was pending would auto-approve the very
    calls the human was prompted about, the re-run would return at the no-gated fast path
    without reaching the gate_event recording site, and the human's decision would vanish from
    the record (gotcha #7: empty must always mean "never asked"). Failures degrade safely: a
    refused shell grant just means the command faces the gate again next batch — diag-logged,
    since a node cannot print."""
    from tools import registry  # lazy, matching the UI: binds the live TOOL_RISK

    for name in decision.get("tools") or []:
        # run_shell never drops a tier (one keypress must not un-gate every future command —
        # it gets the scoped prefix grants below instead). The UI never sends it here, but the
        # resume value is still external input: fail closed.
        if name and name != "run_shell":
            registry.TOOL_RISK[str(name)] = "read_only"
    for grant in decision.get("shell_grants") or []:
        if not isinstance(grant, dict):
            continue
        try:
            ok, msg = policy.grant_shell_prefix(
                str(grant.get("prefix") or ""), str(grant.get("command") or "")
            )
        except Exception as exc:  # resume value is external input — a grant must never kill the turn
            ok, msg = False, str(exc)
        if not ok:
            diag.log(f"approval_node: always-allow shell grant refused — {msg}")


def approval_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """Human-in-the-loop safety gate. Calls within the configured auto-approve tier pass
    straight through. If any pending call exceeds it, pause via `interrupt` and let the user
    decide per batch OR per call.

    The resume value is a bool (True = approve the whole batch, False = reject it),
    `{"approved_ids": [...]}` from the UI's per-call select mode, or the always-allow decision
    dict `{"approved": True, "tools": [...], "shell_grants": [...]}` whose grants are applied
    past the interrupt by `_apply_always_grants`. Rejected calls get a decline
    ToolMessage here (orphaned tool_calls break the next model turn); everything else in the
    batch — ungated calls and per-call-approved ones — still routes to `tools`, which executes
    only the calls that don't already have a ToolMessage. Only a fully-rejected batch goes back
    to the agent to respond without having acted."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    # Quarantine escalation (runtime.quarantine = gate): a previous tool result this turn carried
    # instruction-shaped content, so this batch's arguments may derive from injected text — every
    # call in it faces the human ONCE regardless of risk tier. PEEK here, consume only after the
    # interrupt resolves: LangGraph re-executes this node from the top on resume, so a consuming
    # check would already be spent on the re-run, `gated` would recompute without the escalation,
    # and the user's rejection of the batch would be silently discarded (an all-auto-approved
    # batch would skip the interrupt entirely and run). The interrupt payload carries the flags so
    # the prompt can say why a normally-silent call is suddenly asking.
    escalated = quarantine.gate_pending() if tool_calls else False

    gated = [
        tc
        for tc in tool_calls
        if escalated
        or not policy.approves(tc["name"], risk_of(tc["name"]), tc.get("args"))
    ]

    if not gated:
        return Command(goto="tools")

    # Decision context for the gate's `e(xplain)` answer: the plan step this batch is fulfilling
    # and the agent's pre-action reasoning (the text content of the tool-calling AIMessage) — the
    # same provenance /trace why reconstructs later, surfaced at the moment of decision.
    reasoning = getattr(last, "content", "") or ""
    flags = quarantine.turn_flags()
    decision = interrupt(
        {
            "type": "approval_request",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                    "risk": risk_of(tc["name"]),
                }
                for tc in gated
            ],
            "step": active_step(state.get("plan", [])),
            "reasoning": reasoning if isinstance(reasoning, str) else str(reasoning),
            "quarantine": {"flags": flags} if flags else None,
        }
    )

    # Resolve the decision into the set of approved gated-call ids. Two dict shapes: the per-call
    # select ({"approved_ids": [...]}) and the always-allow decision ({"approved": True,
    # "tools": [...], "shell_grants": [...]}) — the latter's grants are applied here, past the
    # interrupt, never by the UI at decision time (see _apply_always_grants).
    gated_ids = {tc["id"] for tc in gated}
    if isinstance(decision, dict) and "approved_ids" in decision:
        approved_ids = gated_ids & set(decision.get("approved_ids") or [])
    elif isinstance(decision, dict):
        approved_ids = set(gated_ids) if decision.get("approved") else set()
        if approved_ids:
            _apply_always_grants(decision)
    elif decision:
        approved_ids = set(gated_ids)
    else:
        approved_ids = set()

    # Past the interrupt: this runs exactly once, with the human's decision in hand. The one-shot
    # escalation is spent only when the human LET SOMETHING THROUGH — a fully-rejected batch
    # leaves it armed, so a re-issued copy of the call the human just declined (small local
    # models do exactly this despite the decline message) faces the gate again instead of
    # auto-approving right past their 'no'. Re-prompting is bounded by max_iterations and by
    # _skip_rejected_steps retiring the planned work below.
    if escalated and approved_ids:
        quarantine.consume_gate()

    # Record THIS human decision — exactly one structured event per prompt, riding the same
    # delta path tool_events takes into the trace DB. Recorded ONLY on the interrupt path:
    # an auto-approved batch never reaches here, so "gate_events empty" always means "the human
    # was never asked", not "the record was dropped".
    step = active_step(state.get("plan", []))
    event = gate_event(
        gated,
        approved_ids,
        quarantine=bool(escalated),
        step=(step or {}).get("label"),
    )

    if approved_ids == gated_ids:
        return Command(goto="tools", update={"gate_events": [event]})

    # Decline ONLY the rejected calls. (Previously a rejection abandoned the whole batch and a
    # read-only call riding along had to be re-issued next iteration; now it just runs.)
    rejected = [tc for tc in gated if tc["id"] not in approved_ids]
    decline = [
        ToolMessage(
            content=(
                "Execution declined by the user. Do not retry this action; tell the user you "
                "did not perform it."
            ),
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in rejected
    ]

    # Retire the planned step(s) the rejected calls were fulfilling, so the plan stops demanding
    # work the user declined (otherwise lockstep + the nudge re-issue the same call indefinitely).
    update = {"messages": decline, "gate_events": [event]}
    plan = state.get("plan", [])
    if plan:
        rejected_ids = {tc["id"] for tc in rejected}
        update["plan"] = _skip_rejected_steps(
            plan,
            [tc["name"] for tc in rejected],
            # The batch's survivors (approved gated + ungated calls) WILL run and be credited by
            # update_plan — reserve their steps so a rejection doesn't skip the wrong one.
            executing_tools=[tc["name"] for tc in tool_calls if tc["id"] not in rejected_ids],
            called=state.get("tools_called", []),
        )

    # Anything left to run (ungated or approved) still runs; a fully-rejected batch goes back to
    # the agent instead.
    if len(rejected) < len(tool_calls):
        return Command(goto="tools", update=update)
    return Command(goto="agent", update=update)
