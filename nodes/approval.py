"""
Approval node — the human-in-the-loop safety gate (Phase 2).

Whether a call skips the human is ONE question asked of ONE object: `policy.approves(name,
risk, args)` (the tier threshold + the /allow shell allowlist — see policy.py). Anything it
doesn't approve pauses via a LangGraph `interrupt` so the user can decide per batch or per
call. The policy is read live each call, so /config, /risk, /allow, /autoapprove and Shift+Tab
all apply to the very next gate. Resuming with the user's decision is handled in agent.run_turn.

Under the plan/execute engine (2026-07-03 transplant) the execute node emits exactly ONE call
per step, so a batch is normally a singleton — but the node stays batch-shaped (the resume-value
contract, per-call select, and the gate_event record are unchanged). A rejection routes to
`update_plan`, which records the decline onto the current step as a `skipped` incident; rectify
then cancels the remaining steps (a guarded action is reported, never retried or substituted).
The old positional `_skip_rejected_steps` walker is gone with the multiset accounting.
"""

from typing import Literal

from langchain.messages import ToolMessage
from langgraph.types import interrupt, Command

import diag
from trust import policy
from trust import quarantine
from tools.registry import risk_of
from core.state import AgentState, current_step

# The decline observation a rejected call gets. update_plan keys the `skipped` status off this
# text's prefix (nodes/update_plan._DECLINE_PREFIX) — change them together.
DECLINE_TEXT = (
    "Execution declined by the user. Do not retry this action; tell the user you "
    "did not perform it."
)


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
    headless --json "gates" field and the Glass Box's gate_summary — resist letting it grow.

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


def approval_node(state: AgentState) -> Command[Literal["tools", "update_plan"]]:
    """Human-in-the-loop safety gate. Calls within the configured auto-approve tier pass
    straight through. If any pending call exceeds it, pause via `interrupt` and let the user
    decide per batch OR per call.

    The resume value is a bool (True = approve the whole batch, False = reject it),
    `{"approved_ids": [...]}` from the UI's per-call select mode, or the always-allow decision
    dict `{"approved": True, "tools": [...], "shell_grants": [...]}` whose grants are applied
    past the interrupt by `_apply_always_grants`. Rejected calls get a decline
    ToolMessage here (orphaned tool_calls break the next model turn); everything else in the
    batch still routes to `tools`, which executes only the calls that don't already have a
    ToolMessage. A fully-rejected batch routes to `update_plan`, which records the decline onto
    the current plan step as a `skipped` incident."""
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
    # and the execute node's pre-action reasoning (the text content of the tool-calling
    # AIMessage) — the same provenance /trace why reconstructs later, surfaced at the moment of
    # decision.
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
            "step": current_step(state.get("plan", [])),
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
    # leaves it armed, so a re-issued copy of the call the human just declined faces the gate
    # again instead of auto-approving right past their 'no'. Re-issuing is itself rare now:
    # rectify cancels the remaining steps after a guarded outcome.
    if escalated and approved_ids:
        quarantine.consume_gate()

    # Record THIS human decision — exactly one structured event per prompt, riding the same
    # delta path tool_events takes into the trace DB. Recorded ONLY on the interrupt path:
    # an auto-approved batch never reaches here, so "gate_events empty" always means "the human
    # was never asked", not "the record was dropped".
    step = current_step(state.get("plan", []))
    event = gate_event(
        gated,
        approved_ids,
        quarantine=bool(escalated),
        step=(step or {}).get("label"),
    )

    if approved_ids == gated_ids:
        return Command(goto="tools", update={"gate_events": [event]})

    # Decline ONLY the rejected calls (orphaned tool_calls break the next model turn).
    rejected = [tc for tc in gated if tc["id"] not in approved_ids]
    decline = [
        ToolMessage(content=DECLINE_TEXT, tool_call_id=tc["id"], name=tc["name"])
        for tc in rejected
    ]
    update = {"messages": decline, "gate_events": [event]}

    # Anything left to run (ungated or approved) still runs; a fully-rejected batch goes
    # straight to the recorder — the decline lands on the current step as a `skipped` incident,
    # and rectify retires the remaining plan (a guarded action is reported, never retried).
    if len(rejected) < len(tool_calls):
        return Command(goto="tools", update=update)
    return Command(goto="update_plan", update=update)
