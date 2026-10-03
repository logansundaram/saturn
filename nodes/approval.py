"""
Approval node — the human-in-the-loop safety gate.

Whether a call skips the human is ONE question asked of ONE object: `policy.approves(name,
risk, args)` (the tier threshold + the shell allowlist — see policy.py). Anything it
doesn't approve pauses via a LangGraph `interrupt` so the user can decide per batch or per
call. The policy is read live each call, so /config, /policy (risk · allow · open) and Shift+Tab
all apply to the very next gate. Resuming with the user's decision is handled in app/turn.run_turn.

The agent node answers some calls itself (malformed, repeated, declined-before) with
ToolMessages before this node runs, so the batch is the issuing AIMessage's calls MINUS those
already answered — the same walk-back nodes/tools.py does. A fully-rejected batch routes back to `agent`: the decline ToolMessages are
what the model sees, and nodes/agent's declined-repeat guard refuses the same call for the rest
of the turn (a guarded action is reported, never retried or substituted).
"""

from typing import Literal

from langchain.messages import AIMessage, ToolMessage
from langgraph.types import interrupt, Command

import diag
from trust import policy
from trust import quarantine
from tools.registry import DECLARED_RISK, risk_of
from core.state import AgentState, current_step, is_steer_message, is_turn_start, issuing_message

# The decline observation a rejected call gets. The structural `saturn_status: skipped` stamp on
# the ToolMessage is what readers key on (nodes/agent.py's declined-repeat guard + incidents).
DECLINE_TEXT = (
    "Execution declined by the user. Do not retry this action; tell the user you "
    "did not perform it."
)


AIRGAP_NOTE = ("air-gap is on: Saturn cannot see inside a shell command, a shortcut or an MCP "
               "server — approve only if this will not use the network")
SEND_NOTE = "this sends your words to another person; a send always asks, whatever the policy"


def _can_act(name: str) -> bool:
    """Whether a call can send something out or change something — what a quarantine escalation
    exists to put in front of the human. The DECLARED tier counts as well as the live one: a
    tier the user relaxed must not take the tool out from under the escalation."""
    return (quarantine.is_outbound(name) or risk_of(name) != "read_only"
            or DECLARED_RISK.get(name, "destructive") != "read_only")


def provenance(state) -> "tuple[str, str, bool]":
    """(what the user typed, everything else that ENTERED the conversation, whether any of that
    came from an untrusted tool or an attachment) — the three facts quarantine.url_hold reads (the first
    two are what quarantine.handle_hold reads, from the agent's hygiene).
    The model's own messages are skipped: they are what the hold checks, so a URL the model
    wrote in a preamble (the issuing message is already in state) or an earlier answer must not
    vouch for itself. Only the user, a tool result, an attachment or the grounding can."""
    user, seen = [], [str(state.get("attachments") or ""), str(state.get("context") or "")]
    untrusted = bool(state.get("attachments"))
    for m in state.get("messages") or []:
        text = str(getattr(m, "content", "") or "")
        if is_turn_start(m) or is_steer_message(m):
            user.append(text)
            continue
        if isinstance(m, AIMessage):
            continue
        seen.append(text)
        if isinstance(m, ToolMessage) and quarantine.is_untrusted(str(m.name or "")):
            untrusted = True
    return "\n".join(user), "\n".join(seen), untrusted


def _handle_note(tc: dict, state) -> "str | None":
    """Whose number or address a gated call names, for the prompt: the contact card that
    produced it, the user's own typing, or — past the agent's hygiene this should not happen —
    nowhere. A group chat is named with every member (tools.messages.describe_group). A bare +13057108702 at the gate (run 51, 2026-10-02) is not something a person
    can check."""
    from tools.contacts import owner_of

    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
    chat_arg = quarantine.CHAT_ARGS.get(tc.get("name"))
    ref = str((args or {}).get(chat_arg) or "").strip() if chat_arg else ""
    if ref:
        # A group reaches everyone in it: name them all, resolved from Messages now.
        from tools.messages import describe_group
        return f"{tc['name']}: {describe_group(ref)}"
    arg = quarantine.HANDLE_ARGS.get(tc.get("name"))
    handle = str((args or {}).get(arg) or "").strip() if arg else ""
    if not handle:
        return None
    kind = "address" if "@" in handle else "number"
    for m in reversed(state.get("messages") or []):
        if isinstance(m, ToolMessage) and m.name == "search_contacts":
            found = owner_of(handle, str(m.content or ""))
            if found:
                name, label = found
                return f"{tc['name']}: {handle} is {name}'s {label + ' ' if label else ''}{kind} (from search_contacts)"
    user_text, seen_text, _ = provenance(state)
    if quarantine.handle_hold(handle, user_text, "") is None:
        return f"{tc['name']}: {handle} — you typed it"
    if quarantine.handle_hold(handle, user_text, seen_text) is None:
        return None                              # from a tool result that is not a card: nothing to add
    return f"{tc['name']}: {handle} — {quarantine.UNKNOWN_HANDLE_NOTE}"


def _url_holds(tool_calls: list, state) -> dict:
    """{call id: reason} for the web_extract calls whose URL must face the human."""
    fetches = [tc for tc in tool_calls if tc.get("name") == "web_extract"]
    if not fetches:
        return {}
    prov = provenance(state)
    out = {}
    for tc in fetches:
        args = tc.get("args")
        why = quarantine.url_hold(str((args if isinstance(args, dict) else {}).get("url") or ""), *prov)
        if why:
            out[tc["id"]] = why
    return out


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
    later, so this is the single justified persisted exception to the record's
    recompute-everything design. ONE minimal, JSON-serializable shape: the same record feeds the
    headless --json "gates" field and the run export — resist letting it grow.

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
    the session (live registry.TOOL_RISK — exactly what /policy risk <tool> read_only does) and persist
    any scoped run_shell prefix grants through the one policy store (policy.grant_shell_prefix:
    screen -> coverage check with the one matcher -> add).

    The UI COLLECTS these at decision time without mutating anything (it validates with
    grant_shell_prefix(dry_run=True)); they are applied HERE, past the interrupt, because
    LangGraph re-executes this node from the top on resume and `gated` recomputes against the
    live policy — a grant applied while the interrupt was pending would auto-approve the very
    calls the human was prompted about, the re-run would return at the no-gated fast path
    without reaching the gate_event recording site, and the human's decision would vanish from
    the record (an empty gate_events must always mean "never asked"). Failures degrade safely: a
    refused shell grant just means the command faces the gate again next batch — diag-logged,
    since a node cannot print."""
    from tools import registry  # lazy, matching the UI: binds the live TOOL_RISK

    # Every grant carries a LIFETIME (policy.default_grant_scope, default "task"): a task-scoped
    # tier drop registers its own undo with the policy so the next turn starts from the declared
    # tier; persist scope reaches permissions.json; session scope simply stands until Saturn exits.
    scope = policy.default_grant_scope()
    for name in decision.get("tools") or []:
        # run_shell never drops a tier (one keypress must not un-gate every future command —
        # it gets the scoped prefix grants below instead), and neither does a shortcut run or a
        # send (policy.NO_BLANKET_GRANT). The UI never sends them here, but the resume value is
        # still external input: fail closed.
        name = str(name or "")       # before the set lookup: an unhashable entry must not raise
        if not name or name in policy.NO_BLANKET_GRANT:
            continue
        prior = registry.TOOL_RISK.get(name)
        registry.TOOL_RISK[name] = "read_only"
        # Only a tier that actually DROPPED registers an undo. A tool already at read_only is
        # either declared that way (nothing to restore) or already granted earlier this turn —
        # and in the second case `prior` is the read_only THIS grant's predecessor wrote, so a
        # second restorer would re-drop the tier the first one just restored and leave the grant
        # standing for the rest of the process while end_task() reported it expired (fail-open
        # plus a false disclosure). One `a` per tool per turn owns the undo; the rest are no-ops.
        if scope == "persist":
            try:
                policy.set_risk_override(name, "read_only")
            except Exception as exc:  # the live drop stands (this session's decision) — but say so
                diag.log(f"approval_node: tier drop for {name} could not be persisted — {exc}")
        if scope == "task" and prior != "read_only":
            def restore(_n=name, _t=prior):
                if _t is None:
                    registry.TOOL_RISK.pop(_n, None)
                else:
                    registry.TOOL_RISK[_n] = _t
                return _n
            policy.on_task_end(restore)
    for grant in decision.get("shell_grants") or []:
        if not isinstance(grant, dict):
            continue
        try:
            ok, msg = policy.grant_shell_prefix(
                str(grant.get("prefix") or ""), str(grant.get("command") or ""), scope=scope
            )
        except Exception as exc:  # resume value is external input — a grant must never kill the turn
            ok, msg = False, str(exc)
        if not ok:
            diag.log(f"approval_node: always-allow shell grant refused — {msg}")


def approval_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
    """Human-in-the-loop safety gate. Calls within the configured auto-approve tier pass
    straight through. If any pending call exceeds it, pause via `interrupt` and let the user
    decide per batch OR per call.

    The resume value is the literal True (approve the whole batch; anything else that is not one
    of the dict shapes below rejects — fail-closed),
    `{"approved_ids": [...]}` from the UI's per-call select mode, or the always-allow decision
    dict `{"approved": True, "tools": [...], "shell_grants": [...]}` whose grants are applied
    past the interrupt by `_apply_always_grants`. Rejected calls get a decline
    ToolMessage here (orphaned tool_calls break the next model turn); everything else in the
    batch still routes to `tools`, which executes only the calls that don't already have a
    ToolMessage. A fully-rejected batch routes back to `agent`."""
    last, answered = issuing_message(state["messages"])
    tool_calls = [tc for tc in (getattr(last, "tool_calls", None) or [])
                  if tc.get("id") not in answered]

    # Quarantine escalation (runtime.quarantine = gate): a previous tool result this turn carried
    # instruction-shaped content, so this batch's arguments may derive from injected text — every
    # call in it that can act (send or change something) faces the human ONCE regardless of risk
    # tier. A batch of local read-only calls (a plan update, a re-read) is not what the
    # escalation is for: it passes and leaves it armed. PEEK here, consume only after the
    # interrupt resolves: LangGraph re-executes this node from the top on resume, so a consuming
    # check would already be spent on the re-run, `gated` would recompute without the escalation,
    # and the user's rejection of the batch would be silently discarded (an all-auto-approved
    # batch would skip the interrupt entirely and run). The interrupt payload carries the flags so
    # the prompt can say why a normally-silent call is suddenly asking.
    escalated = quarantine.gate_pending() and any(_can_act(tc["name"]) for tc in tool_calls)
    # The URL hold: a read_only fetch still sends its URL (quarantine.url_hold).
    holds = _url_holds(tool_calls, state)

    gated = [
        tc
        for tc in tool_calls
        if (escalated and _can_act(tc["name"]))
        or tc["id"] in holds
        or not policy.approves(tc["name"], risk_of(tc["name"]), tc.get("args"))
    ]

    if not gated:
        return Command(goto="tools")

    # Decision context for the gate's `e(xplain)` answer: the plan step this batch is fulfilling
    # and the agent's pre-action reasoning (the text content of the tool-calling AIMessage) —
    # the same provenance /trace why reconstructs later, surfaced at the moment of decision.
    reasoning = getattr(last, "content", "") or ""
    flags = quarantine.turn_flags()
    # Why a call the tier would have let through is asking anyway — said at the prompt.
    notes = [f"{tc['name']}: {holds[tc['id']]}" for tc in gated if tc["id"] in holds]
    if any(policy.airgap_holds(tc["name"]) for tc in gated):
        notes.append(AIRGAP_NOTE)
    notes += [f"{tc['name']}: {SEND_NOTE}" for tc in gated if policy.always_asks(tc["name"])]
    notes += [n for n in (_handle_note(tc, state) for tc in gated) if n]
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
            "notes": notes or None,
            # The URL-held calls by id: the headless approver denies exactly these under --yolo.
            "held_ids": [tc["id"] for tc in gated if tc["id"] in holds],
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
    elif decision is True:
        approved_ids = set(gated_ids)
    else:
        # Fail-closed on the resume value: ONLY the literal True approves a whole batch. Anything
        # else — False, None, a stray string, an int, an unrecognized dict — is a rejection. The
        # human's approval is never inferred from truthiness.
        approved_ids = set()

    # Past the interrupt: this runs exactly once, with the human's decision in hand. The one-shot
    # escalation is spent only when the human LET SOMETHING THROUGH — a fully-rejected batch
    # leaves it armed, so a re-issued copy of the call the human just declined faces the gate
    # again instead of auto-approving right past their 'no'. (nodes/agent.py's declined-repeat
    # guard usually answers an identical call before it gets here.)
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
        quarantine=bool(escalated or holds),
        step=(step or {}).get("label"),
    )

    if approved_ids == gated_ids:
        return Command(goto="tools", update={"gate_events": [event]})

    # Decline ONLY the rejected calls (orphaned tool_calls break the next model turn). The
    # structural outcome stamp (same contract as nodes/tools.py) is what the recorder keys the
    # `skipped` status off; the DECLINE_TEXT prefix stays as belt-and-braces only.
    rejected = [tc for tc in gated if tc["id"] not in approved_ids]
    decline = [
        ToolMessage(
            content=DECLINE_TEXT,
            tool_call_id=tc["id"],
            name=tc["name"],
            additional_kwargs={"saturn_status": "skipped"},
        )
        for tc in rejected
    ]
    update = {"messages": decline, "gate_events": [event]}

    # Anything left to run (ungated or approved) still runs; a fully-rejected batch goes
    # straight back to the agent, which sees the declines (and whose declined-repeat guard
    # refuses the same call for the rest of the turn — a guarded action is never retried).
    if len(rejected) < len(tool_calls):
        return Command(goto="tools", update=update)
    return Command(goto="agent", update=update)
