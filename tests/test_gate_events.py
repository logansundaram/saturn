"""
Structured human-gate records (state["gate_events"]) — the chain-of-custody piece this wave
added: the approval node's per-prompt event shape (approve-all / reject-all / partial), the
auto-approved-silence rule (no prompt -> no record), the headless --json "gates" derivation,
and /trace why's always-on verification section (the negative case must print, matching the
Glass Box). Offline: the LangGraph interrupt is stubbed; no LLM/graph/network runs.
"""

import json
import types

from trust import quarantine
from nodes import approval as approval_mod
from nodes.approval import approval_node, gate_event
from core.state import summarize_gates


# --- the event builder: ONE shape for --json, the Glass Box, and the signed export -----------

_CALLS = [
    {"id": "c1", "name": "run_shell", "args": {"command": "git status"}},
    {"id": "c2", "name": "write_file", "args": {"path": "a.txt", "content": "x"}},
]


def test_gate_event_approve_all():
    ev = gate_event(_CALLS, {"c1", "c2"})
    assert ev["decision"] == "approved"
    assert ev["calls"] == [
        {"id": "c1", "name": "run_shell", "approved": True},
        {"id": "c2", "name": "write_file", "approved": True},
    ]
    assert ev["quarantine"] is False and ev["step"] is None


def test_gate_event_reject_all():
    ev = gate_event(_CALLS, set())
    assert ev["decision"] == "rejected"
    assert all(not c["approved"] for c in ev["calls"])


def test_gate_event_partial_and_context_fields():
    ev = gate_event(
        _CALLS,
        {"c2"},
        quarantine=True,
        step="apply the fix",
    )
    assert ev["decision"] == "partial"
    assert [c["approved"] for c in ev["calls"]] == [False, True]
    assert ev["quarantine"] is True
    assert ev["step"] == "apply the fix"


def test_gate_event_is_json_serializable():
    # The record rides the trace delta and the --json payload — it must be plain JSON all the
    # way down (gotcha #4: nothing Pydantic, nothing custom).
    ev = gate_event(_CALLS, {"c1"})
    assert json.loads(json.dumps(ev)) == ev


# --- the approval node records exactly one event per PROMPT ----------------------------------


def _msg_with_calls(calls):
    return types.SimpleNamespace(content="about to act", tool_calls=calls)


def _node_state(calls, plan=None):
    return {
        "messages": [_msg_with_calls(calls)],
        "plan": plan or [],
        "tools_called": [],
    }


def _gate_everything(monkeypatch, decision):
    """Force every call to face the (stubbed) human, who answers `decision`."""
    quarantine.reset_turn()
    monkeypatch.setattr(approval_mod.policy, "approves", lambda *a, **k: False)
    monkeypatch.setattr(approval_mod, "interrupt", lambda payload: decision)


def test_node_approve_all_records_event(monkeypatch):
    _gate_everything(monkeypatch, True)
    plan = [{"step_id": 1, "label": "do the work", "status": "pending",
             "intended_tool": "run_shell"}]
    cmd = approval_node(_node_state(list(_CALLS), plan=plan))
    assert cmd.goto == "tools"
    (ev,) = cmd.update["gate_events"]
    assert ev["decision"] == "approved"
    assert [c["name"] for c in ev["calls"]] == ["run_shell", "write_file"]
    assert all(c["approved"] for c in ev["calls"])
    assert ev["step"] == "do the work"


def test_node_reject_all_records_event(monkeypatch):
    _gate_everything(monkeypatch, False)
    cmd = approval_node(_node_state(list(_CALLS)))
    # A fully-rejected batch goes to the recorder: the decline lands on the current step as a
    # `skipped` incident, and rectify retires the remaining plan.
    assert cmd.goto == "update_plan"
    (ev,) = cmd.update["gate_events"]
    assert ev["decision"] == "rejected"
    assert all(not c["approved"] for c in ev["calls"])
    assert ev["step"] is None  # no plan — no active-step label, recorded as unknown not ""
    # The decline ToolMessages still ride the same update (unchanged behavior).
    assert len(cmd.update["messages"]) == 2


def test_node_partial_records_event(monkeypatch):
    _gate_everything(monkeypatch, {"approved_ids": ["c1"]})
    cmd = approval_node(_node_state(list(_CALLS)))
    assert cmd.goto == "tools"  # survivors still run
    (ev,) = cmd.update["gate_events"]
    assert ev["decision"] == "partial"
    assert {c["name"]: c["approved"] for c in ev["calls"]} == {
        "run_shell": True, "write_file": False,
    }


def test_node_always_decision_records_event_and_applies_grants(monkeypatch, isolated_paths):
    # The always-allow answer resumes with the COLLECTED grants dict: the UI mutated nothing
    # while the interrupt was pending (so the re-run still sees the batch gated and this event
    # records — gotcha #7), and the node applies the grants here, past the interrupt.
    fake_registry = types.SimpleNamespace(TOOL_RISK={})
    monkeypatch.setattr("tools.registry", fake_registry, raising=False)
    decision = {
        "approved": True,
        "tools": ["write_file"],
        "shell_grants": [{"prefix": "git status", "command": "git status"}],
    }
    _gate_everything(monkeypatch, decision)
    cmd = approval_node(_node_state(list(_CALLS)))
    assert cmd.goto == "tools"
    (ev,) = cmd.update["gate_events"]
    assert ev["decision"] == "approved"  # the human's `a` is in the record
    # …and the grants applied: the tier drop in the live registry, the prefix in the one store.
    assert fake_registry.TOOL_RISK == {"write_file": "read_only"}
    from trust import policy

    assert policy.shell_allow() == ["git status"]


def test_node_auto_approved_batch_records_nothing(monkeypatch):
    # No prompt -> no record: "gate_events empty" must always mean "the human was never asked".
    quarantine.reset_turn()
    monkeypatch.setattr(approval_mod.policy, "approves", lambda *a, **k: True)
    monkeypatch.setattr(
        approval_mod, "interrupt",
        lambda payload: (_ for _ in ()).throw(AssertionError("gate must not prompt")),
    )
    cmd = approval_node(_node_state(list(_CALLS)))
    assert cmd.goto == "tools"
    assert not cmd.update  # no gate_events delta at all


def test_node_quarantine_escalation_flag_recorded(monkeypatch):
    _gate_everything(monkeypatch, True)
    monkeypatch.setattr(quarantine, "gate_pending", lambda: True)
    consumed = []
    monkeypatch.setattr(quarantine, "consume_gate", lambda: consumed.append(True))
    cmd = approval_node(_node_state(list(_CALLS)))
    (ev,) = cmd.update["gate_events"]
    assert ev["quarantine"] is True
    assert consumed  # the approval spent the one-shot escalation (unchanged behavior)


# --- headless --json: the "gates" field derives from the state dict --------------------------


def test_summarize_gates_counts_and_denied_names():
    events = [
        gate_event(_CALLS, {"c1"}),                       # partial: write_file denied
        gate_event([{"id": "c3", "name": "http_request", "args": {}}], set()),  # rejected
    ]
    assert summarize_gates(events) == {
        "prompted": 3,
        "denied": ["write_file", "http_request"],
    }


def test_summarize_gates_empty_and_garbage_tolerant():
    assert summarize_gates([]) == {"prompted": 0, "denied": []}
    assert summarize_gates(None) == {"prompted": 0, "denied": []}
    assert summarize_gates([None, "junk", {"calls": [None, 7]}]) == {
        "prompted": 0, "denied": [],
    }


# --- /trace why: the verification section always prints --------------------------------------


def test_why_verification_prints_negative_case(capsys):
    from commands.trace import _render_why
    from tui import ui

    run = (3, "what is new?", None, None, "ok", "an answer")
    _render_why(ui, run, [], [])
    out = capsys.readouterr().out
    assert "verification" in out
    assert "rectify judge did not run — every step resolved mechanically." in out


def test_why_verification_prints_rectify_verdict(capsys):
    from commands.trace import _render_why
    from tui import ui

    run = (4, "q", None, None, "ok", "an answer")
    calls = [(1, "rectify", json.dumps(
        {"content": '{"reasoning":"never looked it up","rectify":true}'}))]
    _render_why(ui, run, [], calls)
    out = capsys.readouterr().out
    assert "rectify:" in out and "never looked it up" in out
    assert "did not run" not in out
