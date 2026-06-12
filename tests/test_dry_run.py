"""Dry-run mode — tool_node stubs every call (no execution) when runtime.dry_run is on."""

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from config import get_config
from nodes.tools import tool_node


@pytest.fixture(autouse=True)
def reset_dry_run(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "dry_run", False)
    yield


def _state_with_call(name, args):
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": name, "args": args}],
    )
    return {"messages": [HumanMessage(content="do it"), ai]}


def test_dry_run_stubs_instead_of_executing(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "dry_run", True)
    # An unknown tool would normally come back as an error; under dry-run it must be stubbed
    # WITHOUT ever attempting a lookup/execution.
    out = tool_node(_state_with_call("delete_everything", {"path": "/"}))
    obs = out["messages"][0].content
    assert "[DRY RUN]" in obs
    assert "delete_everything" in obs
    assert "not run" in obs.lower()
    # The plan still advances mechanically: the call is recorded as "called".
    assert out["tools_called"] == ["delete_everything"]
    # And the per-call event is marked ok (it's a successful no-op, not a failure).
    assert out["tool_events"][0]["ok"] is True


def test_real_mode_executes(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "dry_run", False)
    out = tool_node(_state_with_call("calculate", {"expression": "2 + 2"}))
    obs = out["messages"][0].content
    assert "[DRY RUN]" not in obs
    assert "4" in obs


def test_dry_run_attaches_toolmessage_for_each_call(monkeypatch):
    monkeypatch.setitem(get_config()._data["runtime"], "dry_run", True)
    out = tool_node(_state_with_call("run_shell", {"command": "rm -rf /"}))
    assert isinstance(out["messages"][0], ToolMessage)
    assert out["messages"][0].tool_call_id == "c1"
