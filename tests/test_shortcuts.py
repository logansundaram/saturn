"""
Shortcuts (`list_shortcuts` / `run_shortcut`) and the trust wiring a shortcut needs: it is a
process the egress ledger cannot see inside — recorded UNTRACKED, held under air-gap like
`run_shell` — and its gate is relaxed one shortcut at a time (`/policy shortcut`), never by a
blanket always-allow.

Offline: the `shortcuts` CLI never runs (`tools.shortcuts._run` is the process seam).
"""

import subprocess

import pytest

from commands._framework import dispatch
from tools import applescript
from trust import egress, policy


@pytest.fixture
def cli(monkeypatch):
    """Pin macOS and capture `shortcuts` invocations; `.names` is what `shortcuts list` prints,
    `.output` what a run prints."""
    from tools import shortcuts

    class Ctl:
        calls: list = []
        inputs: list = []
        names = ["Lights Off", "Log Water", "Morning Focus"]
        output = ""
        returncode = 0
        stderr = ""

    ctl = Ctl()

    def fake_run(argv, timeout):
        ctl.calls.append(list(argv))
        if argv[:2] == ["shortcuts", "list"]:
            return subprocess.CompletedProcess(argv, 0, "\n".join(ctl.names) + "\n", "")
        if "-i" in argv:
            with open(argv[argv.index("-i") + 1], encoding="utf-8") as fh:
                ctl.inputs.append(fh.read())
        return subprocess.CompletedProcess(argv, ctl.returncode, ctl.output, ctl.stderr)

    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    monkeypatch.setattr(shortcuts, "_run", fake_run)
    return ctl


@pytest.fixture
def gate(isolated_paths, monkeypatch):
    from config import get_config

    runtime = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "auto_approve", "read_only")
    monkeypatch.setitem(runtime, "airgap", False)
    monkeypatch.setattr(policy, "_tier_before_gate_off", None)
    return get_config()


def _tool(name):
    from tools.registry import tools_by_name
    return tools_by_name[name]


def _err(name, args):
    from tools.toolspec import ToolError

    with pytest.raises(ToolError) as info:
        _tool(name).invoke(args)
    return f"Error: {info.value}"


# ── the tools ────────────────────────────────────────────────────────────────────────────────

def test_shortcut_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("list_shortcuts") == "read_only" and "list_shortcuts" not in _UNTRUSTED
    # A shortcut can do anything its owner built it to do — and print anything it fetched.
    assert risk_of("run_shortcut") == "destructive" and "run_shortcut" in _UNTRUSTED


def test_list_shortcuts_returns_the_names(cli):
    assert _tool("list_shortcuts").invoke({}) == ["Lights Off", "Log Water", "Morning Focus"]
    assert cli.calls == [["shortcuts", "list"]]


def test_list_shortcuts_when_there_are_none(cli):
    cli.names = []
    assert _tool("list_shortcuts").invoke({}) == "No shortcuts on this Mac."


def test_run_shortcut_runs_it_by_its_exact_name_and_returns_the_output(cli):
    cli.output = "Living room: off\n"
    out = _tool("run_shortcut").invoke({"name": "lights off"})       # case-insensitive match
    assert out == {"shortcut": "Lights Off", "output": "Living room: off"}
    assert cli.calls[-1] == ["shortcuts", "run", "Lights Off"]


def test_run_shortcut_without_output_says_it_ran(cli):
    out = _tool("run_shortcut").invoke({"name": "Log Water"})
    assert out == {"shortcut": "Log Water", "output": "(the shortcut ran and returned nothing)"}


def test_run_shortcut_passes_input_through_a_file(cli):
    _tool("run_shortcut").invoke({"name": "Log Water", "input": "250 ml"})
    assert cli.inputs == ["250 ml"]
    assert cli.calls[-1][:3] == ["shortcuts", "run", "Log Water"] and "-i" in cli.calls[-1]


def test_run_shortcut_unknown_name_lists_the_real_ones_and_runs_nothing(cli):
    out = _err("run_shortcut", {"name": "Lights On"})
    assert "no shortcut named 'Lights On'" in out and "Lights Off, Log Water, Morning Focus" in out
    assert all(c[:2] == ["shortcuts", "list"] for c in cli.calls)


def test_run_shortcut_failure_and_timeout_are_errors(cli, monkeypatch):
    from tools import shortcuts
    cli.returncode, cli.stderr = 1, "Error: The operation couldn’t be completed."
    assert "couldn’t be completed" in _err("run_shortcut", {"name": "Log Water"})

    def slow(argv, timeout):
        if argv[:2] == ["shortcuts", "list"]:
            return subprocess.CompletedProcess(argv, 0, "Log Water\n", "")
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(shortcuts, "_run", slow)
    out = _err("run_shortcut", {"name": "Log Water"})
    assert "did not finish" in out and "may still be running" in out


def test_shortcut_tools_report_non_mac_honestly(monkeypatch):
    from tools import shortcuts
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert "only available on macOS" in _err("list_shortcuts", {})
    assert "only available on macOS" in _err("run_shortcut", {"name": "x"})


# ── the ledger and the air-gap ───────────────────────────────────────────────────────────────

def test_a_shortcut_run_is_on_the_ledger_as_untracked(cli):
    mark = egress.next_seq()
    _tool("run_shortcut").invoke({"name": "Lights Off"})
    events = egress.events_since(mark)
    assert [(e.channel, e.detail, e.status) for e in events] == [("shortcut", "Lights Off", egress.UNTRACKED)]


def test_a_refused_shortcut_leaves_no_ledger_entry(cli):
    mark = egress.next_seq()
    _err("run_shortcut", {"name": "Nope"})
    assert egress.events_since(mark) == []


def test_airgap_holds_a_shortcut_for_the_human(gate, monkeypatch):
    policy.add_shortcut_allow("Lights Off")
    assert policy.approves("run_shortcut", "destructive", {"name": "Lights Off"})
    monkeypatch.setitem(gate._data["runtime"], "airgap", True)
    assert policy.airgap_holds("run_shortcut")
    assert not policy.approves("run_shortcut", "destructive", {"name": "Lights Off"})


# ── the per-shortcut allowlist ───────────────────────────────────────────────────────────────

def test_a_shortcut_skips_the_gate_only_when_it_is_allowlisted_by_name(gate):
    assert not policy.approves("run_shortcut", "destructive", {"name": "Lights Off"})
    assert policy.add_shortcut_allow("Lights Off") is True
    assert policy.add_shortcut_allow("lights off") is False              # already there
    assert policy.approves("run_shortcut", "destructive", {"name": "lights off"})
    assert not policy.approves("run_shortcut", "destructive", {"name": "Lights Off 2"})
    assert not policy.approves("run_shortcut", "destructive", {"name": ""})
    assert policy.shortcut_allow() == ["Lights Off"]
    assert policy.remove_shortcut_allow("LIGHTS OFF") == "Lights Off"
    assert not policy.approves("run_shortcut", "destructive", {"name": "Lights Off"})
    assert policy.remove_shortcut_allow("Lights Off") is None


def test_the_shortcut_allowlist_survives_a_restart_and_a_garbled_file_fails_closed(gate):
    import json
    policy.add_shortcut_allow("Log Water")
    path = gate.path("permissions")
    assert json.loads(path.read_text(encoding="utf-8"))["shortcut_allow"] == ["Log Water"]
    path.write_text(json.dumps({"shortcut_allow": "Log Water"}), encoding="utf-8")   # a string, not a list
    monkeypatch_problem = policy._LOAD_PROBLEM
    try:
        policy._LOAD_PROBLEM = None
        assert policy.shortcut_allow() == []
        assert not policy.approves("run_shortcut", "destructive", {"name": "L"})
    finally:
        policy._LOAD_PROBLEM = monkeypatch_problem


def test_policy_shortcut_command_adds_lists_and_removes(cli, gate, ctx, capsys):
    dispatch("/policy shortcut", ctx)
    assert "no allowlisted shortcuts" in capsys.readouterr().out
    dispatch("/policy shortcut Lights Off", ctx)
    assert policy.shortcut_allow() == ["Lights Off"]
    assert "Lights Off" in capsys.readouterr().out
    dispatch("/policy shortcut list", ctx)
    assert "Lights Off" in capsys.readouterr().out
    dispatch("/policy shortcut remove Lights Off", ctx)
    assert policy.shortcut_allow() == []
    dispatch("/policy shortcut remove Nope", ctx)
    assert "no such" in capsys.readouterr().out


def test_policy_shortcut_allowlists_only_a_shortcut_that_exists(cli, gate, ctx, capsys):
    # A verb with no name is a typo, not a shortcut called "remove"; and the name must be one
    # of this Mac's shortcuts, stored as the Shortcuts app spells it.
    for line in ("/policy shortcut remove", "/policy shortcut add"):
        dispatch(line, ctx)
        assert "needs the shortcut's name" in capsys.readouterr().out
    dispatch("/policy shortcut Lights Of", ctx)
    out = capsys.readouterr().out
    assert "no shortcut named" in out and "Lights Off" in out
    assert policy.shortcut_allow() == []
    dispatch("/policy shortcut lights off", ctx)
    assert policy.shortcut_allow() == ["Lights Off"]
    cli.names.append("Remove")                                 # a shortcut really called that
    dispatch("/policy shortcut add Remove", ctx)
    assert policy.shortcut_allow() == ["Lights Off", "Remove"]


def test_policy_shortcut_adds_nothing_when_the_names_cannot_be_checked(gate, ctx, capsys, monkeypatch):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    dispatch("/policy shortcut Lights Off", ctx)
    assert "only available on macOS" in capsys.readouterr().out
    assert policy.shortcut_allow() == []


def test_always_allow_at_the_gate_never_drops_the_shortcut_tier(gate):
    """One `a` on run_shortcut must not un-gate every shortcut: no tier drop is collected for
    it, and the approval node refuses one that arrives anyway."""
    from nodes.approval import _apply_always_grants
    from tools import registry
    from tui.ui import approval

    calls = [{"id": "c1", "name": "run_shortcut", "args": {"name": "Lights Off"}},
             {"id": "c2", "name": "create_note", "args": {"title": "t"}}]
    decision = approval._always_allow(calls, lambda _prompt: "")
    assert decision["tools"] == ["create_note"]
    before = registry.TOOL_RISK["run_shortcut"]
    _apply_always_grants({"approved": True, "tools": ["run_shortcut"], "shell_grants": []})
    assert registry.TOOL_RISK["run_shortcut"] == before == "destructive"
