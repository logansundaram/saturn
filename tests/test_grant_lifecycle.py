"""Always-allow grant LIFETIMES (transplanted from the gating isolate; PORTICO's lingering-
authority argument): a gate `a` grant carries a scope — task (dies at the turn boundary,
the default), session (dies with the process), persist (reaches permissions.json) — and each
scope has to actually do what its name says. Before this, an `a` dropped a tool's tier for the
whole session and persisted a shell prefix forever from one keypress: the longest-lived grant
in the gate, with no expiry.
"""

import types

import pytest

from trust import policy


@pytest.fixture
def scope(monkeypatch):
    from config import get_config

    def _set(value):
        monkeypatch.setitem(get_config()._data["runtime"], "grant_scope", value)

    return _set


def _perms(isolated_paths):
    return isolated_paths / "database" / "permissions.json"


# --- shell prefix grants -------------------------------------------------------------------


def test_prefix_grant_not_persisted_by_default(isolated_paths):
    policy.add_shell_allow("git status")
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is True
    assert policy.persisted_shell_allow() == []            # what a fresh process would see
    assert policy.shell_allow_by_scope()["task"] == ["git status"]
    assert not _perms(isolated_paths).exists()             # nothing was even written


def test_prefix_grant_persists_only_when_asked(isolated_paths):
    policy.add_shell_allow("git status", scope="persist")
    assert policy.persisted_shell_allow() == ["git status"]
    policy.end_task()
    assert policy.persisted_shell_allow() == ["git status"]
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is True


def test_prefix_grant_expires_at_task_boundary(isolated_paths):
    policy.add_shell_allow("git status")
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is True
    expired = policy.end_task()
    assert expired["prefixes"] == ["git status"]
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is False


def test_begin_task_expires_a_leftover_from_an_aborted_turn(isolated_paths):
    policy.add_shell_allow("git status")
    policy.begin_task()  # a Ctrl-C'd turn never reached end_task
    assert policy.shell_allow() == []


def test_session_scoped_prefix_survives_the_task_boundary(isolated_paths):
    policy.add_shell_allow("git status", scope="session")
    policy.end_task()
    assert policy.approves("run_shell", "destructive", {"command": "git status"}) is True
    assert policy.persisted_shell_allow() == []


def test_remove_reaches_every_scope_and_indexes_the_effective_list(isolated_paths):
    policy.add_shell_allow("git status")                     # task
    policy.add_shell_allow("git log", scope="session")
    policy.add_shell_allow("ls", scope="persist")
    assert policy.shell_allow() == ["git status", "git log", "ls"]
    assert policy.remove_shell_allow("2") == "git log"
    assert policy.remove_shell_allow("LS") == "ls"
    assert policy.remove_shell_allow("git status") == "git status"
    assert policy.shell_allow() == [] and policy.persisted_shell_allow() == []


def test_unknown_scope_falls_back_to_the_shortest_lifetime(isolated_paths, scope):
    scope("forever")
    assert policy.default_grant_scope() == "task"
    policy.add_shell_allow("git status", scope="bogus")
    assert policy.shell_allow_by_scope()["task"] == ["git status"]


def test_grant_shell_prefix_discloses_the_lifetime(isolated_paths, scope):
    ok, msg = policy.grant_shell_prefix("git status", "git status")
    assert ok and "end of this turn" in msg
    scope("persist")
    ok, msg = policy.grant_shell_prefix("git log", "git log")
    assert ok and "permissions.json" in msg
    assert policy.persisted_shell_allow() == ["git log"]


# --- tier drops through the approval node -------------------------------------------------


def _registry(monkeypatch, **risk):
    fake = types.SimpleNamespace(TOOL_RISK=dict(risk))
    monkeypatch.setattr("tools.registry", fake, raising=False)
    return fake


def test_tier_drop_expires_at_task_boundary(isolated_paths, monkeypatch):
    from nodes.approval import _apply_always_grants

    reg = _registry(monkeypatch, write_file="side_effecting")
    _apply_always_grants({"approved": True, "tools": ["write_file"], "shell_grants": []})
    assert reg.TOOL_RISK["write_file"] == "read_only"       # granted this task
    expired = policy.end_task()                              # ---- task boundary ----
    assert expired["tools"] == ["write_file"]
    assert reg.TOOL_RISK["write_file"] == "side_effecting"   # restored
    assert policy.risk_overrides() == {}                     # never touched the file


def test_session_scoped_tier_drop_survives_the_task_boundary(isolated_paths, monkeypatch, scope):
    from nodes.approval import _apply_always_grants

    scope("session")
    reg = _registry(monkeypatch, write_file="side_effecting")
    _apply_always_grants({"approved": True, "tools": ["write_file"], "shell_grants": []})
    policy.end_task()
    assert reg.TOOL_RISK["write_file"] == "read_only"       # deliberately still relaxed
    assert policy.risk_overrides() == {}


def test_persist_scoped_tier_drop_reaches_the_permissions_file(isolated_paths, monkeypatch, scope):
    from nodes.approval import _apply_always_grants

    scope("persist")
    reg = _registry(monkeypatch, write_file="side_effecting")
    _apply_always_grants({"approved": True, "tools": ["write_file"], "shell_grants": []})
    policy.end_task()
    assert reg.TOOL_RISK["write_file"] == "read_only"
    assert policy.risk_overrides() == {"write_file": "read_only"}   # a new process inherits it


def test_grant_log_records_grant_and_expiry(isolated_paths):
    policy.add_shell_allow("git status")
    policy.end_task()
    assert [e["event"] for e in policy.grant_log()] == ["grant", "expire"]


def test_grant_scope_is_a_trust_key():
    from commands.config import _TRUST_KEYS

    assert "runtime.grant_scope" in _TRUST_KEYS


def test_fresh_turn_opens_a_task(isolated_paths):
    """The REPL/headless per-turn reset is the task boundary's opening half."""
    from app.session import _fresh_turn, _initial_state

    policy.add_shell_allow("git status")
    _fresh_turn(_initial_state(), "hello")
    assert policy.shell_allow() == []


def test_a_failing_restorer_is_named_never_swallowed(isolated_paths):
    """A tier drop whose undo fails would be lingering authority: end_task must NAME it (the REPL
    warns and points at /policy risk reset) instead of swallowing the failure."""
    def bad():
        raise RuntimeError("registry gone")

    policy.on_task_end(bad)
    out = policy.end_task()
    assert out["failed"] and "registry gone" in out["failed"][0]
    assert policy.grant_log()[-1]["failed"]
