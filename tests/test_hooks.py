"""Hooks (pivot #10, 2026-09-29): the user's shell commands on turn-start, turn-end,
before-write and after-write, from $SATURN_HOME/hooks.yaml. A before-write hook that exits
non-zero blocks the write; the file tools never write the hooks file itself. Offline: the
hooks are tiny shell commands writing into tmp_path."""

import json

import pytest

from core import hooks, workspace
from tools.files import edit_file, write_file


@pytest.fixture
def home(tmp_path, monkeypatch, isolated_paths):
    h = tmp_path / "saturn_home"
    h.mkdir()
    monkeypatch.setenv("SATURN_HOME", str(h))
    work = tmp_path / "work"
    work.mkdir()  # set_root falls back to HOME for a folder that doesn't exist
    assert workspace.set_root(work) == work.resolve()
    return h


def _hooks(home, text):
    (home / "hooks.yaml").write_text(text, encoding="utf-8")


def test_no_file_means_no_hooks(home):
    assert hooks.load() == {} and hooks.run("turn-start", query="q") == []
    assert hooks.before_write("x.txt", "write_file") is None


def test_load_accepts_strings_and_mappings_and_skips_junk(home):
    _hooks(home, "turn-start:\n  - echo a\n  - command: echo b\n    timeout: 2\n"
                 "  - {timeout: 3}\nbogus-event:\n  - echo c\n")
    assert hooks.load() == {"turn-start": [{"command": "echo a", "timeout": 10.0},
                                           {"command": "echo b", "timeout": 2.0}]}


def test_problems_name_what_was_skipped(home):
    _hooks(home, 'turn-start:\n  - echo "start: $SATURN_QUERY"\n  - \'echo "ok: quoted"\'\nbogus: [x]\n')
    assert [h["command"] for h in hooks.load()["turn-start"]] == ['echo "ok: quoted"']
    issues = hooks.problems()
    assert len(issues) == 2
    assert "turn-start entry skipped" in issues[0] and "must be quoted" in issues[0]
    assert "unknown event 'bogus'" in issues[1]
    _hooks(home, "turn-start: [unclosed\n")
    assert hooks.problems()[0].startswith("unreadable, no hooks run:")


def test_a_broken_file_is_no_hooks(home):
    _hooks(home, "turn-start: [unclosed\n")
    assert hooks.load() == {}
    _hooks(home, "- just a list\n")
    assert hooks.load() == {}


def test_run_passes_details_in_env_and_stdin(home, tmp_path):
    out = tmp_path / "out"
    _hooks(home, f'turn-end:\n  - printf "%s|%s|" "$SATURN_HOOK_EVENT" "$SATURN_QUERY" > {out}; cat >> {out}\n')
    [r] = hooks.run("turn-end", query="what now", answer="done")
    assert r.code == 0
    event, query, stdin = out.read_text().split("|", 2)
    assert (event, query) == ("turn-end", "what now")
    payload = json.loads(stdin)
    assert payload["answer"] == "done" and payload["event"] == "turn-end"
    assert payload["workspace"] == str(workspace.root())


def test_a_hook_that_hangs_is_cut_off(home):
    _hooks(home, "turn-start:\n  - command: sleep 5\n    timeout: 0.2\n")
    [r] = hooks.run("turn-start")
    assert r.code == -1 and "timed out" in r.output


def test_before_write_can_block_a_write(home):
    _hooks(home, 'before-write:\n  - echo "drafts only" >&2; exit 1\n')
    with pytest.raises(PermissionError) as exc:  # a failed step: the incidents note discloses it
        write_file.invoke({"file_path": "a.txt", "content": "x"})
    assert str(exc.value).startswith("Blocked by your before-write hook") and "drafts only" in str(exc.value)
    assert not (workspace.root() / "a.txt").exists()


def test_write_hooks_see_the_file_and_after_write_fires_once_written(home, tmp_path):
    log = tmp_path / "log"
    _hooks(home, f'before-write:\n  - echo "before $SATURN_TOOL $SATURN_FILE" >> {log}\n'
                 f'after-write:\n  - echo "after $SATURN_TOOL" >> {log}\n')
    assert write_file.invoke({"file_path": "a.txt", "content": "one"}) == "File created successfully"
    assert edit_file.invoke({"file_path": "a.txt", "old_string": "one", "new_string": "two"}).startswith("Edited ")
    target = workspace.root().resolve() / "a.txt"
    assert log.read_text().splitlines() == [
        f"before write_file {target}", "after write_file",
        f"before edit_file {target}", "after edit_file"]


def test_a_refused_edit_fires_no_hook(home, tmp_path):
    log = tmp_path / "log"
    _hooks(home, f"before-write:\n  - echo hit >> {log}\n")
    (workspace.root() / "a.txt").write_text("one", encoding="utf-8")
    edit_file.invoke({"file_path": "a.txt", "old_string": "zzz", "new_string": "two"})
    assert not log.exists()


def test_the_file_tools_never_write_the_hooks_file(home):
    workspace.add(home)
    _hooks(home, "turn-start:\n  - echo hi\n")
    for tool, args in ((write_file, {"file_path": str(home / "hooks.yaml"), "content": "turn-start: [rm -rf ~]"}),
                       (edit_file, {"file_path": str(home / "hooks.yaml"), "old_string": "echo hi",
                                    "new_string": "rm -rf ~"})):
        with pytest.raises(PermissionError, match="Saturn never writes it"):
            tool.invoke(args)
    assert (home / "hooks.yaml").read_text() == "turn-start:\n  - echo hi\n"


def test_the_file_tools_never_write_the_gates_own_files(home, tmp_path, monkeypatch):
    """config.yaml (auto_approve) and permissions.json (saved always-allows) loosen the gate —
    launched from ~ they are inside the workspace, and a write stays refused even approved."""
    import config
    from config import get_config

    fake_config = tmp_path / "data" / "config.yaml"  # never the real one, even if the guard broke
    fake_config.parent.mkdir()
    fake_config.write_text("runtime:\n  auto_approve: read_only\n")
    monkeypatch.setattr(config, "config_path", lambda: fake_config)
    perms = get_config().path("permissions")
    perms.parent.mkdir(parents=True, exist_ok=True)
    perms.write_text("{}")
    workspace.add(fake_config.parent)
    workspace.add(perms.parent)
    for target, why in ((fake_config, "approval gate's included"), (perms, "saved permissions")):
        with pytest.raises(PermissionError, match=why):
            write_file.invoke({"file_path": str(target), "content": "auto_approve: destructive"})
    assert fake_config.read_text() == "runtime:\n  auto_approve: read_only\n"
    assert perms.read_text() == "{}"


def test_run_turn_fires_turn_start_and_turn_end(home, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from langchain.messages import AIMessage

    from app import turn

    log = tmp_path / "log"
    _hooks(home, f'turn-start:\n  - echo "start $SATURN_QUERY" >> {log}\n'
                 f'turn-end:\n  - echo "end $SATURN_ANSWER" >> {log}\n')

    class Graph:
        def stream(self, *_a, **_k):
            return iter(())

        def get_state(self, _config):
            return SimpleNamespace(next=(), values={"messages": [AIMessage(content="42")]})

    turn.run_turn(Graph(), {"current_query": "meaning?"}, {}, approver=lambda v: True)
    assert log.read_text().splitlines() == ["start meaning?", "end 42"]
