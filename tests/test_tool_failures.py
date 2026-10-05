"""A tool call that did not do its job is stamped `error` on its ToolMessage — the stamp the
think decision (`core.think.step_kind`) and the answer's incidents note read. Tools report
failure by RAISING (read_file's not-found convention); a failure returned as a plain string used
to be stamped `done`, so the user was never told the edit or the event did not happen."""

import sys

import pytest
from langchain.messages import AIMessage, HumanMessage

from core import workspace


def _run(name, args):
    from nodes.tools import tool_node

    call = {"name": name, "args": args, "id": "c1", "type": "tool_call"}
    out = tool_node({"messages": [HumanMessage(content="q"), AIMessage(content="", tool_calls=[call])]})
    (msg,) = out["messages"]
    return msg.additional_kwargs["saturn_status"], str(msg.content), out["tool_events"][0]["ok"]


@pytest.fixture
def ws(isolated_paths, tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    workspace.set_root(root)
    (root / "a.txt").write_text("one two", encoding="utf-8")
    return root


@pytest.mark.parametrize("name,args,why", [
    ("edit_file", {"file_path": "a.txt", "old_string": "zzz", "new_string": "x"}, "not found"),
    ("edit_file", {"file_path": "a.txt", "old_string": "o", "new_string": "x"}, "appears 2 times"),
    ("edit_file", {"file_path": "a.txt", "old_string": "one", "new_string": "one"}, "identical"),
    ("edit_file", {"file_path": "nope.txt", "old_string": "a", "new_string": "b"}, "File not found"),
    ("write_file", {"file_path": "/etc/x.txt", "content": "x"}, "Outside"),
    ("list_directory", {"directory": "missing"}, "not a directory"),
    ("search_files", {"pattern": "("}, "Invalid regular expression"),
    ("calculate", {"expression": "1/0"}, "division by zero"),
    ("run_shell", {"command": f'"{sys.executable}" -c "import sys; sys.exit(3)"'}, "exit code 3"),
])
def test_a_failed_call_is_stamped_error(ws, name, args, why):
    status, content, ok = _run(name, args)
    assert status == "error" and not ok, content
    assert why in content


def test_a_successful_call_is_stamped_done(ws):
    status, content, ok = _run("edit_file", {"file_path": "a.txt", "old_string": "one", "new_string": "1"})
    assert status == "done" and ok, content
    assert (ws / "a.txt").read_text() == "1 two"
