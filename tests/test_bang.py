"""`!command` at the prompt (app/bang.py): the user's own shell command, its output attached to
the next message. Offline; runs only the shell builtins it names."""

from app import bang


def test_is_bang_needs_a_command():
    assert bang.is_bang("!ls -la") and bang.is_bang("  !git status")
    for plain in ("!", "!!", "!?", "hello!", "", "  ", "/help"):
        assert not bang.is_bang(plain), plain
    assert bang.command_of("  !git status ") == "git status"


def test_run_returns_output_and_exit_status(tmp_path):
    out, code = bang.run("printf hi; printf err >&2; exit 3", cwd=str(tmp_path))
    assert out == "hi\nerr" and code == 3
    out, code = bang.run("pwd", cwd=str(tmp_path))
    assert out.rstrip("/").endswith(tmp_path.name) and code == 0


def test_run_reports_a_timeout_instead_of_raising(monkeypatch):
    monkeypatch.setattr(bang, "TIMEOUT", 0.2)
    out, code = bang.run("sleep 5")
    assert "timed out" in out and code == 124


def test_attachment_is_fenced_labelled_and_clamped():
    block = bang.attachment("git diff", "x" * 20_000, 0)
    assert block.startswith("### Shell output") and "`!git diff`" in block and "exit status 0" in block
    assert "truncated" in block and block.endswith("```")
    assert len(block) < 13_000
