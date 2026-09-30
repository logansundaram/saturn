"""The startup model warm-up (2026-09-02).

The first turn of every session paid the model LOAD inside the planner call — measured 50 s
and 37 s for "hello" (runs 2 and 8) against 15 s once warm — and read as a hung agent. The
REPL now sends one minimal request in a background thread right after the health check, so the
weights are resident before the user's first query. Fully offline here: the model is a stub.
"""

from langchain.messages import AIMessage

from app import startup
from core import llms


class _Stub:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def invoke(self, msgs, **kw):
        self.calls.append((msgs, kw))
        if self.fail:
            raise RuntimeError("daemon down")
        return AIMessage(content="ok")


def test_warm_model_sends_one_minimal_request_at_the_configured_window(monkeypatch):
    stub = _Stub()
    monkeypatch.setattr("core.llms.get_model", lambda: stub)
    assert startup.warm_model() is True
    assert len(stub.calls) == 1
    opts = stub.calls[0][1]["options"]
    # One token, at the SAME num_ctx every turn uses — Ollama keys the loaded runner on the
    # context size, so a warm-up at a different window would load a runner the turn then evicts.
    assert opts["num_predict"] == 1
    assert opts["num_ctx"] == llms.invoke_kwargs(None, 0.0)["options"]["num_ctx"]


def test_warm_model_never_raises(monkeypatch):
    lines = []
    import diag
    monkeypatch.setattr(diag, "log", lambda s: lines.append(s))
    monkeypatch.setattr("core.llms.get_model", lambda: _Stub(fail=True))
    assert startup.warm_model() is False
    assert any("warm-up" in l for l in lines)


def test_start_warm_up_runs_in_a_daemon_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(startup, "warm_model", lambda: seen.append("warm") or True)
    t = startup.start_warm_up()
    t.join(timeout=5)
    assert t.daemon and not t.is_alive() and seen == ["warm"]
