"""typeahead.InputQueue — the queue surface only (no console polling): push/pop FIFO order and
the late-steer salvage path (agent.main pushes an unconsumed steer correction at turn end so the
user's typed words run as the next message instead of being silently dropped)."""

from core.plan_ops import PauseController
from tui.typeahead import InputQueue


def _queue():
    # A private controller so tests never touch the process-level singleton.
    return InputQueue(controller=PauseController())


def test_push_pop_fifo():
    q = _queue()
    q.push("first")
    q.push("second")
    assert q.pending()
    assert q.pop() == "first"
    assert q.pop() == "second"
    assert q.pop() is None
    assert not q.pending()


def test_push_ignores_blank_lines():
    q = _queue()
    q.push("")
    q.push("   ")
    assert q.pop() is None


def test_push_strips_whitespace():
    q = _queue()
    q.push("  steer text  \n")
    assert q.pop() == "steer text"


def test_push_notifies_on_change():
    seen = []
    q = InputQueue(on_change=lambda buf, n: seen.append(n), controller=PauseController())
    q.push("line")
    assert seen and seen[-1] == 1
    q.pop()
    assert seen[-1] == 0
