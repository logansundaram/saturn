"""budget.py — the session token-budget guard behind runtime.token_budget."""

import pytest

from core import budget
from config import get_config


@pytest.fixture(autouse=True)
def fresh_budget(monkeypatch):
    """Zero the running total and pin the knob to 'off' around every test."""
    budget.reset()
    monkeypatch.setitem(get_config()._data["runtime"], "token_budget", 0)
    yield
    budget.reset()


def _set_limit(monkeypatch, n):
    monkeypatch.setitem(get_config()._data["runtime"], "token_budget", n)


def test_disabled_by_default():
    assert budget.limit() == 0
    budget.add(10_000_000)
    assert not budget.exceeded()
    assert budget.remaining() is None


def test_enforcement_threshold(monkeypatch):
    _set_limit(monkeypatch, 100)
    budget.add(60)
    assert not budget.exceeded()
    assert budget.remaining() == 40
    assert budget.near(0.5)
    assert not budget.near(0.8)
    budget.add(40)
    assert budget.exceeded()
    assert budget.remaining() == 0
    assert budget.spent() == 100


def test_limit_read_live(monkeypatch):
    """A /config edit applies to the very next check — no caching."""
    budget.add(150)
    assert not budget.exceeded()
    _set_limit(monkeypatch, 100)
    assert budget.exceeded()
    _set_limit(monkeypatch, 1000)
    assert not budget.exceeded()


def test_garbage_inputs_ignored(monkeypatch):
    _set_limit(monkeypatch, 100)
    budget.add(None)
    budget.add("not a number")
    budget.add(-50)
    assert budget.spent() == 0
    # And a garbage limit disables instead of crashing.
    _set_limit(monkeypatch, "lots")
    assert budget.limit() == 0
    assert not budget.exceeded()
