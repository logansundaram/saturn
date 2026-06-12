"""
TUI polish helpers — the one-time discovery-hint sentinels (receipt.take_hint), the empty-Esc
pause acknowledgement path (typeahead on_pause), the posture-at-the-prompt derivation
(tui.ui.prompt._posture_flags), and the status bar's trailing key legend. All pure/offline.

NOTE: `tui.ui.prompt` / `tui.ui.response` the ATTRIBUTES are functions (the package re-exports
them flat), so the submodules are reached via importlib.import_module, never `from tui.ui import`.
"""

import importlib

import pytest

from trust import receipt
from core.plan_ops import PauseController
from tui.typeahead import InputQueue


# --- one-time discovery hints (receipt.take_hint) ----------------------------------------------

def test_take_hint_once_per_install(isolated_paths):
    receipt._HINTS_SHOWN.clear()
    try:
        assert receipt.take_hint("polish_test") is True
        assert receipt.take_hint("polish_test") is False  # same session: consumed
        receipt._HINTS_SHOWN.clear()  # simulate a fresh process — the sentinel must hold
        assert receipt.take_hint("polish_test") is False
        assert (isolated_paths / "database" / ".hint_polish_test").exists()
        assert receipt.take_hint("polish_other") is True  # names are independent
    finally:
        receipt._HINTS_SHOWN.clear()


def test_take_hint_unwritable_falls_back_to_once_per_session(monkeypatch):
    # An unwritable/broken sentinel dir must never crash or repeat the hint every answer —
    # the in-memory set still bounds it to once per session.
    receipt._HINTS_SHOWN.clear()
    try:
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr(receipt, "get_config", _boom)
        assert receipt.take_hint("polish_failsafe") is True
        assert receipt.take_hint("polish_failsafe") is False
    finally:
        receipt._HINTS_SHOWN.clear()


# --- empty-Esc acknowledgement (typeahead on_pause, symmetric to on_steer) ----------------------

def test_empty_escape_requests_pause_and_fires_on_pause():
    fired = []
    c = PauseController()
    q = InputQueue(on_pause=lambda: fired.append(True), controller=c)
    q._on_escape()
    assert fired == [True]
    req = c.peek()
    assert req is not None and req.source == "user"


def test_escape_with_text_steers_and_does_not_fire_on_pause():
    steered, paused = [], []
    c = PauseController()
    q = InputQueue(on_steer=steered.append, on_pause=lambda: paused.append(True), controller=c)
    q._buffer = "go deeper"
    q._on_escape()
    assert steered == ["go deeper"]
    assert paused == []
    assert c.peek().source == "steer"


def test_on_pause_errors_never_propagate():
    c = PauseController()
    q = InputQueue(on_pause=lambda: 1 / 0, controller=c)
    q._on_escape()  # a display hiccup must never kill the reader thread
    assert c.peek().source == "user"


def test_pause_note_prints_acknowledgement(capsys):
    from tui import ui

    ui.pause_note()
    assert "pausing for plan review" in capsys.readouterr().out


# --- posture at the prompt (live derivation, same reads as the status bar) ----------------------

def test_posture_flags_read_live_config(monkeypatch):
    mod = importlib.import_module("tui.ui.prompt")
    from config import get_config

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "auto_approve", "read_only")
    monkeypatch.setitem(rt, "airgap", False)
    monkeypatch.setitem(rt, "dry_run", False)
    assert mod._posture_flags() == []  # default posture: nothing to announce

    monkeypatch.setitem(rt, "auto_approve", "destructive")  # the gate is OPEN, not "at a tier"
    monkeypatch.setitem(rt, "airgap", True)
    monkeypatch.setitem(rt, "dry_run", True)
    flags = mod._posture_flags()
    assert [k for _, k in flags] == ["gate", "airgap", "dryrun"]
    assert flags[0][0] == "⚠ GATE OFF"


# --- the styled receipt's kind -> style map ------------------------------------------------------

def test_every_receipt_kind_has_a_style():
    # receipt.trust_spans/turn_spans emit exactly these kinds; the styled renderer must know
    # them all or a trust fact would silently fall back to dim.
    resp = importlib.import_module("tui.ui.response")
    assert {"local", "sent", "blocked", "gated", "unknown"} <= set(resp._TRUST_STYLE)


# --- status-bar key legend ----------------------------------------------------------------------

def test_statusbar_key_legend_trails_the_bar():
    sb = importlib.import_module("tui.ui.statusbar")
    if not sb._RICH:
        pytest.skip("rich not available")
    plain = sb._StatusBar().__rich__().plain
    # Trailing on purpose: the bar trims from the right on narrow terminals, so the legend is
    # the first thing sacrificed.
    assert plain.rstrip().endswith("esc pause · ctrl-c cancel")
