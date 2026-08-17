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
    assert mod._posture_flags() == []  # default posture: nothing to announce

    monkeypatch.setitem(rt, "auto_approve", "destructive")  # the gate is OPEN, not "at a tier"
    monkeypatch.setitem(rt, "airgap", True)
    flags = mod._posture_flags()
    assert [k for _, k in flags] == ["gate", "airgap"]
    assert flags[0][0] == "⚠ GATE OFF"


# --- the styled receipt's kind -> style map ------------------------------------------------------

def test_every_receipt_kind_has_a_style():
    # receipt.trust_spans/turn_spans emit exactly these kinds; the styled renderer must know
    # them all or a trust fact would silently fall back to dim. (No `local` kind since the
    # 2026-07-06 deviation-only receipt — a calm local turn emits no trust spans at all.)
    resp = importlib.import_module("tui.ui.response")
    assert {"sent", "blocked", "gated", "unknown"} <= set(resp._TRUST_STYLE)


# --- live plan rendering (the faithful re-render, 2026-07-06) -----------------------------------

def _step(sid, label, status="pending", tool=None, result=None):
    return {"step_id": sid, "label": label, "status": status,
            "intended_tool": tool, "result": result, "needs_resolution": False}


@pytest.fixture
def fresh_plan_display():
    base = importlib.import_module("tui.ui._base")
    saved = base._plan_seen
    base._plan_seen = {}
    yield base
    base._plan_seen = saved


def test_show_plan_rerenders_full_plan_with_tools_on_every_material_change(
        fresh_plan_display, capsys):
    from tui import ui

    plan = [_step(1, "read the config", "pending", "read_file"),
            _step(2, "summarize it", "pending", None)]
    ui.show_plan(plan)
    out = capsys.readouterr().out
    assert "read the config" in out and "::read_file" in out and "summarize it" in out

    # Same plan again: nothing changed, nothing prints.
    ui.show_plan(plan)
    assert capsys.readouterr().out == ""

    # A step completing (the execute -> update_plan loop) re-renders the WHOLE plan,
    # tools included — not a one-line diff.
    plan2 = [_step(1, "read the config", "done", "read_file", result="ok"),
             _step(2, "summarize it", "pending", None)]
    ui.show_plan(plan2)
    out = capsys.readouterr().out
    assert "read the config" in out and "::read_file" in out and "summarize it" in out


def test_show_plan_folds_the_bare_active_flip(fresh_plan_display, capsys):
    from tui import ui

    plan = [_step(1, "search the corpus", "pending", "search_knowledge_base")]
    ui.show_plan(plan)
    capsys.readouterr()

    # execute stamps the step active in the same delta whose rail line names it — folded.
    ui.show_plan([_step(1, "search the corpus", "active", "search_knowledge_base")])
    assert capsys.readouterr().out == ""

    # ...but the terminal status still renders (the flip was recorded, not lost).
    ui.show_plan([_step(1, "search the corpus", "done", "search_knowledge_base", result="hits")])
    assert "search the corpus" in capsys.readouterr().out


def test_show_plan_renders_a_replan_redraft_even_when_ids_and_statuses_match(
        fresh_plan_display, capsys):
    from tui import ui

    ui.show_plan([_step(1, "list the workspace", "done", "list_directory", result="files"),
                  _step(2, "guess a filename", "pending", "read_file")])
    capsys.readouterr()

    # replan swaps step 2's label/tool but keeps its id and pending status — the old status-only
    # diff printed NOTHING here, silently hiding the redraft.
    ui.show_plan([_step(1, "list the workspace", "done", "list_directory", result="files"),
                  _step(2, "read notes.md", "pending", "read_file")])
    out = capsys.readouterr().out
    assert "read notes.md" in out


# --- status-bar key legend ----------------------------------------------------------------------

def test_statusbar_key_legend_trails_the_bar():
    sb = importlib.import_module("tui.ui.statusbar")
    if not sb._RICH:
        pytest.skip("rich not available")
    plain = sb._StatusBar().__rich__().plain
    # Trailing on purpose: the bar trims from the right on narrow terminals, so the legend is
    # the first thing sacrificed.
    assert plain.rstrip().endswith("esc pause · ctrl-c cancel")


# ── an unknown plan status is rendered as UNKNOWN, never guessed as pending ─────────────────
# (transplanted from the visibility isolate: views over instrumentation, never a guess)


def test_unknown_plan_status_renders_as_unknown_never_pending(monkeypatch):
    from tui.ui import plan as plan_ui

    monkeypatch.setattr(plan_ui, "_RICH", False)
    row = plan_ui._plan_line_bare({"step_id": 1, "label": "x", "status": "garbage"},
                                  show_tool=False)
    assert "?" in row and "garbage" in row
    assert not row.lstrip().startswith("·")  # the pending glyph would be a guess
    # a step carrying no status at all is still pending (the producer's default)
    row = plan_ui._plan_line_bare({"step_id": 1, "label": "x"}, show_tool=False)
    assert row.lstrip().startswith("·")


# ── streaming vs finished measure: the answer must not re-wrap when it lands ─────────────────
# The live tail and the finished markdown rendered at different widths (full terminal vs
# min(term, _BODY_WIDTH)), so on any terminal wider than ~102 columns every line break in the
# answer moved the instant finish() ran. Both now render at min(term, _BODY_WIDTH).


@pytest.mark.parametrize("width", [80, 110, 160])
def test_streaming_tail_and_final_body_share_one_measure(width, monkeypatch):
    resp = importlib.import_module("tui.ui.response")
    base = importlib.import_module("tui.ui._base")
    if not resp._RICH:
        pytest.skip("rich not available")

    from rich.console import Console

    console = Console(width=width, highlight=False, force_terminal=False)
    monkeypatch.setattr(base, "_console", console)
    monkeypatch.setattr(resp, "_console", console)

    expected = min(width, resp._BODY_WIDTH)

    stream = resp.ResponseStream()
    stream._chars = ["word " * 200]
    stream._len = len(stream._chars[0])
    rendered = console.render_lines(resp._constrained(stream._tail()), pad=False)
    tail_max = max(sum(seg.cell_length for seg in line) for line in rendered)

    body = console.render_lines(
        resp.Padding(resp.Text("word " * 200), (0, 0, 0, 2)),
        console.options.update(width=expected), pad=False,
    )
    body_max = max(sum(seg.cell_length for seg in line) for line in body)

    assert tail_max <= expected
    assert body_max <= expected
    # Both actually FILL the shared measure — a clamp that silently narrowed one of them
    # would pass the bounds above while still re-wrapping at finish().
    assert tail_max == body_max
