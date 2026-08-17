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


# ── Tier-2 instrument surfaces ───────────────────────────────────────────────────────────────


def test_each_plan_rerender_is_delimited_by_a_progress_header(fresh_plan_display, capsys):
    """show_plan re-prints the WHOLE plan on every material change — 8-12 times in a normal turn
    — and the rows were bare, interleaved with trace rows, with nothing marking where one
    rendering ended and the next began."""
    from tui import ui

    plan = [{"step_id": i, "label": f"step {i}", "status": "pending", "intended_tool": None}
            for i in range(1, 5)]
    ui.show_plan(plan)
    assert "plan · 0/4" in capsys.readouterr().out

    plan[0]["status"] = "done"
    plan[1]["status"] = "error"      # an incident is finished too — the count is progress, not success
    ui.show_plan(plan)
    out = capsys.readouterr().out
    assert "plan · 2/4" in out
    assert out.count("plan · ") == 1  # exactly one header per re-render


def test_plan_header_reads_the_engine_status_vocabulary():
    # A hand-copied status list would drift the moment a status is added.
    from core.state import TERMINAL_STATUSES

    plan_ui = importlib.import_module("tui.ui.plan")
    plan = [{"step_id": i, "status": s} for i, s in enumerate(TERMINAL_STATUSES, 1)]
    assert plan_ui._finished(plan) == len(TERMINAL_STATUSES)
    assert plan_ui._finished([{"status": "pending"}, {"status": "active"}]) == 0


def test_meter_color_never_wears_the_risk_vocabulary():
    """`bold red` is the posture/risk voice — `⚠ GATE OFF` sits on the same bar and the
    destructive tier wears it at the gate. A busy CPU is load, not risk."""
    base = importlib.import_module("tui.ui._base")
    assert base._meter_color(10) == "green"
    assert base._meter_color(70) == "yellow"
    assert base._meter_color(99) == "red"
    assert "bold" not in base._meter_color(100)


def test_model_rows_stay_aligned_when_a_name_overflows(capsys):
    """`:<26` only pads. One long tag (a `hf.co/…:Q4_K_M` id) pushed the size, detail and
    binding columns out of alignment for every row after it."""
    from tui import ui

    class _Local:
        def __init__(self, name):
            self.name = name
            self.size_h = "3.4G"
            self.parameter_size = "4.7B"
            self.quantization = "Q4_K_M"
            self.is_embedding = False

    ui.show_models(
        [_Local("hf.co/someorg/a-very-long-model-repository-name:Q4_K_M"), _Local("qwen3.5:4b")],
        {}, "4b", "qwen3-embedding:8b",
    )
    rows = [ln for ln in capsys.readouterr().out.splitlines() if "3.4G" in ln]
    assert len(rows) == 2
    # The size column starts at the same offset on both rows — that is the whole point.
    assert len({ln.index("3.4G") for ln in rows}) == 1


def test_air_gap_glyph_is_one_cell_and_shared_by_rail_and_receipt():
    """`⛔` is East-Asian Wide AND emoji-presentation: terminals paint it as a color emoji that
    ignores the `bold red` style and overflows the rail column. `⊘` is the palette's existing
    blocked glyph — one cell, and it takes the style. The rail and the receipt must name the same
    fact with the same glyph."""
    import unicodedata

    from trust import receipt, egress

    trace = importlib.import_module("tui.ui.trace")
    base = importlib.import_module("tui.ui._base")

    text, style = trace._egress_leaf({"channel": "web_search", "host": "h", "status": "blocked"})
    glyph = text[0]
    assert glyph == base._PLAN["blocked"][0]          # the one blocked glyph in the palette
    assert unicodedata.east_asian_width(glyph) != "W"  # one cell, so the rail stays aligned
    assert style == "bold red"                         # …and the style is what carries the alarm

    parts = receipt.trust_parts(
        [egress.EgressEvent(ts="t", channel="web_search", host="h", n_bytes=0,
                            status=egress.BLOCKED)], 0)
    assert any(p.startswith(glyph) for p in parts)


# ── the status bar names the last FINISHED node, never a running one ─────────────────────────
# show_node is fed from a node's *update* event, which LangGraph emits on completion — so the bar
# said `▸ plan` in active styling while `execute` was running, contradicting the `✓ plan` rail
# line directly above it. No active-node signal exists to render (that needs a graph hook).


def test_statusbar_names_the_last_finished_node_in_past_tense(monkeypatch):
    sb = importlib.import_module("tui.ui.statusbar")
    base = importlib.import_module("tui.ui._base")
    if not sb._RICH:
        pytest.skip("rich not available")

    monkeypatch.setattr(base, "_status", dict(base._status, node="plan"))
    plain = sb._StatusBar().__rich__().plain
    assert "✓ plan" in plain          # the node that FINISHED
    assert "▸ plan" not in plain      # never claimed as active


def test_statusbar_seeds_a_started_turn_before_any_node_completes(monkeypatch):
    sb = importlib.import_module("tui.ui.statusbar")
    base = importlib.import_module("tui.ui._base")
    if not sb._RICH:
        pytest.skip("rich not available")

    monkeypatch.setattr(sb, "_live_start", lambda: None)
    monkeypatch.setattr(base, "_status", dict(base._status))
    sb.reset_turn()
    assert base._status["node"] == base._NODE_STARTING
    plain = sb._StatusBar().__rich__().plain
    assert base._NODE_STARTING in plain
    # …without a completion glyph: nothing has completed yet.
    assert f"✓ {base._NODE_STARTING}" not in plain


# ── synthesize's rail row must not land inside the open response block ───────────────────────
# Its update fires when the node COMPLETES — after the answer began streaming — and rich inserts
# a console print above a live display, so the row shoved the streaming answer down mid-stream.
# The row is skipped at normal verbosity; everything ELSE about the node must still land.


def _fresh_trace():
    base = importlib.import_module("tui.ui._base")
    base._trace_started = False
    base._t_last = None
    base._status = dict(base._status, node="", iteration=0, tools=0, tok_per_sec=0.0)
    return base


def test_synthesize_rail_row_is_skipped_at_normal_verbosity(capsys):
    from tui import ui

    base = _fresh_trace()
    ui.set_verbosity("normal")
    ui.show_node("synthesize", {"context_tokens": 5200, "tok_per_sec": 41.0})
    assert "synthesize" not in capsys.readouterr().out
    # …but the metrics still reached the status bar, which is what the receipt echoes.
    assert base._status["tok_per_sec"] == 41.0
    assert base._status["ctx_used"] == 5200


def test_synthesize_rail_row_returns_under_trace_full(capsys):
    from tui import ui

    _fresh_trace()
    try:
        ui.set_verbosity("verbose")
        ui.show_node("synthesize", {"context_tokens": 5200, "tok_per_sec": 41.0})
        assert "synthesize" in capsys.readouterr().out
    finally:
        ui.set_verbosity("normal")


def test_synthesize_keeps_its_row_when_a_trust_leaf_hangs_off_it(capsys):
    """Folding synthesize through _FOLD_NODES would `return` before the metric feed AND before
    the trust annotations — silently costing the receipt its tok/s and dropping the freeze echo,
    an auditable human action. The row is kept whenever a leaf would otherwise be orphaned."""
    from tui import ui

    base = _fresh_trace()
    ui.set_verbosity("normal")
    ui.show_node("synthesize", {"tok_per_sec": 12.0,
                                "answer_buffer": {"state": "frozen", "text": "x"}})
    out = capsys.readouterr().out
    assert "synthesize" in out            # the row is back — the leaf has a parent
    assert "you froze the answer" in out  # …and the auditable event still prints
    assert base._status["tok_per_sec"] == 12.0

    # A bounded record is the same case: the disclosure keeps its row.
    _fresh_trace()
    ui.show_node("synthesize", {"truncated": {"original_chars": 9999, "dropped": ["messages"]}})
    out = capsys.readouterr().out
    assert "synthesize" in out and "record bounded at write time" in out


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
