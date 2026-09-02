"""Session-end learning, gated (core/memory_review + /memory review + /quit, 2026-09-02).

The invariant: nothing reaches the memory file without a y at the review screen. Candidates are
collected mechanically from a turn's state (steer notes → agent, vetoes + gate denials →
negative, unfinished steps → commitments, failed steps → agent) and from a compaction summary
(memo / user / commitments by shape), queued in a pending file that survives a crash, and
accepted one at a time — with the run id stamped as provenance on every accepted fact. All
offline: the model-proposal seam is monkeypatched wherever a test reaches it.
"""

import json

from langchain.messages import AIMessage, HumanMessage

from commands._framework import CommandContext
from core import memory_review as rv
from core.state import STEER_PREFIX
from stores import memory_registry as mr


def _state(**kw):
    base = {"messages": [HumanMessage("rename the reports")], "current_query": "rename the reports",
            "plan": [], "gate_events": [], "plan_vetoes": [], "aborted": False}
    base.update(kw)
    return base


def _step(label, status="done", result="ok", tool=None):
    return {"step_id": 1, "label": label, "status": status, "intended_tool": tool,
            "result": result, "needs_resolution": False}


# ── mechanical candidates ───────────────────────────────────────────────────────────────────


def test_collect_turn_maps_each_signal_to_its_layer():
    state = _state(
        messages=[HumanMessage("old turn"), AIMessage("done"),
                  HumanMessage("rename the reports"),
                  HumanMessage(f"{STEER_PREFIX} keep the original file names as a prefix")],
        plan_vetoes=["delete the originals"],
        gate_events=[{"calls": [{"id": "1", "name": "run_shell", "approved": False}],
                      "decision": "rejected", "quarantine": False, "step": "run the rename"},
                     {"calls": [{"id": "2", "name": "write_file", "approved": True}],
                      "decision": "approved", "quarantine": False, "step": None}],
        plan=[_step("list the reports"), _step("rename them", status="blocked", result="blocked"),
              _step("verify", status="error", result="Error: permission denied on /x")],
    )
    cands = rv.collect_turn(state, run_id=9)
    by_source = {c["source"]: c for c in cands}
    assert set(by_source) == {"steer", "veto", "gate", "unfinished", "failed"}
    assert by_source["steer"]["layer"] == "agent" and "prefix" in by_source["steer"]["text"]
    assert by_source["veto"]["layer"] == "negative" and "delete the originals" in by_source["veto"]["text"]
    assert by_source["gate"]["layer"] == "negative" and "run_shell" in by_source["gate"]["text"]
    assert "run the rename" in by_source["gate"]["text"]
    assert by_source["unfinished"]["layer"] == "commitments" and "rename them" in by_source["unfinished"]["text"]
    assert by_source["failed"]["layer"] == "agent" and "permission denied" in by_source["failed"]["text"]
    assert all(c["run"] == 9 for c in cands)
    # A steer note from an OLDER turn is not this turn's signal.
    assert sum(1 for c in cands if c["source"] == "steer") == 1


def test_collect_turn_is_quiet_on_a_clean_turn_and_reads_aborted_steps():
    assert rv.collect_turn(_state(plan=[_step("a"), _step("b")])) == []
    cands = rv.collect_turn(_state(aborted=True, plan=[_step("a"), _step("b", status="pending",
                                                                      result=None)]))
    assert [c["layer"] for c in cands] == ["commitments"]
    # Approved gates and user-skipped steps are not incidents to learn from.
    assert rv.collect_turn(_state(plan=[_step("s", status="skipped", result="skipped")])) == []


def test_summary_candidates_sort_bullets_by_shape():
    summary = ("- The user prefers answers in metric units\n"
               "- Decided to keep the Q3 deck in the shared drive\n"
               "- Open thread: the invoice export still needs a date filter\n"
               "- ok\n")
    cands = rv.summary_candidates(summary, run_id=3)
    assert [c["layer"] for c in cands] == ["user", "memo", "commitments"]
    assert all(c["source"] == "compaction" and c["run"] == 3 for c in cands)


# ── the pending queue ───────────────────────────────────────────────────────────────────────


def test_pending_queue_round_trips_and_dedups(isolated_paths):
    mr.add_memory("already known")
    added = rv.add_pending([
        rv._candidate("user", "already known", "model"),
        rv._candidate("negative", "do not: x", "veto", 4),
        rv._candidate("negative", "DO NOT: x", "veto", 5),
    ])
    assert added == 1
    assert [c["text"] for c in rv.load_pending()] == ["do not: x"]
    assert rv.pending_path().exists()
    rv.save_pending([])
    assert not rv.pending_path().exists() and rv.load_pending() == []
    rv.pending_path().write_text("{not json", encoding="utf-8")
    assert rv.load_pending() == []


def test_note_compaction_persists_summary_and_queues_bullets(isolated_paths):
    n = rv.note_compaction("- Decided to archive 2025 invoices\n- The user prefers CSV exports", 2)
    assert n == 2
    assert "archive 2025 invoices" in rv.summary_path().read_text(encoding="utf-8")
    assert {c["layer"] for c in rv.load_pending()} == {"memo", "user"}
    assert not mr.entries()  # nothing reached memory


# ── the review screen ───────────────────────────────────────────────────────────────────────


def _scripted(answers):
    it = iter(answers)

    def ask(_prompt, **_kw):
        return next(it, "")

    return ask


def test_review_writes_only_on_yes_with_provenance(isolated_paths):
    cands = [
        rv._candidate("agent", "PDFs need /docs add first", "failed", 7),
        rv._candidate("negative", "do not suggest Postgres", "veto", 7),
        rv._candidate("commitments", "send the summary", "unfinished", 8),
        rv._candidate("memo", "decided X", "compaction", 8),
        rv._candidate("user", "prefers tea", "model", 8),
    ]
    lines = []
    result = rv.run_review(cands, ask=_scripted(["y", "n", "e", "send Petra the summary", "q"]),
                           emit=lines.append)
    assert [c["text"] for c in result["accepted"]] == ["PDFs need /docs add first",
                                                       "send Petra the summary"]
    assert [c["text"] for c in result["rejected"]] == ["do not suggest Postgres"]
    assert [c["text"] for c in result["remaining"]] == ["decided X", "prefers tea"]
    ents = mr.entries()
    assert [(e["text"], e["layer"], e["by"], e["run"]) for e in ents] == [
        ("PDFs need /docs add first", "agent", "inferred", 7),
        ("send Petra the summary", "commitments", "inferred", 8),
    ]
    assert any(line.startswith("  + [agent] PDFs need") for line in lines)


def test_review_all_and_all_no_never_writes(isolated_paths):
    cands = [rv._candidate("user", f"fact {i}", "model") for i in range(3)]
    result = rv.run_review(cands, ask=_scripted(["n", "n", "n"]), emit=lambda _l: None)
    assert not mr._memory_path().exists() and len(result["rejected"]) == 3
    result = rv.run_review(cands, ask=_scripted(["a"]), emit=lambda _l: None)
    assert [e["text"] for e in mr.entries()] == ["fact 0", "fact 1", "fact 2"]
    # An interrupted prompt (ask returns "") reads as no — the safe default.
    result = rv.run_review([rv._candidate("user", "late", "model")], ask=_scripted([]),
                           emit=lambda _l: None)
    assert result["rejected"] and "late" not in [e["text"] for e in mr.entries()]


def test_review_pending_off_tty_keeps_queue_and_skips_model(isolated_paths, monkeypatch):
    from commands import knowledge

    rv.add_pending([rv._candidate("negative", "do not: y", "veto", 1)])
    called = []
    monkeypatch.setattr(rv, "llm_candidates", lambda *_a, **_k: called.append(1) or [])
    ctx = CommandContext(state={"messages": [HumanMessage("hi")]}, make_initial_state=dict,
                         db_path="")
    knowledge.review_pending(ctx, use_llm=False)
    assert not called
    assert [c["text"] for c in rv.load_pending()] == ["do not: y"]
    assert not mr.entries()
    # On /quit with an empty queue the model pass is skipped — a quiet session pays no call.
    rv.save_pending([])
    knowledge.review_pending(ctx, use_llm=True, on_quit=True)
    assert not called


def test_review_pending_runs_screen_on_tty(isolated_paths, monkeypatch):
    from commands import knowledge
    from tui import ui

    rv.add_pending([rv._candidate("user", "prefers tea", "model", 2)])
    monkeypatch.setattr(rv, "llm_candidates", lambda *_a, **_k: [
        rv._candidate("entities", "Petra is the manager", "model", 2)])
    monkeypatch.setattr("sys.stdin", type("T", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr(ui, "ask", _scripted(["y", "n"]))
    ctx = CommandContext(state={"messages": [HumanMessage("hi"), AIMessage("hello")]},
                         make_initial_state=dict, db_path="")
    knowledge.review_pending(ctx, use_llm=True)
    assert [e["text"] for e in mr.entries()] == ["prefers tea"]
    assert rv.load_pending() == []


def test_quit_runs_review_unless_skipped(isolated_paths, monkeypatch):
    from commands import knowledge, system

    calls = []
    monkeypatch.setattr(knowledge, "review_pending", lambda ctx, **kw: calls.append(kw))
    ctx = CommandContext(state={"messages": []}, make_initial_state=dict, db_path="")
    system._quit(ctx, [])
    assert calls == [{"on_quit": True}] and ctx.should_quit
    ctx.should_quit = False
    system._quit(ctx, ["--no-review"])
    assert len(calls) == 1 and ctx.should_quit


def test_memory_command_add_edit_why_and_pending(isolated_paths, capsys):
    from commands import dispatch

    ctx = CommandContext(state={"messages": []}, make_initial_state=dict, db_path="")
    dispatch("/memory add --layer entities 'the deck' means Q3.pptx", ctx)
    dispatch("/memory add I live in Paris", ctx)
    dispatch("/memory add --replaces 2 I live in Berlin", ctx)
    dispatch("/memory edit 1 'the deck' means Q3_final.pptx", ctx)
    rv.add_pending([rv._candidate("negative", "do not: z", "veto", 6)])
    dispatch("/memory", ctx)
    dispatch("/memory why 3", ctx)
    dispatch("/memory pending", ctx)
    dispatch("/memory done 3", ctx)
    out = capsys.readouterr().out
    assert "Remembered #1 (entities)" in out and "replaces #2" in out
    assert "Q3_final" in out and "1 candidate(s) pending review" in out
    assert "you said it" in out and "+ [negative] do not: z" in out
    assert "done: " in out
    assert [e["id"] for e in mr.entries()] == [1]
    assert json.loads(rv.pending_path().read_text(encoding="utf-8"))["candidates"][0]["run"] == 6


# ── review fixes (2026-09-02) ───────────────────────────────────────────────────────────────


def test_interrupt_leaves_the_rest_pending_and_help_reasks(isolated_paths):
    cands = [rv._candidate("user", f"fact {i}", "model") for i in range(3)]
    # Ctrl-C / Ctrl-D arrive as INTERRUPT (ui.ask's on_interrupt) -> stop, nothing dropped.
    result = rv.run_review(cands, ask=_scripted(["y", rv.INTERRUPT]), emit=lambda _l: None)
    assert [c["text"] for c in result["accepted"]] == ["fact 0"]
    assert [c["text"] for c in result["remaining"]] == ["fact 1", "fact 2"] and not result["rejected"]
    # An interrupt at the edit prompt is the same stop.
    result = rv.run_review(cands[1:], ask=_scripted(["e", rv.INTERRUPT]), emit=lambda _l: None)
    assert [c["text"] for c in result["remaining"]] == ["fact 1", "fact 2"]
    # `?` prints the legend and asks the SAME candidate again.
    shown = []
    result = rv.run_review(cands[1:2], ask=_scripted(["?", "y"]), emit=shown.append)
    assert [c["text"] for c in result["accepted"]] == ["fact 1"] and not result["remaining"]
    assert any("y = keep" in line for line in shown)


def test_collect_turn_reads_the_merged_steer_form_too():
    merged = HumanMessage(f"rename the reports\n{STEER_PREFIX} keep the originals", id="m1")
    cands = rv.collect_turn(_state(messages=[merged]), run_id=4)
    assert [c["source"] for c in cands] == ["steer"]
    assert "keep the originals" in cands[0]["text"] and cands[0]["run"] == 4


def test_review_pending_off_tty_never_calls_the_model(isolated_paths, monkeypatch):
    from commands import knowledge

    rv.add_pending([rv._candidate("negative", "do not: y", "veto", 1)])
    called = []
    monkeypatch.setattr(rv, "llm_candidates", lambda *_a, **_k: called.append(1) or [])
    monkeypatch.setattr("sys.stdin", type("T", (), {"isatty": staticmethod(lambda: False)})())
    ctx = CommandContext(state={"messages": [HumanMessage("hi"), AIMessage("hello")]},
                         make_initial_state=dict, db_path="")
    knowledge.review_pending(ctx, use_llm=True)
    assert not called and len(rv.load_pending()) == 1


def test_review_pending_asks_the_model_once_per_transcript(isolated_paths, monkeypatch):
    from commands import knowledge
    from tui import ui

    rv.add_pending([rv._candidate("user", "prefers tea", "model", 2)])
    called = []
    monkeypatch.setattr(rv, "llm_candidates", lambda *_a, **_k: called.append(1) or [])
    monkeypatch.setattr("sys.stdin", type("T", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr(ui, "ask", _scripted(["q", "q"]))
    ctx = CommandContext(state={"messages": [HumanMessage("hi"), AIMessage("hello")]},
                         make_initial_state=dict, db_path="")
    knowledge.review_pending(ctx, use_llm=True)
    knowledge.review_pending(ctx, use_llm=True, on_quit=True)  # same transcript: no second call
    assert called == [1]


def test_quit_survives_an_interrupted_review(isolated_paths, monkeypatch):
    from commands import knowledge, system

    def boom(ctx, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(knowledge, "review_pending", boom)
    ctx = CommandContext(state={"messages": []}, make_initial_state=dict, db_path="")
    system._quit(ctx, [])
    assert ctx.should_quit
