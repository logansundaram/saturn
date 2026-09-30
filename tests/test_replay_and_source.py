"""
Trace replay (commands.trace.export_rows / render_export) and the /source citation drill-down
(commands.trace.lookup_source) — the pure halves of both features. Plus the stdout-honesty
guard: `/trace export -o` refuses a missing/flag-shaped path.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from commands.trace import lookup_source
from commands.trace import _export, export_rows, render_export


# --- replay ----------------------------------------------------------------------------------

def _payload():
    return {
        "saturn_trace_export": 1,
        "saturn_version": "0.1.0",
        "exported_at": "2026-06-10T12:00:00",
        "run": {
            "run_id": 7, "query": "what is 2+2", "started_at": "2026-06-10T11:59:00",
            "ended_at": "2026-06-10T11:59:30", "status": "ok", "response": "4",
        },
        "events": [
            {"seq": 1, "ts": "t1", "node": "tools", "summary": "plan", "data": {"plan": []}},
            {"seq": 2, "ts": "t2", "node": "agent", "summary": "agent", "data": None},
        ],
        "llm_calls": [],
    }


def test_export_rows_shapes_match_show_run():
    run, rows = export_rows(_payload())
    assert run == (7, "what is 2+2", "2026-06-10T11:59:00", "2026-06-10T11:59:30", "ok", "4")
    assert len(rows) == 2
    seq, ts, node, summary, data = rows[0]
    assert (seq, node) == (1, "tools")
    assert json.loads(data) == {"plan": []}   # re-encoded so show_run's decode_json works
    assert rows[1][4] is None                 # None data stays None


def test_render_export_renders(tmp_path, capsys):
    payload = _payload()
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True
    out = capsys.readouterr().out
    assert "replaying exported record" in out
    assert "run #7" in out


def _v2_payload():
    """An export of a v2 loop run (ground → agent → approval → tools → agent), shaped the way
    stores/trace records it: messages pre-serialized to "AIMessage: …" strings."""
    step = {"step_id": 1, "label": "read the notes", "status": "done", "intended_tool": None,
            "result": "done", "needs_resolution": False}
    return {
        "saturn_trace_export": 1, "saturn_version": "2.0.0", "exported_at": "2026-09-30T12:00:00",
        "run": {"run_id": 9, "query": "summarize notes.md", "started_at": "2026-09-30T11:59:00",
                "ended_at": "2026-09-30T11:59:10", "status": "ok", "response": "THE_SUMMARY"},
        "events": [
            {"seq": 1, "ts": "2026-09-30T11:59:01", "node": "ground", "summary": "", "data": {}},
            {"seq": 2, "ts": "2026-09-30T11:59:02", "node": "agent", "summary": "",
             "data": {"iteration": 1,
                      "messages": ["AIMessage: I'll read the file. [tool_calls: read_file]"]}},
            {"seq": 3, "ts": "2026-09-30T11:59:03", "node": "approval", "summary": "",
             "data": {"gate_events": [{"calls": [{"name": "read_file", "approved": True}]}]}},
            {"seq": 4, "ts": "2026-09-30T11:59:04", "node": "tools", "summary": "",
             "data": {"plan": [step],
                      "tool_events": [{"name": "read_file", "args": {"path": "notes.md"},
                                       "dur": 0.01, "ok": True, "result": "x"}],
                      "tool_results": ["read_file(path='notes.md') -> NOTES_BODY"]}},
            {"seq": 5, "ts": "2026-09-30T11:59:09", "node": "agent", "summary": "",
             "data": {"iteration": 2, "messages": ["AIMessage: THE_SUMMARY"]}},
        ],
        "llm_calls": [],
    }


def test_render_export_replays_a_v2_loop_run(tmp_path, capsys):
    f = tmp_path / "run_9.json"
    f.write_text(json.dumps(_v2_payload()), encoding="utf-8")
    assert render_export(str(f)) is True
    out = capsys.readouterr().out
    for node in ("ground", "agent", "approval", "tools"):
        assert node in out                           # every v2 node shows in the drill-down
    assert "I'll read the file." in out              # the agent's pre-call thought
    assert "read the notes" in out                   # the checklist the plan tool recorded
    assert "read_file(" in out and "NOTES_BODY" in out  # the call WITH its output
    assert "you approved read_file" in out           # the human gate decision
    assert "THE_SUMMARY" in out and "final answer" in out


def test_render_export_rejects_non_exports(tmp_path, capsys):
    f = tmp_path / "junk.json"
    f.write_text("{\"foo\": 1}", encoding="utf-8")
    assert render_export(str(f)) is False
    assert render_export(str(tmp_path / "missing.json")) is False


# --- /trace export -o argument guard -----------------------------------------------------------
# A dangling -o used to silently write the default path; `-o --md` wrote a JSON file literally
# named '--md'. Both now refuse with a usage error BEFORE any DB/file work.

def test_export_dash_o_missing_path_refuses(tmp_path, capsys):
    ctx = SimpleNamespace(db_path=str(tmp_path / "none.sqlite"))
    _export(ctx, ["-o"])
    out = capsys.readouterr().out
    assert "needs a path" in out and "nothing written" in out
    assert not (tmp_path / "none.sqlite").exists()  # refused before touching the DB


def test_export_dash_o_flag_shaped_value_refuses(tmp_path, capsys):
    ctx = SimpleNamespace(db_path=str(tmp_path / "none.sqlite"))
    _export(ctx, ["-o", "--md"])
    out = capsys.readouterr().out
    assert "needs a path" in out and "nothing written" in out
    assert not Path("--md").exists()                # the literally-named file is never written
    assert not (tmp_path / "none.sqlite").exists()


# --- /trace context — the full /trace invoke view --------------------------------------------

def _seed_llm_calls_db(tmp_path):
    """A trace DB with one run + one agent llm_calls row whose input carries role-tagged
    messages longer than the invoke preview clip."""
    import sqlite3

    db = str(tmp_path / "trace.sqlite")
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT, "
        "ended_at TEXT, status TEXT, response TEXT);"
        "CREATE TABLE llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, seq INTEGER, "
        "ts TEXT, node TEXT, model TEXT, dur REAL, prompt_tokens INTEGER, output_tokens INTEGER, "
        "input TEXT, output TEXT, status TEXT);"
    )
    conn.execute(
        "INSERT INTO runs (run_id, query, status, response) VALUES (5, 'do a thing', 'ok', 'ans')"
    )
    agent_input = json.dumps([
        {"role": "system", "content": "AGENT_SYSTEM_PROMPT " + "x" * 2000 + " END_OF_PROMPT"},
        {"role": "human", "content": "do a thing"},
    ])
    conn.execute(
        "INSERT INTO llm_calls (run_id, seq, node, model, dur, prompt_tokens, output_tokens, "
        "input, output, status) VALUES (5, 1, 'agent', 'qwen', 1.0, 111, 5, ?, ?, 'ok')",
        (agent_input, json.dumps({"content": "THE_ANSWER", "tool_calls": []})),
    )
    conn.commit()
    conn.close()
    return db


def test_trace_context_is_the_full_invoke_view(tmp_path, capsys):
    from commands.trace import _trace

    ctx = SimpleNamespace(db_path=_seed_llm_calls_db(tmp_path))
    _trace(ctx, ["context"])
    folded = capsys.readouterr().out
    _trace(ctx, ["invoke", "--full"])
    assert folded == capsys.readouterr().out
    assert "END_OF_PROMPT" in folded  # whole messages: the tail of a long prompt is shown
    assert "THE_ANSWER" in folded


def test_trace_context_no_calls(tmp_path, capsys):
    import sqlite3
    from commands.trace import _trace

    db = str(tmp_path / "empty.sqlite")
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT, "
        "ended_at TEXT, status TEXT, response TEXT);"
        "CREATE TABLE llm_calls (id INTEGER PRIMARY KEY, run_id INTEGER, seq INTEGER, ts TEXT, "
        "node TEXT, model TEXT, dur REAL, prompt_tokens INTEGER, output_tokens INTEGER, "
        "input TEXT, output TEXT, status TEXT);"
    )
    conn.commit()
    conn.close()
    _trace(SimpleNamespace(db_path=db), ["context"])
    assert "no LLM calls recorded yet" in capsys.readouterr().out


# --- /source ---------------------------------------------------------------------------------

_STATE = {
    "tool_results": ["web_search(query='x') -> first full result text"],
    "documents_retrieved": ["[source: notes.md] full passage text"],
}


def test_lookup_source_numbers_match_build_sources():
    label, text = lookup_source(_STATE, 1)
    assert label.startswith("web_search(")
    assert text == "web_search(query='x') -> first full result text"
    label2, text2 = lookup_source(_STATE, 2)
    assert "notes.md" in label2
    assert text2 == "[source: notes.md] full passage text"


def test_lookup_source_out_of_range():
    assert lookup_source(_STATE, 0) is None
    assert lookup_source(_STATE, 3) is None
    assert lookup_source({}, 1) is None


# --- /trace run-selector grammar (one parser for every subview) ------------------------------


def test_parse_run_selector_grammar(capsys):
    from commands.trace import _parse_run_selector

    assert _parse_run_selector(["#7"]) == (7, None, False)
    assert _parse_run_selector(["-r", "9"]) == (9, None, False)
    assert _parse_run_selector(["12"]) == (12, None, False)  # bare digits are RUN IDS…
    assert _parse_run_selector(["-l", "20"]) == (None, 20, True)  # …except as the list COUNT
    assert _parse_run_selector(["ls"]) == (None, None, True)
    assert _parse_run_selector([]) == (None, None, False)
    assert _parse_run_selector(["garbage"]) == (None, None, False)
    assert "ignoring" in capsys.readouterr().out


def test_parse_run_selector_consume_hook():
    from commands.trace import _parse_run_selector

    seen = {}

    def consume(low, a, it):
        if low == "--md":
            seen["md"] = True
            return True
        return False

    assert _parse_run_selector(["--md", "#3"], consume=consume) == (3, None, False)
    assert seen == {"md": True}


def test_trace_search_is_cut(tmp_path, capsys):
    """`/trace search` was cut 2026-09-30: no subview, no help line; the old spelling gets the
    run selector's "ignoring" note like the other cut subviews."""
    import commands.trace as trace_cmd
    from commands.trace import _trace

    assert not hasattr(trace_cmd, "_search")
    import stores.trace as trace_store
    assert not hasattr(trace_store, "search_runs")
    from stores.trace import Tracer

    db = str(tmp_path / "db.sqlite")
    Tracer(db).conn.close()
    _trace(SimpleNamespace(db_path=db), ["search", "invoice"])
    assert "ignoring unrecognized argument: 'search'" in capsys.readouterr().out
