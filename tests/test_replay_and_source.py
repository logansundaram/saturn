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
            {"seq": 1, "ts": "t1", "node": "plan", "summary": "plan", "data": {"plan": []}},
            {"seq": 2, "ts": "t2", "node": "tools", "summary": "tools", "data": None},
        ],
        "llm_calls": [],
    }


def test_export_rows_shapes_match_show_run():
    run, rows = export_rows(_payload())
    assert run == (7, "what is 2+2", "2026-06-10T11:59:00", "2026-06-10T11:59:30", "ok", "4")
    assert len(rows) == 2
    seq, ts, node, summary, data = rows[0]
    assert (seq, node) == (1, "plan")
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


# --- /trace context — the context inspector ---------------------------------------------------

def _seed_llm_calls_db(tmp_path):
    """A trace DB with one run + two llm_calls rows (plan + synthesize) whose input carries
    role-tagged messages and whose output is distinct text — enough to exercise the context
    inspector's input-only, per-node rendering."""
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
    plan_input = json.dumps([
        {"role": "system", "content": "PLANNER_SYSTEM_PROMPT_TEXT"},
        {"role": "human", "content": "do a thing"},
    ])
    synth_input = json.dumps([
        {"role": "system", "content": "SYNTH_SYSTEM_PROMPT_TEXT"},
        {"role": "human", "content": "the curated synthesize context"},
    ])
    conn.execute(
        "INSERT INTO llm_calls (run_id, seq, node, model, dur, prompt_tokens, output_tokens, "
        "input, output, status) VALUES (5, 1, 'plan', 'qwen', 1.0, 111, 5, ?, ?, 'ok')",
        (plan_input, json.dumps({"content": "PLAN_OUTPUT_SHOULD_NOT_APPEAR", "tool_calls": []})),
    )
    conn.execute(
        "INSERT INTO llm_calls (run_id, seq, node, model, dur, prompt_tokens, output_tokens, "
        "input, output, status) VALUES (5, 2, 'synthesize', 'qwen', 2.0, 222, 9, ?, ?, 'ok')",
        (synth_input, json.dumps({"content": "SYNTH_OUTPUT_SHOULD_NOT_APPEAR", "tool_calls": []})),
    )
    conn.commit()
    conn.close()
    return db


def test_trace_context_renders_inputs_not_outputs(tmp_path, capsys):
    from commands.trace import _show_llm_context

    ctx = SimpleNamespace(db_path=_seed_llm_calls_db(tmp_path))
    _show_llm_context(ctx, [])
    out = capsys.readouterr().out
    # Both calls' INPUT messages render (what the model was told), per node.
    assert "PLANNER_SYSTEM_PROMPT_TEXT" in out
    assert "the curated synthesize context" in out
    assert "plan" in out and "synthesize" in out
    # The OUTPUTS are deliberately absent — this view is the input side only.
    assert "PLAN_OUTPUT_SHOULD_NOT_APPEAR" not in out
    assert "SYNTH_OUTPUT_SHOULD_NOT_APPEAR" not in out


def test_trace_context_node_filter(tmp_path, capsys):
    from commands.trace import _show_llm_context

    ctx = SimpleNamespace(db_path=_seed_llm_calls_db(tmp_path))
    _show_llm_context(ctx, ["--node", "synth"])
    out = capsys.readouterr().out
    assert "the curated synthesize context" in out
    # The plan call is filtered out.
    assert "PLANNER_SYSTEM_PROMPT_TEXT" not in out


def test_trace_context_no_calls(tmp_path, capsys):
    import sqlite3
    from commands.trace import _show_llm_context

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
    ctx = SimpleNamespace(db_path=db)
    _show_llm_context(ctx, [])
    out = capsys.readouterr().out
    assert "no LLM calls recorded yet" in out


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
