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
