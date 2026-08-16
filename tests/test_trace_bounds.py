"""The trace record drops nothing SILENTLY (transplanted from the visibility isolate, 0.2.1).

An oversized node delta used to be bounded by slicing the JSON text at _DATA_CAP — an
undecodable blob, so `decode_json` fell back to None and the WHOLE delta (tool events, the plan
update) vanished from `/trace`, exports carried `data: null`, and the Glass Box reconstruction
rendered INCOMPLETE for the wrong reason. `_bound_delta` keeps the record parseable: leaf clip
with a halving cap, then per-key salvage with an explicit `truncated` marker naming what was
dropped and the original size — and the replay says so under the node row.
"""

from stores import trace as trace_mod
from stores.trace import Tracer, decode_json


def _fat_tools_delta(n_calls: int, obs_len: int = 12000) -> dict:
    obs = "x" * obs_len
    return {
        "tools_called": ["read_file"] * n_calls,
        "tool_results": [f"read_file(path='a') -> {obs}" for _ in range(n_calls)],
        "tool_events": [
            {"name": "read_file", "args": {"path": "a"}, "result": obs, "dur": 0.1, "ok": True}
            for _ in range(n_calls)
        ],
        "plan": [{"step_id": 1, "label": "read", "status": "done", "intended_tool": "read_file",
                  "result": "x" * 100, "needs_resolution": False}],
    }


def test_oversized_tools_delta_is_stored_parseable():
    for n in (1, 2, 3, 6):
        _summary, data = trace_mod._summarize(_fat_tools_delta(n))
        assert len(data) <= trace_mod._DATA_CAP, n
        delta = decode_json(data, None)
        assert isinstance(delta, dict), "a mid-token slice would decode to None"
        # nothing dropped: the plan update AND every tool event survive (leaves clipped)
        assert len(delta["tool_events"]) == n
        assert delta["plan"][0]["status"] == "done"
        assert "truncated" not in delta  # no key had to be dropped


def test_huge_plan_delta_keeps_what_fits_and_names_what_it_dropped():
    plan = [{"step_id": i, "label": f"step {i} " + "y" * 60, "status": "pending",
             "intended_tool": None, "result": None, "needs_resolution": False}
            for i in range(1, 400)]
    _s, data = trace_mod._summarize({"plan": plan, "iteration": 3})
    assert len(data) <= trace_mod._DATA_CAP + 400
    delta = decode_json(data, None)
    assert isinstance(delta, dict)
    assert delta["iteration"] == 3
    assert delta["truncated"]["dropped"] == ["plan"]
    assert delta["truncated"]["original_chars"] > trace_mod._DATA_CAP


def test_small_delta_is_untouched():
    delta = {"iteration": 1, "plan": [{"step_id": 1, "label": "a", "status": "done"}]}
    _s, data = trace_mod._summarize(delta)
    assert decode_json(data, None) == delta


def test_tracer_round_trip_of_a_fat_delta(tmp_path):
    t = Tracer(str(tmp_path / "db.sqlite"))
    rid = t.start_run("th", "q")
    t.log_event(rid, "tools", _fat_tools_delta(3))
    (data,) = t.conn.execute("SELECT data FROM events").fetchone()
    assert isinstance(decode_json(data, None), dict)


def test_replay_discloses_a_bounded_record(capsys):
    from tui import ui

    delta = {"iteration": 3, "truncated": {"original_chars": 99999, "dropped": ["plan"],
                                          "note": "delta exceeded the record cap"}}
    ui.show_node("plan", delta)
    out = capsys.readouterr().out
    assert "record bounded at write time" in out and "plan" in out
