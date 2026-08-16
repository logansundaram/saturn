"""The stream loop is fail-soft on DISPLAY (transplanted from the visibility isolate): a render
bug (a hostile step dict, a width edge case) must never raise out of run_turn, record a healthy
turn as `error`, and lose the answer. The tracer's log_event runs first and is untouched by the
guard; the failure lands as one warn line + a diag entry and the loop continues."""

import types

from app import turn as turn_mod


class _Tracer:
    def __init__(self):
        self.events = []

    def log_event(self, run_id, node, delta):
        self.events.append((run_id, node, node))


def test_render_error_is_one_line_and_the_loop_continues(monkeypatch, capsys):
    tracer = _Tracer()
    calls = {"node": 0, "plan": 0}

    def boom_show_node(node, delta):
        calls["node"] += 1
        raise TypeError("hostile delta")

    def show_plan(plan):
        calls["plan"] += 1

    monkeypatch.setattr(turn_mod.ui, "show_node", boom_show_node)
    monkeypatch.setattr(turn_mod.ui, "show_plan", show_plan)
    on_update = turn_mod._make_on_update(tracer, 7, show_ui=True)

    # Does not raise; the trace still records the delta.
    on_update("execute", {"iteration": 1})
    on_update("update_plan", {"plan": [{"step_id": 1, "label": "x", "status": "done"}]})
    assert [e[1] for e in tracer.events] == ["execute", "update_plan"]
    assert calls["node"] == 2
    # A failing show_node does not skip the plan render (each ui call is guarded on its own).
    assert calls["plan"] == 1
    out = capsys.readouterr().out
    assert "display error" in out and "hostile delta" in out


def test_healthy_render_path_is_unchanged(monkeypatch):
    tracer = _Tracer()
    seen = []
    monkeypatch.setattr(turn_mod.ui, "show_node", lambda n, d: seen.append(("node", n)))
    monkeypatch.setattr(turn_mod.ui, "show_plan", lambda p: seen.append(("plan", len(p))))
    on_update = turn_mod._make_on_update(tracer, 1, show_ui=True)
    on_update("plan", {"plan": [{"step_id": 1}]})
    assert seen == [("node", "plan"), ("plan", 1)]
