"""The ask gate (transplanted from the engine isolate): three DETERMINISTIC rules `execute` applies
before an `ask_user` step generates a call — none of which asks whether a question was
"necessary" (that is judgment, and the judge is the thing that failed):

  1. a budget — the second `ask_user` of a turn does not run (`error`: finish from what is known;
     rectify 4a redrafts around the missing answer);
  2. search-first — the REQUEST names a source the engine can look in (my notes / the knowledge
     base / search…), nothing has been searched yet, and the plan's answer is to ask the user
     (`error`: rectify 4a redrafts toward the search). Read from the human's words only;
  3. no dangling question — no step follows the ask, so its answer feeds nothing (`error`:
     rectify 4a redrafts toward the step that consumes the answer, or a plain cannot-do step;
     a second lone ask ends the run through the no-call guard).

A question the USER asked for in their own words ("ask me which colour") is exempt from 2 and 3 —
that is the interrupting-tool seam, and a gate that could not see it would break it outright.
Measured: `dev.absence.kb_miss` 3/5 → 5/5.
"""

from langchain.messages import HumanMessage

from core import request_intent as ri
from nodes import execute as ex
from nodes import rectify as rc


def _step(step_id, label, tool=None, result=None, status="pending"):
    return {"step_id": step_id, "label": label, "status": status, "intended_tool": tool,
            "result": result, "needs_resolution": False}


def _state(query, plan, **kw):
    base = {"messages": [HumanMessage(query)], "plan": plan, "current_query": query,
            "context": "", "iteration": 0, "replans": 0}
    base.update(kw)
    return base


def _never_generate(monkeypatch):
    def boom(tool, ctx):
        raise AssertionError("the gate must refuse before a generation is spent")

    monkeypatch.setattr(ex, "_generate_tool_call", boom)


def _generate_ok(monkeypatch):
    monkeypatch.setattr(ex, "_generate_tool_call", lambda tool, ctx: ({"question": "?"}, None, None))


def _no_judge(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not reach the judge")

    monkeypatch.setattr(rc, "structured", boom)


# ── the detectors (core/request_intent, a leaf) ─────────────────────────────────────────────


def test_invites_a_question():
    assert ri.invites_a_question("Ask me which colour to use, then save it")
    assert ri.invites_a_question("check with the user before deleting")
    assert ri.invites_a_question("let me choose the format")
    assert not ri.invites_a_question("Search my notes and tell me when my passport expires.")
    assert not ri.invites_a_question(None)


def test_names_searchable_source():
    assert ri.names_searchable_source("Search my notes and tell me when my passport expires.")
    assert ri.names_searchable_source("look up the late fee in the knowledge base")
    assert ri.names_searchable_source("check my files for the invoice")
    assert not ri.names_searchable_source("What is 847 * 293?")
    assert not ri.names_searchable_source("Text Solveig to let her know the Kestrel is delayed.")


# ── the gate in execute ─────────────────────────────────────────────────────────────────────


def test_ask_before_search_is_refused(monkeypatch):
    _never_generate(monkeypatch)
    state = _state("Search my notes and tell me when my passport expires.",
                   [_step(1, "Ask the user when their passport expires", "ask_user")])
    out = ex.execute_node(state)
    s = out["plan"][0]
    assert s["status"] == "error" and s["result"].startswith(ex.ASK_GATE_PREFIX)


def test_ask_allowed_once_the_search_has_run(monkeypatch):
    _generate_ok(monkeypatch)
    state = _state("Search my notes for the renewal date, and if it is not there ask me.",
                   [_step(1, "Search the notes", "search_knowledge_base", "no matching passages",
                          "done"),
                    _step(2, "Ask the user", "ask_user")])
    assert ex.execute_node(state)["messages"][-1].tool_calls


def test_trailing_ask_that_feeds_nothing_is_refused(monkeypatch):
    _never_generate(monkeypatch)
    state = _state("Send an email to Petra reminding her about the March rotation.",
                   [_step(1, "Read roster_march.txt", "read_file", "on call: Petra", "done"),
                    _step(2, "Ask the user for the email address of Petra", "ask_user")])
    out = ex.execute_node(state)
    assert out["plan"][1]["status"] == "error"
    assert out["plan"][1]["result"].startswith(ex.DANGLING_ASK_TEXT)


def test_dangling_ask_redrafts_toward_the_step_that_uses_the_answer(monkeypatch):
    # 2026-09-06 (runs 51-54): the 9b drafts "ask the time" for a calendar request with NO
    # create_calendar_event step after it. Skipping the ask as a guarded outcome ended the run
    # with "I cannot" and an incident, and the user's next message hit the same wall. The
    # dangling ask is a redraft: keep the question, add the consuming step by reference.
    _no_judge(monkeypatch)
    state = _state("make an appointment for me tomorrow to get groceries",
                   [_step(1, "Ask the user what time they want the appointment", "ask_user",
                          ex.DANGLING_ASK_TEXT, "error")])
    out = rc.rectify_node(state)
    assert out["rectify"] is True
    assert out.get("plan") is None  # nothing cancelled
    assert "keep" in out["reasoning"].lower() and "needs_resolution" in out["reasoning"]
    assert "search" not in out["reasoning"].lower().split("if no")[0]  # not the search redraft


def test_dangling_ask_still_refuses_a_substitute_action(monkeypatch):
    # The redraft text must keep the Aug-16 property: an action with no tool ("send an email")
    # is stated as impossible, never replaced by a possible one (a write claiming it sent).
    _no_judge(monkeypatch)
    state = _state("Send an email to Petra reminding her about the March rotation.",
                   [_step(1, "Read roster_march.txt", "read_file", "Petra", "done"),
                    _step(2, "Ask for the email of Petra", "ask_user",
                          ex.DANGLING_ASK_TEXT, "error")])
    out = rc.rectify_node(state)
    assert out["rectify"] is True
    assert "cannot carry out" in out["reasoning"]
    assert "never substitute" in out["reasoning"].lower()


def test_second_dangling_ask_ends_the_run(monkeypatch):
    # The redraft is bounded by the no-call guard: a planner that drafts the lone ask twice
    # (ask_user never executed this turn) lands at an honest synthesize, not a third redraft.
    _no_judge(monkeypatch)
    state = _state("make an appointment for me tomorrow to get groceries",
                   [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
                    _step(2, "Ask the time again", "ask_user", ex.DANGLING_ASK_TEXT, "error")],
                   replans=1)
    out = rc.rectify_node(state)
    assert out["rectify"] is False
    assert [s["status"] for s in out["plan"]] == ["error", "error"]


def test_trailing_ask_the_user_invited_still_runs(monkeypatch):
    _generate_ok(monkeypatch)
    state = _state("Read the palette file, then ask me which colour to use.",
                   [_step(1, "Read palette.txt", "read_file", "red, blue", "done"),
                    _step(2, "Ask which colour", "ask_user")])
    assert ex.execute_node(state)["messages"][-1].tool_calls


def test_a_lone_ask_the_user_did_not_invite_is_refused(monkeypatch):
    _never_generate(monkeypatch)
    state = _state("Text Solveig to let her know the Kestrel is delayed.",
                   [_step(1, "Ask for the number of Solveig", "ask_user")])
    out = ex.execute_node(state)
    assert out["plan"][0]["status"] == "error"
    assert out["plan"][0]["result"].startswith(ex.DANGLING_ASK_TEXT)


def test_ask_allowed_when_the_request_names_no_searchable_source(monkeypatch):
    _generate_ok(monkeypatch)
    state = _state("Ask me which colour to use, then save that colour to depot/choice.txt",
                   [_step(1, "Ask which colour", "ask_user")])
    assert ex.execute_node(state)["messages"][-1].tool_calls


def test_second_ask_of_a_turn_is_refused(monkeypatch):
    _never_generate(monkeypatch)
    state = _state("Ask me which colour to use, then save that colour to depot/choice.txt",
                   [_step(1, "Ask again", "ask_user")],
                   tool_events=[{"name": "ask_user", "args": {"question": "colour?"}, "ok": True}])
    out = ex.execute_node(state)
    assert out["plan"][0]["status"] == "error"
    assert "limit" in out["plan"][0]["result"]
    assert out["plan"][0]["result"].startswith(ex.ASK_GATE_PREFIX)


# ── rectify 4a: the redraft ─────────────────────────────────────────────────────────────────


def test_ask_budget_refusal_does_not_cancel_the_remaining_plan(monkeypatch):
    _no_judge(monkeypatch)
    state = _state("Ask me the title, then ask me the author, then save both to book.txt",
                   [_step(1, "Ask the title", "ask_user", "Kestrel", "done"),
                    _step(2, "Ask the author", "ask_user",
                          ex.ASK_GATE_PREFIX + " this turn has already put 1 question(s)", "error"),
                    _step(3, "Save to book.txt", "write_file")])
    out = rc.rectify_node(state)
    assert out["rectify"] is True
    assert out.get("plan") is None or out["plan"][2].get("result") is None


def test_ask_refusal_routes_to_a_search_redraft(monkeypatch):
    _no_judge(monkeypatch)
    state = _state("Search my notes and tell me when my passport expires.",
                   [_step(1, "Ask the user", "ask_user",
                          ex.ASK_GATE_PREFIX + " nothing has been searched yet", "error")])
    out = rc.rectify_node(state)
    assert out["rectify"] is True and "search" in out["reasoning"].lower()


def test_a_lone_ask_past_the_redraft_budget_lands_without_a_judge_call(monkeypatch):
    # Review 2026-09-06: 4a's `replans < 2` gate let a lone ask refused AFTER two replans fall
    # through to the judge, which asked for another redraft that reproduced the lone ask
    # (two extra model calls) before 1b finally cancelled. The old `skipped` posture ended the
    # run at once; the redrafted posture must land just as directly once its budget is spent.
    _no_judge(monkeypatch)
    state = _state("make an appointment for me tomorrow to get groceries",
                   [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error")],
                   replans=2)
    out = rc.rectify_node(state)
    assert out["rectify"] is False
    assert out.get("plan") is None  # nothing pending to cancel; route lands at synthesize


# ── supersession: the refusal is record, not incident, once the redraft has ASKED ───────────
#
# Stamped on the plan itself (status `superseded`, by update_plan when a later ask_user lands
# `done`), so every reader — the incidents block, the write gate, the rail, /trace, the
# headless status — agrees, instead of one consumer filtering what the others still disclose.


def _update_plan_state(plan, observation):
    from langchain.messages import ToolMessage

    return {"plan": plan, "messages": [HumanMessage("q"), ToolMessage(content=observation,
                                                                         tool_call_id="c1")]}


def test_update_plan_supersedes_the_dangling_refusal_once_the_ask_ran():
    from nodes.update_plan import update_plan_node

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
            _step(2, "Ask the time", "ask_user")]
    out = update_plan_node(_update_plan_state(plan, "3pm"))
    assert [s["status"] for s in out["plan"]] == ["superseded", "done"]
    assert out["plan"][0]["result"].startswith(ex.DANGLING_ASK_TEXT)  # the record stays
    assert plan[0]["status"] == "error"  # state's plan never mutated in place


def test_update_plan_leaves_the_refusal_alone_when_something_else_ran():
    from nodes.update_plan import update_plan_node

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
            _step(2, "Read notes.md", "read_file")]
    out = update_plan_node(_update_plan_state(plan, "contents"))
    assert [s["status"] for s in out["plan"]] == ["error", "done"]
    # a budget refusal is not a dangling one, and a later ask does not supersede it
    plan = [_step(1, "Ask again", "ask_user",
                  ex.ASK_GATE_PREFIX + " this turn has already put 1 question(s)", "error"),
            _step(2, "Ask the time", "ask_user")]
    out = update_plan_node(_update_plan_state(plan, "3pm"))
    assert out["plan"][0]["status"] == "error"


def test_a_superseded_ask_is_not_an_incident_anywhere():
    from core.state import INCIDENT_STATUSES, TERMINAL_STATUSES, incident_steps
    from nodes.synthesize import incidents_block, plan_outcomes_block

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "superseded"),
            _step(2, "Ask the time", "ask_user", "3pm", "done"),
            _step(3, "Create the event at 3pm", "create_calendar_event", "created", "done")]
    assert "superseded" in TERMINAL_STATUSES and "superseded" not in INCIDENT_STATUSES
    assert incident_steps(plan) == [] and incidents_block(plan) == []
    # the outcomes narrative no longer carries the refusal's "what cannot be done" wording
    outcomes = plan_outcomes_block(plan)
    assert "superseded" in outcomes and "cannot be done" not in outcomes


def test_a_dangling_ask_never_asked_again_is_still_disclosed():
    from nodes.synthesize import incidents_block

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
            _step(2, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error")]
    assert len(incidents_block(plan)) == 2
    budget = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
              _step(2, "Ask again", "ask_user",
                    ex.ASK_GATE_PREFIX + " this turn has already put 1 question(s)", "error")]
    assert len(incidents_block(budget)) == 2


def test_a_superseded_ask_does_not_arm_the_write_gate(monkeypatch):
    # Review 2026-09-06: with the refusal stamped `error`, `_write_gate` read it as a failed
    # producer and fired the judge on a plan that never searched or failed — and with the judge
    # unavailable it failed closed, so "make a note in notes.md" never wrote the note.
    def boom(*a, **k):
        raise AssertionError("the gate must not arm on a superseded refusal")

    monkeypatch.setattr(ex, "structured", boom)
    plan = [_step(1, "Ask what to note", "ask_user", ex.DANGLING_ASK_TEXT, "superseded"),
            _step(2, "Ask what to note", "ask_user", "buy milk", "done"),
            _step(3, "Write the note to notes.md", "write_file")]
    assert ex._write_gate(_state("make a note in notes.md", plan), plan[2]) is None

