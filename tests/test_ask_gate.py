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


# ── synthesize: a superseded refusal is not an incident ─────────────────────────────────────


def test_a_dangling_ask_the_redraft_asked_is_not_disclosed_as_a_failure():
    # Runs 55-58: the refused lone ask stayed in the plan as `error`, so the answer's INCIDENTS
    # block told the synthesizer the question "did NOT complete" — and the 9b wrote "I cannot"
    # under a turn that had asked it. The record keeps the refusal (the no-call guard counts
    # it); the disclosure does not, once a later ask_user step actually ran.
    from nodes.synthesize import incidents_block

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
            _step(2, "Ask the time", "ask_user", "3pm", "done"),
            _step(3, "Create the event at 3pm", "create_calendar_event", "created", "done")]
    assert incidents_block(plan) == []


def test_a_dangling_ask_never_asked_again_is_still_disclosed():
    from nodes.synthesize import incidents_block

    plan = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
            _step(2, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error")]
    assert len(incidents_block(plan)) == 2
    budget = [_step(1, "Ask the time", "ask_user", ex.DANGLING_ASK_TEXT, "error"),
              _step(2, "Ask again", "ask_user",
                    ex.ASK_GATE_PREFIX + " this turn has already put 1 question(s)", "error")]
    assert len(incidents_block(budget)) == 2
