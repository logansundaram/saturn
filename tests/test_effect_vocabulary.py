"""Effect authorization beyond the workspace (2026-09-06).

The Apple Calendar / Notes / Mail tools and `schedule_notification` are side_effecting, so a
replan-drafted step for any of them faces `request_authorized` — whose vocabulary was WORKSPACE
verbs only. "make an appointment for me tomorrow to get groceries" authorized nothing, and the
create_calendar_event step the ask-gate redraft added was refused as an unauthorized effect (run
55) — a claim about the user's intent their own words contradict. The same refusal hit every
two-step notify/calendar plan whose effect step is resolved by reference after current_time.

The direction-of-error rule still holds: a noun that doubles as a verb ("my schedule", "the
book", "an email") never counts, and "send" stays out — draft_mail is the closest honest tool
and "draft" / "email" / "write" authorize it.
"""

import pytest
from langchain.messages import HumanMessage

from core import plan_context as pc
from core import request_intent as ri


@pytest.mark.parametrize("request_text", [
    "make an appointment for me tomorrow to get groceries",
    "remind me at 5pm to call mom",
    "schedule a dentist visit friday 9am",
    "book a meeting with Sam tomorrow at 10",
    "draft an email to Petra about the rotation",
    "email Petra the totals",
    "set up a meeting with the team on monday",
    "put an event on my calendar for the game",
    "notify me when it is 3pm",
])
def test_scheduling_and_mail_requests_are_state_changes(request_text):
    assert ri.wants_state_change(request_text) is True


@pytest.mark.parametrize("request_text", [
    "what is on my schedule today",
    "read the book summary in notes.md",
    "review the emails I received today",
    "is there a draft in my mailbox",
    "list my calendar events for tomorrow",
    "send a text to Solveig",   # "send" deliberately absent: no tool sends, and a write is not sending
    # Review 2026-09-06: effect verbs in QUESTION and query-idiom positions authorized nothing
    # the user asked for — "remind me what …" is "tell me", and a past form in a request is a
    # statement about the world, never a request to act on it.
    "read vendor_terms.txt and remind me what the late fee is",
    "remind me of the wifi password in notes.md",
    "is the dentist appointment scheduled?",
    "did you notify Sam about the rotation?",
    "have you booked the table yet",
    "I booked a table already, what time was it",
    "the email I drafted yesterday — what did it say",
])
def test_reading_a_calendar_or_mailbox_is_not_a_state_change(request_text):
    assert ri.wants_state_change(request_text) is False


@pytest.mark.parametrize("request_text", [
    "remind me that the rent is due tomorrow",
    "remind me about the dentist at 3",
    "remind me to call mom",
    "scheduling a dentist visit for friday, add it to the calendar",
    "can you schedule a meeting with Sam tomorrow at 10",   # a polite imperative, not a question
    "could you remind me to call mom at 5",
    "will you notify me when it is 3pm",
])
def test_reminder_requests_still_count(request_text):

    assert ri.wants_state_change(request_text) is True


def test_a_query_with_an_effect_verb_authorizes_no_injected_effect():
    # The direction of error this module exists to avoid: an injected "create a calendar
    # event" in the file's contents must not ride "remind me what" into authorization.
    st = _state("read vendor_terms.txt and remind me what the late fee is")
    assert not pc.request_authorized(
        st, _step("create_calendar_event", "Create the event Pay $500 tomorrow"))



def _state(query):
    return {"current_query": query, "messages": [HumanMessage(query)]}


def _step(tool, label):
    return {"origin": pc.ORIGIN_REPLAN, "intended_tool": tool, "label": label,
            "step_id": 2, "status": "pending", "result": None, "needs_resolution": True}


def test_a_replanned_calendar_event_is_authorized_by_an_appointment_request():
    st = _state("make an appointment for me tomorrow to get groceries")
    assert pc.request_authorized(
        st, _step("create_calendar_event", "Create the event at the time the user gives"))


def test_a_replanned_reminder_is_authorized_by_remind_me():
    st = _state("remind me at 5pm to call mom")
    assert pc.request_authorized(st, _step("schedule_notification", "Schedule the reminder"))


def test_a_calendar_read_request_still_authorizes_no_replanned_effect():
    st = _state("what is on my schedule today")
    assert not pc.request_authorized(st, _step("create_calendar_event", "Create an event"))


# ── arming: effect authorization needs RESULTS, and a refusal is not a result ───────────────


def _lone_ask_state(query, **kw):
    st = {"messages": [HumanMessage(query)], "plan": [
        {"step_id": 1, "label": "Ask the time", "status": "error", "intended_tool": "ask_user",
         "result": "error: ask_user was not executed: no step follows this question",
         "needs_resolution": False}],
        "current_query": query, "context": "", "iteration": 1, "replans": 0,
        "plan_vetoes": [], "revoked_writes": [], "tool_events": []}
    st.update(kw)
    return st


def test_results_exist_reads_what_ran_not_what_has_a_result():
    assert not pc.results_exist(_lone_ask_state("x"))
    assert not pc.results_exist(_lone_ask_state(
        "x", tool_events=[{"name": "ask_user", "args": {"question": "?"}, "ok": True}]))
    assert pc.results_exist(_lone_ask_state(
        "x", tool_events=[{"name": "read_file", "args": {}, "ok": True}]))


def test_a_redraft_before_any_tool_ran_is_not_an_effect_of_results(monkeypatch):
    # Run 57: "make an appointmenet for me to get groceries tomrrow" (typo, so no vocabulary
    # match). The only "result" was the ask gate's own text; the calendar step the redraft added
    # was refused as an unauthorized effect of results that did not exist.
    from core.structured import _PlanItem, _PlanOut
    from nodes import replan as rp

    monkeypatch.setattr(rp, "planner_sys_msg", lambda: HumanMessage(content="sys"))
    monkeypatch.setattr(rp, "registered_tools", lambda: [])
    monkeypatch.setattr(rp, "plan_format", lambda tools: {})
    monkeypatch.setattr(rp, "structured", lambda *a, **k: _PlanOut(plan=[
        _PlanItem(description="Ask the time", tool="ask_user", needs_resolution=False),
        _PlanItem(description="Create the event at the time given", tool="create_calendar_event",
                  needs_resolution=True)]))
    out = rp.replan_node(_lone_ask_state("make an appointmenet for me to get groceries tomrrow",
                                         reasoning="keep the question, add the consuming step"))
    assert [s["status"] for s in out["plan"][1:]] == ["pending", "pending"]
    assert "origin" not in out["plan"][2]
