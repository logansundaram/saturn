"""
The write side of the native macOS app tools: `append_note`, `update_calendar_event` /
`delete_calendar_event`, `reply_mail` / `update_mail` — plus the reader fields they lean on
(`recurring` on an event, `replied` on a message).

Offline like tests/test_macos_apps.py: osascript never runs (conftest's `mac` controller
captures the script and feeds canned output).
"""

from datetime import datetime

import pytest

from tools.applescript import GS, RS, US

NOW = datetime(2026, 9, 6, 10, 0, 0).astimezone()


@pytest.fixture
def clock(monkeypatch):
    import tools.calendar as cal
    monkeypatch.setattr(cal, "_now", lambda: NOW)
    return NOW


def _tool(name):
    from tools.registry import tools_by_name
    return tools_by_name[name]


def _err(name, args):
    from tools.toolspec import ToolError

    with pytest.raises(ToolError) as info:
        _tool(name).invoke(args)
    return f"Error: {info.value}"


def _rows(*rows):
    return RS.join(US.join(r) for r in rows) + RS


def test_the_write_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    for name in ("append_note", "update_calendar_event", "reply_mail", "update_mail"):
        assert risk_of(name) == "side_effecting", name
        assert name not in _UNTRUSTED, name
    assert risk_of("delete_calendar_event") == "destructive"   # nothing restores a deleted event


# ── Notes: append ────────────────────────────────────────────────────────────────────────────

def test_append_note_adds_lines_to_the_end_of_the_body(mac):
    mac.reply(f"ok{US}x-coredata://A/ICNote/p7{US}Errands\n")
    out = _tool("append_note").invoke({"note": "Errands", "text": "call the gas company\nbuy <milk> & eggs"})
    assert out == {"id": "x-coredata://A/ICNote/p7", "title": "Errands",
                   "appended": "call the gas company\nbuy <milk> & eggs"}
    s = mac.script()
    assert mac.calls[0] == ["open", "-gja", "Notes"]
    # The body is HTML: one <div> per line, markup escaped, appended to the existing body.
    assert "set body of n to (body of n) &" in s
    assert "<div>call the gas company</div><div>buy &lt;milk&gt; &amp; eggs</div>" in s
    assert 'whose name is "Errands"' in s


def test_append_note_by_id(mac):
    mac.reply(f"ok{US}x-coredata://A/ICNote/p7{US}Errands\n")
    _tool("append_note").invoke({"note": "x-coredata://A/ICNote/p7", "text": "x"})
    assert 'note id "x-coredata://A/ICNote/p7"' in mac.script()


def test_append_note_keeps_a_blank_line(mac):
    mac.reply(f"ok{US}id{US}T\n")
    _tool("append_note").invoke({"note": "T", "text": "a\n\nb"})
    assert "<div>a</div><div><br></div><div>b</div>" in mac.script()


def test_append_note_missing_note_is_an_error(mac):
    mac.reply(f"none{US}\n")
    out = _err("append_note", {"note": "nope", "text": "x"})
    assert "no note is titled 'nope'" in out and "search_notes" in out


def test_append_note_never_picks_among_notes_that_merely_contain_the_title(mac):
    # A read may take the first title containing the text; a WRITE has no undo, and the gate
    # showed only what the model typed — so it lands on the one note with that exact title.
    mac.reply(f"none{US}Shopping — gift ideas{GS}Shopping list{GS}\n")
    out = _err("append_note", {"note": "shopping", "text": "milk"})
    assert "no note is titled 'shopping'" in out
    assert "'Shopping — gift ideas'" in out and "'Shopping list'" in out
    s = mac.script()
    assert s.index('return "none"') < s.index("set body of n to")   # decided before any write
    assert "set n to item 1 of found" in s and "if (count of found) > 1" in s


def test_append_note_refuses_two_notes_with_the_same_title(mac):
    mac.reply(f"many{US}2\n")
    out = _err("append_note", {"note": "Shopping list", "text": "milk"})
    assert "2 notes are titled 'Shopping list'" in out and "id" in out


def test_read_note_still_finds_a_note_by_part_of_its_title(mac):
    mac.reply(f"id{US}Shopping list{US}Notes{US}2026-09-01T10:00{US}milk")
    assert _tool("read_note").invoke({"note": "shopping"})["title"] == "Shopping list"
    assert 'whose name contains "shopping"' in mac.script()


def test_append_note_refuses_a_locked_note(mac):
    mac.reply("locked\n")
    assert "locked" in _err("append_note", {"note": "Diary", "text": "x"})


def test_append_note_refuses_a_note_with_attachments(mac):
    # Rewriting `body` from a script is the only way to append, and it is not known to keep a
    # note's attachments — so the tool checks before it writes and leaves such a note alone.
    mac.reply("attachments\n")
    out = _err("append_note", {"note": "Receipts", "text": "x"})
    assert "attachments" in out and "Notes" in out
    assert "if (count of attachments of n) > 0" in mac.script()


def test_append_note_refuses_empty_input(mac):
    assert _err("append_note", {"note": "", "text": "x"}).startswith("Error:")
    assert _err("append_note", {"note": "T", "text": "  "}).startswith("Error:")
    assert mac.calls == []


# ── Calendar: recurring flag, update, delete ─────────────────────────────────────────────────

def test_list_events_reports_whether_an_event_recurs(mac, clock):
    mac.reply(_rows(
        ("Work", "uid-1", "Standup", "2026-09-08T09:30", "2026-09-08T09:45", "false", "", "FREQ=DAILY;INTERVAL=1"),
        ("Home", "uid-2", "Dentist", "2026-09-09T14:00", "2026-09-09T15:00", "false", "", ""),
    ))
    out = _tool("list_calendar_events").invoke({"start": "2026-09-07", "end": "2026-09-10"})
    assert [e["recurring"] for e in out] == [True, False]
    assert "recurrence of e" in mac.script()


def test_update_event_moves_it_and_keeps_the_duration(mac, clock):
    mac.reply(f"ok{US}Dentist{US}2026-09-16T14:00{US}2026-09-16T15:00{US}0\n")
    out = _tool("update_calendar_event").invoke(
        {"uid": "uid-2", "calendar": "Home", "start": "2026-09-16T14:00"})
    assert out == {"uid": "uid-2", "calendar": "Home", "title": "Dentist",
                   "start": "2026-09-16T14:00", "end": "2026-09-16T15:00"}
    s = mac.script()
    assert 'first event of calendar "Home" whose uid is "uid-2"' in s
    assert "my mkdate(2026, 9, 16, 50400)" in s
    assert "set d2 to d1 + dur" in s             # no `end` given: the event keeps its length
    # Calendar refuses a save where start is not before end, so the order of the two
    # assignments depends on the direction of the move (probed live 2026-10-01).
    assert "if d1 ≥ (end date of e) then" in s


def test_update_event_with_explicit_end_title_and_location(mac, clock):
    mac.reply(f"ok{US}Checkup{US}2026-09-16T14:00{US}2026-09-16T14:30{US}0\n")
    _tool("update_calendar_event").invoke({
        "uid": "u", "calendar": "Home", "start": "2026-09-16T14:00", "end": "2026-09-16T14:30",
        "title": "Checkup", "location": "Main St"})
    s = mac.script()
    assert "set d2 to my mkdate(2026, 9, 16, 52200)" in s
    assert 'set summary of e to "Checkup"' in s and 'set location of e to "Main St"' in s


def test_update_event_bare_clock_time_keeps_the_events_own_day(mac, clock):
    # "Move Friday's review to 3pm": 15:00 is a time ON THE EVENT'S DAY, not the next 15:00
    # from now — the gate showed start='15:00', and the event must not jump to today.
    mac.reply(f"ok{US}Review{US}2026-09-11T15:00{US}2026-09-11T16:00{US}0\n")
    out = _tool("update_calendar_event").invoke({"uid": "u", "calendar": "Work", "start": "15:00"})
    assert out["start"] == "2026-09-11T15:00"
    s = mac.script()
    assert "set d1 to my atclock(start date of e, 54000)" in s and "on atclock(d, secs)" in s
    assert "set d2 to d1 + dur" in s
    assert "my mkdate(2026, 9, 6" not in s and "my mkdate(2026, 9, 7" not in s   # never "today"/"tomorrow"


def test_update_event_bare_clock_end_follows_the_start(mac, clock):
    mac.reply(f"ok{US}Review{US}2026-09-11T15:00{US}2026-09-11T15:30{US}0\n")
    _tool("update_calendar_event").invoke({"uid": "u", "calendar": "Work", "start": "3pm", "end": "3:30pm"})
    s = mac.script()
    assert "set d2 to my atclock(d1, 55800)" in s
    assert "if d2 ≤ d1 then set d2 to d2 + (1 * days)" in s    # 23:00 to 01:00 ends the next day
    # an end alone moves on the event's own end day, and never to before its start
    mac.reply(f"order\n")
    out = _err("update_calendar_event", {"uid": "u", "calendar": "Work", "end": "9:00"})
    assert "at or before" in out
    s = mac.script()
    assert "set d2 to my atclock(end date of e, 32400)" in s
    assert s.index('return "order"') < s.index("set end date of e to d2")


def test_update_event_title_only_leaves_the_times_alone(mac, clock):
    mac.reply(f"ok{US}New{US}2026-09-09T14:00{US}2026-09-09T15:00{US}0\n")
    _tool("update_calendar_event").invoke({"uid": "u", "calendar": "Home", "title": "New"})
    s = mac.script()
    assert "set start date of e" not in s and "set end date of e" not in s


def test_update_event_sets_the_notes_and_nothing_else(mac, clock):
    # "Edit the description to basketball with MK" (run 74, 2026-10-05): create took `notes`,
    # update did not, so the request could not be done.
    mac.reply(f"ok{US}Basketball{US}2026-10-06T16:00{US}2026-10-06T18:30{US}0\n")
    out = _tool("update_calendar_event").invoke(
        {"uid": "u", "calendar": "Home", "notes": 'basketball with "MK"'})
    assert out == {"uid": "u", "calendar": "Home", "title": "Basketball",
                   "start": "2026-10-06T16:00", "end": "2026-10-06T18:30",
                   "notes": 'basketball with "MK"'}
    s = mac.script()
    assert 'set description of e to "basketball with \\"MK\\""' in s
    assert "set start date of e" not in s and "set summary of e" not in s


def test_update_event_without_notes_reports_none(mac, clock):
    mac.reply(f"ok{US}New{US}2026-09-09T14:00{US}2026-09-09T15:00{US}0\n")
    out = _tool("update_calendar_event").invoke({"uid": "u", "calendar": "Home", "title": "New"})
    assert "notes" not in out and "set description of e" not in mac.script()


def test_update_event_needs_something_to_change(mac, clock):
    out = _err("update_calendar_event", {"uid": "u", "calendar": "Home"})
    assert "nothing to change" in out
    assert mac.calls == []


def test_update_event_refuses_a_recurring_event_without_whole_series(mac, clock):
    mac.reply("recurring\n")
    out = _err("update_calendar_event", {"uid": "u", "calendar": "Work", "start": "2026-09-16T09:00"})
    assert "recurring" in out and "whole_series" in out
    assert "if rec is not \"\" then return \"recurring\"" in mac.script()


def test_update_event_whole_series_skips_the_recurring_check(mac, clock):
    mac.reply(f"ok{US}Standup{US}2026-09-16T09:00{US}2026-09-16T09:15{US}0\n")
    _tool("update_calendar_event").invoke(
        {"uid": "u", "calendar": "Work", "start": "2026-09-16T09:00", "whole_series": True})
    assert 'return "recurring"' not in mac.script()


def test_update_event_says_when_attendees_may_be_notified(mac, clock):
    mac.reply(f"ok{US}1:1{US}2026-09-16T09:00{US}2026-09-16T09:30{US}2\n")
    out = _tool("update_calendar_event").invoke(
        {"uid": "u", "calendar": "Work", "start": "2026-09-16T09:00"})
    assert "2 attendee(s)" in out["note"]


def test_update_event_unknown_uid_is_a_plain_error(mac, clock):
    mac.reply("", 1, 'execution error: Calendar got an error: Can’t get event 1 of calendar "Home" '
                     'whose uid = "nope". Invalid index. (-1719)')
    out = _err("update_calendar_event", {"uid": "nope", "calendar": "Home", "title": "x"})
    assert "no event with uid 'nope' in calendar 'Home'" in out and "list_calendar_events" in out


def test_update_event_end_before_start_is_an_error(mac, clock):
    out = _err("update_calendar_event", {"uid": "u", "calendar": "Home",
                                         "start": "2026-09-16T14:00", "end": "2026-09-16T13:00"})
    assert "at or before start" in out
    assert mac.calls == []


def test_delete_event_returns_what_it_removed(mac, clock):
    mac.reply(f"ok{US}Dentist{US}2026-09-09T14:00{US}2026-09-09T15:00{US}0{US}Main St\n")
    out = _tool("delete_calendar_event").invoke({"uid": "uid-2", "calendar": "Home"})
    # The whole event comes back: nothing else could recreate it.
    assert out == {"deleted": {"uid": "uid-2", "calendar": "Home", "title": "Dentist",
                               "start": "2026-09-09T14:00", "end": "2026-09-09T15:00",
                               "location": "Main St"}}
    s = mac.script()
    assert "delete e" in s and 'first event of calendar "Home" whose uid is "uid-2"' in s


def test_delete_event_refuses_a_recurring_event_without_whole_series(mac, clock):
    mac.reply("recurring\n")
    out = _err("delete_calendar_event", {"uid": "u", "calendar": "Work"})
    assert "recurring" in out and "every occurrence" in out


def test_delete_event_says_when_attendees_may_be_notified(mac, clock):
    mac.reply(f"ok{US}1:1{US}2026-09-09T14:00{US}2026-09-09T15:00{US}3{US}\n")
    out = _tool("delete_calendar_event").invoke({"uid": "u", "calendar": "Work"})
    assert "3 attendee(s)" in out["note"]


# ── Mail: replied flag, reply draft, triage ──────────────────────────────────────────────────

def test_list_mail_reports_whether_a_message_was_replied_to(mac):
    mac.reply(_rows(
        ("101", "Petra <p@x.com>", "Thursday", "2026-09-06T10:45", "true", "false"),
        ("99", "A Friend <f@x.org>", "dinner", "2026-09-05T19:00", "true", "true"),
    ))
    out = _tool("list_mail").invoke({})
    assert [m["replied"] for m in out] == [False, True]
    assert "was replied to of m" in mac.script()


def test_reply_mail_opens_an_unsent_reply_with_the_original_quoted(mac):
    mac.reply(f"true{US}Re: Thursday{US}p@x.com\n")
    out = _tool("reply_mail").invoke({"id": 101, "body": "Thursday works.\nI'll be 10 minutes late."})
    assert out == {"to": ["p@x.com"], "subject": "Re: Thursday",
                   "note": "opened in Mail as an unsent reply for you to review and send; nothing was sent"}
    s = mac.script()
    assert "whose id is 101" in s
    assert "reply m opening window yes reply to all no" in s
    assert '"Thursday works." & linefeed & "I\'ll be 10 minutes late."' in s
    assert "my quoted(" in s                     # the original, "> "-prefixed, under the reply
    assert "send" not in s.replace("sender", "")


def test_reply_mail_never_reads_the_reply_body_before_setting_it(mac):
    # Probed live 2026-10-01: reading `content of r` before the set leaves the reply empty.
    mac.reply(f"true{US}Re: x{US}a@b.c\n")
    _tool("reply_mail").invoke({"id": 1, "body": "ok"})
    s = mac.script()
    assert s.index("set content of r to") < s.index("content of r as string")


def test_reply_mail_reply_all(mac):
    mac.reply(f"true{US}Re: x{US}a@b.c, d@e.f\n")
    out = _tool("reply_mail").invoke({"id": 1, "body": "ok", "reply_all": True})
    assert out["to"] == ["a@b.c", "d@e.f"]
    assert "reply to all yes" in mac.script()


def test_reply_mail_says_when_the_text_did_not_land(mac):
    mac.reply(f"false{US}Re: x{US}a@b.c\n")
    out = _err("reply_mail", {"id": 1, "body": "my words"})
    assert "reply window is open" in out and "nothing was sent" in out and "my words" in out


def test_reply_mail_with_windows_line_endings_still_confirms_the_text(mac):
    # The check that the text landed compares the first line; a stray \r made it fail.
    mac.reply(f"true{US}Re: x{US}a@b.c\n")
    _tool("reply_mail").invoke({"id": 1, "body": "Thursday works.\r\nSee you then."})
    s = mac.script()
    assert 'contains "Thursday works.")' in s and "\r" not in s


def test_reply_mail_missing_message_and_empty_body(mac):
    mac.reply("")
    assert "no message with id 7" in _err("reply_mail", {"id": 7, "body": "x"})
    n = len(mac.calls)
    assert _err("reply_mail", {"id": 7, "body": "  "}).startswith("Error:")
    assert _err("reply_mail", {"id": "seven", "body": "x"}).startswith("Error:")
    assert len(mac.calls) == n


def test_reply_mail_is_not_egress(mac):
    from trust import egress
    mac.reply(f"true{US}Re: x{US}a@b.c\n")
    mark = egress.next_seq()
    _tool("reply_mail").invoke({"id": 1, "body": "ok"})
    assert egress.events_since(mark) == []


def test_update_mail_marks_a_batch_in_one_call(mac):
    mac.reply(_rows(("101", "Deals"), ("99", "Re: dinner")))
    out = _tool("update_mail").invoke({"ids": [101, 99, 55], "action": "read"})
    assert out == {"action": "read", "mailbox": "inbox",
                   "changed": [{"id": "101", "subject": "Deals"}, {"id": "99", "subject": "Re: dinner"}],
                   "missing": ["55"]}
    s = mac.script()
    assert s.count("messages of inbox") == 1     # one mailbox reference per call (Mail hangs otherwise)
    assert "whose id is 101 or id is 99 or id is 55" in s
    assert "set read status of m to true" in s


@pytest.mark.parametrize("action,expected", [
    ("unread", "set read status of m to false"),
    ("flag", "set flagged status of m to true"),
    ("unflag", "set flagged status of m to false"),
    ("trash", "move m to trash mailbox"),
])
def test_update_mail_actions(mac, action, expected):
    mac.reply(_rows(("1", "s")))
    _tool("update_mail").invoke({"ids": "1", "action": action})
    assert expected in mac.script()


def test_update_mail_move_needs_a_destination(mac):
    assert "destination" in _err("update_mail", {"ids": [1], "action": "move"})
    assert mac.calls == []
    mac.reply(_rows(("1", "s")))
    out = _tool("update_mail").invoke({"ids": [1], "action": "move", "destination": "Receipts"})
    assert out["destination"] == "Receipts"
    assert 'move m to (first item of (mailboxes whose name is "Receipts"))' in mac.script()


def test_update_mail_moves_from_the_last_message_backwards(mac):
    # A move changes what "message 3 of the mailbox" means; walking backwards keeps every
    # remaining reference valid.
    mac.reply(_rows(("1", "s")))
    _tool("update_mail").invoke({"ids": [1, 2], "action": "trash"})
    assert "from (count of ms) to 1 by -1" in mac.script()


def test_update_mail_accepts_a_comma_separated_string_of_ids(mac):
    mac.reply(_rows(("3", "a"), ("4", "b")))
    out = _tool("update_mail").invoke({"ids": "3, 4", "action": "flag"})
    assert [c["id"] for c in out["changed"]] == ["3", "4"]


def test_update_mail_refuses_bad_input(mac):
    assert "unknown action" in _err("update_mail", {"ids": [1], "action": "archive"})
    assert "numeric" in _err("update_mail", {"ids": ["abc"], "action": "read"})
    assert "at least one" in _err("update_mail", {"ids": [], "action": "read"})
    assert "at most 50" in _err("update_mail", {"ids": list(range(1, 60)), "action": "read"})
    assert mac.calls == []


def test_update_mail_nothing_found_is_an_error(mac):
    mac.reply("")
    assert "none of the messages" in _err("update_mail", {"ids": [5, 6], "action": "read"})


def test_update_mail_uses_the_long_timeout_and_is_not_egress(mac, monkeypatch):
    from tools import applescript
    from trust import egress
    seen = {}
    real = applescript.run

    def spy(script, timeout=30.0, **kw):
        seen["timeout"] = timeout
        return real(script, timeout, **kw)

    monkeypatch.setattr(applescript, "run", spy)
    mac.reply(_rows(("1", "s")))
    mark = egress.next_seq()
    _tool("update_mail").invoke({"ids": [1], "action": "read"})
    assert seen["timeout"] == 90.0
    assert egress.events_since(mark) == []
