"""
Native macOS app tools: the `tools/applescript.py` runner seam, the Notes tools
(`search_notes` / `read_note` / `create_note`) and the Calendar tools (`list_calendar_events` /
`create_calendar_event`).

Fully offline and platform-independent: `osascript` never runs (the runner's process seam is
captured and fed canned output) and the platform selector is pinned per test, so the macOS path
is exercised on the Linux CI leg and the honest-refusal path on a Mac.
"""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from tools import applescript
from tools.applescript import RS, US, AppleScriptError


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────

NOW = datetime(2026, 9, 6, 10, 0, 0).astimezone()


@pytest.fixture
def mac(monkeypatch):
    """Pin the platform to macOS and capture osascript invocations. Yields a controller whose
    `.calls` lists every argv and whose `.reply(stdout)` sets the next canned output."""

    class Ctl:
        calls: list[list[str]] = []
        stdout = ""
        returncode = 0
        stderr = ""

        def reply(self, stdout, returncode=0, stderr=""):
            self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

        def script(self, i=-1):
            argv = self.calls[i]
            return argv[argv.index("-e") + 1] if "-e" in argv else argv[-1]

    ctl = Ctl()

    def fake_run(argv, timeout):
        ctl.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, ctl.returncode, ctl.stdout, ctl.stderr)

    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    monkeypatch.setattr(applescript, "_run", fake_run)
    return ctl


# ── the runner ───────────────────────────────────────────────────────────────────────────────

def test_run_refuses_off_macos(monkeypatch):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    with pytest.raises(AppleScriptError, match="only available on macOS.*linux"):
        applescript.run('return "x"')


def test_run_returns_stdout_and_passes_the_script(mac):
    mac.reply("hello\n")
    assert applescript.run('return "hello"') == "hello"
    assert mac.calls[0][0] == "osascript"
    assert mac.script() == 'return "hello"'


def test_run_opens_the_app_hidden_before_scripting_it(mac):
    mac.reply("ok\n")
    applescript.run("x", app="Calendar")
    assert mac.calls[0] == ["open", "-gja", "Calendar"]     # osascript alone gets -600 (probed live)
    assert mac.calls[1][0] == "osascript"


def test_notes_and_calendar_tools_open_their_app_first(mac, clock):
    mac.reply("")
    _tool("search_notes").invoke({"query": "x"})
    assert mac.calls[0] == ["open", "-gja", "Notes"]
    _tool("list_calendar_events").invoke({})
    assert mac.calls[-2] == ["open", "-gja", "Calendar"]


def test_run_keeps_a_trailing_empty_field(mac):
    mac.reply(f"a{US}\n")                      # str.strip() would eat the US — a real bug once
    assert applescript.run("x") == f"a{US}"


def test_run_translates_automation_denied(mac):
    mac.reply("", 1, "execution error: Not authorized to send Apple events to Notes. (-1743)")
    with pytest.raises(AppleScriptError, match="denied.*Notes.*Automation"):
        applescript.run('tell application "Notes" to count of notes')


def test_run_translates_app_not_running(mac):
    mac.reply("", 1, "execution error: Calendar got an error: Application isn’t running. (-600)")
    with pytest.raises(AppleScriptError, match="Calendar.*not running"):
        applescript.run('tell application "Calendar" to name of calendars')


def test_run_surfaces_other_failures_verbatim(mac):
    mac.reply("", 1, "execution error: Can’t get note \"zzz\". (-1728)")
    with pytest.raises(AppleScriptError, match="Can’t get note"):
        applescript.run("x")


def test_run_reports_timeout(mac, monkeypatch):
    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)
    monkeypatch.setattr(applescript, "_run", slow)
    with pytest.raises(AppleScriptError, match="timed out after 5"):
        applescript.run("x", timeout=5)


def test_quote_escapes_and_keeps_newlines():
    q = applescript.quote('say "hi"\\ \nline two')
    assert q == '"say \\"hi\\"\\\\ " & linefeed & "line two"'


def test_records_splits_on_separators_and_drops_trailing_newline():
    out = f"a{US}b{US}c{RS}d{US}e{US}f\n"
    assert applescript.records(out) == [["a", "b", "c"], ["d", "e", "f"]]
    assert applescript.records("") == []
    assert applescript.records("\n") == []
    assert applescript.records(f"a{US}b{RS}") == [["a", "b"]]   # a trailing RS is not a record


# ── Notes ────────────────────────────────────────────────────────────────────────────────────

def _tool(name):
    from tools.registry import tools_by_name
    return tools_by_name[name]


def _err(name, args):
    """A refused call RAISES ToolError (stamped error by the tools node); returns the
    observation the model sees, "Error: …"."""
    from tools.toolspec import ToolError

    with pytest.raises(ToolError) as info:
        _tool(name).invoke(args)
    return f"Error: {info.value}"


def test_notes_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of, tools_by_name
    from tools.toolspec import _UNTRUSTED
    for name in ("search_notes", "read_note"):
        assert risk_of(name) == "read_only"
        assert name in _UNTRUSTED, name
    assert risk_of("create_note") == "side_effecting"
    assert "create_note" not in _UNTRUSTED
    assert "create_note" in tools_by_name


def test_search_notes_quotes_the_query_and_parses_records(mac):
    mac.reply(
        f"x-coredata://A/ICNote/p1{US}Dentist \"Tues\"{US}Notes{US}2026-09-01T09:15{RS}"
        f"x-coredata://A/ICNote/p2{US}Groceries{US}Lists{US}2026-08-30T18:00\n"
    )
    out = _tool("search_notes").invoke({"query": 'dentist "tues"', "limit": 5})
    assert out == [
        {"id": "x-coredata://A/ICNote/p1", "title": 'Dentist "Tues"', "folder": "Notes", "modified": "2026-09-01T09:15"},
        {"id": "x-coredata://A/ICNote/p2", "title": "Groceries", "folder": "Lists", "modified": "2026-08-30T18:00"},
    ]
    script = mac.script()
    assert '"dentist \\"tues\\""' in script
    assert 'tell application "Notes"' in script


def test_search_notes_limit_keeps_the_most_recent(mac):
    mac.reply(
        f"id-old{US}Old{US}Notes{US}2026-01-01T00:00{RS}"
        f"id-new{US}New{US}Notes{US}2026-09-01T00:00{RS}"
    )
    out = _tool("search_notes").invoke({"query": "", "limit": 1})
    assert [n["id"] for n in out] == ["id-new"]


def test_search_notes_hides_recently_deleted(mac):
    mac.reply(f"id-1{US}Gone{US}Recently Deleted{US}2026-09-01T00:00{RS}id-2{US}Kept{US}Notes{US}2026-08-01T00:00{RS}")
    out = _tool("search_notes").invoke({"query": ""})
    assert [n["id"] for n in out] == ["id-2"]


def test_search_notes_no_match_is_a_note_not_an_error(mac):
    mac.reply("")
    out = _tool("search_notes").invoke({"query": "zzz"})
    assert out == "No notes match 'zzz'."


def test_read_note_by_title_returns_the_plaintext_body(mac):
    body = "Line one\nLine two, with a comma | and a pipe"
    mac.reply(f"x-coredata://A/ICNote/p1{US}Dentist{US}Notes{US}2026-09-01T09:15{US}{body}\n")
    out = _tool("read_note").invoke({"note": "Dentist"})
    assert out == {"id": "x-coredata://A/ICNote/p1", "title": "Dentist", "folder": "Notes",
                   "modified": "2026-09-01T09:15", "body": body}
    assert '"Dentist"' in mac.script()


def test_read_note_by_id_uses_the_id(mac):
    mac.reply(f"x-coredata://A/ICNote/p9{US}T{US}F{US}2026-09-01T09:15{US}b\n")
    _tool("read_note").invoke({"note": "x-coredata://A/ICNote/p9"})
    assert 'note id "x-coredata://A/ICNote/p9"' in mac.script()


def test_read_note_missing_is_an_error(mac):
    mac.reply("")
    out = _err("read_note", {"note": "nope"})
    assert out.startswith("Error:") and "nope" in out


def test_create_note_writes_title_body_and_folder(mac):
    mac.reply("x-coredata://A/ICNote/p42\n")
    out = _tool("create_note").invoke({"title": "Trip", "body": "pack\nsocks", "folder": "Travel"})
    assert out == {"id": "x-coredata://A/ICNote/p42", "title": "Trip", "folder": "Travel"}
    s = mac.script()
    assert '"Trip"' in s and '"pack<br>socks"' in s and 'folder "Travel"' in s


def test_create_note_escapes_markup_in_the_body(mac):
    mac.reply("id\n")
    _tool("create_note").invoke({"title": "T", "body": "a <b>& b"})
    assert '"a &lt;b&gt;&amp; b"' in mac.script()


def test_create_note_defaults_to_the_default_folder(mac):
    mac.reply("x-coredata://A/ICNote/p43\n")
    out = _tool("create_note").invoke({"title": "Quick", "body": "x"})
    assert out["folder"] == "" and 'folder "' not in mac.script()


def test_create_note_refuses_empty_title(mac):
    out = _err("create_note", {"title": "  ", "body": "x"})
    assert out.startswith("Error:") and mac.calls == []


def test_notes_tools_report_non_mac_honestly(monkeypatch):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    out = _err("search_notes", {"query": "x"})
    assert out.startswith("Error:") and "only available on macOS" in out and "linux" in out


# ── Calendar ─────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def clock(monkeypatch):
    import tools.calendar as cal
    monkeypatch.setattr(cal, "_now", lambda: NOW)
    return NOW


def test_calendar_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("list_calendar_events") == "read_only"
    assert "list_calendar_events" in _UNTRUSTED
    assert risk_of("create_calendar_event") == "side_effecting"
    assert "create_calendar_event" not in _UNTRUSTED


def _event_rows(*rows):
    return RS.join(US.join(r) for r in rows) + RS


def test_list_events_parses_and_sorts_by_start(mac, clock):
    mac.reply(_event_rows(
        ("Work", "uid-2", "Standup", "2026-09-08T09:30", "2026-09-08T09:45", "false", "Zoom"),
        ("Home", "uid-1", "Labor Day", "2026-09-07T00:00", "2026-09-08T00:00", "true", ""),
    ))
    out = _tool("list_calendar_events").invoke({"start": "2026-09-07", "end": "2026-09-09"})
    assert out == [
        {"calendar": "Home", "uid": "uid-1", "title": "Labor Day", "start": "2026-09-07T00:00",
         "end": "2026-09-08T00:00", "all_day": True, "location": ""},
        {"calendar": "Work", "uid": "uid-2", "title": "Standup", "start": "2026-09-08T09:30",
         "end": "2026-09-08T09:45", "all_day": False, "location": "Zoom"},
    ]
    s = mac.script()
    assert "my mkdate(2026, 9, 7, 0)" in s and "my mkdate(2026, 9, 9, 0)" in s
    assert 'tell application "Calendar"' in s


def test_list_events_defaults_to_the_coming_week(mac, clock):
    mac.reply("")
    out = _tool("list_calendar_events").invoke({})
    s = mac.script()
    assert "my mkdate(2026, 9, 6, 0)" in s and "my mkdate(2026, 9, 13, 0)" in s
    assert out == "No events between 2026-09-06T00:00 and 2026-09-13T00:00."


def test_list_events_filters_by_calendar_names(mac, clock):
    mac.reply("")
    _tool("list_calendar_events").invoke({"calendars": "Work, Home"})
    s = mac.script()
    assert '{"Work", "Home"}' in s and "every calendar" not in s


def test_list_events_no_filter_walks_every_calendar(mac, clock):
    mac.reply("")
    _tool("list_calendar_events").invoke({})
    assert "every calendar" in mac.script()


def test_list_events_accepts_relative_times_and_end_before_start_is_an_error(mac, clock):
    mac.reply("")
    _tool("list_calendar_events").invoke({"start": "tomorrow at 09:00", "end": "in 2 days"})
    s = mac.script()
    assert "my mkdate(2026, 9, 7, 32400)" in s and "my mkdate(2026, 9, 8, 36000)" in s
    out = _err("list_calendar_events", {"start": "2026-09-09", "end": "2026-09-08"})
    assert out.startswith("Error:") and "before" in out


def test_list_events_accepts_the_bare_days_its_description_offers(mac, clock):
    """'What's on my calendar today?' — the schema says 'today' works, so it must."""
    mac.reply("")
    _tool("list_calendar_events").invoke({"start": "today", "end": "tomorrow"})
    s = mac.script()
    assert "my mkdate(2026, 9, 6, 0)" in s and "my mkdate(2026, 9, 7, 0)" in s
    _tool("list_calendar_events").invoke({"start": "today", "end": "in 1 week"})
    assert "my mkdate(2026, 9, 13, 36000)" in mac.script()
    _tool("list_calendar_events").invoke({"start": "next monday", "end": "next friday"})
    assert "my mkdate(2026, 9, 7, 0)" in mac.script() and "my mkdate(2026, 9, 11, 0)" in mac.script()


def test_event_times_with_an_offset_land_at_the_local_wall_clock(mac, clock):
    """Calendar dates are local wall-clock. '15:00Z' is 15:00 UTC, which is some other hour
    here — the fields are converted, and the observation reports the local time it wrote."""
    mac.reply("uid-9\n")
    start = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
    local = start.astimezone()
    out = _tool("create_calendar_event").invoke(
        {"calendar": "Work", "title": "Sync", "start": "2026-09-08T15:00Z"})
    secs = local.hour * 3600 + local.minute * 60
    assert f"my mkdate({local.year}, {local.month}, {local.day}, {secs})" in mac.script()
    assert out["start"] == local.isoformat(timespec="minutes")[:16]
    # an offset far from any real zone, so the test bites wherever it runs
    far = datetime(2026, 9, 8, 15, 0, tzinfo=timezone(timedelta(hours=14))).astimezone()
    _tool("create_calendar_event").invoke(
        {"calendar": "Work", "title": "Sync", "start": "2026-09-08T15:00+14:00"})
    assert f"my mkdate({far.year}, {far.month}, {far.day}, {far.hour * 3600 + far.minute * 60})" in mac.script()


def test_list_events_bad_time_is_an_error(mac, clock):
    out = _err("list_calendar_events", {"start": "whenever"})
    assert out.startswith("Error:") and "whenever" in out and mac.calls == []


def test_create_event_writes_all_fields_and_returns_the_uid(mac, clock):
    mac.reply("uid-77\n")
    out = _tool("create_calendar_event").invoke({
        "calendar": "Work", "title": "Design review", "start": "2026-09-08T14:00",
        "end": "2026-09-08T15:30", "location": "Room 4", "notes": "bring the \"spec\"",
    })
    assert out == {"uid": "uid-77", "calendar": "Work", "title": "Design review",
                   "start": "2026-09-08T14:00", "end": "2026-09-08T15:30"}
    s = mac.script()
    assert 'calendar "Work"' in s and 'summary:"Design review"' in s
    assert "my mkdate(2026, 9, 8, 50400)" in s and "my mkdate(2026, 9, 8, 55800)" in s
    assert 'location:"Room 4"' in s and 'description:"bring the \\"spec\\""' in s


def test_create_event_defaults_to_one_hour(mac, clock):
    mac.reply("uid-1\n")
    out = _tool("create_calendar_event").invoke({"calendar": "Home", "title": "Coffee", "start": "tomorrow at 10:00"})
    assert out["start"] == "2026-09-07T10:00" and out["end"] == "2026-09-07T11:00"


def test_create_event_refuses_empty_title_and_calendar(mac, clock):
    assert _err("create_calendar_event", {"calendar": "", "title": "x", "start": "10:00"}).startswith("Error:")
    assert _err("create_calendar_event", {"calendar": "Home", "title": " ", "start": "10:00"}).startswith("Error:")
    assert mac.calls == []


def test_create_event_relays_an_unknown_calendar(mac, clock):
    mac.reply("", 1, 'execution error: Calendar got an error: Can’t get calendar "Nope". (-1728)')
    out = _err("create_calendar_event", {"calendar": "Nope", "title": "x", "start": "10:00"})
    assert out.startswith("Error:") and 'calendar "Nope"' in out


def test_calendar_tools_report_non_mac_honestly(monkeypatch, clock):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    out = _err("list_calendar_events", {})
    assert out.startswith("Error:") and "only available on macOS" in out


# ── Mail ─────────────────────────────────────────────────────────────────────────────────────

def test_mail_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    for name in ("list_mail", "search_mail", "read_mail"):
        assert risk_of(name) == "read_only"
        assert name in _UNTRUSTED, name
    assert risk_of("draft_mail") == "side_effecting"
    assert "draft_mail" not in _UNTRUSTED


def test_list_mail_parses_records_newest_first(mac):
    mac.reply(_event_rows(
        ("101", "Costco <c@costco.com>", "Deals", "2026-09-06T10:45", "false"),
        ("99", "A Friend <f@x.org>", "Re: dinner", "2026-09-05T19:00", "true"),
    ))
    out = _tool("list_mail").invoke({})
    assert out == [
        {"id": "101", "mailbox": "inbox", "from": "Costco <c@costco.com>", "subject": "Deals",
         "date": "2026-09-06T10:45", "unread": True},
        {"id": "99", "mailbox": "inbox", "from": "A Friend <f@x.org>", "subject": "Re: dinner",
         "date": "2026-09-05T19:00", "unread": False},
    ]
    s = mac.script()
    assert mac.calls[0] == ["open", "-gja", "Mail"]
    assert "messages 1 thru 10 of inbox" in s


def test_list_mail_unread_only_filters_in_python_over_the_newest(mac):
    mac.reply(_event_rows(
        ("3", "a", "read one", "2026-09-06T10:00", "true"),
        ("2", "b", "unread one", "2026-09-06T09:00", "false"),
        ("1", "c", "unread two", "2026-09-06T08:00", "false"),
    ))
    out = _tool("list_mail").invoke({"unread_only": True, "limit": 1})
    assert [m["id"] for m in out] == ["2"]
    s = mac.script()
    assert "whose" not in s                      # `whose read status is false` hung Mail for 15 min
    assert "messages 1 thru 25 of inbox" in s    # a bounded window of the newest instead (~0.25s/message)


def test_list_mail_unread_only_says_so_when_none_in_the_window(mac):
    mac.reply(_event_rows(("3", "a", "read one", "2026-09-06T10:00", "true")))
    out = _tool("list_mail").invoke({"unread_only": True})
    assert out == "No unread messages among the newest 25 in inbox."


def test_list_mail_standard_and_named_mailboxes(mac):
    mac.reply("")
    _tool("list_mail").invoke({"mailbox": "Sent"})
    assert "of sent mailbox" in mac.script()
    _tool("list_mail").invoke({"mailbox": "Receipts"})
    assert 'mailboxes whose name is "Receipts"' in mac.script()


def test_list_mail_unknown_mailbox_is_a_plain_error(mac):
    mac.reply("", 1, 'execution error: Mail got an error: Can’t get item 1 of every mailbox whose name = "Nope". Invalid index. (-1719)')
    out = _err("list_mail", {"mailbox": "Nope"})
    assert out == "Error: no mailbox named 'Nope' (use inbox, sent, drafts, junk, trash, or a folder/label name)"


def test_search_mail_matches_subject_or_sender(mac):
    mac.reply(_event_rows(("7", "bmw@x.com", "Your BMW order", "2026-09-05T19:00", "true")))
    out = _tool("search_mail").invoke({"query": "BMW"})
    assert out[0]["subject"] == "Your BMW order"
    s = mac.script()
    assert 'subject contains "BMW" or sender contains "BMW"' in s


def test_search_mail_no_match(mac):
    mac.reply("")
    assert _tool("search_mail").invoke({"query": "zzz"}) == "No messages match 'zzz'."


def test_read_mail_returns_headers_and_body(mac):
    body = "Hi,\n\nPlease verify.\n-- \nR"
    mac.reply(f"7{US}R <r@y.com>{US}me@x.com{US}Fwd: verify{US}2026-09-05T19:00{US}{body}\n")
    out = _tool("read_mail").invoke({"id": 7})
    assert out == {"id": "7", "mailbox": "inbox", "from": "R <r@y.com>", "to": "me@x.com",
                   "subject": "Fwd: verify", "date": "2026-09-05T19:00", "body": body}
    assert "(messages of inbox whose id is 7)" in mac.script()   # `message id N` is Mail's header property, a syntax error


def test_read_mail_missing_is_an_error(mac):
    mac.reply("")
    out = _err("read_mail", {"id": 12345})
    assert out.startswith("Error:") and "12345" in out


def test_mail_tools_use_the_long_query_timeout(mac, monkeypatch):
    seen = []
    monkeypatch.setattr(applescript, "_run", lambda argv, timeout: (seen.append(timeout), subprocess.CompletedProcess(argv, 0, "", ""))[1])
    _tool("list_mail").invoke({})
    assert max(seen) >= 90     # a killed osascript leaves its event queued in Mail; never fire-and-forget


def test_read_mail_rejects_a_non_numeric_id(mac):
    out = _err("read_mail", {"id": "abc"})
    assert out.startswith("Error:") and mac.calls == []


def test_draft_mail_opens_a_visible_unsent_draft(mac):
    mac.reply("ok\n")
    out = _tool("draft_mail").invoke({
        "to": "a@x.com, b@y.org", "subject": "Hello", "body": "line 1\nline 2", "cc": "c@z.net",
    })
    assert out == {"to": ["a@x.com", "b@y.org"], "cc": ["c@z.net"], "subject": "Hello",
                   "note": "opened in Mail as an unsent draft for you to review and send; nothing was sent"}
    s = mac.script()
    assert "visible:true" in s and 'subject:"Hello"' in s and '"line 1" & linefeed & "line 2"' in s
    assert 'to recipient with properties {address:"a@x.com"}' in s
    assert 'to recipient with properties {address:"b@y.org"}' in s
    assert 'cc recipient with properties {address:"c@z.net"}' in s
    assert "send" not in s.replace("sender", "")


def test_draft_mail_refuses_missing_recipient_or_subject(mac):
    assert _err("draft_mail", {"to": "", "subject": "x", "body": "y"}).startswith("Error:")
    assert _err("draft_mail", {"to": "a@x.com", "subject": " ", "body": "y"}).startswith("Error:")
    assert mac.calls == []


def test_draft_mail_is_not_egress(mac):
    from trust import egress
    mac.reply("ok\n")
    mark = egress.next_seq()
    _tool("draft_mail").invoke({"to": "a@x.com", "subject": "s", "body": "b"})
    assert egress.events_since(mark) == []
