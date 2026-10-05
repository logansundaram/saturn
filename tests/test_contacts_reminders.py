"""
Contacts (`search_contacts`) and Reminders (`list_reminders` / `create_reminder` /
`complete_reminder`) — the two AppleScript integrations pivot #3 names.

Offline: osascript never runs (conftest's `mac` controller).
"""

from datetime import datetime

import pytest

from tools.applescript import RS, US

NOW = datetime(2026, 9, 6, 10, 0, 0).astimezone()
FS, GS = "\x1c", "\x1d"


@pytest.fixture
def clock(monkeypatch):
    import tools.reminders as rem
    monkeypatch.setattr(rem, "_now", lambda: NOW)
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


def test_contacts_and_reminders_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    # A contact card and a shared reminders list both carry text someone else wrote.
    for name in ("search_contacts", "list_reminders"):
        assert risk_of(name) == "read_only"
        assert name in _UNTRUSTED, name
    for name in ("create_reminder", "complete_reminder"):
        assert risk_of(name) == "side_effecting"
        assert name not in _UNTRUSTED, name


# ── Contacts ─────────────────────────────────────────────────────────────────────────────────

def test_search_contacts_returns_addresses_numbers_and_birthday(mac):
    mac.reply("2" + RS + _rows(
        ("Petra Novak", "Acme", f"_$!<Work>!$_{FS}petra@acme.com{GS}home{FS}pn@x.org{GS}",
         f"mobile{FS}+1 555 010 2000{GS}", "1988-03-02T12:00"),
        ("Sam Roth", "", "", f"{FS}555-0101{GS}", "1604-11-14T12:00"),
    ))
    out = _tool("search_contacts").invoke({"query": "a"})
    assert out == [
        {"name": "Petra Novak", "organization": "Acme",
         "emails": [{"label": "work", "value": "petra@acme.com"}, {"label": "home", "value": "pn@x.org"}],
         "phones": [{"label": "mobile", "value": "+1 555 010 2000"}],
         "birthday": "1988-03-02"},
        # A card with no year stores 1604; the day is what the user knows.
        {"name": "Sam Roth", "phones": [{"label": "", "value": "555-0101"}], "birthday": "11-14"},
    ]
    s = mac.script()
    assert mac.calls[0] == ["open", "-gja", "Contacts"]
    assert 'id of every person whose name contains "a"' in s
    # Per-person loops over a `whose` result cost ~0.6s a person; by-id fetches ~0.2s (probed).
    assert "person id" in s


def test_search_contacts_caps_the_people_fetched_and_says_so(mac):
    mac.reply("37" + RS + _rows(("A One", "", "", "", "")))
    out = _tool("search_contacts").invoke({"query": "a", "limit": 1})
    assert out["contacts"] == [{"name": "A One"}]
    assert "1 of 37" in out["note"]
    assert "if k ≥ 1 then exit repeat" in mac.script()


def test_search_contacts_ranks_a_name_match_before_a_substring(mac):
    """Contacts' own order put "Ian" fifth behind Brian and Christiana (2026-10-02); the cut to
    `limit` must come AFTER the ranking — exact name, then a first/last name that starts with
    the query, then any name containing it — and a card in several tiers is fetched once."""
    mac.reply("0" + RS)
    _tool("search_contacts").invoke({"query": "ian", "limit": 2})
    s = _osascripts(mac)[0]                       # the ranked query; a names fetch follows the miss
    exact = s.index('whose name is "ian" or first name is "ian"')
    lead = s.index('whose first name starts with "ian"')
    loose = s.index('whose name contains "ian"')
    assert exact < lead < loose
    assert "repeat with i in (exact & lead & loose)" in s
    assert "if ids does not contain i then set end of ids to i" in s
    # the count and the cut read the ranked, de-duplicated list
    assert s.index("set out to ((count of ids)") > s.index("end repeat")
    assert s.index("if k ≥ 2 then exit repeat") > s.index("repeat with i in ids")

def _scripted(monkeypatch, mac, *outputs):
    """osascript answers each call with the next canned output, in order."""
    import subprocess

    from tools import applescript

    queue = list(outputs)

    def fake_run(argv, timeout):
        mac.calls.append(list(argv))
        out = queue.pop(0) if argv[0] == "osascript" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(applescript, "_run", fake_run)


def _osascripts(mac):
    """The scripts run so far, in order (the hidden app-open commands between them skipped)."""
    return [argv[argv.index("-e") + 1] for argv in mac.calls if argv[0] == "osascript"]


def test_search_contacts_falls_back_to_the_closest_names_on_a_typo(mac, monkeypatch):
    """"stanly" (run 45, 2026-10-02) matched nothing: no card has that substring. The fallback
    reads every name once, picks the closest, and fetches those cards — saying it did."""
    _scripted(monkeypatch, mac,
              "0" + RS,                                                      # the ranked query: nothing
              GS.join(["Brian Ling", "Stanley Kim", "Ian Smith", "Christiana Lee"]),
              "1" + RS + _rows(("Stanley Kim", "", "", f"mobile{FS}+1 650 309 3315{GS}", "")))
    out = _tool("search_contacts").invoke({"query": "stanly"})
    assert out["contacts"] == [{"name": "Stanley Kim", "phones": [{"label": "mobile", "value": "+1 650 309 3315"}]}]
    assert "no contact is named 'stanly'" in out["note"] and "closest" in out["note"]
    s = _osascripts(mac)[2]
    assert 'name is "Stanley Kim"' in s and "Brian" not in s                 # only the close names


def test_search_contacts_no_match_and_empty_query(mac, monkeypatch):
    # nothing matches, and no name is close either: the names are fetched once, then the honest no
    _scripted(monkeypatch, mac, "0" + RS, GS.join(["Petra Novak", "Sam Roth"]))
    assert _tool("search_contacts").invoke({"query": "zzz"}) == "No contacts match 'zzz'."
    scripts = _osascripts(mac)
    assert len(scripts) == 2 and "name of every person" in scripts[1]
    n = len(mac.calls)
    assert "name" in _err("search_contacts", {"query": "  "})
    assert len(mac.calls) == n


def test_search_contacts_quotes_the_query(mac):
    mac.reply("0" + RS)
    _tool("search_contacts").invoke({"query": 'O"Brien'})
    assert 'contains "O\\"Brien"' in _osascripts(mac)[0]


def test_contacts_report_non_mac_honestly(monkeypatch):
    from tools import applescript
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert "only available on macOS" in _err("search_contacts", {"query": "x"})


# ── Reminders ────────────────────────────────────────────────────────────────────────────────

def test_list_reminders_sorts_due_first_and_flags_overdue(mac, clock):
    mac.reply(_rows(
        ("x-apple-reminder://A", "Home", "Buy honey", "", "", "false"),
        ("x-apple-reminder://B", "Home", "Call the dentist", "2026-09-05T09:00", "ask about x-ray", "true"),
        ("x-apple-reminder://C", "Work", "Send invoice", "2026-09-08T17:00", "", "false"),
    ))
    out = _tool("list_reminders").invoke({})
    assert out == [
        {"id": "x-apple-reminder://B", "list": "Home", "title": "Call the dentist",
         "due": "2026-09-05T09:00", "overdue": True, "notes": "ask about x-ray", "flagged": True},
        {"id": "x-apple-reminder://C", "list": "Work", "title": "Send invoice", "due": "2026-09-08T17:00"},
        {"id": "x-apple-reminder://A", "list": "Home", "title": "Buy honey"},
    ]
    s = mac.script()
    assert mac.calls[0] == ["open", "-gja", "Reminders"]
    assert "set ls to lists" in s
    # One `properties of` fetch per list: a per-reminder loop costs over a second each (probed).
    assert "properties of (reminders of l whose completed is false)" in s


def test_list_reminders_in_one_named_list(mac, clock):
    mac.reply("")
    out = _tool("list_reminders").invoke({"list": "Groceries"})
    assert out == "No open reminders in Groceries."
    assert 'set ls to {list "Groceries"}' in mac.script()


def test_list_reminders_none_and_unknown_list(mac, clock):
    mac.reply("")
    assert _tool("list_reminders").invoke({}) == "No open reminders."
    mac.reply("", 1, 'execution error: Reminders got an error: Can’t get list "Nope". (-1728)')
    assert "no Reminders list named 'Nope'" in _err("list_reminders", {"list": "Nope"})


def test_create_reminder_with_due_date_notes_and_list(mac, clock):
    mac.reply(f"ok{US}x-apple-reminder://N{US}Home\n")
    out = _tool("create_reminder").invoke(
        {"title": "Call the dentist", "due": "2026-09-07T09:00", "list": "Home", "notes": "ask about x-ray"})
    assert out == {"id": "x-apple-reminder://N", "list": "Home", "title": "Call the dentist",
                   "due": "2026-09-07T09:00"}
    s = mac.script()
    assert 'set l to list "Home"' in s
    assert 'name:"Call the dentist"' in s and 'body:"ask about x-ray"' in s
    assert "due date:my mkdate(2026, 9, 7, 32400)" in s


def test_create_reminder_without_a_date_goes_to_the_default_list(mac, clock):
    mac.reply(f"ok{US}x-apple-reminder://N{US}Reminders\n")
    out = _tool("create_reminder").invoke({"title": "Buy honey"})
    assert out == {"id": "x-apple-reminder://N", "list": "Reminders", "title": "Buy honey"}
    s = mac.script()
    assert "set l to default list" in s and "due date" not in s


def test_create_reminder_names_the_lists_when_there_is_no_usable_one(mac, clock):
    # This Mac has no default list (probed live 2026-10-01: only a shared list exists), and a
    # misspelled list is the same dead end: say which lists there are.
    mac.reply(f"nolist{US}Family{GS}Groceries{GS}\n")
    out = _err("create_reminder", {"title": "Buy honey"})
    assert "no default list" in out and "Family, Groceries" in out
    out = _err("create_reminder", {"title": "Buy honey", "list": "Nope"})
    assert "no Reminders list named 'Nope'" in out and "Family, Groceries" in out


def test_create_reminder_refuses_an_empty_title_and_a_past_time(mac, clock):
    assert _err("create_reminder", {"title": " "}).startswith("Error:")
    assert _err("create_reminder", {"title": "x", "due": "2026-09-01T09:00"}).startswith("Error:")
    assert mac.calls == []


def test_complete_reminder_by_id_and_by_title(mac, clock):
    mac.reply("Call the dentist\n")
    out = _tool("complete_reminder").invoke({"reminder": "x-apple-reminder://B"})
    assert out == {"completed": "Call the dentist"}
    s = mac.script()
    assert 'reminder id "x-apple-reminder://B"' in s and "set completed of r to true" in s
    _tool("complete_reminder").invoke({"reminder": "Call the dentist"})
    assert 'first reminder whose name is "Call the dentist" and completed is false' in mac.script()


def test_complete_reminder_missing_is_an_error(mac, clock):
    mac.reply("", 1, "execution error: Reminders got an error: Can’t get reminder id \"x\". (-1728)")
    assert "no open reminder matches" in _err("complete_reminder", {"reminder": "x-apple-reminder://zz"})
    n = len(mac.calls)
    assert _err("complete_reminder", {"reminder": ""}).startswith("Error:")
    assert len(mac.calls) == n


def test_reminders_are_not_egress(mac, clock):
    from trust import egress
    mac.reply(f"ok{US}id{US}Home\n")
    mark = egress.next_seq()
    _tool("create_reminder").invoke({"title": "x"})
    assert egress.events_since(mark) == []


def test_a_date_only_reminder_is_not_overdue_on_its_own_day(mac, clock):
    # No time of day = due at that day's midnight; it is late once the day is over.
    mac.reply(_rows(
        ("x-apple-reminder://A", "Home", "Renew passport", "2026-09-06T00:00", "", "false"),
        ("x-apple-reminder://B", "Home", "Pay rent", "2026-09-05T00:00", "", "false"),
        ("x-apple-reminder://C", "Home", "Stand-up", "2026-09-06T09:00", "", "false"),
    ))
    out = {r["title"]: r.get("overdue", False) for r in _tool("list_reminders").invoke({})}
    assert out == {"Renew passport": False, "Pay rent": True, "Stand-up": True}
