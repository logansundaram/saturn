"""
Messages: `send_message` — the one tool that sends the user's words to another person, so an
egress chokepoint that ALWAYS faces the human — and `read_messages`, the history reader over
`~/Library/Messages/chat.db`.

Offline: osascript never runs (conftest's `mac`), and the history is a throwaway SQLite file
with the columns the reader uses.
"""

import sqlite3
import subprocess
import types
from datetime import datetime, timezone

import pytest

from tools import applescript
from tools.applescript import FS, GS, RS, US
from trust import egress, policy, quarantine


def _tool(name):
    from tools.registry import tools_by_name
    return tools_by_name[name]


def _err(name, args):
    from tools.toolspec import ToolError

    with pytest.raises(ToolError) as info:
        _tool(name).invoke(args)
    return f"Error: {info.value}"


@pytest.fixture
def gate(isolated_paths, monkeypatch):
    from config import get_config

    runtime = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(runtime, "auto_approve", "read_only")
    monkeypatch.setitem(runtime, "airgap", False)
    monkeypatch.setattr(policy, "_tier_before_gate_off", None)
    return get_config()


# ── send_message: the tool ───────────────────────────────────────────────────────────────────

def test_message_tools_are_registered_with_the_right_trust():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    assert risk_of("send_message") == "destructive" and "send_message" not in _UNTRUSTED
    assert quarantine.is_outbound("send_message")
    assert risk_of("read_messages") == "read_only" and "read_messages" in _UNTRUSTED


def test_send_message_hands_the_text_to_messages(mac, gate):
    mac.reply("ok\n")
    out = _tool("send_message").invoke({"to": "+1 (555) 010-2000", "text": 'Running 15 min late — "sorry"'})
    assert out["to"] == "+15550102000" and out["text"] == 'Running 15 min late — "sorry"'
    assert "handed to Messages" in out["note"] and "not confirmed" in out["note"]
    s = mac.script()
    assert 'first account whose service type is iMessage' in s
    assert 'send "Running 15 min late — \\"sorry\\"" to participant "+15550102000" of svc' in s


def test_send_message_takes_an_email_handle(mac, gate):
    mac.reply("ok\n")
    assert _tool("send_message").invoke({"to": "sam@icloud.com", "text": "hi"})["to"] == "sam@icloud.com"


def test_send_message_refuses_a_name_and_points_at_contacts(mac, gate):
    # The gate shows the arguments: a number the user can check, never a name Messages would
    # have to guess a person from.
    mark = egress.next_seq()
    out = _err("send_message", {"to": "Sam", "text": "hi"})
    assert "search_contacts" in out and "phone number or email" in out
    assert _err("send_message", {"to": "+15550102000", "text": "  "}).startswith("Error:")
    assert mac.calls == [] and egress.events_since(mark) == []


def test_send_message_is_on_the_egress_ledger(mac, gate):
    mac.reply("ok\n")
    mark = egress.next_seq()
    _tool("send_message").invoke({"to": "+15550102000", "text": "héllo"})
    (ev,) = egress.events_since(mark)
    assert (ev.channel, ev.host, ev.status) == ("message", "+15550102000", egress.SENT)
    assert ev.n_bytes == len("héllo".encode("utf-8"))


def test_a_failed_send_is_an_error_and_still_on_the_ledger(mac, gate):
    # Recorded before the attempt, like every other chokepoint: the ledger never under-reports.
    mac.reply("", 1, "execution error: Messages got an error: Can’t get participant. (-1728)")
    mark = egress.next_seq()
    assert "Can’t get participant" in _err("send_message", {"to": "+15550102000", "text": "hi"})
    assert len(egress.events_since(mark)) == 1


def test_airgap_refuses_a_send_before_anything_runs(mac, gate, monkeypatch):
    monkeypatch.setitem(gate._data["runtime"], "airgap", True)
    mark = egress.next_seq()
    out = _err("send_message", {"to": "+15550102000", "text": "hi"})
    assert "Air-gap is ON" in out and "Nothing was sent" in out
    assert mac.calls == []
    assert [e.status for e in egress.events_since(mark)] == [egress.BLOCKED]


def test_send_message_times_out_without_a_retry(mac, gate, monkeypatch):
    from tools import applescript

    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(applescript, "_run", slow)
    out = _err("send_message", {"to": "+15550102000", "text": "hi"})
    assert "may or may not have been sent" in out and "do not send it again" in out


def test_send_message_never_launches_anything_off_macos(monkeypatch, gate):
    from tools import applescript
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    mark = egress.next_seq()
    assert "only available on macOS" in _err("send_message", {"to": "+15550102000", "text": "hi"})
    assert egress.events_since(mark) == []


# ── send_message: the gate ───────────────────────────────────────────────────────────────────

def test_a_send_always_faces_the_human(gate):
    prev = policy.tier()
    try:
        assert not policy.approves("send_message", "destructive", {})
        policy.set_gate_off(True)                                    # /policy open, --yolo
        assert policy.approves("run_shell", "destructive", {"command": "ls"})
        assert not policy.approves("send_message", "destructive", {})
        assert not policy.approves("send_message", "read_only", {})  # a /policy risk override
    finally:
        policy.set_tier(prev)
        policy._tier_before_gate_off = None


def test_always_allow_never_covers_a_send(gate):
    from nodes.approval import _apply_always_grants
    from tools import registry
    from tui.ui import approval

    decision = approval._always_allow(
        [{"id": "c1", "name": "send_message", "args": {"to": "+1555", "text": "x"}}], lambda _p: "")
    assert decision["tools"] == []
    _apply_always_grants({"approved": True, "tools": ["send_message"], "shell_grants": []})
    assert registry.TOOL_RISK["send_message"] == "destructive"


def test_headless_yolo_still_refuses_a_send(gate, capsys):
    from app import headless

    prev = policy.tier()
    try:
        policy.set_gate_off(True)
        decision = headless.headless_approver({"type": "approval_request", "tool_calls": [
            {"id": "c1", "name": "send_message", "args": {"to": "+1555", "text": "x"}},
            {"id": "c2", "name": "create_note", "args": {"title": "t"}},
        ]})
        assert decision == {"approved_ids": ["c2"]}
        assert "send_message" in capsys.readouterr().err
    finally:
        policy.set_tier(prev)
        policy._tier_before_gate_off = None


def test_headless_never_offers_yolo_for_a_send(gate, capsys):
    from app import headless

    call = {"id": "c1", "name": "send_message", "args": {"to": "+1555", "text": "x"}}
    assert headless.headless_approver({"type": "approval_request", "tool_calls": [call]}) is False
    err = capsys.readouterr().err
    assert "--yolo would not allow send_message" in err and "re-run with --yolo" not in err


def test_the_gate_prompt_says_why_a_send_is_asking(gate, monkeypatch):
    from nodes import approval as approval_mod

    seen = {}
    quarantine.reset_turn()
    policy_tier = policy.tier()
    try:
        policy.set_gate_off(True)                                    # nothing else would prompt
        monkeypatch.setattr(approval_mod, "interrupt", lambda payload: seen.update(payload) or True)
        call = {"id": "c1", "name": "send_message", "args": {"to": "+1555", "text": "x"}}
        state = {"messages": [types.SimpleNamespace(content="", tool_calls=[call])], "plan": []}
        cmd = approval_mod.approval_node(state)
        assert cmd.goto == "tools"
        assert seen["tool_calls"][0]["name"] == "send_message"
        assert any("always asks" in n for n in seen["notes"])
    finally:
        policy.set_tier(policy_tier)
        policy._tier_before_gate_off = None


def _gate_notes(monkeypatch, msgs):
    """The notes the approval prompt shows for a gated send at the end of `msgs`."""
    from nodes import approval as approval_mod

    seen = {}
    quarantine.reset_turn()
    policy_tier = policy.tier()
    try:
        policy.set_gate_off(True)
        monkeypatch.setattr(approval_mod, "interrupt", lambda payload: seen.update(payload) or True)
        approval_mod.approval_node({"messages": msgs, "plan": [], "context": ""})
    finally:
        policy.set_tier(policy_tier)
        policy._tier_before_gate_off = None
    return seen["notes"]


def test_the_gate_prompt_names_who_a_number_belongs_to(gate, monkeypatch):
    """Run 51 showed a bare +13057108702 at the gate. The card that produced it is in the
    conversation, so the prompt can say whose number it is."""
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    card = ("[{'name': 'Ian Smith', 'organization': 'Acme', "
            "'phones': [{'label': 'mobile', 'value': '(305) 710-8702'}]}, "
            "{'name': 'Brian Ianson', 'phones': [{'label': 'home', 'value': '(312) 879-2860'}]}]")
    send = {"id": "c2", "name": "send_message", "args": {"to": "+13057108702", "text": "hi"}}
    msgs = [HumanMessage(content="text ian hi"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "search_contacts", "args": {"query": "Ian"}}]),
            ToolMessage(content=card, tool_call_id="c1", name="search_contacts"),
            AIMessage(content="", tool_calls=[send])]
    notes = _gate_notes(monkeypatch, msgs)
    assert any("+13057108702 is Ian Smith" in n and "mobile" in n for n in notes), notes


def test_the_gate_prompt_says_when_the_user_typed_the_number_or_nobody_did(gate, monkeypatch):
    from langchain.messages import AIMessage, HumanMessage

    send = {"id": "c1", "name": "send_message", "args": {"to": "+14155550199", "text": "hi"}}
    typed = [HumanMessage(content="text 415-555-0199 hi"), AIMessage(content="", tool_calls=[send])]
    assert any("+14155550199" in n and "you typed" in n for n in _gate_notes(monkeypatch, typed))

    nowhere = [HumanMessage(content="text ian hi"), AIMessage(content="", tool_calls=[send])]
    notes = _gate_notes(monkeypatch, nowhere)
    assert any("+14155550199" in n and quarantine.UNKNOWN_HANDLE_NOTE in n for n in notes), notes


def test_owner_of_reads_a_contact_card_however_the_number_is_written():
    from tools.contacts import owner_of

    card = ("[{'name': \"Sam O'Brien\", 'emails': [{'label': 'home', 'value': 'sam@example.com'}], "
            "'phones': [{'label': 'mobile', 'value': '+1 555 010 2000'}, {'label': 'work', 'value': '555-0101'}]}, "
            "{'name': 'Ian Smith', 'phones': [{'label': '', 'value': '(305) 710-8702'}]}]")
    assert owner_of("+15550102000", card) == ("Sam O'Brien", "mobile")
    assert owner_of("5550101", card) == ("Sam O'Brien", "work")
    assert owner_of("SAM@example.com", card) == ("Sam O'Brien", "home")
    assert owner_of("3057108702", card) == ("Ian Smith", "")
    assert owner_of("+13128792860", card) is None
    assert owner_of("+13128792860", "not a card at all") is None


# ── read_messages ────────────────────────────────────────────────────────────────────────────

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def _ns(dt: datetime) -> int:
    return int((dt - APPLE_EPOCH).total_seconds() * 1_000_000_000)


def _blob(text: str) -> bytes:
    """An `attributedBody` typedstream as Messages writes it: the text sits after the NSString
    class marker, behind a one-byte length (or 0x81 + two little-endian bytes when ≥ 128)."""
    raw = text.encode("utf-8")
    length = bytes([len(raw)]) if len(raw) < 128 else b"\x81" + len(raw).to_bytes(2, "little")
    return (b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00"
            b"\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+"
            + length + raw + b"\x86\x84\x02iI\x01\x0c\x92\x84\x84\x84\x0cNSDictionary\x00")


@pytest.fixture
def history(tmp_path, monkeypatch):
    """A chat.db with two conversations: Sam (+15550102000) and a group."""
    from tools import messages

    path = tmp_path / "chat.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
                              handle_id INTEGER, date INTEGER, is_from_me INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+15550102000'), (2, 'jonah@example.com');
        INSERT INTO chat VALUES (1, 'iMessage;-;+15550102000', '+15550102000', ''),
                                (2, 'iMessage;+;chat99', 'chat99', 'Dinner club');
    """)
    rows = [
        (1, "Dinner at 7?", None, 1, datetime(2026, 9, 5, 18, 0, tzinfo=timezone.utc), 0, 1),
        (2, None, _blob("Yes — the Thai place"), 1, datetime(2026, 9, 5, 18, 5, tzinfo=timezone.utc), 1, 1),
        (3, None, _blob("x" * 300), 2, datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc), 0, 2),
        (4, "See you at dinner", None, 2, datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc), 0, 2),
        (5, None, None, 1, datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc), 0, 1),   # an image: no text
    ]
    for rowid, text, blob, handle, when, mine, chat in rows:
        db.execute("INSERT INTO message VALUES (?,?,?,?,?,?)", (rowid, text, blob, handle, _ns(when), mine))
        db.execute("INSERT INTO chat_message_join VALUES (?,?)", (chat, rowid))
    db.commit()
    db.close()
    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    monkeypatch.setattr(messages, "_db_path", lambda: path)
    # The group names come from the Messages app; no test may ask the real one.
    monkeypatch.setattr(messages, "_groups", lambda: [])
    return path


def test_typedstream_text_short_and_long():
    from tools.messages import _attributed_text
    assert _attributed_text(_blob("Yes — the Thai place")) == "Yes — the Thai place"
    assert _attributed_text(_blob("y" * 500)) == "y" * 500
    assert _attributed_text(b"not a typedstream") == ""
    assert _attributed_text(None) == ""


def test_read_messages_newest_first_with_decoded_bodies(history):
    out = _tool("read_messages").invoke({})
    assert [m["text"] for m in out] == ["See you at dinner", "x" * 300, "Yes — the Thai place", "Dinner at 7?"]
    assert out[0]["chat"] == f"{_ref('chat99')} · Dinner club" and out[0]["from"] == "jonah@example.com"
    assert out[2]["from"] == "me" and out[2]["chat"] == "+15550102000"
    local = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc).astimezone().isoformat(timespec="minutes")[:16]
    assert out[0]["when"] == local


def test_read_messages_with_one_person_matches_the_number_however_it_is_written(history):
    out = _tool("read_messages").invoke({"contact": "(555) 010-2000"})
    assert [m["text"] for m in out] == ["Yes — the Thai place", "Dinner at 7?"]
    out = _tool("read_messages").invoke({"contact": "jonah@example.com", "limit": 1})
    assert [m["text"] for m in out] == ["See you at dinner"]


def test_read_messages_filters_by_text_including_decoded_bodies(history):
    out = _tool("read_messages").invoke({"query": "thai"})
    assert [m["text"] for m in out] == ["Yes — the Thai place"]
    assert _tool("read_messages").invoke({"query": "zzz"}) == "No messages match."


def test_one_persons_messages_are_found_however_far_back_they_are(history, monkeypatch):
    # The person is matched in SQL, not among the newest N rows of every chat: a busy group
    # chat must not turn "when did I last talk to Sam?" into "no history".
    from tools import messages
    monkeypatch.setattr(messages, "_SCAN", 3)                  # the three NEWEST rows hold no text of Sam's
    out = _tool("read_messages").invoke({"contact": "+1 555 010 2000"})
    assert [m["text"] for m in out] == ["Yes — the Thai place", "Dinner at 7?"]
    assert _tool("read_messages").invoke({"contact": "+1 555 999 0000"}) == \
        "No messages with +15559990000 in the history."


def test_a_text_search_says_when_it_only_covered_the_newest_messages(history, monkeypatch):
    from tools import messages
    monkeypatch.setattr(messages, "_SCAN", 2)
    out = _tool("read_messages").invoke({"query": "thai"})     # the match is older than the scan
    assert out.startswith("No messages match in the newest 2") and "older" in out
    out = _tool("read_messages").invoke({"query": "dinner", "limit": 5})
    assert out[0]["text"] == "See you at dinner"
    assert "newest 2" in out[-1]["note"]                       # found some, but not everything was read


def test_read_messages_says_how_to_grant_access_when_macos_denies_it(history, monkeypatch):
    from tools import messages

    def denied(path):
        raise sqlite3.OperationalError("unable to open database file: authorization denied")

    monkeypatch.setattr(messages, "_connect", denied)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    out = _err("read_messages", {})
    assert "Full Disk Access" in out and "Privacy & Security" in out
    # The grant goes to the app that launched Saturn, named — a 4b told the user to grant it
    # to Messages when the text said only "the terminal app".
    assert "Visual Studio Code" in out and "not Messages" in out


def test_the_access_remedy_names_the_launching_app_or_says_which_one_it_means(monkeypatch):
    from tools import messages

    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert messages._terminal_app() == "iTerm"
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert messages._terminal_app() == "Terminal"
    monkeypatch.setenv("TERM_PROGRAM", "WarpTerminal")
    assert messages._terminal_app() == "Warp"
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")             # unknown: the raw name, not a guess
    assert messages._terminal_app() == "ghostty"
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert messages._terminal_app() == "the app you launched Saturn from"


def test_read_messages_opens_the_history_read_only(history):
    from tools import messages
    db = messages._connect(history)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM message")
    db.close()


def test_read_messages_reports_non_mac_honestly(monkeypatch):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert "only available on macOS" in _err("read_messages", {})


def test_read_messages_is_not_egress(history):
    mark = egress.next_seq()
    _tool("read_messages").invoke({})
    assert egress.events_since(mark) == []


# ── the benchmark never acts on the user's real world ────────────────────────────────────────

def test_the_benchmark_approver_never_lets_a_send_or_an_app_write_through():
    """benchmark.py answers the gate itself so a run does not block on a human. That yes covers
    the workspace tools its tasks are about — never a text message, a shortcut, or a change to
    the user's real mail, calendar, notes or reminders."""
    import benchmark

    prompted: list = []
    decision = benchmark.bench_approver({"type": "approval_request", "tool_calls": [
        {"id": "c1", "name": "write_file", "args": {}},
        {"id": "c2", "name": "send_message", "args": {}},
        {"id": "c3", "name": "run_shortcut", "args": {}},
        {"id": "c4", "name": "update_mail", "args": {}},
        {"id": "c5", "name": "delete_calendar_event", "args": {}},
        {"id": "c6", "name": "draft_mail", "args": {}},
        {"id": "c7", "name": "run_shell", "args": {}},
    ]}, prompted)
    assert decision == {"approved_ids": ["c1", "c7"]}
    assert prompted == ["write_file", "send_message", "run_shortcut", "update_mail",
                        "delete_calendar_event", "draft_mail", "run_shell"]
    assert benchmark.bench_approver({"type": "approval_request", "tool_calls": [
        {"id": "c1", "name": "write_file", "args": {}}]}, prompted) is True
    assert benchmark.bench_approver({"type": "pause"}, prompted) is True


# ── group chats: find_group_chats, send_message(chat=), read_messages(chat=) ──────────────────

FAMILY = ("any;+;chat111", "Family", [("+15550000001", "Mom"), ("+15550000002", "Dad"),
                                       ("+15550102000", "Sam Lee")])
SAM_ALEX = ("any;+;chat222", "", [("+15550102000", "Sam Lee"), ("alex@example.com", "Alex Kim")])
SAM_ALEX_JO = ("any;+;chat333", "", [("+15550102000", "Sam Lee"), ("alex@example.com", "Alex Kim"),
                                      ("+15550000004", "Jo Park")])
CLIMBING = ("any;+;0ed3abc9", "Climbing 🧗", [("alex@example.com", "Alex Kim"), ("+15550000004", "Jo Park"),
                                              ("+15550000009", "")])
BOOK_A = ("any;+;chat555", "Book club", [("+15550000001", "Mom"), ("+15550000004", "Jo Park")])
BOOK_B = ("any;+;chat666", "Book club 2", [("+15550000002", "Dad"), ("+15550000004", "Jo Park")])
GROUPS = (FAMILY, SAM_ALEX, SAM_ALEX_JO, CLIMBING, BOOK_A, BOOK_B)


def _as_groups(*groups) -> str:
    """The output the group listing script prints: one record per group chat — its id, its name
    and its people as handle/full-name pairs."""
    return "".join(US.join([guid, name, "".join(f"{h}{FS}{n}{GS}" for h, n in people)]) + RS
                   for guid, name, people in groups)


def _ref(guid):
    from tools.messages import chat_ref
    return chat_ref(guid)


def test_find_group_chats_is_a_read_only_untrusted_reader():
    from tools.registry import risk_of
    from tools.toolspec import _UNTRUSTED
    # any member can rename a group: its name is someone else's text
    assert risk_of("find_group_chats") == "read_only" and "find_group_chats" in _UNTRUSTED


def test_a_chat_ref_is_short_stable_and_the_same_from_either_source():
    from tools.messages import chat_ref, is_chat_ref
    ref = chat_ref("any;+;chat111")
    assert is_chat_ref(ref) and len(ref) == 6 and ref == chat_ref("any;+;chat111")
    # chat.db spells the service, the scripting bridge says "any": one chat, one ref
    assert chat_ref("iMessage;+;chat111") == ref == chat_ref("chat111")
    assert chat_ref("any;+;chat222") != ref
    assert not is_chat_ref("+15550102000") and not is_chat_ref("Family") and not is_chat_ref("g12")


def test_refs_lengthen_when_two_groups_would_share_one(mac, monkeypatch):
    from tools import messages
    monkeypatch.setattr(messages, "_digest", lambda ident: ("abcde" + ident[-1]) * 7)
    mac.reply(_as_groups(("any;+;chat1", "A", [("+15550000001", "Mom")]),
                         ("any;+;chat2", "B", [("+15550000002", "Dad")])))
    refs = [g["ref"] for g in messages._groups()]
    assert refs == ["gabcde1", "gabcde2"]


def test_the_group_listing_reads_every_group_and_only_groups(mac):
    from tools import messages
    mac.reply(_as_groups(FAMILY, ("any;-;+15550102000", "", [("+15550102000", "Sam Lee")]), CLIMBING))
    groups = messages._groups()
    assert [g["name"] for g in groups] == ["Family", "Climbing 🧗"]
    assert groups[0]["people"][2] == {"handle": "+15550102000", "name": "Sam Lee"}
    # a person with no card is shown by their handle
    assert groups[1]["people"][2] == {"handle": "+15550000009", "name": "+15550000009"}
    s = mac.script()
    assert 'tell application "Messages"' in s and ";+;" in s and "participants" in s


def test_find_group_chats_by_the_groups_name(mac):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"query": "the family chat"})
    assert out == [{"chat": _ref(FAMILY[0]), "name": "Family", "people": ["Mom", "Dad", "Sam Lee"], "size": 3}]
    # an exact name beats a name that only contains the query
    out = _tool("find_group_chats").invoke({"query": "book club"})
    assert [g["name"] for g in out if "chat" in g] == ["Book club"] and "exactly" in out[-1]["note"]


def test_find_group_chats_by_the_people_in_it_ranks_the_exact_group_first(mac):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"query": "Sam and Alex"})
    # One group IS exactly who was named: it alone is returned, and the note says so. Listing
    # the larger group beside it made a 9b ask "which one?" (live run, 2026-10-03).
    assert out[0]["chat"] == _ref(SAM_ALEX[0])
    assert out[0]["name"] is None and out[0]["people"] == ["Sam Lee", "Alex Kim"]
    assert len(out) == 2 and "exactly" in out[1]["note"] and "1 other" in out[1]["note"]
    assert "ask" not in out[1]["note"]


def test_find_group_chats_says_to_ask_when_the_match_is_not_decisive(mac):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"query": "book"})
    assert [g["name"] for g in out if "chat" in g] == ["Book club", "Book club 2"]
    assert "ask_user" in out[-1]["note"]
    out = _tool("find_group_chats").invoke({"query": "Jo"})    # in four groups, exactly in none
    assert len([g for g in out if "chat" in g]) == 4 and "ask_user" in out[-1]["note"]


def test_find_group_chats_matches_a_handle_and_a_name_together(mac):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"query": "climbing alex"})
    assert [g["chat"] for g in out] == [_ref(CLIMBING[0])]


def test_find_group_chats_with_no_match_says_groups_are_not_created_here(mac):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"query": "Priya and Jordan"})
    assert isinstance(out, str) and "No group chat" in out and "start" in out and "Messages" in out
    # the user asked for a GROUP: separate one-to-one texts are not a substitute (a 9b sent
    # two of them in the live run, 2026-10-03)
    assert "one by one" in out and "unless the user asks" in out


def test_find_group_chats_with_no_query_lists_the_groups(mac, monkeypatch):
    mac.reply(_as_groups(*GROUPS))
    out = _tool("find_group_chats").invoke({"limit": 2})
    assert [g["name"] for g in out[:2]] == ["Family", None]
    assert "6 group chats" in out[-1]["note"]


def test_find_group_chats_reports_non_mac_honestly(monkeypatch):
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert "only available on macOS" in _err("find_group_chats", {"query": "family"})


# send_message(chat=)

def test_send_message_to_a_group_sends_to_the_chat_itself(mac, gate):
    mac.reply(_as_groups(*GROUPS))
    ref = _ref(FAMILY[0])
    out = _tool("send_message").invoke({"chat": ref, "text": 'Dinner at 7 — "ok"?'})
    assert out["chat"] == ref and out["group"] == "Family"
    assert out["to"] == ["+15550000001", "+15550000002", "+15550102000"]
    assert out["people"] == ["Mom", "Dad", "Sam Lee"]
    s = mac.script()
    assert 'send "Dinner at 7 — \\"ok\\"?" to chat id "any;+;chat111"' in s
    assert "participant" not in s


def test_a_group_send_puts_every_recipient_on_the_ledger(mac, gate):
    mac.reply(_as_groups(*GROUPS))
    mark = egress.next_seq()
    _tool("send_message").invoke({"chat": _ref(SAM_ALEX[0]), "text": "héllo"})
    events = egress.events_since(mark)
    assert [(e.channel, e.host, e.status) for e in events] == [
        ("message", "+15550102000", egress.SENT), ("message", "alex@example.com", egress.SENT)]
    assert all(e.n_bytes == len("héllo".encode("utf-8")) for e in events)


def test_airgap_refuses_a_group_send_before_it_is_sent(mac, gate, monkeypatch):
    monkeypatch.setitem(gate._data["runtime"], "airgap", True)
    mac.reply(_as_groups(*GROUPS))
    mark = egress.next_seq()
    out = _err("send_message", {"chat": _ref(FAMILY[0]), "text": "hi"})
    assert "Air-gap is ON" in out
    assert not any("send " in mac.script(i) for i in range(len(mac.calls)) if "-e" in mac.calls[i])
    assert [e.status for e in egress.events_since(mark)] == [egress.BLOCKED]


def test_a_group_send_that_times_out_is_not_retried(mac, gate, monkeypatch):
    from tools import messages
    monkeypatch.setattr(messages, "_groups", lambda: messages._parse_groups(_as_groups(*GROUPS)))

    def slow(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(applescript, "_run", slow)
    out = _err("send_message", {"chat": _ref(FAMILY[0]), "text": "hi"})
    assert "may or may not have been sent" in out and "do not send it again" in out


def test_send_message_to_a_chat_that_does_not_exist_sends_nothing(mac, gate):
    mac.reply(_as_groups(*GROUPS))
    mark = egress.next_seq()
    out = _err("send_message", {"chat": "g000000", "text": "hi"})
    assert "find_group_chats" in out
    assert egress.events_since(mark) == []
    assert all("send " not in mac.script(i) for i in range(len(mac.calls)) if "-e" in mac.calls[i])


def test_send_message_takes_one_recipient_or_one_chat_never_both_or_neither(mac, gate):
    mark = egress.next_seq()
    both = _err("send_message", {"to": "+15550102000", "chat": _ref(FAMILY[0]), "text": "hi"})
    assert "not both" in both
    neither = _err("send_message", {"text": "hi"})
    assert "search_contacts" in neither and "find_group_chats" in neither
    assert mac.calls == [] and egress.events_since(mark) == []


def test_a_name_in_to_points_at_both_lookups(mac, gate):
    out = _err("send_message", {"to": "the family chat", "text": "hi"})
    assert "search_contacts" in out and "find_group_chats" in out and "chat=" in out


def test_a_group_send_always_faces_the_human_even_headless_with_yolo(gate, capsys):
    from app import headless

    prev = policy.tier()
    try:
        policy.set_gate_off(True)
        call = {"id": "c1", "name": "send_message", "args": {"chat": "g7f3a2b", "text": "x"}}
        assert not policy.approves("send_message", "destructive", call["args"])
        decision = headless.headless_approver({"type": "approval_request", "tool_calls": [call]})
        assert decision in (False, {"approved_ids": []})
    finally:
        policy.set_tier(prev)
        policy._tier_before_gate_off = None


# routing: the argument names say which path a call is on

@pytest.mark.parametrize("name, args, want", [
    ("send_message", {"to": "+15550102000", "text": "t"}, {"to": "+15550102000", "text": "t"}),
    ("send_message", {"chat": "gabc123", "text": "t"}, {"chat": "gabc123", "text": "t"}),
    # a value in the wrong slot whose form says where it belongs is moved, not refused
    ("send_message", {"to": "gabc123", "text": "t"}, {"chat": "gabc123", "text": "t"}),
    ("send_message", {"chat": "+1 555 010 2000", "text": "t"}, {"to": "+1 555 010 2000", "text": "t"}),
    ("send_message", {"to": "+15550102000", "chat": "", "text": "t"}, {"to": "+15550102000", "text": "t"}),
    ("read_messages", {"contact": "gabc123"}, {"chat": "gabc123"}),
    ("read_messages", {"chat": "sam@example.com", "query": "x"}, {"contact": "sam@example.com", "query": "x"}),
    ("read_messages", {}, {}),
    ("search_contacts", {"query": "Sam"}, {"query": "Sam"}),       # not a messaging tool: untouched
])
def test_route_target_keeps_or_moves_the_recipient(name, args, want):
    from tools.messages import route_target
    assert route_target(name, args) == (want, None)


@pytest.mark.parametrize("name, args, words", [
    ("send_message", {"to": "+15550102000", "chat": "gabc123", "text": "t"}, ["not both"]),
    ("send_message", {"text": "t"}, ["search_contacts", "find_group_chats"]),
    ("read_messages", {"contact": "+15550102000", "chat": "gabc123"}, ["not both"]),
    # a NAME never reaches the gate: a 4b sent to='Priya Jordan' (live run, 2026-10-03)
    ("send_message", {"to": "Priya Jordan", "text": "t"}, ["search_contacts", "find_group_chats", "not a name"]),
    ("read_messages", {"contact": "Sam"}, ["search_contacts", "not a name"]),
    ("send_message", {"chat": "the family chat", "text": "t"}, ["find_group_chats"]),
])
def test_route_target_refuses_two_targets_or_a_send_with_none(name, args, words):
    from tools.messages import route_target
    _args, problem = route_target(name, args)
    assert problem and all(w in problem for w in words)


# the gate names everyone a group send reaches

def test_the_gate_prompt_lists_every_member_of_a_group(gate, mac, monkeypatch):
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    mac.reply(_as_groups(*GROUPS))
    ref = _ref(FAMILY[0])
    send = {"id": "c2", "name": "send_message", "args": {"chat": ref, "text": "hi"}}
    msgs = [HumanMessage(content="tell the family chat hi"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "find_group_chats", "args": {"query": "family"}}]),
            ToolMessage(content=f"[{{'chat': '{ref}', 'name': 'Family'}}]", tool_call_id="c1", name="find_group_chats"),
            AIMessage(content="", tool_calls=[send])]
    notes = _gate_notes(monkeypatch, msgs)
    (note,) = [n for n in notes if ref in n]
    assert '"Family"' in note and "3 people" in note
    for who in ("Mom (+15550000001)", "Dad (+15550000002)", "Sam Lee (+15550102000)"):
        assert who in note


def test_the_gate_prompt_says_when_a_group_cannot_be_resolved(gate, mac, monkeypatch):
    from langchain.messages import AIMessage, HumanMessage

    mac.reply(_as_groups(*GROUPS))
    send = {"id": "c1", "name": "send_message", "args": {"chat": "g000000", "text": "hi"}}
    notes = _gate_notes(monkeypatch, [HumanMessage(content="g000000 hi"), AIMessage(content="", tool_calls=[send])])
    assert any("g000000" in n and "not found" in n for n in notes), notes


# read_messages(chat=)

@pytest.fixture
def group_history(history, monkeypatch):
    """The history fixture's Dinner club group, with Messages naming its people."""
    from tools import messages
    dinner = ("any;+;chat99", "Dinner club", [("jonah@example.com", "Jonah Reyes"), ("+15550102000", "Sam Lee")])
    monkeypatch.setattr(messages, "_groups", lambda: messages._parse_groups(_as_groups(dinner, FAMILY)))
    return _ref(dinner[0])


def test_read_messages_reads_one_whole_group_with_names(group_history):
    out = _tool("read_messages").invoke({"chat": group_history})
    assert [m["text"] for m in out] == ["See you at dinner", "x" * 300]
    assert all(m["from"] == "Jonah Reyes" and m["group"] is True for m in out)
    assert out[0]["chat"] == f"{group_history} · Dinner club"


def test_every_listing_labels_a_group_row_so_the_model_can_follow_it(group_history):
    out = _tool("read_messages").invoke({})
    assert out[0]["chat"] == f"{group_history} · Dinner club" and out[0]["group"] is True
    assert out[2]["chat"] == "+15550102000" and "group" not in out[2]       # a 1:1 row is unchanged


def test_an_unnamed_group_is_labelled_by_its_people(history, monkeypatch):
    from tools import messages
    db = sqlite3.connect(history)
    db.execute("UPDATE chat SET display_name = '' WHERE ROWID = 2")
    db.commit()
    db.close()
    people = [("jonah@example.com", "Jonah Reyes"), ("+15550102000", "Sam Lee"), ("+15550000001", "Mom"),
              ("+15550000002", "Dad"), ("+15550000004", "Jo Park")]
    monkeypatch.setattr(messages, "_groups", lambda: messages._parse_groups(_as_groups(("any;+;chat99", "", people))))
    out = _tool("read_messages").invoke({})
    assert out[0]["chat"] == f"{_ref('chat99')} · with Jonah Reyes, Sam Lee, Mom +2"


def test_read_messages_keeps_handles_when_messages_cannot_name_anyone(history, monkeypatch):
    from tools import messages

    from tools.toolspec import ToolError

    def denied():
        raise ToolError("macOS denied automation access to Messages")

    monkeypatch.setattr(messages, "_groups", denied)
    out = _tool("read_messages").invoke({"chat": _ref("chat99")})
    assert [m["from"] for m in out] == ["jonah@example.com", "jonah@example.com"]
    assert out[0]["chat"] == f"{_ref('chat99')} · Dinner club"


def test_read_messages_with_an_unknown_chat_points_at_find_group_chats(history):
    out = _tool("read_messages").invoke({"chat": "g000000"})
    assert isinstance(out, str) and "g000000" in out and "find_group_chats" in out


def test_read_messages_refuses_a_contact_and_a_chat_together(history):
    assert "not both" in _err("read_messages", {"contact": "+15550102000", "chat": _ref("chat99")})


def test_a_contact_filter_still_finds_their_group_messages_labelled(group_history):
    out = _tool("read_messages").invoke({"contact": "jonah@example.com"})
    assert out[0]["chat"] == f"{group_history} · Dinner club" and out[0]["group"] is True
