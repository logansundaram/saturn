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
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
                              handle_id INTEGER, date INTEGER, is_from_me INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+15550102000'), (2, 'jonah@example.com');
        INSERT INTO chat VALUES (1, '+15550102000', ''), (2, 'chat99', 'Dinner club');
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
    assert out[0]["chat"] == "Dinner club" and out[0]["from"] == "jonah@example.com"
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
    out = _err("read_messages", {})
    assert "Full Disk Access" in out and "Privacy & Security" in out


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
