"""
Messages tools — send_message, read_messages.

`send_message` sends an iMessage through the Messages app (AppleScript). It is the one tool
here that puts the user's words in front of another person, so it is wired as an EGRESS
CHOKEPOINT (spec: docs/superpowers/specs/2026-09-06-macos-apps.md):

  - `egress.check` first (air-gap refuses), then `egress.record` with the recipient as the host
    label — recorded before the attempt, like every chokepoint, so the ledger never
    under-reports;
  - tier `destructive`, and `trust/policy.ALWAYS_ASKS` on top: no tier, open gate, `/policy
    risk` override or always-allow lets a send through, and headless refuses it even with
    --yolo. The human reads the exact recipient and text first, every time;
  - `tests/test_no_new_egress.py` only sees network-client imports, and `osascript` is not one
    — `tests/test_messages.py` pins the check/record wiring instead.

`to` is a phone number or an email address, never a name: the gate shows the arguments, and a
handle is something the user can check (`search_contacts` resolves a name). The script was
compiled, not sent, when this was written (2026-10-01) — nothing is sent without the user.
A send is "handed to Messages": a recipient who is not on iMessage fails later, inside
Messages, where a script cannot see it.

`read_messages` reads the history from `~/Library/Messages/chat.db` (AppleScript has no
message element). macOS only opens that file for an app with Full Disk Access; without it the
tool says how to grant it. On recent macOS `message.text` is often NULL and the text lives in
`attributedBody`, an NSAttributedString typedstream — `_attributed_text` pulls the string out.
The schema is undocumented; the reader uses six long-stable columns and nothing else. It is
`read_only` AND `untrusted=True` (texts are written by other people), opened read-only, and not
egress.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools import applescript
from tools.applescript import AppleScriptError, quote
from tools.toolspec import ToolError, register_tool
from trust import egress

_SEND_TIMEOUT = 60.0
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE = re.compile(r"\+?[\d\s().-]{7,}")


def _handle(to: str) -> "str | None":
    """`to` as a Messages handle — an email address as written, a phone number reduced to its
    digits (and a leading +) — or None when it is neither (a name, say)."""
    to = str(to or "").strip()
    if _EMAIL.fullmatch(to):
        return to
    if _PHONE.fullmatch(to):
        digits = re.sub(r"\D", "", to)
        if len(digits) >= 7:
            return ("+" if to.startswith("+") else "") + digits
    return None


@register_tool("destructive")
def send_message(to: str, text: str):
    """Send a text message (iMessage) to one person through the Messages app. `to` is their
    phone number or iMessage email address — get it from search_contacts, never guess it and
    never pass a name. `text` is the exact message. The user is always shown the recipient and
    the text and must approve before anything is sent."""
    handle = _handle(to)
    text = str(text or "")
    if handle is None:
        raise ToolError("`to` must be a phone number or email address, not a name — look the "
                        "person up with search_contacts and pass the number from their card")
    if not text.strip():
        raise ToolError("send_message needs the text to send")
    applescript.mac_only()
    blocked = egress.check("message", handle, text)
    if blocked:
        raise ToolError(blocked)
    egress.record("message", handle, text, provider="imessage", n_bytes=len(text.encode("utf-8")))
    script = f"""
tell application "Messages"
  set svc to first account whose service type is iMessage
  send {quote(text)} to participant {quote(handle)} of svc
end tell
return "ok\""""
    try:
        applescript.run(script, timeout=_SEND_TIMEOUT, app="Messages")
    except AppleScriptError as exc:
        if "timed out" in str(exc):
            raise ToolError("Messages did not answer in time; the message may or may not have "
                            "been sent — check Messages, and do not send it again") from exc
        raise ToolError(str(exc)) from exc
    return {"to": handle, "text": text,
            "note": "handed to Messages for delivery as an iMessage; delivery is not confirmed "
                    "— a recipient who is not on iMessage shows 'Not Delivered' in Messages"}


# ── the history reader ───────────────────────────────────────────────────────────────────────

_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
_MAX = 100
_SCAN = 4000          # newest rows examined per call (a text filter runs in Python: bodies decode there)
_NS_MARKER = b"NSString"


def _db_path() -> Path:
    return Path.home() / "Library" / "Messages" / "chat.db"


def _connect(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)


def _attributed_text(blob) -> str:
    """The message text inside an `attributedBody` typedstream, or "". The string follows the
    NSString class marker and a five-byte preamble, behind its length: one byte, or 0x81 and two
    little-endian bytes for 128 and over (0x82: four). Anything else is not a message body."""
    if not blob:
        return ""
    data = bytes(blob)
    at = data.find(_NS_MARKER)
    if at < 0:
        return ""
    body = data[at + len(_NS_MARKER) + 5:]
    if not body:
        return ""
    if body[0] == 0x81:
        size, start = int.from_bytes(body[1:3], "little"), 3
    elif body[0] == 0x82:
        size, start = int.from_bytes(body[1:5], "little"), 5
    else:
        size, start = body[0], 1
    return body[start:start + size].decode("utf-8", errors="replace")


def _when(raw) -> str:
    """Messages' timestamp (nanoseconds since 2001 on current macOS, seconds on old ones) as
    local wall-clock time."""
    try:
        value = int(raw or 0)
    except (TypeError, ValueError):
        return ""
    seconds = value / 1_000_000_000 if value > 10**11 else value
    return (_APPLE_EPOCH + timedelta(seconds=seconds)).astimezone().isoformat(timespec="minutes")[:16]


def _same_person(contact: str, *handles) -> bool:
    """Whether `contact` names one of `handles`: emails compare case-insensitively, phone
    numbers by their last ten digits, however either side is punctuated."""
    want = contact.strip().lower()
    digits = re.sub(r"\D", "", want)
    for h in handles:
        h = str(h or "").lower()
        if not h:
            continue
        if "@" in want:
            if want == h:
                return True
        elif len(digits) >= 7 and re.sub(r"\D", "", h)[-10:] == digits[-10:]:
            return True
    return False


_QUERY = """
SELECT m.text, m.attributedBody, m.date, m.is_from_me, h.id, c.chat_identifier, c.display_name
FROM message m
LEFT JOIN handle h ON h.ROWID = m.handle_id
LEFT JOIN chat_message_join j ON j.message_id = m.ROWID
LEFT JOIN chat c ON c.ROWID = j.chat_id
{where}
ORDER BY m.date DESC
LIMIT ?
"""


def _person_filter(db, contact: str) -> "tuple[str, list] | None":
    """The WHERE clause keeping one person's messages, or None when nobody in the history is
    `contact`. The person is found among the handles and chats first (small tables, compared by
    `_same_person` however the number is punctuated) and the messages are then selected by
    row id — so their texts are found however many newer messages other chats hold."""
    handles = [rowid for rowid, hid in db.execute("SELECT ROWID, id FROM handle")
               if _same_person(contact, hid)]
    chats = [rowid for rowid, cid in db.execute("SELECT ROWID, chat_identifier FROM chat")
             if _same_person(contact, cid)]
    clauses = [f"{column} IN ({', '.join('?' * len(ids))})"
               for column, ids in (("m.handle_id", handles), ("j.chat_id", chats)) if ids]
    if not clauses:
        return None
    return "WHERE " + " OR ".join(clauses), handles + chats


@register_tool("read_only", untrusted=True)
def read_messages(contact: str = "", query: str = "", limit: int = 20):
    """Read the user's text-message history from the Messages app, newest first. `contact`
    narrows it to one person — their phone number or email address (search_contacts gives it);
    `query` keeps only messages containing that text. Returns when, from ('me' for the user),
    chat and text. Use it for "what did Sam say about …" and "when did I last talk to …"."""
    applescript.mac_only()
    contact, needle = str(contact or "").strip(), str(query or "").strip().lower()
    limit = max(1, min(int(limit or 20), _MAX))
    person = _handle(contact) if contact else None
    if contact and person is None:
        raise ToolError("`contact` must be a phone number or email address — look the person "
                        "up with search_contacts first")
    scan = _SCAN if (contact or needle) else limit * 4
    try:
        db = _connect(_db_path())
        try:
            where, ids = "", []
            if person:
                found = _person_filter(db, person)
                if found is None:
                    return f"No messages with {person} in the history."
                where, ids = found
            rows = db.execute(_QUERY.format(where=where), (*ids, scan)).fetchall()
        finally:
            db.close()
    except sqlite3.Error as exc:
        if "authorization denied" in str(exc) or "unable to open" in str(exc):
            raise ToolError(
                "macOS did not let this terminal read the Messages history. Give the terminal "
                "app Full Disk Access under System Settings > Privacy & Security > Full Disk "
                "Access, restart it, and ask again") from exc
        raise ToolError(f"the Messages history could not be read: {exc}") from exc
    out = []
    for text, blob, date, mine, handle, chat_id, chat_name in rows:
        body = (text or "").strip() or _attributed_text(blob).strip()
        if not body:
            continue                      # an attachment or a reaction: nothing to read
        if needle and needle not in body.lower():
            continue
        out.append({"when": _when(date), "from": "me" if mine else (handle or "unknown"),
                    "chat": chat_name or chat_id or handle or "", "text": body})
        if len(out) >= limit:
            break
    # Fewer than asked for AND the scan stopped at its cap: older messages were never read,
    # so "none" (or "these are all") would be a claim about history nobody looked at.
    if len(out) < limit and len(rows) >= scan and needle:
        searched = f"the newest {scan:,} messages" + (f" with {person}" if person else "")
        if not out:
            return (f"No messages match in {searched}; older ones were not searched — narrow "
                    "it with contact=, or try a more specific query.")
        out.append({"note": f"only {searched} were searched; older ones may match too"})
    return out or "No messages match."
