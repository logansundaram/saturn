"""
Messages tools — send_message, read_messages, find_group_chats.

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

Group chats (spec: docs/superpowers/specs/2026-10-03-imessage-group-chats-design.md). A group
is named by a short CHAT REF (`g` + hex, `chat_ref`) that only `find_group_chats` hands out; the
argument name says which path a call is on — `to=`/`contact=` for one person, `chat=` for a
group, never both (`route_target`, which the agent's hygiene runs before any gate). Groups are
found, never created. `find_group_chats` reads the Messages app's own chat list over AppleScript
(`_groups`: one bulk pass, ~0.2-0.8s for 30-50 chats measured 2026-10-03, and no Full Disk
Access) — the `chats` element that failed with -10000 on 2026-09-06 iterates now. A group send
goes to the chat itself (`send … to chat id`), records one egress event per recipient, and the
gate names every member (`describe_group`, resolved live at approval).
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools import applescript
from tools.applescript import AS_FS, AS_GS, AS_RS, AS_US, FS, GS, AppleScriptError, quote, records
from tools.toolspec import ToolError, register_tool
from trust import egress

_SEND_TIMEOUT = 60.0

# The app a Full Disk Access grant must go to is the one that launched Saturn — macOS charges
# the request to that process, not to Messages and not to Python. TERM_PROGRAM is how the
# terminals announce themselves; a name not listed is shown as written rather than guessed.
_TERMINALS = {"apple_terminal": "Terminal", "iterm.app": "iTerm", "vscode": "Visual Studio Code",
              "warpterminal": "Warp", "hyper": "Hyper", "tmux": "tmux"}


def _terminal_app() -> str:
    """The launching app by name, for the remedy text. A 4b told the user to grant Messages
    the access when the text said only "the terminal app" — the app has to be named, and the
    wrong candidate ruled out."""
    raw = (os.environ.get("TERM_PROGRAM") or "").strip()
    if not raw:
        return "the app you launched Saturn from"
    return _TERMINALS.get(raw.lower(), raw)
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


# ── group chats ──────────────────────────────────────────────────────────────────────────────

_REF_LEN = 5
_REF = re.compile(r"g[0-9a-f]{5,40}")


def _digest(ident: str) -> str:
    """The hash behind a chat ref. It is taken over the identifier AFTER the service prefix
    (`any;+;chat8…` from the Messages app, `iMessage;+;chat8…` in chat.db, or the bare
    `chat8…`), so both sources name one chat by one ref."""
    return hashlib.sha1(str(ident or "").rsplit(";", 1)[-1].encode("utf-8")).hexdigest()


def chat_ref(ident: str, n: int = _REF_LEN) -> str:
    """The short ref the model passes as `chat=`: `g` + the first `n` hex digits of the digest.
    A raw `any;+;chat823749…` is 30 characters a 9b miscopies; six it does not."""
    return "g" + _digest(ident)[:n]


def is_chat_ref(text) -> bool:
    return bool(_REF.fullmatch(str(text or "").strip()))


def _matches_ref(ident: str, ref: str) -> bool:
    return _digest(ident).startswith(str(ref or "").strip()[1:])


def _assign_refs(idents: list) -> list:
    """One ref per identifier, lengthened for all of them until no two share one — so a ref is
    never ambiguous, and `_matches_ref` still finds a chat by its longer ref."""
    n = _REF_LEN
    while True:
        refs = [chat_ref(i, n) for i in idents]
        if len(set(refs)) == len(refs) or n >= 40:
            return refs
        n += 1


# One record per group chat: its id, its name ("" when unnamed) and its people as
# handle FS full-name GS pairs. 1:1 chats (`;-;` in the id) are skipped in the script itself.
# The ids are bound to a variable first: `repeat with g in (id of chats)` hands Messages a lazy
# reference it cannot resolve (-1700, run live 2026-10-03).
_GROUPS_SCRIPT = f"""
tell application "Messages"
  set out to ""
  set ids to id of chats
  repeat with g in ids
    set g to contents of g
    if g contains ";+;" then
      set c to chat id g
      set nm to name of c
      if nm is missing value then set nm to ""
      set hs to handle of participants of c
      set fs to full name of participants of c
      set ps to ""
      repeat with i from 1 to count of hs
        set f to item i of fs
        if f is missing value then set f to ""
        set ps to ps & (item i of hs) & {AS_FS} & f & {AS_GS}
      end repeat
      set out to out & g & {AS_US} & nm & {AS_US} & ps & {AS_RS}
    end if
  end repeat
  return out
end tell"""


def _parse_groups(output: str) -> list:
    """[{ref, guid, name, people: [{handle, name}]}] from the listing script's output, in the
    order Messages lists its chats. A person with no Contacts card is named by their handle."""
    rows = []
    for rec in records(output):
        if len(rec) < 3 or ";+;" not in rec[0]:
            continue
        people = []
        for item in rec[2].split(GS):
            if FS not in item:
                continue
            handle, name = (part.strip() for part in item.split(FS, 1))
            if handle:
                people.append({"handle": handle, "name": name or handle})
        rows.append({"guid": rec[0].strip(), "name": rec[1].strip(), "people": people})
    for row, ref in zip(rows, _assign_refs([r["guid"] for r in rows])):
        row["ref"] = ref
    return rows


def _groups() -> list:
    """Every group chat the Messages app knows, with its people. The seam tests replace."""
    try:
        return _parse_groups(applescript.run(_GROUPS_SCRIPT, app="Messages"))
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc


def _group(ref: str) -> "dict | None":
    ref = str(ref or "").strip()
    return next((g for g in _groups() if g["ref"] == ref), None) if is_chat_ref(ref) else None


def _label(name: str, people: list) -> str:
    """A group as the model and the user read it: its name, or the people in it."""
    if name:
        return name
    names = [p["name"] if isinstance(p, dict) else str(p) for p in people]
    if not names:
        return "unnamed group"
    more = f" +{len(names) - 3}" if len(names) > 3 else ""
    return "with " + ", ".join(names[:3]) + more


def describe_group(ref: str) -> str:
    """Who a group send reaches, for the gate prompt — resolved from Messages NOW, never taken
    from the model's words: `group g7f3a2 "Family" — 3 people: Mom (+1555…), …`."""
    try:
        group = _group(ref)
    except ToolError as exc:
        return f"group {ref} could not be looked up in Messages ({exc}) — the send will be refused"
    if group is None:
        return f"group {ref} was not found in Messages — the send will be refused"
    title = f'group {ref} "{group["name"]}"' if group["name"] else f"unnamed group {ref}"
    people = ", ".join(f'{p["name"]} ({p["handle"]})' if p["name"] != p["handle"] else p["handle"]
                       for p in group["people"])
    n = len(group["people"])
    return f"{title} — {n} {'person' if n == 1 else 'people'}: {people}"


# Which argument names one person, per messaging tool, and whether a call needs a target.
_TARGETS = {"send_message": ("to", True), "read_messages": ("contact", False)}


def route_target(name: str, args: dict) -> "tuple[dict, str | None]":
    """(the call's arguments with its recipient in the right slot, the problem or None). One
    person is `to`/`contact`, a group is `chat`, never both. A value whose FORM says which it
    is (a `g…` ref, a phone number or email) is moved to its slot instead of refused — the gate
    shows the corrected call. The agent's hygiene runs this before any gate; the tools run it
    again, so a direct call is held to the same rule."""
    if name not in _TARGETS or not isinstance(args, dict):
        return args, None
    key, required = _TARGETS[name]
    raw = args.get(key)
    person, chat = str(raw or "").strip(), str(args.get("chat") or "").strip()
    if person and not chat and is_chat_ref(person):
        person, chat = "", person
    elif chat and not person and _handle(chat):
        person, raw, chat = chat, args.get("chat"), ""
    if person and chat:
        return args, (f"{name} takes {key}= (one person) or chat= (a group chat from "
                      "find_group_chats), not both — make one call per conversation")
    if required and not person and not chat:
        return args, (f"{name} needs a recipient: for one person, {key}= their number or address "
                      "from search_contacts; for a group conversation, chat= the ref from "
                      "find_group_chats")
    # A name never reaches the gate as a recipient (a 4b sent to='Priya Jordan', 2026-10-03).
    if person and _handle(person) is None:
        return args, (f"`{key}` must be a phone number or email address, not a name — for one "
                      "person, look them up with search_contacts and use the number or address "
                      "from their card; for a group conversation, find it with find_group_chats "
                      "and pass chat= instead")
    if chat and not is_chat_ref(chat):
        return args, (f"`chat` must be a chat ref from find_group_chats (like g7f3a2), not "
                      f"{chat!r} — find the group with find_group_chats first")
    out = {k: v for k, v in args.items() if k not in (key, "chat")}
    if person:
        out[key] = raw
    if chat:
        out["chat"] = chat
    return out, None


_STOPWORDS = {"the", "a", "an", "and", "&", "with", "group", "groupchat", "chat", "gc", "thread",
              "text", "texts", "my", "our", "in", "of", "to", "conversation"}


def _words(text: str) -> list:
    return [w for w in re.split(r"[^\w@+.]+", str(text or "").lower()) if w]


def _rank(group: dict, query: str, terms: list) -> "tuple | None":
    """(tier, size) for a group the query matches, or None. Every term must match the group's
    name or one of its people. Tier 0: the query IS the name; 1: every term is in the name;
    2: the terms name exactly the group's people; 3: anything else that matches."""
    name_words = _words(group["name"])
    in_name = [t for t in terms if any(w.startswith(t) for w in name_words)]
    matched_people = set()
    for t in terms:
        hits = {i for i, p in enumerate(group["people"])
                if any(w.startswith(t) for w in _words(p["name"]))
                or (len(t) >= 4 and t in p["handle"].lower())}
        if not hits and t not in in_name:
            return None
        matched_people |= hits
    size = len(group["people"])
    norm = " ".join(_words(query))
    if group["name"] and (" ".join(name_words) in (norm, " ".join(terms))):
        return 0, size
    if len(in_name) == len(terms):
        return 1, size
    if len(matched_people) == size:
        return 2, size
    return 3, size


@register_tool("read_only", untrusted=True)
def find_group_chats(query: str = "", limit: int = 10):
    """Find an EXISTING group text conversation in Messages — by its name ("family chat") or by
    who is in it ("Sam Alex"). Returns each match's chat ref, name and people; pass the ref as
    chat= to send_message or read_messages. Only for a GROUP the user means: to text or read
    ONE person, use search_contacts instead. Groups cannot be created here. If several match
    and none is exactly what the user named, ask the user which one."""
    applescript.mac_only()
    limit = max(1, min(int(limit or 10), _MAX))
    groups = _groups()
    terms = [t for t in _words(query) if t not in _STOPWORDS and len(t) > 1]
    if not str(query or "").strip() or not terms:
        out = [_found(g) for g in groups[:limit]]
        if len(groups) > limit or not out:
            out.append({"note": f"{len(groups)} group chats in all; pass a group name or the "
                                "people in it to narrow the list"})
        return out
    ranked = sorted(((r, i, g) for i, g in enumerate(groups) if (r := _rank(g, query, terms))),
                    key=lambda x: (x[0], x[1]))
    if not ranked:
        return (f"No group chat matches {query!r}. Groups are only found here, never created. "
                "Do not text the people one by one instead unless the user asks for that — "
                "tell the user there is no such group; they can start it in Messages.")
    top = ranked[0][0][0]
    if len(ranked) == 1:
        return [_found(ranked[0][2])]
    if top < 3 and sum(1 for r, _i, _g in ranked if r[0] == top) == 1:
        # Decisive: exactly the name, or exactly the people named. It alone is returned — with
        # the larger groups listed beside it a 9b asked "which one?" (live run, 2026-10-03).
        others = len(ranked) - 1
        what = ("its name is exactly what was asked for", "it is the only group whose name matches",
                "its people are exactly the people named")[top]
        return [_found(ranked[0][2]),
                {"note": f"this is the group: {what}. {others} other group"
                         f"{'s' if others > 1 else ''} matched less closely and "
                         f"{'were' if others > 1 else 'was'} left out — use this one unless the "
                         "user said otherwise"}]
    out = [_found(g) for _r, _i, g in ranked[:limit]]
    out.append({"note": f"{len(ranked)} groups match and none is exactly what was named — "
                        "ask the user which one (ask_user), naming the people in each"})
    return out


def _found(group: dict) -> dict:
    return {"chat": group["ref"], "name": group["name"] or None,
            "people": [p["name"] for p in group["people"]], "size": len(group["people"])}


@register_tool("destructive")
def send_message(text: str, to: str = "", chat: str = ""):
    """Send a text message (iMessage) through the Messages app, to ONE person or to ONE
    existing group chat — never both. One person: `to` is their phone number or iMessage email
    from search_contacts ("text Sam" means Sam alone, even if Sam is in groups). A group
    conversation the user names ("the family chat", "Sam and Alex's group"): `chat` is the ref
    from find_group_chats. Never guess either, never pass a name. `text` is the exact message.
    The user is always shown who receives it and the text, and must approve first."""
    args, problem = route_target("send_message", {"to": to, "chat": chat})
    if problem:
        raise ToolError(problem)
    text = str(text or "")
    if not text.strip():
        raise ToolError("send_message needs the text to send")
    if args.get("chat"):
        return _send_to_group(args["chat"], text)
    handle = _handle(args["to"])                      # route_target refused anything else
    applescript.mac_only()
    blocked = egress.check("message", handle, text)
    if blocked:
        raise ToolError(blocked)
    egress.record("message", handle, text, provider="imessage", n_bytes=len(text.encode("utf-8")))
    _send("set svc to first account whose service type is iMessage\n"
          f"  send {quote(text)} to participant {quote(handle)} of svc")
    return {"to": handle, "text": text,
            "note": "handed to Messages for delivery as an iMessage; delivery is not confirmed "
                    "— a recipient who is not on iMessage shows 'Not Delivered' in Messages"}


def _send_to_group(ref: str, text: str) -> dict:
    """The group path: resolve the ref to the chat Messages knows NOW, refuse before anything is
    sent when it is gone, then send to the chat itself with one ledger event per recipient —
    the ledger's host stays one person, as on the 1:1 path."""
    applescript.mac_only()
    group = _group(ref)
    if group is None:
        raise ToolError(f"there is no group chat {ref} in Messages — find the group with "
                        "find_group_chats and use the chat ref it returns")
    handles = [p["handle"] for p in group["people"]]
    blocked = egress.check("message", f"group {ref}", text)
    if blocked:
        raise ToolError(blocked)
    n_bytes = len(text.encode("utf-8"))
    for handle in handles:
        egress.record("message", handle, text, provider="imessage", n_bytes=n_bytes)
    _send(f"send {quote(text)} to chat id {quote(group['guid'])}")
    return {"chat": ref, "group": _label(group["name"], group["people"]), "to": handles,
            "people": [p["name"] for p in group["people"]], "text": text,
            "note": "handed to Messages for delivery to the group; delivery is not confirmed"}


def _send(line: str) -> None:
    """Run one `send` inside Messages. A timeout is NOT retried: the Apple event may still be
    executing, so a second send could deliver the text twice."""
    script = f"""
tell application "Messages"
  {line}
end tell
return "ok\""""
    try:
        applescript.run(script, timeout=_SEND_TIMEOUT, app="Messages")
    except AppleScriptError as exc:
        if "timed out" in str(exc):
            raise ToolError("Messages did not answer in time; the message may or may not have "
                            "been sent — check Messages, and do not send it again") from exc
        raise ToolError(str(exc)) from exc


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
SELECT m.text, m.attributedBody, m.date, m.is_from_me, h.id, c.chat_identifier, c.display_name,
       c.guid
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


def _history_refs(db) -> dict:
    """{chat ROWID: ref} for every group chat in the history, assigned the way `_groups`
    assigns them, so a row's label carries the ref find_group_chats hands out."""
    rows = [(rowid, guid) for rowid, guid in db.execute("SELECT ROWID, guid FROM chat")
            if guid and ";+;" in guid]
    return {rowid: ref for (rowid, _g), ref in zip(rows, _assign_refs([g for _r, g in rows]))}


def _chat_filter(db, ref: str, refs: dict) -> "tuple[str, list] | None":
    """The WHERE clause keeping one group's messages, or None when no group in the history is
    `ref`. An exact ref first; a prefix match covers a ref the app lengthened differently."""
    ids = [rowid for rowid, r in refs.items() if r == ref]
    if not ids:
        guids = dict(db.execute("SELECT ROWID, guid FROM chat"))
        ids = [rowid for rowid in refs if _matches_ref(guids.get(rowid) or "", ref)]
    if not ids:
        return None
    return f"WHERE j.chat_id IN ({', '.join('?' * len(ids))})", ids


def _history_members(db, rowids: set) -> dict:
    """{chat ROWID: [handle, …]} from chat.db — the fallback label for an unnamed group when the
    Messages app cannot name its people. An older schema without the join table gives {}."""
    if not rowids:
        return {}
    out: dict = {}
    try:
        for chat_id, hid in db.execute(
                "SELECT k.chat_id, h.id FROM chat_handle_join k JOIN handle h ON h.ROWID = k.handle_id"):
            if chat_id in rowids:
                out.setdefault(chat_id, []).append(hid)
    except sqlite3.Error:
        return {}
    return out


def _key(handle: str) -> str:
    h = str(handle or "").strip().lower()
    return h if "@" in h else re.sub(r"\D", "", h)[-10:]


def _app_names() -> "tuple[dict, dict]":
    """({ref: group}, {handle key: name}) from the Messages app — the names a group's rows are
    shown with. Soft: a Messages app that cannot be asked leaves the handles as they are."""
    try:
        groups = _groups()
    except ToolError:
        return {}, {}
    names = {_key(p["handle"]): p["name"] for g in groups for p in g["people"]}
    return {g["ref"]: g for g in groups}, names


@register_tool("read_only", untrusted=True)
def read_messages(contact: str = "", query: str = "", limit: int = 20, chat: str = ""):
    """Read the user's text-message history from the Messages app, newest first. `contact`
    narrows it to ONE person — their phone number or email from search_contacts (their messages
    in group chats are included). `chat` reads ONE whole group conversation — the ref from
    find_group_chats ("what's the family chat saying"). Never both. `query` keeps only messages
    containing that text. Returns when, from ('me' for the user), chat and text; a group row is
    labelled with its chat ref."""
    applescript.mac_only()
    args, problem = route_target("read_messages", {"contact": contact, "chat": chat})
    if problem:
        raise ToolError(problem)
    contact, ref = str(args.get("contact") or "").strip(), str(args.get("chat") or "").strip()
    needle = str(query or "").strip().lower()
    limit = max(1, min(int(limit or 20), _MAX))
    person = _handle(contact) if contact else None    # route_target refused a name
    scan = _SCAN if (contact or ref or needle) else limit * 4
    try:
        db = _connect(_db_path())
        try:
            refs = _history_refs(db)
            where, ids = "", []
            if person:
                found = _person_filter(db, person)
                if found is None:
                    return f"No messages with {person} in the history."
                where, ids = found
            elif ref:
                found = _chat_filter(db, ref, refs)
                if found is None:
                    return (f"No group chat {ref} in the Messages history — find the group with "
                            "find_group_chats and use the chat ref it returns.")
                where, ids = found
            rows = db.execute(_QUERY.format(where=where), (*ids, scan)).fetchall()
            guid_rowid = {guid: rowid for rowid, guid in db.execute("SELECT ROWID, guid FROM chat")}
            group_rows = {guid_rowid.get(r[7]) for r in rows if r[7] and ";+;" in r[7]} - {None}
            members = _history_members(db, group_rows)
        finally:
            db.close()
    except sqlite3.Error as exc:
        if "authorization denied" in str(exc) or "unable to open" in str(exc):
            app = _terminal_app()
            raise ToolError(
                f"macOS did not let {app} read the Messages history. Give {app} — not Messages "
                "— Full Disk Access under System Settings > Privacy & Security > Full Disk "
                f"Access, then restart {app} and ask again") from exc
        raise ToolError(f"the Messages history could not be read: {exc}") from exc
    by_ref, names = _app_names() if group_rows else ({}, {})
    out = []
    for text, blob, date, mine, handle, chat_id, chat_name, guid in rows:
        body = (text or "").strip() or _attributed_text(blob).strip()
        if not body:
            continue                      # an attachment or a reaction: nothing to read
        if needle and needle not in body.lower():
            continue
        rowid = guid_rowid.get(guid) if guid and ";+;" in guid else None
        if rowid is None:
            out.append({"when": _when(date), "from": "me" if mine else (handle or "unknown"),
                        "chat": chat_name or chat_id or handle or "", "text": body})
        else:
            gref = refs.get(rowid) or chat_ref(guid)
            app_group = by_ref.get(gref)
            people = (app_group["people"] if app_group
                      else [names.get(_key(h), h) for h in members.get(rowid, [])])
            sender = "me" if mine else names.get(_key(handle), handle or "unknown")
            out.append({"when": _when(date), "from": sender,
                        "chat": f"{gref} · {_label(chat_name or '', people)}", "group": True,
                        "text": body})
        if len(out) >= limit:
            break
    # Fewer than asked for AND the scan stopped at its cap: older messages were never read,
    # so "none" (or "these are all") would be a claim about history nobody looked at.
    if len(out) < limit and len(rows) >= scan and needle:
        searched = f"the newest {scan:,} messages" + (f" with {person}" if person else "")
        if not out:
            return (f"No messages match in {searched}; older ones were not searched — narrow "
                    "it with contact= or chat=, or try a more specific query.")
        out.append({"note": f"only {searched} were searched; older ones may match too"})
    return out or "No messages match."
