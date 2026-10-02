"""
Apple Mail tools — list_mail, search_mail, read_mail, draft_mail, reply_mail, update_mail.

Six tools over the Mail app via AppleScript (`tools/applescript.py`). The three readers are
`read_only` AND `untrusted=True`: email is the canonical prompt-injection vector, so every
subject line and body is scanned and fenced by the quarantine exactly like a web page before
the model reads it. `draft_mail` composes a VISIBLE, UNSENT draft in Mail for the human to
review and send themselves — it is `side_effecting` (it creates a draft) but it is NOT egress:
nothing leaves the machine until the human presses Send in Mail. `reply_mail` is the same
thing in a thread: Mail's own `reply` (so the recipient and the In-Reply-To headers are right),
with the text set once the window exists and the original quoted underneath. `update_mail` is
triage — read / unread / flag / unflag / move / trash — over a LIST of ids, so one cleanup is
one gate prompt and one mailbox reference; a trashed message is in the Trash, not gone.

Deliberately not here (decided 2026-09-06; see docs/superpowers/specs/2026-09-06-macos-apps.md):
`send_mail` — sending IS egress and would be a new chokepoint (egress.check/record wiring, a
`destructive` tier, tests/test_no_new_egress.py's docstring). The draft covers the common case
with the human as the send button.

Measured 2026-09-06 (22k-message inbox): newest-N listing ~5s cold, a subject/sender filter
~1.5s, one body ~3s; a `date received` filter took 19s and is never used. Every `messages of
<mailbox>` reference makes Mail materialize the whole mailbox, so the tools issue exactly one
such reference per call. Message ids are Mail's per-account integer `id`, stable within a
mailbox, so `read_mail` takes id + mailbox. (`message id N of inbox` is a syntax error — `message
id` is the RFC header property — so the lookup is a `whose id is N` filter, which Mail evaluates
in memory on the already-materialized list, unlike `read status`, whose filter hung it.)
"""

from __future__ import annotations

from tools import applescript
from tools.applescript import AS_RS, AS_US, ISO_HANDLERS, AppleScriptError, quote, records
from tools.toolspec import ToolError, register_tool

# A killed osascript does NOT cancel its Apple event: Mail keeps grinding through it on its main
# thread, serially, and every retry queues behind it (observed 2026-09-06 — six timed-out queries
# left Mail unresponsive for minutes). So the timeout is generous, and the tools never retry.
_QUERY_TIMEOUT = 90.0

_STANDARD = {"inbox": "inbox", "sent": "sent mailbox", "drafts": "drafts mailbox",
             "junk": "junk mailbox", "trash": "trash mailbox", "outbox": "outbox"}
_MAX = 50


def _mailbox_expr(name: str) -> str:
    """The AppleScript expression for a mailbox: the five standard ones by keyword, anything
    else by name across every account (Gmail labels show up as mailboxes)."""
    key = (name or "inbox").strip().lower()
    if key in _STANDARD:
        return _STANDARD[key]
    return f"first item of (mailboxes whose name is {quote(name.strip())})"


_ROW = (f"(id of m as string) & {AS_US} & (sender of m) & {AS_US} & (subject of m) & {AS_US} "
        f"& my iso(date received of m) & {AS_US} & (read status of m as string) & {AS_US} "
        f"& (was replied to of m as string) & {AS_RS}")


def _error(exc: AppleScriptError, mailbox: str) -> ToolError:
    """Mail's own errors as the ToolError to raise, with the one everyday case made plain."""
    if "Invalid index" in str(exc) and "every mailbox whose name" in str(exc):
        return ToolError(f"no mailbox named {mailbox!r} (use inbox, sent, drafts, junk, trash, "
                         f"or a folder/label name)")
    return ToolError(str(exc))


def _rows(out: str, mailbox: str) -> list[dict]:
    return [
        {"id": r[0], "mailbox": mailbox, "from": r[1], "subject": r[2], "date": r[3],
         "unread": r[4] == "false", "replied": r[5] == "true"}
        for r in records(out) if len(r) == 6
    ]


# Newest messages scanned for `unread_only`. A `whose read status is false` filter hung Mail for
# 15 minutes; a bounded window costs ~1s + 0.25s per message however the properties are fetched
# (per-message loop, `subject of (messages 1 thru N …)`, or a `{id, subject, …} of` record —
# all measured 2026-09-06), so 25 is 7–12s.
_WINDOW = 25


@register_tool("read_only", untrusted=True)
def list_mail(mailbox: str = "inbox", limit: int = 10, unread_only: bool = False):
    """List the newest messages in an Apple Mail mailbox ('inbox' by default; also 'sent',
    'drafts', 'junk', 'trash', or a folder/label name). Returns id, from, subject, date and the
    unread / replied flags, newest first. `unread_only` keeps only unread ones among the newest 25.
    Follow up with read_mail for a message's body."""
    mailbox = str(mailbox or "inbox").strip() or "inbox"
    limit = max(1, min(int(limit or 10), _MAX))
    box = _mailbox_expr(mailbox)
    take = _WINDOW if unread_only else limit
    script = f"""
set out to ""
tell application "Mail"
  set ms to messages of {box}
  if (count of ms) > {take} then set ms to messages 1 thru {take} of {box}
  repeat with r in ms
    set m to contents of r
    set out to out & {_ROW}
  end repeat
end tell
return out
{ISO_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Mail")
    except AppleScriptError as exc:
        raise _error(exc, mailbox) from exc
    rows = _rows(out, mailbox)
    if unread_only:
        rows = [m for m in rows if m["unread"]]
        if not rows:
            return f"No unread messages among the newest {_WINDOW} in {mailbox}."
    return rows[:limit] if rows else f"No messages in {mailbox}."


@register_tool("read_only", untrusted=True)
def search_mail(query: str, mailbox: str = "inbox", limit: int = 10):
    """Search an Apple Mail mailbox for messages whose subject or sender contains `query`.
    Returns id, from, subject, date and unread flag, newest first."""
    q = str(query or "").strip()
    if not q:
        raise ToolError("search_mail needs a non-empty query")
    mailbox = str(mailbox or "inbox").strip() or "inbox"
    limit = max(1, min(int(limit or 10), _MAX))
    script = f"""
set out to ""
set k to 0
tell application "Mail"
  set ms to (messages of {_mailbox_expr(mailbox)} whose subject contains {quote(q)} or sender contains {quote(q)})
  repeat with r in ms
    set m to contents of r
    set out to out & {_ROW}
    set k to k + 1
    if k ≥ {limit} then exit repeat
  end repeat
end tell
return out
{ISO_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Mail")
    except AppleScriptError as exc:
        raise _error(exc, mailbox) from exc
    rows = _rows(out, mailbox)
    return rows if rows else f"No messages match {q!r}."


@register_tool("read_only", untrusted=True)
def read_mail(id: int | str, mailbox: str = "inbox"):
    """Read one Apple Mail message: from, to, subject, date and the plain-text body. `id` is
    the id from list_mail / search_mail, `mailbox` the mailbox it was listed from."""
    try:
        mid = int(id)
    except (TypeError, ValueError):
        raise ToolError(f"read_mail needs a numeric message id from list_mail, not {id!r}")
    mailbox = str(mailbox or "inbox").strip() or "inbox"
    script = f"""
tell application "Mail"
  set ms to (messages of {_mailbox_expr(mailbox)} whose id is {mid})
  if (count of ms) is 0 then return ""
  set m to item 1 of ms
  set tos to ""
  repeat with t in to recipients of m
    if tos is not "" then set tos to tos & ", "
    set tos to tos & (address of t)
  end repeat
  return (id of m as string) & {AS_US} & (sender of m) & {AS_US} & tos & {AS_US} & (subject of m) ¬
    & {AS_US} & my iso(date received of m) & {AS_US} & (content of m)
end tell
{ISO_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Mail")
    except AppleScriptError as exc:
        raise _error(exc, mailbox) from exc
    parts = out.split(applescript.US, 5)
    if len(parts) != 6:
        raise ToolError(f"no message with id {mid} in {mailbox}")
    return {"id": parts[0], "mailbox": mailbox, "from": parts[1], "to": parts[2],
            "subject": parts[3], "date": parts[4], "body": parts[5]}


def _addresses(text: str) -> list[str]:
    return [a.strip() for a in str(text or "").replace(";", ",").split(",") if a.strip()]


@register_tool("side_effecting")
def draft_mail(to: str, subject: str, body: str, cc: str = ""):
    """Compose an email in Apple Mail as an UNSENT draft, opened on screen for the user to
    review and send themselves. `to` / `cc` are comma-separated addresses. Nothing is sent by
    this tool. Use it whenever the user asks to write, draft, or prepare an email."""
    tos, ccs = _addresses(to), _addresses(cc)
    subject = str(subject or "").strip()
    if not tos:
        raise ToolError("draft_mail needs at least one recipient address")
    if not subject:
        raise ToolError("an email draft needs a non-empty subject")
    recips = "\n".join(
        [f"    make new to recipient with properties {{address:{quote(a)}}}" for a in tos]
        + [f"    make new cc recipient with properties {{address:{quote(a)}}}" for a in ccs]
    )
    script = f"""
tell application "Mail"
  set m to make new outgoing message with properties {{subject:{quote(subject)}, content:{quote(str(body or ""))}, visible:true}}
  tell m
{recips}
  end tell
  activate
  return "ok"
end tell"""
    try:
        applescript.run(script, app="Mail")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    return {"to": tos, "cc": ccs, "subject": subject,
            "note": "opened in Mail as an unsent draft for you to review and send; nothing was sent"}


# The "> "-quoted original under a reply. Capped: an AppleScript string built line by line is
# quadratic, and a reply does not need a newsletter quoted in full.
_QUOTE_MAX_LINES = 150

_QUOTED = f"""
on quoted(t)
  set out to ""
  set k to 0
  repeat with p in paragraphs of t
    set out to out & "> " & p & linefeed
    set k to k + 1
    if k ≥ {_QUOTE_MAX_LINES} then
      set out to out & "> […]" & linefeed
      exit repeat
    end if
  end repeat
  return out
end quoted
"""


@register_tool("side_effecting")
def reply_mail(id: int | str, body: str, mailbox: str = "inbox", reply_all: bool = False):
    """Reply to an email: opens an UNSENT reply in Apple Mail — addressed to the sender, in the
    same thread, the original quoted under your text — for the user to review and send
    themselves. `id` and `mailbox` come from list_mail / search_mail; `body` is the reply text
    only (no greeting to the quoted message, no quote). reply_all=true also addresses everyone
    on the original. Nothing is sent by this tool. Use it for "reply to …"; draft_mail starts a
    new, unthreaded message."""
    try:
        mid = int(id)
    except (TypeError, ValueError):
        raise ToolError(f"reply_mail needs a numeric message id from list_mail, not {id!r}")
    body = str(body or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        raise ToolError("reply_mail needs the reply text")
    mailbox = str(mailbox or "inbox").strip() or "inbox"
    probe = body.split("\n", 1)[0][:60].rstrip()
    # Mail takes the text only once the reply window exists, and a script that READS the
    # reply's content before setting it leaves the reply empty (both probed live 2026-10-01):
    # a fixed one-second wait, one set, then the read that confirms it. No retry.
    script = f"""
tell application "Mail"
  set ms to (messages of {_mailbox_expr(mailbox)} whose id is {mid})
  if (count of ms) is 0 then return ""
  set m to item 1 of ms
  set q to "On " & my iso(date received of m) & ", " & (sender of m) & " wrote:" & linefeed ¬
    & my quoted(content of m as string)
  set r to reply m opening window yes reply to all {"yes" if reply_all else "no"}
  delay 1
  set content of r to {quote(body)} & linefeed & linefeed & q
  set landed to ((content of r as string) contains {quote(probe)})
  set tos to ""
  repeat with t in to recipients of r
    if tos is not "" then set tos to tos & ", "
    set tos to tos & (address of t)
  end repeat
  activate
  return (landed as string) & {AS_US} & (subject of r) & {AS_US} & tos
end tell
{ISO_HANDLERS}{_QUOTED}"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Mail")
    except AppleScriptError as exc:
        raise _error(exc, mailbox) from exc
    parts = out.split(applescript.US)
    if len(parts) != 3:
        raise ToolError(f"no message with id {mid} in {mailbox}")
    if parts[0] != "true":
        raise ToolError("a reply window is open in Mail but it did not take the text; nothing was "
                        f"sent — the user can paste it in: {body}")
    return {"to": _addresses(parts[2]), "subject": parts[1],
            "note": "opened in Mail as an unsent reply for you to review and send; nothing was sent"}


_ACTIONS = {
    "read": "set read status of m to true",
    "unread": "set read status of m to false",
    "flag": "set flagged status of m to true",
    "unflag": "set flagged status of m to false",
    "trash": "move m to trash mailbox",
    "move": None,   # needs `destination`
}
_MAX_IDS = 50


def _ids(ids) -> list[int]:
    raw = [p for p in str(ids).replace(";", ",").split(",")] if isinstance(ids, (str, int)) else list(ids or [])
    out = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError:
            raise ToolError(f"update_mail needs numeric message ids from list_mail, not {item!r}")
    return list(dict.fromkeys(out))


@register_tool("side_effecting")
def update_mail(ids: list[int | str] | str, action: str, mailbox: str = "inbox", destination: str = ""):
    """Triage Apple Mail messages, several at once: `action` is 'read', 'unread', 'flag',
    'unflag', 'trash' (moves them to the Trash), or 'move' (to the mailbox named by
    `destination`). `ids` is a list of message ids from list_mail / search_mail, all in
    `mailbox`. Pass every message of one cleanup in ONE call. Returns which messages changed."""
    action = str(action or "").strip().lower()
    if action not in _ACTIONS:
        raise ToolError(f"unknown action {action!r}; use one of: " + ", ".join(_ACTIONS))
    wanted = _ids(ids)
    if not wanted:
        raise ToolError("update_mail needs at least one message id")
    if len(wanted) > _MAX_IDS:
        raise ToolError(f"update_mail takes at most {_MAX_IDS} messages per call, not {len(wanted)}")
    mailbox = str(mailbox or "inbox").strip() or "inbox"
    destination = str(destination or "").strip()
    if action == "move":
        if not destination:
            raise ToolError("action 'move' needs `destination`, the mailbox to move the messages to")
        act = f"move m to ({_mailbox_expr(destination)})"
    else:
        act = _ACTIONS[action]
    match = " or ".join(f"id is {n}" for n in wanted)
    # One mailbox reference (Mail materializes the mailbox on each), what was found recorded
    # BEFORE anything moves, then the action from the last message backwards: a move changes
    # what the later positions refer to.
    script = f"""
set out to ""
tell application "Mail"
  set ms to (messages of {_mailbox_expr(mailbox)} whose {match})
  repeat with r in ms
    set m to contents of r
    set out to out & (id of m as string) & {AS_US} & (subject of m) & {AS_RS}
  end repeat
  repeat with i from (count of ms) to 1 by -1
    set m to item i of ms
    {act}
  end repeat
end tell
return out"""
    try:
        out = applescript.run(script, timeout=_QUERY_TIMEOUT, app="Mail")
    except AppleScriptError as exc:
        raise _error(exc, destination if action == "move" and "whose name" in str(exc) else mailbox) from exc
    changed = [{"id": r[0], "subject": r[1]} for r in records(out) if len(r) == 2]
    if not changed:
        raise ToolError(f"none of the messages {wanted} are in {mailbox}; nothing was changed")
    found = {c["id"] for c in changed}
    result = {"action": action, "mailbox": mailbox, "changed": changed,
              "missing": [str(n) for n in wanted if str(n) not in found]}
    if action == "move":
        result["destination"] = destination
    return result
