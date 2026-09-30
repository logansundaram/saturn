"""
Apple Mail tools — list_mail, search_mail, read_mail, draft_mail.

Four tools over the Mail app via AppleScript (`tools/applescript.py`). The three readers are
`read_only` AND `untrusted=True`: email is the canonical prompt-injection vector, so every
subject line and body is scanned and fenced by the quarantine exactly like a web page before
the model reads it. `draft_mail` composes a VISIBLE, UNSENT draft in Mail for the human to
review and send themselves — it is `side_effecting` (it creates a draft) but it is NOT egress:
nothing leaves the machine until the human presses Send in Mail.

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
        f"& my iso(date received of m) & {AS_US} & (read status of m as string) & {AS_RS}")


def _error(exc: AppleScriptError, mailbox: str) -> ToolError:
    """Mail's own errors as the ToolError to raise, with the one everyday case made plain."""
    if "Invalid index" in str(exc) and "every mailbox whose name" in str(exc):
        return ToolError(f"no mailbox named {mailbox!r} (use inbox, sent, drafts, junk, trash, "
                         f"or a folder/label name)")
    return ToolError(str(exc))


def _rows(out: str, mailbox: str) -> list[dict]:
    return [
        {"id": r[0], "mailbox": mailbox, "from": r[1], "subject": r[2], "date": r[3],
         "unread": r[4] == "false"}
        for r in records(out) if len(r) == 5
    ]


# Newest messages scanned for `unread_only`. A `whose read status is false` filter hung Mail for
# 15 minutes; a bounded window costs ~1s + 0.25s per message however the properties are fetched
# (per-message loop, `subject of (messages 1 thru N …)`, or a `{id, subject, …} of` record —
# all measured 2026-09-06), so 25 is 7–12s.
_WINDOW = 25


@register_tool("read_only", untrusted=True)
def list_mail(mailbox: str = "inbox", limit: int = 10, unread_only: bool = False):
    """List the newest messages in an Apple Mail mailbox ('inbox' by default; also 'sent',
    'drafts', 'junk', 'trash', or a folder/label name). Returns id, from, subject, date and
    unread flag, newest first. `unread_only` keeps only unread ones among the newest 25.
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
