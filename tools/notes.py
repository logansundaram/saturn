"""
Apple Notes tools — search_notes, read_note, create_note, append_note.

Four tools over the Notes app via AppleScript (`tools/applescript.py`). The two readers are
`read_only` AND `untrusted=True`: a note may be shared from someone else, or hold text pasted
from a web page or an email, so its content is treated exactly like a web fetch — scanned and
fenced by the quarantine before the model reads it. `create_note` and `append_note` change user
data, so they are `side_effecting` and face the approval gate with the exact text shown.

`append_note` rewrites the note's HTML `body` with the new lines added (the only way a script
can append; verified live 2026-10-01: the title and the existing lines survive). A locked note,
or one holding attachments, is refused before anything is written — a body rewrite is not known
to keep attachments. It writes to a note id or to the one note with exactly that title, never
to "the first title containing it" (`_locate_exactly`).

Every failure — not macOS, Automation denied, no such note — raises ToolError: an `Error: …`
observation the model must relay, never invent around, stamped error. Nothing here is egress.
"""

from __future__ import annotations

from tools import applescript
from tools.applescript import (AS_GS, AS_RS, AS_US, GS, ISO_HANDLERS, AppleScriptError, quote,
                               records)
from tools.toolspec import ToolError, register_tool

_MAX_MATCHES = 200   # hard cap on records fetched before Python sorts and applies `limit`
_TRASH = "Recently Deleted"   # deleted notes linger here for 30 days; search never lists them

# `name of container of n` fails with -1700 inside Notes; the two-step form resolves.


@register_tool("read_only", untrusted=True)
def search_notes(query: str = "", limit: int = 10):
    """Search Apple Notes on this Mac by title or content. Returns the most recently modified
    matching notes (id, title, folder, modified); an empty `query` lists the most recent notes.
    Follow up with read_note to get a note's full text."""
    q = str(query or "").strip()
    limit = max(1, min(int(limit or 10), _MAX_MATCHES))
    script = f"""
set q to {quote(q)}
set out to ""
set k to 0
tell application "Notes"
  if q is "" then
    set found to notes
  else
    set found to (every note whose name contains q or plaintext contains q)
  end if
  repeat with r in found
    set n to contents of r
    set c to container of n
    set out to out & (id of n) & {AS_US} & (name of n) & {AS_US} & (name of c) ¬
      & {AS_US} & my iso(modification date of n) & {AS_RS}
    set k to k + 1
    if k ≥ {_MAX_MATCHES} then exit repeat
  end repeat
end tell
return out
{ISO_HANDLERS}"""
    try:
        rows = records(applescript.run(script, app="Notes"))
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    notes = [{"id": r[0], "title": r[1], "folder": r[2], "modified": r[3]}
             for r in rows if len(r) == 4 and r[2] != _TRASH]
    notes.sort(key=lambda n: n["modified"], reverse=True)
    if not notes:
        return f"No notes match {q!r}." if q else "No notes found."
    return notes[:limit]


def _locate(ref: str) -> str:
    """The script lines that bind `n` to the note `ref` names — an id, else an exact title, else
    the first title containing it — or return "" from the script when nothing matches."""
    if ref.startswith("x-coredata://"):
        return f"set n to note id {quote(ref)}"
    return f"""set found to (every note whose name is {quote(ref)})
  if (count of found) is 0 then set found to (every note whose name contains {quote(ref)})
  if (count of found) is 0 then return ""
  set n to item 1 of found"""


_MAX_NEAR = 5   # titles offered when a write names no note exactly


def _locate_exactly(ref: str) -> str:
    """`_locate` for a WRITE: an id, or the ONE note with exactly that title. A write has no
    undo and the gate showed only what the model typed, so it never picks among titles that
    merely contain the text: the script returns "none" (with those titles, to choose from) or
    "many" instead of writing."""
    if ref.startswith("x-coredata://"):
        return f"set n to note id {quote(ref)}"
    return f"""set found to (every note whose name is {quote(ref)})
  if (count of found) is 0 then
    set near to ""
    repeat with t in (name of every note whose name contains {quote(ref)})
      set near to near & t & {AS_GS}
    end repeat
    return "none" & {AS_US} & near
  end if
  if (count of found) > 1 then return "many" & {AS_US} & ((count of found) as string)
  set n to item 1 of found"""


@register_tool("read_only", untrusted=True)
def read_note(note: str):
    """Read one Apple Note's full text. `note` is a note id from search_notes (preferred) or a
    title (exact match first, then the first title containing it)."""
    ref = str(note or "").strip()
    if not ref:
        raise ToolError("read_note needs a note id or title")
    script = f"""
tell application "Notes"
  {_locate(ref)}
  set c to container of n
  return (id of n) & {AS_US} & (name of n) & {AS_US} & (name of c) ¬
    & {AS_US} & my iso(modification date of n) & {AS_US} & (plaintext of n)
end tell
{ISO_HANDLERS}"""
    try:
        out = applescript.run(script, app="Notes")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    parts = out.split(applescript.US, 4)
    if len(parts) != 5:
        raise ToolError(f"no note matches {ref!r}")
    return {"id": parts[0], "title": parts[1], "folder": parts[2], "modified": parts[3], "body": parts[4]}


@register_tool("side_effecting")
def create_note(title: str, body: str = "", folder: str = ""):
    """Create a new Apple Note with `title` and plain-text `body` (newlines kept), in `folder`
    (an existing Notes folder name) or the default folder when omitted. Returns the new note's id."""
    title = str(title or "").strip()
    if not title:
        raise ToolError("a note needs a non-empty title")
    folder = str(folder or "").strip()
    text = _html(str(body or ""))
    target = f" at folder {quote(folder)}" if folder else ""
    script = f"""
tell application "Notes"
  set n to make new note{target} with properties {{name:{quote(title)}, body:{quote(text)}}}
  return id of n
end tell"""
    try:
        note_id = applescript.run(script, app="Notes")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    return {"id": note_id, "title": title, "folder": folder}


@register_tool("side_effecting")
def append_note(note: str, text: str):
    """Add `text` to the END of an existing Apple Note (each line becomes its own line in the
    note); the rest of the note is kept. `note` is a note id from search_notes (preferred) or
    the note's exact title. Use this for "add X to my … note" — create_note would start a
    second note."""
    ref = str(note or "").strip()
    text = str(text or "").replace("\r\n", "\n").strip("\n")
    if not ref:
        raise ToolError("append_note needs a note id or title")
    if not text.strip():
        raise ToolError("append_note needs the text to add")
    script = f"""
tell application "Notes"
  {_locate_exactly(ref)}
  if password protected of n then return "locked"
  if (count of attachments of n) > 0 then return "attachments"
  set body of n to (body of n) & {quote(_divs(text))}
  return "ok" & {AS_US} & (id of n) & {AS_US} & (name of n)
end tell"""
    try:
        out = applescript.run(script, app="Notes")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    if out == "locked":
        raise ToolError(f"the note {ref!r} is locked; unlock it in Notes and add the text there")
    if out == "attachments":
        raise ToolError(f"the note {ref!r} holds attachments, which a scripted edit could drop; "
                        "nothing was changed — add the text in Notes")
    parts = out.split(applescript.US)
    if parts[0] == "none":
        near = [t for t in (parts[1] if len(parts) > 1 else "").split(GS) if t][:_MAX_NEAR]
        hint = ("; notes with that in the title: " + ", ".join(repr(t) for t in near)
                + " — pass the exact title of the one meant") if near else ""
        raise ToolError(f"no note is titled {ref!r}{hint} (search_notes finds a note and its id); "
                        "nothing was changed")
    if parts[0] == "many":
        raise ToolError(f"{parts[1] if len(parts) > 1 else 'several'} notes are titled {ref!r}; "
                        "pass the id of the one meant (search_notes gives the ids); nothing "
                        "was changed")
    if len(parts) != 3 or parts[0] != "ok":
        raise ToolError(f"no note matches {ref!r}")
    return {"id": parts[1], "title": parts[2], "appended": text}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _divs(text: str) -> str:
    """`text` as the HTML Notes stores for typed lines: one <div> per line, an empty line as
    <div><br></div> (what Notes itself writes for a blank line)."""
    return "".join(f"<div>{_escape(line) or '<br>'}</div>" for line in text.split("\n"))


def _html(text: str) -> str:
    """Notes treats `body` as HTML: escape the three characters that would otherwise be parsed
    as markup, so the user's text lands verbatim, and turn newlines into <br> (a bare linefeed
    is collapsed to a space — verified live 2026-09-06)."""
    return "<br>".join(_escape(text).replace("\r\n", "\n").split("\n"))
