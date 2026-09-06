"""
Apple Notes tools — search_notes, read_note, create_note.

Three tools over the Notes app via AppleScript (`tools/applescript.py`). The two readers are
`read_only` AND `untrusted=True`: a note may be shared from someone else, or hold text pasted
from a web page or an email, so its content is treated exactly like a web fetch — scanned and
fenced by the quarantine before the model reads it. `create_note` changes user data, so it is
`side_effecting` and faces the approval gate with the exact title/body/folder shown.

Every failure — not macOS, Automation denied, no such note — comes back as an `Error: …`
string the model must relay, never invent around. Nothing here is egress.
"""

from __future__ import annotations

from tools import applescript
from tools.applescript import AS_RS, AS_US, ISO_HANDLERS, AppleScriptError, quote, records
from tools.toolspec import register_tool

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
        return f"Error: {exc}"
    notes = [{"id": r[0], "title": r[1], "folder": r[2], "modified": r[3]}
             for r in rows if len(r) == 4 and r[2] != _TRASH]
    notes.sort(key=lambda n: n["modified"], reverse=True)
    if not notes:
        return f"No notes match {q!r}." if q else "No notes found."
    return notes[:limit]


@register_tool("read_only", untrusted=True)
def read_note(note: str):
    """Read one Apple Note's full text. `note` is a note id from search_notes (preferred) or a
    title (exact match first, then the first title containing it)."""
    ref = str(note or "").strip()
    if not ref:
        return "Error: read_note needs a note id or title"
    if ref.startswith("x-coredata://"):
        locate = f"set n to note id {quote(ref)}"
    else:
        locate = f"""set found to (every note whose name is {quote(ref)})
  if (count of found) is 0 then set found to (every note whose name contains {quote(ref)})
  if (count of found) is 0 then return ""
  set n to item 1 of found"""
    script = f"""
tell application "Notes"
  {locate}
  set c to container of n
  return (id of n) & {AS_US} & (name of n) & {AS_US} & (name of c) ¬
    & {AS_US} & my iso(modification date of n) & {AS_US} & (plaintext of n)
end tell
{ISO_HANDLERS}"""
    try:
        out = applescript.run(script, app="Notes")
    except AppleScriptError as exc:
        return f"Error: {exc}"
    parts = out.split(applescript.US, 4)
    if len(parts) != 5:
        return f"Error: no note matches {ref!r}"
    return {"id": parts[0], "title": parts[1], "folder": parts[2], "modified": parts[3], "body": parts[4]}


@register_tool("side_effecting")
def create_note(title: str, body: str = "", folder: str = ""):
    """Create a new Apple Note with `title` and plain-text `body` (newlines kept), in `folder`
    (an existing Notes folder name) or the default folder when omitted. Returns the new note's id."""
    title = str(title or "").strip()
    if not title:
        return "Error: a note needs a non-empty title"
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
        return f"Error: {exc}"
    return {"id": note_id, "title": title, "folder": folder}


def _html(text: str) -> str:
    """Notes treats `body` as HTML: escape the three characters that would otherwise be parsed
    as markup, so the user's text lands verbatim, and turn newlines into <br> (a bare linefeed
    is collapsed to a space — verified live 2026-09-06)."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "<br>".join(esc.replace("\r\n", "\n").split("\n"))
