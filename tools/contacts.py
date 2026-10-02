"""
Apple Contacts tool — search_contacts.

One reader over the Contacts app via AppleScript (`tools/applescript.py`): a name becomes the
email addresses, phone numbers and birthday on the card, so "reply to Petra" and "text Sam"
resolve to a real address instead of a guessed one. `read_only` AND `untrusted=True`: a card
can arrive shared or synced from elsewhere, so its text is scanned and fenced like a web page.

Measured 2026-10-01 (171 cards): `id of (people whose name contains …)` 0.7s, then ~0.2s per
person fetched by id. A loop over the `whose` result itself is ~0.6s per person, and bulk
`<property> of (people whose …)` forms are ~5s each on a broad match — so the script resolves
ids once and fetches at most `limit` people by id. Nothing here is egress.

The ids are ranked before the cut: a card whose name IS the query (whole, first, last or
nickname), then one whose first or last name starts with it, then any name containing it.
Contacts' own order put "Ian" fifth behind Brian, Brian Ling, Christiana and "Roommate Asian"
(2026-10-02) — one more substring match and `limit=5` would have cut the person asked for.
Measured 2026-10-02 (172 cards): each narrow `whose` ~0.4s, so ranking costs ~1s; a broad
query ("a", 120 matches) is ~6s in any form.
"""

from __future__ import annotations

from tools import applescript
from tools.applescript import (AS_FS, AS_GS, AS_RS, AS_US, FS, GS, ISO_HANDLERS, RS,
                               AppleScriptError, quote, records)
from tools.toolspec import ToolError, register_tool

_MAX = 25

_HANDLERS = ISO_HANDLERS + f"""
on pairs(labels, vals)
  set s to ""
  repeat with j from 1 to count of vals
    set l to item j of labels
    if l is missing value then set l to ""
    set s to s & l & {AS_FS} & (item j of vals) & {AS_GS}
  end repeat
  return s
end pairs
"""


def _label(raw: str) -> str:
    """Contacts' built-in labels arrive wrapped (`_$!<Work>!$_`); a custom one arrives bare."""
    raw = raw.strip()
    if raw.startswith("_$!<") and raw.endswith(">!$_"):
        raw = raw[4:-4]
    return raw.lower()


def _pairs(field: str) -> list[dict]:
    out = []
    for item in field.split(GS):
        if FS not in item:
            continue
        label, value = item.split(FS, 1)
        if value.strip():
            out.append({"label": _label(label), "value": value.strip()})
    return out


def _birthday(iso: str) -> str:
    """The card's birthday as a date; a card saved without a year stores 1604, so only the
    month and day are reported for it."""
    day = iso[:10]
    return day[5:] if day.startswith("1604-") else day


@register_tool("read_only", untrusted=True)
def search_contacts(query: str, limit: int = 10):
    """Look up people in Apple Contacts by name (or part of one). Returns each match's name,
    organization, email addresses, phone numbers and birthday. Use it to get the address or
    number for a person the user names before drafting mail or sending a message — never guess
    an address. Several matches mean the user must say which one."""
    q = str(query or "").strip()
    if not q:
        raise ToolError("search_contacts needs a name to look for")
    limit = max(1, min(int(limit or 10), _MAX))
    script = f"""
set out to ""
tell application "Contacts"
  set exact to id of every person whose name is {quote(q)} or first name is {quote(q)} ¬
    or last name is {quote(q)} or nickname is {quote(q)}
  set lead to id of every person whose first name starts with {quote(q)} ¬
    or last name starts with {quote(q)}
  set loose to id of every person whose name contains {quote(q)}
  set ids to {{}}
  repeat with i in (exact & lead & loose)
    set i to contents of i
    if ids does not contain i then set end of ids to i
  end repeat
  set out to ((count of ids) as string) & {AS_RS}
  set k to 0
  repeat with i in ids
    set p to person id i
    set org to organization of p
    if org is missing value then set org to ""
    set bd to birth date of p
    if bd is missing value then
      set bds to ""
    else
      set bds to my iso(bd)
    end if
    set out to out & (name of p) & {AS_US} & org ¬
      & {AS_US} & my pairs(label of emails of p, value of emails of p) ¬
      & {AS_US} & my pairs(label of phones of p, value of phones of p) ¬
      & {AS_US} & bds & {AS_RS}
    set k to k + 1
    if k ≥ {limit} then exit repeat
  end repeat
end tell
return out
{_HANDLERS}"""
    try:
        out = applescript.run(script, timeout=60.0, app="Contacts")
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    head, _, rest = out.partition(RS)
    total = int(head) if head.strip().isdigit() else 0
    people = []
    for r in records(rest):
        if len(r) != 5:
            continue
        person = {"name": r[0]}
        if r[1]:
            person["organization"] = r[1]
        if emails := _pairs(r[2]):
            person["emails"] = emails
        if phones := _pairs(r[3]):
            person["phones"] = phones
        if r[4]:
            person["birthday"] = _birthday(r[4])
        people.append(person)
    if not people:
        return f"No contacts match {q!r}."
    if total > len(people):
        return {"contacts": people,
                "note": f"showing {len(people)} of {total} matches — use a fuller name to narrow it"}
    return people
