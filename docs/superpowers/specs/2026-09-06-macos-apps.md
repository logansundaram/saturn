# Native macOS app tools — what shipped, what was deferred, and why

Date: 2026-09-06. Status: Notes, Calendar and Mail (read + draft) shipped 2026-09-06.
Extended 2026-10-01 (the section at the end): the write side of all three, Contacts,
Reminders, Shortcuts, the browser tab and Finder selection, `send_message` and the Messages
history reader. `send_mail` is still deferred.

## Shipped

| Module | Tools | Tier / trust |
|---|---|---|
| `tools/applescript.py` | the runner (`run(script, app=)`, `quote`, `records`, `ISO_HANDLERS`) | — |
| `tools/notes.py` | `search_notes`, `read_note` · `create_note` | read_only + untrusted · side_effecting |
| `tools/calendar.py` | `list_calendar_events` · `create_calendar_event` | read_only + untrusted · side_effecting |
| `tools/mail.py` | `list_mail`, `search_mail`, `read_mail` · `draft_mail` | read_only + untrusted · side_effecting, **not egress** |

All AppleScript via `osascript`; the target app is opened hidden first (`open -gja`) because
osascript alone answers -600 to a closed Calendar even when told to `launch`. Nothing here is
egress: `draft_mail` composes a visible, unsent draft and the human presses Send in Mail.

## Probe findings that fixed the design

- **EventKit is out.** A terminal-launched Python only gets Calendar access through EventKit
  if the *terminal app* carries `NSCalendarsUsageDescription` (iTerm does; Terminal.app and VS
  Code don't) — the request returns denied with no prompt. AppleScript automation prompts
  uniformly ("Terminal wants to control Calendar") from any terminal. The pyobjc EventKit
  package was installed for the probe and removed.
- **Timings (this Mac).** Notes search 0.3s warm. Calendar: 6.5s warm / 15s cold for a window
  across eight calendars, 0.7s narrowed to two by name. Mail (22k-message inbox): newest-N
  ~5s cold, subject/sender filter ~1.5s, one body ~3s, a `date received` filter 19s (never
  used).
- **Notes quirks.** `name of container of n` fails with -1700; `set c to container of n`
  then `name of c` works. A bare newline in a new note's body collapses to a space — `<br>`
  is required, and `<`, `>`, `&` must be escaped because `body` is HTML. Deleted notes linger
  in "Recently Deleted" and are filtered out of search.
- **A killed osascript does not cancel its Apple event.** Mail kept executing each timed-out
  `messages of inbox` query on its main thread (sampled: `MFMailbox messages` →
  `copyOfAllMessagesWithOptions`, a full-mailbox SQLite walk), and every later query — and
  every "is it alive yet" poll — queued behind it. Six timed-out probes left Mail unresponsive
  to scripting for minutes at 80% CPU. Hence: Mail tools use a 90s timeout, issue exactly one
  mailbox reference per call, look messages up by `message id N of <mailbox>` instead of a
  `whose id is` filter (1.2s — Mail evaluates it on the materialized list, unlike `read status`,
  which hung), and never retry after a timeout (the error says so to the model).
- **Mail costs ~1s + 0.25s per message listed, however you ask.** A per-message property loop,
  `subject of (messages 1 thru 50 of inbox)`, and a `{id, subject, …} of (…)` record fetch all
  landed at 12–20s for 50 messages; batching through a list *variable* is a -1728 error. So
  `unread_only` scans a 25-message window in Python rather than filtering in Mail, and the
  listing cap is 50. On this Gmail account `inbox` resolves to "[Gmail]/All Mail" (22k rows).
- **`delete` on an outgoing message or a draft does nothing visible;** `close window saving no`
  closes the compose window, `move … to trash mailbox` removes a persisted draft.
- **Separators.** Script output is RS/US-delimited. `str.strip()` treats those bytes as
  whitespace and ate an empty trailing field — the runner strips line endings only.

## Deferred — and what it would take

_2026-10-01: `send_message` and the history reader below shipped, wired as this section says;
`send_mail` remains deferred. The text is kept as the reasoning of record._

### `send_mail` / `send_message` (sending)

Sending IS egress in this project's definition (bytes leave the machine), but
`tests/test_no_new_egress.py` only catches network-client *imports*, and `osascript` is not
one — so a send tool would slip past the guard unless wired deliberately:

1. `egress.check(channel, host)` first (air-gap refuses), `egress.record(...)` on success with
   the recipient as the host label, so the ledger and status bar count it.
2. Tier `destructive`, so it faces the gate regardless of `runtime.auto_approve`.
3. Add the module to the chokepoint list in CLAUDE.md, `docs/ARCHITECTURE.md`, and the
   guard test's docstring.
4. Mail: `send m` on an outgoing message (verified the draft path; the verb compiles).
   Messages: `send "text" to participant "+1555…" of (first account whose service type is
   iMessage)` compiles (`osacompile`); not executed.

Decision 2026-09-06: not now. The draft covers the common case with the human as the send
button, and it keeps the egress surface unchanged.

### Messages history (reading texts)

Not possible through AppleScript: `chats` iteration fails with -10000 and there is no message
element. `find_message_contact` would work (participants list name + handle; verified) but is
pointless without the history reader. The only route is `~/Library/Messages/chat.db`:

- Requires **Full Disk Access** for the terminal app (authorization denied otherwise —
  verified with sqlite3 `-readonly`).
- On recent macOS `message.text` is often NULL; the text lives in `attributedBody` as an
  NSAttributedString typedstream blob that needs a custom parser.
- The schema is undocumented and shifts across releases.

Decision 2026-09-06: skip. Revisit only with an explicit FDA grant and a small typedstream
decoder, both behind a `read_messages` tool marked `untrusted=True`.

### Other platforms

Every tool registers everywhere and answers "only available on macOS (this is linux)" — the
notify precedent, so the planner catalog is stable across platforms.

## 2026-10-01 — the write side, five more apps, and sending

| Module | Tools added | Tier / trust |
|---|---|---|
| `tools/notes.py` | `append_note` | side_effecting |
| `tools/calendar.py` | `update_calendar_event` · `delete_calendar_event` | side_effecting · destructive |
| `tools/mail.py` | `reply_mail` · `update_mail` | side_effecting, **not egress** |
| `tools/contacts.py` | `search_contacts` | read_only + untrusted |
| `tools/reminders.py` | `list_reminders` · `create_reminder`, `complete_reminder` | read_only + untrusted · side_effecting |
| `tools/shortcuts.py` | `list_shortcuts` · `run_shortcut` | read_only · destructive + untrusted, `UNTRACKED` on the ledger, held by air-gap |
| `tools/desktop.py` | `read_browser_tab` · `finder_selection` | read_only + untrusted · read_only |
| `tools/messages.py` | `send_message` · `read_messages` | destructive + **egress** + always asks · read_only + untrusted |
| `tools/files.py` | `move_file`; Spotlight behind `search_files` | side_effecting |
| `core/mentions.py`, `/copy` | `@clipboard` | the user's own act; no tool |

### Probe findings (this Mac, 2026-10-01)

- **Every Apple event has a price, and it differs by app.** Reminders: about a second per
  event, so a per-reminder loop was 8.6s for six reminders; `properties of (reminders of
  <list> whose completed is false)` is one event per list (~2s) and the records format without
  further events. Contacts: a loop over `people whose name contains …` is ~0.6s a person and
  a bulk `<property> of (people whose …)` is ~5s per property on a broad match (119 of 171
  cards); `id of (people whose …)` once (0.7s) then `person id …` is ~0.2s a person.
- **Reminders has no default list here** (`default list` and `default account` are -1728; the
  only list is a shared one). `create_reminder` names the lists when it cannot pick one. The
  scripting dictionary has no recurrence and no location: "every Sunday" and "when I get
  home" cannot be set.
- **Calendar refuses a save where start is not before end**, so moving an event writes the two
  dates in an order that depends on the direction of the move. A uid lookup in one named
  calendar is 0.5s. A script reaches a recurring event only as the series, hence
  `whole_series`.
- **Notes append works by rewriting `body`** (`body & "<div>…</div>"`): title and existing
  lines survive. Not verified: what a rewrite does to a checklist or an attachment — a note
  with attachments is refused; a checklist cannot be detected from a script.
- **A Mail reply takes its text only if the script never reads the reply's content first.**
  `reply m opening window yes`, wait one second, `set content` — worked five times out of
  five; the same script with one read of `content of r` before the set left the reply empty
  every time. Setting the content replaces Mail's own quote, so the tool builds the quoted
  original itself (capped at 150 lines). `reply … opening window no` leaves an autosaved,
  empty draft behind; the tool always opens the window.
- **Mail has no archive verb**, and on this Gmail account `inbox` is "[Gmail]/All Mail" —
  `update_mail` offers `move` with a named mailbox instead of an `archive` action.
- **Safari gives the page text as a property** (`text of current tab`) with no setting to
  turn on; verified on a live page. Chrome returns URL and title; its text needs *Allow
  JavaScript from Apple Events* (off here — the tool degraded as designed).
- **Spotlight is a candidate source, not a judge.** For one phrase in this repo it returned
  28 files where a line-wise grep finds 17: it matches across line breaks, and it missed two
  files grep finds. So every candidate is re-matched by the regex, and a folder small enough
  to walk in two seconds is still walked whole. From `~`: a phrase with no match answers in
  2.0s (the old walk: 10s cap, 80s uncapped), "tax return" in ~5s with hits inside PDFs.
  Opening fifteen PDFs took 11s, hence the three-second document budget.
- **`pbpaste` reads the clipboard with no prompt** on this macOS. The clipboard is still not
  a tool: it is attached only when the user types `@clipboard`.
- **`chat.db` is `authorization denied` without Full Disk Access** (still true). The reader
  and the typedstream decoder are tested against a synthetic database only.

### Not executed

- `send_message` was compiled (`osacompile`) and unit-tested, never run: sending a real
  message needs the user. First real use should be to yourself.
- `run_shortcut` was never run against a real shortcut (each does something real). Whether
  `shortcuts run` prints a shortcut's output on a pipe, as the tool assumes, is unverified.
- `read_messages` against the real history (no Full Disk Access).
- `update_mail` `move` / `trash` (flag and unflag were run and reverted on one message).
- `update_calendar_event` / `delete_calendar_event` on a recurring event or one with
  attendees; `append_note` on a shared note, a checklist or a note with attachments.
  A checklist is the likely casualty: Notes hands a script a checklist as a plain list, so the
  body rewrite may drop the checkboxes and their ticks. A script cannot create a checklist to
  try it on — check by hand before relying on `append_note` for a checklist note.
- `update_calendar_event` with a bare clock time (`atclock`, 2026-10-01 review): the scripts
  compile and the handler was run on its own; no real event was moved with it.
- `append_note`'s exact-title lookup: the no-such-title path was run read-only against Notes
  (returns `none`); the two-notes-one-title path was not.

