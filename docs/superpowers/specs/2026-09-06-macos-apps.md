# Native macOS app tools — what shipped, what was deferred, and why

Date: 2026-09-06. Status: Notes, Calendar and Mail (read + draft) shipped; sending and
Messages deferred by decision.

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
