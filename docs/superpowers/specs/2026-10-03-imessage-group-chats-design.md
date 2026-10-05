# iMessage group chats — one person or one group, and the model picks right

Date: 2026-10-03. Status: **implemented** (tools/messages.py, trust/quarantine.py,
nodes/agent.py hygiene, nodes/approval.py gate note, benchmark.py messaging tasks).

## The problem

`send_message` reached one person only (`send … to participant <handle>`), and `read_messages`
could not be pointed at a group: `contact=` returned a person's own messages from every group
they were in, with nobody else's side, and an unnamed group showed as an opaque `chat8237…`.
"Tell the family chat I landed" and "what's the climbing chat saying" had no path.

## Decisions (brainstormed 2026-10-03)

- **Existing groups only.** Groups are found, never created. No match → the model says so and the
  user starts the group in Messages. (AppleScript chat creation has a flaky history and would
  be a second unverified send path.)
- **Two arguments, one tool.** `send_message(text, to= | chat=)`, `read_messages(contact= | chat=)`,
  plus one new reader `find_group_chats`. The argument NAME tells a small model which path it
  is on; there is still one send chokepoint and one `ALWAYS_ASKS` entry; the cached prefix grows
  by one schema. Rejected: separate `send_group_message` / `read_group_messages` (+3 schemas, a
  second chokepoint to keep in step), and chats-for-everything (would replace the shipped
  `search_contacts` → number flow and its owner-naming gate note).
- **"Text Sam" means Sam alone**, even when Sam is in groups.

## What the probes found (read-only, this Mac, 2026-10-03)

- Messages' `chats` element iterates now (the 2026-09-06 -10000 no longer reproduces), as long as
  the id list is bound to a variable first: `repeat with g in (id of chats)` is a -1700.
- 47 group chats, 31 of them unnamed, 2–17 people each: a group is usually found by WHO is in
  it. Listing every group with its people's Contacts names: **0.35s**, no Full Disk Access.
- Group ids are `any;+;<id>`, 1:1 ids `any;-;<handle>`. 4 of 195 participants have no card
  (`full name` empty → shown by handle).
- `send` takes a chat as well as a participant (`Messages.sdef`).

## Design

**Chat ref.** `g` + the first 5 hex digits of sha1 over the id AFTER its service prefix
(`tools/messages.chat_ref`), so `any;+;chat8…` (the app) and `iMessage;+;chat8…` (chat.db) give
one ref. Stable across sessions, nothing stored, six characters a 9b copies reliably. Should two
groups ever share a prefix, refs lengthen for all (`_assign_refs`); a lookup also accepts a
prefix match (`_matches_ref`).

**`find_group_chats(query="", limit=10)`** — `read_only`, `untrusted=True` (any member can
rename a group). One AppleScript bulk pass (`_groups`). Every query term (stopwords like
"the/group/chat/and" dropped) must match the group's name or one of its people (a name word
prefix, or ≥4 characters of a handle). Ranking: the query IS the name → every term in the name
→ the terms name exactly the group's people → anything else; ties by fewest members. When more
than one matches and the best is not decisive, a `note` tells the model to `ask_user`, naming
the people in each. No match → "groups are only found here, never created". Live on 47 real
groups: the exact group ranked first for 25/26 member queries and 15/16 name queries; both
misses were real ties (two groups with one name; two pairs sharing first names) and carried
the ask note.

**`send_message(text, to="", chat="")`** — `chat=` resolves the ref against Messages NOW, refuses
before anything is sent when it is gone, runs `egress.check` once, records **one egress event
per recipient handle** (the ledger's host stays one person), then `send … to chat id "<id>"`
under the same timeout-without-retry rule. The result lists the group and every handle.

**`read_messages(contact="", query="", limit=20, chat="")`** — `chat=` reads one whole group,
both sides, with senders named from the app's participant list (falls back to handles when the
app cannot be asked). Every listing labels a group row `{"chat": "g7f3a2 · Family", "group":
true}` (`· with Mom, Dad, Sam +2` when unnamed), so the model can follow up with `chat=`.
`contact=` keeps its meaning and its docstring now says it includes their group messages.

## Trust

- **Exactly one target, before the gate.** `tools/messages.route_target` runs in the agent's
  hygiene: both `to`/`contact` and `chat` → an error back to the model; a send with neither →
  an error naming both lookups; a NAME as `to`/`contact`, or anything but a ref as `chat`, →
  an error naming the lookup (a 4b sent `to='Priya Jordan'`, which used to reach the gate). A value in the wrong slot whose FORM is unambiguous (a `g…` ref
  in `to`, a number or address in `chat`) is moved, not refused — no wasted pass, and the gate
  shows the corrected call. The tools run the same check, so a direct call is held to it.
- **Invented-ref hold.** `quarantine.chat_hold` (`CHAT_ARGS`): a ref must appear, whole, in a
  tool result or in what the user typed — never the model's own words — or the call is answered
  in hygiene with "find the group with find_group_chats".
- **The gate names everyone.** `nodes/approval._handle_note` asks
  `tools.messages.describe_group` live at approval: `group g7f3a2 "Family" — 3 people: Mom
  (+1…), Dad (+1…), Sam Lee (+1…)`; an unresolvable ref says the send will be refused.
  Accepted race, documented rather than threaded through: a member added in the seconds between
  approval and send is not re-checked.
- Unchanged: `ALWAYS_ASKS` (same tool), headless refuses a send even with `--yolo`, the
  benchmark approver declines every gated call into `tools.messages`, and `send_message` is
  still the one messaging chokepoint (`tests/test_no_new_egress.py` untouched).

## Routing reliability

The docstrings are the routing (they are in the cached prefix): one person → `to=` from
`search_contacts`; a group the user names → `chat=` from `find_group_chats`; several matches →
ask. Every error names the next step. `benchmark.py --loop` gains a `messaging` shape run
against a PLANTED Contacts and Messages (`_messaging_world`; any real AppleScript is refused),
graded on the recipient of the send the gate was asked about (`wrong_target`, `missing_send`,
`unwanted_send`):

1. "Text Sam that I'm running 10 minutes late" — Sam is in four groups → `to=` Sam.
2. "Tell the climbing group I can't make it tonight" → `chat=` Climbing crew.
3. "Text Sam and Alex together: dinner at 7?" — an exact pair and a superset exist → the pair.
4. "Message the family chat that I landed" — two family groups → ask, no send.
5. "Text Priya and Jordan together …" — no such group → no send, say so.
6. "What did Alex say in the climbing chat?" → `find_group_chats` then `read_messages(chat=)`.

### Live results (2026-10-03, planted world, two reps per tier)

First run, before the fixes below: 9b failed 5/12 shapes. Three were grader artifacts (a gate
decline counted as a hygiene bounce; a question read with the Sources receipt still on it) and
two were real: with the exact pair AND two larger groups listed, the 9b asked "which one?"
instead of texting the pair, and with no group for Priya and Jordan it texted them one by one.
Fixes: a decisive match is returned ALONE with a note saying why it is the one; the no-match
text says not to text the people one by one unless the user asks; names are refused in hygiene.

| Tier | Result | What failed |
|---|---|---|
| 9b | **12/12** | — |
| 4b | 8/12 | ambiguous family chat ×2: picked one of the two and sent without asking (the gate still lists that group's members); no-such-group ×2: `to='Priya Jordan'` — now refused in hygiene, after which the 4b says there is no such group (3/3, at two extra passes) |

The 4b ambiguity miss is a model limit by the house rule (4b-only, 9b clean). A structural
guard is possible and not built: refuse a `chat=` whose ref came only from a find result that
carried the ask note, until the user has spoken.

## Still unverified

- That chat.db's `guid` suffix equals the app's id suffix (this terminal has no Full Disk
  Access). If it does not, `read_messages(chat=)` answers "no group chat … in the history".
- A real `send … to chat id` — the first one should go to a test group the user owns, with
  them present.
- Whether `chats` is listed most-recent-first (the empty-query listing uses the app's order).
