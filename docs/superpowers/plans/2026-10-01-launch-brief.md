# Launch Brief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before the first prompt, show one dim, capped block of what needs the user today: overdue reminders and commitments, what is left of today's calendar, unread mail they have not answered, and tomorrow morning. Gathered without a model and never added to the conversation. `/brief` shows it on demand.

**Architecture:** `core/brief.py` calls the existing reader tools' functions directly (`list_calendar_events`, `list_reminders`, `list_mail`) plus the memory registry, one daemon thread per source. It turns each result into `Item(bucket, key, text)` lines, making every line inert (control and bidi characters neutralised), and `compose()` sorts them into urgency buckets under a line cap. `commands/brief.py` owns the one renderer, the `/brief` command and the launch view. `app/repl.py` starts the gather after the splash, waits up to 2 s before the first prompt, and shows any source that lands later once, between prompts.

**Tech Stack:** Python 3.11+, the existing AppleScript readers (`tools/applescript.py`), `threading`, rich via `tui.ui`, pytest (offline: conftest's `mac` controller captures every `osascript`).

**Spec:** `docs/pivot.md` item 6 ("A launch brief"), `docs/advantages.md` §3 ("a launch brief turns 'open the terminal' into 'here is your day'") and §5.2 (overnight precompute — a follow-up here), `docs/research.md` Part 2 ("The brief (plan 3)"), and the Design section below.

---

## Design

### The problem

Opening Saturn starts from nothing: a person who wants to know what is on today has to ask,
wait for a model pass and a tool round per app, and ask again tomorrow. The readers that would
answer it already exist: Calendar, Reminders and Mail through AppleScript, and the memory file's
`commitments` (with `due=`) and `memo` layers. Pivot #6 asks for one dim block at session start, built from read-only calls and
cached for the session, with `/brief` to re-run it and a config switch to turn it off.

### What the products teach (background)

- **Finite and bucketed, not a feed.** Microsoft's Copilot "Today" sorts what it finds into
  missed / needs attention now / can wait
  (blogs.microsoft.com/blog/2026/09/25/introducing-the-new-copilot-with-home-code-and-autopilot/);
  ChatGPT Pulse shows roughly five to ten cards that expire after a day (reported through
  secondary write-ups; the OpenAI page returned 403). → a hard cap per bucket and overall, with
  `+N more — /brief --all`, ordered by urgency: overdue → today → waiting → coming up → notes.
- **Silence is a feature.** OpenClaw's heartbeat suppresses its output when there is nothing to
  say (github.com/openclaw/openclaw/blob/main/docs/gateway/heartbeat.md). → an empty launch
  brief prints nothing at all; an explicit `/brief` says "nothing on today" in one line.
- **Dates are where proactive assistants lose trust.** Reviews of Poke (saner.ai/blogs/poke-reviews)
  single out wrong dates and time zones and noisy nudges. → day buckets compare local
  wall-clock DATES, never `now ± 24 h`, and the tests pin all-day events, an event across
  midnight, a reminder due at 23:59, a date-only reminder on its own day, and a DST change day.
- **"Waiting on you" is the mail signal people value** in Superhuman and Fyxer. Mail cannot say
  which messages need an answer. Its listing does carry `unread` and `replied` flags, so the
  deterministic half is: unread and not replied, among the newest 25 in the inbox.

### Approaches considered

1. **A deterministic brief (chosen).** Call the reader functions directly, render the result.
   No model call, so it adds nothing to any turn. It cannot hallucinate a meeting, and it cannot
   be prompt-injected into acting, because no model reads it. Subjects and senders are still
   other people's text, so they are made inert and clipped before rendering. The brief is never
   put in the prompt or in state.
2. **The brief is the first turn.** Run a real agent turn with the readers auto-approved. That
   costs at least two model passes at every launch (a cold first pass on a 9b is several
   seconds), and it puts untrusted mail text in front of the model before the user has typed a
   word. A crafted subject line would get a model pass with tools bound and no human request to
   anchor it. Rejected.
3. **An overnight precompute** (advantages §5.2): a launchd job builds the brief at 6 am and the
   launch only prints it. The launch is instant, but it adds a background process, a cache file
   of private data on disk, and staleness (a 6 am brief at 3 pm). It is a natural follow-up once
   routines exist (`docs/research.md` P1). Listed under "Not in this plan".

### The chosen design

- **Sources** (each on its own daemon thread, each failing on its own):
  - *calendar*: `list_calendar_events(start=<today>, end=<today+2>)` — one bulk AppleScript
    query across every calendar (≈0.7–6.5 s warm; the cost is the calendars' size, not the
    window). Today's events that have not ended (all-day ones first, "since yesterday" for one
    that crossed midnight) go to *today*; tomorrow's events before noon, and tomorrow's all-day
    events, go to *coming up*.
  - *reminders*: `list_reminders()` — one `properties of (reminders of l whose completed is
    false)` event per list (≈2 s). The reader's own `overdue` flag decides lateness, so there
    is one rule. A reminder due today goes to *today* and one due tomorrow to *coming up*;
    undated reminders are a list, not a day, and stay out.
  - *mail*: `list_mail(mailbox="inbox", limit=25)` — one mailbox reference, as CLAUDE.md
    requires (≈7 s on a large inbox). Unread and not replied → *waiting*, newest first.
  - *memory*: `entries("commitments")` and `entries("memo")`. A commitment whose `due=` starts
    with an ISO date goes to *overdue* / *today* / *coming up* (within 7 days). Undated ones
    become one count line. The last 3 memo entries go to *notes*.
- **Caps:** 5 lines per bucket, 15 overall, in bucket order; `/brief --all` lifts both.
- **Rendering point.** `tui/ui/prompt.py` reads the line with prompt_toolkit without
  `patch_stdout`, so anything printed while the prompt is open would tear the input line. The
  gather starts right after the splash. Starting it earlier would import the reader modules
  from a thread before the registry, which would reorder the registered tools, and the bound
  tool catalog is part of the cached prefix. The rest of startup runs while the gather does.
  Just before the first prompt, `show_at_launch` waits up to 2 s and shows what has landed. A
  source still gathering gets one dim line, and what it brings is shown once at the next prompt
  boundary under `── brief, continued` (`show_late`). Nothing ever prints while a prompt is
  open.
- **First launch and permissions — off by default.** The first Apple event to each app raises
  a macOS "Terminal wants to control Calendar" dialog. Three of those firing from background
  threads on a first launch, behind the `/models` page, is a permission-prompt storm the user
  did not ask for. So `runtime.brief` ships `false`. The first launch prints one hint line, and
  `/brief on` turns it on and gathers once right away, so the dialogs appear while the user is
  watching the result they asked for. After that, a denied app costs nothing: macOS answers
  `-1743` instantly, `tools/applescript.run` turns that into a sentence, and the launch view
  names the source on one dim line (`/brief` gives the reason). This departs from pivot #6's
  "`runtime.brief: false` turns it off" (default on); the dialogs are the reason.
- **Session cache.** `brief.start()` returns the gather in flight if there is one. Apple events
  to one app run one after another, so a second gather would only queue behind the first. A
  finished gather under 10 minutes old is reused; `--refresh` replaces it. The header says when
  it was gathered.
- **Not egress, not a run.** Apple events and the memory file stay on the machine
  (`tests/test_no_new_egress.py` needs no change). No model is called and the agent takes no
  action, so nothing goes in the trace DB; each source's time goes to `diag.log`.

### Assumptions made without asking

1. Default **off** (above), with a first-launch hint and `/brief on|off` persisting `runtime.brief` through `config.persist`. A `config.yaml` seeded before the key existed has no line to edit (`config._set_yaml_scalar` raises `KeyError`), so the command then says, word for word, what line to add.
2. "Mail waiting on a reply" means **unread and not replied, among the newest 25** in the inbox. The other "awaiting reply" signal, mail the user *sent* that got no answer, needs the Sent mailbox and a match against the inbox: two mailbox references, so two scripts. That is under "Not in this plan".
3. The **memo digest** is in, as the lowest bucket (*notes*, 3 lines). The pivot names it; it is a cheap local read.
4. **Ended events are left out** of *today* (a 3 pm launch does not list the 9 am stand-up); all-day events stay.
5. `@brief` (attaching the brief to a turn) is left out: it is the user's explicit act, but it would put other people's subjects in the prompt and needs the attachment quarantine path. It is under "Not in this plan".
6. Wait budgets: **2.0 s** before the first prompt (`_LAUNCH_WAIT_S`) and **30 s** for `/brief` (`_COMMAND_WAIT_S`, Ctrl-C stops waiting). The reader tools keep their own 90 s osascript timeout, and the brief only stops *waiting*. That matters because killing osascript does not cancel the Apple event (`tools/applescript.py`), so a late result still fills the cache.
7. **Inert text:** the sibling plan `2026-10-01-terminal-escape-sanitising.md` adds the shared neutraliser `textutil.visible_controls(text)`. It turns every C0/C1 control except TAB and LF into a visible picture (ESC → `␛`) and removes SGR colour codes; it deliberately leaves bidi and zero-width marks alone outside the gate. `research.md` orders that plan first. If `visible_controls` exists when you reach Task 1, write `_inert` in the shared form given in Task 1 Step 3; otherwise use the local form there and switch when it lands. Both forms pass the same tests (checked against a dry run of each), and `test_inert_text_carries_no_control_or_bidi_characters` is the regression pin. The brief strips bidi and zero-width marks itself because each line is one line of someone else's words.
8. A **suite-wide guard** against real Apple events is added to `tests/conftest.py`: `applescript._run` raises unless a test uses the `mac` fixture. It was found while dry-running this plan: an early version of the `--refresh` test gathered from the real Calendar, Reminders and Mail on the development Mac (read-only queries, but still).

### Not in this plan

- **Sent mail with no reply** (Superhuman's "awaiting reply"): a second reader over the Sent mailbox, matched by subject/recipient against the inbox — its own small plan; it would also feed the `commitments` layer.
- **A brief feedback file** (`~/.saturn/brief.md`, "always include…/never show…" in plain words, the Pulse "curate" idea).
- **Overnight precompute** through the `notify/` launchd seam, once routines exist (`docs/research.md` P1).
- **`@brief`** as an attachment, fenced as untrusted like other reader output.
- **Printing late sections above an open prompt** (prompt_toolkit `patch_stdout`): it touches every prompt in the session; the next-boundary print is enough until dogfooding says otherwise.

---

## Global Constraints

- Tests are fully offline and never touch the real apps: every osascript is captured by conftest's `mac` fixture (`applescript._run`). From Task 3 on, an uncaptured call fails the test (`_no_real_apple_events`).
- `isolated_paths` for anything that reads or writes the memory file or other configured paths.
- No model call anywhere in this feature; no LLM seam is involved.
- `diag.log()`, never `print()`, for diagnostics; user-facing lines go through `tui.ui` (`note`, `section`, `table`) or `commands._framework._print`.
- Every command accepts `--help` (the dispatcher handles a first/last `--help` token).
- AppleScript rules (CLAUDE.md): fetch in bulk, never per item; never two `messages of <mailbox>` references in one Mail script; scripts go through `applescript.run(script, app=)`. This plan writes no new script: it calls the existing readers.
- The brief adds no model call to any turn and does not delay the first prompt by more than `_LAUNCH_WAIT_S` (2.0 s), and only when `runtime.brief` is on.
- The brief never enters `state` or the prompt (`ctx.state` is untouched; `nodes/ground.py` is unchanged).
- Not egress: `tests/test_no_new_egress.py` stays unchanged and passing.
- User-visible changes go under `## [Unreleased]` in `CHANGELOG.md`.
- Commit messages: `area: what changed`, lowercase; end with the attribution lines the session's system reminder gives.

## Review Focus

1. **Ctrl-C while `/brief` waits.** `commands._framework.dispatch` catches `Exception`, not `KeyboardInterrupt`, so an unhandled Ctrl-C inside a handler would end the REPL. Expected: "stopped waiting", back at the prompt. Pinned by `test_ctrl_c_while_waiting_does_not_end_the_session` (Task 4).
2. **Yesterday's all-day event and the event across midnight.** The calendar query's `end date ≥ start` returns yesterday's all-day event (it ends at today 00:00). Expected: not in *today*; a night event that started yesterday shows as "until 09:00 … (since yesterday)"; a DST change day buckets by date. Pinned by `test_calendar_today_left_tomorrow_morning_and_nothing_else` and `test_calendar_buckets_by_wall_clock_date_on_a_dst_change_day` (Task 1).
3. **An escape sequence or a bidi override in a subject, title or reminder.** Expected: shown as visible inert text (`␛]52;…`), never sent to the terminal. Pinned by `test_inert_text_carries_no_control_or_bidi_characters` (Task 1); every item builder routes text through `_inert`.
4. **A source that is denied, closed or not on macOS.** Expected: that source becomes one quiet line at launch (`brief: mail not available — /brief says why`) and a full reason under `/brief`; the other sources still show; three identical "only available on macOS" reasons collapse to one line. Pinned by `test_a_failing_source_becomes_unavailable_and_the_rest_still_lands` (Task 3), `test_compose_groups_unavailable_sources_by_reason` (Task 2), `test_launch_view_names_unavailable_sources_quietly` (Task 5).
5. **A source that lands after the prompt opened, or a `/brief --refresh` in between.** Expected: the late lines print once at the next prompt boundary, never twice, and never for a gather that `--refresh` replaced. Pinned by `test_late_sources_show_once_at_the_next_prompt` and `test_a_refreshed_brief_drops_the_launch_gathers_late_lines` (Task 5).

Known and accepted, not tested: a mail question typed in the first seconds after launch waits behind the brief's in-flight Mail query (Mail runs Apple events one at a time; bounded by that one listing, ≈7 s on a large inbox). The manual check in Task 7 measures it.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `core/brief.py` | Create | Items from each reader (`calendar_items`, `reminder_items`, `mail_items`, `memory_items`), `_inert`, `compose()` into buckets under caps, `Gather` (a thread per source), `start()` / `current()` / `reset()` / `enabled()`. No UI imports. |
| `commands/brief.py` | Create | The one renderer (`render`), `/brief [--all \| --refresh \| on \| off]`, the launch view (`show_at_launch`, `show_late`, `first_run_hint`). |
| `commands/__init__.py` | Modify | Add `"brief"` to `_COMMAND_MODULES`. |
| `commands/system.py` | Modify | `/brief` in the `knowledge & workspace` group and in `_DAILY`. |
| `app/repl.py` | Modify | Start the gather after the splash; the first-run hint; `show_at_launch` before the first prompt; `show_late` at each prompt boundary. |
| `config.default.yaml` | Modify | `runtime.brief: false` with its comment. |
| `tests/conftest.py` | Modify | `_no_real_apple_events` autouse guard. |
| `tests/test_brief.py` | Create | Every test below. |
| `CHANGELOG.md`, `CLAUDE.md`, `docs/pivot.md`, `docs/ARCHITECTURE.md` | Modify | Task 6. |

## Interfaces at a glance

- `core.brief.Item(bucket: str, key: str, text: str)` (frozen), `core.brief.Unavailable(reason: str)` (frozen).
- `core.brief.BUCKETS = ("overdue", "today", "waiting", "coming up", "notes")`, `SOURCES = ("calendar", "reminders", "mail", "memory")`.
- `calendar_items(events, now) -> list[Item]`, `reminder_items(rows, now)`, `mail_items(rows, now)`, `memory_items(commitments, memos, now)`. Each accepts the reader's "No …" string and returns `[]` for it.
- `core.brief.Brief(gathered, buckets: dict[str, list[str]], hidden: dict[str, int], unavailable: dict[str, list[str]], pending: list[str])`, with `.empty`.
- `core.brief.compose(results: dict[str, list[Item] | Unavailable], gathered: datetime, *, pending=(), capped=True) -> Brief`.
- `core.brief.Gather`: `.begin()`, `.wait(timeout) -> bool`, `.done()`, `.pending()`, `.results()`, `.age()`, `.compose(*, capped=True, only=None)`, `.shown: set`, `.now`.
- `core.brief.start(*, sources=None, refresh=False) -> Gather`, `current() -> Gather | None`, `reset()`, `enabled() -> bool`.
- `commands.brief.render(b, *, title="brief", footer="full"|"quiet")`, `show_at_launch(g, wait=2.0)`, `show_late(g) -> bool`, `first_run_hint()`.
- **Dependency on a sibling plan:** `core.brief._inert(text, n=72) -> str` uses `textutil.visible_controls(text) -> str` from `2026-10-01-terminal-escape-sanitising.md` once that exists (shared form, Task 1 Step 3). Until then it uses a local regex. Its signature and its test are the same either way.

---

### Task 1: The brief's items — inert text, dates, and the four sources as lines

**Files:**
- Create: `core/brief.py`
- Create: `tests/test_brief.py`

**Interfaces:**
- Consumes: `textutil.clip(s, n)` (collapses whitespace, truncates with `…`); the readers' output shapes — `list_calendar_events` → `[{"calendar","uid","title","start","end","all_day","location","recurring"}]` or `"No events between …"`; `list_reminders` → `[{"id","list","title","due"?,"overdue"?,"notes"?,"flagged"?}]` or `"No open reminders."`; `list_mail` → `[{"id","mailbox","from","subject","date","unread","replied"}]` or `"No messages in …"`; memory `entries(layer)` dicts with `id`, `text`, `due`, `date`.
- Produces: `Item`, `Unavailable`, `BUCKETS`, `SOURCES`, `PER_BUCKET`, `TOTAL`, `MAIL_WINDOW`, `MEMO_LINES`, `COMMITMENT_DAYS`, `FRESH_S`, `_inert`, `_day_label`, `calendar_items`, `reminder_items`, `mail_items`, `memory_items`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brief.py` with the shared helpers and the Task 1 tests (later tasks append their sections to this file):

```python
"""
The launch brief (core/brief.py, commands/brief.py): today at a glance, gathered without a
model. Offline: osascript never runs (conftest's `mac` controller), the memory file is a tmp
file (`isolated_paths`), and the gathering threads run fake sources gated by Events — no sleeps.
"""

import threading
from datetime import datetime

import pytest

from core import brief
from core.brief import Item, Unavailable
from tools.applescript import RS, US

NOW = datetime(2026, 10, 2, 8, 0).astimezone()      # a Friday morning


def _ev(title, start, end, all_day=False, location=""):
    return {"calendar": "Home", "uid": "u", "title": title, "start": start, "end": end,
            "all_day": all_day, "location": location, "recurring": False}


def _rows(*rows):
    return RS.join(US.join(r) for r in rows) + RS


def _gated(result):
    """A fake source that returns `result` once its Event is set."""
    ev = threading.Event()

    def fn(now):
        assert ev.wait(5), "test never released this source"
        if isinstance(result, Exception):
            raise result
        return result

    return fn, ev


# ── Task 1: inert text and dates ──────────────────────────────────────────────────────────────

def test_inert_text_carries_no_control_or_bidi_characters():
    evil = "Invoice \x1b]52;c;ZXZpbA==\x07 due\x1b[2K\x1b[1A ‮exe.pdf​\nnext\x9b31m"
    out = brief._inert(evil, 200)
    assert "\x1b" not in out and "\x07" not in out and "\x9b" not in out
    assert "‮" not in out and "​" not in out and "\n" not in out
    assert "␛]52;c;ZXZpbA==" in out          # the planted sequence is visible, not live
    assert brief._inert(out, 200) == out          # idempotent
    assert brief._inert("Café ☕ — Zoë's 1:1", 200) == "Café ☕ — Zoë's 1:1"   # ordinary text untouched
    assert len(brief._inert("x" * 500)) == 72


def test_day_labels_are_dates_not_offsets():
    today = NOW.date()
    assert brief._day_label("2026-10-02T23:59", today) == "today"
    assert brief._day_label("2026-10-01", today) == "yesterday"
    assert brief._day_label("2026-10-03T00:00", today) == "tomorrow"
    assert brief._day_label("2026-09-30", today) == "Wed 30 Sep"
    assert brief._day_label("someday", today) == "someday"


# ── Task 1: calendar items ────────────────────────────────────────────────────────────────────

def test_calendar_today_left_tomorrow_morning_and_nothing_else():
    events = [
        _ev("Stand-up", "2026-10-02T07:00", "2026-10-02T07:30"),               # over already
        _ev("1:1 with Petra", "2026-10-02T09:30", "2026-10-02T10:00", location="Room 4"),
        _ev("Sam's birthday", "2026-10-02T00:00", "2026-10-03T00:00", all_day=True),
        _ev("Yesterday's holiday", "2026-10-01T00:00", "2026-10-02T00:00", all_day=True),
        _ev("Night shift", "2026-10-01T22:00", "2026-10-02T09:00"),             # crosses midnight
        _ev("Dentist", "2026-10-03T08:30", "2026-10-03T09:00"),
        _ev("Lunch", "2026-10-03T12:30", "2026-10-03T13:30"),                   # tomorrow afternoon
        _ev("Lisbon trip", "2026-10-03T00:00", "2026-10-06T00:00", all_day=True),
    ]
    items = brief.calendar_items(events, NOW)
    assert [(i.bucket, i.text) for i in items] == [
        ("today", "09:30–10:00  1:1 with Petra · Room 4"),
        ("today", "all day  Sam's birthday"),
        ("today", "until 09:00  Night shift (since yesterday)"),
        ("coming up", "tomorrow 08:30  Dentist"),
        ("coming up", "tomorrow all day  Lisbon trip"),
    ]


def test_calendar_no_events_string_is_no_items():
    assert brief.calendar_items("No events between 2026-10-02T00:00 and 2026-10-04T00:00.", NOW) == []


def test_calendar_buckets_by_wall_clock_date_on_a_dst_change_day():
    # US clocks fall back on 2026-11-01: that day has 25 hours. Late on the 1st, "tomorrow" is
    # still the 2nd — the bucket is a date, never now + 24 h (which would land on the 2nd 22:30).
    late = datetime(2026, 11, 1, 23, 30).astimezone()
    events = [_ev("Early train", "2026-11-02T06:00", "2026-11-02T07:00"),
              _ev("Late film", "2026-11-01T23:00", "2026-11-02T01:00")]
    items = brief.calendar_items(events, late)
    assert [(i.bucket, i.text) for i in items] == [
        ("coming up", "tomorrow 06:00  Early train"),
        ("today", "23:00–01:00  Late film"),
    ]


# ── Task 1: reminders, mail, memory ───────────────────────────────────────────────────────────

def test_reminders_overdue_today_tomorrow_and_undated_skipped():
    rows = [
        {"id": "a", "list": "Home", "title": "Call the dentist", "due": "2026-09-30T09:00", "overdue": True},
        {"id": "b", "list": "Home", "title": "Pay rent", "due": "2026-10-02T00:00"},
        {"id": "c", "list": "Home", "title": "Ring Mom", "due": "2026-10-02T23:59"},
        {"id": "d", "list": "Work", "title": "Send invoice", "due": "2026-10-03T17:00"},
        {"id": "e", "list": "Home", "title": "Buy honey"},
        {"id": "f", "list": "Home", "title": "Renew passport", "due": "2026-10-20T00:00"},
    ]
    assert [(i.bucket, i.text) for i in brief.reminder_items(rows, NOW)] == [
        ("overdue", "reminder · Call the dentist (due Wed 30 Sep 09:00)"),
        ("today", "reminder · Pay rent"),
        ("today", "reminder · Ring Mom (23:59)"),
        ("coming up", "tomorrow 17:00  reminder · Send invoice"),
    ]
    assert brief.reminder_items("No open reminders.", NOW) == []


def test_mail_waiting_is_unread_and_unanswered_newest_first():
    rows = [
        {"id": "3", "from": "Petra Novak <petra@example.com>", "subject": "Thursday?",
         "date": "2026-10-02T07:40", "unread": True, "replied": False},
        {"id": "2", "from": "news@shop.example", "subject": "Sale", "date": "2026-10-01T12:00",
         "unread": False, "replied": False},
        {"id": "1", "from": '"The Landlord" <ll@example.com>', "subject": "Lease renewal",
         "date": "2026-09-29T09:00", "unread": True, "replied": False},
        {"id": "0", "from": "sam@example.com", "subject": "dinner", "date": "2026-09-28T09:00",
         "unread": True, "replied": True},
    ]
    assert [i.text for i in brief.mail_items(rows, NOW)] == [
        "Petra Novak · Thursday? (today)",
        "The Landlord · Lease renewal (Tue 29 Sep)",
    ]


def test_memory_commitments_by_due_date_and_the_memo_digest():
    commitments = [
        {"id": 4, "text": "Send Jonah the photos", "due": "2026-09-30"},
        {"id": 5, "text": "Book the vet", "due": "2026-10-02"},
        {"id": 6, "text": "Lease notice", "due": "2026-10-07"},
        {"id": 7, "text": "Tax return", "due": "2027-04-15"},          # beyond the week
        {"id": 8, "text": "Call Jonah", "due": None},
        {"id": 9, "text": "Fix bike", "due": "after-the-trip"},       # a hand edit
    ]
    memos = [{"date": f"2026-09-{d}", "text": f"note {d}"} for d in (26, 27, 28, 29)]
    assert [(i.bucket, i.text) for i in brief.memory_items(commitments, memos, NOW)] == [
        ("overdue", "commitment #4 · Send Jonah the photos (due Wed 30 Sep)"),
        ("today", "commitment #5 · Book the vet (due today)"),
        ("coming up", "commitment #6 · Lease notice (due Wed 7 Oct)"),
        ("coming up", "2 open commitment(s) without a date — /memory"),
        ("notes", "Sun 27 Sep · note 27"),
        ("notes", "Mon 28 Sep · note 28"),
        ("notes", "Tue 29 Sep · note 29"),
    ]
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'core.brief'`.

- [ ] **Step 3: Write the module with the item builders**

Create `core/brief.py`. This is the module's head and its first two sections; Tasks 2 and 3 append the rest. The import block already lists what those tasks use.

```python
"""
The launch brief — today at a glance, gathered without a model.

`start()` reads four sources on one background thread each — Calendar (today, and tomorrow
until noon), Reminders (overdue, due today or tomorrow), Mail (unread and unanswered among the
newest `MAIL_WINDOW` in the inbox) and memory (commitments due within `COMMITMENT_DAYS`, the
last `MEMO_LINES` memo entries) — and `compose()` sorts what came back into urgency buckets
(`BUCKETS`: overdue → today → waiting → coming up → notes) under a line cap. No model call:
nothing for a chat turn to wait on, nothing to hallucinate, nothing a mail subject can steer.
The brief is printed for the user (commands/brief.py) and never enters the prompt or state.

Every source degrades on its own: a closed app, a denied Automation permission or a non-macOS
platform becomes an `Unavailable(reason)` for that source, never an exception. Event titles,
reminder text, senders and subjects are written by other people, so every line is made inert
(`_inert`) before it is rendered — an escape sequence in a subject could otherwise rewrite the
screen or set the clipboard.

Times are the readers' local wall-clock strings (`YYYY-MM-DDTHH:MM`, tools/applescript.iso),
so the day buckets compare DATES, never now ± 24 h: a DST change day is just a date, and an
invitation from another time zone is already in local time when Calendar reports it.

Not egress: Apple events and the memory file stay on this machine.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import diag
from textutil import clip

BUCKETS = ("overdue", "today", "waiting", "coming up", "notes")
SOURCES = ("calendar", "reminders", "mail", "memory")
PER_BUCKET = 5          # lines a bucket shows before "+N more"
TOTAL = 15              # lines the whole brief shows before "+N more"
MAIL_WINDOW = 25        # newest inbox messages scanned (list_mail's own cost: ~1 s + 0.25 s each)
MEMO_LINES = 3
COMMITMENT_DAYS = 7
FRESH_S = 600.0         # a finished gather younger than this is reused by /brief
_TEXT_CAP = 72


@dataclass(frozen=True)
class Item:
    bucket: str     # one of BUCKETS
    key: str        # sort key inside the bucket (a local timestamp, or a fixed rank)
    text: str       # one display line, already inert


@dataclass(frozen=True)
class Unavailable:
    reason: str


# ── inert text ────────────────────────────────────────────────────────────────────────────────

# C0/C1 controls, DEL, zero-width and bidi-control characters. ESC becomes a visible ␛ so a
# planted sequence reads as what it is; every other one becomes a space (tabs and newlines too:
# a brief line is one line). The local form, until textutil.visible_controls exists
# (docs/superpowers/plans/2026-10-01-terminal-escape-sanitising.md) — then use the shared form.
_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩﻿]")


def _inert(text, n: int = _TEXT_CAP) -> str:
    s = _UNSAFE.sub(lambda m: "␛" if m.group() == "\x1b" else " ", str(text or ""))
    return clip(s, n)


# ── dates ─────────────────────────────────────────────────────────────────────────────────────


def _stamp(now: datetime) -> str:
    """`now` as the readers report times: local wall clock to the minute."""
    return now.strftime("%Y-%m-%dT%H:%M")


def _day(d: date) -> str:
    return d.isoformat()


def _day_label(when, today: date) -> str:
    """'today' / 'yesterday' / 'tomorrow' / 'Wed 30 Sep' for an ISO date or timestamp."""
    try:
        d = date.fromisoformat(str(when)[:10])
    except ValueError:
        return _inert(when, 16)
    delta = (d - today).days
    if delta == 0:
        return "today"
    if delta == -1:
        return "yesterday"
    if delta == 1:
        return "tomorrow"
    return f"{d:%a} {d.day} {d:%b}"


# ── the four sources, as items ────────────────────────────────────────────────────────────────


def calendar_items(events, now: datetime) -> list[Item]:
    """Today's events that have not ended, and tomorrow's until noon. `events` is
    list_calendar_events' output: a list of dicts, or its 'No events…' string."""
    if not isinstance(events, list):
        return []
    today, tomorrow = now.date(), now.date() + timedelta(days=1)
    t0, t1 = f"{_day(today)}T00:00", f"{_day(tomorrow)}T00:00"
    noon1 = f"{_day(tomorrow)}T12:00"
    t2 = f"{_day(tomorrow + timedelta(days=1))}T00:00"
    cur = _stamp(now)
    out = []
    for e in events:
        start, end = str(e.get("start") or ""), str(e.get("end") or "")
        title = _inert(e.get("title") or "(untitled)")
        loc = _inert(e.get("location") or "", 32)
        what = f"{title} · {loc}" if loc else title
        if start < t1 and end > t0:
            # Overlaps today. Strictly: yesterday's all-day event ends AT today 00:00 — the
            # calendar query's `end date ≥` returns it, and it is not today's.
            if e.get("all_day"):
                out.append(Item("today", "0", f"all day  {what}"))  # "0" sorts before any time
            elif end <= cur:
                continue                                           # already over
            elif start < t0:
                out.append(Item("today", start, f"until {end[11:16]}  {what} (since yesterday)"))
            else:
                out.append(Item("today", start, f"{start[11:16]}–{end[11:16]}  {what}"))
        elif t1 <= start < t2 and (e.get("all_day") or start < noon1):
            label = "all day" if e.get("all_day") else start[11:16]
            out.append(Item("coming up", start, f"tomorrow {label}  {what}"))
    return out


def reminder_items(rows, now: datetime) -> list[Item]:
    """Overdue reminders, and those due today or tomorrow. `rows` is list_reminders' output;
    its own `overdue` flag decides lateness — one rule, the reader's: a date-only reminder is
    late once its day is over, a timed one once its time has passed."""
    if not isinstance(rows, list):
        return []
    today = now.date()
    out = []
    for r in rows:
        due = str(r.get("due") or "")
        if not due:
            continue                                   # an undated reminder is a list, not a day
        title = _inert(r.get("title") or "(untitled)")
        clock = "" if due.endswith("T00:00") else due[11:16]
        at = f" {clock}" if clock else ""
        if r.get("overdue"):
            out.append(Item("overdue", due, f"reminder · {title} (due {_day_label(due, today)}{at})"))
        elif due[:10] == _day(today):
            out.append(Item("today", due, f"reminder · {title}" + (f" ({clock})" if clock else "")))
        elif due[:10] == _day(today + timedelta(days=1)):
            out.append(Item("coming up", due, f"tomorrow{at}  reminder · {title}"))
    return out


_ADDRESS = re.compile(r"\s*<[^>]*>\s*$")


def _sender(raw) -> str:
    """'Petra Novak <petra@x.com>' → 'Petra Novak'; a bare address stays an address."""
    s = str(raw or "").strip()
    name = _ADDRESS.sub("", s).strip().strip('"').strip()
    return name or s.strip("<>")


def mail_items(rows, now: datetime) -> list[Item]:
    """Unread messages not replied to, among the newest MAIL_WINDOW in the inbox, newest
    first — the deterministic half of "waiting on you" (Mail cannot say which need an answer)."""
    if not isinstance(rows, list):
        return []
    today = now.date()
    out = []
    for i, m in enumerate(rows):
        if not m.get("unread") or m.get("replied"):
            continue
        who = _inert(_sender(m.get("from")), 28)
        subject = _inert(m.get("subject") or "(no subject)", 56)
        out.append(Item("waiting", f"{i:04d}", f"{who} · {subject} ({_day_label(m.get('date'), today)})"))
    return out


_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}")


def memory_items(commitments, memos, now: datetime) -> list[Item]:
    """Commitments with a due date (overdue, today, within COMMITMENT_DAYS), a count of the
    undated ones, and the last MEMO_LINES memo entries. `due=` is free text in the memory file
    (hand edits happen): only a leading ISO date counts as a date."""
    today = now.date()
    horizon = today + timedelta(days=COMMITMENT_DAYS)
    out, undated = [], 0
    for e in commitments:
        due = str(e.get("due") or "")
        try:
            d = date.fromisoformat(due[:10]) if _ISO_DAY.match(due) else None
        except ValueError:
            d = None
        if d is None:
            undated += 1
            continue
        tag = f"commitment #{e['id']}" if e.get("id") else "commitment"
        line = f"{tag} · {_inert(e.get('text'))} (due {_day_label(due, today)})"
        if d < today:
            out.append(Item("overdue", _day(d), line))
        elif d == today:
            out.append(Item("today", _day(d), line))
        elif d <= horizon:
            out.append(Item("coming up", _day(d) + "~", line))  # after that day's timed items
    if undated:
        out.append(Item("coming up", "~", f"{undated} open commitment(s) without a date — /memory"))
    for e in list(memos)[-MEMO_LINES:]:
        out.append(Item("notes", str(e.get("date") or ""),
                        f"{_day_label(e.get('date'), today)} · {_inert(e.get('text'))}"))
    return out
```

**The shared form.** If `textutil.visible_controls` already exists (the terminal-escape plan
landed first), replace the `_UNSAFE` block and `_inert` above with the block below, and import
it with `from textutil import clip, visible_controls`:

```python
# Other people's text, made inert: every terminal control becomes a visible picture through the
# shared neutraliser (ESC -> ␛; textutil.visible_controls), and bidi / zero-width marks — shown
# by code point at the gate, harmless in an answer — become spaces here, where a line is one
# line of someone else's words. clip() then folds tabs and newlines into single spaces.
_BIDI = re.compile(r"[\u061c\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def _inert(text, n: int = _TEXT_CAP) -> str:
    return clip(_BIDI.sub(" ", visible_controls(text)), n)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add core/brief.py tests/test_brief.py
git commit -m "brief: today's lines from calendar, reminders, mail and memory, made inert"
```

---

### Task 2: Compose — urgency buckets under a cap

**Files:**
- Modify: `core/brief.py` (append the composing section)
- Modify: `tests/test_brief.py` (append)

**Interfaces:**
- Consumes: `Item`, `Unavailable`, `BUCKETS`, `SOURCES`, `PER_BUCKET`, `TOTAL` (Task 1).
- Produces: `Brief` (dataclass: `gathered`, `buckets`, `hidden`, `unavailable`, `pending`, property `empty`) and `compose(results, gathered, *, pending=(), capped=True) -> Brief`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brief.py`:

```python
# ── Task 2: compose ───────────────────────────────────────────────────────────────────────────

def test_compose_orders_buckets_by_urgency_and_items_by_key():
    results = {
        "calendar": [Item("today", "2026-10-02T09:30", "09:30 1:1"), Item("today", "0", "all day x")],
        "reminders": [Item("overdue", "2026-09-30T09:00", "reminder late")],
        "mail": [Item("waiting", "0000", "Petra")],
        "memory": [Item("notes", "2026-09-29", "a note")],
    }
    b = brief.compose(results, NOW)
    assert list(b.buckets) == list(brief.BUCKETS)
    assert b.buckets["overdue"] == ["reminder late"]
    assert b.buckets["today"] == ["all day x", "09:30 1:1"]
    assert not b.empty and not b.pending and not b.unavailable


def test_compose_caps_each_bucket_and_the_whole_brief():
    many = {"mail": [Item("waiting", f"{i:04d}", f"m{i}") for i in range(9)],
            "calendar": [Item("today", f"2026-10-02T1{i}:00", f"e{i}") for i in range(7)],
            "memory": [Item("notes", f"2026-09-2{i}", f"n{i}") for i in range(6)]}
    b = brief.compose(many, NOW)
    assert b.buckets["today"] == ["e0", "e1", "e2", "e3", "e4"] and b.hidden["today"] == 2
    assert b.buckets["waiting"] == ["m0", "m1", "m2", "m3", "m4"] and b.hidden["waiting"] == 4
    assert b.buckets["notes"] == ["n0", "n1", "n2", "n3", "n4"] and b.hidden["notes"] == 1
    many["memory"].append(Item("coming up", "2026-10-03", "c0"))
    b = brief.compose(many, NOW)                   # 5 + 5 + 1 leaves room for 4 of the notes
    assert b.buckets["coming up"] == ["c0"]
    assert b.buckets["notes"] == ["n0", "n1", "n2", "n3"] and b.hidden["notes"] == 2
    full = brief.compose(many, NOW, capped=False)
    assert len(full.buckets["waiting"]) == 9 and full.hidden["waiting"] == 0


def test_compose_groups_unavailable_sources_by_reason():
    why = "this tool is only available on macOS (this is linux)"
    b = brief.compose({"calendar": Unavailable(why), "reminders": Unavailable(why),
                       "mail": Unavailable(why), "memory": []}, NOW)
    assert b.unavailable == {why: ["calendar", "reminders", "mail"]}
    assert b.empty
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q -k compose`
Expected: 3 failed — `AttributeError: module 'core.brief' has no attribute 'compose'`.

- [ ] **Step 3: Append the composing section to `core/brief.py`**

```python
# ── composing ─────────────────────────────────────────────────────────────────────────────────


@dataclass
class Brief:
    gathered: datetime
    buckets: dict = field(default_factory=dict)       # bucket -> [line, …] shown
    hidden: dict = field(default_factory=dict)        # bucket -> lines past the cap
    unavailable: dict = field(default_factory=dict)   # reason -> [source, …]
    pending: list = field(default_factory=list)       # sources still gathering

    @property
    def empty(self) -> bool:
        return not any(self.buckets.values()) and not any(self.hidden.values())


def compose(results: dict, gathered: datetime, *, pending=(), capped: bool = True) -> Brief:
    """Sort every source's items into BUCKETS (in that order, each by its key; equal keys keep
    SOURCES order) under PER_BUCKET and TOTAL; `capped=False` shows everything. Sources that
    failed are grouped by reason, so three "only available on macOS" become one line."""
    items: list[Item] = []
    unavailable: dict = {}
    for name in SOURCES:
        r = results.get(name)
        if isinstance(r, Unavailable):
            unavailable.setdefault(r.reason, []).append(name)
        elif r:
            items.extend(r)
    shown, hidden, room = {}, {}, TOTAL
    for b in BUCKETS:
        lines = [i.text for i in sorted((i for i in items if i.bucket == b), key=lambda i: i.key)]
        take = len(lines) if not capped else max(0, min(PER_BUCKET, room, len(lines)))
        shown[b], hidden[b] = lines[:take], len(lines) - take
        room -= take
    return Brief(gathered, shown, hidden, unavailable, [n for n in SOURCES if n in set(pending)])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add core/brief.py tests/test_brief.py
git commit -m "brief: compose into overdue, today, waiting, coming up, notes under a cap"
```

---

### Task 3: Gathering — a thread per source, the session cache, the switch, and the real-apps guard

**Files:**
- Modify: `core/brief.py` (append the gathering section)
- Modify: `config.default.yaml` (the `runtime.brief` key)
- Modify: `tests/conftest.py` (the `_no_real_apple_events` guard)
- Modify: `tests/test_brief.py` (append)

**Interfaces:**
- Consumes: `compose`, `Brief` (Task 2); the item builders (Task 1); `tools.calendar.list_calendar_events`, `tools.reminders.list_reminders`, `tools.mail.list_mail` (LangChain `StructuredTool`s — `.func` is the decorated function, which raises `ToolError` on failure); `stores.memory_registry.entries`; `tools.applescript._platform`; `config.get_config().get`.
- Produces: `Gather`, `start(*, sources=None, refresh=False) -> Gather`, `current()`, `reset()`, `enabled() -> bool`, the private sources `_calendar`, `_reminders`, `_mail`, `_memory`, `_default_sources()`, and `_now()`.

- [ ] **Step 1: Guard the real apps first**

Before any test in this task can reach a reader, make it impossible for a test to drive the
real apps. Insert directly above the `mac` fixture in `tests/conftest.py` (the `@pytest.fixture`
line before `def mac(monkeypatch):`):

```python
@pytest.fixture(autouse=True)
def _no_real_apple_events(monkeypatch):
    """No test may drive the user's real apps: osascript (and the `open -gja` launch before it)
    fails loudly unless the test asked for the captured `mac` controller below, whose own patch
    replaces this one. Found writing the launch brief: a /brief --refresh test gathered from the
    real Calendar, Reminders and Mail."""
    from tools import applescript

    def refuse(argv, timeout):
        raise AssertionError(f"a test tried to run {argv[0]!r} against the real apps — use the `mac` fixture")

    monkeypatch.setattr(applescript, "_run", refuse)
```

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: every test passes (1,301 at the time of writing, plus Tasks 1–2's 11). Any test that now fails with "a test tried to run 'osascript' against the real apps" was driving the real apps. Give it the `mac` fixture; never weaken the guard.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_brief.py`:

```python
# ── Task 3: gathering ─────────────────────────────────────────────────────────────────────────

# Every test starts and ends without a session brief (start() reuses one in flight).
@pytest.fixture(autouse=True)
def _fresh_brief():
    brief.reset()
    yield
    brief.reset()


def test_a_failing_source_becomes_unavailable_and_the_rest_still_lands():
    from tools.toolspec import ToolError

    boom, ev1 = _gated(ToolError("macOS denied automation access to Mail; allow this terminal"))
    ok, ev2 = _gated([Item("today", "0", "all day x")])
    g = brief.start(sources={"mail": boom, "calendar": ok})
    ev1.set(), ev2.set()
    assert g.wait(5)
    b = g.compose()
    assert b.buckets["today"] == ["all day x"]
    assert b.unavailable == {"macOS denied automation access to Mail; allow this terminal": ["mail"]}


def test_wait_returns_what_landed_and_names_what_is_pending():
    slow, ev_slow = _gated([Item("waiting", "0000", "Petra")])
    fast, ev_fast = _gated([Item("today", "0", "all day x")])
    g = brief.start(sources={"mail": slow, "calendar": fast})
    ev_fast.set()
    assert g.wait(0.2) is False
    b = g.compose()
    assert b.buckets["today"] == ["all day x"] and b.pending == ["mail"]
    ev_slow.set()
    assert g.wait(5) and g.compose().pending == []


def test_start_reuses_an_in_flight_or_fresh_gather_and_refresh_replaces_a_finished_one():
    fn, ev = _gated([])
    g1 = brief.start(sources={"memory": fn})
    assert brief.start(refresh=True) is g1          # in flight: a second gather would queue behind it
    ev.set()
    assert g1.wait(5)
    assert brief.start() is g1                      # finished and fresh
    fn2, ev2 = _gated([])
    g2 = brief.start(sources={"memory": fn2}, refresh=True)
    assert g2 is not g1 and brief.current() is g2
    ev2.set()


def test_the_real_sources_read_in_bulk_and_one_mailbox(mac, monkeypatch, isolated_paths):
    import tools.reminders as rem

    monkeypatch.setattr(rem, "_now", lambda: NOW)
    mac.reply(_rows(("Home", "u1", "1:1 with Petra", "2026-10-02T09:30", "2026-10-02T10:00",
                     "false", "", "")))
    assert [i.text for i in brief._calendar(NOW)] == ["09:30–10:00  1:1 with Petra"]
    assert mac.calls[-2] == ["open", "-gja", "Calendar"]
    s = mac.script()
    assert "my mkdate(2026, 10, 2, 0)" in s and "my mkdate(2026, 10, 4, 0)" in s

    mac.reply(_rows(("x-apple-reminder://B", "Home", "Pay rent", "2026-10-02T00:00", "", "false")))
    assert [i.text for i in brief._reminders(NOW)] == ["reminder · Pay rent"]
    assert "properties of (reminders of l whose completed is false)" in mac.script()

    mac.reply(_rows(("7", "Petra <p@x>", "Thursday?", "2026-10-02T07:40", "false", "false")))
    assert [i.text for i in brief._mail(NOW)] == ["Petra · Thursday? (today)"]
    s = mac.script()
    assert "messages 1 thru 25 of inbox" in s and "sent mailbox" not in s   # one mailbox, 25 newest

    from stores.memory_registry import add_memory

    add_memory("Book the vet", layer="commitments", due="2026-10-02")
    assert [i.text for i in brief._memory(NOW)] == ["commitment #1 · Book the vet (due today)"]


def test_the_brief_is_not_egress(mac, monkeypatch, isolated_paths):
    import tools.reminders as rem
    from trust import egress

    monkeypatch.setattr(rem, "_now", lambda: NOW)
    monkeypatch.setattr(brief, "_now", lambda: NOW)
    mac.reply("")
    mark = egress.next_seq()
    g = brief.start()
    assert g.wait(5)
    assert egress.events_since(mark) == []


def test_enabled_needs_macos_and_the_setting(monkeypatch):
    from config import get_config
    from tools import applescript

    monkeypatch.setitem(get_config()._data.setdefault("runtime", {}), "brief", True)
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    assert brief.enabled() is False
    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    assert brief.enabled() is True
    monkeypatch.setitem(get_config()._data["runtime"], "brief", False)
    assert brief.enabled() is False


def test_no_test_reaches_the_real_apps(monkeypatch):
    from tools import applescript

    monkeypatch.setattr(applescript, "_platform", lambda: "darwin")
    with pytest.raises(AssertionError, match="use the `mac` fixture"):
        applescript.run('return "x"', app="Calendar")


def test_the_template_ships_the_brief_off():
    import yaml
    from pathlib import Path

    data = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.default.yaml").read_text())
    assert data["runtime"]["brief"] is False
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: `19 errors`. The new autouse `_fresh_brief` fixture calls `brief.reset()`, which does not exist yet (`AttributeError: module 'core.brief' has no attribute 'reset'`), so every test in the file errors at setup. Once Step 5 lands, the new tests also check `start`, `_calendar`, `enabled` and the template's `runtime.brief`.

- [ ] **Step 4: Add the switch to `config.default.yaml`**

Under `runtime:`, directly after the `grant_scope: task` line:

```yaml
  # The launch brief (/brief, core/brief.py): one dim block before the first prompt — overdue
  # reminders and commitments, what is left of today's calendar, unread mail you have not
  # answered, tomorrow morning. No model call; read-only; never added to the conversation.
  # Off until `/brief on`: the first gather is when macOS asks for access to Calendar,
  # Reminders and Mail, and that should happen while you watch, not behind a launch. macOS only.
  brief: false
```

- [ ] **Step 5: Append the gathering section to `core/brief.py`**

```python
# ── gathering ─────────────────────────────────────────────────────────────────────────────────


def _reason(exc: Exception) -> str:
    return clip(str(exc) or type(exc).__name__, 160)


class Gather:
    """One brief being gathered: a daemon thread per source, results kept as they land. Safe to
    read from any thread; nothing here prints. `shown` is the renderer's record of which
    sources the user has already seen (commands/brief.py)."""

    def __init__(self, sources: dict, now: datetime):
        self.now = now
        self.started = time.monotonic()
        self.shown: set = set()
        self._sources = dict(sources)
        self._results: dict = {}
        self._lock = threading.Lock()
        self._done = {name: threading.Event() for name in self._sources}

    def _run(self, name: str, fn) -> None:
        t = time.perf_counter()
        try:
            result = fn(self.now)
        except Exception as exc:  # a source never takes the brief down
            result = Unavailable(_reason(exc))
        with self._lock:
            self._results[name] = result
        self._done[name].set()
        diag.log(f"brief : {name} {time.perf_counter() - t:.2f}s")

    def begin(self) -> "Gather":
        for name, fn in self._sources.items():
            threading.Thread(target=self._run, args=(name, fn), name=f"brief-{name}",
                             daemon=True).start()
        return self

    def wait(self, timeout: float) -> bool:
        """Block up to `timeout` seconds for every source; True when all have landed."""
        deadline = time.monotonic() + max(0.0, timeout)
        for ev in self._done.values():
            if not ev.wait(max(0.0, deadline - time.monotonic())):
                return False
        return True

    def done(self) -> bool:
        return all(ev.is_set() for ev in self._done.values())

    def pending(self) -> list:
        return [n for n, ev in self._done.items() if not ev.is_set()]

    def results(self) -> dict:
        with self._lock:
            return dict(self._results)

    def age(self) -> float:
        return time.monotonic() - self.started

    def compose(self, *, capped: bool = True, only=None) -> Brief:
        results = self.results()
        if only is not None:
            results = {n: r for n, r in results.items() if n in only}
            return compose(results, self.now, capped=capped)
        return compose(results, self.now, pending=self.pending(), capped=capped)


def _calendar(now: datetime) -> list[Item]:
    from tools.calendar import list_calendar_events

    today = now.date()
    return calendar_items(list_calendar_events.func(
        start=_day(today), end=_day(today + timedelta(days=2))), now)


def _reminders(now: datetime) -> list[Item]:
    from tools.reminders import list_reminders

    return reminder_items(list_reminders.func(), now)


def _mail(now: datetime) -> list[Item]:
    from tools.mail import list_mail

    return mail_items(list_mail.func(mailbox="inbox", limit=MAIL_WINDOW), now)


def _memory(now: datetime) -> list[Item]:
    from stores import memory_registry as mem

    return memory_items(mem.entries("commitments"), mem.entries("memo"), now)


def _default_sources() -> dict:
    # Looked up at call time, so a test that replaces one source replaces it here.
    return {"calendar": _calendar, "reminders": _reminders, "mail": _mail, "memory": _memory}


def _now() -> datetime:
    return datetime.now().astimezone()


_current: "Gather | None" = None
_current_lock = threading.Lock()


def start(*, sources: "dict | None" = None, refresh: bool = False) -> Gather:
    """The session's brief: the one in flight if there is one (Apple events to one app run one
    after another, so a second gather would only queue behind the first), the last one if it
    finished under FRESH_S ago and `refresh` is false, else a new one."""
    global _current
    with _current_lock:
        g = _current
        if g is not None and (not g.done() or (not refresh and g.age() < FRESH_S)):
            return g
        _current = Gather(sources or _default_sources(), _now()).begin()
        return _current


def current() -> "Gather | None":
    return _current


def reset() -> None:
    """Forget the session's brief (tests)."""
    global _current
    with _current_lock:
        _current = None


def enabled() -> bool:
    """`runtime.brief`, on macOS only — whether an interactive launch gathers a brief."""
    from config import get_config
    from tools import applescript

    return applescript._platform() == "darwin" and bool(get_config().get("runtime.brief", False))
```

- [ ] **Step 6: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: `19 passed`.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: every test passes.

- [ ] **Step 7: Commit**

```bash
git add core/brief.py config.default.yaml tests/conftest.py tests/test_brief.py
git commit -m "brief: gather each source on its own thread; tests can no longer reach the real apps"
```

---

### Task 4: `/brief` — the renderer and the command

**Files:**
- Create: `commands/brief.py`
- Modify: `commands/__init__.py` (`_COMMAND_MODULES`)
- Modify: `commands/system.py` (`_GROUPS`, `_DAILY`)
- Modify: `tests/test_brief.py` (append)

**Interfaces:**
- Consumes: `core.brief.start`, `Gather.wait/compose/done/shown`, `BUCKETS`, `SOURCES`, `Brief` (Tasks 2–3); `tui.ui.section(title, subtitle)`, `ui.table(rows, styles)` (cells may be `(text, style)` tuples; rendered as `rich.Text`, so markup in a subject is never interpreted), `ui.note(msg)`; `config.persist(key)` (raises `KeyError` when the user's `config.yaml` has no such line); `commands._framework.command`, `_print`.
- Produces: `commands.brief.render(b, *, title="brief", footer="full")`, `_mark_shown(g, b)`, the `/brief` command; `_LAUNCH_WAIT_S` is defined here for Task 5.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brief.py`:

```python
# ── Task 4: /brief ────────────────────────────────────────────────────────────────────────────

def _landed(items_by_source):
    """A finished gather over fixed results, installed as the session's brief."""
    g = brief.start(sources={n: (lambda now, r=r: r) for n, r in items_by_source.items()})
    assert g.wait(5)
    return g


def test_brief_command_renders_buckets_in_order_with_labels(ctx, capsys):
    from commands import dispatch

    _landed({"reminders": [Item("overdue", "1", "reminder · Call the dentist (due Wed 30 Sep)")],
             "calendar": [Item("today", "2026-10-02T09:30", "09:30–10:00  1:1 with Petra")],
             "mail": [Item("waiting", "0000", "Petra · Thursday? (today)")]})
    dispatch("/brief", ctx)
    out = capsys.readouterr().out
    assert "── brief" in out and "gathered" in out
    assert out.index("overdue") < out.index("today") < out.index("waiting")
    assert "reminder · Call the dentist" in out and "Petra · Thursday?" in out
    assert ctx.state == {}                       # printed for the user, never added to the conversation


def test_brief_command_says_so_when_there_is_nothing(ctx, capsys):
    from commands import dispatch

    _landed({"calendar": [], "memory": []})
    dispatch("/brief", ctx)
    assert "nothing on today" in capsys.readouterr().out


def test_brief_command_names_unavailable_sources_with_the_reason(ctx, capsys):
    from commands import dispatch

    _landed({"calendar": Unavailable("macOS denied automation access to Calendar; allow this terminal"),
             "memory": [Item("notes", "x", "Tue 29 Sep · a note")]})
    dispatch("/brief", ctx)
    out = capsys.readouterr().out
    assert "calendar: not available — macOS denied automation access to Calendar" in out
    assert "a note" in out


def test_brief_all_lifts_the_cap(ctx, capsys):
    from commands import dispatch

    _landed({"mail": [Item("waiting", f"{i:04d}", f"sender{i} · s") for i in range(8)]})
    dispatch("/brief", ctx)
    out = capsys.readouterr().out
    assert "sender4" in out and "sender5" not in out and "+3 more — /brief --all" in out
    dispatch("/brief --all", ctx)
    assert "sender7" in capsys.readouterr().out


def test_ctrl_c_while_waiting_does_not_end_the_session(ctx, capsys, monkeypatch):
    from commands import dispatch

    fn, ev = _gated([])
    brief.start(sources={"mail": fn})

    def interrupted(self, timeout):
        raise KeyboardInterrupt

    monkeypatch.setattr(brief.Gather, "wait", interrupted)
    dispatch("/brief", ctx)                      # must return, not raise
    assert "stopped waiting" in capsys.readouterr().out
    ev.set()


def test_brief_on_persists_and_gathers_off_turns_it_off(ctx, capsys, mac, recording_persist,
                                                        monkeypatch):
    from commands import dispatch
    from config import get_config

    monkeypatch.setitem(get_config()._data.setdefault("runtime", {}), "brief", False)
    _landed({"memory": []})
    dispatch("/brief on", ctx)
    assert get_config().get("runtime.brief") is True and recording_persist == ["runtime.brief"]
    out = capsys.readouterr().out
    assert "shows before the first prompt of every launch" in out and "nothing on today" in out
    dispatch("/brief off", ctx)
    assert get_config().get("runtime.brief") is False


def test_brief_on_says_how_to_keep_it_when_config_has_no_line(ctx, capsys, mac, monkeypatch):
    import config
    from commands import dispatch

    monkeypatch.setitem(config.get_config()._data.setdefault("runtime", {}), "brief", False)

    def no_line(key):
        raise KeyError(f"{key} not found in config.yaml as an editable scalar")

    monkeypatch.setattr(config, "persist", no_line)
    _landed({"memory": []})
    dispatch("/brief on", ctx)
    out = capsys.readouterr().out
    assert "for this session" in out and "brief: true" in out and "runtime:" in out


def test_brief_on_refuses_off_macos(ctx, capsys, monkeypatch):
    from commands import dispatch
    from config import get_config
    from tools import applescript

    monkeypatch.setitem(get_config()._data.setdefault("runtime", {}), "brief", False)
    monkeypatch.setattr(applescript, "_platform", lambda: "linux")
    dispatch("/brief on", ctx)
    assert "macOS apps" in capsys.readouterr().out and get_config().get("runtime.brief") is False


def test_brief_rejects_an_unknown_argument(ctx, capsys):
    from commands import dispatch

    dispatch("/brief tomorrow", ctx)
    assert "unknown argument: tomorrow" in capsys.readouterr().out
    assert brief.current() is None               # nothing was gathered


def test_brief_is_a_daily_command_in_help(ctx, capsys):
    from commands import dispatch

    dispatch("/help", ctx)
    assert "/brief" in capsys.readouterr().out
    dispatch("/brief --help", ctx)
    assert "waiting" in capsys.readouterr().out and brief.current() is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: 10 failed. The output reads `unknown command: /brief - try /help`, so the assertions on "── brief", "nothing on today" and so on fail, and `test_brief_is_a_daily_command_in_help` finds no `/brief` in `/help`.

- [ ] **Step 3: Create `commands/brief.py`**

This is the module without its launch-view section, which Task 5 appends.

```python
"""
/brief — today at a glance (core/brief.py), and the launch view that shares its renderer.

The brief is read-only and model-free, and it is printed for the user only: nothing here
touches `ctx.state`, so a brief never becomes part of the conversation.
"""

from __future__ import annotations

from commands._framework import command, _print
from core import brief

_LAUNCH_WAIT_S = 2.0     # the longest the first prompt waits for the launch brief
_COMMAND_WAIT_S = 30.0   # /brief waits this long, then shows what came back
_LABEL_STYLE = {"overdue": "yellow"}
_USAGE = "/brief [--all | --refresh | on | off]"


def render(b: "brief.Brief", *, title: str = "brief", footer: str = "full") -> None:
    """The one renderer: a `── brief` section, one aligned row per line under its bucket label,
    `+N more` where a cap cut, then the footer — `full` names every unavailable source with
    its reason (/brief), `quiet` only names them (the launch view, which prints every launch)."""
    from tui import ui

    g = b.gathered
    if not b.empty:
        ui.section(title, f"{g:%a} {g.day} {g:%b} · gathered {g:%H:%M}")
        rows = []
        for bucket in brief.BUCKETS:
            lines = b.buckets.get(bucket) or []
            for i, line in enumerate(lines):
                rows.append(((bucket if i == 0 else "", _LABEL_STYLE.get(bucket, "dim")), line))
            if b.hidden.get(bucket):
                rows.append((("" if lines else bucket, "dim"),
                             (f"+{b.hidden[bucket]} more — /brief --all", "dim")))
        ui.table(rows, styles=["dim", None])
    if footer == "full":
        for reason, names in b.unavailable.items():
            ui.note(f"{', '.join(names)}: not available — {reason}")
    else:
        names = [n for group in b.unavailable.values() for n in group]
        if names:
            ui.note(f"brief: {', '.join(names)} not available — /brief says why")
    if b.pending:
        ui.note(f"brief: still gathering {', '.join(b.pending)}")


def _mark_shown(g: "brief.Gather", b: "brief.Brief") -> None:
    g.shown.update(n for n in brief.SOURCES if n not in b.pending)


@command(
    "brief",
    "Today at a glance: events, reminders, unread mail, commitments.",
    usage=_USAGE,
    details="""
One block, no model call, nothing sent anywhere:

  overdue     reminders and commitments past their date
  today       events still ahead today (all-day ones first), reminders and commitments due
  waiting     unread mail you have not replied to, among the newest 25 in your inbox
  coming up   tomorrow until noon, commitments due this week
  notes       your last three memo entries

  /brief             show it (reuses one gathered in the last ten minutes)
  /brief --all       show every line (the brief stops at 5 a bucket, 15 in all)
  /brief --refresh   gather it again now
  /brief on | off    show it before the first prompt of every launch (runtime.brief)

Calendar, Reminders and Mail are read through AppleScript: the first /brief is when macOS
asks for access to each app, and Mail can take ten seconds on a large inbox. A source that
cannot be read says why and the rest still shows. Subjects and titles are other people's
words: they are shown as plain text, and the brief never becomes part of the conversation.
macOS only (memory works everywhere).
""",
)
def _brief(ctx, args):
    sub = args[0].lower() if args else ""
    if sub in ("on", "off"):
        return _toggle(sub == "on")
    if sub not in ("", "--all", "-a", "all", "--refresh", "-r", "refresh"):
        _print(f"  unknown argument: {args[0]} — usage: {_USAGE}")
        return
    _show(refresh=sub in ("--refresh", "-r", "refresh"), capped=sub not in ("--all", "-a", "all"))


def _show(*, refresh: bool = False, capped: bool = True) -> None:
    from tui import ui

    g = brief.start(refresh=refresh)
    if not g.done():
        ui.note("gathering — Calendar and Mail can take a few seconds (Ctrl-C stops waiting)")
        try:
            g.wait(_COMMAND_WAIT_S)
        except KeyboardInterrupt:  # dispatch catches Exception only: Ctrl-C would end the REPL
            ui.note("stopped waiting — the brief keeps gathering; /brief shows it")
            return
    b = g.compose(capped=capped)
    _mark_shown(g, b)
    if b.empty and not b.unavailable and not b.pending:
        ui.note("nothing on today — no events left, nothing overdue or due, no unread mail waiting")
        return
    render(b)


def _toggle(on: bool) -> None:
    import config
    from tools import applescript
    from tui import ui

    if on and applescript._platform() != "darwin":
        ui.note("the launch brief reads Calendar, Reminders and Mail — macOS apps; "
                "/brief still shows your commitments here")
        return
    config.get_config().set("runtime.brief", on)
    state = "shows before the first prompt of every launch" if on else "no longer shows at launch"
    try:
        path = config.persist("runtime.brief")
        ui.note(f"the brief {state} (saved to {path.name})")
    except (KeyError, ValueError):
        # A config.yaml seeded before this key existed has no line to edit, and persist only
        # edits lines in place.
        ui.note(f"the brief {state} for this session — to keep it, add `brief: "
                f"{'true' if on else 'false'}` under `runtime:` in {config._CONFIG_PATH}")
    if on:
        _show()  # the first gather is when macOS asks for access: let it happen while you watch
```

- [ ] **Step 4: Register the module and list it in `/help`**

In `commands/__init__.py`, add the first entry of `_COMMAND_MODULES`:

```python
_COMMAND_MODULES = [
    "brief",         # /brief — today at a glance (core/brief.py), and the launch view
    "config",        # /config — owns the persist seam others import
```

In `commands/system.py`, replace the `knowledge & workspace` row of `_GROUPS` and the `_DAILY` line:

```python
    ("knowledge & workspace", ("add-dir", "brief", "docs", "init", "memory", "rm-dir", "undo")),
```

```python
_DAILY: tuple[str, ...] = ("brief", "memory", "policy", "trace", "help", "quit")
```

(`tests/test_help.py` checks that every registered command sits in exactly one group, alphabetical inside it, and that every `_DAILY` name shows in bare `/help`.)

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_brief.py tests/test_help.py -q`
Expected: all pass (29 in `test_brief.py`).

- [ ] **Step 6: Commit**

```bash
git add commands/brief.py commands/__init__.py commands/system.py tests/test_brief.py
git commit -m "brief: /brief shows today at a glance; on|off sets the launch brief"
```

---

### Task 5: The launch view — before the first prompt, and late sources between prompts

**Files:**
- Modify: `commands/brief.py` (append the launch-view section)
- Modify: `app/repl.py` (four insertions in `run_repl`)
- Modify: `tests/test_brief.py` (append)

**Interfaces:**
- Consumes: `render`, `_mark_shown`, `_LAUNCH_WAIT_S` (Task 4); `core.brief.start`, `current`, `enabled`, `Gather.compose(only=…)` (Task 3).
- Produces: `commands.brief.show_at_launch(g, wait=_LAUNCH_WAIT_S) -> None`, `show_late(g) -> bool`, `first_run_hint() -> None`; `run_repl` gathers when `runtime.brief` is on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brief.py`:

```python
# ── Task 5: the launch view ───────────────────────────────────────────────────────────────────

def test_launch_view_is_silent_when_there_is_nothing(capsys):
    from commands.brief import show_at_launch

    g = _landed({"calendar": [], "mail": [], "memory": []})
    show_at_launch(g, wait=0)
    assert capsys.readouterr().out == ""


def test_launch_view_names_unavailable_sources_quietly(capsys):
    from commands.brief import show_at_launch

    g = _landed({"mail": Unavailable("macOS denied automation access to Mail; allow this terminal "
                                     "under System Settings > Privacy & Security > Automation")})
    show_at_launch(g, wait=0)
    out = capsys.readouterr().out
    assert "brief: mail not available — /brief says why" in out and "System Settings" not in out


def test_late_sources_show_once_at_the_next_prompt(capsys):
    from commands.brief import show_at_launch, show_late

    slow, ev = _gated([Item("waiting", "0000", "Petra · Thursday? (today)")])
    g = brief.start(sources={"mail": slow, "calendar": lambda now: [Item("today", "0", "all day  x")]})
    for _ in range(100):                         # let the instant source land
        if "calendar" in g.results():
            break
        threading.Event().wait(0.01)
    show_at_launch(g, wait=0)
    out = capsys.readouterr().out
    assert "all day  x" in out and "still gathering mail" in out and "Petra" not in out
    assert show_late(g) is False and capsys.readouterr().out == ""   # mail not landed yet
    ev.set()
    assert g.wait(5)
    assert show_late(g) is True
    out = capsys.readouterr().out
    assert "brief, continued" in out and "Petra" in out and "all day  x" not in out
    assert show_late(g) is True and capsys.readouterr().out == ""   # once


def test_a_refreshed_brief_drops_the_launch_gathers_late_lines(ctx, capsys, monkeypatch):
    from commands import dispatch
    from commands.brief import show_late

    g = _landed({"memory": [Item("notes", "x", "old note")]})
    monkeypatch.setattr(brief, "_default_sources", lambda: {"memory": lambda now: []})
    dispatch("/brief --refresh", ctx)            # g finished: --refresh replaces it
    capsys.readouterr()
    assert brief.current() is not g and show_late(g) is True
    assert capsys.readouterr().out == ""


def test_headless_never_gathers_a_brief():
    import inspect

    import app.headless

    assert "brief" not in inspect.getsource(app.headless)


def test_the_repl_starts_and_shows_the_launch_brief():
    import inspect

    import app.repl

    src = inspect.getsource(app.repl.run_repl)
    assert "_brief.enabled()" in src and "show_at_launch" in src and "show_late" in src
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: 5 failed. Four fail on `ImportError: cannot import name 'show_at_launch'` (or `show_late`), and `test_the_repl_starts_and_shows_the_launch_brief` finds no `_brief.enabled()` in `run_repl`. `test_headless_never_gathers_a_brief` already passes: it is a guard.

- [ ] **Step 3: Append the launch view to `commands/brief.py`**

```python
# ── the launch view (app/repl.py) ─────────────────────────────────────────────────────────────


def show_at_launch(g: "brief.Gather", wait: float = _LAUNCH_WAIT_S) -> None:
    """Before the first prompt: wait up to `wait` seconds, then show what came back. Silent
    when there is nothing to show; a source still gathering is shown once it lands, at the next
    prompt (show_late) — never while the prompt is open, where it would tear the input line."""
    g.wait(wait)
    b = g.compose()
    render(b, footer="quiet")
    _mark_shown(g, b)


def show_late(g: "brief.Gather") -> bool:
    """At a prompt boundary after launch: show the sources that landed after the launch view,
    once. True when nothing is left to show (the caller stops asking). A gather that is no
    longer the session's (`/brief --refresh` replaced it) is dropped unshown."""
    if brief.current() is not g:
        return True
    late = [n for n in brief.SOURCES if n in g.results() and n not in g.shown]
    if late:
        b = g.compose(only=late)
        render(b, title="brief, continued", footer="quiet")
        g.shown.update(late)
    return g.done() and all(n in g.shown for n in brief.SOURCES)


def first_run_hint() -> None:
    """One line on the first launch (macOS only): the brief exists and is off."""
    from tools import applescript
    from tui import ui

    if applescript._platform() == "darwin":
        ui.note("/brief shows today's events, reminders and unread mail · /brief on shows it at "
                "every launch (macOS asks once for access to each app)")
```

- [ ] **Step 4: Wire it into `app/repl.py::run_repl`**

Four insertions; each shows the existing lines around it.

(a) Start the gather right after the splash. Replace

```python
    if ingest_warning:
        ui.warn(ingest_warning)
    tracer = Tracer(DB_PATH)
```

with

```python
    if ingest_warning:
        ui.warn(ingest_warning)
    # The launch brief (core/brief.py), when `runtime.brief` is on: gathered on background
    # threads while the rest of startup runs, shown just before the first prompt. Started only
    # now, after the splash imported the tool registry — a reader imported from a thread first
    # would reorder the registered tools, and the tool catalog is part of the cached prefix.
    from core import brief as _brief

    _launch_brief = _brief.start() if _brief.enabled() else None
    tracer = Tracer(DB_PATH)
```

(b) The first-run hint. Inside `if _first_run:`, replace

```python
        _health_check()
        try:
            _setup_sentinel.parent.mkdir(parents=True, exist_ok=True)
```

with

```python
        _health_check()
        from commands.brief import first_run_hint

        first_run_hint()
        try:
            _setup_sentinel.parent.mkdir(parents=True, exist_ok=True)
```

(c) The launch view, last before the prompt. Replace

```python
    except Exception as exc:
        diag.log(f"memory review: pending check failed: {exc}")
```

with

```python
    except Exception as exc:
        diag.log(f"memory review: pending check failed: {exc}")

    # The launch brief, last before the prompt: a short wait for what is ready, silence when
    # there is nothing. A launch never depends on it.
    if _launch_brief is not None:
        try:
            from commands.brief import show_at_launch

            show_at_launch(_launch_brief)
        except Exception as exc:
            diag.log(f"brief: launch view failed: {exc}")
            _launch_brief = None
```

(d) Late sources between prompts. Replace

```python
    while True:
        # The idle prompt's exit semantics mirror the gate/ask
```

with

```python
    while True:
        # A launch-brief source that landed after the prompt first opened is shown here, once,
        # between prompts — never while one is open, where it would tear the input line.
        if _launch_brief is not None:
            try:
                from commands.brief import show_late

                if show_late(_launch_brief):
                    _launch_brief = None
            except Exception as exc:
                diag.log(f"brief: late view failed: {exc}")
                _launch_brief = None
        # The idle prompt's exit semantics mirror the gate/ask
```

- [ ] **Step 5: Run the tests, then the whole suite**

Run: `.venv/bin/python -m pytest tests/test_brief.py -q`
Expected: `35 passed`.

Run: `.venv/bin/python -m pytest tests/ -q && .venv/bin/python -c "import app.repl"`
Expected: every test passes; the import prints nothing.

- [ ] **Step 6: Commit**

```bash
git add commands/brief.py app/repl.py tests/test_brief.py
git commit -m "brief: the launch view — before the first prompt, late sources between prompts"
```

---

### Task 6: Docs — what is true now

**Files:**
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `docs/pivot.md`, `docs/ARCHITECTURE.md`

- [ ] **Step 1: `CHANGELOG.md`** — first bullet under `## [Unreleased]` → `### Added`:

```markdown
- **Your day at a glance.** `/brief` shows, in one short block, what needs you today: overdue
  reminders and commitments, what is left of today's calendar (all-day events first), unread
  mail you have not replied to among the newest 25 in your inbox, tomorrow morning, and your
  last few memo notes. No model is involved and nothing leaves your Mac; subjects and titles
  are shown as plain text, and the brief is never added to the conversation. It stops at five
  lines a section and fifteen in all (`/brief --all` shows everything) and reuses what it
  gathered in the last ten minutes (`/brief --refresh` gathers again). `/brief on` shows it
  before the first prompt of every launch; it is off until you turn it on, because the first
  brief is when macOS asks for access to Calendar, Reminders and Mail. An app Saturn cannot
  read is named in one line and the rest still shows.
```

- [ ] **Step 2: `CLAUDE.md`.** Insert this paragraph directly after the paragraph that begins "`notify/` is the scheduled-notification seam":

```markdown
`core/brief.py` + `commands/brief.py` are the launch brief (`/brief`, `runtime.brief`, off by
default): the Calendar / Reminders / Mail readers' functions and the memory file, called
directly on one thread per source — no model call, never added to state or the prompt, every
line made inert (`_inert`: other people's text). The REPL starts it after the splash (a reader
imported from a thread before the registry would reorder the tool catalog), waits at most
`_LAUNCH_WAIT_S` before the first prompt, and prints a late source only between prompts.
```

Then extend the sentence "`tests/conftest.py` gives every test an empty `SATURN_HOME` and a throwaway `HOME`." to:

```markdown
`tests/conftest.py` gives every test an empty `SATURN_HOME` and a throwaway `HOME`, and fails
any `osascript` a test runs without the `mac` fixture (`_no_real_apple_events`).
```

- [ ] **Step 3: `docs/pivot.md`.** Change the heading `### 6. A launch brief (1 day)` to:

```markdown
### 6. A launch brief (1 day) — shipped <today's date, YYYY-MM-DD> (`core/brief.py`, `/brief`; deterministic, no model call; off until `/brief on`, because the first gather raises macOS's Automation dialogs)
```

(Write the actual date of the commit; this is the one value the plan cannot know.)

- [ ] **Step 4: `docs/ARCHITECTURE.md`.** In the `core/` table, add after the `hooks.py` row:

```markdown
| `brief.py` | The launch brief: Calendar / Reminders / Mail / memory read on one thread each, sorted into overdue · today · waiting · coming up · notes under a cap, every line inert. No model call; never enters state. Rendered by `commands/brief.py`. |
```

In the `commands/` paragraph, change "Themed modules: `conversation.py` (/clear /resume)," to "Themed modules: `brief.py` (/brief + the launch view), `conversation.py` (/clear /resume),".

- [ ] **Step 5: Check and commit**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: every test passes (docs only).

```bash
git add CHANGELOG.md CLAUDE.md docs/pivot.md docs/ARCHITECTURE.md
git commit -m "docs: the launch brief — changelog, CLAUDE.md, pivot #6 shipped, architecture map"
```

---

### Task 7: Verify on the real Mac (manual — the user's own apps)

This is the one step that drives the real Calendar, Reminders and Mail, and only the person
whose apps they are should run it. Nothing here is automated.

- [ ] **Step 1:** `saturn` in a terminal → `/brief`. Expected: macOS asks once per app ("Terminal wants to control Calendar/Reminders/Mail"). After you allow them, the block shows with "gathered HH:MM". Note how long Mail took (`SATURN_DEBUG=1` shows `brief : mail N.NNs`).
- [ ] **Step 2:** Deny one app (System Settings › Privacy & Security › Automation) → `/brief --refresh`. Expected: `mail: not available — macOS denied automation access to Mail; …`, and the rest still shows.
- [ ] **Step 3:** `/brief on`, quit, relaunch. Expected: the block (or nothing, on an empty day) right before the first prompt, at most ~2 s after the startup lines. A source still gathering is named on one dim line, and `── brief, continued` follows after your first command.
- [ ] **Step 4:** Relaunch and immediately ask "what's in my inbox?". Note the extra wait behind the brief's Mail query (Review Focus, "known and accepted"). If it is much worse than ~7 s, consider dropping `MAIL_WINDOW` to 10.
- [ ] **Step 5:** `/brief off`, relaunch. Expected: no brief and no wait.

Record anything surprising as a dated note under pivot #6 in `docs/pivot.md`.

---

## Self-Review (done while writing)

**1. Spec coverage.** Pivot #6: today's events (Task 1 `calendar_items`), mail waiting on a reply (Task 1 `mail_items`, unread and unanswered; sent-with-no-reply is deferred and listed), open commitments (Task 1 `memory_items`), the memo digest (*notes*), read-only and no background work after printing (Tasks 3 and 5: daemon threads end with their one read), cached for the session (Task 3 `start` reuse + `FRESH_S`), `/brief` re-runs it (Task 4 `--refresh`), `runtime.brief` (Task 3 key, Task 4 `on|off`). The research additions: caps and `+N more` (Task 2), silence on empty (Task 5 `test_launch_view_is_silent_when_there_is_nothing`; the command's one line in Task 4), date and time-zone buckets (Task 1 tests), awaiting reply (Task 1, with the sent-mail half deferred), follow-ups (Not in this plan). The brief's own constraints: never in the prompt (Task 4 `ctx.state == {}`; `nodes/ground.py` untouched), headless never (Task 5 guard), not egress (Task 3), non-macOS and denied apps (Tasks 2, 3, 5), and it never blocks the prompt beyond 2 s (Task 5). No gap found.

**2. Placeholder scan.** Every code step carries its full code, and every edit quotes the exact text it replaces. The one value left open is the shipped date in Task 6 Step 3, which is the commit date by definition.

**3. Type consistency.** `Item(bucket, key, text)`, `Unavailable(reason)`, `compose(results, gathered, *, pending, capped) -> Brief`, `Gather.compose(*, capped, only)`, `start(*, sources, refresh)`, `render(b, *, title, footer)`, `show_at_launch(g, wait)`, `show_late(g) -> bool`, and `first_run_hint()` are used with these signatures in every later task and test. `_mark_shown(g, b)` is defined in Task 4 and used in Tasks 4 and 5.

**4. Review Focus.** Each of the five lines names its test and the task that owns it. All of them are written out above. Test counts were checked against a dry run of the plan's code in a scratch copy of the tree (1336 tests passing, offline).
